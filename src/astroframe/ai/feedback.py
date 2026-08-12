"""Banco local de aprendizagem: recompensa/punição por avaliação em estrelas.

Cada utilização (imagem ou vídeo processado) é registada como **uma linha**
na base SQLite `~/.astroframe/feedback.db` — um log de ajustes que serve de
treino para o sistema:

- o que foi usado (parâmetros + origem: predefinidos/metadados/estrelas);
- como correu (métricas + estrelas calculadas);
- o que o utilizador avaliou (estrelas manuais, com peso reforçado);
- **o que se ajustou para a próxima vez, como e porquê** (rationale legível).

`apply_learned` carrega os deltas acumulados do perfil (tipo + resolução +
câmara + ISO) no início de cada execução; `nudge_params` converte as métricas
fracas em correções dirigidas (punição) e as boas em reforço (recompensa).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from astroframe.ai.score import StarRating
from astroframe.config import AstroFrameConfig
from astroframe.core.stabilizer import DiskDetection

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    profile TEXT NOT NULL,
    kind TEXT NOT NULL,
    source TEXT,
    params_used TEXT NOT NULL,
    params_origin TEXT NOT NULL,
    metrics TEXT NOT NULL,
    stars_calc REAL NOT NULL,
    stars_user REAL,
    nudge TEXT NOT NULL,
    rationale TEXT NOT NULL
);
"""

# Gamas seguras por parâmetro (a recompensa/punição nunca sai destes limites).
_PARAM_BOUNDS = {
    "clahe.clip_limit": (0.5, 6.0),
    "denoise.h": (1.0, 20.0),
    "unsharp.amount": (0.0, 2.0),
    "polish.corona_scale": (1.0, 3.0),
    "polish.feather": (0.0, 0.1),
}

# "Punição": como corrigir cada métrica fraca; "recompensa": o que manter
# quando a métrica é excelente. Cada delta é (campo, delta, porque_curto).
_RULES = {
    "background": {
        "bad": [("polish.feather", 0.005, "fundo com brilho residual → feather +0.005")],
        "good": [("polish.feather", -0.002, "fundo limpo → feather −0.002 (retém mais coroa)")],
    },
    "limb": {
        "bad": [
            ("denoise.h", 0.8, "limbo irregular (ruído na borda) → denoise.h +0.8"),
            ("unsharp.amount", 0.1, "limbo irregular → nitidez +0.1 para refinar a borda"),
        ],
        "good": [("denoise.h", -0.3, "limbo perfeito → denoise.h −0.3 (menos plastificação)")],
    },
    "noise": {
        "bad": [("denoise.h", 1.0, "ruído alto na coroa → denoise.h +1.0")],
        "good": [("denoise.h", -0.3, "ruído baixo → denoise.h −0.3 (preserva detalhe)")],
    },
    "contrast": {
        "bad": [("clahe.clip_limit", 0.2, "contraste pobre → CLAHE clip_limit +0.2")],
        "good": [("clahe.clip_limit", -0.1, "contraste excelente → CLAHE clip_limit −0.1 (naturalidade)")],
    },
    "reflections": {
        "bad": [
            ("polish.corona_scale", -0.1, "reflexos na coroa → coroa mantida −0.1 (recorta ghosts)"),
        ],
        "good": [("polish.corona_scale", 0.05, "sem reflexos → coroa mantida +0.05 (maior gama)")],
    },
}


def profile_for(kind: str, width: int, height: int, camera: str = "", iso: int | None = None) -> str:
    """Perfil de aprendizagem: tipo + escala de resolução + câmara + ISO."""
    bucket = "small" if max(width, height or 0) < 1080 else "large"
    iso_bucket = "low" if not iso else ("hi" if iso > 1600 else "mid")
    raw = f"{kind}|{bucket}|{camera or 'camera'}|{iso_bucket}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return digest


@dataclass
class RunRecord:
    """Uma linha do banco: utilização + avaliação + ajuste aprendido."""

    id: int
    ts: str
    profile: str
    kind: str
    source: str
    params_used: dict
    params_origin: dict
    metrics: dict
    stars_calc: float
    stars_user: float | None
    nudge: dict
    rationale: str


def _default_db_path(config: AstroFrameConfig | None) -> Path:
    override = os.environ.get("ASTROFRAME_FEEDBACK_DB")
    raw = override or (config or AstroFrameConfig()).feedback.db_path
    return Path(raw).expanduser()


class FeedbackDB:
    """Base local SQLite (append-only) com o log de aprendizagem.

    Thread-safe por utilização (uma conexão por operação); o ficheiro é
    criado com permissões restritas para não expor material fotográfico.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path).expanduser() if path else _default_db_path(None)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path)
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(_SCHEMA)
            conn.commit()

    def add_run(
        self,
        kind: str,
        profile: str,
        params_used: dict,
        params_origin: dict,
        rating: StarRating,
        stars_user: float | None,
        nudge: dict,
        rationale: str | None,
        source: str = "",
    ) -> int:
        """Regista uma utilização como uma nova linha (append-only)."""
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO runs (ts, profile, kind, source, params_used, params_origin,"
                " metrics, stars_calc, stars_user, nudge, rationale)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    profile,
                    kind,
                    source or "",
                    json.dumps(params_used, default=str),
                    json.dumps(params_origin, default=str),
                    json.dumps(rating.metrics, default=str),
                    float(rating.stars),
                    float(stars_user) if stars_user is not None else None,
                    json.dumps(nudge, default=str),
                    rationale or "",
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def history(self, profile: str, limit: int = 12) -> list[RunRecord]:
        """Últimas utilizações do perfil (para o log visível na interface)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, ts, profile, kind, source, params_used, params_origin, metrics,"
                " stars_calc, stars_user, nudge, rationale FROM runs WHERE profile = ?"
                " ORDER BY id DESC LIMIT ?",
                (profile, limit),
            ).fetchall()
        return [
            RunRecord(
                id=row[0],
                ts=row[1],
                profile=row[2],
                kind=row[3],
                source=row[4],
                params_used=json.loads(row[5] or "{}"),
                params_origin=json.loads(row[6] or "{}"),
                metrics=json.loads(row[7] or "{}"),
                stars_calc=row[8],
                stars_user=row[9],
                nudge=json.loads(row[10] or "{}"),
                rationale=row[11],
            )
            for row in rows
        ]

    def recent_nudges(self, profile: str, limit: int = 1) -> list[dict]:
        rows = self.history(profile, limit=limit)
        return [row.nudge for row in rows]

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0])


def _clamp_delta(field_name: str, current: float, delta: float) -> float:
    low, high = _PARAM_BOUNDS.get(field_name, (0.0, 1e9))
    next_value = current + delta
    if next_value < low:
        return low - current
    if next_value > high:
        return high - current
    return delta


def _set_param(cfg: AstroFrameConfig, field_name: str, value: float) -> None:
    section, name = field_name.split(".")
    setattr(getattr(cfg, section), name, float(value))


def _get_param(cfg: AstroFrameConfig, field_name: str) -> float:
    section, name = field_name.split(".")
    return float(getattr(getattr(cfg, section), name))


def apply_learned(cfg: AstroFrameConfig, profile: str, db: FeedbackDB | None = None) -> AstroFrameConfig:
    """Aplica os deltas aprendidos (EMA) no início de cada execução.

    Devolve uma cópia da configuração com os deltas somados aos valores
    atuais — a "memória" da IA entre utilizações. Com `feedback.enabled`
    desativado (e sem `db` explícito) devolve a configuração inalterada.
    """
    if db is None:
        if not cfg.feedback.enabled:
            return cfg
        db = FeedbackDB()
    nudges = db.recent_nudges(profile, limit=1)
    if not nudges:
        return cfg
    import copy

    adjusted = copy.deepcopy(cfg)
    for field_name, delta in nudges[0].items():
        if field_name in _PARAM_BOUNDS:
            current = _get_param(adjusted, field_name)
            _set_param(adjusted, field_name, current + _clamp_delta(field_name, current, delta))
    return adjusted


def nudge_params(
    rating: StarRating,
    cfg: AstroFrameConfig,
    lr: float | None = None,
    user_weight: float | None = None,
) -> tuple[dict, str]:
    """Converte a avaliação em ajustes dirigidos, com justificação.

    - estrelas < 3 → punição (corrige as métricas fracas);
    - estrelas ≥ 4 → recompensa (reforça/relaxa o que está bom);
    - entre 3 e 4 → ajustes suaves das métricas fracas apenas.

    Devolve (deltas {campo: delta}, rationale legível).
    """
    config = cfg or AstroFrameConfig()
    alpha = lr if lr is not None else config.feedback.learning_rate
    weight = user_weight if user_weight is not None else 1.0
    stars = rating.stars
    deltas: dict[str, float] = {}
    reasons: list[str] = []

    if stars >= 4.0:
        state = "good"
        tag = "recompensa"
    elif stars < 3.0:
        state = "bad"
        tag = "punição"
    else:
        state = "mixed"
        tag = "ajuste fino"

    for metric, value in rating.metrics.items():
        rule = _RULES.get(metric)
        if not rule:
            continue
        if state == "mixed" and value >= 0.85:
            continue
        if state == "good" and value >= 0.85:
            for field_name, delta, why in rule["good"]:
                deltas[field_name] = deltas.get(field_name, 0.0) + delta * alpha * weight
                reasons.append(why)
        elif state == "bad" and value < 0.85:
            for field_name, delta, why in rule["bad"]:
                deltas[field_name] = deltas.get(field_name, 0.0) + delta * alpha * weight
                reasons.append(why)

    summary = f"avaliação {stars:.1f}★ → {tag}: " + (
        "; ".join(reasons) if reasons else "sem alterações necessárias"
    )
    return deltas, summary


def record_run(
    db: FeedbackDB,
    kind: str,
    profile: str,
    cfg: AstroFrameConfig,
    origin: dict,
    rating: StarRating,
    stars_user: float | None = None,
    source: str = "",
    lr: float | None = None,
    user_weight: float | None = None,
) -> RunRecord:
    """Regista a utilização completa: uma linha nova por avaliação.

    O `nudge` guardado é o que será aplicado (EMA) na próxima execução do
    mesmo perfil — "o que o sistema ajustou, como e porquê".
    """
    deltas, rationale = nudge_params(
        rating,
        cfg,
        lr=lr,
        user_weight=(
            user_weight
            if user_weight is not None
            else (cfg.feedback.user_weight if stars_user is not None else None)
        ),
    )
    params_used = json.loads(json.dumps(cfg.to_dict(), default=str))
    row_id = db.add_run(
        kind=kind,
        profile=profile,
        params_used=params_used,
        params_origin=origin,
        rating=rating,
        stars_user=stars_user,
        nudge=deltas,
        rationale=rationale,
        source=source,
    )
    run = db.history(profile, limit=1)
    if run:
        return run[0]
    return RunRecord(
        row_id,
        "",
        profile,
        kind,
        source,
        params_used,
        origin,
        rating.metrics,
        rating.stars,
        stars_user,
        deltas,
        rationale,
    )


def origin_for(cfg: AstroFrameConfig) -> dict:
    """Origem dos parâmetros atuais (para o log): predef/metadados/estrelas.

    Representação curta e estável: campo → "default" (não se sabe a origem)
    ou o estado atual, já que a origem exata é anotada pela UI.
    """
    pairs = (
        ("clahe", "clip_limit"),
        ("denoise", "h"),
        ("unsharp", "amount"),
        ("polish", "corona_scale"),
    )
    return {f"{k}.{name}": _get_param(cfg, f"{k}.{name}") for k, name in pairs}


def detection_id(detection: DiskDetection | None) -> str:
    """Identificador estável do disco (usado como fonte no log)."""
    if detection is None:
        return "none"
    return f"{detection.cx}x{detection.cy}r{detection.radius}"
