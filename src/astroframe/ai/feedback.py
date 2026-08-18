"""Banco local de aprendizagem: recompensa/punição por avaliação em estrelas.

Cada utilização (imagem ou vídeo processado) é registada como **uma linha**
na base SQLite `~/.astroframe/feedback.db` — um log de ajustes que serve de
treino para o sistema:

- o que foi usado (parâmetros + origem: predefinidos/metadados/estrelas);
- como correu (métricas + estrelas calculadas);
- o que o utilizador avaliou (estrelas manuais, com peso reforçado);
- **o que se ajustou para a próxima vez, como e porquê** (rationale legível).

Além das tabelas `runs`/`tuning`, o banco guarda:

- `logs` — log local do sistema (componentes, níveis, mensagens);
- `models` — artefactos das redes neuronais (caminho `.npz`, métricas,
  tamanho do dataset e fonte), com **lógica de campeão**: `add_model`
  compara o resultado novo com o melhor registado (`champion`) e, se for
  melhor, promove-o; se for pior, o treino seguinte parte dos pesos do
  campeão (`warm-start`).

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

from astroframe.ai.params import FEEDBACK_PARAMS
from astroframe.ai.params import bounds as param_bounds
from astroframe.ai.score import StarRating
from astroframe.config import AstroFrameConfig
from astroframe.core.stabilizer import DiskDetection
from astroframe.paths import data_root

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
CREATE TABLE IF NOT EXISTS tuning (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    profile TEXT NOT NULL,
    base_params TEXT NOT NULL,
    deltas TEXT NOT NULL,
    report TEXT NOT NULL,
    source TEXT
);
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    level TEXT NOT NULL,
    component TEXT NOT NULL,
    message TEXT NOT NULL,
    details TEXT
);
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    round INTEGER,
    path TEXT NOT NULL,
    metrics TEXT NOT NULL,
    dataset INTEGER NOT NULL DEFAULT 0,
    source TEXT,
    promoted INTEGER NOT NULL DEFAULT 0
);
"""

# Métrica de comparação do campeão por tipo de modelo (chave em `metrics`).
# "score" é 0–100 vs guia manual (deteção); "mean_delta" são estrelas de
# melhoria média (imagem). Tipos sem entrada usam "score" se existir.
MODEL_METRICS: dict[str, str] = {
    "disk_filter": "score",
    "enhancer": "mean_delta",
    "lstm_traj": "score",
    "lstm_tuner": "score",
}

# Gamas seguras por parâmetro (a recompensa/punição nunca sai destes limites).
# Valores vindos do registry unificado (`astroframe.ai.params`).
_PARAM_BOUNDS = {path: param_bounds(path) for path in FEEDBACK_PARAMS}

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
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return data_root() / path


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
            conn.executescript(_SCHEMA)
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
        return self._rows_to_records(rows)

    def history_all(self, limit: int = 32) -> list[RunRecord]:
        """Últimas utilizações de **todos** os perfis (para o treino da LSTM)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, ts, profile, kind, source, params_used, params_origin, metrics,"
                " stars_calc, stars_user, nudge, rationale FROM runs"
                " ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return self._rows_to_records(rows)

    @staticmethod
    def _rows_to_records(rows) -> list[RunRecord]:
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

    # ------------------------------------------------------------ auto-tuning

    def add_tuning(
        self,
        profile: str,
        base_params: dict,
        deltas: dict,
        report: dict,
        source: str = "autotune",
    ) -> int:
        """Regista uma otimização completa (append-only) na tabela `tuning`."""
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO tuning (ts, profile, base_params, deltas, report, source)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    profile,
                    json.dumps(base_params, default=str),
                    json.dumps(deltas, default=str),
                    json.dumps(report, default=str),
                    source,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def tuning_history(self, profile: str, limit: int = 12) -> list[dict]:
        """Últimas otimizações do perfil (para o log visível na interface)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ts, base_params, deltas, report, source FROM tuning"
                " WHERE profile = ? ORDER BY id DESC LIMIT ?",
                (profile, limit),
            ).fetchall()
        return [
            {
                "ts": row[0],
                "base_params": json.loads(row[1] or "{}"),
                "deltas": json.loads(row[2] or "{}"),
                "report": json.loads(row[3] or "{}"),
                "source": row[4],
            }
            for row in rows
        ]

    def recent_tuning(self, profile: str, limit: int = 1) -> list[dict]:
        """Últimos deltas de auto-tuning do perfil (vazio se não houver)."""
        return [row["deltas"] for row in self.tuning_history(profile, limit=limit)]

    def reset_tuning(self) -> int:
        """Apaga o histórico de auto-tuning; devolve o nº de linhas removidas."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM tuning")
            conn.commit()
            return int(cur.rowcount)

    def log(self, level: str, component: str, message: str, details: dict | None = None) -> int:
        """Regista uma entrada no log local do sistema (`logs`)."""
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO logs (ts, level, component, message, details)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    level,
                    component,
                    message,
                    json.dumps(details, default=str) if details is not None else None,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def logs(self, limit: int = 100, component: str | None = None) -> list[dict]:
        """Últimas entradas do log (opcionalmente de um componente)."""
        with self._connect() as conn:
            if component:
                rows = conn.execute(
                    "SELECT ts, level, component, message, details FROM logs"
                    " WHERE component = ? ORDER BY id DESC LIMIT ?",
                    (component, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT ts, level, component, message, details FROM logs"
                    " ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [
            {
                "ts": row[0],
                "level": row[1],
                "component": row[2],
                "message": row[3],
                "details": json.loads(row[4]) if row[4] else None,
            }
            for row in rows
        ]

    def add_model(
        self,
        kind: str,
        path: str | Path,
        metrics: dict,
        dataset_size: int = 0,
        source: str = "training",
        round: int | None = None,
        metric_name: str | None = None,
    ) -> dict:
        """Regista um artefacto de rede neuronal e aplica a lógica de campeão.

        O resultado novo é comparado com o **melhor registado** (`champion`)
        pela métrica do tipo (`metrics[metric_name]`): se for **estritamente
        melhor** promove-o (`promoted=1`); senão regista `promoted=0` e o
        treino seguinte deve partir dos pesos do campeão (warm-start).
        Devolve `{"promoted", "previous", "champion"}` para o chamador
        decidir se copia o `.npz` para o caminho canónico.
        """
        metric = metric_name or MODEL_METRICS.get(kind, "score")
        if metric not in metrics or metrics[metric] is None:
            raise ValueError(f"Métrica '{metric}' ausente para o modelo {kind}.")
        previous = self.champion(kind)
        value = float(metrics[metric])
        promoted = previous is None or value > float(previous["metrics"].get(metric, -float("inf")))
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO models (ts, kind, round, path, metrics, dataset, source, promoted)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    kind,
                    round,
                    str(Path(path).expanduser()),
                    json.dumps(metrics, default=str),
                    int(dataset_size),
                    source,
                    1 if promoted else 0,
                ),
            )
            conn.commit()
            row_id = int(cur.lastrowid)
        return {
            "promoted": promoted,
            "previous": previous,
            "champion": self._champion_row(kind),
            "id": row_id,
        }

    def _champion_row(self, kind: str) -> dict | None:
        metric = MODEL_METRICS.get(kind, "score")
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, ts, kind, round, path, metrics, dataset, source, promoted"
                " FROM models WHERE kind = ? ORDER BY id ASC",
                (kind,),
            ).fetchall()
        best: dict | None = None
        best_value = -float("inf")
        for row in rows:
            metrics = json.loads(row[5] or "{}")
            value = float(metrics.get(metric, -float("inf")))
            if value > best_value:
                best_value = value
                best = self._model_row(row)
        return best

    def champion(self, kind: str) -> dict | None:
        """Melhor modelo registado do tipo (a métrica do `MODEL_METRICS`)."""
        return self._champion_row(kind)

    def model_history(self, kind: str, limit: int = 12) -> list[dict]:
        """Últimos artefactos do tipo (para o log visível na interface)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, ts, kind, round, path, metrics, dataset, source, promoted"
                " FROM models WHERE kind = ? ORDER BY id DESC LIMIT ?",
                (kind, limit),
            ).fetchall()
        return [self._model_row(row) for row in rows]

    @staticmethod
    def _model_row(row) -> dict:
        return {
            "id": row[0],
            "ts": row[1],
            "kind": row[2],
            "round": row[3],
            "path": row[4],
            "metrics": json.loads(row[5] or "{}"),
            "dataset": row[6],
            "source": row[7],
            "promoted": bool(row[8]),
        }


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

    Junta duas fontes, sempre com clamp do registry unificado:

    - os **nudges** do feedback por estrelas (`runs`, os 5 parâmetros
      visuais de `_PARAM_BOUNDS`);
    - os **deltas do auto-tuning** (`tuning`, qualquer parâmetro registado).

    Devolve uma cópia da configuração com os deltas somados aos valores
    atuais — a "memória" da IA entre utilizações. Com `feedback.enabled`
    desativado (e sem `db` explícito) devolve a configuração inalterada.
    """
    if db is None:
        if not cfg.feedback.enabled:
            return cfg
        db = FeedbackDB()
    import copy

    adjusted = copy.deepcopy(cfg)
    changed = False
    nudges = db.recent_nudges(profile, limit=1)
    if nudges:
        changed = True
        for field_name, delta in nudges[0].items():
            if field_name in _PARAM_BOUNDS:
                current = _get_param(adjusted, field_name)
                _set_param(adjusted, field_name, current + _clamp_delta(field_name, current, delta))
    tuning = db.recent_tuning(profile, limit=1)
    if tuning:
        changed = True
        from astroframe.ai import params as pparams

        for field_name, delta in tuning[0].items():
            if field_name in pparams.PARAM_SPECS:
                current = _get_param(adjusted, field_name)
                _set_param(
                    adjusted,
                    field_name,
                    pparams.clamp_value(field_name, current + delta),
                )
    return adjusted if changed else cfg


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
