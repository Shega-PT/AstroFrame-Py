"""Auto-tuning: otimização determinística e limitada de todos os parâmetros.

Estratégia estável, rápida e segura para ajustar **qualquer** parâmetro da
pipeline (`clahe`, `denoise`, `unsharp`, `stabilizer`, `lucky`, `stacking`,
`polish`, `score`), a partir das amostras de `samples/`:

1. **Proxy de avaliação rápido** (`ProxyEval`) — cada amostra é reduzida a
   ~480p e avaliada com a pipeline real: deteção (IoU/recall/precisão vs o
   guia manual de `calibration.json`) + melhoria (`score_image`, 0–5
   estrelas). O resultado é **guardado em cache** por hash dos parâmetros
   efetivos — só o que muda é reavaliado.
2. **Hill-climbing limitado** (`BoundedHillClimb`) — determinístico (seed
   fixa), com momentum por parâmetro, passos adaptativos, recozimento
   opcional (aceitar pioras para escapar de mínimos locais), patience e
   orçamento de tempo. **Nunca sai das gamas seguras** do registry
   (`astroframe.ai.params`).
3. **Persistência** — cada otimização é registada na tabela `tuning` do
   SQLite (mesma `FeedbackDB`); `apply_learned` aplica os deltas na próxima
   execução do mesmo perfil. `--reset` apaga o histórico.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from astroframe.ai import params as pparams
from astroframe.ai.feedback import FeedbackDB
from astroframe.ai.score import score_image
from astroframe.calibration.scan import load_frame, scan_samples
from astroframe.calibration.store import CalibrationStore
from astroframe.calibration.validate import validate_all
from astroframe.config import AstroFrameConfig
from astroframe.core.enhancer import enhance_image
from astroframe.core.stabilizer import DiskDetection, find_all_disks
from astroframe.paths import calibration_json

logger = logging.getLogger(__name__)

TUNED_VERSION = 2
TUNED_KIND = "astroframe-tuned"
DEFAULT_EXPORT_NAME = "trained_config.json"
DEFAULT_PROFILE = "tuning"

# Proxy: lado mínimo de trabalho (~480p) e nº de amostras pontuadas para as
# estrelas (a denoise é o passo mais lento; 3 frames chegam para o proxy).
_WORK_MIN_SIDE = 480


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _detection_component(report) -> float | None:
    """Componente de deteção 0–1 (recall 0.4 · precisão 0.3 · IoU 0.3), ou
    None quando não há ground truth para pontuar."""
    if not report.has_ground_truth:
        return None
    return 0.4 * report.recall + 0.3 * report.precision + 0.3 * report.mean_iou


@dataclass
class TuneReport:
    """Avaliação de uma configuração no proxy (0–1 cada componente)."""

    objective: float = 0.0
    stars: float = 0.0
    detection: float | None = None
    recall: float = 0.0
    precision: float = 0.0
    mean_iou: float = 0.0
    elapsed_s: float = 0.0
    n_items: int = 0
    n_scored: int = 0

    def to_dict(self) -> dict:
        return {
            "objective": round(self.objective, 4),
            "stars": round(self.stars, 4),
            "detection": round(self.detection, 4) if self.detection is not None else None,
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "mean_iou": round(self.mean_iou, 4),
            "elapsed_s": round(self.elapsed_s, 3),
            "n_items": self.n_items,
            "n_scored": self.n_scored,
        }


@dataclass
class TuneResult:
    """Resultado de uma otimização: melhor configuração + deltas + relatório."""

    config: AstroFrameConfig
    deltas: dict[str, float]
    base: AstroFrameConfig
    report: TuneReport
    evaluations: int = 0
    lines: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "deltas": self.deltas,
            "report": self.report.to_dict(),
            "evaluations": self.evaluations,
        }


class ProxyEval:
    """Avaliação rápida de configurações sobre as amostras de `samples/`.

    As imagens/frames são reduzidas para ~480p (sem nunca aumentar), a
    deteção é comparada com o guia manual e a melhoria é pontuada com
    estrelas. Os relatórios são guardados em cache por hash dos parâmetros
    efetivos — a otimização só reavalia o que muda.
    """

    def __init__(
        self,
        samples_dir: str | Path,
        work_scale: float = 0.5,
        frames_per_sample: int = 3,
        detection_weight: float = 0.6,
        seed: int = 42,
    ):
        self.samples_dir = Path(samples_dir)
        self.work_scale = float(work_scale)
        self.frames_per_sample = max(1, frames_per_sample)
        self.detection_weight = _clamp01(detection_weight)
        self.seed = seed
        self.samples = scan_samples(self.samples_dir)
        self.store = CalibrationStore(calibration_json(self.samples_dir))
        self._cache: dict[str, TuneReport] = {}

    def _scale_for(self, height: int, width: int) -> float:
        """Fator de redução: ~480p no lado mínimo e nunca acima de work_scale."""
        half = min(height, width)
        if half <= _WORK_MIN_SIDE:
            return 1.0
        return min(self.work_scale, _WORK_MIN_SIDE / half)

    def _frame_for(self, sample) -> tuple[np.ndarray, float]:
        """Frame reduzido (BGR) da amostra + fator de escala aplicado."""
        frame = load_frame(sample)
        h, w = frame.shape[:2]
        scale = self._scale_for(h, w)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return frame, scale

    def _cache_key(self, config: AstroFrameConfig) -> str:
        values = {path: round(pparams.get_param(config, path), 6) for path in pparams.PARAM_SPECS}
        raw = json.dumps(values, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:16]

    def evaluate(self, config: AstroFrameConfig) -> TuneReport:
        """Avalia a configuração no proxy (com cache por parâmetros efetivos)."""
        key = self._cache_key(config)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        start = time.monotonic()
        rows: list[tuple[str, list[DiskDetection], list[DiskDetection]]] = []
        stars_total = 0.0
        n_scored = 0
        scored = 0
        for sample in self.samples:
            try:
                frame, scale = self._frame_for(sample)
            except Exception as exc:
                logger.warning("Amostra ignorada no auto-tuning (%s): %s", sample.label, exc)
                continue
            item = self.store.get_item(sample.key)
            gt = [self._scaled(circle, scale) for circle in (list(item.circles) if item else [])]
            detected = find_all_disks(frame, config)
            rows.append((sample.label, gt, detected))
            if scored < self.frames_per_sample and detected:
                enhanced = enhance_image(frame, config)
                rating = score_image(enhanced, detected[0], config)
                stars_total += rating.stars
                n_scored += 1
                scored += 1
        report = validate_all(rows)
        stars = stars_total / n_scored if n_scored else 0.0
        detection = _detection_component(report)
        if detection is None:
            objective = stars / 5.0
        else:
            objective = (self.detection_weight * detection + (1.0 - self.detection_weight) * stars / 5.0) / (
                self.detection_weight + (1.0 - self.detection_weight)
            )
        elapsed = time.monotonic() - start
        result = TuneReport(
            objective=_clamp01(objective),
            stars=stars,
            detection=detection,
            recall=report.recall,
            precision=report.precision,
            mean_iou=report.mean_iou,
            elapsed_s=elapsed,
            n_items=len(rows),
            n_scored=n_scored,
        )
        self._cache[key] = result
        return result

    @staticmethod
    def _scaled(circle: DiskDetection, scale: float) -> DiskDetection:
        """Ground truth reduzido à escala de trabalho (elipse preservada)."""
        if scale >= 1.0:
            return circle
        ry = int(circle.ry * scale) if circle.ry is not None else None
        return DiskDetection(int(circle.cx * scale), int(circle.cy * scale), int(circle.radius * scale), ry)

    def clear_cache(self) -> None:
        self._cache.clear()


class BoundedHillClimb:
    """Hill-climbing limitado com momentum, recozimento e orçamento de tempo.

    Determinístico (seed fixa): percorre os parâmetros na ordem do registry,
    tenta +step e −step (via cache do proxy), aceita a melhor se melhorar
    ≥ `improve_eps`; momentum duplica o passo após 2 aceites seguidos na
    mesma direção; falhas reduzem o passo a metade (mín. step/8). Com
    `anneal`, aceita pioras com probabilidade exp(−Δ/T), `T` a decair por
    passada — escapa de mínimos locais mantendo-se dentro das gamas seguras.
    """

    def __init__(
        self,
        specs: list[pparams.ParamSpec],
        budget_s: float = 60.0,
        seed: int = 42,
        anneal: bool = True,
        patience: int = 3,
        improve_eps: float = 1e-4,
    ):
        self.specs = list(specs)
        self.budget_s = float(budget_s)
        self.seed = seed
        self.anneal = bool(anneal)
        self.patience = patience
        self.improve_eps = float(improve_eps)
        self._rng = np.random.default_rng(seed)

    def optimize(
        self,
        evaluate: callable,
        base: AstroFrameConfig,
        start_deltas: dict[str, float] | None = None,
    ) -> TuneResult:
        """Otimiza a partir da configuração base (e, opcionalmente, dos deltas
        previstos pela LSTM); devolve a melhor configuração encontrada."""
        deltas: dict[str, float] = dict(start_deltas or {})
        start_cfg = pparams.apply_deltas(base, deltas)
        base_report = evaluate(start_cfg)
        best = TuneResult(
            config=copy.deepcopy(start_cfg),
            deltas=dict(deltas),
            base=copy.deepcopy(base),
            report=base_report,
            evaluations=1,
        )
        current_obj = base_report.objective
        steps = {spec.path: spec.step for spec in self.specs}
        momentum: dict[str, int] = {}
        temperature = 0.5
        start = time.monotonic()
        evaluations = 1
        passes_without_improve = 0
        pass_idx = 0

        while time.monotonic() - start < self.budget_s and passes_without_improve < self.patience:
            improved_pass = False
            for spec in self.specs:
                if time.monotonic() - start >= self.budget_s:
                    break
                # Parâmetros caros (denoise) são tentados só nas passadas pares,
                # para o orçamento não arder todo no passo mais lento.
                if spec.costly and pass_idx % 2 == 1:
                    continue
                step = steps[spec.path]
                up = dict(deltas)
                up[spec.path] = up.get(spec.path, 0.0) + step
                down = dict(deltas)
                down[spec.path] = down.get(spec.path, 0.0) - step
                up_cfg = pparams.apply_deltas(base, up)
                down_cfg = pparams.apply_deltas(base, down)
                up_report = evaluate(up_cfg)
                down_report = evaluate(down_cfg)
                evaluations += 2
                if up_report.objective >= down_report.objective:
                    direction, moved, candidate_obj = +1, up, up_report.objective
                else:
                    direction, moved, candidate_obj = -1, down, down_report.objective
                better = candidate_obj - current_obj
                if better >= self.improve_eps:
                    deltas = moved
                    current_obj = candidate_obj
                    if best.report.objective < current_obj:
                        best = TuneResult(
                            config=pparams.apply_deltas(base, deltas),
                            deltas=dict(deltas),
                            base=copy.deepcopy(base),
                            report=up_report if direction > 0 else down_report,
                            evaluations=evaluations,
                        )
                    momentum[spec.path] = direction
                    if momentum.get(spec.path, 0) == direction:
                        steps[spec.path] = min(spec.step * 4.0, step * 1.5)
                    improved_pass = True
                elif self.anneal and self._rng.random() < float(np.exp(-max(0.0, -better) / temperature)):
                    deltas = moved
                    current_obj = candidate_obj
                    momentum[spec.path] = direction
                else:
                    momentum[spec.path] = 0
                    steps[spec.path] = max(spec.step / 8.0, step / 2.0)
            if not improved_pass:
                passes_without_improve += 1
            else:
                passes_without_improve = 0
            temperature *= 0.9
            pass_idx += 1

        best.report.elapsed_s = time.monotonic() - start
        best.evaluations = evaluations
        best.lines = tuning_table_lines(best)
        return best


def tuning_table_lines(result: TuneResult) -> list[str]:
    """Linhas do relatório: parâmetro · base · ajustado · delta (só alterados)."""
    base = result.base
    cfg = result.config
    lines: list[str] = []
    changed = [
        path
        for path in pparams.PARAM_SPECS
        if abs(pparams.get_param(cfg, path) - pparams.get_param(base, path)) > 1e-9
    ]
    for path in sorted(changed):
        spec = pparams.spec(path)
        delta = pparams.get_param(cfg, path) - pparams.get_param(base, path)
        lines.append(
            f"{path}: {pparams.get_param(base, path):g} → {pparams.get_param(cfg, path):g} "
            f"({delta:+g}, passo {spec.step:g})"
        )
    if not changed:
        lines.append("Nenhum parâmetro ajustado — a configuração base já é a melhor.")
    report = result.report
    lines.append(
        f"Objetivo {report.objective:.3f} · estrelas {report.stars:.1f}/5"
        + (f" · deteção {report.detection:.3f}" if report.detection is not None else "")
        + f" · {result.evaluations} avaliações · {report.elapsed_s:.1f}s"
    )
    return lines


def export_trained_config(
    config: AstroFrameConfig,
    deltas: dict[str, float],
    report: TuneReport,
    path: str | Path,
) -> Path:
    """Exporta a configuração otimizada (JSON) para alimentar o sistema real.

    Inclui todos os parâmetros efetivos (`params`), os deltas, o relatório
    do proxy e a secção `stabilizer` (compatível com o export antigo).
    """
    path = Path(path)
    effective = pparams.apply_deltas(config, deltas)
    data: dict = {
        "version": TUNED_VERSION,
        "kind": TUNED_KIND,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "deltas": {k: round(v, 6) for k, v in deltas.items()},
        "params": {p: pparams.get_param(effective, p) for p in pparams.PARAM_SPECS},
        "stabilizer": {
            spec.name: pparams.get_param(effective, spec.path) for spec in pparams.specs("detect")
        },
        "report": report.to_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_autotune(
    samples_dir: str | Path,
    config: AstroFrameConfig | None = None,
    budget_s: float = 60.0,
    seed: int = 42,
    anneal: bool = True,
    params_filter: str | None = None,
    export_path: str | Path | None = None,
    profile: str = DEFAULT_PROFILE,
    db: FeedbackDB | None = None,
    work_scale: float = 0.5,
    frames_per_sample: int = 3,
    detection_weight: float = 0.6,
) -> TuneResult:
    """Orquestra uma otimização completa e regista-a no banco de aprendizagem."""
    config = config or AstroFrameConfig()
    specs = pparams.specs()
    if params_filter:
        wanted = {name.strip() for name in params_filter.split(",") if name.strip()}
        specs = [spec for spec in specs if spec.name in wanted or spec.path in wanted]
    if not specs:
        raise ValueError("Nenhum parâmetro selecionado para o auto-tuning.")
    proxy = ProxyEval(
        samples_dir,
        work_scale=work_scale,
        frames_per_sample=frames_per_sample,
        detection_weight=detection_weight,
        seed=seed,
    )
    optimizer = BoundedHillClimb(specs, budget_s=budget_s, seed=seed, anneal=anneal)
    start_deltas = _lstm_seed(proxy, config, db)
    result = optimizer.optimize(proxy.evaluate, config, start_deltas=start_deltas)
    if export_path:
        export_trained_config(result.config, result.deltas, result.report, export_path)
    if db is not None or config.feedback.enabled:
        db = db or FeedbackDB()
        db.add_tuning(profile, config.to_dict(), result.deltas, result.report.to_dict(), "autotune")
    return result


def _lstm_seed(
    proxy: ProxyEval,
    base: AstroFrameConfig,
    db: FeedbackDB | None,
) -> dict[str, float]:
    """Pré-seed LSTM: deltas previstos do histórico do perfil, se melhorarem
    o objetivo do proxy. Sem modelo/histórico (ou sem ganho) devolve `{}` —
    a otimização parte da base, falhando sempre em silêncio."""
    if db is None:
        return {}
    try:
        from astroframe.ai.lstm import LSTMTuner

        tuner = LSTMTuner.load()
        if tuner is None:
            return {}
        history = db.history_all(limit=32)
        deltas = tuner.predict_next_delta(history)
        if not deltas:
            return {}
        seeded = pparams.apply_deltas(base, deltas)
        if proxy.evaluate(seeded).objective > proxy.evaluate(base).objective + 1e-4:
            return deltas
    except Exception as exc:  # pragma: no cover - falha silenciosa garantida
        logger.warning("Pré-seed LSTM indisponível: %s", exc)
    return {}
