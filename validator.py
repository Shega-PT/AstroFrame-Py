"""Ponto de entrada da validação: treino da deteção por recompensa/punição.

Equivalente ao `calibrate.py`, mas em vez de ajustares os círculos à mão,
**validas o que a deteção automática fez**: o sistema analisa cada imagem e
frame de `samples/` e apresenta os círculos/elipses detetados; por cada forma
tens apenas dois botões:

- **Válido** → o sistema é recompensado (os parâmetros relaxam ligeiramente);
- **Rejeitado** → o sistema é punido (os parâmetros apertam: `param2`,
  `param1`, `dp`, o desfoque gaussiano e a tolerância a discos ocultos são
  ajustados) e a imagem é **reavaliada** de imediato, determinando as formas
  novamente. Os pesos do treino nunca tocam em tamanhos nem distâncias — o
  raio mínimo e a distância entre centros são derivados da resolução.

O loop repete-se até 100% das imagens e frames de vídeo estarem processados.
`samples/calibration.json` (a saída da calibração manual) serve de **guia**:
os círculos manuais aparecem desenhados na imagem, cada forma pendente mostra
o IoU contra o guia e o relatório final compara o resultado aceite com a
calibração manual (recall, precisão, IoU, score 0–100).

**Séries de treino**: cada passagem completa (manual ou automática) é uma
*série*. No fim de cada série o estado é gravado no JSON de treino
(`samples/validator_state.json`), que no início da série seguinte alimenta o
sistema com os pesos/deltas aprendidos — carrega em **Novo treino** para
continuar a treinar a partir do ponto onde paraste.

**Treino automático**: a janela **Treino automático…** faz o mesmo esquema,
sem intervenção humana: compara cada deteção com a calibração e valida se o
IoU for ≥ um limiar **sempre ≥ 0.90** (ajustável na UI ou com `--iou`, para
aumentar a dificuldade) que **sobe conforme o número de validações aumenta**
— a margem de erro encolhe com o treino; rejeita (e reavalia) tudo o resto.

No fim de qualquer série a configuração treinada é exportada para
`samples/trained_config.json` (por omissão), prontinha para alimentar o
sistema "real" (`--export` muda o caminho), e é mostrado o **relatório final
em pop-up**: discrepâncias por amostra vs calibração manual (IoU, falsos
negativos/positivos, erros do centro e do raio) e as recompensas/punições
com os seus pesos — editáveis ali mesmo para as séries seguintes.

Uso:
    python validator.py                     # janela desktop (tkinter)
    python validator.py --samples samples
    python validator.py --config config.yaml
    python validator.py --reset-state       # apaga progresso, deltas e séries
    python validator.py --check             # relatório sem interface
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import queue
import shutil
import sys
import threading
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from astroframe.ai import params as pparams
from astroframe.ai.cnn import DiskFilter, SmallCNN, disk_patch, fit_classifier
from astroframe.ai.feedback import FeedbackDB
from astroframe.ai.score import score_image, stars_text
from astroframe.core.enhancer import enhance_image
from astroframe.calibration.scan import SampleRef, load_frame, scan_samples
from astroframe.calibration.store import CalibrationStore
from astroframe.calibration.validate import CalibrationReport, shape_iou, validate_all, validate_item
from astroframe.config import AstroFrameConfig
from astroframe.core.stabilizer import DiskDetection, find_all_disks, find_disks_for_calibration
from astroframe.paths import (
    calibration_json,
    logs_ia_dir,
    migrate_legacy,
    setup_logging,
    staging_dir,
    train_dir,
    weights_dir,
)

logger = logging.getLogger(__name__)

try:  # tkinter é stdlib mas o módulo só é necessário em runtime
    import tkinter as tk
    from tkinter import messagebox, ttk

    from PIL import Image, ImageTk
except ImportError as exc:  # pragma: no cover
    tk = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    ImageTk = None  # type: ignore[assignment]
    _TK_IMPORT_ERROR = exc
else:
    _TK_IMPORT_ERROR = None

# --------------------------------------------------------------------------
# Constantes do treino
# --------------------------------------------------------------------------

# Parâmetros treináveis: apenas ajustes de acumulador (Hough), desfoque e
# tolerância a discos ocultos — **nunca** tamanhos nem distâncias (o raio
# mínimo e a distância entre centros são derivados da resolução da imagem,
# sem valores em px treináveis). Os limites, passos e deltas vêm do registry
# unificado (`astroframe.ai.params`) — fonte única de verdade do treino.
TRAINABLE_PARAMS = tuple(spec.name for spec in pparams.specs("detect"))

# Punição (forma rejeitada): apertar a deteção para o falso positivo
# desaparecer. Recompensa (forma válida): relaxar um pouco (25%) para não
# ficar demasiado estrito e deixar de detetar astros reais pequenos.
PUNISH_DELTAS = pparams.default_punish_deltas()
REWARD_DELTAS = pparams.default_reward_deltas()

# Gamas seguras dos parâmetros treináveis (o delta nunca sai destes limites).
# `max_radius` não é treinado — só é limitado quando exportado.
DELTA_BOUNDS = {spec.name: (spec.low, spec.high) for spec in pparams.specs("detect")}
DELTA_BOUNDS["max_radius"] = (50.0, 5000.0)

MAX_REEVALS_PER_SAMPLE = 12
MAX_JUDGMENTS_PER_SAMPLE = 100
STATE_VERSION = 2
DEFAULT_STATE_NAME = "validator_state.json"
DEFAULT_EXPORT_NAME = "trained_config.json"

# Rede neuronal da deteção (classificador disco/ruído) no treino automático:
# os patches recolhidos (positivos = guia, negativos = falsos positivos +
# recortes aleatórios) re-treina a CNN entre séries (warm-start do campeão);
# os candidatos da deteção são filtrados por confiança durante o julgamento.
CNN_THRESHOLD_DEFAULT = 0.5
CNN_RANDOM_NEGATIVES_PER_SAMPLE = 4
CNN_MODEL_DIR = staging_dir()
CNN_CANONICAL_PATH = weights_dir() / "disk_filter.npz"

# Treino automático: o mínimo de correspondência com o guia manual (IoU) é
# sempre ≥ 0.90; a UI (e a CLI com `--iou`) pode aumentar o valor para subir
# a dificuldade — nunca abaixo do mínimo.
AUTO_IOU_MIN = 0.90
AUTO_IOU_MAX = 0.99

# Curva do treino automático: o limiar exigido começa em `AUTO_IOU_START` (o
# mínimo ajustável, ≥ 0.90) e cresce exponencialmente até `AUTO_IOU_END`
# conforme o número de validações acumuladas — a margem de erro encolhe e a
# dificuldade aumenta. `AUTO_IOU_RATE` controla a rapidez dessa subida.
AUTO_IOU_START = AUTO_IOU_MIN
AUTO_IOU_END = AUTO_IOU_MAX
AUTO_IOU_RATE = 200.0

# Cores do desenho
COLOR_GT = "#4da6ff"
COLOR_ACCEPTED = "#3ee66f"
COLOR_REJECTED = "#ff5c5c"
COLOR_PENDING = "#3ee66f"
COLOR_CURRENT = "#ffd23f"

# Pré-visualização em tempo real do treino automático: no máximo um frame
# enviado para a UI a cada intervalo (a deteção é muito mais rápida que o
# desenho; o throttle impede a fila de acumular).
PREVIEW_THROTTLE_S = 0.15


def circle_to_dict(circle: DiskDetection) -> dict:
    """Forma → dict (serialização do estado)."""
    raw = {"cx": circle.cx, "cy": circle.cy, "radius": circle.radius}
    if circle.ry is not None:
        raw["ry"] = circle.ry
    return raw


def circle_from_dict(raw: dict) -> DiskDetection:
    """dict → forma (desserialização do estado)."""
    ry = raw.get("ry")
    return DiskDetection(
        int(raw["cx"]), int(raw["cy"]), int(raw["radius"]), int(ry) if ry is not None else None
    )


def same_shape(a: DiskDetection, b: DiskDetection, threshold: float = 0.5) -> bool:
    """Duas formas representam o mesmo objeto (IoU ≥ limiar)."""
    return shape_iou(a, b) >= threshold


def filter_unjudged(
    detected: list[DiskDetection], accepted: list[DiskDetection], rejected: list[DiskDetection]
) -> list[DiskDetection]:
    """Formas detetadas ainda não julgadas (fora das aceites/rejeitadas)."""
    return [
        shape
        for shape in detected
        if not any(same_shape(shape, known) for known in accepted)
        and not any(same_shape(shape, known) for known in rejected)
    ]


def persistent_rejected(rejected: list[DiskDetection], detected: list[DiskDetection]) -> list[DiskDetection]:
    """Rejeitadas que continuam a aparecer na última deteção (falsos positivos teimosos)."""
    return [shape for shape in rejected if any(same_shape(shape, found) for found in detected)]


def nudge_deltas(
    deltas: dict[str, float], event: str, config: AstroFrameConfig | None = None
) -> dict[str, float]:
    """Aplica recompensa (`accept`) ou punição (`reject`) aos deltas acumulados.

    Com `config` fornecido, cada delta é limitado à gama que mantém o valor
    efetivo (base + delta) dentro de `DELTA_BOUNDS`.
    """
    table = REWARD_DELTAS if event == "accept" else PUNISH_DELTAS
    out = dict(deltas)
    for key, delta in table.items():
        value = out.get(key, 0.0) + delta
        if config is not None:
            base_value = float(getattr(config.stabilizer, key))
            low, high = DELTA_BOUNDS[key]
            value = min(high - base_value, max(low - base_value, value))
        out[key] = value
    return out


def effective_params(config: AstroFrameConfig, deltas: dict[str, float]) -> dict[str, int | float]:
    """Parâmetros de deteção resultantes de aplicar os deltas aprendidos.

    Os parâmetros inteiros são arredondados e o kernel do desfoque fica
    sempre ímpar; `max_radius` entra sem delta (apenas limitado à gama).
    O clamp vem do registry unificado (`astroframe.ai.params`).
    """
    base = config.stabilizer
    params: dict[str, int | float] = {}
    for key in TRAINABLE_PARAMS:
        spec = pparams.spec_by_name(key)
        params[key] = pparams.clamp_value(spec.path, float(getattr(base, key)) + deltas.get(key, 0.0))
    params["max_radius"] = pparams.clamp_value("stabilizer.max_radius", float(base.max_radius))
    return params


def apply_effective(config: AstroFrameConfig, deltas: dict[str, float]) -> None:
    """Aplica os deltas aprendidos ao `config` (mutação de `config.stabilizer`)."""
    for key, value in effective_params(config, deltas).items():
        setattr(config.stabilizer, key, value)


def best_gt_iou(shape: DiskDetection, ground_truth: list[DiskDetection]) -> float | None:
    """IoU máximo da forma contra o guia manual, ou None sem guia."""
    if not ground_truth:
        return None
    return max(shape_iou(shape, gt) for gt in ground_truth)


def auto_iou_threshold(
    n: int,
    start: float = AUTO_IOU_START,
    end: float = AUTO_IOU_END,
    rate: float = AUTO_IOU_RATE,
) -> float:
    """Limiar IoU exigido após `n` validações acumuladas.

    Cresce de `start` (≥ `AUTO_IOU_MIN`) para `end` exponencialmente — nunca
    desce abaixo do mínimo ajustável, só sobe a dificuldade com o treino.
    """
    if n <= 0:
        return float(start)
    return float(min(end, start + (end - start) * (1.0 - math.exp(-n / rate))))


# --------------------------------------------------------------------------
# Persistência do treino
# --------------------------------------------------------------------------


class ValidatorState:
    """Progresso do treino em JSON: deltas aprendidos + julgamentos por amostra.

    Ficheiros inexistentes, JSON inválido ou versões desconhecidas resultam
    num estado vazio (nunca levantam exceções), como `CalibrationStore`.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.deltas: dict[str, float] = {}
        self.rewards = 0
        self.punishments = 0
        self.samples: dict[str, dict] = {}
        self.round = 0
        self.rounds: list[dict] = []
        self.weights: dict = self._default_weights()
        self.cnn_positives = 0
        self.cnn_negatives = 0
        self.cnn_series: list[dict] = []
        self.load()

    @staticmethod
    def _default_weights() -> dict:
        """Pesos e IoU guardados ('Salvar' no relatório final).

        Os pesos sobrevivem a `--reset-state` (o treino em si é apagado,
        mas a preferência do utilizador fica) e são aplicados no arranque.
        """
        return {
            "reward": {key: REWARD_DELTAS.get(key, 0.0) for key in TRAINABLE_PARAMS},
            "punish": {key: PUNISH_DELTAS.get(key, 0.0) for key in TRAINABLE_PARAMS},
            "iou": float(AUTO_IOU_MIN),
        }

    def load(self) -> None:
        self.deltas = {}
        self.rewards = 0
        self.punishments = 0
        self.samples = {}
        self.round = 0
        self.rounds = []
        self.weights = self._default_weights()
        self.cnn_positives = 0
        self.cnn_negatives = 0
        self.cnn_series = []
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            logger.warning("Estado de validação ilegível (%s), a começar vazio: %s", self.path, exc)
            return
        if not isinstance(data, dict) or data.get("version") not in (1, STATE_VERSION):
            logger.warning("Estado de validação com versão desconhecida (%s), a começar vazio", self.path)
            return
        self.deltas = {k: float(v) for k, v in (data.get("deltas") or {}).items()}
        self.rewards = int(data.get("rewards", 0))
        self.punishments = int(data.get("punishments", 0))
        self.round = int(data.get("round", 0))
        self.rounds = list(data.get("rounds") or [])
        self.cnn_positives = int(data.get("cnn_positives", 0))
        self.cnn_negatives = int(data.get("cnn_negatives", 0))
        self.cnn_series = list(data.get("cnn_series") or [])
        raw_weights = data.get("weights")
        if isinstance(raw_weights, dict):
            for kind, table in (("reward", self.weights["reward"]), ("punish", self.weights["punish"])):
                raw = raw_weights.get(kind)
                if isinstance(raw, dict):
                    for key in table:
                        if key in raw:
                            table[key] = float(raw[key])
            if isinstance(raw_weights.get("iou"), (int, float)):
                self.weights["iou"] = float(raw_weights["iou"])
        for key, raw in (data.get("samples") or {}).items():
            if not isinstance(raw, dict):
                continue
            record = {
                "accepted": [circle_from_dict(c) for c in raw.get("accepted", [])],
                "rejected": [circle_from_dict(c) for c in raw.get("rejected", [])],
                "done": bool(raw.get("done", False)),
            }
            self.samples[key] = record

    def save(self) -> None:
        data = {
            "version": STATE_VERSION,
            "deltas": self.deltas,
            "rewards": self.rewards,
            "punishments": self.punishments,
            "round": self.round,
            "rounds": self.rounds,
            "weights": self.weights,
            "cnn_positives": self.cnn_positives,
            "cnn_negatives": self.cnn_negatives,
            "cnn_series": self.cnn_series,
            "samples": {
                key: {
                    "accepted": [circle_to_dict(c) for c in record["accepted"]],
                    "rejected": [circle_to_dict(c) for c in record["rejected"]],
                    "done": record["done"],
                }
                for key, record in self.samples.items()
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def record(self, key: str) -> dict:
        """Registo da amostra (criado em falta) — as listas são `DiskDetection`."""
        record = self.samples.get(key)
        if record is None:
            record = {"accepted": [], "rejected": [], "done": False}
            self.samples[key] = record
        return record

    def is_done(self, key: str) -> bool:
        record = self.samples.get(key)
        return bool(record and record.get("done"))

    def reset(self) -> None:
        """Apaga o treino (progresso, deltas, séries) — mantém os **pesos**
        gravados com 'Salvar', que sobrevivem de propósito a `--reset-state`."""
        self.deltas = {}
        self.rewards = 0
        self.punishments = 0
        self.samples = {}
        self.round = 0
        self.rounds = []
        self.cnn_positives = 0
        self.cnn_negatives = 0
        self.cnn_series = []
        self.save()

    def done_count(self, samples: list[SampleRef]) -> int:
        return sum(1 for sample in samples if self.is_done(sample.key))

    def clear_progress(self) -> None:
        """Repõe o progresso das amostras (mantém deltas e contadores)."""
        self.samples = {}
        self.save()

    def begin_round(self, mode: str) -> int:
        """Começa uma série nova: incrementa o número, tira um instantâneo dos
        deltas/contadores atuais (para o balanço da série) e repõe o progresso.

        O JSON da série anterior (deltas acumulados) já está no estado e é o
        ponto de partida desta nova série.
        """
        self.round += 1
        self.rounds.append(
            {
                "round": self.round,
                "mode": mode,
                "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "ended": None,
                "rewards0": self.rewards,
                "punishments0": self.punishments,
                "deltas0": dict(self.deltas),
                "rewards": 0,
                "punishments": 0,
                "deltas": dict(self.deltas),
                "score": None,
                "recall": None,
                "precision": None,
                "mean_iou": None,
            }
        )
        self.clear_progress()
        return self.round

    def end_round(self, metrics: dict | None = None) -> None:
        """Fecha a série atual com o balanço e as métricas vs guia."""
        if not self.rounds or self.rounds[-1].get("round") != self.round:
            return
        record = self.rounds[-1]
        record["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
        record["rewards"] = self.rewards - record.get("rewards0", 0)
        record["punishments"] = self.punishments - record.get("punishments0", 0)
        record["deltas"] = dict(self.deltas)
        if metrics:
            for key in ("score", "recall", "precision", "mean_iou"):
                if metrics.get(key) is not None:
                    record[key] = metrics[key]
        self.save()


def apply_state_weights(state: ValidatorState) -> None:
    """Aplica os pesos/IoU guardados no estado às tabelas de treino.

    Chamado no arranque (janela e `--check`): os valores
    gravados com 'Salvar' no relatório final substituem os pesos por
    omissão até nova edição.
    """
    for key, value in state.weights.get("reward", {}).items():
        REWARD_DELTAS[key] = float(value)
    for key, value in state.weights.get("punish", {}).items():
        PUNISH_DELTAS[key] = float(value)


def state_iou(state: ValidatorState) -> float:
    """IoU mínimo guardado no estado (aplicado no arranque)."""
    return max(AUTO_IOU_MIN, min(float(state.weights.get("iou", AUTO_IOU_MIN)), AUTO_IOU_MAX))


# --------------------------------------------------------------------------
# Sessão de validação (lógica do loop, testável sem ecrã)
# --------------------------------------------------------------------------


class ValidationSession:
    """Máquina do treino: julgamento, punição/recompensa e reavaliação.

    As ações devolvem uma string: `"present"` (há forma pendente para
    mostrar), `"redetect"` (reavaliar a imagem com os parâmetros novos) ou
    `"complete"` (amostra terminada).
    """

    def __init__(
        self,
        samples: list[SampleRef],
        store: CalibrationStore,
        config: AstroFrameConfig,
        state: ValidatorState,
    ):
        self.samples = samples
        self.store = store
        self.config = config
        self.state = state
        self.current_index = 0
        self.accepted: list[DiskDetection] = []
        self.rejected: list[DiskDetection] = []
        self.detected: list[DiskDetection] = []
        self.pending: list[DiskDetection] = []
        self.reevals = 0
        self.judgments = 0

    def detect_config(self) -> AstroFrameConfig:
        """Cópia da configuração com os deltas aprendidos aplicados à deteção.

        A configuração original fica intacta (é a base fixa dos deltas); cada
        deteção usa esta cópia, senão a base deslizava e corrompia o treino.
        """
        adjusted = copy.deepcopy(self.config)
        apply_effective(adjusted, self.state.deltas)
        return adjusted

    @property
    def sample(self) -> SampleRef:
        return self.samples[self.current_index]

    @property
    def ground_truth(self) -> list[DiskDetection]:
        item = self.store.get_item(self.sample.key)
        return list(item.circles) if item is not None else []

    @property
    def current(self) -> DiskDetection | None:
        return self.pending[0] if self.pending else None

    def start(self, index: int) -> None:
        self.current_index = index
        record = self.state.record(self.sample.key)
        self.accepted = list(record["accepted"])
        self.rejected = list(record["rejected"])
        self.detected = []
        self.pending = []
        self.reevals = 0
        self.judgments = 0

    def apply_detection(self, shapes: list[DiskDetection]) -> str:
        """Resultado da deteção: decide apresentar, punir+reavaliar ou terminar."""
        self.detected = list(shapes)
        self.pending = filter_unjudged(shapes, self.accepted, self.rejected)
        if self.pending:
            return "present"
        stubborn = persistent_rejected(self.rejected, shapes)
        if stubborn and self._can_redetect():
            for _ in stubborn:
                self._punish()
            return "redetect"
        return "complete"

    def accept(self) -> str:
        return self._judge(True)

    def reject(self) -> str:
        return self._judge(False)

    def _judge(self, valid: bool) -> str:
        if not self.pending:
            return "complete"
        shape = self.pending.pop(0)
        self.judgments += 1
        if valid:
            self.accepted.append(shape)
            self.state.rewards += 1
            self.state.deltas = nudge_deltas(self.state.deltas, "accept", self.config)
        else:
            self.rejected.append(shape)
            self.state.punishments += 1
            self.state.deltas = nudge_deltas(self.state.deltas, "reject", self.config)
        self._persist()
        if self.judgments >= MAX_JUDGMENTS_PER_SAMPLE:
            return "complete"
        if self.pending:
            return "present"
        if self._can_redetect():
            return "redetect"
        return "complete"

    def complete(self) -> None:
        record = self.state.record(self.sample.key)
        record["accepted"] = list(self.accepted)
        record["rejected"] = list(self.rejected)
        record["done"] = True
        self.state.save()

    def _can_redetect(self) -> bool:
        if self.reevals >= MAX_REEVALS_PER_SAMPLE:
            return False
        self.reevals += 1
        return True

    def _punish(self) -> None:
        self.state.punishments += 1
        self.state.deltas = nudge_deltas(self.state.deltas, "reject", self.config)

    def _persist(self) -> None:
        record = self.state.record(self.sample.key)
        record["accepted"] = list(self.accepted)
        record["rejected"] = list(self.rejected)
        record["done"] = False
        self.state.save()


# --------------------------------------------------------------------------
# Relatórios
# --------------------------------------------------------------------------


def sample_done_text(session: ValidationSession) -> str:
    """Resumo de uma amostra terminada (julgamentos + concordância com o guia)."""
    gt = session.ground_truth
    n_accepted, n_rejected = len(session.accepted), len(session.rejected)
    if gt:
        item = validate_item(session.sample.label, gt, session.accepted)
        iou = f"{item.mean_iou:.2f}" if item.mean_iou is not None else "—"
        return (
            f"✓ {session.sample.label}: {n_accepted} válida(s) + {n_rejected} rejeitada(s) "
            f"· guia: {item.n_matched}/{item.n_manual} · IoU {iou}"
        )
    return f"✓ {session.sample.label}: {n_accepted} válida(s) + {n_rejected} rejeitada(s) · sem guia"


def build_global_report(
    samples: list[SampleRef], store: CalibrationStore, state: ValidatorState
) -> tuple[CalibrationReport, list[str]] | None:
    """Relatório final: resultado aceite vs calibration.json, por amostra concluída."""
    rows: list[tuple[str, list[DiskDetection], list[DiskDetection]]] = []
    for sample in samples:
        record = state.samples.get(sample.key)
        if not record or not record.get("done"):
            continue
        item = store.get_item(sample.key)
        gt = list(item.circles) if item is not None else []
        rows.append((sample.label, gt, list(record["accepted"])))
    if not rows:
        return None
    report = validate_all(rows)
    lines = [
        (
            f"Score global vs guia manual: {report.score:.1f}/100"
            if report.score is not None
            else "Sem guia (calibration.json) para pontuar."
        ),
        (
            f"Recall {report.recall * 100:.0f}% · Precisão {report.precision * 100:.0f}%"
            f" · IoU médio {report.mean_iou:.2f}"
        ),
        f"Total: {report.total_matched} emparelhado(s), "
        f"{report.total_false_negatives} falso(s) negativo(s), "
        f"{report.total_false_positives} falso(s) positivo(s).",
    ]
    if report.mean_center_error is not None:
        lines.append(f"Erro médio do centro: {report.mean_center_error:.1f} px")
    if report.mean_radius_error_pct is not None:
        lines.append(f"Erro médio do raio: {report.mean_radius_error_pct:+.0f}%")
    lines.append("")
    for item in report.items:
        iou = f"{item.mean_iou:.2f}" if item.mean_iou is not None else "—"
        lines.append(
            f"{item.label}: {item.n_matched}/{item.n_manual} · IoU {iou} · "
            f"FN {item.n_false_negatives} FP {item.n_false_positives}"
        )
    return report, lines


def rounds_text(state: ValidatorState) -> list[str]:
    """Histórico das séries concluídas (para o relatório final)."""
    lines = []
    for record in state.rounds:
        if record.get("ended") is None:
            continue
        score = f"{record['score']:.1f}" if record.get("score") is not None else "—"
        delta_p2 = record["deltas"].get("param2", 0.0)
        lines.append(
            f"Série {record['round']} ({record.get('mode', '?')}): "
            f"+{record.get('rewards', 0)} ✓ / {record.get('punishments', 0)} ✗ · "
            f"score {score} · Δ param2 {delta_p2:+.1f}"
        )
    return lines


def export_trained(
    state: ValidatorState,
    config: AstroFrameConfig,
    path: str | Path | None = None,
    report: CalibrationReport | None = None,
) -> Path:
    """Exporta a configuração treinada (JSON) para alimentar o sistema real.

    O ficheiro tem os parâmetros efetivos da deteção (base + deltas
    aprendidos), os deltas, as estatísticas e o score vs guia — pode ser
    consultado pela pipeline ou fundido num `config.yaml`.
    """
    path = Path(path)
    params = effective_params(config, state.deltas)
    full_deltas = {}
    for key, value in state.deltas.items():
        try:
            full_deltas[pparams.spec_by_name(key).path] = value
        except KeyError:
            continue
    effective = pparams.apply_deltas(config, full_deltas)
    data: dict = {
        "version": STATE_VERSION,
        "kind": "astroframe-trained",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rounds": state.round,
        "stats": {"rewards": state.rewards, "punishments": state.punishments},
        "deltas": dict(state.deltas),
        "stabilizer": params,
        "params": {p: pparams.get_param(effective, p) for p in pparams.PARAM_SPECS},
    }
    if report is not None and report.score is not None:
        data["score"] = {
            "score": report.score,
            "recall": report.recall,
            "precision": report.precision,
            "mean_iou": report.mean_iou,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def metrics_from_report(report: CalibrationReport | None) -> dict | None:
    """Métricas vs guia de um relatório, para `ValidatorState.end_round`."""
    if report is None or report.score is None:
        return None
    return {
        "score": report.score,
        "recall": report.recall,
        "precision": report.precision,
        "mean_iou": report.mean_iou,
    }


# --------------------------------------------------------------------------
# Treino automático
# --------------------------------------------------------------------------


@dataclass
class AutoSeriesReport:
    """Balanço de uma série automática."""

    series: int
    samples_done: int
    samples_total: int
    rewards: int
    punishments: int
    threshold_end: float
    report: CalibrationReport | None = None
    errors: list[str] = field(default_factory=list)
    stopped: bool = False
    cnn_positives: int = 0
    cnn_negatives: int = 0


class AutoTrainer:
    """Treino automático: valida/rejeita deteções comparando-as com o guia.

    Cada deteção é comparada com a calibração manual: se o IoU máximo contra
    o guia for ≥ `iou_threshold` (mínimo sempre ≥ 0.90, ajustável pela UI
    para aumentar a dificuldade), é aceite (recompensa); senão é rejeitada
    (punição + reavaliação da imagem), no mesmo esquema do treino manual.
    O limiar começa no mínimo escolhido e sobe até `AUTO_IOU_END` conforme o
    número de validações acumuladas. Amostras sem guia são concluídas sem
    julgamentos (não há critério).

    Com `collect_patches`, a série recolhe **patches para a CNN de deteção**:
    positivos (círculos do guia) e negativos (formas rejeitadas + recortes
    aleatórios que não tocam o guia). Com `cnn_model_path` (o campeão da
    série anterior), os candidatos do Hough são filtrados por confiança da
    CNN antes do julgamento — a lista detetada nunca é esvaziada.
    """

    def __init__(
        self,
        samples: list[SampleRef],
        store: CalibrationStore,
        config: AstroFrameConfig,
        state: ValidatorState,
        iou_threshold: float = AUTO_IOU_MIN,
        collect_patches: bool = True,
        cnn_model_path: str | Path | None = None,
        cnn_threshold: float = CNN_THRESHOLD_DEFAULT,
        seed: int = 42,
    ):
        self.samples = samples
        self.store = store
        self.config = config
        self.state = state
        # Mínimo ajustável pela UI/CLI: nunca abaixo de 0.90.
        self.iou_threshold = max(AUTO_IOU_MIN, min(float(iou_threshold), AUTO_IOU_MAX))
        self.iou_start = self.iou_threshold
        self.iou_end = AUTO_IOU_MAX
        self.iou_rate = AUTO_IOU_RATE
        self.collect_patches = collect_patches
        self.cnn_threshold = float(cnn_threshold)
        self.seed = int(seed)
        self.positives: list[np.ndarray] = []
        self.negatives: list[np.ndarray] = []
        if collect_patches and cnn_model_path:
            model = SmallCNN.load(cnn_model_path)
            self._filter = DiskFilter(model=model) if model is not None else None
        else:
            self._filter = None

    def _random_negatives(self, frame: np.ndarray, gt: list[DiskDetection], label: str) -> None:
        """Recortes aleatórios determinísticos que não tocam o guia."""
        h, w = frame.shape[:2]
        if h < 32 or w < 32:
            return
        rng = np.random.default_rng(self.seed + zlib.crc32(label.encode("utf-8")))
        max_r = max(8, min(h, w) // 12)
        for _ in range(CNN_RANDOM_NEGATIVES_PER_SAMPLE):
            radius = int(rng.integers(8, max_r + 1))
            cx = int(rng.integers(radius, w - radius))
            cy = int(rng.integers(radius, h - radius))
            candidate = DiskDetection(cx, cy, radius)
            iou = best_gt_iou(candidate, gt)
            if iou is not None and iou >= 0.3:
                continue
            self.negatives.append(disk_patch(frame, cx, cy, radius))

    def run_series(
        self,
        progress: callable | None = None,
        should_stop: callable | None = None,
        on_detect: callable | None = None,
    ) -> AutoSeriesReport:
        """Uma passagem completa por todas as amostras, sem intervenção humana.

        `on_detect(frame, detected, label, threshold)` é chamado a cada
        deteção (inicial e reavaliações) — a UI usa-o para a pré-visualização
        em tempo real.
        """
        self.state.clear_progress()
        session = ValidationSession(self.samples, self.store, self.config, self.state)
        rewards0 = self.state.rewards
        punishments0 = self.state.punishments
        threshold = self.iou_start
        done = 0
        stopped = False
        errors: list[str] = []
        cnn_positives0 = len(self.positives)
        cnn_negatives0 = len(self.negatives)
        for idx in range(len(self.samples)):
            if should_stop is not None and should_stop():
                stopped = True
                break
            sample = self.samples[idx]
            session.start(idx)
            gt = session.ground_truth
            if not gt:
                session.complete()
                done += 1
                if progress is not None:
                    progress(done, len(self.samples), sample.label, threshold)
                continue
            try:
                frame = load_frame(sample)
            except Exception as exc:
                errors.append(f"{sample.label}: erro ao ler ({exc})")
                continue
            if self.collect_patches:
                h, w = frame.shape[:2]
                for circle in gt:
                    if 0 <= circle.cx < w and 0 <= circle.cy < h and circle.radius > 0:
                        self.positives.append(disk_patch(frame, circle.cx, circle.cy, circle.radius))
                self._random_negatives(frame, gt, sample.label)
            detected = find_disks_for_calibration(frame, session.detect_config(), expected_n=len(gt))
            if self._filter is not None:
                detected = self._filter.filter_disks(detected, frame, self.cnn_threshold)
            if on_detect is not None:
                on_detect(frame, detected, sample.label, threshold)
            while True:
                shape = session.current
                action = session.apply_detection(detected)
                if action == "present":
                    threshold = auto_iou_threshold(
                        self.state.rewards + self.state.punishments,
                        self.iou_start,
                        self.iou_end,
                        self.iou_rate,
                    )
                    iou = best_gt_iou(session.current, gt)
                    rejected = iou is None or iou < threshold
                    action = session.reject() if rejected else session.accept()
                    if rejected and self.collect_patches and shape is not None:
                        self.negatives.append(disk_patch(frame, shape.cx, shape.cy, shape.radius))
                    if action == "present":
                        continue
                if action == "redetect":
                    detected = find_disks_for_calibration(frame, session.detect_config(), expected_n=len(gt))
                    if self._filter is not None:
                        detected = self._filter.filter_disks(detected, frame, self.cnn_threshold)
                    if on_detect is not None:
                        on_detect(frame, detected, sample.label, threshold)
                    continue
                break
            session.complete()
            done += 1
            if progress is not None:
                progress(done, len(self.samples), sample.label, threshold)
        result = build_global_report(self.samples, self.store, self.state)
        return AutoSeriesReport(
            series=self.state.round,
            samples_done=done,
            samples_total=len(self.samples),
            rewards=self.state.rewards - rewards0,
            punishments=self.state.punishments - punishments0,
            threshold_end=threshold,
            report=result[0] if result else None,
            errors=errors,
            stopped=stopped,
            cnn_positives=len(self.positives) - cnn_positives0,
            cnn_negatives=len(self.negatives) - cnn_negatives0,
        )


# --------------------------------------------------------------------------
# Rede neuronal da deteção (treino entre séries)
# --------------------------------------------------------------------------


def classifier_accuracy(
    model: SmallCNN,
    positives: list[np.ndarray],
    negatives: list[np.ndarray],
    seed: int,
) -> float:
    """Precisão do classificador numa divisão 80/20 determinística."""
    X = np.concatenate([np.array(positives), np.array(negatives)], axis=0)
    y = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
    if len(X) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    n_val = max(1, int(len(X) * 0.2))
    val_idx = order[:n_val]
    batch = np.stack([X[i] for i in val_idx])[:, None]
    probs = model.predict_class(batch)
    labels = y[val_idx].astype(int)
    return float(np.mean((probs >= 0.5) == (labels == 1)))


def train_classifier_round(
    positives: list[np.ndarray],
    negatives: list[np.ndarray],
    state: ValidatorState,
    round_n: int,
    score: float | None,
    epochs: int = 60,
    champion_path: str | Path | None = None,
    db: FeedbackDB | None = None,
    seed: int = 42,
) -> dict:
    """Treina/continua o classificador CNN com os patches acumulados.

    Warm-start a partir do campeão (`champion_path`) quando existe; o
    candidato é guardado em staging (`Logs/weights/staging/`) e **comparado
    com o melhor registado no banco** (`add_model`): se for melhor promove-o
    (copia para o caminho canónico `Logs/weights/disk_filter.npz`); se for
    pior, a série seguinte parte dos pesos do campeão (warm-start). Sem
    patches suficientes devolve `None` (o Hough segue como hoje). O relatório
    de cada ronda fica em `Logs/logs/ia/`.
    """
    if len(positives) < 2 or len(negatives) < 2:
        if db is not None:
            db.log("info", "validator", f"Série {round_n}: sem patches suficientes para a CNN")
        return {"skipped": True}
    warm = SmallCNN.load(champion_path) if champion_path else None
    model, fit = fit_classifier(positives, negatives, model=warm, epochs=epochs, seed=seed)
    accuracy = classifier_accuracy(model, positives, negatives, seed)
    staged = CNN_MODEL_DIR / f"disk_filter_r{round_n}.npz"
    staged.parent.mkdir(parents=True, exist_ok=True)
    model.save(staged)
    metrics = {
        "score": float(score) if score is not None else 0.0,
        "accuracy": accuracy,
        "best_loss": float(fit.best_loss),
    }
    metric_name = "score" if score is not None else "accuracy"
    db = db or FeedbackDB()
    result = db.add_model(
        "disk_filter",
        staged,
        metrics,
        dataset_size=len(positives) + len(negatives),
        source="validator-auto",
        round=round_n,
        metric_name=metric_name,
    )
    champion = result["champion"]
    score_text = f"{score:.1f}" if score is not None else "—"
    if result["promoted"]:
        shutil.copyfile(staged, CNN_CANONICAL_PATH)
        db.log("info", "validator", f"Série {round_n}: novo campeão CNN (score {score_text})", metrics)
    else:
        db.log(
            "info",
            "validator",
            f"Série {round_n}: CNN pior que o campeão (score {score_text}); próxima série parte do campeão",
            metrics,
        )
    record = {
        "round": round_n,
        "accuracy": accuracy,
        "score": score,
        "promoted": bool(result["promoted"]),
        "dataset": len(positives) + len(negatives),
    }
    report_path = logs_ia_dir() / f"disk_filter_round_{round_n}.json"
    report_path.write_text(json.dumps({**record, **metrics}, indent=2, ensure_ascii=False), encoding="utf-8")
    state.cnn_series.append(record)
    state.save()
    return {
        "skipped": False,
        "accuracy": accuracy,
        "promoted": bool(result["promoted"]),
        "staged": staged,
        "champion_path": champion["path"] if champion else None,
    }


# --------------------------------------------------------------------------
# Interface desktop (tkinter)
# --------------------------------------------------------------------------


class ValidatorTkApp:
    """Janela de validação: imagem + formas detetadas e botões Válido/Rejeitado."""

    def __init__(
        self,
        root: tk.Tk,
        samples: list[SampleRef],
        store: CalibrationStore,
        config: AstroFrameConfig,
        state: ValidatorState,
        samples_root: str | Path | None = None,
    ):
        if tk is None:  # pragma: no cover
            raise RuntimeError(f"tkinter indisponível: {_TK_IMPORT_ERROR}")
        self.root = root
        self.samples = samples
        self.store = store
        self.config = config
        self.state = state
        self.samples_root = Path(samples_root) if samples_root else None

        self.session = ValidationSession(samples, store, config, state)
        apply_state_weights(state)
        self.auto_iou_min = state_iou(state)
        self.frame: np.ndarray | None = None
        self.img_w = 0
        self.img_h = 0
        self.scale = 1.0
        self.ox = 0.0
        self.oy = 0.0
        self._photo: ImageTk.PhotoImage | None = None
        self._photo_scale: float | None = None
        self._pan_start: tuple[int, int] | None = None

        self._queue: queue.Queue[tuple] = queue.Queue()
        self._busy = False
        self._job_id = 0
        self._pending_shown: list[DiskDetection] = []
        self._training = False

        self._build_ui()
        self.root.after(50, self._poll_queue)
        if self.state.round == 0:
            self.state.begin_round("manual")
        if self.samples:
            self.goto(self._first_undone())
        else:  # pragma: no cover
            self.status.set("Sem amostras na pasta de exemplos.")

    # ---------------------------------------------------------------- UI --

    def _build_ui(self) -> None:
        self.root.title("AstroFrame — Validação (treino da deteção)")
        self.root.geometry("1280x800")

        paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(paned, bg="#1c1c1e", highlightthickness=0)
        paned.add(self.canvas, weight=3)

        panel = ttk.Frame(paned, padding=8, width=320)
        paned.add(panel, weight=1)

        row = 0
        ttk.Label(panel, text=f"Amostras ({len(self.samples)})", font=("", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1

        self.listbox = tk.Listbox(panel, height=8, activestyle="none", exportselection=False)
        self.listbox.grid(row=row, column=0, columnspan=2, sticky="ew")
        sb = ttk.Scrollbar(panel, orient=tk.VERTICAL, command=self.listbox.yview)
        sb.grid(row=row, column=2, sticky="ns")
        self.listbox.configure(yscrollcommand=sb.set)
        for sample in self.samples:
            done = "✓ " if self.state.is_done(sample.key) else "· "
            self.listbox.insert(tk.END, done + sample.label)
        self.listbox.bind("<<ListboxSelect>>", self._on_listbox_select)
        row += 1

        nav = ttk.Frame(panel)
        nav.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(nav, text="◀ Anterior", command=lambda: self.goto(-1, relative=True)).pack(side=tk.LEFT)
        ttk.Button(nav, text="Seguinte ▶", command=lambda: self.goto(+1, relative=True)).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Button(nav, text="Próx. pendente", command=self.goto_next_undone).pack(side=tk.RIGHT)
        row += 1

        self.progress = ttk.Progressbar(panel, maximum=max(1, len(self.samples)))
        self.progress.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        row += 1
        self.progress_label = ttk.Label(panel, text="")
        self.progress_label.grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        ttk.Separator(panel).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1

        self.pending_info = ttk.Label(
            panel, text="", foreground="#0a7", wraplength=300, font=("", 11, "bold")
        )
        self.pending_info.grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        self.gt_hint = ttk.Label(panel, text="", foreground="#666", wraplength=300)
        self.gt_hint.grid(row=row, column=0, columnspan=2, sticky="w", pady=(2, 6))
        row += 1

        buttons = ttk.Frame(panel)
        buttons.grid(row=row, column=0, columnspan=2, sticky="ew")
        self.valid_btn = tk.Button(
            buttons,
            text="✓ Válido  [V]",
            command=self.accept,
            bg="#1e7a3a",
            fg="white",
            font=("", 11, "bold"),
            padx=12,
            pady=6,
        )
        self.valid_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.reject_btn = tk.Button(
            buttons,
            text="✗ Rejeitado  [R]",
            command=self.reject,
            bg="#9c2b2b",
            fg="white",
            font=("", 11, "bold"),
            padx=12,
            pady=6,
        )
        self.reject_btn.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(6, 0))
        row += 1

        ttk.Separator(panel).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1

        self.stats_label = ttk.Label(panel, text="", wraplength=300)
        self.stats_label.grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        self.params_label = ttk.Label(panel, text="", foreground="#555", wraplength=300)
        self.params_label.grid(row=row, column=0, columnspan=2, sticky="w", pady=(2, 0))
        row += 1

        ttk.Separator(panel).grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        row += 1
        self.stars_label_var = tk.StringVar(value="")
        self.stars_label = ttk.Label(panel, textvariable=self.stars_label_var, font=("", 10, "bold"), foreground="#c90")
        self.stars_label.grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Label(panel, text="Avaliação manual (arrasta):", foreground="#555").grid(
            row=row, column=0, sticky="w"
        )
        self.stars_var = tk.DoubleVar(value=3.0)
        self.stars_scale = ttk.Scale(
            panel, from_=0.0, to=5.0, variable=self.stars_var, orient=tk.HORIZONTAL, command=self._on_stars_change
        )
        self.stars_scale.grid(row=row, column=1, sticky="ew")
        row += 1
        self.stars_detail_var = tk.StringVar(value="")
        self.stars_detail = ttk.Label(panel, textvariable=self.stars_detail_var, foreground="#666", wraplength=300)
        self.stars_detail.grid(row=row, column=0, columnspan=2, sticky="w", pady=(2, 0))
        row += 1

        ttk.Button(panel, text="Recomeçar amostra", command=self.restart_sample).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(6, 0)
        )
        row += 1

        train_row = ttk.Frame(panel)
        train_row.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.train_btn = ttk.Button(train_row, text="Novo treino", command=self.new_round)
        self.train_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.auto_btn = ttk.Button(train_row, text="Treino automático…", command=self.open_auto_train)
        self.auto_btn.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(6, 0))
        row += 1
        ttk.Label(
            panel,
            text="Novo treino = série seguinte com os ajustes desta série.",
            foreground="#666",
            wraplength=300,
        ).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        self.status = tk.StringVar(value="")
        ttk.Label(panel, textvariable=self.status, foreground="#0a7", wraplength=300).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        row += 1

        self.report = tk.Text(panel, height=12, width=40, state=tk.DISABLED)
        self.report.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        row += 1

        panel.columnconfigure(1, weight=1)

        self.canvas.bind("<ButtonPress-1>", self._on_press_pan)
        self.canvas.bind("<B1-Motion>", self._on_drag_pan)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, 1.0 / 1.15))
        self.canvas.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, 1.15))

        for key in ("v", "V"):
            self.root.bind(f"<Key-{key}>", lambda _e: self.accept())
        for key in ("r", "R"):
            self.root.bind(f"<Key-{key}>", lambda _e: self.reject())
        self.root.bind("<Prior>", lambda e: self.goto(-1, relative=True))
        self.root.bind("<Next>", lambda e: self.goto(+1, relative=True))

    # ------------------------------------------------------------ amostras --

    def _first_undone(self) -> int:
        for i, sample in enumerate(self.samples):
            if not self.state.is_done(sample.key):
                return i
        return 0

    def goto_next_undone(self) -> None:
        for offset in range(1, len(self.samples) + 1):
            index = (self.session.current_index + offset) % len(self.samples)
            if not self.state.is_done(self.samples[index].key):
                self.goto(index)
                return

    def goto(self, index_or_delta: int, relative: bool = False) -> None:
        if not self.samples:
            return
        index = (
            (self.session.current_index + index_or_delta) % len(self.samples) if relative else index_or_delta
        )
        self.load_sample(index)

    def _on_listbox_select(self, _event=None) -> None:
        selection = self.listbox.curselection()
        if selection:
            self.load_sample(selection[0])

    def load_sample(self, index: int) -> None:
        sample = self.samples[index]
        self.session.start(index)
        self._job_id += 1
        self._busy = False
        self._pending_shown = []
        self._report("")
        self.stars_var.set(3.0)
        self.stars_label_var.set("")
        self.stars_detail_var.set(stars_text(3.0))
        try:
            frame = load_frame(sample)
        except Exception as exc:
            self.status.set(f"Erro ao carregar: {exc}")
            return
        self.frame = frame
        self.img_h, self.img_w = frame.shape[:2]

        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(index)
        self.listbox.see(index)
        self.fit_view()
        self.redraw()
        self._update_labels()

        if self.state.is_done(sample.key):
            self.status.set(
                f"Amostra completa (✓ {len(self.session.accepted)} válida(s), "
                f"✗ {len(self.session.rejected)} rejeitada(s))."
            )
            self._set_buttons_enabled(False)
            return
        self._set_buttons_enabled(False)
        self.status.set("A analisar imagem…")
        self._start_detect()

    # ------------------------------------------------------------- deteção --

    def _start_detect(self) -> None:
        job_id = self._job_id
        frame = self.frame.copy() if self.frame is not None else None
        detect_config = self.session.detect_config()
        expected_n = len(self.session.ground_truth)
        messages = self._queue

        def work() -> None:
            try:
                assert frame is not None
                detected = find_disks_for_calibration(frame, detect_config, expected_n=expected_n)
                messages.put(("detect", job_id, detected, None))
            except Exception as exc:  # pragma: no cover
                messages.put(("detect", job_id, None, exc))

        self._busy = True
        threading.Thread(target=work, daemon=True).start()

    def _poll_queue(self) -> None:
        try:
            while True:
                message = self._queue.get_nowait()
                kind = message[0]
                if kind == "detect":
                    self._on_detect_done(*message[1:])
        except queue.Empty:
            pass
        self.root.after(50, self._poll_queue)

    def _on_detect_done(
        self, job_id: int, detected: list[DiskDetection] | None, exc: Exception | None
    ) -> None:
        if job_id != self._job_id:
            return
        self._busy = False
        if exc is not None:
            self.status.set(f"Erro na deteção: {exc}")
            return
        action = self.session.apply_detection(detected or [])
        if action == "redetect":
            self.status.set("Rejeição aplicada — a reavaliar a imagem com os novos parâmetros…")
            self._set_buttons_enabled(False)
            self._start_detect()
            return
        if action == "complete":
            self._finalize_sample()
            return
        self._present_pending()

    # ------------------------------------------------------------ julgamento --

    def accept(self) -> None:
        if self._busy or not self.session.pending:
            return
        self._handle_action(self.session.accept())

    def reject(self) -> None:
        if self._busy or not self.session.pending:
            return
        self._handle_action(self.session.reject())

    def _handle_action(self, action: str) -> None:
        if action == "redetect":
            self.status.set("A reavaliar a imagem com os novos parâmetros…")
            self._set_buttons_enabled(False)
            self._start_detect()
            return
        if action == "complete":
            self._finalize_sample()
            return
        self._present_pending()

    def _present_pending(self) -> None:
        pending = self.session.pending
        if not pending:
            self._finalize_sample()
            return
        self._pending_shown = list(pending)
        gt = self.session.ground_truth
        iou = best_gt_iou(pending[0], gt)
        if iou is None:
            self.gt_hint.config(text="Sem guia (calibration.json) para esta amostra.", foreground="#888")
        elif iou >= 0.5:
            self.gt_hint.config(text=f"Corresponde ao guia manual (IoU {iou:.2f}).", foreground="#0a7")
        else:
            self.gt_hint.config(
                text=f"Sem correspondência no guia manual (IoU {iou:.2f}).", foreground="#a60"
            )
        self.status.set(
            f"Forma {1}/{len(pending)} pendente — centro ({pending[0].cx}, {pending[0].cy}), "
            f"raio {pending[0].radius}"
        )
        self._set_buttons_enabled(True)
        self._update_labels()
        self.redraw()

    def _finalize_sample(self) -> None:
        self._pending_shown = []
        self._set_buttons_enabled(False)
        self._apply_stars_to_sample()
        self.session.complete()
        done = self.state.done_count(self.samples)
        total = len(self.samples)
        self.status.set(f"✓ Amostra completa ({done}/{total} processadas, {total - done} pendentes).")
        self._report(sample_done_text(self.session))
        self.listbox.delete(self.session.current_index)
        self.listbox.insert(self.session.current_index, "✓ " + self.session.sample.label)
        self._update_labels()
        self.redraw()
        if done >= total:
            self._show_final_report()
        else:
            self.root.after(600, self.goto_next_undone)

    def _show_final_report(self) -> None:
        self.status.set("Treino concluído: 100% das imagens e frames processados.")
        result = build_global_report(self.samples, self.store, self.state)
        if result is None:
            self._report("Sem amostras concluídas para pontuar.")
            return
        _report, lines = result
        self.state.end_round(metrics_from_report(_report))
        export_path = export_trained(self.state, self.config, self._export_path(), _report)
        lines.insert(0, f"Série {self.state.round} (manual) concluída.")
        lines.insert(1, f"Recompensas: {self.state.rewards} · Punições: {self.state.punishments}")
        lines.insert(2, self._params_text())
        lines.insert(3, "")
        history = rounds_text(self.state)
        if history:
            lines.insert(4, "Séries anteriores:")
            for offset, line in enumerate(history, start=5):
                lines.insert(offset, line)
            lines.insert(5 + len(history), "")
        lines.extend(
            [
                "",
                f"Config treinada exportada: {export_path}",
                "Carrega em 'Novo treino' para começar a série seguinte com estes ajustes.",
            ]
        )
        self._report("\n".join(lines))
        FinalReportWindow(self, _report, self.state, lines)
        print("=" * 72)
        print("AstroFrame — validação concluída (100% processado)")
        print("\n".join(lines))
        print("=" * 72)

    def _export_path(self) -> Path:
        return Path(self.samples_root or "samples") / DEFAULT_EXPORT_NAME

    def new_round(self) -> None:
        """Série seguinte: usa o JSON desta série como ponto de partida."""
        if self._busy or self._training or not self.samples:
            return
        if not messagebox.askyesno(
            "Novo treino",
            "Começar uma nova série de treino manual?\n\n"
            "O progresso das amostras é reposto, mas os deltas aprendidos\n"
            "(pesos/configs) desta série são mantidos e usados no início\n"
            "da série seguinte.",
        ):
            return
        self.state.begin_round("manual")
        self.refresh_list()
        self._report("")
        self.status.set(
            f"Série {self.state.round}: a começar com os ajustes da série anterior — {self._params_text()}"
        )
        self.goto(self._first_undone())

    def open_auto_train(self) -> None:
        if self._training or not self.samples:
            return
        AutoTrainWindow(self)

    def set_training_active(self, active: bool) -> None:
        self._training = active
        self._set_buttons_enabled(False if active else bool(self.session.pending))
        self.train_btn.config(state="disabled" if active else "normal")
        self.auto_btn.config(state="disabled" if active else "normal")

    def training_finished(self) -> None:
        self.set_training_active(False)
        self.refresh_list()
        self._update_labels()
        self.status.set("Treino automático concluído.")

    def refresh_list(self) -> None:
        self.listbox.delete(0, tk.END)
        for sample in self.samples:
            done = "✓ " if self.state.is_done(sample.key) else "· "
            self.listbox.insert(tk.END, done + sample.label)

    def restart_sample(self) -> None:
        if self._busy or self._training:
            return
        record = self.state.record(self.session.sample.key)
        record["accepted"] = []
        record["rejected"] = []
        record["done"] = False
        self.state.save()
        self.load_sample(self.session.current_index)

    # ----------------------------------------------------------------- estrelas --

    def _on_stars_change(self, _value: str) -> None:
        stars = self.stars_var.get()
        self.stars_detail_var.set(stars_text(stars))

    def _compute_auto_stars(self) -> float:
        """Estrelas automáticas da imagem atual (0–5)."""
        if self.frame is None:
            return 0.0
        enhanced = enhance_image(self.frame, self.session.detect_config())
        detection = self.session.current if self.session.pending else (
            self.session.accepted[0] if self.session.accepted else None
        )
        return float(score_image(enhanced, detection, self.session.config).stars)

    def _apply_stars_to_sample(self) -> None:
        """Grava as estrelas (auto ou manual) no registo da amostra."""
        auto_stars = self._compute_auto_stars()
        manual_stars = self.stars_var.get()
        record = self.state.record(self.session.sample.key)
        record["stars_auto"] = round(auto_stars, 1)
        record["stars_user"] = round(manual_stars, 1)
        self.state.save()
        self.stars_label_var.set(f"{auto_stars:.1f}★ auto · {manual_stars:.1f}★ manual")

    # ------------------------------------------------------------- desenho --

    def fit_view(self) -> None:
        if self.frame is None:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 900, 700
        self.scale = max(0.05, min(cw / self.img_w, ch / self.img_h))
        self.ox = (cw - self.img_w * self.scale) / 2.0
        self.oy = (ch - self.img_h * self.scale) / 2.0
        self._photo_scale = None

    def _zoom_at(self, cx: float, cy: float, factor: float) -> None:
        if self.frame is None:
            return
        ix = (cx - self.ox) / self.scale
        iy = (cy - self.oy) / self.scale
        self.scale = max(0.02, min(40.0, self.scale * factor))
        self.ox = cx - ix * self.scale
        self.oy = cy - iy * self.scale
        self._photo_scale = None
        self.redraw()

    def _on_wheel(self, event) -> None:
        factor = 1.0 / 1.15 if event.delta > 0 else 1.15
        self._zoom_at(event.x, event.y, factor)

    def _on_press_pan(self, event) -> None:
        self._pan_start = (event.x, event.y)
        self._pan_ox, self._pan_oy = self.ox, self.oy

    def _on_drag_pan(self, event) -> None:
        if self._pan_start is None:
            return
        self.ox = self._pan_ox + (event.x - self._pan_start[0])
        self.oy = self._pan_oy + (event.y - self._pan_start[1])
        self.redraw()

    def redraw(self) -> None:
        if self.frame is None:
            return
        canvas = self.canvas
        canvas.delete("all")
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        canvas.create_rectangle(0, 0, cw, ch, fill="#1c1c1e")

        if self._photo_scale != self.scale:
            rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            size = (max(1, round(self.img_w * self.scale)), max(1, round(self.img_h * self.scale)))
            scaled = pil.resize(size, Image.LANCZOS)
            self._photo = ImageTk.PhotoImage(scaled, master=self.canvas)
            self._photo_scale = self.scale
        canvas.create_image(self.ox, self.oy, anchor=tk.NW, image=self._photo)

        for shape in self.session.ground_truth:
            self._draw_shape(shape, COLOR_GT, width=1, dash=(5, 3))
        for shape in self.session.accepted:
            self._draw_shape(shape, COLOR_ACCEPTED, width=2)
            self._draw_label(shape, "✓", COLOR_ACCEPTED)
        for shape in self.session.rejected:
            self._draw_shape(shape, COLOR_REJECTED, width=2)
            self._draw_label(shape, "✗", COLOR_REJECTED)
        for i, shape in enumerate(self.session.pending):
            color = COLOR_CURRENT if i == 0 else COLOR_PENDING
            width = 4 if i == 0 else 2
            self._draw_shape(shape, color, width=width)
            self._draw_label(shape, str(i + 1), color)

    def _draw_shape(self, shape: DiskDetection, color: str, width: int, dash: tuple | None = None) -> None:
        rx = shape.radius * self.scale
        ry = (shape.ry if shape.ry is not None else shape.radius) * self.scale
        cx = shape.cx * self.scale + self.ox
        cy = shape.cy * self.scale + self.oy
        self.canvas.create_oval(cx - rx, cy - ry, cx + rx, cy + ry, outline=color, width=width, dash=dash)

    def _draw_label(self, shape: DiskDetection, text: str, color: str) -> None:
        rx = shape.radius * self.scale
        ry = (shape.ry if shape.ry is not None else shape.radius) * self.scale
        cx = shape.cx * self.scale + self.ox
        cy = shape.cy * self.scale + self.oy
        self.canvas.create_text(
            cx - rx, cy - ry - 8, text=text, fill=color, font=("", 11, "bold"), anchor="sw"
        )

    # ------------------------------------------------------------- painel --

    def _set_buttons_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled and not self._training else "disabled"
        self.valid_btn.config(state=state)
        self.reject_btn.config(state=state)

    def _update_labels(self) -> None:
        done = self.state.done_count(self.samples)
        total = len(self.samples)
        self.progress["value"] = done
        self.progress_label.config(
            text=f"Série {self.state.round} · Processadas: {done}/{total} "
            f"({100 * done // total if total else 0}%)"
        )
        self.stats_label.config(
            text=f"Recompensas: {self.state.rewards} · Punições: {self.state.punishments} · "
            f"Score: {self.state.rewards - self.state.punishments}"
        )
        self.params_label.config(text=self._params_text())

    def _params_text(self) -> str:
        params = effective_params(self.config, self.state.deltas)
        base = self.config.stabilizer
        parts = []
        for key in TRAINABLE_PARAMS:
            value = params[key]
            delta = value - float(getattr(base, key))
            delta_text = f"{int(delta):+d}" if isinstance(value, int) else f"{delta:+.2f}"
            value_text = f"{value}" if isinstance(value, int) else f"{value:.2f}"
            parts.append(f"{key}={value_text} ({delta_text})")
        return "Parâmetros treinados: " + "  ".join(parts)

    def _report(self, text: str) -> None:
        self.report.config(state=tk.NORMAL)
        self.report.delete("1.0", tk.END)
        self.report.insert("1.0", text)
        self.report.config(state=tk.DISABLED)

    # ------------------------------------------------------------ ciclo --

    def run(self) -> None:
        self.root.after(50, self._poll_queue)
        self.root.mainloop()


class Tooltip:
    """Balão de ajuda simples: Toplevel junto ao cursor, some no Leave.

    Ligado a um widget (tipicamente um rótulo "ⓘ"): ao entrar com o rato
    abre uma janelinha sem decoração perto do cursor com o texto; ao sair
    fecha-a. Nunca rouba o foco.
    """

    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self._top: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, event) -> None:
        if self._top is not None:
            return
        self._top = tk.Toplevel(self.widget)
        self._top.wm_overrideredirect(True)
        self._top.wm_geometry(f"+{event.x_root + 14}+{event.y_root + 14}")
        tk.Label(
            self._top,
            text=self.text,
            justify=tk.LEFT,
            bg="#2b2b2e",
            fg="#eee",
            padx=10,
            pady=6,
            wraplength=340,
        ).pack()
        self._top.lift()

    def _hide(self, _event=None) -> None:
        if self._top is not None:
            self._top.destroy()
            self._top = None


# Texto de ajuda (ⓘ) de cada parâmetro treinável no relatório final.
_PARAM_TOOLTIPS = {
    "param2": "Limiar de acumulador do Hough: alto filtra ruído, baixo apanha bordos de contraste fraco.",
    "param1": "Sensibilidade do gradiente (Canny) antes do Hough: sobe para "
    "ignorar texturas, desce para apanhar limbo suave.",
    "dp": "Inverso da resolução do acumulador: alto acelera (menos precisão), "
    "baixo afia o centro do círculo.",
    "gaussian_kernel_size": "Tamanho do kernel do desfoque gaussiano aplicado "
    "antes do Hough (ímpar forçado).",
    "gaussian_sigma": "Sigma do desfoque: sobe com o ruído da imagem.",
}


class AutoTrainWindow:
    """Janela de treino automático: séries completas sem intervenção humana.

    Inicia `N` séries seguidas; cada uma usa o JSON (deltas) da anterior e a
    margem de erro encolhe com o número de validações. O trabalho pesado
    corre numa thread de fundo e a UI é atualizada por fila (Tk é
    single-threaded).
    """

    def __init__(self, app: ValidatorTkApp):
        if tk is None:  # pragma: no cover
            raise RuntimeError(f"tkinter indisponível: {_TK_IMPORT_ERROR}")
        self.app = app
        self.top = tk.Toplevel(app.root)
        self.top.title("AstroFrame — Treino automático")
        self.top.geometry("760x700")
        self._stop = False
        self._running = False
        self._queue: queue.Queue[tuple] = queue.Queue()
        self._preview_enabled = True
        self._last_preview = 0.0
        self._preview_photo: ImageTk.PhotoImage | None = None
        self._build_ui()
        self.top.after(50, self._poll)
        self.top.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        panel = ttk.Frame(self.top, padding=10)
        panel.pack(fill=tk.BOTH, expand=True)

        row = 0
        ttk.Label(panel, text="Treino automático", font=("", 12, "bold")).grid(row=row, column=0, sticky="w")
        row += 1
        ttk.Label(
            panel,
            text=(
                "O sistema deteta, compara com calibration.json e valida/rejeita sozinho, "
                "com margem de erro cada vez menor. Cada série usa o JSON da anterior."
            ),
            foreground="#666",
            wraplength=700,
        ).grid(row=row, column=0, sticky="w", pady=(0, 8))
        row += 1

        controls = ttk.Frame(panel)
        controls.grid(row=row, column=0, sticky="w")
        ttk.Label(controls, text="Nº de séries:").pack(side=tk.LEFT)
        self.series_var = tk.IntVar(value=3)
        ttk.Spinbox(controls, from_=1, to=20, textvariable=self.series_var, width=5).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        self.start_btn = tk.Button(
            controls, text="Iniciar", command=self.start, bg="#1e7a3a", fg="white", font=("", 10, "bold")
        )
        self.start_btn.pack(side=tk.LEFT, padx=(12, 0))
        self.stop_btn = tk.Button(
            controls, text="Parar", command=self.stop, state=tk.DISABLED, bg="#9c2b2b", fg="white"
        )
        self.stop_btn.pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(controls, text="Fechar", command=self._on_close).pack(side=tk.RIGHT)
        row += 1

        preview_row = ttk.Frame(panel)
        preview_row.grid(row=row, column=0, sticky="w", pady=(4, 0))
        self.preview_var = tk.BooleanVar(value=True)
        self.preview_var.trace_add("write", self._on_preview_toggle)
        ttk.Checkbutton(
            preview_row, text="Pré-visualizar deteções em tempo real", variable=self.preview_var
        ).pack(side=tk.LEFT)
        ttk.Label(preview_row, text="ⓘ", foreground="#0a7", cursor="hand2").pack(side=tk.LEFT, padx=(4, 0))
        Tooltip(
            preview_row.winfo_children()[-1],
            "Mostra cada imagem a ser analisada com os círculos detetados "
            "(no máximo 1 imagem a cada 0,15 s). Desliga para acelerar o treino.",
        )
        row += 1

        iou_row = ttk.Frame(panel)
        iou_row.grid(row=row, column=0, sticky="ew", pady=(6, 0))
        self.iou_var = tk.DoubleVar(value=self.app.auto_iou_min)
        iou_label = ttk.Label(iou_row, text="IoU mínimo com o guia (≥ 0.90):")
        iou_label.pack(side=tk.LEFT)
        Tooltip(
            iou_label,
            "Limiar mínimo de correspondência (IoU) entre a deteção e o guia "
            "manual: quanto maior, mais difícil é o treino (rejeita mais).",
        )
        self.iou_scale = ttk.Scale(
            iou_row,
            from_=AUTO_IOU_MIN,
            to=AUTO_IOU_MAX,
            variable=self.iou_var,
            command=self._on_iou_slider,
        )
        self.iou_scale.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(6, 0))
        self.iou_label = ttk.Label(iou_row, text=f"{self.iou_var.get():.2f}", width=5)
        self.iou_label.pack(side=tk.LEFT, padx=(4, 0))
        row += 1
        ttk.Label(
            panel,
            text="Quanto maior o IoU mínimo, maior a dificuldade do treino.",
            foreground="#666",
        ).grid(row=row, column=0, sticky="w", pady=(0, 4))
        row += 1

        cnn_row = ttk.Frame(panel)
        cnn_row.grid(row=row, column=0, sticky="w", pady=(2, 0))
        self.cnn_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            cnn_row,
            text="Treinar CNN de deteção entre séries (filtra falsos positivos)",
            variable=self.cnn_var,
        ).pack(side=tk.LEFT)
        cnn_hint = ttk.Label(cnn_row, text="ⓘ", foreground="#0a7", cursor="hand2")
        cnn_hint.pack(side=tk.LEFT, padx=(4, 0))
        Tooltip(
            cnn_hint,
            "Recolhe patches das séries (positivos = guia, negativos = falsos positivos "
            "+ recortes aleatórios), re-treina o classificador entre séries e usa-o no "
            "julgamento seguinte. O melhor resultado de cada treino é comparado com o "
            "campeão registado: só é promovido se for estritamente melhor.",
        )
        row += 1

        self.series_label = ttk.Label(panel, text="", font=("", 10, "bold"))
        self.series_label.grid(row=row, column=0, sticky="w", pady=(8, 2))
        row += 1
        self.progress = ttk.Progressbar(panel, maximum=100)
        self.progress.grid(row=row, column=0, sticky="ew")
        row += 1
        self.status = tk.StringVar(value="Pronto — carrega em 'Iniciar'.")
        ttk.Label(panel, textvariable=self.status, wraplength=700, foreground="#0a7").grid(
            row=row, column=0, sticky="w", pady=(2, 0)
        )
        row += 1

        preview_frame = ttk.LabelFrame(panel, text="Pré-visualização")
        preview_frame.grid(row=row, column=0, sticky="ew", pady=(6, 0))
        self.preview = tk.Canvas(preview_frame, bg="#1c1c1e", height=210, highlightthickness=0)
        self.preview.pack(fill=tk.BOTH, expand=True)
        self.preview.bind("<Configure>", lambda _e: self._redraw_preview())
        row += 1

        self.report = tk.Text(panel, height=10, state=tk.DISABLED)
        self.report.grid(row=row, column=0, sticky="ew", pady=(6, 0))
        row += 1

    def _on_iou_slider(self, _value=None) -> None:
        self.iou_label.config(text=f"{self.iou_var.get():.2f}")
        self.app.auto_iou_min = float(self.iou_var.get())

    def _on_preview_toggle(self, *_args) -> None:
        self._preview_enabled = bool(self.preview_var.get())
        if not self._preview_enabled:
            self.preview.delete("all")
            self._preview_photo = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop = False
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        series_total = int(self.series_var.get())
        iou_threshold = float(self.iou_var.get())
        cnn_enabled = bool(self.cnn_var.get())
        self._report(f"Iniciando treino automático ({series_total} série(s))…\n")
        self.app.set_training_active(True)
        state = self.app.state
        samples = self.app.samples
        store = self.app.store
        config = self.app.config
        messages = self._queue
        positives: list[np.ndarray] = []
        negatives: list[np.ndarray] = []
        champion: dict = {"path": None}

        def work() -> None:
            try:
                for current in range(1, series_total + 1):
                    if self._stop:
                        messages.put(("stopped", current, None))
                        return
                    state.begin_round("auto")
                    trainer = AutoTrainer(
                        samples,
                        store,
                        config,
                        state,
                        iou_threshold=iou_threshold,
                        collect_patches=cnn_enabled,
                        cnn_model_path=champion["path"] if cnn_enabled else None,
                    )
                    report = trainer.run_series(
                        progress=self._progress_msg,
                        should_stop=lambda: self._stop,
                        on_detect=self._preview_msg,
                    )
                    state.end_round(metrics_from_report(report.report))
                    messages.put(("series_done", current, report))
                    if cnn_enabled:
                        positives.extend(trainer.positives)
                        negatives.extend(trainer.negatives)
                        score = report.report.score if report.report is not None else None
                        result = train_classifier_round(
                            positives,
                            negatives,
                            state,
                            current,
                            score,
                            epochs=60,
                            champion_path=champion["path"],
                        )
                        if result is not None and not result["skipped"]:
                            if result["champion_path"] is not None:
                                champion["path"] = result["champion_path"]
                            messages.put(("cnn_done", current, result))
                        else:
                            messages.put(("cnn_done", current, {"skipped": True}))
                    if self._stop:
                        messages.put(("stopped", current, None))
                        return
                messages.put(("all_done", None, None))
            except Exception as exc:  # pragma: no cover
                messages.put(("error", None, exc))

        threading.Thread(target=work, daemon=True).start()

    def stop(self) -> None:
        self._stop = True
        self.stop_btn.config(state=tk.DISABLED)
        self.status.set("A parar… (termina a amostra atual)")

    def _progress_msg(self, done: int, total: int, label: str, threshold: float) -> None:
        self._queue.put(("progress", done, total, label, threshold))

    def _preview_msg(self, frame, detected, label, threshold) -> None:
        """`on_detect` do treino (corre na thread de fundo): envia o frame para
        a UI com throttle — nunca mais de 1 a cada `PREVIEW_THROTTLE_S`."""
        if not self._preview_enabled:
            return
        now = time.monotonic()
        if now - self._last_preview < PREVIEW_THROTTLE_S:
            return
        self._last_preview = now
        self._queue.put(("preview", frame, list(detected), label))

    def _poll(self) -> None:
        try:
            while True:
                message = self._queue.get_nowait()
                kind = message[0]
                if kind == "progress":
                    _kind, done, total, label, threshold = message
                    self.progress["value"] = 100 * done // total if total else 0
                    self.status.set(f"[{done}/{total}] {label} — IoU mínimo exigido {threshold:.2f}")
                elif kind == "preview":
                    _kind, frame, detected, label = message
                    self._draw_preview(frame, detected, label)
                elif kind == "series_done":
                    _kind, current, report = message
                    self.series_label.config(
                        text=f"Série {current} concluída: {report.samples_done}/"
                        f"{report.samples_total} amostras · +{report.rewards} ✓ / "
                        f"{report.punishments} ✗ · IoU mín. final {report.threshold_end:.2f}"
                    )
                    self._append_report(
                        f"Série {current}: {report.samples_done}/{report.samples_total}"
                        f" amostras · +{report.rewards} ✓ / {report.punishments} ✗"
                    )
                    if report.cnn_positives or report.cnn_negatives:
                        self._append_report(
                            f"  patches CNN recolhidos: +{report.cnn_positives} ✓ / {report.cnn_negatives} ✗"
                        )
                    for error in report.errors:
                        self._append_report(f"  ! {error}")
                elif kind == "cnn_done":
                    _kind, current, result = message
                    if result.get("skipped"):
                        self._append_report(f"Série {current}: sem patches suficientes para treinar a CNN")
                    else:
                        self._append_report(
                            f"Série {current}: CNN re-treinada — precisão "
                            f"{100 * result['accuracy']:.1f}% "
                            f"· {'PROMOVIDA (novo campeão)' if result['promoted'] else 'mantém o campeão'}"
                        )
                elif kind == "stopped":
                    self._finish("Treino interrompido (séries parciais concluídas).")
                    return
                elif kind == "all_done":
                    self._show_final()
                    return
                elif kind == "error":
                    _kind, _unused, exc = message
                    self._finish(f"Erro no treino automático: {exc}")
                    return
        except queue.Empty:
            pass
        self.top.after(50, self._poll)

    def _show_final(self) -> None:
        state = self.app.state
        self.progress["value"] = 100
        self.status.set("Treino automático concluído.")
        result = build_global_report(self.app.samples, self.app.store, state)
        export_path = export_trained(
            state, self.app.config, self.app._export_path(), result[0] if result else None
        )
        lines = [f"Treino automático concluído — {state.round} série(s) no total."]
        history = rounds_text(state)
        lines.extend(history)
        lines.append("")
        if result is not None:
            _report, report_lines = result
            lines.extend(report_lines)
        lines.append("")
        if state.cnn_series:
            lines.append(
                f"CNN de deteção: {state.cnn_positives} positivos / {state.cnn_negatives} "
                f"negativos recolhidos · {len(state.cnn_series)} treino(s) entre séries:"
            )
            for record in state.cnn_series:
                promoted = "PROMOVIDA" if record["promoted"] else "mantém campeão"
                lines.append(
                    f"  série {record['round']}: precisão {100 * record['accuracy']:.1f}% · {promoted}"
                )
        lines.append("")
        lines.append(f"Config treinada exportada: {export_path}")
        self._report("\n".join(lines))
        FinalReportWindow(self.app, result[0] if result else None, self.app.state, lines)
        print("=" * 72)
        print("AstroFrame — treino automático concluído")
        print("\n".join(lines))
        print("=" * 72)
        self._finish("Treino automático concluído.")

    def _finish(self, status_text: str) -> None:
        self._running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status.set(status_text)
        self.app.training_finished()

    def _append_report(self, text: str) -> None:
        self.report.config(state=tk.NORMAL)
        self.report.insert(tk.END, text + "\n")
        self.report.see(tk.END)
        self.report.config(state=tk.DISABLED)

    def _report(self, text: str) -> None:
        self.report.config(state=tk.NORMAL)
        self.report.delete("1.0", tk.END)
        self.report.insert("1.0", text)
        self.report.config(state=tk.DISABLED)

    # ----------------------------------------------------- pré-visualização --

    def _draw_preview(self, frame, detected, label) -> None:
        if not self._preview_enabled:
            return
        self._preview_frame = frame
        self._preview_detected = list(detected)
        self._preview_label = label
        self._redraw_preview()

    def _redraw_preview(self) -> None:
        frame = getattr(self, "_preview_frame", None)
        if frame is None or not self._preview_enabled:
            return
        h, w = frame.shape[:2]
        cw, ch = self.preview.winfo_width(), self.preview.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 700, 210
        scale = min(cw / w, ch / h)
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._preview_photo = ImageTk.PhotoImage(
            Image.fromarray(rgb).resize(size, Image.LANCZOS), master=self.preview
        )
        self.preview.delete("all")
        ox = (cw - size[0]) / 2.0
        oy = (ch - size[1]) / 2.0
        self.preview.create_image(ox, oy, anchor=tk.NW, image=self._preview_photo)
        for shape in self._preview_detected:
            x1, y1 = ox + (shape.cx - shape.radius) * scale, oy + (shape.cy - shape.radius) * scale
            x2, y2 = ox + (shape.cx + shape.radius) * scale, oy + (shape.cy + shape.radius) * scale
            self.preview.create_oval(x1, y1, x2, y2, outline=COLOR_ACCEPTED, width=2)
        self.preview.create_text(
            8, 8, anchor="nw", text=self._preview_label, fill="#ccc", font=("", 9, "bold")
        )

    def _on_close(self) -> None:
        self._stop = True
        self.top.destroy()


class FinalReportWindow:
    """Pop-up do relatório final: discrepâncias vs guia + pesos editáveis.

    Mostra, numa janela separada:

    - o resumo global (score, recall, precisão, IoU médio vs calibração
      manual) e as **discrepâncias por amostra** (formas manuais vs aceites,
      emparelhamentos, IoU, falsos negativos/positivos, erro do centro e do
      raio);
    - as **recompensas e punições** da série (contadores) com os **pesos**
      (deltas) que o treino aplica a cada parâmetro treinável (nunca a
      tamanhos nem distâncias) — editáveis em tempo real, com ajuda ⓘ;
    - os deltas aprendidos acumulados e o **IoU mínimo** do treino automático
      (≥ 0.90), que também se altera aqui.

    "Aplicar" usa os pesos e o IoU mínimo já nas próximas séries; "Salvar"
    grava-os no `validator_state.json` para **sobreviverem a `--reset-state`**
    e serem reaplicados no próximo arranque. "Fechar" apenas fecha a janela.
    """

    WEIGHT_KEYS = TRAINABLE_PARAMS

    def __init__(
        self,
        app: ValidatorTkApp,
        report: CalibrationReport | None,
        state: ValidatorState,
        lines: list[str],
    ):
        if tk is None:  # pragma: no cover
            raise RuntimeError(f"tkinter indisponível: {_TK_IMPORT_ERROR}")
        self.app = app
        self.report = report
        self.state = state
        self._vars: dict[tuple[str, str], tk.DoubleVar] = {}
        self.top = tk.Toplevel(app.root)
        self.top.title("AstroFrame — Relatório final do treino")
        self.top.geometry("920x720")
        self.top.transient(app.root)
        self._build_ui(report, lines)
        self._load_weights()

    # ---------------------------------------------------------------- UI --

    def _build_ui(self, report: CalibrationReport | None, lines: list[str]) -> None:
        panel = ttk.Frame(self.top, padding=10)
        panel.pack(fill=tk.BOTH, expand=True)

        if report is not None and report.score is not None:
            status = (
                f"Score global vs guia manual: {report.score:.1f}/100 · "
                f"Recall {report.recall * 100:.0f}% · Precisão {report.precision * 100:.0f}%"
                f" · IoU médio {report.mean_iou:.2f}"
            )
        else:
            status = "Sem guia (calibration.json) para pontuar."
        ttk.Label(panel, text=status, font=("", 11, "bold"), foreground="#0a7").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 6)
        )

        # -- discrepâncias por amostra --------------------------------------
        table_frame = ttk.LabelFrame(panel, text="Discrepâncias entre o sistema e a calibração manual")
        table_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(0, 8))
        columns = (
            "amostra",
            "manual",
            "sistema",
            "emparelhados",
            "iou",
            "fn",
            "fp",
            "centro",
            "raio",
        )
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)
        headers = {
            "amostra": ("Amostra", 170),
            "manual": ("Manual", 55),
            "sistema": ("Sistema", 55),
            "emparelhados": ("Emparelh.", 70),
            "iou": ("IoU", 50),
            "fn": ("FN", 40),
            "fp": ("FP", 40),
            "centro": ("Δ centro px", 75),
            "raio": ("Δ raio %", 70),
        }
        for key, (text, width) in headers.items():
            self.tree.heading(key, text=text)
            self.tree.column(key, width=width, anchor="center" if key != "amostra" else "w")
        scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        for item in report.items if report is not None else []:
            iou = f"{item.mean_iou:.2f}" if item.mean_iou is not None else "—"
            center = f"{item.mean_center_error:.1f}" if item.mean_center_error is not None else "—"
            radius = f"{item.mean_radius_error_pct:+.0f}" if item.mean_radius_error_pct is not None else "—"
            self.tree.insert(
                "",
                tk.END,
                values=(
                    item.label,
                    item.n_manual,
                    item.n_detected,
                    item.n_matched,
                    iou,
                    item.n_false_negatives,
                    item.n_false_positives,
                    center,
                    radius,
                ),
            )

        # -- recompensas/punições com pesos ----------------------------------
        weights = ttk.Frame(panel)
        weights.grid(row=2, column=0, columnspan=2, sticky="nsew")
        weights.columnconfigure(1, weight=1)
        ttk.Label(
            weights,
            text=f"Recompensas: {self.state.rewards} ✓ · Punições: {self.state.punishments} ✗",
            font=("", 10, "bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))

        ttk.Label(weights, text="Pesos do treino (alterar e carregar em 'Aplicar'):", foreground="#555").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(0, 2)
        )
        ttk.Label(weights, text="Parâmetro").grid(row=2, column=0, sticky="w", padx=(12, 6))
        ttk.Label(weights, text="Recompensa (válido)").grid(row=2, column=1, sticky="w")
        ttk.Label(weights, text="Punição (rejeitado)").grid(row=2, column=2, sticky="w", padx=(12, 0))
        for r, key in enumerate(self.WEIGHT_KEYS, start=3):
            name = ttk.Frame(weights)
            name.grid(row=r, column=0, sticky="w", padx=(12, 6))
            ttk.Label(name, text=key).pack(side=tk.LEFT)
            help_label = ttk.Label(name, text="ⓘ", foreground="#0a7", cursor="hand2")
            help_label.pack(side=tk.LEFT, padx=(4, 0))
            Tooltip(help_label, _PARAM_TOOLTIPS.get(key, ""))
            self._vars[("reward", key)] = tk.DoubleVar(value=REWARD_DELTAS.get(key, 0.0))
            ttk.Spinbox(
                weights,
                from_=-100.0,
                to=100.0,
                increment=0.25,
                width=8,
                textvariable=self._vars[("reward", key)],
            ).grid(row=r, column=1, sticky="w")
            self._vars[("punish", key)] = tk.DoubleVar(value=PUNISH_DELTAS.get(key, 0.0))
            ttk.Spinbox(
                weights,
                from_=-100.0,
                to=100.0,
                increment=0.25,
                width=8,
                textvariable=self._vars[("punish", key)],
            ).grid(row=r, column=2, sticky="w", padx=(12, 0))

        deltas = (
            ", ".join(f"{key} {value:+.1f}" for key, value in sorted(self.state.deltas.items())) or "nenhum"
        )
        ttk.Label(
            weights,
            text=f"Deltas aprendidos até agora: {deltas}",
            foreground="#666",
        ).grid(row=3 + len(self.WEIGHT_KEYS), column=0, columnspan=3, sticky="w", pady=(6, 0))

        # -- IoU mínimo + ações ---------------------------------------------
        actions = ttk.Frame(panel)
        actions.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(actions, text="IoU mínimo do treino automático (≥ 0.90):").pack(side=tk.LEFT)
        self.iou_var = tk.DoubleVar(value=self.app.auto_iou_min)
        ttk.Scale(
            actions,
            from_=AUTO_IOU_MIN,
            to=AUTO_IOU_MAX,
            variable=self.iou_var,
            command=lambda _v: self.iou_value_label.config(text=f"{self.iou_var.get():.2f}"),
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(6, 0))
        self.iou_value_label = ttk.Label(actions, text=f"{self.iou_var.get():.2f}", width=5)
        self.iou_value_label.pack(side=tk.LEFT, padx=(4, 0))
        self.apply_btn = tk.Button(
            actions,
            text="Aplicar",
            command=self.apply,
            bg="#1e7a3a",
            fg="white",
            font=("", 10, "bold"),
            padx=12,
        )
        self.apply_btn.pack(side=tk.RIGHT, padx=(12, 0))
        self.save_btn = tk.Button(
            actions,
            text="Salvar",
            command=self.save,
            bg="#2d6da8",
            fg="white",
            font=("", 10, "bold"),
            padx=12,
        )
        self.save_btn.pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(actions, text="Fechar", command=self.top.destroy).pack(side=tk.RIGHT)
        self.status_var = tk.StringVar(value="")
        self.status = ttk.Label(actions, textvariable=self.status_var, foreground="#0a7")
        self.status.pack(side=tk.RIGHT, padx=(8, 0))

        self._full_report = "\n".join(lines)
        ttk.Label(panel, text="Relatório completo:").grid(row=4, column=0, columnspan=2, sticky="w")
        self.text = tk.Text(panel, height=8, state=tk.DISABLED)
        self.text.grid(row=5, column=0, columnspan=2, sticky="nsew")
        self._fill_text()

        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)
        panel.rowconfigure(5, weight=1)

    def _fill_text(self) -> None:
        self.text.config(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", self._full_report)
        self.text.config(state=tk.DISABLED)

    def _load_weights(self) -> None:
        for (kind, key), var in self._vars.items():
            source = REWARD_DELTAS if kind == "reward" else PUNISH_DELTAS
            var.set(source.get(key, 0.0))

    def apply(self) -> None:
        """Usa os pesos editados e o IoU mínimo já nas próximas séries."""
        self._write_weights()
        self.app.auto_iou_min = float(self.iou_var.get())
        self.status_var.set("Pesos e IoU mínimo aplicados ✓ (Salvar para guardar)")

    def save(self) -> None:
        """Grava os pesos e o IoU mínimo no estado — sobrevivem a `--reset-state`
        e são aplicados no arranque seguinte."""
        self._write_weights()
        self.state.weights["iou"] = float(self.iou_var.get())
        self.app.auto_iou_min = float(self.iou_var.get())
        self.state.save()
        self.status_var.set("Pesos e IoU mínimo salvos ✓")

    def _write_weights(self) -> None:
        for (kind, key), var in self._vars.items():
            target = REWARD_DELTAS if kind == "reward" else PUNISH_DELTAS
            value = float(var.get())
            target[key] = value
            self.state.weights[kind][key] = value


def build_app(
    root: tk.Tk,
    samples_dir: str = "samples",
    config_path: str | None = None,
    state_path: str | Path | None = None,
) -> ValidatorTkApp:
    """Constrói a janela (sem `mainloop`), para testes e para `run`."""
    config = AstroFrameConfig.from_yaml(config_path) if config_path else AstroFrameConfig()
    samples = scan_samples(samples_dir)
    store = CalibrationStore(calibration_json(samples_dir))
    state = ValidatorState(state_path or train_dir() / DEFAULT_STATE_NAME)
    return ValidatorTkApp(root, samples, store, config, state, samples_root=samples_dir)


def run(
    samples_dir: str = "samples", config_path: str | None = None, state_path: str | Path | None = None
) -> None:
    """Lança a interface desktop de validação."""
    if tk is None:  # pragma: no cover
        raise SystemExit(
            "tkinter não está disponível neste ambiente.\nNo Debian/Ubuntu: sudo apt install python3-tk"
        )
    root = tk.Tk()
    build_app(root, samples_dir=samples_dir, config_path=config_path, state_path=state_path).run()


# --------------------------------------------------------------------------
# Modo sem interface
# --------------------------------------------------------------------------


def run_check(
    samples_dir: str = "samples", config_path: str | None = None, state_path: str | Path | None = None
) -> int:
    """`--check`: deteta em todas as amostras (com os deltas aprendidos) e
    imprime o relatório contra `calibration.json`, sem janela."""
    config = AstroFrameConfig.from_yaml(config_path) if config_path else AstroFrameConfig()
    state = ValidatorState(state_path or train_dir() / DEFAULT_STATE_NAME)
    apply_state_weights(state)
    apply_effective(config, state.deltas)
    samples = scan_samples(samples_dir)
    store = CalibrationStore(calibration_json(samples_dir))

    print(f"AstroFrame — verificação automática da deteção ({len(samples)} amostras)")
    deltas_text = " · ".join(f"{key} {state.deltas.get(key, 0):+.2f}" for key in TRAINABLE_PARAMS)
    print(f"Deltas aprendidos: {deltas_text}")
    rows: list[tuple[str, list[DiskDetection], list[DiskDetection]]] = []
    errors: list[str] = []
    for sample in samples:
        item = store.get_item(sample.key)
        gt = list(item.circles) if item is not None else []
        try:
            frame = load_frame(sample)
            detected = find_disks_for_calibration(frame, config, expected_n=len(gt))
        except Exception as exc:
            errors.append(f"{sample.label}: erro ({exc})")
            continue
        rows.append((sample.label, gt, detected))

    report = validate_all(rows)
    if report.score is not None:
        print(f"Score vs guia manual: {report.score:.1f}/100")
    print(
        f"Recall {report.recall * 100:.0f}% · Precisão {report.precision * 100:.0f}%"
        f" · IoU médio {report.mean_iou:.2f}"
    )
    print(
        f"Total: {report.total_matched} emparelhado(s), {report.total_false_negatives} "
        f"falso(s) negativo(s), {report.total_false_positives} falso(s) positivo(s)."
    )
    for item in report.items:
        iou = f"{item.mean_iou:.2f}" if item.mean_iou is not None else "—"
        print(f"  {item.label}: {item.n_matched}/{item.n_manual} · IoU {iou}")
    for error in errors:
        print(f"  ! {error}")
    return 0


def run_auto_headless(
    samples_dir: str = "samples",
    config_path: str | None = None,
    state_path: str | Path | None = None,
    series: int = 3,
    export_path: str | Path | None = None,
    iou: float | None = None,
    epochs: int = 60,
    cnn: bool = True,
    cnn_threshold: float = CNN_THRESHOLD_DEFAULT,
) -> int:
    """Treino headless mantido declarado (sem entrada na CLI desde a remoção
    do auto-treino): N séries de treino automático, com exportação final.

    Pode ser chamado programaticamente pelos testes; a interface usa o
    julgamento manual. Entre séries, os patches recolhidos re-treiam a CNN de
    deteção (warm-start do campeão); o candidato é comparado com o melhor
    registado no banco e só é promovido se for estritamente melhor. Sem `iou`
    explícito usa o valor guardado no estado (pesos 'Salvar').
    """
    config = AstroFrameConfig.from_yaml(config_path) if config_path else AstroFrameConfig()
    state = ValidatorState(state_path or train_dir() / DEFAULT_STATE_NAME)
    apply_state_weights(state)
    iou_threshold = iou if iou is not None else state_iou(state)
    samples = scan_samples(samples_dir)
    store = CalibrationStore(calibration_json(samples_dir))
    db = FeedbackDB()

    print(
        f"AstroFrame — treino automático ({len(samples)} amostras, {series} série(s), "
        f"IoU mínimo {iou_threshold:.2f}, CNN {'ligada' if cnn else 'desligada'})"
    )
    positives: list[np.ndarray] = []
    negatives: list[np.ndarray] = []
    champion_path: str | Path | None = None
    for current in range(1, series + 1):
        state.begin_round("auto")
        trainer = AutoTrainer(
            samples,
            store,
            config,
            state,
            iou_threshold=iou_threshold,
            collect_patches=cnn,
            cnn_model_path=champion_path if cnn else None,
            cnn_threshold=cnn_threshold,
        )
        report = trainer.run_series(
            progress=lambda done, total, label, threshold: print(
                f"  [{done}/{total}] {label} — IoU mínimo {threshold:.2f}"
            )
        )
        state.end_round(metrics_from_report(report.report))
        print(
            f"Série {current}: {report.samples_done}/{report.samples_total} amostras · "
            f"+{report.rewards} ✓ / {report.punishments} ✗ · "
            f"IoU mínimo final {report.threshold_end:.2f}"
        )
        for error in report.errors:
            print(f"  ! {error}")
        if report.report is not None and report.report.score is not None:
            print(
                f"  Score vs guia: {report.report.score:.1f}/100 · "
                f"Recall {report.report.recall * 100:.0f}% · "
                f"Precisão {report.report.precision * 100:.0f}%"
            )
        if cnn:
            positives.extend(trainer.positives)
            negatives.extend(trainer.negatives)
            if report.cnn_positives or report.cnn_negatives:
                print(
                    f"  patches CNN recolhidos: +{report.cnn_positives} ✓ / "
                    f"{report.cnn_negatives} ✗ (acumulado: {len(positives)} / {len(negatives)})"
                )
            score = report.report.score if report.report is not None else None
            result = train_classifier_round(
                positives,
                negatives,
                state,
                current,
                score,
                epochs=epochs,
                champion_path=champion_path,
                db=db,
            )
            if result is not None and not result["skipped"]:
                if result["champion_path"] is not None:
                    champion_path = result["champion_path"]
                print(
                    f"  CNN: precisão {100 * result['accuracy']:.1f}% · "
                    f"{'PROMOVIDA (novo campeão)' if result['promoted'] else 'mantém o campeão'}"
                )
            elif result is not None:
                print("  CNN: sem patches suficientes para treinar")

    result = build_global_report(samples, store, state)
    final_report = result[0] if result else None
    export_path = export_trained(
        state, config, export_path or train_dir() / DEFAULT_EXPORT_NAME, final_report
    )
    print(f"Config treinada exportada: {export_path}")
    if result is not None:
        _report, lines = result
        print("\n".join(lines))
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="validator",
        description=(
            "AstroFrame — validação/treino da deteção: recompensa e punição por forma, "
            "com Logs/train/calibration.json como guia, até 100% processado."
        ),
    )
    parser.add_argument(
        "--samples",
        default="samples",
        help="Pasta com as imagens e vídeos de exemplo (varrida recursivamente)",
    )
    parser.add_argument("--config", default=None, help="Caminho para um config.yaml")
    parser.add_argument(
        "--state",
        default=None,
        help=f"Ficheiro de estado do treino (omissão: Logs/train/{DEFAULT_STATE_NAME})",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Apaga o progresso, os deltas e o histórico de séries antes de começar",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Modo sem interface: deteta todas as amostras e imprime o relatório",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    migrate_legacy()
    setup_logging("validator.log")
    args = build_parser().parse_args(argv)
    state_path = Path(args.state) if args.state else train_dir() / DEFAULT_STATE_NAME
    if args.reset_state and state_path.exists():
        ValidatorState(state_path).reset()
        print(f"Estado de validação reposto: {state_path}")
    try:
        if args.check:
            return run_check(args.samples, args.config, state_path)
        run(samples_dir=args.samples, config_path=args.config, state_path=state_path)
    except Exception:
        logging.exception("Falha ao arrancar a validação")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
