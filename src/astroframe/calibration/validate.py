"""Validação da deteção automática contra o ground truth manual.

`validate_all` compara os círculos guardados pelo utilizador (calibração)
com a saída de `find_all_disks` em **todas** as amostras e agrega as métricas
num relatório global com score 0–100 (recall 0.4 · precisão 0.3 · IoU médio 0.3)
e sugestões de parâmetros (`suggest_parameters`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from astroframe.config import AstroFrameConfig
from astroframe.core.stabilizer import DiskDetection

IOU_THRESHOLD = 0.5

_RECALL_WEIGHT = 0.4
_PRECISION_WEIGHT = 0.3
_IOU_WEIGHT = 0.3


def circle_iou(a: DiskDetection, b: DiskDetection) -> float:
    """Intersecção sobre união de dois círculos (0–1)."""
    r1, r2 = float(a.radius), float(b.radius)
    d = math.hypot(a.cx - b.cx, a.cy - b.cy)
    if d >= r1 + r2:
        return 0.0
    if d == 0.0:
        return 1.0 if r1 == r2 else float(min(r1, r2) ** 2 / max(r1, r2) ** 2)
    if d + min(r1, r2) <= max(r1, r2):
        return float(min(r1, r2) ** 2 / max(r1, r2) ** 2)
    part1 = r1 * r1 * math.acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
    part2 = r2 * r2 * math.acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
    part3 = 0.5 * math.sqrt(max(0.0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)))
    inter = part1 + part2 - part3
    union = math.pi * (r1 * r1 + r2 * r2) - inter
    return float(max(0.0, min(1.0, inter / union)))


def match_circles(
    manual: list[DiskDetection],
    detected: list[DiskDetection],
    iou_threshold: float = IOU_THRESHOLD,
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    """Corresponde círculos manuais a detetados (greedy por IoU descrescente).

    Devolve (pares, manuais sem par, detetados sem par).
    """
    scores: list[tuple[float, int, int]] = []
    for i, m in enumerate(manual):
        for j, d in enumerate(detected):
            score = circle_iou(m, d)
            if score >= iou_threshold:
                scores.append((score, i, j))
    used_m: set[int] = set()
    used_d: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, i, j in sorted(scores, key=lambda item: item[0], reverse=True):
        if i in used_m or j in used_d:
            continue
        pairs.append((i, j))
        used_m.add(i)
        used_d.add(j)
    return pairs, set(range(len(manual))) - used_m, set(range(len(detected))) - used_d


@dataclass
class ItemReport:
    """Métricas de um item de calibração."""

    label: str
    n_manual: int
    n_detected: int
    n_matched: int
    n_false_negatives: int
    n_false_positives: int
    mean_iou: float | None = None
    mean_center_error: float | None = None
    mean_radius_error_pct: float | None = None


def validate_item(label: str, manual: list[DiskDetection], detected: list[DiskDetection]) -> ItemReport:
    """Compara a deteção de uma amostra com o ground truth manual."""
    pairs, unmatched_m, unmatched_d = match_circles(manual, detected)
    ious = [circle_iou(manual[i], detected[j]) for i, j in pairs]
    centers = [math.hypot(manual[i].cx - detected[j].cx, manual[i].cy - detected[j].cy) for i, j in pairs]
    radii = [(detected[j].radius - manual[i].radius) / manual[i].radius * 100.0 for i, j in pairs]
    return ItemReport(
        label=label,
        n_manual=len(manual),
        n_detected=len(detected),
        n_matched=len(pairs),
        n_false_negatives=len(unmatched_m),
        n_false_positives=len(unmatched_d),
        mean_iou=float(sum(ious) / len(ious)) if ious else None,
        mean_center_error=float(sum(centers) / len(centers)) if centers else None,
        mean_radius_error_pct=float(sum(radii) / len(radii)) if radii else None,
    )


@dataclass
class CalibrationReport:
    """Relatório global: agregados de todas as amostras + score 0–100."""

    items: list[ItemReport] = field(default_factory=list)
    total_manual: int = 0
    total_detected: int = 0
    total_matched: int = 0
    total_false_negatives: int = 0
    total_false_positives: int = 0
    recall: float = 0.0
    precision: float = 0.0
    mean_iou: float = 0.0
    mean_center_error: float | None = None
    mean_radius_error_pct: float | None = None
    score: float | None = None

    @property
    def has_ground_truth(self) -> bool:
        return self.total_manual > 0


def validate_all(items: list[tuple[str, list[DiskDetection], list[DiskDetection]]]) -> CalibrationReport:
    """Agrega os relatórios de várias amostras (label, manual, detetado)."""
    reports = [validate_item(label, manual, detected) for label, manual, detected in items]
    report = CalibrationReport(items=reports)
    report.total_manual = sum(r.n_manual for r in reports)
    report.total_detected = sum(r.n_detected for r in reports)
    report.total_matched = sum(r.n_matched for r in reports)
    report.total_false_negatives = sum(r.n_false_negatives for r in reports)
    report.total_false_positives = sum(r.n_false_positives for r in reports)
    report.recall = report.total_matched / report.total_manual if report.total_manual else 0.0
    report.precision = report.total_matched / report.total_detected if report.total_detected else 0.0
    matched_ious = [r.mean_iou for r in reports if r.mean_iou is not None]
    report.mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
    centers = [r.mean_center_error for r in reports if r.mean_center_error is not None]
    report.mean_center_error = sum(centers) / len(centers) if centers else None
    radii = [r.mean_radius_error_pct for r in reports if r.mean_radius_error_pct is not None]
    report.mean_radius_error_pct = sum(radii) / len(radii) if radii else None
    if report.has_ground_truth:
        report.score = 100.0 * (
            _RECALL_WEIGHT * report.recall
            + _PRECISION_WEIGHT * report.precision
            + _IOU_WEIGHT * report.mean_iou
        )
    return report


def suggest_parameters(report: CalibrationReport, config: AstroFrameConfig | None = None) -> list[str]:
    """Sugestões (PT) de ajuste de parâmetros a partir do relatório global."""
    config = config or AstroFrameConfig()
    if not report.has_ground_truth:
        return [
            "Sem círculos manuais guardados — desenha os astros na amostra e "
            "carrega em 'Guardar ajustes' antes de validar."
        ]
    suggestions: list[str] = []
    if report.total_false_negatives:
        cfg = config.stabilizer
        suggestions.append(
            f"Há {report.total_false_negatives} disco(s) manual(is) não detetado(s): confirma se o astro "
            f"entra no intervalo de raios [min_radius={cfg.min_radius}, max_radius={cfg.max_radius}] px "
            f"e considera baixar `min_radius` (ou `param2`, atualmente {cfg.param2}, para detetar "
            "bordos de contraste mais fraco)."
        )
    if report.total_false_positives:
        cfg = config.stabilizer
        suggestions.append(
            f"Há {report.total_false_positives} deteção(ões) sem correspondência manual: considera subir "
            f"`min_radius` ou `param2` (atualmente {cfg.param2}) para filtrar círculos falsos."
        )
    if report.mean_radius_error_pct is not None and report.mean_radius_error_pct <= -10.0:
        suggestions.append(
            f"O raio detetado é sistematicamente menor que o manual ({report.mean_radius_error_pct:.0f}% "
            "em média): o contorno detetado está dentro do bordo real — pondera usar `contour_fallback` "
            "ou afinar o refinamento do raio."
        )
    if not suggestions:
        suggestions.append(
            "Sem ajustes sugeridos — a deteção automática alinha-se bem com o ground truth manual."
        )
    return suggestions
