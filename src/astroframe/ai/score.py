"""Avaliação automática da qualidade (0–5 estrelas) com métricas explicáveis.

A pontuação mede a imagem **antes** do polimento (sobre o resultado
estabilizado + CLAHE + denoise + nitidez), senão um fundo 100% preto daria
sempre 5 estrelas sem mérito:

- **background** — fração do fundo (fora do disco + coroa) suficientemente
  escura; penaliza reflexos vagos, brilho ambiente e vinheta clara;
- **limb** — redondeza do limbo: sobreposição (IoU) entre o disco de brilho
  real e um círculo perfeito do raio detetado;
- **noise** — variância do Laplaciano na coroa (ruído baixo → nota alta);
- **contrast** — gama dinâmica (p99/p50) dentro da coroa;
- **reflections** — penalização por reflexos da lente (ghosts) encontrados.

Cada métrica vale 0..1; a média ponderada (pesos em `score:` do YAML) é
multiplicada por 5 e arredondada a 1 casa.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from astroframe.config import AstroFrameConfig
from astroframe.core.stabilizer import GHOST_RADIUS_RATIO, DiskDetection, find_all_disks

_STAR_TEXT = {
    5: "Excelente",
    4: "Muito bom",
    3: "Bom",
    2: "Fraco",
    1: "Mau",
    0: "Inaceitável",
}


@dataclass(frozen=True)
class StarRating:
    """Avaliação 0–5 estrelas com as métricas que a justificam."""

    stars: float
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return stars_text(self.stars)


def stars_text(stars: float) -> str:
    """Representação textual estável (★ cheias + ½ + ☆ vazias + etiqueta)."""
    full = int(np.clip(round(stars), 0, 5))
    half = 1 if 0.25 <= stars - int(stars) < 0.75 else 0
    empty = 5 - full - half
    stars_str = "★" * full + ("½" if half else "") + "☆" * empty
    return f"{stars_str} {stars:.1f} — {_STAR_TEXT[full]}"


def _background_purity(image: np.ndarray, cx: int, cy: int, radius: float, config: AstroFrameConfig) -> float:
    """Fração dos pixels fora do disco+coroa com brilho 'preto suficiente'."""
    height, width = image.shape[:2]
    ys, xs = np.ogrid[:height, :width]
    keep = np.hypot(xs - cx, ys - cy) <= radius * config.polish.corona_scale
    sample = image[~keep]
    if sample.size == 0:
        return 1.0
    mean = float(np.mean(sample))
    return float(np.clip(1.0 - mean / 64.0, 0.0, 1.0))


def _limb_roundness(image: np.ndarray, cx: int, cy: int, radius: float, cfg) -> float:
    """IoU entre o disco de brilho (limiarizado) e o círculo perfeito detetado."""
    height, width = image.shape[:2]
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lo = max(16.0, float(np.percentile(gray, 90)) * cfg.limb_min_dark)
    bright = (gray > lo).astype(np.uint8)
    ys, xs = np.ogrid[:height, :width]
    perfect = (np.hypot(xs - cx, ys - cy) <= radius * cfg.edge_radius).astype(np.uint8)
    union = float(np.logical_or(bright, perfect).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(bright, perfect).sum()) / union


def _noise_level(image: np.ndarray, cx: int, cy: int, radius: float, config: AstroFrameConfig) -> float:
    """Variância do Laplaciano na coroa (anel logo fora do limbo), normalizada.

    0 = limpo, 1 = ruidoso. O anel começa a 1.05×raio para não contar o
    gradiente do próprio limbo (que existe mesmo numa imagem sem ruído).
    """
    height, width = image.shape[:2]
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    ys, xs = np.ogrid[:height, :width]
    distance = np.hypot(xs - cx, ys - cy)
    annulus = (distance >= radius * 1.05) & (distance <= radius * config.polish.corona_scale)
    values = laplacian[annulus]
    if values.size == 0:
        return 1.0
    var = float(np.var(values))
    return float(np.clip(var / 900.0, 0.0, 1.0))


def _contrast_range(image: np.ndarray, cx: int, cy: int, radius: float, config: AstroFrameConfig) -> float:
    """Razão p99/p50 da intensidade na coroa (contraste útil do limbo do astro)."""
    height, width = image.shape[:2]
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ys, xs = np.ogrid[:height, :width]
    annulus = np.hypot(xs - cx, ys - cy) <= radius * config.polish.corona_scale
    values = gray[annulus]
    if values.size == 0:
        return 0.0
    lo = max(1.0, float(np.percentile(values, 50)))
    ratio = float(np.percentile(values, 99)) / lo
    return float(np.clip((ratio - 1.0) / 4.0, 0.0, 1.0))


def _reflection_penalty(
    image: np.ndarray,
    cx: int,
    cy: int,
    radius: float,
    cfg,
    disks: list[DiskDetection] | None = None,
) -> float:
    """0..1: 1.0 = sem reflexos; cada círculo-ghost relevante reduz a nota.

    Um disco é um reflexo da lente (ghost) se for **pequeno face ao astro
    maior** (raio < `GHOST_RADIUS_RATIO` × o do primário) e afastado dele;
    discos grandes são outros corpos celestes reais e não penalizam.

    `disks` permite passar as deteções já calculadas pelo chamador (evita
    voltar a correr o Hough dentro da avaliação — caro e recursivo);
    sem ele, a deteção é feita aqui com a configuração dada.
    """
    if disks is None:
        disks = find_all_disks(image, cfg)
    penalty = 0.0
    for disk in disks:
        if disk.radius < cfg.polish.reflection_min_radius:
            continue
        if disk.cx == cx and disk.cy == cy:
            continue
        if abs(disk.cx - cx) < 4 and abs(disk.cy - cy) < 4:
            continue
        if disk.radius >= GHOST_RADIUS_RATIO * radius:
            continue
        penalty += 0.5
    return float(np.clip(1.0 - penalty, 0.0, 1.0))


def score_image(
    image: np.ndarray,
    detection: DiskDetection | None = None,
    config: AstroFrameConfig | None = None,
    disks: list[DiskDetection] | None = None,
) -> StarRating:
    """Avalia uma imagem processada (0–5 estrelas) com métricas explicáveis.

    Sem `detection`, a avaliação usa apenas ruído/contraste globais e vale
    no máximo `score.background_weight + score.noise_weight + score.contrast_weight`
    das estrelas (as restantes pesam 0) — nunca falha.

    `disks` são as deteções já calculadas pelo chamador (reutilizadas na
    penalização por reflexos sem voltar a correr o Hough).
    """
    config = config or AstroFrameConfig()
    cfg = config.score
    metrics: dict[str, float] = {}

    if detection is not None and detection.radius > 0:
        cx, cy, radius = detection.cx, detection.cy, detection.radius
        metrics["background"] = _background_purity(image, cx, cy, radius, config)
        metrics["limb"] = _limb_roundness(image, cx, cy, radius, cfg)
        metrics["noise"] = 1.0 - _noise_level(image, cx, cy, radius, config)
        metrics["contrast"] = _contrast_range(image, cx, cy, radius, config)
        metrics["reflections"] = _reflection_penalty(image, cx, cy, radius, config, disks)
        weights = {
            "background": cfg.background_weight,
            "limb": cfg.limb_weight,
            "noise": cfg.noise_weight,
            "contrast": cfg.contrast_weight,
            "reflections": cfg.reflection_weight,
        }
    else:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        var = float(np.var(cv2.Laplacian(gray, cv2.CV_32F)))
        metrics["noise"] = float(np.clip(1.0 - var / 900.0, 0.0, 1.0))
        lo = max(1.0, float(np.percentile(gray, 50)))
        ratio = float(np.percentile(gray, 99)) / lo
        metrics["contrast"] = float(np.clip((ratio - 1.0) / 4.0, 0.0, 1.0))
        metrics["background"] = 0.0
        metrics["limb"] = 0.0
        metrics["reflections"] = 0.0
        weights = {
            "background": 0.0,
            "limb": 0.0,
            "noise": cfg.noise_weight,
            "contrast": cfg.contrast_weight,
            "reflections": 0.0,
        }

    total = sum(weights.values())
    stars = round(5.0 * sum(metrics[k] * weights[k] for k in metrics) / total, 1) if total > 0 else 0.0
    return StarRating(stars=float(np.clip(stars, 0.0, 5.0)), metrics=metrics)


def score_from_stars(stars: float) -> StarRating:
    """Cria um StarRating apenas com o valor (usado para feedback manual)."""
    return StarRating(stars=float(np.clip(stars, 0.0, 5.0)), metrics={})
