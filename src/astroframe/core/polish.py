"""Polimento final: fundo preto, contorno perfeitamente redondo e remoção de reflexos.

Depois da melhoria (CLAHE + denoise + nitidez), o polimento "lava" a imagem:

- **fundo preto** — tudo fora do disco + coroa (`corona_scale` × raio) passa a
  preto puro (0);
- **contorno redondo** — a máscara do disco é suavizada com um *feather*
  gaussiano proporcional ao raio, dando um limbo suave e circular (sem dentes
  de ruído);
- **reflexos removidos** — círculos secundários detetados (reflexos internos
  da lente / ghosts) são escurecidos na coroa com o mesmo feather.

Sem deteção válida, a imagem é devolvida inalterada (`polish_image` nunca
falha no fluxo principal).
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from astroframe.config import AstroFrameConfig
from astroframe.core.stabilizer import DiskDetection, find_all_disks

logger = logging.getLogger(__name__)


def _feather_mask(shape: tuple[int, int], cx: int, cy: int, radius: float, feather: float) -> np.ndarray:
    """Máscara circular com borda suavizada (0..1), redonda e sem dentes."""
    height, width = shape
    ys, xs = np.ogrid[:height, :width]
    inside = np.hypot(xs - cx, ys - cy) <= radius
    if feather <= 0.0:
        return inside.astype(np.float32)
    blur = max(3, int(round(feather * radius)) | 1)
    return cv2.GaussianBlur(inside.astype(np.float32), (blur, blur), 0)


def _applies_to(image: np.ndarray, detection: DiskDetection, config: AstroFrameConfig) -> bool:
    """Verifica se há algo para polir (deteção válida e dentro dos limites)."""
    cfg = config.polish
    if not cfg.enabled or detection.radius <= 0:
        return False
    height, width = image.shape[:2]
    return 0 <= detection.cx < width and 0 <= detection.cy < height


def polish_image(
    image: np.ndarray,
    detection: DiskDetection | None,
    config: AstroFrameConfig | None = None,
) -> np.ndarray:
    """Aplica o polimento (fundo preto + contorno suave + reflexos removidos).

    `detection` deve usar as coordenadas da própria `image` (após
    estabilização, o disco está centrado). Sem disco detetado devolve a
    imagem inalterada.
    """
    config = config or AstroFrameConfig()
    cfg = config.polish
    if detection is None or not _applies_to(image, detection, config):
        return image

    height, width = image.shape[:2]
    cx, cy = int(detection.cx), int(detection.cy)
    radius = float(detection.radius)

    if cfg.black_background:
        keep = _feather_mask((height, width), cx, cy, radius * cfg.corona_scale, cfg.feather)
    else:
        keep = None

    if cfg.remove_reflections:
        for disk in find_all_disks(image, config):
            if disk.radius < cfg.reflection_min_radius:
                continue
            if disk.cx == cx and disk.cy == cy:
                continue
            if abs(disk.cx - cx) < 2 and abs(disk.cy - cy) < 2:
                continue
            if math.hypot(disk.cx - cx, disk.cy - cy) < radius:
                continue
            refl = _feather_mask((height, width), disk.cx, disk.cy, disk.radius, cfg.feather)
            keep = 1.0 - refl if keep is None else np.minimum(keep, 1.0 - refl)

    if keep is None:
        return image

    result = np.asarray(image).astype(np.float32)
    result = result * keep[..., None] if result.ndim == 3 else result * keep
    return np.clip(result, 0, 255).astype(np.uint8)
