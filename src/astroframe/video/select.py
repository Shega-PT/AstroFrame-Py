"""Lucky imaging: descarte de frames borrados por motion blur.

A nitidez é medida pela variância do Laplaciano (cv2.Laplacian(...).var()).
Frames capturados durante movimentos rápidos da câmara ficam desfocados e
apresentam valores baixos — são ignorados. O limiar pode ser definido à mão
ou estimado estatisticamente a partir do próprio vídeo (percentil das
nitidezes), eliminando a necessidade de calibrar um valor fixo.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import cv2
import numpy as np

from astroframe.config import AstroFrameConfig

logger = logging.getLogger(__name__)


def sharpness(frame: np.ndarray, config: AstroFrameConfig | None = None) -> float:
    """Variância do Laplaciano: valores mais altos = frames mais nítidos."""
    config = config or AstroFrameConfig()
    cfg = config.lucky
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    gray = cv2.GaussianBlur(gray, (cfg.gaussian_kernel_size, cfg.gaussian_kernel_size), cfg.gaussian_sigma)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_sharpness_threshold(scores: Sequence[float], percentile: float = 25.0) -> float:
    """Limiar de nitidez estimado a partir das estatísticas do próprio vídeo."""
    if not scores:
        return 0.0
    return float(np.percentile(scores, percentile))


def select_sharp_frames(
    frames: Sequence[np.ndarray],
    config: AstroFrameConfig | None = None,
    minimum: float | None = None,
) -> list[tuple[int, np.ndarray, float]]:
    """Devolve (índice, frame, nitidez) dos frames acima do limiar.

    O limiar usado é, por ordem: o argumento `minimum`, o `min_sharpness`
    da configuração, ou o percentil estimado da própria sequência.
    """
    config = config or AstroFrameConfig()
    scored = [(i, frame, sharpness(frame, config)) for i, frame in enumerate(frames)]

    if minimum is not None:
        threshold = minimum
    elif config.lucky.min_sharpness is not None:
        threshold = config.lucky.min_sharpness
    else:
        threshold = estimate_sharpness_threshold(
            [score for _, _, score in scored], config.lucky.sharpness_percentile
        )
    logger.info("Limiar de nitidez: %.2f", threshold)

    return [(i, frame, score) for i, frame, score in scored if score >= threshold]
