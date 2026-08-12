"""Melhoria automática de imagem: CLAHE + denoising + máscara de nitidez."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from astroframe.config import AstroFrameConfig

logger = logging.getLogger(__name__)


def clahe_enhance(image: np.ndarray, config: AstroFrameConfig) -> np.ndarray:
    """Equalização adaptativa (CLAHE) no canal L do LAB, preservando as cores originais."""
    cfg = config.clahe
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    tile = max(1, min(cfg.tile_grid_size, *lightness.shape[:2]))
    clahe = cv2.createCLAHE(clipLimit=cfg.clip_limit, tileGridSize=(tile, tile))
    lightness = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR)


def denoise(image: np.ndarray, config: AstroFrameConfig) -> np.ndarray:
    """Redução de ruído Non-Local Means (especialmente útil em fotos com ISO elevado)."""
    cfg = config.denoise
    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        cfg.h,
        cfg.h,
        cfg.template_window_size,
        cfg.search_window_size,
    )


def unsharp_mask(image: np.ndarray, config: AstroFrameConfig) -> np.ndarray:
    """Máscara de nitidez: destaca as bordas exatas da Lua sobre o disco solar."""
    cfg = config.unsharp
    gaussian = cv2.GaussianBlur(image, (0, 0), cfg.sigma)
    return cv2.addWeighted(image, 1.0 + cfg.amount, gaussian, -cfg.amount, 0)


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return image


def enhance_image(
    image: np.ndarray,
    config: AstroFrameConfig | None = None,
    use_denoise: bool = True,
) -> np.ndarray:
    """Pipeline de melhoria: CLAHE (LAB) -> denoising -> máscara de nitidez.

    `use_denoise=False` omite o denoising (o passo mais lento); útil para
    vídeos grandes via `astroframe video --fast`.
    """
    config = config or AstroFrameConfig()
    image = _as_bgr(image)
    image = clahe_enhance(image, config)
    if use_denoise:
        image = denoise(image, config)
    return unsharp_mask(image, config)
