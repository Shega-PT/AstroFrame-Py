"""Orquestra o pipeline completo: estabilizar -> melhorar."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from astroframe.config import AstroFrameConfig
from astroframe.core.enhancer import enhance_image
from astroframe.core.stabilizer import DiskDetection, center_and_stabilize

logger = logging.getLogger(__name__)


@dataclass
class ProcessResult:
    original: np.ndarray
    stabilized: np.ndarray
    enhanced: np.ndarray
    detection: DiskDetection | None


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return image


def process_image(image: np.ndarray, config: AstroFrameConfig | None = None) -> ProcessResult:
    """Aplica estabilização geométrica e melhoria automática a uma imagem/frame."""
    config = config or AstroFrameConfig()
    bgr = _to_bgr(image)
    stabilized, detection = center_and_stabilize(bgr, config)
    enhanced = enhance_image(stabilized, config)
    return ProcessResult(original=bgr, stabilized=stabilized, enhanced=enhanced, detection=detection)


def process_path(path: str | Path, config: AstroFrameConfig | None = None) -> ProcessResult:
    """Lê um ficheiro de imagem e processa-o com a pipeline completa."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Não foi possível ler a imagem: {path}")
    return process_image(image, config)
