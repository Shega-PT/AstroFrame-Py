"""Orquestra o pipeline completo: estabilizar -> melhorar."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from astroframe.config import AstroFrameConfig
from astroframe.core.enhancer import enhance_image
from astroframe.core.polish import polish_image
from astroframe.core.stabilizer import DiskDetection, center_and_stabilize

logger = logging.getLogger(__name__)


@dataclass
class ProcessResult:
    original: np.ndarray
    stabilized: np.ndarray
    enhanced: np.ndarray
    enhanced_raw: np.ndarray
    detection: DiskDetection | None


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return image


def process_image(image: np.ndarray, config: AstroFrameConfig | None = None) -> ProcessResult:
    """Aplica estabilização geométrica, melhoria e polimento final.

    `enhanced_raw` é o resultado **antes** do polimento — é nele que a
    avaliação (score) mede o ruído/redondeza/reflexos, para o mérito não
    ser "comprado" pelo fundo preto do polimento.
    """
    config = config or AstroFrameConfig()
    bgr = _to_bgr(image)
    stabilized, detection = center_and_stabilize(bgr, config)
    raw = enhance_image(stabilized, config)
    if detection is not None:
        center = DiskDetection(stabilized.shape[1] // 2, stabilized.shape[0] // 2, detection.radius)
        enhanced = polish_image(raw, center, config)
    else:
        enhanced = raw
    return ProcessResult(
        original=bgr,
        stabilized=stabilized,
        enhanced=enhanced,
        enhanced_raw=raw,
        detection=detection,
    )


def process_path(path: str | Path, config: AstroFrameConfig | None = None) -> ProcessResult:
    """Lê um ficheiro de imagem e processa-o com a pipeline completa."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Não foi possível ler a imagem: {path}")
    return process_image(image, config)
