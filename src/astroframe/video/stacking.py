"""Stacking: combinação dos melhores frames para reduzir ruído (ISO alto)."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from astroframe.config import AstroFrameConfig, StackingConfig
from astroframe.video.select import sharpness

logger = logging.getLogger(__name__)


def stack_frames(frames: Sequence[np.ndarray], stacking: StackingConfig | None = None) -> np.ndarray:
    """Combina frames (mediana ou média) num único frame com menos ruído."""
    if not frames:
        raise ValueError("stack_frames requer pelo menos um frame")
    stacking = stacking or StackingConfig()

    height, width = frames[0].shape[:2]
    for frame in frames[1:]:
        if frame.shape[:2] != (height, width):
            raise ValueError(
                f"stack_frames requer frames com a mesma resolução: {(height, width)} vs {frame.shape[:2]}"
            )

    if height * width > 1920 * 1080 and len(frames) > 10:
        logger.warning(
            "Stacking de %d frames acima de 1080p pode exigir muita memória (float32).",
            len(frames),
        )

    converted = [frame.astype(np.float32) for frame in frames]
    if stacking.use_median:
        result = np.median(converted, axis=0)
    else:
        result = np.mean(converted, axis=0)
    return np.clip(result, 0, 255).astype(np.uint8)


def select_best(
    frames: Sequence[np.ndarray],
    n_best: int,
    config: AstroFrameConfig | None = None,
) -> list[np.ndarray]:
    """Devolve os N frames mais nítidos da sequência."""
    if n_best < 1:
        raise ValueError("n_best deve ser >= 1")
    scored = sorted(
        ((sharpness(frame, config), frame) for frame in frames),
        key=lambda item: item[0],
        reverse=True,
    )
    return [frame for _, frame in scored[:n_best]]
