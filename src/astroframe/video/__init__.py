"""Processamento de vídeo: leitura de frames, lucky imaging e stacking."""

from __future__ import annotations

from astroframe.video.reader import FrameReader
from astroframe.video.select import estimate_sharpness_threshold, select_sharp_frames, sharpness
from astroframe.video.stacking import select_best, stack_frames

__all__ = [
    "FrameReader",
    "estimate_sharpness_threshold",
    "select_sharp_frames",
    "sharpness",
    "select_best",
    "stack_frames",
]
