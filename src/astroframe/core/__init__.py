"""Núcleo da pipeline: estabilização geométrica, melhoria automática e orquestração."""

from __future__ import annotations

from astroframe.core.pipeline import ProcessResult, process_image, process_path
from astroframe.core.stabilizer import AntiJitterStabilizer, DiskDetection

__all__ = ["ProcessResult", "process_image", "process_path", "AntiJitterStabilizer", "DiskDetection"]
