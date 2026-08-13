"""Pacote de calibração: varrimento de exemplos, ground truth e validação."""

from __future__ import annotations

from astroframe.calibration.circles import circles_to_layers, layers_to_circles
from astroframe.calibration.scan import (
    DEFAULT_FRAMES_PER_VIDEO,
    IMAGE_EXTS,
    VIDEO_EXTS,
    SampleRef,
    item_key,
    item_label,
    load_frame,
    sample_video_frames,
    scan_samples,
)
from astroframe.calibration.store import STORE_VERSION, CalibrationItem, CalibrationStore
from astroframe.calibration.validate import (
    IOU_THRESHOLD,
    CalibrationReport,
    ItemReport,
    circle_iou,
    match_circles,
    suggest_parameters,
    validate_all,
    validate_item,
)

__all__ = [
    "DEFAULT_FRAMES_PER_VIDEO",
    "IMAGE_EXTS",
    "VIDEO_EXTS",
    "SampleRef",
    "item_key",
    "item_label",
    "load_frame",
    "sample_video_frames",
    "scan_samples",
    "STORE_VERSION",
    "CalibrationItem",
    "CalibrationStore",
    "IOU_THRESHOLD",
    "CalibrationReport",
    "ItemReport",
    "circle_iou",
    "match_circles",
    "suggest_parameters",
    "validate_all",
    "validate_item",
    "circles_to_layers",
    "layers_to_circles",
]
