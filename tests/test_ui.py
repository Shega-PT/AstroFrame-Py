"""Testes dos limites UI <-> pipeline (convenção de cores Gradio/OpenCV)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from astroframe.config import AstroFrameConfig
from astroframe.ui.gradio_app import _best_frame_from_video, _from_pipeline, _to_pipeline
from astroframe.video.select import sharpness


def test_to_pipeline_converte_rgb_para_bgr():
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb[0, 0] = (255, 0, 0)
    bgr = _to_pipeline(rgb)
    np.testing.assert_array_equal(bgr[0, 0], (0, 0, 255))


def test_from_pipeline_converte_bgr_para_rgb():
    bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    bgr[0, 0] = (0, 0, 255)
    rgb = _from_pipeline(bgr)
    np.testing.assert_array_equal(rgb[0, 0], (255, 0, 0))


def test_round_trip_cores():
    image = np.arange(10 * 10 * 3, dtype=np.uint8).reshape(10, 10, 3)
    np.testing.assert_array_equal(_from_pipeline(_to_pipeline(image)), image)


def test_passthrough_escala_de_cinza():
    gray = np.zeros((5, 5), dtype=np.uint8)
    np.testing.assert_array_equal(_to_pipeline(gray), gray)


def _write_video(path, frames: list[np.ndarray]) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (width, height))
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def _make_pattern(sharp: bool) -> np.ndarray:
    frame = np.full((60, 80, 3), 15, dtype=np.uint8)
    cv2.circle(frame, (40, 30), 12, (220,) * 3, -1)
    if sharp:
        return frame
    return cv2.GaussianBlur(frame, (15, 15), 5)


def test_best_frame_seleciona_o_frame_mais_nitido(tmp_path):
    sharp, blurry = _make_pattern(True), _make_pattern(False)
    video = tmp_path / "test.avi"
    _write_video(video, [blurry, sharp, blurry])
    best = _best_frame_from_video(str(video), AstroFrameConfig())
    assert sharpness(best) >= sharpness(sharp) * 0.9
    assert sharpness(best) > sharpness(blurry) * 2


def test_best_frame_video_sem_frames_levanta_erro(tmp_path):
    video = tmp_path / "empty.avi"
    height, width, fourcc = 60, 80, cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(video), fourcc, 10.0, (width, height))
    writer.release()
    with pytest.raises(ValueError, match="sem frames"):
        _best_frame_from_video(str(video), AstroFrameConfig())
