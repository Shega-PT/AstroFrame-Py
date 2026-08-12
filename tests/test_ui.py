"""Testes dos limites UI <-> pipeline (convenção de cores Gradio/OpenCV)."""

from __future__ import annotations

import numpy as np

from astroframe.ui.gradio_app import _from_pipeline, _to_pipeline


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
