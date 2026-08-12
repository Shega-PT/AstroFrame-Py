"""Testes do estabilizador geométrico."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from astroframe.config import AstroFrameConfig
from astroframe.core.stabilizer import (
    AntiJitterStabilizer,
    _intensity_centroid,
    center_and_stabilize,
    find_disk_center,
)
from tests.helpers import center_tolerance, make_disk_image


def test_find_disk_center_deteta_disco():
    image, cx, cy = make_disk_image()
    detection = find_disk_center(image, AstroFrameConfig())
    assert detection is not None
    assert abs(detection.cx - cx) <= 5
    assert abs(detection.cy - cy) <= 5


def test_center_and_stabilize_centraliza_disco():
    image, _, _ = make_disk_image()
    height, width = image.shape[:2]
    stabilized, _ = center_and_stabilize(image, AstroFrameConfig())
    detection = find_disk_center(stabilized, AstroFrameConfig())
    assert detection is not None
    tol = center_tolerance(height, width)
    assert abs(detection.cx - width // 2) <= tol
    assert abs(detection.cy - height // 2) <= tol


def test_center_and_stabilize_sem_disco_devolve_original():
    empty = np.zeros((100, 100, 3), dtype=np.uint8)
    result, detection = center_and_stabilize(empty, AstroFrameConfig())
    assert detection is None
    np.testing.assert_array_equal(result, empty)


def test_center_and_stabilize_funciona_com_ruido():
    image, _, _ = make_disk_image(add_noise=True)
    height, width = image.shape[:2]
    stabilized, _ = center_and_stabilize(image, AstroFrameConfig())
    detection = find_disk_center(stabilized, AstroFrameConfig())
    assert detection is not None
    tol = center_tolerance(height, width)
    assert abs(detection.cx - width // 2) <= tol
    assert abs(detection.cy - height // 2) <= tol


@pytest.mark.parametrize("offset", [(0, 0), (100, 60), (-80, -50)])
def test_center_and_stabilize_varios_desvios(offset):
    image, _, _ = make_disk_image(offset=offset)
    height, width = image.shape[:2]
    stabilized, detection = center_and_stabilize(image, AstroFrameConfig())
    assert detection is not None
    centered = find_disk_center(stabilized, AstroFrameConfig())
    tol = center_tolerance(height, width)
    assert abs(centered.cx - width // 2) <= tol
    assert abs(centered.cy - height // 2) <= tol


def test_intensity_centroid_imagem_uniforme_mantem_centro():
    uniform = np.full((50, 50), 100, dtype=np.uint8)
    assert _intensity_centroid(uniform, 25, 25, 10) == (25, 25)


def test_intensity_centroid_mascara_pequena_mantem_centro():
    gray = np.zeros((50, 50), dtype=np.uint8)
    gray[25, 25] = 255
    assert _intensity_centroid(gray, 25, 25, 10) == (25, 25)


def test_intensity_centroid_refina_para_centro_intensidade():
    gray = np.zeros((60, 60), dtype=np.uint8)
    cv2.circle(gray, (30, 30), 8, 200, -1)
    cx, cy = _intensity_centroid(gray, 30, 30, 12)
    assert abs(cx - 30) <= 2
    assert abs(cy - 30) <= 2


def test_deteccao_em_meia_resolucao_para_frames_grandes():
    image = np.zeros((1280, 1360, 3), dtype=np.uint8)
    cv2.circle(image, (680, 640), 280, (200,) * 3, -1)
    detection = find_disk_center(image, AstroFrameConfig())
    assert detection is not None
    assert abs(detection.cx - 680) <= 15
    assert abs(detection.cy - 640) <= 15


def test_antijitter_primeiro_frame_sem_deteccao_devolve_inalterado(monkeypatch):
    empty = np.zeros((100, 120, 3), dtype=np.uint8)
    engine = AntiJitterStabilizer(alpha=0.5)
    monkeypatch.setattr("astroframe.core.stabilizer.find_disk_center", lambda image, config=None: None)
    out, detection = engine.stabilize(empty)
    assert detection is None
    np.testing.assert_array_equal(out, empty)
