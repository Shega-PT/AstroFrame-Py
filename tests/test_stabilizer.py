"""Testes do estabilizador geométrico."""

from __future__ import annotations

import numpy as np
import pytest

from astroframe.config import AstroFrameConfig
from astroframe.core.stabilizer import center_and_stabilize, find_disk_center
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
