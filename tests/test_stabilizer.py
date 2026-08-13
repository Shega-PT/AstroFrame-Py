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
    find_all_disks,
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
    monkeypatch.setattr("astroframe.core.stabilizer.find_all_disks", lambda image, config=None: [])
    out, detection = engine.stabilize(empty)
    assert detection is None
    np.testing.assert_array_equal(out, empty)


def test_find_all_disks_devolve_multiplos_ordenados_por_raio():
    image = np.zeros((360, 480, 3), dtype=np.uint8)
    cv2.circle(image, (300, 200), 90, (200,) * 3, -1)
    cv2.circle(image, (120, 110), 35, (140,) * 3, -1)
    config = AstroFrameConfig()
    config.stabilizer.min_dist = 30
    disks = find_all_disks(image, config)
    assert len(disks) >= 2
    assert disks[0].radius >= disks[1].radius
    assert abs(disks[0].cx - 300) <= 5
    assert abs(disks[1].cx - 120) <= 12


def test_find_all_disks_imagem_vazia_nao_engana():
    assert find_all_disks(np.zeros((100, 100, 3), dtype=np.uint8), AstroFrameConfig()) == []


def test_find_all_disks_com_contorno_fallback_adiciona_sem_duplicar(monkeypatch):
    image, cx, cy = make_disk_image()
    config = AstroFrameConfig()
    config.stabilizer.param2 = 300
    config.stabilizer.contour_fallback = True
    disks = find_all_disks(image, config)
    assert disks, "contorno devia recuperar o disco"
    assert abs(disks[0].cx - cx) <= 5


def test_find_all_disks_duplicados_hough_contorno_nao_repetem(monkeypatch):
    image, cx, cy = make_disk_image()

    def fake_hough(gray, *args, **kwargs):
        return np.array([[[cx, cy, 90]]], dtype=np.float64)

    monkeypatch.setattr("astroframe.core.stabilizer.cv2.HoughCircles", fake_hough)
    config = AstroFrameConfig()
    config.stabilizer.min_radius = 20
    disks = find_all_disks(image, config)
    centers = [(d.cx, d.cy) for d in disks]
    assert centers.count((cx, cy)) == 1


def test_find_all_disks_circulos_concentricos_fundem_no_maior(monkeypatch):
    image, cx, cy = make_disk_image()

    def fake_hough(gray, *args, **kwargs):
        return np.array([[[cx, cy, 90], [cx + 3, cy + 2, 40]]], dtype=np.float64)

    monkeypatch.setattr("astroframe.core.stabilizer.cv2.HoughCircles", fake_hough)
    config = AstroFrameConfig()
    config.stabilizer.min_radius = 20
    disks = find_all_disks(image, config)
    assert len(disks) == 1
    assert abs(disks[0].radius - 90) <= 2


def test_find_all_disks_reflexo_afastado_mantido(monkeypatch):
    image, cx, cy = make_disk_image()

    def fake_hough(gray, *args, **kwargs):
        return np.array([[[cx, cy, 90], [cx + 120, cy - 40, 35]]], dtype=np.float64)

    monkeypatch.setattr("astroframe.core.stabilizer.cv2.HoughCircles", fake_hough)
    config = AstroFrameConfig()
    config.stabilizer.min_radius = 20
    disks = find_all_disks(image, config)
    assert len(disks) == 2


def test_find_all_disks_limita_numero_de_discos(monkeypatch):
    image, cx, cy = make_disk_image()

    def fake_hough(gray, *args, **kwargs):
        circles = [
            [cx, cy, 90],
            [80, 60, 80],
            [80, 220, 70],
            [80, 310, 60],
            [450, 60, 50],
            [450, 300, 40],
        ]
        return np.array([circles], dtype=np.float64)

    monkeypatch.setattr("astroframe.core.stabilizer.cv2.HoughCircles", fake_hough)
    config = AstroFrameConfig()
    config.stabilizer.min_radius = 20
    disks = find_all_disks(image, config)
    assert len(disks) == 5


def test_find_all_disks_anula_em_disco_coberto_sem_anel(monkeypatch):
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    def fake_hough(gray, *args, **kwargs):
        return np.array([[[100, 100, 212], [80, 80, 140]]], dtype=np.float64)

    monkeypatch.setattr("astroframe.core.stabilizer.cv2.HoughCircles", fake_hough)
    config = AstroFrameConfig()
    config.stabilizer.min_radius = 20
    disks = find_all_disks(image, config)
    assert len(disks) == 2


def test_antijitter_last_detection_mantido_em_frame_sem_deteccao(monkeypatch):
    image1, cx, cy = make_disk_image()
    blank = np.zeros_like(image1)
    engine = AntiJitterStabilizer(alpha=0.5)
    real = find_all_disks

    def fake(image, config=None):
        if image is blank:
            return []
        return real(image, config)

    monkeypatch.setattr("astroframe.core.stabilizer.find_all_disks", fake)
    stabilized, detection = engine.stabilize(image1)
    assert detection is not None
    assert engine.last_detection is not None
    out, detection2 = engine.stabilize(blank)
    assert detection2 is None
    assert engine.last_detection is not None
    assert engine.last_detection.radius == 90


def test_antijitter_last_detection_properties():
    engine = AntiJitterStabilizer(alpha=0.5)
    assert engine.last_detection is None
    assert engine.last_all_disks == []


def test_antijitter_last_detection_suave_sem_deteccao_guardada():
    engine = AntiJitterStabilizer()
    engine._smooth = (120.0, 80.0)
    det = engine.last_detection
    assert det is not None
    assert det.cx == 120 and det.cy == 80


def test_antijitter_last_all_disks_retidos_sem_deteccao(monkeypatch):
    image1, cx, cy = make_disk_image()
    blank = np.zeros_like(image1)
    engine = AntiJitterStabilizer(alpha=0.5)
    real = find_all_disks

    def fake(image, config=None):
        if image is blank:
            return []
        return real(image, config)

    monkeypatch.setattr("astroframe.core.stabilizer.find_all_disks", fake)
    engine.stabilize(image1)
    engine.stabilize(blank)
    assert engine.last_all_disks
    assert engine.last_all_disks[0].radius >= 85
