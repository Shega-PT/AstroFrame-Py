"""Testes do módulo de vídeo: lucky imaging, stacking e anti-trepidação."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from astroframe.config import AstroFrameConfig
from astroframe.core.stabilizer import AntiJitterStabilizer
from astroframe.video.select import estimate_sharpness_threshold, select_sharp_frames, sharpness
from astroframe.video.stacking import select_best, stack_frames
from tests.helpers import make_disk_image


def _sharp_and_blurred():
    sharp, _, _ = make_disk_image()
    blurred = cv2.GaussianBlur(sharp, (21, 21), 0)
    return sharp, blurred


def test_sharpness_distingue_nitido_de_borrado():
    sharp, blurred = _sharp_and_blurred()
    assert sharpness(sharp) > sharpness(blurred)


def test_sharpness_aceita_escala_de_cinza():
    sharp, _ = _sharp_and_blurred()
    gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
    assert sharpness(gray) > 0


def test_estimate_sharpness_threshold():
    scores = [1.0, 2.0, 3.0, 4.0]
    assert estimate_sharpness_threshold(scores, 25.0) == pytest.approx(np.percentile(scores, 25.0))
    assert estimate_sharpness_threshold([], 25.0) == 0.0


def test_select_sharp_frames_com_minimo_explicito():
    sharp, blurred = _sharp_and_blurred()
    seq = [blurred, sharp, blurred]
    threshold = (sharpness(sharp) + sharpness(blurred)) / 2
    selected = select_sharp_frames(seq, minimum=threshold)
    assert [index for index, _, _ in selected] == [1]


def test_select_sharp_frames_estimativa_automatica():
    sharp, blurred = _sharp_and_blurred()
    seq = [blurred, sharp, blurred]
    selected = select_sharp_frames(seq)
    indices = [index for index, _, _ in selected]
    assert 1 in indices


def test_select_best_ordena_por_nitidez():
    sharp, blurred = _sharp_and_blurred()
    best = select_best([blurred, sharp, blurred], n_best=1)
    assert len(best) == 1
    np.testing.assert_array_equal(best[0], sharp)


def test_select_best_requer_n_positivo():
    with pytest.raises(ValueError):
        select_best([np.zeros((10, 10, 3), dtype=np.uint8)], 0)


def test_stack_frames_mediana_remove_outlier():
    base = np.full((30, 40, 3), 50, dtype=np.uint8)
    f1 = base.copy()
    f1[5, 5] = 250
    f2 = f1.copy()
    f3 = base.copy()
    f3[5, 5] = 0
    stacked = stack_frames([f1, f2, f3])
    assert stacked[5, 5, 0] == 250
    assert stacked[0, 0, 0] == 50


def test_stack_frames_erro_com_shapes_diferentes():
    a = np.zeros((30, 40, 3), dtype=np.uint8)
    b = np.zeros((40, 30, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="mesma resolução"):
        stack_frames([a, b])


def test_stack_frames_vazio():
    with pytest.raises(ValueError):
        stack_frames([])


def test_antijitter_reutiliza_deslocamento_sem_deteção():
    config = AstroFrameConfig()
    config.stabilizer.contour_fallback = False
    config.stabilizer.auto_crop = False
    engine = AntiJitterStabilizer(config=config, alpha=1.0)

    frame1, _, _ = make_disk_image(offset=(60, -40))
    _, detection1 = engine.stabilize(frame1)
    assert detection1 is not None

    frame2 = np.full((360, 480, 3), 30, dtype=np.uint8)
    out2, detection2 = engine.stabilize(frame2)
    assert detection2 is None

    dx = 480 // 2 - detection1.cx
    dy = 360 // 2 - detection1.cy
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    expected = cv2.warpAffine(frame2, matrix, (480, 360))
    np.testing.assert_array_equal(out2, expected)


def test_antijitter_aplica_ema_no_centroide():
    config = AstroFrameConfig()
    config.stabilizer.contour_fallback = False
    config.stabilizer.auto_crop = False
    engine = AntiJitterStabilizer(config=config, alpha=0.3)

    frame1, _, _ = make_disk_image(offset=(60, -40))
    out1, detection1 = engine.stabilize(frame1)
    frame2, _, _ = make_disk_image(offset=(150, 100))
    out2, detection2 = engine.stabilize(frame2)
    assert detection1 is not None and detection2 is not None

    smooth_x = 0.3 * detection2.cx + 0.7 * detection1.cx
    smooth_y = 0.3 * detection2.cy + 0.7 * detection1.cy
    dx = 480 // 2 - int(round(smooth_x))
    dy = 360 // 2 - int(round(smooth_y))
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    expected = cv2.warpAffine(frame2, matrix, (480, 360))
    np.testing.assert_array_equal(out2, expected)

    pure_dx = 480 // 2 - detection2.cx
    assert abs(dx) < abs(pure_dx)
