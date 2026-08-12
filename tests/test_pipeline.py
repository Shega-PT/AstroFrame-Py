"""Testes do pipeline completo (estabilizar + melhorar)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from astroframe.config import AstroFrameConfig
from astroframe.core.pipeline import process_image, process_path
from astroframe.core.stabilizer import find_disk_center
from tests.helpers import center_tolerance, make_disk_image


def test_process_image_completo():
    image, _, _ = make_disk_image(offset=(40, -30), add_noise=True)
    result = process_image(image, AstroFrameConfig())
    assert result.original.shape == result.stabilized.shape == result.enhanced.shape
    assert result.detection is not None
    assert result.original.dtype == result.stabilized.dtype == np.uint8

    height, width = result.stabilized.shape[:2]
    centered = find_disk_center(result.stabilized, AstroFrameConfig())
    assert centered is not None
    tol = center_tolerance(height, width)
    assert abs(centered.cx - width // 2) <= tol
    assert abs(centered.cy - height // 2) <= tol


def test_process_image_aceita_escala_de_cinza():
    gray, _, _ = make_disk_image()
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    result = process_image(gray, AstroFrameConfig())
    assert result.enhanced.ndim == 3
    assert result.enhanced.shape[:2] == gray.shape


def test_process_path_inexistente_levanta_erro():
    with pytest.raises(ValueError):
        process_path("/caminho/que/nao/existe.png")


def test_process_image_aceita_rgba():
    rgba, _, _ = make_disk_image()
    rgba = cv2.cvtColor(rgba, cv2.COLOR_BGR2RGBA)
    result = process_image(rgba, AstroFrameConfig())
    assert result.enhanced.ndim == 3
    assert result.enhanced.shape == (360, 480, 3)


def test_process_image_imagem_minuscula_nao_crasha():
    tiny = np.zeros((6, 6, 3), dtype=np.uint8)
    result = process_image(tiny, AstroFrameConfig())
    assert result.enhanced.shape == tiny.shape


def test_process_path_com_ficheiro_temporario(tmp_path):
    image, _, _ = make_disk_image()
    path = tmp_path / "foto.jpg"
    cv2.imwrite(str(path), image)
    result = process_path(path, AstroFrameConfig())
    assert result.detection is not None
    assert result.enhanced.shape == image.shape
