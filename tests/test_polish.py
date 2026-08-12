"""Testes do polimento final (fundo preto, contorno redondo, reflexos)."""

from __future__ import annotations

import cv2
import numpy as np

from astroframe.config import AstroFrameConfig
from astroframe.core.polish import _feather_mask, polish_image
from astroframe.core.stabilizer import DiskDetection


def _disk_image(height: int = 300, width: int = 360, radius: int = 80) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(image, (width // 2, height // 2), radius, (200,) * 3, -1)
    return image


def test_polish_fundo_fora_da_coroa_fica_preto():
    image = np.full((300, 360, 3), 90, dtype=np.uint8)
    cfg = AstroFrameConfig()
    cfg.polish.corona_scale = 1.0
    detection = DiskDetection(180, 150, 60)
    out = polish_image(image, detection, cfg)
    assert out[5, 5].tolist() == [0, 0, 0]
    assert out[180, 150].tolist() == [90, 90, 90]


def test_polish_disco_permanece_intacto_dentro_do_raio():
    image = _disk_image()
    cfg = AstroFrameConfig()
    cfg.polish.corona_scale = 1.0
    cfg.polish.remove_reflections = False
    out = polish_image(image, DiskDetection(180, 150, 80), cfg)
    assert out[150, 180].tolist() == [200, 200, 200]


def test_polish_contorno_suave_sem_dentes():
    image = _disk_image()
    cfg = AstroFrameConfig()
    cfg.polish.corona_scale = 1.0
    cfg.polish.feather = 0.03
    cfg.polish.remove_reflections = False
    out = polish_image(image, DiskDetection(180, 150, 80), cfg)
    edge = out[150, 180 + 80]
    assert 0 < int(edge[0]) < 200


def test_polish_sem_deteccao_devolve_inalterado():
    image = _disk_image()
    out = polish_image(image, None, AstroFrameConfig())
    np.testing.assert_array_equal(out, image)


def test_polish_deteccao_fora_da_imagem_devolve_inalterado():
    image = _disk_image()
    cfg = AstroFrameConfig()
    out = polish_image(image, DiskDetection(-50, -50, 80), cfg)
    np.testing.assert_array_equal(out, image)


def test_polish_raio_zero_ou_negativo_devolve_inalterado():
    image = _disk_image()
    assert polish_image(image, DiskDetection(180, 150, 0), AstroFrameConfig()) is not None
    assert polish_image(image, DiskDetection(180, 150, -5), AstroFrameConfig()) is not None
    np.testing.assert_array_equal(polish_image(image, DiskDetection(180, 150, 0), AstroFrameConfig()), image)


def test_polish_desativado_devolve_inalterado():
    image = _disk_image()
    cfg = AstroFrameConfig()
    cfg.polish.enabled = False
    cfg.polish.corona_scale = 2.0
    out = polish_image(image, DiskDetection(180, 150, 80), cfg)
    np.testing.assert_array_equal(out, image)


def test_polish_reflexos_removidos_da_coroa(monkeypatch):
    image = np.full((300, 360, 3), 60, dtype=np.uint8)
    ghost = DiskDetection(70, 60, 25)
    main = DiskDetection(180, 150, 80)
    cfg = AstroFrameConfig()
    cfg.polish.corona_scale = 1.6
    cfg.polish.reflection_min_radius = 5
    monkeypatch.setattr(
        "astroframe.core.polish.find_all_disks",
        lambda img, config=None: [main, ghost],
    )
    out = polish_image(image, main, cfg)
    assert out[60, 70].tolist() == [0, 0, 0]
    assert out[150, 180].tolist() == [60, 60, 60]


def test_polish_sem_black_background_mantem_fundo(monkeypatch):
    image = np.full((300, 360, 3), 60, dtype=np.uint8)
    cfg = AstroFrameConfig()
    cfg.polish.black_background = False
    cfg.polish.remove_reflections = False
    out = polish_image(image, DiskDetection(180, 150, 80), cfg)
    assert out[5, 5].tolist() == [60, 60, 60]


def test_polish_reflexoes_primeiro_circulo_nao_se_remove(monkeypatch):
    image = np.full((300, 360, 3), 60, dtype=np.uint8)
    main = DiskDetection(180, 150, 80)
    cfg = AstroFrameConfig()
    cfg.polish.corona_scale = 1.0
    monkeypatch.setattr(
        "astroframe.core.polish.find_all_disks",
        lambda img, config=None: [main, DiskDetection(180, 150, 40)],
    )
    out = polish_image(image, main, cfg)
    assert out[150, 180].tolist() == [60, 60, 60]


def test_polish_imagem_escala_cinza():
    gray = np.zeros((300, 360), dtype=np.uint8)
    cv2.circle(gray, (180, 150), 80, 200, -1)
    cfg = AstroFrameConfig()
    cfg.polish.corona_scale = 1.0
    cfg.polish.remove_reflections = False
    out = polish_image(gray, DiskDetection(180, 150, 80), cfg)
    assert out.shape == gray.shape
    assert out[150, 180] == 200
    assert out[5, 5] == 0


def test_polish_ignora_reflexos_inferiores_ao_minimo(monkeypatch):
    image = np.full((300, 360, 3), 60, dtype=np.uint8)
    main = DiskDetection(180, 150, 80)
    cfg = AstroFrameConfig()
    cfg.polish.black_background = False
    cfg.polish.reflection_min_radius = 8
    monkeypatch.setattr(
        "astroframe.core.polish.find_all_disks",
        lambda img, config=None: [main, DiskDetection(70, 60, 4)],
    )
    out = polish_image(image, main, cfg)
    assert out[60, 70].tolist() == [60, 60, 60]


def test_feather_mask_sem_feather_redonda():
    mask = _feather_mask((100, 100), 50, 50, 20, 0.0)
    assert mask[50, 50] == 1.0
    assert mask[50, 75] == 0.0
    assert mask[50, 76] == 0.0
