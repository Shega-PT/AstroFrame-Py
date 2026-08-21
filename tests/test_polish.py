"""Testes do polimento por astros (realce individual + remontagem sem costuras)."""

from __future__ import annotations

import cv2
import numpy as np

from astroframe.config import AstroFrameConfig
from astroframe.core.polish import _band_mask, _feather_mask, polish_image
from astroframe.core.stabilizer import DiskDetection


def _disk_image(height: int = 300, width: int = 360, radius: int = 80) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(image, (width // 2, height // 2), radius, (200,) * 3, -1)
    return image


def test_polish_fundo_fora_da_linha_de_recorte_e_media_do_fundo_original():
    image = np.full((300, 360, 3), 90, dtype=np.uint8)
    cfg = AstroFrameConfig()
    cfg.polish.corona_scale = 1.0
    out = polish_image(image, DiskDetection(180, 150, 60), cfg)
    assert out[5, 5].tolist() == [90, 90, 90]
    assert out[180, 150].tolist() == [90, 90, 90]


def test_polish_black_background_opcional_preto_puro():
    image = np.full((300, 360, 3), 90, dtype=np.uint8)
    cfg = AstroFrameConfig()
    cfg.polish.black_background = True
    cfg.polish.corona_scale = 1.0
    out = polish_image(image, DiskDetection(180, 150, 60), cfg)
    assert out[5, 5].tolist() == [0, 0, 0]
    assert out[180, 150].tolist() == [90, 90, 90]


def test_polish_anel_entre_astro_e_recorte_fica_diluido():
    image = np.full((300, 360, 3), 30, dtype=np.uint8)
    cv2.circle(image, (180, 150), 60, (200,) * 3, -1)
    cfg = AstroFrameConfig()
    cfg.polish.corona_scale = 2.0
    cfg.polish.brightness = 0.0
    out = polish_image(image, DiskDetection(180, 150, 60), cfg)
    inside = out[150, 180]
    mid = out[150, 270]
    outside = out[150, 320]
    assert outside.tolist() == [30, 30, 30]
    assert 30 < int(mid[0]) < int(inside[0])
    assert int(inside[0]) >= 200


def test_polish_disco_permanece_intacto_dentro_do_raio():
    image = _disk_image()
    cfg = AstroFrameConfig()
    cfg.polish.corona_scale = 1.0
    cfg.polish.remove_reflections = False
    cfg.polish.brightness = 0.0
    out = polish_image(image, DiskDetection(180, 150, 80), cfg)
    assert out[150, 180].tolist() == [200, 200, 200]


def test_polish_contorno_suave_sem_dentes():
    image = _disk_image()
    cfg = AstroFrameConfig()
    cfg.polish.corona_scale = 1.0
    cfg.polish.feather = 0.03
    cfg.polish.remove_reflections = False
    cfg.polish.brightness = 0.0
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
    cfg.polish.brightness = 0.0
    monkeypatch.setattr(
        "astroframe.core.polish.find_all_disks",
        lambda img, config=None: [main, ghost],
    )
    out = polish_image(image, main, cfg)
    assert out[60, 70].tolist() == [60, 60, 60]
    assert out[150, 180].tolist() == [60, 60, 60]


def test_polish_reflexos_removidos_com_black_background(monkeypatch):
    image = np.full((300, 360, 3), 60, dtype=np.uint8)
    ghost = DiskDetection(70, 60, 25)
    main = DiskDetection(180, 150, 80)
    cfg = AstroFrameConfig()
    cfg.polish.corona_scale = 1.6
    cfg.polish.reflection_min_radius = 5
    cfg.polish.black_background = True
    cfg.polish.brightness = 0.0
    monkeypatch.setattr(
        "astroframe.core.polish.find_all_disks",
        lambda img, config=None: [main, ghost],
    )
    out = polish_image(image, main, cfg)
    assert out[60, 70].tolist() == [0, 0, 0]


def test_polish_reflexo_mantido_quando_remocao_desativada(monkeypatch):
    image = np.full((300, 360, 3), 60, dtype=np.uint8)
    ghost = DiskDetection(70, 60, 25)
    main = DiskDetection(180, 150, 80)
    cfg = AstroFrameConfig()
    cfg.polish.corona_scale = 1.6
    cfg.polish.remove_reflections = False
    cfg.polish.brightness = 0.0
    monkeypatch.setattr(
        "astroframe.core.polish.find_all_disks",
        lambda img, config=None: [main, ghost],
    )
    out = polish_image(image, main, cfg)
    assert out[60, 70].tolist() == [60, 60, 60]


def test_polish_astro_fora_da_imagem_ignorado_sem_falhar(monkeypatch):
    image = np.full((300, 360, 3), 60, dtype=np.uint8)
    main = DiskDetection(180, 150, 80)
    cfg = AstroFrameConfig()
    cfg.polish.remove_reflections = False
    cfg.polish.brightness = 0.0
    monkeypatch.setattr(
        "astroframe.core.polish.find_all_disks",
        lambda img, config=None: [main, DiskDetection(-100, 50, 30), DiskDetection(900, 500, 40)],
    )
    out = polish_image(image, main, cfg)
    assert out.shape == image.shape
    assert out[150, 180].tolist() == [60, 60, 60]


def test_polish_fundo_sem_pixels_fora_do_recorte_nao_falha():
    image = np.full((100, 100, 3), 90, dtype=np.uint8)
    cfg = AstroFrameConfig()
    cfg.polish.corona_scale = 1.0
    cfg.polish.brightness = 0.0
    out = polish_image(image, DiskDetection(50, 50, 80), cfg)
    assert out.shape == image.shape


def test_polish_sem_background_fill_nem_black_devolve_inalterado():
    image = _disk_image()
    cfg = AstroFrameConfig()
    cfg.polish.background_fill = False
    cfg.polish.black_background = False
    out = polish_image(image, DiskDetection(180, 150, 80), cfg)
    np.testing.assert_array_equal(out, image)


def test_polish_dois_astros_tratados_individualmente_sem_cortes(monkeypatch):
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.circle(image, (140, 150), 60, (150, 150, 150), -1)
    cv2.circle(image, (300, 150), 40, (180, 180, 180), -1)
    cfg = AstroFrameConfig()
    cfg.polish.brightness = 0.5
    cfg.polish.black_background = True
    monkeypatch.setattr(
        "astroframe.core.polish.find_all_disks",
        lambda img, config=None: [DiskDetection(140, 150, 60), DiskDetection(300, 150, 40)],
    )
    out = polish_image(image, DiskDetection(140, 150, 60), cfg)
    assert out[150, 140].tolist() == [150, 150, 150]  # 1º astro uniforme preservado intacto
    assert out[150, 300].tolist() == [180, 180, 180]  # 2º astro uniforme preservado intacto
    assert out[150, 385].tolist() == [0, 0, 0]  # fundo fora do recorte


def test_polish_astro_escuro_uniforme_nao_recebe_brilho(monkeypatch):
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.circle(image, (180, 150), 80, (235, 235, 235), -1)
    cv2.circle(image, (90, 90), 30, (12, 12, 12), -1)
    cfg = AstroFrameConfig()
    cfg.polish.brightness = 0.5
    monkeypatch.setattr(
        "astroframe.core.polish.find_all_disks",
        lambda img, config=None: [DiskDetection(180, 150, 80), DiskDetection(90, 90, 30)],
    )
    out = polish_image(image, DiskDetection(180, 150, 80), cfg)
    assert out[90, 90].tolist() == [12, 12, 12]  # silhueta escura preservada
    assert int(out[150, 180][0]) > 200  # astro claro realçado


def test_polish_corpo_real_fora_do_primario_nao_e_removido(monkeypatch):
    image = np.full((300, 360, 3), 60, dtype=np.uint8)
    main = DiskDetection(180, 150, 80)
    body = DiskDetection(300, 60, 50)  # 50 ≥ 0.35 × 80 → corpo real, não ghost
    cfg = AstroFrameConfig()
    cfg.polish.black_background = True
    cfg.polish.brightness = 0.0
    monkeypatch.setattr(
        "astroframe.core.polish.find_all_disks",
        lambda img, config=None: [main, body],
    )
    out = polish_image(image, main, cfg)
    assert out[60, 300].tolist() == [60, 60, 60]  # mantido (não preenchido com o fundo)


def test_polish_ghost_pequeno_fora_do_primario_removido(monkeypatch):
    image = np.full((300, 360, 3), 60, dtype=np.uint8)
    main = DiskDetection(180, 150, 80)
    ghost = DiskDetection(300, 60, 20)  # 20 < 0.35 × 80 → reflexo da lente
    cfg = AstroFrameConfig()
    cfg.polish.black_background = True
    cfg.polish.brightness = 0.0
    monkeypatch.setattr(
        "astroframe.core.polish.find_all_disks",
        lambda img, config=None: [main, ghost],
    )
    out = polish_image(image, main, cfg)
    assert out[60, 300].tolist() == [0, 0, 0]  # removido, preenchido com o fundo preto


def test_polish_imagem_escala_cinza():
    gray = np.zeros((300, 360), dtype=np.uint8)
    cv2.circle(gray, (180, 150), 80, 200, -1)
    cfg = AstroFrameConfig()
    cfg.polish.corona_scale = 1.0
    cfg.polish.remove_reflections = False
    cfg.polish.brightness = 0.0
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


def test_band_mask_um_dentro_e_desce_ate_ao_recorte():
    band = _band_mask((100, 100), 50, 50, 20.0, 40.0, 0.0)
    assert band[50, 50] == 1.0
    assert band[50, 70] == 1.0
    assert 0.0 < band[50, 72] < 1.0
    assert band[50, 90] == 0.0
