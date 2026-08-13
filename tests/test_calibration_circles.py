"""Testes da conversão círculos <-> camadas RGBA do ImageEditor."""

from __future__ import annotations

import numpy as np

from astroframe.calibration.circles import circles_to_layers, layers_to_circles
from astroframe.core.stabilizer import DiskDetection


def _rgba_layer(mask: np.ndarray) -> np.ndarray:
    layer = np.zeros((*mask.shape, 4), dtype=np.uint8)
    layer[..., 3] = mask.astype(np.uint8) * 255
    layer[..., :3] = (0, 255, 0)
    return layer


def test_circles_para_layers_roundtrip():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    circles = [DiskDetection(40, 30, 15), DiskDetection(90, 70, 10)]
    value = circles_to_layers(image, circles)
    assert value["background"].shape == (100, 120, 3)
    assert len(value["layers"]) == 2
    extracted = layers_to_circles(value["layers"])
    assert len(extracted) == 2
    assert extracted[0] == DiskDetection(40, 30, 15)
    assert extracted[1] == DiskDetection(90, 70, 10)


def test_layers_pintura_dupla_na_mesma_camada_gera_dois_circulos():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:20, 10:20] = True
    mask[70:90, 70:90] = True
    extracted = layers_to_circles([_rgba_layer(mask)])
    assert len(extracted) == 2
    assert extracted[0] == DiskDetection(14, 14, 4)
    assert extracted[1] == DiskDetection(79, 79, 9)


def test_layers_vazias_e_none():
    assert layers_to_circles(None) == []
    assert layers_to_circles([]) == []
    assert layers_to_circles([None]) == []


def test_layer_sem_alpha_usa_rgb():
    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    rgb[5:15, 5:15] = (10, 200, 30)
    extracted = layers_to_circles([rgb])
    assert extracted == [DiskDetection(9, 9, 4)]


def test_circulo_no_bordo_nao_estoura():
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    value = circles_to_layers(image, [DiskDetection(2, 2, 5)])
    (extracted,) = layers_to_circles(value["layers"])
    assert 0 <= extracted.cx <= 49
    assert 0 <= extracted.cy <= 49


def test_layer_escala_cinza_2d_usa_valor_nao_zero():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    assert layers_to_circles([mask]) == [DiskDetection(9, 9, 4)]


def test_layer_com_mascara_vazia_e_ignorada():
    empty_rgba = np.zeros((40, 40, 4), dtype=np.uint8)
    full_rgba = np.zeros((40, 40, 4), dtype=np.uint8)
    full_rgba[5:15, 5:15, 3] = 255
    assert layers_to_circles([empty_rgba, full_rgba]) == [DiskDetection(9, 9, 4)]
