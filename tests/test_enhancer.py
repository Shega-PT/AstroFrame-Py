"""Testes do melhorador automático de imagem."""

from __future__ import annotations

import cv2
import numpy as np

from astroframe.config import AstroFrameConfig, DenoiseConfig
from astroframe.core.enhancer import denoise, enhance_image
from tests.helpers import make_disk_image, make_noisy_image


def test_enhance_image_preserva_forma_e_tipo():
    image, _, _ = make_disk_image(add_noise=True)
    out = enhance_image(image, AstroFrameConfig())
    assert out.dtype == np.uint8
    assert out.shape == image.shape


def test_enhance_image_aumenta_contraste():
    image, _, _ = make_disk_image()
    out = enhance_image(image, AstroFrameConfig())
    assert out.std() > image.std()


def test_denoise_reduz_ruido():
    noisy = make_noisy_image()
    config = AstroFrameConfig()
    config.denoise = DenoiseConfig(h=15)
    out = denoise(noisy, config)
    assert out.std() < noisy.std() * 0.5


def test_enhance_image_aceita_escala_de_cinza():
    gray = cv2.cvtColor(make_disk_image()[0], cv2.COLOR_BGR2GRAY)
    out = enhance_image(gray, AstroFrameConfig())
    assert out.ndim == 3
    assert out.dtype == np.uint8


def test_enhance_image_imagem_minuscula_nao_crasha():
    tiny = np.full((4, 4, 3), 100, dtype=np.uint8)
    out = enhance_image(tiny, AstroFrameConfig())
    assert out.shape == tiny.shape


def test_enhance_image_aceita_rgba():
    image, _, _ = make_disk_image()
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    out = enhance_image(rgba, AstroFrameConfig())
    assert out.ndim == 3 and out.shape[2] == 3


def test_enhance_image_aceita_bgr_ja_convertido():
    image, _, _ = make_disk_image()
    out = enhance_image(image, AstroFrameConfig())
    assert out.shape == image.shape


def test_enhance_image_pode_omitir_denoise():
    image, _, _ = make_disk_image(add_noise=True)
    out = enhance_image(image, AstroFrameConfig(), use_denoise=False)
    assert out.shape == image.shape
    assert out.dtype == np.uint8
