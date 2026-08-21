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


def test_enhance_image_com_cnn_enhance_aplica_residual(monkeypatch):
    from astroframe.ai.cnn import ResidualEnhancer, SmallCNN, fit_residual

    image, _, _ = make_disk_image(add_noise=True)
    rng = np.random.default_rng(0)
    clean = image[:, :, 0].astype(np.float64) / 255.0
    noisy = np.clip(clean + rng.normal(0, 0.05, clean.shape), 0, 1)
    pairs = [
        (np.clip(noisy * 255, 0, 255).astype(np.uint8), (clean * 255).astype(np.uint8)) for _ in range(6)
    ]
    model, _ = fit_residual(pairs, model=SmallCNN(mode="residual", k=2, seed=0), epochs=2, seed=0)
    enhancer = ResidualEnhancer(model=model)
    monkeypatch.setattr("astroframe.core.enhancer._cnn_enhancer", enhancer)

    config = AstroFrameConfig()
    config.ai.cnn_enhance = True
    out = enhance_image(image, config)
    assert out.dtype == np.uint8
    assert out.shape == image.shape
