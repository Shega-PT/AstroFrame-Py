"""Fábricas de imagens sintéticas partilhadas entre os testes."""

from __future__ import annotations

import cv2
import numpy as np


def center_tolerance(height: int, width: int) -> int:
    """Tolerância aceitável para o centro do disco (auto-crop re-escala a imagem)."""
    return max(5, min(height, width) // 30)


def make_disk_image(
    height: int = 360,
    width: int = 480,
    radius: int = 90,
    offset: tuple[int, int] = (60, -40),
    brightness: int = 200,
    add_noise: bool = False,
) -> tuple[np.ndarray, int, int]:
    """Imagem sintética: disco claro (Sol/Lua) sobre fundo escuro, desviado do centro."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cx = width // 2 + offset[0]
    cy = height // 2 + offset[1]
    cv2.circle(image, (cx, cy), radius, (brightness,) * 3, -1)
    if add_noise:
        noise = np.random.default_rng(0).normal(0, 12, image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image, cx, cy


def make_noisy_image(height: int = 240, width: int = 320) -> np.ndarray:
    """Imagem uniforme com ruído gaussiano (simula fotos com ISO alto)."""
    rng = np.random.default_rng(42)
    base = np.full((height, width, 3), 60, dtype=np.uint8)
    noise = rng.normal(0, 25, base.shape)
    return np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
