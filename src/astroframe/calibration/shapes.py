"""Ferramentas de formas para a calibração: círculos, elipses e transformações.

As camadas do `gr.ImageEditor` são imagens RGBA sobre o fundo (o frame da
amostra). Este módulo desenha formas geométricas diretamente numa camada nova
(círculo ou elipse) e transforma camadas existentes — **mover** (translação de
pixels) e **redimensionar** (escala em torno do próprio centro) — preservando
qualquer traço de pincel desenhado pelo utilizador.

As transformações são ao nível da imagem (não de uma lista de círculos), por
isso funcionam tanto para formas puras como para detalhes pintados à mão.
"""

from __future__ import annotations

import cv2
import numpy as np

from astroframe.calibration.circles import _FILL, _FILL_ALPHA, _RING_ALPHA

CIRCLE = "círculo"
ELLIPSE = "elipse"

_SHAPES = (CIRCLE, ELLIPSE)


def normalize_shape(shape: str | None) -> str:
    """Normaliza o nome da forma escolhida na interface."""
    return ELLIPSE if (shape or "").strip().lower() == ELLIPSE else CIRCLE


def shape_layer(
    shape: str | None,
    diameter: float,
    ratio: float,
    cx: int,
    cy: int,
    frame_size: tuple[int, int],
) -> np.ndarray:
    """Camada RGBA com um círculo ou elipse centrado em (cx, cy).

    `diameter` é o eixo maior em px; `ratio` (0–1) reduz o eixo menor quando
    `shape` é uma elipse (círculo ignora a proporção).
    """
    height, width = frame_size[:2]
    layer = np.zeros((height, width, 4), dtype=np.uint8)
    rx = max(1, int(round(diameter / 2)))
    ry = max(1, int(round(rx * ratio))) if normalize_shape(shape) == ELLIPSE else rx
    center = (int(np.clip(cx, 0, width - 1)), int(np.clip(cy, 0, height - 1)))
    axes = (max(1, int(min(rx, width))), max(1, int(min(ry, height))))
    cv2.ellipse(layer, center, axes, 0, 0, 360, (*_FILL, _FILL_ALPHA), -1)
    cv2.ellipse(layer, center, axes, 0, 0, 360, (*_FILL, _RING_ALPHA), 1)
    return layer


def _layer_mask(layer: np.ndarray) -> np.ndarray:
    """Máscara binária do conteúdo da camada (alpha ou RGB, consoante o formato)."""
    if layer.ndim == 3 and layer.shape[2] == 4:
        return layer[..., 3] > 0
    if layer.ndim == 3 and layer.shape[2] == 3:
        return np.any(layer[..., :3] > 0, axis=-1)
    return layer > 0


def content_bounds(layer: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box do conteúdo da camada: (x0, y0, x1, y1) ou None se vazia."""
    if layer is None or layer.size == 0:
        return None
    mask = _layer_mask(layer)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def content_center(layer: np.ndarray) -> tuple[int, int] | None:
    """Centro do bounding box do conteúdo da camada, ou None se vazia."""
    bounds = content_bounds(layer)
    if bounds is None:
        return None
    x0, y0, x1, y1 = bounds
    return (x0 + x1) // 2, (y0 + y1) // 2


def translate_layer(layer: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Devolve a camada deslocada `dx`/`dy` px (a imagem mantém o tamanho)."""
    if not dx and not dy:
        return layer
    height, width = layer.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(layer, matrix, (width, height), flags=cv2.INTER_LINEAR)


def scale_layer(layer: np.ndarray, target_diameter: float) -> np.ndarray:
    """Redimensiona o conteúdo da camada à volta do próprio centro.

    O eixo maior do conteúdo passa a medir `target_diameter` px (proporções e
    traços de pincel preservados, como toda a camada é re-escalada).
    """
    bounds = content_bounds(layer)
    if bounds is None:
        return layer
    x0, y0, x1, y1 = bounds
    current = max(1, max(x1 - x0, y1 - y0))
    factor = max(0.05, float(target_diameter) / current)
    if abs(factor - 1.0) < 0.005:
        return layer
    height, width = layer.shape[:2]
    center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, 0, factor)
    return cv2.warpAffine(layer, matrix, (width, height), flags=cv2.INTER_LINEAR)


def recenter_layer(layer: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """Move o conteúdo da camada para que o seu centro fique em (cx, cy)."""
    center = content_center(layer)
    if center is None:
        return layer
    return translate_layer(layer, cx - center[0], cy - center[1])
