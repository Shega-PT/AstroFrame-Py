"""Conversão entre círculos (`DiskDetection`) e camadas RGBA do ImageEditor.

O `gr.ImageEditor` recebe o fundo (a imagem da amostra) mais uma **camada por
círculo** (disco translúcido + bordo opaco) — camadas são arrastáveis na
interface, o que permite **mover** os círculos; pintar por cima com o pincel
**adiciona** (o ponto é reconvertido em círculo) e a borracha **remove**.

`layers_to_circles` devolve um círculo por componente conexa de cada camada
(duas pinturas separadas na mesma camada = dois círculos).
"""

from __future__ import annotations

import cv2
import numpy as np

from astroframe.core.stabilizer import DiskDetection

_FILL = (0, 255, 0)
_RING_ALPHA = 255
_FILL_ALPHA = 90


def _circle_layer(shape: tuple[int, int], disk: DiskDetection) -> np.ndarray:
    """Camada RGBA com o círculo do astro (preenchimento translúcido + bordo)."""
    height, width = shape[:2]
    layer = np.zeros((height, width, 4), dtype=np.uint8)
    cv2.circle(layer, (disk.cx, disk.cy), disk.radius, (*_FILL, _FILL_ALPHA), -1)
    cv2.circle(layer, (disk.cx, disk.cy), disk.radius, (*_FILL, _RING_ALPHA), 1)
    return layer


def circles_to_layers(image: np.ndarray, circles: list[DiskDetection]) -> dict:
    """Valor do `gr.ImageEditor`: fundo (RGB) + uma camada RGBA por círculo."""
    return {"background": np.asarray(image), "layers": [_circle_layer(image.shape[:2], d) for d in circles]}


def _layer_mask(layer: np.ndarray) -> np.ndarray:
    """Máscara binária do conteúdo da camada (alpha ou RGB, consoante o formato)."""
    if layer.ndim == 3 and layer.shape[2] == 4:
        return layer[..., 3] > 0
    if layer.ndim == 3 and layer.shape[2] == 3:
        return np.any(layer[..., :3] > 0, axis=-1)
    return layer > 0


def layers_to_circles(layers) -> list[DiskDetection]:
    """Extrai os círculos desenhados pelo utilizador (um por componente conexa)."""
    circles: list[DiskDetection] = []
    for layer in layers or ():
        if layer is None:
            continue
        mask = _layer_mask(layer)
        if not mask.any():
            continue
        count, labels = cv2.connectedComponents(mask.astype(np.uint8))
        for label in range(1, count):
            ys, xs = np.where(labels == label)
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2
            radius = max(1, max(x1 - x0, y1 - y0) // 2)
            circles.append(DiskDetection(cx, cy, radius))
    return circles
