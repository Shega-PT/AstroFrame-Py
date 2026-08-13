"""Iterador frame-a-frame de ficheiros de vídeo (OpenCV VideoCapture)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class FrameReader:
    """Itera sobre os frames de um vídeo, libertando o recurso no fim."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {self.path}")

    @property
    def fps(self) -> float:
        return float(self.cap.get(cv2.CAP_PROP_FPS))

    @property
    def frame_count(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def size(self) -> tuple[int, int]:
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def __iter__(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame

    def frame_at(self, index: int) -> np.ndarray:
        """Devolve o frame no índice `index` (0-based), posicionando a leitura.

        Levanta `ValueError` se o frame não puder ser lido.
        """
        index = max(0, index)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            raise ValueError(f"Falha ao ler o frame {index} de {self.path}")
        return frame

    def close(self) -> None:
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()
