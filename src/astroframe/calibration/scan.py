"""Varrimento da pasta de exemplos (`samples/`) usada na calibração.

Cada imagem é um item de calibração; cada vídeo contribui com N frames
amostrados de forma **determinística** (`sample_video_frames`), para que a
validação automática seja reproduzível entre execuções. A pasta é varrida
recursivamente e suporta qualquer organização interna (ex.: subpastas por
tipo de astro — eclipse, lua, sol, planetas).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from astroframe.video.reader import FrameReader

IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
VIDEO_EXTS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".m4v"})

DEFAULT_FRAMES_PER_VIDEO = 8


@dataclass(frozen=True)
class SampleRef:
    """Um item de calibração: uma imagem ou um frame amostrado de um vídeo.

    `key` é a chave estável no store (`path_relativo#frame`); `label` é o
    texto apresentado na interface.
    """

    kind: str
    path: Path
    frame: int | None
    key: str
    label: str


def item_key(relpath: str | Path, frame: int | None = None) -> str:
    """Chave estável de um item: path relativo (+ `#frame` para vídeos)."""
    rel = Path(relpath).as_posix()
    return f"{rel}#{frame}" if frame is not None else rel


def item_label(kind: str, relpath: str | Path, frame: int | None = None) -> str:
    """Texto apresentado na interface: `IMG path` / `VID path #frame`."""
    tag = "IMG" if kind == "image" else "VID"
    base = f"{tag} {Path(relpath).as_posix()}"
    return f"{base} #{frame}" if frame is not None else base


def sample_video_frames(frame_count: int, n: int = DEFAULT_FRAMES_PER_VIDEO) -> list[int]:
    """Índices de frames equidistantes (meios-intervalos, determinísticos).

    Com `frame_count` desconhecido (0) devolve os primeiros `n` índices; com
    menos frames que `n`, devolve todos.
    """
    n = max(1, n)
    if frame_count <= 0:
        return list(range(n))
    if frame_count <= n:
        return list(range(frame_count))
    return [min(frame_count - 1, int((i + 0.5) * frame_count / n)) for i in range(n)]


def _kind_for(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTS:
        return "image"
    if suffix in VIDEO_EXTS:
        return "video"
    return None


def scan_samples(root: str | Path, frames_per_video: int = DEFAULT_FRAMES_PER_VIDEO) -> list[SampleRef]:
    """Varre a pasta recursivamente e devolve as amostras ordenadas.

    Vídeos ilegíveis são ignorados (com log de aviso); as suas frames amostradas
    são geradas a partir do `frame_count` real do ficheiro.
    """
    import logging

    logger = logging.getLogger(__name__)
    root = Path(root)
    items: list[SampleRef] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        kind = _kind_for(path)
        if kind is None:
            continue
        rel = path.relative_to(root).as_posix()
        if kind == "image":
            items.append(SampleRef("image", path, None, item_key(rel), item_label("image", rel)))
            continue
        try:
            with FrameReader(path) as reader:
                count = reader.frame_count
        except ValueError:
            logger.warning("Vídeo ignorado na calibração (ilegível): %s", rel)
            continue
        for frame in sample_video_frames(count, frames_per_video):
            items.append(
                SampleRef("video", path, frame, item_key(rel, frame), item_label("video", rel, frame))
            )
    return items


def load_frame(sample: SampleRef) -> np.ndarray:
    """Lê a imagem ou o frame do vídeo correspondente à amostra (BGR)."""
    if sample.kind == "image":
        frame = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Não foi possível ler a imagem: {sample.path}")
        return frame
    with FrameReader(sample.path) as reader:
        return reader.frame_at(sample.frame)
