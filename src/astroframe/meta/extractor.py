"""Extração de metadados de imagens e vídeos, sem dependências novas.

Implementação própria (MIT) inspirada na cascata do repositório MetadataExplorer
(ffprobe -> OpenCV/PIL), com o objetivo de alimentar a otimização automática:
- Vídeo: ffprobe (se instalado) fornece codec/bitrate/duração; o OpenCV
  (`cv2.VideoCapture`) garante dimensões, fps e contagem de frames — sempre
  disponível, sem dependências externas.
- Imagem: PIL (Pillow, já dependência do Gradio) lê formato, dimensões, modo,
  DPI e EXIF (ISO, exposição, abertura, distância focal, câmara, data).

Toda a falha de leitura é silenciosa: os campos ficam a `None` em vez de
abortar o processamento.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2

try:
    from PIL import ExifTags, Image
except ImportError:  # pragma: no cover - ambiente sem Pillow
    Image = None
    ExifTags = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp", ".gif", ".heic", ".heif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".mpeg", ".mpg", ".ts"}

_FFPROBE_TIMEOUT = 30

# Tags EXIF mais relevantes para a otimização (nomes do PIL).
_EXIF_ISO = 34855
_EXIF_EXPOSURE = 33434
_EXIF_FNUMBER = 33437
_EXIF_FOCAL = 37386
_EXIF_MAKE = 271
_EXIF_MODEL = 272
_EXIF_DATETIME = 36867


@dataclass
class MediaMetadata:
    """Metadados normalizados de uma imagem ou vídeo.

    `raw` mantém a informação completa (JSON do ffprobe/EXIF) para exibição;
    os restantes campos são os valores interpretados e usados pelas sugestões.
    """

    path: str = ""
    kind: str = "unknown"
    width: int | None = None
    height: int | None = None
    aspect_ratio: float | None = None
    fps: float | None = None
    frame_count: int | None = None
    duration: float | None = None
    codec: str | None = None
    bitrate: float | None = None
    format_name: str | None = None
    iso: int | None = None
    exposure_time: float | None = None
    focal_length: float | None = None
    aperture: float | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    captured_at: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def _classify(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return "unknown"


def _to_float(value: Any) -> float | None:
    """Converte valores EXIF (tuplos racionais, Fraction, números) em float."""
    if value is None:
        return None
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, tuple) and len(value) == 2 and value[1] != 0:
        return float(value[0]) / float(value[1])
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def aspect_text(width: int | None, height: int | None) -> str | None:
    """Devolve a proporção como texto '16:9', ou None se desconhecida."""
    if not width or not height:
        return None
    divisor = math.gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def _extract_image_metadata(path: Path, meta: MediaMetadata) -> None:
    if Image is None:  # pragma: no cover - ambiente sem Pillow
        return
    with Image.open(path) as image:
        meta.width = int(image.width)
        meta.height = int(image.height)
        meta.aspect_ratio = meta.width / meta.height if meta.height else None
        dpi = image.info.get("dpi")
        meta.raw["format"] = image.format
        meta.raw["mode"] = image.mode
        if dpi and isinstance(dpi, tuple) and len(dpi) == 2:
            meta.raw["dpi"] = tuple(int(round(d)) for d in dpi)
        exif = image.getexif()
        tags = {}
        if ExifTags is not None:
            tags = {ExifTags.TAGS.get(tag_id, tag_id): value for tag_id, value in exif.items()}
        meta.raw["exif"] = {key: (str(value)[:1000]) for key, value in tags.items()}
        meta.iso = _to_int(exif.get(_EXIF_ISO))
        meta.exposure_time = _to_float(exif.get(_EXIF_EXPOSURE))
        meta.aperture = _to_float(exif.get(_EXIF_FNUMBER))
        meta.focal_length = _to_float(exif.get(_EXIF_FOCAL))
        meta.camera_make = _as_text(exif.get(_EXIF_MAKE))
        meta.camera_model = _as_text(exif.get(_EXIF_MODEL))
        meta.captured_at = _as_text(exif.get(_EXIF_DATETIME))


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_text(value: Any) -> str | None:
    text = str(value).strip()
    return text or None


def _ffprobe(path: Path) -> dict[str, Any] | None:
    """Metadados ricos do vídeo via ffprobe (JSON). None se indisponível/falhar."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=_FFPROBE_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _extract_video_metadata(path: Path, meta: MediaMetadata) -> None:
    capture = cv2.VideoCapture(str(path))
    if capture.isOpened():
        meta.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        meta.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
        meta.aspect_ratio = meta.width / meta.height if meta.width and meta.height else None
        meta.fps = float(capture.get(cv2.CAP_PROP_FPS)) or None
        meta.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        capture.release()

    probe = _ffprobe(path)
    if probe is None:
        return
    meta.raw["ffprobe"] = probe
    stream = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), None)
    if stream is not None:
        meta.codec = stream.get("codec_name")
        if meta.width is None:
            meta.width = stream.get("width")
        if meta.height is None:
            meta.height = stream.get("height")
            meta.aspect_ratio = meta.width / meta.height if meta.width and meta.height else None
        if meta.fps is None:
            meta.fps = _parse_fps(stream)
        meta.duration = _to_float(stream.get("duration"))
        meta.raw["video_stream"] = {key: value for key, value in stream.items() if value is not None}
    fmt = probe.get("format")
    if fmt is not None:
        meta.format_name = fmt.get("format_name")
        meta.bitrate = _to_float(fmt.get("bit_rate"))
        if meta.duration is None:
            meta.duration = _to_float(fmt.get("duration"))
        meta.raw["format"] = {key: value for key, value in fmt.items() if value is not None}


def _parse_fps(stream: dict[str, Any]) -> float | None:
    for key in ("avg_frame_rate", "r_frame_rate"):
        text = stream.get(key)
        if text is None:
            continue
        if isinstance(text, str) and "/" in text:
            numerator, denominator = text.split("/", 1)
            try:
                if float(denominator):
                    return float(numerator) / float(denominator)
            except ValueError:
                continue
        else:
            value = _to_float(text)
            if value is not None:
                return value
    return None


def extract_metadata(path: str | Path) -> MediaMetadata:
    """Lê os metadados de uma imagem ou vídeo, devolvendo um `MediaMetadata`."""
    path = Path(path)
    meta = MediaMetadata(path=str(path), kind=_classify(path))

    if meta.kind == "image":
        try:
            _extract_image_metadata(path, meta)
        except Exception:
            pass
    elif meta.kind == "video":
        try:
            _extract_video_metadata(path, meta)
        except Exception:
            pass

    return meta
