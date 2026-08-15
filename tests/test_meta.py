"""Testes da leitura de metadados e das sugestões de otimização (pacote meta/)."""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pytest

from astroframe.meta.extractor import (
    MediaMetadata,
    _as_text,
    _ffprobe,
    _parse_fps,
    _to_float,
    _to_int,
    aspect_text,
    extract_metadata,
)
from astroframe.meta.suggest import (
    _format_bitrate,
    _format_duration,
    _format_exposure,
    suggest_config,
    summary_fields,
)


def _write_video(path: Path, frames: list[np.ndarray], fps: float = 10.0) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def _make_frame() -> np.ndarray:
    frame = np.full((60, 80, 3), 15, dtype=np.uint8)
    cv2.circle(frame, (40, 30), 12, (220,) * 3, -1)
    return frame


# ---------------------------------------------------------------------------
# extract_metadata — vídeo (fallback OpenCV)
# ---------------------------------------------------------------------------


def test_video_metadata_via_opencv(tmp_path):
    video = tmp_path / "clip.avi"
    _write_video(video, [_make_frame() for _ in range(6)], fps=12.0)
    meta = extract_metadata(video)
    assert meta.kind == "video"
    assert (meta.width, meta.height) == (80, 60)
    assert meta.aspect_ratio == pytest.approx(80 / 60)
    assert meta.fps == pytest.approx(12.0)
    assert meta.frame_count == 6


def test_video_metadata_aplica_campos_do_ffprobe(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    probe = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30/1",
                "duration": "12.5",
            }
        ],
        "format": {"format_name": "mov,mp4,m4a", "bit_rate": "6000000", "duration": "12.5"},
    }
    monkeypatch.setattr("astroframe.meta.extractor._ffprobe", lambda path: probe)
    meta = extract_metadata(video)
    assert meta.codec == "h264"
    assert meta.width == 1920
    assert meta.duration == pytest.approx(12.5)
    assert meta.bitrate == 6_000_000
    assert meta.format_name == "mov,mp4,m4a"
    assert meta.fps == pytest.approx(30.0)


def test_video_ffprobe_nao_sobrescreve_dimensoes_do_opencv(tmp_path, monkeypatch):
    video = tmp_path / "clip.avi"
    _write_video(video, [_make_frame() for _ in range(2)])
    probe = {
        "streams": [{"codec_type": "video", "codec_name": "mjpeg", "width": 320, "height": 240}],
        "format": {},
    }
    monkeypatch.setattr("astroframe.meta.extractor._ffprobe", lambda path: probe)
    meta = extract_metadata(video)
    assert (meta.width, meta.height) == (80, 60)
    assert meta.aspect_ratio == pytest.approx(80 / 60)
    assert meta.codec == "mjpeg"


# ---------------------------------------------------------------------------
# extract_metadata — imagem (PIL + EXIF)
# ---------------------------------------------------------------------------


def test_image_metadata_exif(tmp_path):
    from PIL import Image

    image = Image.new("RGB", (640, 480), (30, 30, 30))
    exif = Image.Exif()
    exif[34855] = 3200  # ISOSpeedRatings
    exif[33434] = (1, 500)  # ExposureTime
    exif[33437] = (28, 10)  # FNumber f/2.8
    exif[37386] = (50, 1)  # FocalLength
    exif[271] = "CanonTest"  # Make
    exif[272] = "EOS-Test"  # Model
    exif[36867] = "2026:08:12 12:00:00"  # DateTimeOriginal
    path = tmp_path / "foto.jpg"
    image.save(str(path), exif=exif)

    meta = extract_metadata(path)
    assert meta.kind == "image"
    assert (meta.width, meta.height) == (640, 480)
    assert meta.aspect_ratio == pytest.approx(4 / 3)
    assert meta.iso == 3200
    assert meta.exposure_time == pytest.approx(1 / 500)
    assert meta.aperture == pytest.approx(2.8)
    assert meta.focal_length == pytest.approx(50.0)
    assert meta.camera_make == "CanonTest"
    assert meta.camera_model == "EOS-Test"
    assert meta.captured_at == "2026:08:12 12:00:00"
    assert meta.raw["format"] == "JPEG"
    assert "exif" in meta.raw


def test_image_metadata_imagem_plain(tmp_path):
    from PIL import Image

    path = tmp_path / "foto.png"
    Image.new("RGB", (100, 200)).save(str(path))
    meta = extract_metadata(path)
    assert meta.kind == "image"
    assert (meta.width, meta.height) == (100, 200)
    assert meta.iso is None
    assert meta.raw["mode"] == "RGB"


def test_metadata_tipo_desconhecido(tmp_path):
    path = tmp_path / "dados.bin"
    path.write_bytes(b"x")
    meta = extract_metadata(path)
    assert meta.kind == "unknown"
    assert meta.width is None
    assert meta.raw == {}


def test_metadata_imagem_com_falha_interna_devolve_vazio(tmp_path, monkeypatch):
    path = tmp_path / "foto.jpg"
    path.write_bytes(b"nao e imagem")
    monkeypatch.setattr(
        "astroframe.meta.extractor._extract_image_metadata",
        lambda path, meta: (_ for _ in ()).throw(RuntimeError("corrompido")),
    )
    meta = extract_metadata(path)
    assert meta.kind == "image"
    assert meta.width is None


def test_metadata_video_com_falha_interna_devolve_vazio(tmp_path, monkeypatch):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"nao e video")
    monkeypatch.setattr(
        "astroframe.meta.extractor._extract_video_metadata",
        lambda path, meta: (_ for _ in ()).throw(RuntimeError("corrompido")),
    )
    meta = extract_metadata(path)
    assert meta.kind == "video"
    assert meta.width is None


# ---------------------------------------------------------------------------
# helpers internos do extractor
# ---------------------------------------------------------------------------


def test_aspect_text():
    assert aspect_text(1920, 1080) == "16:9"
    assert aspect_text(400, 300) == "4:3"
    assert aspect_text(None, 1080) is None
    assert aspect_text(0, 0) is None


def test_to_float_variantes():
    assert _to_float(Fraction(1, 500)) == pytest.approx(0.002)
    assert _to_float((1, 500)) == pytest.approx(0.002)
    assert _to_float("12.5") == pytest.approx(12.5)
    assert _to_float(7) == 7.0
    assert _to_float(None) is None
    assert _to_float("abc") is None
    assert _to_float(float("inf")) is None
    assert _to_float((1, 0)) is None


def test_to_int_e_as_text():
    assert _to_int("42") == 42
    assert _to_int("x") is None
    assert _as_text("  valor  ") == "valor"
    assert _as_text("   ") is None
    assert _as_text(None) == "None"


def test_parse_fps():
    assert _parse_fps({"avg_frame_rate": "30/1"}) == 30.0
    assert _parse_fps({"avg_frame_rate": "0/0", "r_frame_rate": "25/1"}) == 25.0
    assert _parse_fps({"avg_frame_rate": "abc/def", "r_frame_rate": "25/1"}) == 25.0
    assert _parse_fps({"avg_frame_rate": "29.97"}) == pytest.approx(29.97)
    assert _parse_fps({"avg_frame_rate": "garbage"}) is None
    assert _parse_fps({}) is None


def test_ffprobe_ausente_devolve_none(monkeypatch):
    def boom(*args, **kwargs):
        raise FileNotFoundError("ffprobe não instalado")

    monkeypatch.setattr("astroframe.meta.extractor.subprocess.run", boom)
    assert _ffprobe(Path("x.mp4")) is None


def test_ffprobe_timeout_devolve_none(monkeypatch):
    def boom(*args, **kwargs):
        raise subprocess.TimeoutExpired("ffprobe", 30)

    monkeypatch.setattr("astroframe.meta.extractor.subprocess.run", boom)
    assert _ffprobe(Path("x.mp4")) is None


def test_ffprobe_falha_ou_json_invalido(monkeypatch):
    class Result:
        def __init__(self, returncode=0, stdout=""):
            self.returncode = returncode
            self.stdout = stdout

    monkeypatch.setattr("astroframe.meta.extractor.subprocess.run", lambda *a, **k: Result(1, ""))
    assert _ffprobe(Path("x.mp4")) is None

    monkeypatch.setattr("astroframe.meta.extractor.subprocess.run", lambda *a, **k: Result(0, "não é json"))
    assert _ffprobe(Path("x.mp4")) is None

    monkeypatch.setattr(
        "astroframe.meta.extractor.subprocess.run",
        lambda *a, **k: Result(0, json.dumps({"format": {"format_name": "mp4"}})),
    )
    data = _ffprobe(Path("x.mp4"))
    assert data is not None
    assert data["format"]["format_name"] == "mp4"


# ---------------------------------------------------------------------------
# suggest_config
# ---------------------------------------------------------------------------


def test_suggest_config_raios_por_resolucao():
    meta = MediaMetadata(width=1920, height=1080)
    cfg = suggest_config(meta)
    assert cfg.stabilizer.max_radius == int(1080 * 0.45)


def test_suggest_config_iso_aumenta_denoise_e_nitidez():
    meta = MediaMetadata(width=100, height=100, iso=3200)
    cfg = suggest_config(meta)
    assert cfg.denoise.h == pytest.approx(2.0 + 3200 / 1600 * 4.0)
    assert cfg.unsharp.amount == 0.6


def test_suggest_config_iso_baixo_mantem_nitidez_base():
    meta = MediaMetadata(width=100, height=100, iso=800)
    cfg = suggest_config(meta)
    assert cfg.unsharp.amount == 0.4


def test_suggest_config_bitrate_baixo_reduz_denoise():
    meta = MediaMetadata(width=1920, height=1080, fps=30, bitrate=3_000_000, iso=3200)
    cfg = suggest_config(meta)
    assert cfg.denoise.h == pytest.approx(round((2.0 + 2 * 4.0) * 0.7, 1))


def test_suggest_config_bitrate_alto_nao_mexe():
    meta = MediaMetadata(width=1920, height=1080, fps=30, bitrate=50_000_000, iso=3200)
    cfg = suggest_config(meta)
    assert cfg.denoise.h == pytest.approx(2.0 + 3200 / 1600 * 4.0)


def test_suggest_config_sem_metadados_usa_padroes():
    from astroframe.config import AstroFrameConfig

    cfg = suggest_config(MediaMetadata())
    assert isinstance(cfg, AstroFrameConfig)
    assert cfg.denoise.h == 5.0
    assert cfg.stabilizer.max_radius == 400


# ---------------------------------------------------------------------------
# summary_fields
# ---------------------------------------------------------------------------


def test_summary_fields_video_completo():
    meta = MediaMetadata(
        kind="video",
        width=1920,
        height=1080,
        fps=30,
        frame_count=900,
        duration=30.0,
        codec="h264",
        bitrate=8_000_000,
        format_name="mp4",
        captured_at="2026-08-12",
    )
    fields = summary_fields(meta)
    assert "1920x1080 · 16:9" in fields["Proporção (aspect ratio)"]
    assert fields["FPS"] == "30"
    assert fields["Frames"] == "900"
    assert fields["Duração"] == "0m 30s"
    assert fields["Codec"] == "H264"
    assert fields["Bitrate"] == "8.0 Mbps"
    assert fields["Formato"] == "mp4"


def test_summary_fields_imagem_com_exif():
    meta = MediaMetadata(
        kind="image",
        width=640,
        height=480,
        iso=1600,
        exposure_time=0.002,
        aperture=2.8,
        focal_length=50,
        camera_make="Canon",
        camera_model="EOS R",
    )
    fields = summary_fields(meta)
    assert fields["ISO"] == "1600"
    assert fields["Exposição"] == "1/500s"
    assert fields["Abertura"] == "f/2.8"
    assert fields["Distância focal"] == "50 mm"
    assert fields["Câmara"] == "Canon EOS R"


def test_summary_fields_vazias():
    assert summary_fields(MediaMetadata()) == {}


def test_formatadores():
    assert _format_duration(None) is None
    assert _format_duration(75) == "1m 15s"
    assert _format_duration(3661) == "1h 01m 01s"
    assert _format_bitrate(None) is None
    assert _format_bitrate(500_000) == "500 kbps"
    assert _format_bitrate(8_500_000) == "8.5 Mbps"
    assert _format_exposure(None) is None
    assert _format_exposure(2) == "2s"
    assert _format_exposure(0.002) == "1/500s"
