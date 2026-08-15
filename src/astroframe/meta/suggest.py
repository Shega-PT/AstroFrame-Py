"""Tradução de metadados em parâmetros otimizados da pipeline.

Heurísticas documentadas (não são "ciência exata", apenas pontos de partida
sensatos que depois podem ser ajustados pelo utilizador na interface):

- **Resolução -> teto de raio do estabilizador**: o disco ocupa tipicamente
  15–45% da dimensão mínima do frame; o teto do Hough é posto a essa escala
  (em vez do valor fixo por omissão). O raio mínimo e a distância entre
  centros são derivados automaticamente da resolução (sem valores em px).
- **ISO -> denoising/nitidez**: ISO alto implica ruído mais forte; `denoise.h`
  sobe linearmente com o ISO (2 a 15) e a nitidez aumenta ligeiramente.
- **Bitrate -> denoising**: vídeos com bitrate muito baixo (menos de ~0,1 bits
  por pixel-frame, ≈6 Mbps em 1080p@30) já estão fortemente comprimidos — o
  ruído é menor e `denoise.h` é reduzido para não embrulhar detalhes.
"""

from __future__ import annotations

from astroframe.config import AstroFrameConfig
from astroframe.meta.extractor import MediaMetadata, aspect_text

_SUGGESTED_ISO_BASE = 1600.0
_MAX_DENOISE = 15.0
_COMPRESSED_BPP_THRESHOLD = 0.1


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def suggest_config(meta: MediaMetadata) -> AstroFrameConfig:
    """Devolve uma cópia da configuração ajustada aos metadados."""
    cfg = AstroFrameConfig()

    if meta.width and meta.height:
        half = min(meta.width, meta.height)
        cfg.stabilizer.max_radius = int(_clamp(half * 0.45, 50, 2000))

    if meta.iso:
        cfg.denoise.h = round(_clamp(2.0 + meta.iso / _SUGGESTED_ISO_BASE * 4.0, 2.0, _MAX_DENOISE), 1)
        cfg.unsharp.amount = 0.4 if meta.iso < 1600 else 0.6

    if (
        meta.bitrate
        and meta.width
        and meta.height
        and meta.fps
        and meta.bitrate / (meta.width * meta.height * meta.fps) < _COMPRESSED_BPP_THRESHOLD
    ):
        cfg.denoise.h = round(cfg.denoise.h * 0.7, 1)

    return cfg


def _format_duration(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def _format_bitrate(bps: float | None) -> str | None:
    if bps is None:
        return None
    if bps >= 1_000_000:
        return f"{bps / 1_000_000:.1f} Mbps"
    return f"{bps / 1000:.0f} kbps"


def _format_exposure(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    if seconds >= 1:
        return f"{seconds:g}s"
    return f"1/{round(1 / seconds):d}s"


def summary_fields(meta: MediaMetadata) -> dict[str, str]:
    """Metadados legíveis para o painel de "proporção/qualidade" da interface."""
    fields: dict[str, str] = {}

    ratio = aspect_text(meta.width, meta.height)
    if meta.width and meta.height:
        fields["Proporção (aspect ratio)"] = f"{meta.width}x{meta.height}" + (f" · {ratio}" if ratio else "")

    if meta.kind in ("image", "video"):
        fields["Tipo de ficheiro"] = meta.kind
    if meta.format_name:
        fields["Formato"] = meta.format_name

    if meta.fps:
        fields["FPS"] = f"{meta.fps:g}"
    if meta.frame_count:
        fields["Frames"] = str(meta.frame_count)
    duration = _format_duration(meta.duration)
    if duration:
        fields["Duração"] = duration
    codec = meta.codec or ""
    if codec:
        fields["Codec"] = codec.upper()
    bitrate = _format_bitrate(meta.bitrate)
    if bitrate:
        fields["Bitrate"] = bitrate

    if meta.iso:
        fields["ISO"] = str(meta.iso)
    exposure = _format_exposure(meta.exposure_time)
    if exposure:
        fields["Exposição"] = exposure
    if meta.aperture:
        fields["Abertura"] = f"f/{meta.aperture:g}"
    if meta.focal_length:
        fields["Distância focal"] = f"{meta.focal_length:g} mm"
    camera = " ".join(part for part in (meta.camera_make, meta.camera_model) if part)
    if camera:
        fields["Câmara"] = camera
    if meta.captured_at:
        fields["Data/hora"] = meta.captured_at

    return fields
