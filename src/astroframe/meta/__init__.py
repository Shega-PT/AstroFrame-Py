"""Leitura de metadados de imagens e vídeos para otimização automática."""

from __future__ import annotations

from astroframe.meta.extractor import MediaMetadata, extract_metadata
from astroframe.meta.suggest import suggest_config, summary_fields

__all__ = ["MediaMetadata", "extract_metadata", "suggest_config", "summary_fields"]
