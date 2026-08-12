"""Interfaces de utilizador: Gradio (web) e linha de comando."""

from __future__ import annotations

from astroframe.ui.cli import main
from astroframe.ui.gradio_app import build_app, run

__all__ = ["build_app", "run", "main"]
