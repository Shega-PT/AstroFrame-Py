"""Permite executar via `python -m astroframe`."""

from __future__ import annotations

import sys

from astroframe.ui.cli import main

if __name__ == "__main__":
    sys.exit(main())
