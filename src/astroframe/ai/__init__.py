"""IA opcional (interpolação de movimento RIFE).

Este módulo não é importado por defeito em lado nenhum da pipeline e
requer PyTorch (extra opcional `astroframe[rife]`).
"""

from __future__ import annotations

from astroframe.ai.rife import RifeInterpolator

__all__ = ["RifeInterpolator"]
