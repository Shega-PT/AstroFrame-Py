"""IA do AstroFrame.

- `rife` — interpolação de movimento opcional (requer PyTorch, extra
  `astroframe[rife]`); não é importada por defeito.
- `score` — avaliação automática da qualidade (0–5 estrelas) com métricas.
- `feedback` — banco local de aprendizagem (recompensa/punição por estrelas).
"""

from __future__ import annotations

from astroframe.ai.feedback import FeedbackDB, apply_learned, nudge_params, record_run
from astroframe.ai.score import StarRating, score_from_stars, score_image, stars_text

__all__ = [
    "FeedbackDB",
    "StarRating",
    "apply_learned",
    "nudge_params",
    "record_run",
    "score_from_stars",
    "score_image",
    "stars_text",
]
