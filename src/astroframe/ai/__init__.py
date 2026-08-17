"""IA do AstroFrame.

- `params` — registry unificado dos parâmetros ajustáveis (limites seguros,
  passos e deltas de treino de toda a pipeline).
- `tuner` — auto-tuning determinístico e limitado de todos os parâmetros
  (hill-climbing com proxy de avaliação rápida e persistência por perfil).
- `lstm` — pequena LSTM (NumPy/torch) para antecipar a trajetória do
  centroide em frames sem deteção (anti-trepidação temporal).
- `cnn` — pequena CNN (NumPy/torch) com duas cabeças: residual (melhoria
  aprendida de imagens) e classificadora (filtro de falsos positivos).
- `rife` — interpolação de movimento opcional (requer PyTorch, extra
  `astroframe[rife]`); não é importada por defeito.
- `score` — avaliação automática da qualidade (0–5 estrelas) com métricas.
- `feedback` — banco local de aprendizagem (recompensa/punição por estrelas).
"""

from __future__ import annotations

from astroframe.ai.cnn import DiskFilter, ResidualEnhancer, SmallCNN, fit_classifier, fit_residual
from astroframe.ai.feedback import FeedbackDB, apply_learned, nudge_params, record_run
from astroframe.ai.lstm import LSTMTuner, TrajectoryPredictor, train_trajectory_model
from astroframe.ai.score import StarRating, score_from_stars, score_image, stars_text
from astroframe.ai.tuner import BoundedHillClimb, ProxyEval, run_autotune

__all__ = [
    "BoundedHillClimb",
    "DiskFilter",
    "FeedbackDB",
    "LSTMTuner",
    "ProxyEval",
    "ResidualEnhancer",
    "SmallCNN",
    "StarRating",
    "TrajectoryPredictor",
    "apply_learned",
    "fit_classifier",
    "fit_residual",
    "nudge_params",
    "record_run",
    "run_autotune",
    "score_from_stars",
    "score_image",
    "stars_text",
    "train_trajectory_model",
]