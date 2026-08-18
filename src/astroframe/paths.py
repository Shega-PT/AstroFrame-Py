"""Caminhos centrais dos dados do AstroFrame.

Tudo o que o sistema produz fica dentro de `Logs/` na raiz do projeto
(sujeita ao `ASTROFRAME_DATA_DIR` para ambientes isolados, ex.: testes):

- `Logs/logs/ia/`     — relatórios de treino das redes neuronais;
- `Logs/logs/system/` — logs do sistema (ficheiro) e base de aprendizagem
  (SQLite);
- `Logs/train/`       — JSONs de calibração, validação e treino;
- `Logs/weights/`     — pesos finais de uso (modelos canónicos) e staging
  (`staging/`) com os candidatos por ronda.

`migrate_legacy()` copia uma única vez os artefactos da localização antiga
(`~/.astroframe/`) para a estrutura nova.
"""

from __future__ import annotations

import logging
import os
import shutil
from logging.handlers import RotatingFileHandler
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_ROOT = "ASTROFRAME_DATA_DIR"
_LEGACY_DIR = Path("~/.astroframe").expanduser()


def data_root() -> Path:
    """Raiz dos dados (`ASTROFRAME_DATA_DIR` ou `Logs/` do projeto)."""
    raw = os.environ.get(_ENV_ROOT)
    return Path(raw).expanduser() if raw else _REPO_ROOT / "Logs"


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def logs_ia_dir() -> Path:
    """Relatórios de treino das redes neuronais (`Logs/logs/ia/`)."""
    return _ensure(data_root() / "logs" / "ia")


def logs_system_dir() -> Path:
    """Logs do sistema e base de aprendizagem (`Logs/logs/system/`)."""
    return _ensure(data_root() / "logs" / "system")


def train_dir() -> Path:
    """JSONs de calibração, validação e treino (`Logs/train/`)."""
    return _ensure(data_root() / "train")


def weights_dir() -> Path:
    """Pesos finais de uso (`Logs/weights/`)."""
    return _ensure(data_root() / "weights")


def staging_dir() -> Path:
    """Candidatos por ronda, antes de promoção (`Logs/weights/staging/`)."""
    return _ensure(weights_dir() / "staging")


def feedback_db_path() -> Path:
    """Base de aprendizagem (SQLite) do sistema."""
    return logs_system_dir() / "feedback.db"


def calibration_json(samples_dir: Path | str | None = None) -> Path:
    """Ground truth de calibração (`Logs/train/calibration.json`).

    Com `samples_dir` e sem ficheiro global, cai para
    `<samples_dir>/calibration.json` — compatível com projetos antigos.
    """
    target = train_dir() / "calibration.json"
    if samples_dir is not None and not target.exists():
        local = Path(samples_dir) / "calibration.json"
        if local.exists():
            return local
    return target


def migrate_legacy() -> None:
    """Cópia única dos artefactos antigos para a estrutura `Logs/`.

    Copia os modelos e a base de aprendizagem de `~/.astroframe/` (pré-v0.8)
    quando `Logs/weights/` ainda não tem pesos, e o `samples/calibration.json`
    para `Logs/train/` quando este ainda não existe. A origem nunca é apagada.
    """
    if _LEGACY_DIR.is_dir() and not any(weights_dir().glob("*.npz")):
        for src in sorted(_LEGACY_DIR.glob("*.npz")):
            shutil.copy2(src, weights_dir() / src.name)
        legacy_db = _LEGACY_DIR / "feedback.db"
        if legacy_db.exists():
            shutil.copy2(legacy_db, feedback_db_path())
    legacy_calibration = _REPO_ROOT / "samples" / "calibration.json"
    target = train_dir() / "calibration.json"
    if legacy_calibration.exists() and not target.exists():
        shutil.copy2(legacy_calibration, target)


def setup_logging(log_filename: str) -> None:
    """Liga o log de ficheiro em `Logs/logs/system/<log_filename>`.

    Mantém o console ativo; o ficheiro roda quando atinge 1 MiB (3 cópias
    antigas). Chamada nos pontos de entrada (CLI, validator, enhancer...).
    """
    handler = RotatingFileHandler(
        logs_system_dir() / log_filename,
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)