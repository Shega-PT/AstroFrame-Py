"""Fixtures partilhadas.

Nota: o GC cíclico é desligado durante a sessão — no Python 3.12, o GC a
correr numa thread de trabalho durante o bootstrap (com cobertura ativa)
aborta o processo (`Fatal Python error: Aborted`); a recolha é feita de forma
segura na main thread, após cada teste.
"""

from __future__ import annotations

import gc

import pytest

from tests.helpers import make_disk_image, make_noisy_image


@pytest.fixture(scope="session", autouse=True)
def _gc_desligado_durante_a_sessao():
    gc.disable()
    yield
    gc.enable()


@pytest.fixture(autouse=True)
def _recolha_de_lixo_segura():
    yield
    gc.collect()


@pytest.fixture
def disk_image():
    return make_disk_image()


@pytest.fixture
def noisy_image():
    return make_noisy_image()


@pytest.fixture(autouse=True)
def _ai_isolado(tmp_path, monkeypatch):
    """Isola todos os artefactos de IA (banco, modelos canónicos e staging).

    Sem isto, `FeedbackDB()`, `DiskFilter()` e o treino entre séries
    escreveriam nos `Logs/` reais do projeto durante os testes. O
    `ASTROFRAME_DATA_DIR` redireciona toda a estrutura `Logs/` (logs, train,
    weights) para a pasta temporária; o banco e os caminhos canónicos são
    apontados explicitamente para não dependerem de módulos já importados.
    """
    import enhancer_trainer
    import validator

    import astroframe.ai.cnn as cnn

    monkeypatch.setenv("ASTROFRAME_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ASTROFRAME_FEEDBACK_DB", str(tmp_path / "feedback.db"))
    monkeypatch.setattr(cnn, "_FILTER_MODEL", tmp_path / "disk_filter.npz")
    monkeypatch.setattr(cnn, "_ENHANCER_MODEL", tmp_path / "enhancer_cnn.npz")
    monkeypatch.setattr(validator, "CNN_MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(validator, "CNN_CANONICAL_PATH", tmp_path / "disk_filter.npz")
    monkeypatch.setattr(enhancer_trainer, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(enhancer_trainer, "ENHANCER_CANONICAL_PATH", tmp_path / "enhancer_cnn.npz")
    yield
