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
