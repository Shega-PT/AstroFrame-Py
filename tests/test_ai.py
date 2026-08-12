"""Testes do wrapper RIFE opcional, sem instalar PyTorch.

Um módulo `torch` falso permite exercitar o código de interpolação (`_infer`)
de forma determinística; `import torch` durante os testes devolve o falso.
"""

from __future__ import annotations

import builtins
import contextlib
import types

import numpy as np
import pytest

from astroframe.ai.rife import RifeInterpolator


def _bare_interpolator() -> RifeInterpolator:
    """Instância sem `__init__` (que exige torch); útil para os ramos sem PyTorch."""
    return object.__new__(RifeInterpolator)


def test_available_false_sem_torch():
    assert RifeInterpolator.available() is False


def test_interpolate_sem_torch_levanta():
    frame = np.zeros((2, 2, 3), np.uint8)
    with pytest.raises(RuntimeError, match="astroframe\\[rife\\]"):
        _bare_interpolator().interpolate(frame, frame)


@pytest.fixture
def rife_com_torch_disponivel(monkeypatch):
    """Força `available()` a True para testar a validação sem instalar PyTorch."""
    monkeypatch.setattr(
        "astroframe.ai.rife.RifeInterpolator.available",
        staticmethod(lambda: True),
    )
    return _bare_interpolator()


def test_interpolate_n_interp_negativo_levanta(rife_com_torch_disponivel):
    frame = np.zeros((2, 2, 3), np.uint8)
    with pytest.raises(ValueError, match=">= 0"):
        rife_com_torch_disponivel.interpolate(frame, frame, -1)


def test_interpolate_n_interp_zero_devolve_vazio(rife_com_torch_disponivel):
    frame = np.zeros((2, 2, 3), np.uint8)
    assert rife_com_torch_disponivel.interpolate(frame, frame, 0) == []


# ---------------------------------------------------------------------------
# PyTorch falso para cobrir __init__ e _infer
# ---------------------------------------------------------------------------


class _FakeTensor:
    def __init__(self, value: np.ndarray | None = None):
        self._value = value

    def __add__(self, other):
        return self

    def __truediv__(self, other):
        return self

    def float(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def unsqueeze(self, *args):
        return self

    def squeeze(self, *args):
        if self._value is not None:
            self._value = np.squeeze(self._value) if not args else np.squeeze(self._value, args)
        return self

    def permute(self, *args):
        if self._value is not None:
            self._value = self._value.transpose(args)
        return self

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return self._value

    def __getitem__(self, item):
        return self


class _FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, inputs, timestep, *args):
        height, width = inputs.numpy().shape[1:]
        return _FakeTensor(np.zeros((1, 3, height, width), dtype=np.float32))


@pytest.fixture
def torch_falso(monkeypatch):
    """Substitui `import torch` por um módulo falso funcional para os testes."""
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.hub = types.SimpleNamespace(load=lambda *a, **k: _FakeModel())
    torch.from_numpy = lambda array: _FakeTensor(array)
    torch.tensor = lambda *a, **k: _FakeTensor()
    torch.cat = lambda tensors, *a, **k: _FakeTensor(tensors[0].numpy())
    torch.no_grad = contextlib.nullcontext
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            return torch
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    return torch


def test_available_true_com_torch_instalado(torch_falso):
    assert RifeInterpolator.available() is True


def test_interpolate_completo_com_torch_fingido(torch_falso):
    interp = RifeInterpolator("repo")
    frames = np.full((4, 5, 3), 100, dtype=np.uint8)
    out = interp.interpolate(frames, frames, 1)
    assert len(out) == 1
    assert out[0].dtype == np.uint8
    assert out[0].shape == (4, 5, 3)


def test_interpolate_varios_frames_intermedios(torch_falso):
    interp = RifeInterpolator("repo")
    frames = np.zeros((4, 5, 3), dtype=np.uint8)
    assert len(interp.interpolate(frames, frames, 3)) == 3
