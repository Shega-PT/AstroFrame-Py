"""Testes da UI desktop do enhancer_trainer (Tk real, janela oculta).

A suíte usa `DISPLAY` (Tk real com a janela retirada do ecrã) para exercitar
`EnhancerTkApp` — o loop de eventos é bombeado manualmente com
`root.update()` e o treino real é substituído por `monkeypatch` para ser
determinístico e rápido.
"""

from __future__ import annotations

import queue
import time
from pathlib import Path

import cv2
import enhancer_trainer as et
import numpy as np
import pytest

from astroframe.calibration.scan import SampleRef
from astroframe.calibration.store import CalibrationItem, CalibrationStore
from astroframe.config import AstroFrameConfig
from astroframe.core.stabilizer import DiskDetection
from tests.helpers import make_disk_image

CIRCLE = DiskDetection(300, 140, 90)


@pytest.fixture()
def root():
    import tkinter as tk

    tk_root = tk.Tk()
    tk_root.withdraw()
    yield tk_root
    try:
        tk_root.destroy()
    except tk.TclError:
        pass


def make_app(root, tmp_path: Path, n: int = 2) -> et.EnhancerTkApp:
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    store = CalibrationStore(samples_dir / "calibration.json")
    samples = []
    for i in range(n):
        name = f"sample_{i}.jpg"
        image, cx, cy = make_disk_image()
        cv2.imwrite(str(samples_dir / name), image)
        store.items[name] = CalibrationItem(name, "image", None, 480, 360, [CIRCLE])
        samples.append(SampleRef("image", samples_dir / name, None, name, name))
    store.save()
    state = et.EnhancerState(tmp_path / "enhancer_state.json")
    return et.EnhancerTkApp(samples, store, AstroFrameConfig(), state)


def test_app_carrega_primeira_amostra(root, tmp_path):
    app = make_app(root, tmp_path)
    assert len(app.samples) == 2
    assert "sample_0.jpg" in app.info.get()
    assert "pares: 0" in app.info.get()
    assert app.left.cget("image") != ""
    assert app.right.cget("image") != ""


def test_app_aceite_rejeitado_e_navegacao(root, tmp_path):
    app = make_app(root, tmp_path)
    app._accept()
    assert len(app.pairs) == 1
    assert app.state.pairs_positive == 1
    assert "pares: 1" in app.info.get()
    assert app.index == 1
    app._reject()
    assert len(app.pairs) == 2
    assert app.state.pairs_identity == 1
    app._prev()
    assert app.index == 1
    app._prev()
    assert app.index == 0
    x, y = app.pairs[0]
    assert x.shape == (360, 480)
    assert np.array_equal(x, y)  # sem campeão, a CNN não altera a imagem
    x2, y2 = app.pairs[1]
    assert np.array_equal(x2, y2)


def test_app_sem_amostras(root, tmp_path):
    app = et.EnhancerTkApp(
        [], CalibrationStore(tmp_path / "c.json"), AstroFrameConfig(), et.EnhancerState(tmp_path / "s.json")
    )
    assert "Sem amostras" in app.status.get()
    app._accept()
    assert "Sem imagem para julgar" in app.status.get()
    app._train_now()
    assert app.status.get() == "Sem imagem para julgar."  # sem amostras: sai cedo


def test_app_treinar_agora_usa_agente_fake(root, tmp_path, monkeypatch):
    app = make_app(root, tmp_path)
    app._accept()
    app._accept()

    calls: dict = {"n": 0}

    def fake_train_round(pairs, state, round_n, **kwargs):
        calls["n"] += 1
        calls["pairs"] = pairs
        return {"skipped": False, "mean_delta": 0.8, "promoted": True, "champion_path": None}

    monkeypatch.setattr(et, "train_enhancer_round", fake_train_round)
    app._train_now()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and app.status.get() == "A treinar a CNN residual…":
        root.update()
        if not app._train_results.empty():
            app._poll_train()
        time.sleep(0.01)
    root.update()
    assert calls["n"] == 1
    assert len(calls["pairs"]) == 2
    assert "PROMOVIDA" in app.status.get()


def test_app_treinar_agora_erro(root, tmp_path, monkeypatch):
    app = make_app(root, tmp_path)
    app._accept()
    app._accept()

    def failing(*args, **kwargs):
        raise RuntimeError("falha sintética")

    monkeypatch.setattr(et, "train_enhancer_round", failing)
    app._train_now()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and app.status.get() == "A treinar a CNN residual…":
        root.update()
        if not app._train_results.empty():
            app._poll_train()
        time.sleep(0.01)
    root.update()
    assert "Erro no treino" in app.status.get()


def test_app_erro_de_processamento_de_amostra(root, tmp_path):
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    (samples_dir / "quebrada.jpg").write_bytes(b"nao e imagem")
    store = CalibrationStore(samples_dir / "calibration.json")
    samples = [SampleRef("image", samples_dir / "quebrada.jpg", None, "q.jpg", "q.jpg")]
    app = et.EnhancerTkApp(samples, store, AstroFrameConfig(), et.EnhancerState(tmp_path / "s.json"))
    assert "erro ao processar" in app.status.get()


def test_run_gui_inicia_janela(root, tmp_path, monkeypatch):
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    image, cx, cy = make_disk_image()
    cv2.imwrite(str(samples_dir / "a.jpg"), image)
    store = CalibrationStore(samples_dir / "calibration.json")
    store.items["a.jpg"] = CalibrationItem("a.jpg", "image", None, 480, 360, [CIRCLE])
    store.save()
    created: list = []

    def fake_app(samples, store, config, state):
        created.append(samples)

    monkeypatch.setattr(et, "EnhancerTkApp", fake_app)
    assert et.run_gui(str(samples_dir), state_path=str(tmp_path / "s.json")) == 0
    assert created and len(created[0]) == 1


def test_app_show_sem_imagem_precomputada(root, tmp_path):
    app = make_app(root, tmp_path)
    before = app.left.cget("image")
    app._precomputed = []
    app._show()
    assert app.left.cget("image") == before  # sem imagem: não altera nada


def test_app_treinar_com_um_par_so(root, tmp_path):
    app = make_app(root, tmp_path)
    app._accept()
    app._train_now()
    assert "Precisas de pelo menos 2 pares" in app.status.get()


def test_app_treinar_resultado_skipped(root, tmp_path, monkeypatch):
    app = make_app(root, tmp_path)
    app._accept()
    app._accept()

    def fake_train_round(pairs, state, round_n, **kwargs):
        return {"skipped": True}

    monkeypatch.setattr(et, "train_enhancer_round", fake_train_round)
    app._train_now()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and app.status.get() == "A treinar a CNN residual…":
        root.update()
        if not app._train_results.empty():
            app._poll_train()
        time.sleep(0.01)
    root.update()
    assert "sem pares suficientes" in app.status.get()


def test_app_poll_train_diretamente(root, tmp_path):
    app = make_app(root, tmp_path)
    app._train_results = queue.Queue()
    app._train_results.put("mensagem direta")
    app._poll_train()
    assert app.status.get() == "mensagem direta"
    app._poll_train()  # fila vazia → reagenda (sem crash)


def test_main_module_guard(tmp_path, monkeypatch):
    import runpy

    samples = tmp_path / "samples"
    samples.mkdir()
    monkeypatch.setattr(et.sys, "argv", ["enhancer_trainer", "--samples", str(samples)])
    with pytest.raises(SystemExit):
        runpy.run_path(str(et.__file__), run_name="__main__")


def test_app_treinar_agora_atualiza_campeao(root, tmp_path, monkeypatch):
    app = make_app(root, tmp_path)
    app._accept()
    app._accept()

    def fake_train_round(pairs, state, round_n, **kwargs):
        return {
            "skipped": False,
            "mean_delta": 0.8,
            "promoted": True,
            "champion_path": str(tmp_path / "campeao.npz"),
        }

    monkeypatch.setattr(et, "train_enhancer_round", fake_train_round)
    app._train_now()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and app.status.get() == "A treinar a CNN residual…":
        root.update()
        if not app._train_results.empty():
            app._poll_train()
        time.sleep(0.01)
    root.update()
    assert app._champion_path == str(tmp_path / "campeao.npz")
    assert "PROMOVIDA" in app.status.get()
