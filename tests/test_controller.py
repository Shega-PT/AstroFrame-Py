"""Testes do controlador sempre ligado (background thread + FallbackNet)."""

from __future__ import annotations

import threading
import time

import pytest

from astroframe.ai.controller import Controller, ControllerState, FallbackNet
from astroframe.ai.feedback import FeedbackDB, record_run
from astroframe.ai.lstm import LSTMTuner
from astroframe.ai.score import score_from_stars
from astroframe.config import AstroFrameConfig


@pytest.fixture
def db(tmp_path):
    return FeedbackDB(tmp_path / "fb.db")


def _rating(stars: float, metrics: dict | None = None) -> object:
    rating = score_from_stars(stars)
    rating.metrics.update(metrics or {})
    return rating


def _populate_db(db: FeedbackDB, n: int = 5, stars: float = 4.0) -> None:
    cfg = AstroFrameConfig()
    origin = {"clahe.clip_limit": 1.0, "denoise.h": 5.0, "unsharp.amount": 0.8, "polish.corona_scale": 1.6}
    for i in range(n):
        m = {"background": 0.9, "limb": 0.85, "noise": 0.8, "contrast": 0.9, "reflections": 0.9}
        record_run(db, "image", "test_profile", cfg, origin, _rating(stars, m), source=f"img_{i}.jpg")


# ------------------------------------------------------------------- FallbackNet --


def test_fallback_net_sempre_disponivel(db):
    net = FallbackNet(db, "test_profile")
    assert net.available is True


def test_fallback_net_sem_runs_devolve_config_identica(db):
    cfg = AstroFrameConfig()
    net = FallbackNet(db, "test_profile")
    result = net.predict(cfg)
    # Sem runs → apply_learned devolve a mesma config
    assert result is cfg


def test_fallback_net_aplica_nudges(db):
    _populate_db(db, n=3, stars=2.0)  # estrelas baixas → punição
    cfg = AstroFrameConfig()
    net = FallbackNet(db, "test_profile")
    result = net.predict(cfg)
    # Com nudges gravados, a config deve ter mudado
    assert result is not cfg
    # denoise.h deve ter aumentado (punição por noise/limb baixo)
    assert result.denoise.h >= cfg.denoise.h


# ------------------------------------------------------------------- Controller --


def test_controller_tuner_disponivel():
    cfg = AstroFrameConfig()
    tuner = LSTMTuner(n_hidden=4, seed=0)
    # LSTMTuner.cell é None até fit() — simular modelo carregado
    from astroframe.ai.lstm import LSTMCell

    tuner.cell = LSTMCell(9, 4, rng=tuner._rng)
    ctrl = Controller(cfg, "test_profile", tuner=tuner)
    assert ctrl.tuner_available is True


def test_controller_tuner_indisponivel(db):
    cfg = AstroFrameConfig()
    ctrl = Controller(cfg, "test_profile", db=db, tuner=None)
    assert ctrl.tuner_available is False


def test_controller_apply_now_sem_runs(db):
    cfg = AstroFrameConfig()
    ctrl = Controller(cfg, "test_profile", db=db)
    deltas = ctrl.apply_now()
    assert deltas == {}


def test_controller_apply_now_com_runs_fallback(db):
    _populate_db(db, n=5, stars=2.0)
    cfg = AstroFrameConfig()
    original_h = cfg.denoise.h
    ctrl = Controller(cfg, "test_profile", db=db, tuner=None)
    deltas = ctrl.apply_now()
    # FallbackNet aplica nudges → deltas não vazios
    assert len(deltas) > 0
    assert ctrl.state.lstm_used is False
    assert ctrl.state.last_apply_ts > 0


def test_controller_apply_now_com_runs_lstm(tmp_path):
    db = FeedbackDB(tmp_path / "fb.db")
    _populate_db(db, n=10, stars=4.0)
    cfg = AstroFrameConfig()
    tuner = LSTMTuner(n_hidden=4, seed=0)
    # Treinar o tuner com o histórico
    history = db.history_all(limit=32)
    if len(history) >= 2:
        tuner.fit(history, epochs=4, seq_len=4)
    ctrl = Controller(cfg, "test_profile", db=db, tuner=tuner)
    if ctrl.tuner_available:
        deltas = ctrl.apply_now()
        # LSTM pode devolver deltas ou vazio se previsão < threshold
        assert isinstance(deltas, dict)
        assert ctrl.state.lstm_used is True or len(deltas) == 0


def test_controller_start_stop():
    cfg = AstroFrameConfig()
    ctrl = Controller(cfg, "test_profile", interval=0.1)
    ctrl.start()
    assert ctrl.running is True
    time.sleep(0.3)  # permitir 2-3 ticks
    ctrl.stop()
    assert ctrl.running is False
    assert ctrl.state.ticks >= 2


def test_controller_start_idempotente():
    cfg = AstroFrameConfig()
    ctrl = Controller(cfg, "test_profile", interval=0.1)
    ctrl.start()
    ctrl.start()  # segunda vez não deve criar nova thread
    assert ctrl.running is True
    ctrl.stop()


def test_controller_stop_sem_start():
    cfg = AstroFrameConfig()
    ctrl = Controller(cfg, "test_profile")
    ctrl.stop()  # não deve levantar exceção
    assert ctrl.running is False


def test_controller_thread_e_daemon():
    cfg = AstroFrameConfig()
    ctrl = Controller(cfg, "test_profile", interval=0.1)
    ctrl.start()
    assert ctrl._thread is not None
    assert ctrl._thread.daemon is True
    ctrl.stop()


def test_controller_ticks_conta(db):
    _populate_db(db, n=3, stars=3.0)
    cfg = AstroFrameConfig()
    ctrl = Controller(cfg, "test_profile", db=db, interval=0.05)
    ctrl.start()
    time.sleep(0.3)
    ctrl.stop()
    assert ctrl.state.ticks >= 3


def test_controller_fallback_usado_quando_lstm_indisponivel(db):
    _populate_db(db, n=3, stars=2.0)
    cfg = AstroFrameConfig()
    original_h = cfg.denoise.h
    ctrl = Controller(cfg, "test_profile", db=db, tuner=None, interval=0.1)
    ctrl.start()
    time.sleep(0.3)
    ctrl.stop()
    assert ctrl.state.lstm_used is False
    assert cfg.denoise.h != original_h or ctrl.state.ticks > 0
