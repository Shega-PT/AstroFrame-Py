"""Testes da pequena LSTM (célula, tuner de deltas e trajetória)."""

from __future__ import annotations

import numpy as np
import pytest

from astroframe.ai.lstm import (
    FitHistory,
    LSTMCell,
    LSTMTuner,
    TrajectoryPredictor,
    _normalize_delta,
    make_sequences,
    run_features,
    target_deltas,
    torch_available,
    train_trajectory_model,
)
from astroframe.ai.params import FEEDBACK_PARAMS

# ------------------------------------------------------------------ célula --


def test_lstm_cell_forward_formas():
    rng = np.random.default_rng(0)
    cell = LSTMCell(3, 4, rng)
    seq = rng.normal(0, 1, (5, 3))
    h, cache = cell.forward(seq)
    assert h.shape == (4,)
    assert len(cache) == 5
    assert cell.forward_full(seq).shape == (5, 4)


def test_lstm_cell_h0_fornecido():
    cell = LSTMCell(3, 4)
    seq = np.zeros((2, 3))
    h, _ = cell.forward(seq, np.ones(4))
    assert h.shape == (4,)


def test_lstm_cell_backward_exata_por_diferencas_finitas():
    rng = np.random.default_rng(3)
    cell = LSTMCell(2, 3, rng)
    seq = rng.normal(0, 1, (4, 2))
    h, cache = cell.forward(seq)
    dh = rng.normal(0, 1, (3,))
    grads = cell.backward(seq, cache, dh)

    def loss():
        h2, _ = cell.forward(seq)
        return float(np.sum(h2 * dh))

    eps = 1e-6
    for name, P in (("W", cell.W), ("U", cell.U), ("b", cell.b)):
        analytic = grads[name]
        numeric = np.zeros_like(P)
        for i in range(P.size):
            orig = P.ravel()[i]
            P.ravel()[i] = orig + eps
            l1 = loss()
            P.ravel()[i] = orig - eps
            l2 = loss()
            P.ravel()[i] = orig
            numeric.ravel()[i] = (l1 - l2) / (2 * eps)
        assert np.max(np.abs(numeric - analytic)) < 1e-6, name


def test_lstm_cell_backward_sem_dh_devolve_gradientes():
    cell = LSTMCell(2, 3)
    seq = np.zeros((2, 2))
    h, cache = cell.forward(seq)
    grads = cell.backward(seq, cache)
    assert grads["W"].shape == cell.W.shape


def test_lstm_cell_roundtrip(tmp_path):
    rng = np.random.default_rng(5)
    cell = LSTMCell(3, 4, rng)
    path = cell.save(tmp_path / "cell.npz")
    loaded = LSTMCell.load(path)
    assert loaded is not None
    assert loaded.n_in == 3 and loaded.n_hidden == 4
    seq = rng.normal(0, 1, (3, 3))
    assert np.allclose(cell.forward(seq)[0], loaded.forward(seq)[0])


def test_lstm_cell_load_inexistente_e_corrompido(tmp_path):
    assert LSTMCell.load(tmp_path / "nao_existe.npz") is None
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"isto nao e um npz")
    assert LSTMCell.load(bad) is None


def test_torch_available_falso_sem_pytorch():
    assert torch_available() is False


# -------------------------------------------------- características/alvos --


class _FakeRun:
    def __init__(self, stars=4.0, metrics=None, nudge=None):
        self.stars_calc = stars
        self.stars_user = None
        self.metrics = metrics or {}
        self.nudge = nudge or {}


def test_run_features_nove_dimensoes():
    run = _FakeRun(
        stars=4.0,
        metrics={"background": 1.0, "limb": 0.8, "noise": 0.6, "contrast": 0.9, "reflections": 0.7},
        nudge={"denoise.h": 2.0, "clahe.clip_limit": 0.4},
    )
    features = run_features(run)
    assert features.shape == (9,)
    assert features[0] == pytest.approx(0.8)
    assert features[1] == pytest.approx(0.8)
    assert features[2] == pytest.approx(1.0)
    assert np.all(features >= 0.0) and np.all(features <= 1.0)


def test_run_features_sem_metricas():
    run = _FakeRun(stars=2.5)
    features = run_features(run)
    assert features[2:7].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_normalize_delta_limitado():
    assert _normalize_delta("denoise.h", 0.0) == 0.0
    assert _normalize_delta("denoise.h", 100.0) == 1.0
    assert _normalize_delta("denoise.h", -100.0) == -1.0


def test_target_deltas_alinhado_com_feedback_params():
    run = _FakeRun(nudge={"denoise.h": 4.0, "unsharp.amount": 0.1})
    target = target_deltas(run)
    assert target.shape == (len(FEEDBACK_PARAMS),)
    assert target[FEEDBACK_PARAMS.index("denoise.h")] == pytest.approx(1.0)
    assert target[FEEDBACK_PARAMS.index("unsharp.amount")] == pytest.approx(0.1 / 0.4)


def test_make_sequences_janelas():
    history = [_FakeRun(stars=float(i)) for i in range(5)]
    X, y = make_sequences(history, seq_len=3)
    assert X.shape == (4, 3, 9)
    assert y.shape == (4, len(FEEDBACK_PARAMS))
    assert np.allclose(X[0][-1], run_features(history[0]))
    assert np.allclose(X[1][-1], run_features(history[1]))
    seq3 = np.stack([run_features(history[1]), run_features(history[2]), run_features(history[3])])
    assert np.allclose(X[3], seq3)


def test_make_sequences_insuficiente_vazio():
    X, y = make_sequences([_FakeRun()], seq_len=3)
    assert X.shape == (0, 3, 9)
    assert y.shape == (0, len(FEEDBACK_PARAMS))


# ------------------------------------------------------------- LSTMTuner --


def test_lstm_tuner_fit_e_previsao(tmp_path):
    history = [
        _FakeRun(stars=3.0, metrics={"noise": 0.4}, nudge={"denoise.h": 1.0}),
        _FakeRun(stars=3.5, metrics={"noise": 0.5}, nudge={"denoise.h": 2.0}),
        _FakeRun(stars=4.0, metrics={"noise": 0.7}, nudge={"denoise.h": 3.0}),
        _FakeRun(stars=4.5, metrics={"noise": 0.8}, nudge={"denoise.h": 4.0}),
        _FakeRun(stars=5.0, metrics={"noise": 0.9}, nudge={"denoise.h": 4.0}),
    ]
    tuner = LSTMTuner(n_hidden=8, seed=1)
    fit = tuner.fit(history, epochs=30, lr=0.1)
    assert isinstance(fit, FitHistory)
    assert fit.epochs >= 1
    assert fit.best_loss != float("inf")
    prediction = tuner.predict_next_delta(history)
    assert set(prediction) <= set(FEEDBACK_PARAMS)
    assert prediction.get("denoise.h", 0.0) > 0.0


def test_lstm_tuner_sem_historico():
    tuner = LSTMTuner(n_hidden=8)
    fit = tuner.fit([], epochs=10)
    assert fit.epochs == 0
    assert tuner.predict_next_delta([]) == {}
    assert tuner.predict_next_delta([_FakeRun()]) == {}


def test_lstm_tuner_roundtrip(tmp_path):
    tuner = LSTMTuner(n_hidden=8, seed=2)
    path = tuner.save(tmp_path / "lstm.npz")
    loaded = LSTMTuner.load(path)
    assert loaded is not None
    assert loaded.n_hidden == 8
    history = [_FakeRun(stars=float(i)) for i in range(4)]
    assert loaded.predict_next_delta(history) == tuner.predict_next_delta(history)


def test_lstm_tuner_load_inexistente(tmp_path):
    assert LSTMTuner.load(tmp_path / "nao_existe.npz") is None


# ------------------------------------------------------------ trajetória --


def test_trajectory_linear_prevê_proxima_posicao():
    pred = TrajectoryPredictor()
    for i in range(5):
        pred.push(100.0 + 20.0 * i, 50.0 + 10.0 * i)
    assert pred.predict() == pytest.approx((200.0, 100.0), abs=1e-6)
    assert len(pred) == 5


def test_trajectory_sem_historico_devolve_none():
    pred = TrajectoryPredictor()
    assert pred.predict() is None
    pred.push(1.0, 1.0)
    assert pred.predict() is None


def test_trajectory_clear():
    pred = TrajectoryPredictor()
    pred.push(1.0, 1.0)
    pred.push(2.0, 2.0)
    pred.clear()
    assert len(pred) == 0
    assert pred.predict() is None


def test_trajectory_maxlen_respeitado():
    pred = TrajectoryPredictor(maxlen=3)
    for i in range(10):
        pred.push(float(i), 0.0)
    assert len(pred) == 3


def test_trajectory_lstm_sem_modelo_faz_linear(tmp_path):
    pred = TrajectoryPredictor(use_lstm=True, model_path=tmp_path / "nao_existe.npz")
    for i in range(4):
        pred.push(float(i), float(i))
    assert pred.predict() == pytest.approx((4.0, 4.0), abs=1e-6)


def test_train_trajectory_model_guarda_e_prediz(tmp_path):
    model_path = tmp_path / "traj.npz"
    trajectories = [
        [(float(100 + 10 * t), float(50 + 5 * t)) for t in range(10)] for _ in range(4)
    ]
    path = train_trajectory_model(trajectories, path=model_path, seed=3, epochs=20)
    assert path == model_path
    assert model_path.exists()
    pred = TrajectoryPredictor(use_lstm=True, model_path=model_path)
    for i in range(6):
        pred.push(100.0 + 10.0 * i, 50.0 + 5.0 * i)
    x, y = pred.predict()
    assert abs(x - 160.0) < 12.0
    assert abs(y - 80.0) < 12.0


def test_train_trajectory_model_sem_trajetorias_levanta(tmp_path):
    with pytest.raises(ValueError, match="Sem trajetórias"):
        train_trajectory_model([[ (1.0, 1.0) ]], path=tmp_path / "t.npz")


def test_trajectory_model_path_por_omissao():
    from astroframe.ai.lstm import trajectory_model_path

    assert trajectory_model_path().name == "lstm.npz"
    assert trajectory_model_path("/tmp/x.npz") == pytest.importorskip("pathlib").Path("/tmp/x.npz")