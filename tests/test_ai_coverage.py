"""Cobertura de ramos dos módulos de IA e integrações (gaps do registo)."""

from __future__ import annotations

import builtins
import types

import cv2
import numpy as np
import pytest

from astroframe.ai.cnn import DiskFilter, SmallCNN, fit_classifier, fit_residual
from astroframe.ai.feedback import FeedbackDB, apply_learned, record_run
from astroframe.ai.lstm import LSTMCell, LSTMTuner, TrajectoryPredictor, torch_available
from astroframe.ai.score import score_from_stars
from astroframe.ai.tuner import ProxyEval, TuneReport, _lstm_seed
from astroframe.calibration.store import CalibrationItem, CalibrationStore
from astroframe.config import AstroFrameConfig
from astroframe.core.enhancer import enhance_image
from astroframe.core.stabilizer import AntiJitterStabilizer, DiskDetection, find_all_disks


def _disk_frame(size: int = 300, dx: int = 0, dy: int = 0, radius: int = 60) -> np.ndarray:
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(frame, (size // 2 + dx, size // 2 + dy), radius, (220,) * 3, -1)
    return frame


def _make_samples(root, calibration: bool = True, broken: bool = False):
    root = root / "samples"
    root.mkdir(parents=True, exist_ok=True)
    store = CalibrationStore(root / "calibration.json")
    for i in range(2):
        cv2.imwrite(str(root / f"a{i}.jpg"), _disk_frame(size=200, radius=50))
        store.items[f"a{i}.jpg"] = CalibrationItem(
            f"a{i}.jpg", "image", None, 200, 200, [DiskDetection(100, 100, 50)]
        )
    if broken:
        (root / "b0.jpg").write_bytes(b"nao e uma imagem")
    if calibration:
        store.save()
    return root


def _tiny_classifier(tmp_path) -> SmallCNN:
    rng = np.random.default_rng(0)
    pos, neg = [], []
    for _ in range(4):
        p = np.zeros((48, 48))
        p[20:28, 20:28] = 1.0
        pos.append(p)
        neg.append(np.clip(rng.normal(0.5, 0.3, (48, 48)), 0, 1))
    model, _ = fit_classifier(pos, neg, epochs=2, seed=1)
    return model


# ------------------------------------------------------------------- cnn --


def test_smallcnn_load_versao_desconhecida(tmp_path):
    model = SmallCNN(mode="residual")
    path = model.save(tmp_path / "m.npz")
    data = np.load(path)
    np.savez(
        path,
        schema_version=99,
        mode="residual",
        k=8,
        n_in=1,
        conv1_w=data["conv1_w"],
        conv1_b=data["conv1_b"],
        conv2_w=data["conv2_w"],
        conv2_b=data["conv2_b"],
        conv3_w=data["conv3_w"],
        conv3_b=data["conv3_b"],
    )
    assert SmallCNN.load(path) is None


def test_fit_residual_com_uma_amostra_usar_val_como_treino():
    np.random.default_rng(0)
    img = np.zeros((48, 48))
    img[10:38, 10:38] = 0.5
    model, report = fit_residual([(img, img)], epochs=2, seed=2)
    assert report.best_loss != float("inf")


def test_disk_filter_patch_crop_vazio(tmp_path):
    filtro = DiskFilter(model=None, model_path=tmp_path / "ausente.npz")
    assert filtro.patch(np.zeros((0, 0)), 0, 0, 4).shape == (48, 48)


def test_disk_filter_com_modelo_filtra_sem_esvaziar(tmp_path, monkeypatch):
    import astroframe.ai.cnn as cnn

    model = _tiny_classifier(tmp_path)
    model_path = tmp_path / "filter.npz"
    model.save(model_path)
    monkeypatch.setattr(cnn, "_FILTER_MODEL", model_path)
    cfg = AstroFrameConfig()
    cfg.ai.disk_filter = 0.99
    frame = _disk_frame()
    disks = find_all_disks(frame, cfg)
    assert len(disks) >= 1


def test_residual_enhancer_sem_fallback_carregado_do_disco(tmp_path, monkeypatch):
    import astroframe.ai.cnn as cnn
    import astroframe.core.enhancer as enhancer

    rng = np.random.default_rng(0)
    clean = np.zeros((64, 64))
    clean[16:48, 16:48] = 0.5
    pairs = [(np.clip(clean + rng.normal(0, 0.03, clean.shape), 0, 1), clean) for _ in range(4)]
    model, _ = fit_residual(pairs, model=SmallCNN(mode="residual"), epochs=3, seed=3)
    model_path = tmp_path / "enhancer.npz"
    model.save(model_path)
    monkeypatch.setattr(cnn, "_ENHANCER_MODEL", model_path)
    monkeypatch.setattr(enhancer, "_cnn_enhancer", None)
    cfg = AstroFrameConfig()
    cfg.ai.cnn_enhance = True
    base = enhance_image(_disk_frame(), AstroFrameConfig())
    enhanced = enhance_image(_disk_frame(), cfg)
    assert enhanced.shape == base.shape


# ------------------------------------------------------------------ lstm --


def test_torch_available_verdadeiro_com_import_fingido(monkeypatch):
    torch = types.ModuleType("torch")
    original = builtins.__import__

    def fake(name, *args, **kwargs):
        if name == "torch":
            return torch
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)
    assert torch_available() is True


def test_lstm_cell_load_versao_desconhecida(tmp_path):
    cell = LSTMCell(2, 4)
    path = cell.save(tmp_path / "c.npz")
    data = np.load(path)
    np.savez(path, schema_version=99, n_in=2, n_hidden=4, W=data["W"], U=data["U"], b=data["b"])
    assert LSTMCell.load(path) is None


def test_lstm_tuner_fit_early_stop_por_patience():
    class _Run:
        def __init__(self, nudge):
            self.stars_calc = 4.0
            self.stars_user = None
            self.metrics = {"noise": 0.5}
            self.nudge = nudge

    # características idênticas com alvos diferentes → a loss estagna e o
    # patience (early-stop) termina o treino antes do limite de épocas.
    history = [_Run({"denoise.h": 2.0})] * 10 + [_Run({"denoise.h": -2.0})] * 4
    tuner = LSTMTuner(n_hidden=6, seed=1)
    fit = tuner.fit(history, epochs=60, patience=3, lr=0.02)
    assert fit.epochs < 60
    assert fit.best_loss != float("inf")


def test_lstm_tuner_load_versao_desconhecida_e_corrompido(tmp_path):
    tuner = LSTMTuner(n_hidden=8)
    path = tuner.save(tmp_path / "t.npz")
    data = np.load(path)
    np.savez(
        path,
        schema_version=99,
        n_hidden=8,
        W=data["W"],
        U=data["U"],
        b=data["b"],
        head=data["head"],
        head_bias=data["head_bias"],
    )
    assert LSTMTuner.load(path) is None
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"lixo")
    assert LSTMTuner.load(bad) is None


def test_trajectory_predict_lstm_celula_incompativel(tmp_path):
    cell = LSTMCell(3, 8)
    path = cell.save(tmp_path / "cell3.npz")
    pred = TrajectoryPredictor(use_lstm=True, model_path=path)
    for i in range(4):
        pred.push(float(i), 0.0)
    assert pred._predict_lstm() is None
    assert pred.predict() == pytest.approx((4.0, 0.0), abs=1e-6)


# -------------------------------------------------------------- feedback --


def test_feedback_history_all_e_reset_tuning(tmp_path):

    db = FeedbackDB(tmp_path / "fb.db")
    cfg = AstroFrameConfig()
    rating = score_from_stars(4.0)
    rating.metrics.update(noise=0.6)
    record_run(db, "image", "p1", cfg, {}, rating)
    record_run(db, "image", "p2", cfg, {}, rating)
    history = db.history_all()
    assert len(history) == 2
    assert history[0].id == 2
    db.add_tuning("p1", cfg.to_dict(), {"denoise.h": 1.0}, {"objective": 0.5})
    assert db.reset_tuning() == 1
    assert db.recent_tuning("p1") == []


def test_apply_learned_aplica_deltas_de_tuning(tmp_path):
    db = FeedbackDB(tmp_path / "fb.db")
    cfg = AstroFrameConfig()
    db.add_tuning("perfil", cfg.to_dict(), {"clahe.clip_limit": 0.6}, {"objective": 0.7})
    adjusted = apply_learned(cfg, "perfil", db=db)
    assert adjusted.clahe.clip_limit == pytest.approx(3.6)
    assert cfg.clahe.clip_limit == pytest.approx(3.0)


def test_apply_learned_tuning_respeita_limites(tmp_path):
    db = FeedbackDB(tmp_path / "fb.db")
    cfg = AstroFrameConfig()
    db.add_tuning("perfil", cfg.to_dict(), {"clahe.clip_limit": 99.0}, {})
    adjusted = apply_learned(cfg, "perfil", db=db)
    assert adjusted.clahe.clip_limit == 6.0


# ------------------------------------------------------------------ tuner --


def test_proxy_sem_ground_truth_objetivo_so_estrelas(tmp_path):
    samples = _make_samples(tmp_path, calibration=False)
    proxy = ProxyEval(samples, work_scale=0.5)
    report = proxy.evaluate(AstroFrameConfig())
    assert report.detection is None
    assert report.objective == pytest.approx(report.stars / 5.0)
    assert report.to_dict()["detection"] is None


def test_proxy_ignora_amostra_partida(tmp_path):
    samples = _make_samples(tmp_path, broken=True)
    proxy = ProxyEval(samples, work_scale=0.5)
    report = proxy.evaluate(AstroFrameConfig())
    assert report.n_items == 2


def test_proxy_escala_ground_truth_elipse(tmp_path):
    root = tmp_path / "samples"
    root.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / "a0.jpg"), _disk_frame(size=200, radius=50))
    store = CalibrationStore(root / "calibration.json")
    store.items["a0.jpg"] = CalibrationItem(
        "a0.jpg", "image", None, 200, 200, [DiskDetection(100, 100, 50, ry=40)]
    )
    store.save()
    proxy = ProxyEval(root, work_scale=0.5)
    assert proxy.evaluate(AstroFrameConfig()).n_items == 1


class _Proxy:
    """Proxy de avaliação fingido: objetivo melhora quando o delta é aplicado."""

    def __init__(self):
        self.calls = 0

    def evaluate(self, config):
        self.calls += 1
        if config.clahe.clip_limit > 3.0:
            return TuneReport(objective=0.9)
        return TuneReport(objective=0.4)


def test_lstm_seed_com_previsao_que_melhora(monkeypatch):
    class _Tuner:
        @staticmethod
        def load():
            return _Tuner()

        def predict_next_delta(self, history):
            return {"clahe.clip_limit": 0.3}

    monkeypatch.setattr("astroframe.ai.lstm.LSTMTuner", _Tuner)
    db = FeedbackDB(__import__("tempfile").mkdtemp() + "/fb.db")
    deltas = _lstm_seed(_Proxy(), AstroFrameConfig(), db)
    assert deltas == {"clahe.clip_limit": 0.3}


# ------------------------------------------------------------- integração --


def test_stabilizer_trajetoria_prevê_sem_deteção():
    cfg = AstroFrameConfig()
    cfg.ai.lstm_trajectory = True
    eng = AntiJitterStabilizer(config=cfg)
    for dx in (0, 10, 20, 30):
        eng.stabilize(_disk_frame(dx=dx))
    before = eng._smooth
    frame, detection = eng.stabilize(np.zeros((300, 300, 3), dtype=np.uint8))
    assert detection is None
    assert eng._smooth != before
    assert frame is not None


def test_stabilizer_trajetoria_sem_historico_nao_prevê():
    cfg = AstroFrameConfig()
    cfg.ai.lstm_trajectory = True
    eng = AntiJitterStabilizer(config=cfg)
    frame, detection = eng.stabilize(np.zeros((300, 300, 3), dtype=np.uint8))
    assert detection is None
    assert eng._smooth is None


def test_stabilizer_trajetoria_empurra_centroide_suavizado():
    cfg = AstroFrameConfig()
    cfg.ai.lstm_trajectory = True
    eng = AntiJitterStabilizer(config=cfg)
    eng.stabilize(_disk_frame())
    assert len(eng._trajectory) == 1


# -------------------------------------------------- early-stop e variantes --


def test_fit_residual_early_stop_por_patience():
    rng = np.random.default_rng(0)
    x = np.clip(rng.normal(0.5, 0.2, (48, 48)), 0, 1)
    pairs = [(x, np.clip(rng.normal(0.5, 0.5, (48, 48)), 0, 1)) for _ in range(8)]
    model, report = fit_residual(pairs, epochs=60, lr=0.001, seed=1)
    assert report.epochs < 60


def test_fit_classifier_uma_amostra_por_classe():
    rng = np.random.default_rng(0)
    pos = [np.clip(rng.normal(0.5, 0.1, (48, 48)), 0, 1)]
    neg = [np.clip(rng.normal(0.5, 0.1, (48, 48)), 0, 1)]
    model, report = fit_classifier(pos, neg, epochs=2, seed=3)
    assert report.best_loss != float("inf")


def test_fit_classifier_sem_negativas_parte_da_validacao():
    # só patches positivos → a validação é também o conjunto de treino
    rng = np.random.default_rng(0)
    pos = [np.clip(rng.normal(0.5, 0.1, (48, 48)), 0, 1)]
    model, report = fit_classifier(pos, [], epochs=2, seed=3)
    assert report.best_loss != float("inf")


def test_fit_classifier_sem_patches_erro():
    with pytest.raises(ValueError):
        fit_classifier([], [])


def test_fit_classifier_early_stop_por_patience():
    rng = np.random.default_rng(0)
    pos = [np.clip(rng.normal(0.5, 0.05, (48, 48)), 0, 1) for _ in range(8)]
    neg = [np.clip(rng.normal(0.5, 0.05, (48, 48)), 0, 1) for _ in range(8)]
    model, report = fit_classifier(pos, neg, epochs=60, lr=0.001, seed=1)
    assert report.epochs < 60


def test_lstm_tuner_fit_com_uma_janela():
    class _Run:
        def __init__(self, stars):
            self.stars_calc = stars
            self.stars_user = None
            self.metrics = {"noise": 0.5}
            self.nudge = {}

    tuner = LSTMTuner(n_hidden=6, seed=1)
    fit = tuner.fit([_Run(1.0), _Run(2.0)], epochs=5)
    assert fit.best_loss != float("inf")


def test_tune_result_to_dict():
    from astroframe.ai.tuner import TuneResult

    cfg = AstroFrameConfig()
    result = TuneResult(config=cfg, deltas={}, base=cfg, report=TuneReport(objective=0.5), evaluations=3)
    data = result.to_dict()
    assert data["report"]["objective"] == 0.5
    assert data["evaluations"] == 3


def test_proxy_redimensiona_amostra_grande_com_elipse(tmp_path):
    root = tmp_path / "samples"
    root.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / "a0.jpg"), _disk_frame(size=1200, radius=300))
    store = CalibrationStore(root / "calibration.json")
    store.items["a0.jpg"] = CalibrationItem(
        "a0.jpg", "image", None, 1200, 1200, [DiskDetection(600, 600, 300, ry=250)]
    )
    store.save()
    proxy = ProxyEval(root, work_scale=0.5)
    report = proxy.evaluate(AstroFrameConfig())
    assert report.n_items == 1


def test_lstm_seed_predicao_vazia(monkeypatch):
    class _Tuner:
        @staticmethod
        def load():
            return _Tuner()

        def predict_next_delta(self, history):
            return {}

    monkeypatch.setattr("astroframe.ai.lstm.LSTMTuner", _Tuner)
    db = FeedbackDB(__import__("tempfile").mkdtemp() + "/fb.db")
    assert _lstm_seed(_Proxy(), AstroFrameConfig(), db) == {}


def test_lstm_seed_excecao_falha_em_silencio(monkeypatch, caplog):
    import logging

    def _load():
        raise RuntimeError("sem modelo")

    monkeypatch.setattr("astroframe.ai.lstm.LSTMTuner.load", staticmethod(_load))
    db = FeedbackDB(__import__("tempfile").mkdtemp() + "/fb.db")
    with caplog.at_level(logging.WARNING):
        assert _lstm_seed(_Proxy(), AstroFrameConfig(), db) == {}
    assert any("Pré-seed LSTM" in record.getMessage() for record in caplog.records)
