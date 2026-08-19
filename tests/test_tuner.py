"""Testes do auto-tuning (proxy + hill-climbing + persistência)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from astroframe.ai import params as pparams
from astroframe.ai.feedback import FeedbackDB
from astroframe.ai.tuner import (
    BoundedHillClimb,
    ProxyEval,
    TuneReport,
    TuneResult,
    _lstm_seed,
    export_trained_config,
    run_autotune,
    tuning_table_lines,
)
from astroframe.calibration.store import CalibrationItem, CalibrationStore
from astroframe.config import AstroFrameConfig
from astroframe.core.stabilizer import DiskDetection


def make_samples(root: Path, n: int = 2, size: int = 200, radius: int = 50) -> Path:
    """Pasta `samples` com imagens sintéticas + calibration.json."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    store = CalibrationStore(root / "calibration.json")
    for i in range(n):
        image = np.zeros((size, size, 3), dtype=np.uint8)
        cv2.circle(image, (size // 2, size // 2), radius, (200, 200, 200), -1)
        cv2.imwrite(str(root / f"a{i}.jpg"), image)
        store.items[f"a{i}.jpg"] = CalibrationItem(
            f"a{i}.jpg",
            "image",
            None,
            size,
            size,
            [DiskDetection(size // 2, size // 2, radius)],
        )
    store.save()
    return root


# ------------------------------------------------------------------ proxy --


def test_proxy_avalia_e_guarda_em_cache(tmp_path):
    samples = make_samples(tmp_path)
    proxy = ProxyEval(samples, work_scale=0.5, seed=7)
    assert len(proxy.samples) == 2
    cfg = AstroFrameConfig()
    first = proxy.evaluate(cfg)
    assert first.n_items == 2
    second = proxy.evaluate(cfg)
    assert second is first
    proxy.clear_cache()
    assert proxy.evaluate(cfg) is not first


def test_proxy_cache_key_deterministico(tmp_path):
    samples = make_samples(tmp_path)
    proxy = ProxyEval(samples)
    cfg = AstroFrameConfig()
    assert proxy._cache_key(cfg) == proxy._cache_key(AstroFrameConfig())
    cfg.clahe.clip_limit = 2.0
    assert proxy._cache_key(cfg) != proxy._cache_key(AstroFrameConfig())


def test_proxy_scale_nunca_aumenta(tmp_path):
    samples = make_samples(tmp_path)
    proxy = ProxyEval(samples, work_scale=0.25)
    assert proxy._scale_for(200, 200) == 1.0
    assert proxy._scale_for(4000, 4000) == pytest.approx(480 / 4000)


def test_proxy_usa_work_scale_480p(tmp_path):
    samples = make_samples(tmp_path)
    proxy = ProxyEval(samples, work_scale=0.5)
    assert proxy._scale_for(2000, 2000) == pytest.approx(480 / 2000)


# ------------------------------------------------------------ hill-climbing --


class _FakeEval:
    """Avaliação determinística: objetivo máximo no delta 0.5 do parâmetro."""

    def __init__(self):
        self.calls = 0

    def __call__(self, config: AstroFrameConfig) -> TuneReport:
        self.calls += 1
        delta = pparams.get_param(config, "clahe.clip_limit") - AstroFrameConfig().clahe.clip_limit
        return TuneReport(objective=max(0.0, 1.0 - abs(delta - 0.5)))


def test_hill_climb_encontra_melhor_delta():
    base = AstroFrameConfig()
    optimizer = BoundedHillClimb(pparams.specs("enhance"), budget_s=5.0, seed=1, anneal=False)
    fake = _FakeEval()
    result = optimizer.optimize(fake, base)
    assert result.evaluations >= 3
    assert result.deltas["clahe.clip_limit"] == pytest.approx(0.5, abs=0.05)
    assert result.report.objective == pytest.approx(1.0, abs=1e-4)


def test_hill_climb_nao_muta_a_base():
    base = AstroFrameConfig()
    before = base.to_dict()
    BoundedHillClimb(pparams.specs("enhance"), budget_s=0.0, seed=1).optimize(_FakeEval(), base)
    assert base.to_dict() == before


def test_hill_climb_orcamento_zero_somente_avalia_a_base():
    fake = _FakeEval()
    result = BoundedHillClimb(pparams.specs("enhance"), budget_s=0.0, seed=1).optimize(
        fake, AstroFrameConfig()
    )
    assert result.evaluations == 1


def test_hill_climb_com_start_deltas_aplica_seed():
    base = AstroFrameConfig()
    fake = _FakeEval()
    result = BoundedHillClimb(pparams.specs("enhance"), budget_s=0.0, seed=1).optimize(
        fake, base, start_deltas={"clahe.clip_limit": 0.5}
    )
    assert result.config.clahe.clip_limit == pytest.approx(3.5)


def test_hill_climb_anneal_aceita_piora_sem_sair_das_gamas():
    base = AstroFrameConfig()
    rng = np.random.default_rng(0)
    calls = 0

    def evaluator(config):
        nonlocal calls
        calls += 1
        delta = pparams.get_param(config, "clahe.clip_limit") - base.clahe.clip_limit
        return TuneReport(objective=max(0.0, 1.0 - abs(delta - 0.5)) + 0.01 * rng.random())

    optimizer = BoundedHillClimb(pparams.specs("enhance"), budget_s=2.0, seed=3)
    result = optimizer.optimize(evaluator, base)
    assert 0.5 <= result.config.clahe.clip_limit <= 6.0
    assert calls >= 1


def test_tuning_table_lines_sem_alteracoes():
    base = AstroFrameConfig()
    result = TuneResult(config=base, deltas={}, base=base, report=TuneReport(objective=0.5))
    lines = tuning_table_lines(result)
    assert "Nenhum parâmetro ajustado" in lines[0]
    assert "Objetivo 0.500" in lines[-1]


def test_tuning_table_lines_com_alteracoes():
    base = AstroFrameConfig()
    adjusted = pparams.apply_deltas(base, {"clahe.clip_limit": 0.4})
    result = TuneResult(
        config=adjusted,
        deltas={"clahe.clip_limit": 0.4},
        base=base,
        report=TuneReport(objective=0.8, stars=4.0, detection=0.9),
    )
    lines = tuning_table_lines(result)
    assert "clahe.clip_limit" in lines[0]
    assert "Objetivo 0.800" in lines[-1]


def test_export_trained_config_json(tmp_path):
    base = AstroFrameConfig()
    report = TuneReport(objective=0.9, stars=4.5, detection=0.8, recall=0.8, precision=0.7)
    path = export_trained_config(base, {"clahe.clip_limit": 0.3}, report, tmp_path / "out" / "cfg.json")
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 2
    assert data["kind"] == "astroframe-tuned"
    assert data["params"]["clahe.clip_limit"] == pytest.approx(3.3)
    assert "param2" in data["stabilizer"]
    assert data["report"]["objective"] == 0.9


def test_export_trained_config_clampa_deltas(tmp_path):
    base = AstroFrameConfig()
    report = TuneReport(objective=0.5)
    path = export_trained_config(base, {"clahe.clip_limit": 99.0}, report, tmp_path / "cfg.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["params"]["clahe.clip_limit"] == 6.0


# -------------------------------------------------------------- orquestração --


def test_run_autotune_completo_com_db(tmp_path):
    samples = make_samples(tmp_path)
    db = FeedbackDB(tmp_path / "fb.db")
    result = run_autotune(
        samples,
        budget_s=0.5,
        seed=11,
        profile="teste",
        db=db,
        work_scale=0.5,
    )
    assert isinstance(result, TuneResult)
    assert result.evaluations >= 1
    history = db.tuning_history("teste")
    assert len(history) == 1
    assert history[0]["source"] == "autotune"
    assert history[0]["deltas"] == result.deltas
    assert db.recent_tuning("teste") == [history[0]["deltas"]]
    assert db.recent_tuning("outro") == []


def test_run_autotune_exporta_e_regista_por_omissao(tmp_path):
    samples = make_samples(tmp_path)
    db = FeedbackDB(tmp_path / "fb.db")
    out = tmp_path / "trained_config.json"
    result = run_autotune(
        samples,
        budget_s=0.3,
        export_path=out,
        db=db,
        work_scale=0.5,
    )
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["deltas"] == result.deltas


def test_run_autotune_params_filter(tmp_path):
    samples = make_samples(tmp_path)
    result = run_autotune(
        samples,
        budget_s=0.3,
        params_filter="clip_limit,param2",
        db=FeedbackDB(tmp_path / "fb.db"),
        work_scale=0.5,
    )
    assert set(result.deltas) <= {"clahe.clip_limit", "stabilizer.param2"}


def test_run_autotune_sem_parametros_levanta(tmp_path):
    samples = make_samples(tmp_path)
    with pytest.raises(ValueError, match="Nenhum parâmetro"):
        run_autotune(samples, params_filter="nao.existe", budget_s=0.1)


def test_lstm_seed_sem_db_e_vazio(tmp_path):
    samples = make_samples(tmp_path)
    proxy = ProxyEval(samples)
    assert _lstm_seed(proxy, AstroFrameConfig(), None) == {}


def test_lstm_seed_com_db_sem_modelo_e_vazio(tmp_path):
    samples = make_samples(tmp_path)
    proxy = ProxyEval(samples)
    db = FeedbackDB(tmp_path / "fb.db")
    assert _lstm_seed(proxy, AstroFrameConfig(), db) == {}
