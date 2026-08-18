"""Testes do enhancer_trainer.py — CNN de edição (residual), manual e automática."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import enhancer_trainer as et
import numpy as np
import pytest

from astroframe.ai.cnn import SmallCNN
from astroframe.ai.feedback import FeedbackDB
from astroframe.calibration.store import CalibrationItem, CalibrationStore
from astroframe.config import AstroFrameConfig
from astroframe.core.stabilizer import DiskDetection
from astroframe.paths import train_dir
from tests.helpers import make_disk_image

CIRCLE = DiskDetection(300, 140, 90)


def make_samples_dir(tmp_path: Path, n: int = 2) -> Path:
    root = tmp_path / "samples"
    root.mkdir()
    store = CalibrationStore(root / "calibration.json")
    for i in range(n):
        name = f"sample_{i}.jpg"
        image, cx, cy = make_disk_image()
        cv2.imwrite(str(root / name), image)
        store.items[name] = CalibrationItem(name, "image", None, 480, 360, [CIRCLE])
    store.save()
    return root


def make_pairs(n: int = 4, tile: int = et.TILE, seed: int = 7) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    x = rng.random((n, tile, tile))
    y = np.clip(x + rng.normal(0.0, 0.05, x.shape), 0.0, 1.0)
    return list(zip(x, y, strict=True))


def make_state(tmp_path: Path) -> et.EnhancerState:
    return et.EnhancerState(tmp_path / "enhancer_state.json")


# ------------------------------------------------------------------- canal L --


def test_l_channel_retorna_float_0_1():
    bgr = np.zeros((10, 12, 3), dtype=np.uint8)
    bgr[:, :, 0] = 255
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    light = et._l_channel(bgr)
    assert light.shape == (10, 12)
    assert light.dtype == np.float64
    assert 0.0 <= light.min() and light.max() <= 1.0
    assert light[0, 0] == pytest.approx(lab[0, 0, 0] / 255.0)


# ----------------------------------------------------------------- degradação --


def test_degrade_aplica_ruido_e_desfoque_deterministico():
    image = make_disk_image()[0]
    rng1 = np.random.default_rng(3)
    rng2 = np.random.default_rng(3)
    a = et.degrade(image, rng1)
    b = et.degrade(image, rng2)
    assert a.shape == image.shape
    assert np.array_equal(a, b)
    assert not np.array_equal(a, image)


def test_crop_pairs_forma_e_determinismo():
    clean = make_disk_image()[0]
    rng1 = np.random.default_rng(5)
    rng2 = np.random.default_rng(5)
    degraded = et.degrade(clean, rng1)
    pairs = et.crop_pairs(clean, degraded, rng2)
    assert pairs
    for x, y in pairs:
        assert x.shape == (et.TILE, et.TILE)
        assert y.shape == (et.TILE, et.TILE)
    assert np.array_equal(pairs[0][0], et.crop_pairs(clean, degraded, np.random.default_rng(5))[0][0])


def test_crop_pairs_imagem_pequena_vazio():
    small = np.zeros((20, 20, 3), dtype=np.uint8)
    assert et.crop_pairs(small, small, np.random.default_rng(1)) == []


def test_synthetic_pairs_reais(tmp_path):
    root = make_samples_dir(tmp_path, 1)
    samples = et.scan_samples(str(root))
    pairs = et.synthetic_pairs(samples[0], AstroFrameConfig(), np.random.default_rng(9))
    assert pairs
    assert len(pairs) <= et.MAX_CROPS_PER_SAMPLE


# ----------------------------------------------------------------- avaliação --


def test_evaluate_pairs_vazio():
    assert et.evaluate_pairs(SmallCNN(mode="residual", seed=1), [], 1) == {"mean_delta": 0.0, "mse": 1.0}


def test_evaluate_pairs_metricas_validas():
    pairs = make_pairs()
    model, _ = et.fit_residual(pairs, epochs=3, seed=2)
    result = et.evaluate_pairs(model, pairs, 3)
    assert 0.0 <= result["mean_delta"] <= 1.0
    assert result["mse"] >= 0.0
    assert result["mean_delta"] == pytest.approx(1.0 - result["mse"])


def test_evaluate_pairs_deterministico():
    pairs = make_pairs()
    model, _ = et.fit_residual(pairs, epochs=2, seed=2)
    assert et.evaluate_pairs(model, pairs, 3) == et.evaluate_pairs(model, pairs, 3)


# ------------------------------------------------------------------ estado --


def test_state_roundtrip(tmp_path):
    state = make_state(tmp_path)
    state.begin_round("auto")
    state.end_round({"mean_delta": 0.7, "promoted": True})
    state.pairs_positive = 5
    state.save()
    loaded = make_state(tmp_path)
    assert loaded.round == 1
    assert loaded.rounds[0]["mean_delta"] == 0.7
    assert loaded.rounds[0]["promoted"] is True
    assert loaded.pairs_positive == 5


def test_state_ilegivel_comeca_vazio(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{inválido", encoding="utf-8")
    state = et.EnhancerState(path)
    assert state.round == 0 and state.rounds == []


def test_state_versao_desconhecida_comeca_vazio(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": 99, "round": 7}), encoding="utf-8")
    state = et.EnhancerState(path)
    assert state.round == 0


def test_state_reset(tmp_path):
    state = make_state(tmp_path)
    state.begin_round("auto")
    state.pairs_positive = 3
    state.reset()
    loaded = make_state(tmp_path)
    assert loaded.round == 0 and loaded.pairs_positive == 0 and loaded.series == []


def test_state_end_round_ignora_round_desconhecido(tmp_path):
    state = make_state(tmp_path)
    state.round = 5
    state.end_round({"mean_delta": 0.5})
    assert state.rounds == []


# ---------------------------------------------------------- treino + campeão --


def test_train_enhancer_round_sem_pares_suficientes(tmp_path, monkeypatch):
    db = FeedbackDB(tmp_path / "fb.db")
    state = make_state(tmp_path)
    result = et.train_enhancer_round([], state, 1, db=db)
    assert result == {"skipped": True}
    assert state.series == []
    rows = db.logs(component="enhancer")
    assert rows and "sem pares suficientes" in rows[0]["message"]


def test_train_enhancer_round_promove_campeao(tmp_path):
    db = FeedbackDB(tmp_path / "fb.db")
    state = make_state(tmp_path)
    pairs = make_pairs(6)
    result = et.train_enhancer_round(pairs, state, 1, epochs=3, db=db)
    assert result["skipped"] is False
    assert result["promoted"] is True
    assert result["staged"].exists()
    assert et.ENHANCER_CANONICAL_PATH.exists()
    assert et.ENHANCER_CANONICAL_PATH.read_bytes() == result["staged"].read_bytes()
    assert state.series[0]["round"] == 1 and state.series[0]["promoted"] is True
    champion = db.champion("enhancer")
    assert champion is not None and champion["path"] == str(result["staged"])
    assert db.champion("disk_filter") is None


def test_train_enhancer_round_mantem_campeao_quando_pior(tmp_path):
    db = FeedbackDB(tmp_path / "fb.db")
    state = make_state(tmp_path)
    pairs = make_pairs(6, seed=11)
    first = et.train_enhancer_round(pairs, state, 1, epochs=4, db=db, seed=10)
    second = et.train_enhancer_round(pairs, state, 2, epochs=4, db=db, seed=10)
    assert first["promoted"] is True
    assert second["promoted"] is False
    assert second["champion_path"] == str(first["staged"])
    assert state.series[-1]["promoted"] is False
    champion = db.champion("enhancer")
    assert champion["path"] == str(first["staged"])
    logs = [r["message"] for r in db.logs(component="enhancer")]
    assert any("pior que o campeão" in message for message in logs)


def test_train_enhancer_round_warm_start_do_campeao(tmp_path):
    db = FeedbackDB(tmp_path / "fb.db")
    state = make_state(tmp_path)
    pairs = make_pairs(6, seed=13)
    first = et.train_enhancer_round(pairs, state, 1, epochs=3, db=db, seed=12)
    second = et.train_enhancer_round(
        pairs, state, 2, epochs=3, db=db, seed=12, champion_path=first["staged"]
    )
    assert second["skipped"] is False
    assert second["mean_delta"] >= 0.0


# ------------------------------------------------------------- headless auto --


def test_run_auto_headless_promove_e_exporta(tmp_path):
    root = make_samples_dir(tmp_path)
    assert et.run_auto_headless(
        str(root), series=2, epochs=2, export_path=str(tmp_path / "modelo.npz")
    ) == 0
    assert (tmp_path / "modelo.npz").exists()
    assert et.ENHANCER_CANONICAL_PATH.exists()
    state = et.EnhancerState(train_dir() / et.DEFAULT_STATE_NAME)
    assert state.round == 2
    assert len(state.series) == 2
    assert state.series[0]["promoted"] is True
    db = FeedbackDB()
    assert db.champion("enhancer") is not None


def test_run_auto_headless_sem_amostras(tmp_path):
    root = tmp_path / "samples"
    root.mkdir()
    assert et.run_auto_headless(str(root), series=1, epochs=2) == 0
    assert not et.ENHANCER_CANONICAL_PATH.exists()


def test_run_auto_headless_continua_do_campeao(tmp_path):
    root = make_samples_dir(tmp_path)
    et.run_auto_headless(str(root), series=1, epochs=2)
    assert et.run_auto_headless(str(root), series=1, epochs=2) == 0
    state = et.EnhancerState(train_dir() / et.DEFAULT_STATE_NAME)
    assert state.round == 2


def test_run_check_imprime_relatorio(tmp_path, capsys):
    root = make_samples_dir(tmp_path)
    assert et.run_check(str(root)) == 0
    out = capsys.readouterr().out
    assert "verificação da melhoria CNN" in out
    assert "★" in out


# ------------------------------------------------------------------ relatório --


def test_build_report_com_e_sem_cnn(tmp_path):
    root = make_samples_dir(tmp_path, 1)
    samples = et.scan_samples(str(root))
    store = CalibrationStore(root / "calibration.json")
    report = et.build_report(samples, store, AstroFrameConfig(), pairs_count=7)
    assert report.samples_done == 1
    assert report.pairs == 7
    assert report.stars_cnn is not None and report.stars_plain is not None
    assert report.errors == []


def test_build_report_erro_de_leitura(tmp_path):
    root = tmp_path / "samples"
    root.mkdir()
    (root / "quebrada.jpg").write_bytes(b"nao e imagem")
    samples = et.scan_samples(str(root))
    report = et.build_report(samples, CalibrationStore(root / "calibration.json"), AstroFrameConfig())
    assert report.samples_done == 0
    assert report.stars_cnn is None
    assert report.errors and "erro ao ler" in report.errors[0]


def test_report_lines():
    report = et.EnhancerReport(1, 1, 4.2, 3.8, 3)
    lines = report.lines()
    assert "4.2★" in lines[0] and "3.8★" in lines[0]
    assert "Pares recolhidos: 3" in lines[1]
    assert et.EnhancerReport(0, 2, None, None, 0).lines()[0] == "Melhoria CNN: sem amostras avaliadas"


# ----------------------------------------------------------------------- CLI --


def test_main_reset_state(tmp_path, capsys):
    root = tmp_path / "samples"
    root.mkdir()
    state = et.EnhancerState(train_dir() / et.DEFAULT_STATE_NAME)
    state.begin_round("auto")
    assert et.main(["--samples", str(root), "--reset-state"]) == 0
    assert "reposto" in capsys.readouterr().out


def test_main_check_retorna_0(tmp_path):
    root = make_samples_dir(tmp_path)
    assert et.main(["--samples", str(root), "--check"]) == 0


def test_main_auto_retorna_0(tmp_path):
    root = make_samples_dir(tmp_path, 1)
    assert et.main(["--samples", str(root), "--auto", "--series", "1", "--epochs", "2"]) == 0


def test_build_parser_tem_fluxos():
    parser = et.build_parser()
    args = parser.parse_args(["--auto", "--series", "5", "--epochs", "9", "--seed", "3"])
    assert args.auto and args.series == 5 and args.epochs == 9 and args.seed == 3
    args2 = parser.parse_args(["--check"])
    assert args2.check is True


def test_sample_stars_diferente_do_limiar():
    frame = make_disk_image()[0]
    config = AstroFrameConfig()
    stars = et.sample_stars(frame, CIRCLE, config, with_cnn=True)
    assert 0.0 <= stars <= 5.0

def test_detection_sem_guia_devolve_none():
    store = CalibrationStore("/tmp/inexistente/calibration.json")
    assert et._detection(store, "qualquer.jpg") is None
    assert et._detection(store, "outra.jpg") is None


def test_report_lines_ramo_sem_deteção():
    lines = et.EnhancerReport(1, 1, 4.2, None, 0).lines()
    assert "4.2★" in lines[0] and "sem deteção" in lines[0]


def test_build_report_lines_com_erros(tmp_path):
    root = tmp_path / "samples"
    root.mkdir()
    (root / "quebrada.jpg").write_bytes(b"nao e imagem")
    samples = et.scan_samples(str(root))
    report = et.build_report(samples, CalibrationStore(root / "calibration.json"), AstroFrameConfig())
    assert any("!" in line for line in report.lines())


def test_run_auto_headless_erro_de_leitura_registado(tmp_path, capsys):
    root = make_samples_dir(tmp_path, 1)
    (root / "quebrada.jpg").write_bytes(b"nao e imagem")
    assert et.run_auto_headless(str(root), series=1, epochs=2) == 0
    assert "erro" in capsys.readouterr().out


def test_main_config_invalido_devolve_1(tmp_path):
    root = make_samples_dir(tmp_path, 1)
    bad = tmp_path / "bad.yaml"
    bad.write_text(":", encoding="utf-8")
    assert et.main(["--samples", str(root), "--check", "--config", str(bad)]) == 1


def test_run_check_com_estado_anterior(tmp_path, capsys):
    root = make_samples_dir(tmp_path)
    state = et.EnhancerState(train_dir() / et.DEFAULT_STATE_NAME)
    state.begin_round("auto")
    state.pairs_positive = 4
    state.save()
    assert et.run_check(str(root)) == 0
    out = capsys.readouterr().out
    assert "Treinos anteriores: 1 série(s), 4 pares válidos" in out
