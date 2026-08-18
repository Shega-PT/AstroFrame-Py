"""Testes da validação: deltas de treino, estado, sessão e relatórios."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import validator
from validator import (
    AUTO_IOU_END,
    AUTO_IOU_MAX,
    AUTO_IOU_MIN,
    AUTO_IOU_START,
    DELTA_BOUNDS,
    MAX_REEVALS_PER_SAMPLE,
    PUNISH_DELTAS,
    REWARD_DELTAS,
    AutoTrainer,
    ValidationSession,
    ValidatorState,
    apply_effective,
    apply_state_weights,
    auto_iou_threshold,
    best_gt_iou,
    build_global_report,
    circle_from_dict,
    circle_to_dict,
    classifier_accuracy,
    effective_params,
    export_trained,
    filter_unjudged,
    main,
    metrics_from_report,
    nudge_deltas,
    persistent_rejected,
    rounds_text,
    run_auto_headless,
    same_shape,
    sample_done_text,
    train_classifier_round,
)

from astroframe.ai.cnn import fit_classifier
from astroframe.ai.feedback import FeedbackDB
from astroframe.calibration.scan import SampleRef
from astroframe.calibration.store import CalibrationItem, CalibrationStore
from astroframe.calibration.validate import validate_all
from astroframe.config import AstroFrameConfig
from astroframe.core.stabilizer import DiskDetection
from astroframe.paths import train_dir
from tests.helpers import make_disk_image

CIRCLE = DiskDetection(100, 100, 50)


def make_sample(key: str = "a.jpg", kind: str = "image") -> SampleRef:
    return SampleRef(kind, Path(key), None, key, f"IMG {key}")


def make_session(tmp_path, gt=None, deltas=None, sample_key: str = "a.jpg") -> ValidationSession:
    config = AstroFrameConfig()
    store = CalibrationStore(tmp_path / "calibration.json")
    if gt is not None:
        store.items[sample_key] = CalibrationItem(sample_key, "image", None, 640, 480, list(gt))
    state = ValidatorState(tmp_path / "validator_state.json")
    state.deltas = dict(deltas or {})
    session = ValidationSession([make_sample(sample_key)], store, config, state)
    session.start(0)
    return session


# ------------------------------------------------------------------ deltas --


def test_nudge_punish_aumenta_os_deltas():
    deltas = nudge_deltas({}, "reject")
    assert deltas["param2"] == PUNISH_DELTAS["param2"]
    assert deltas["param1"] == PUNISH_DELTAS["param1"]
    assert deltas["dp"] == PUNISH_DELTAS["dp"]
    assert deltas["occluded_ratio"] == PUNISH_DELTAS["occluded_ratio"]


def test_nudge_reward_reduz_os_deltas():
    deltas = nudge_deltas({}, "accept")
    assert deltas["param2"] == REWARD_DELTAS["param2"]
    assert deltas["occluded_ring"] == REWARD_DELTAS["occluded_ring"]


def test_nudge_nao_muda_os_deltas_dos_outros():
    deltas = nudge_deltas({"param2": 3.0}, "reject")
    assert deltas["param2"] == pytest.approx(4.0)


def test_nudge_respeita_os_limites():
    config = AstroFrameConfig()
    config.stabilizer.param2 = DELTA_BOUNDS["param2"][1]
    assert nudge_deltas({"param2": 0.0}, "reject", config)["param2"] == 0.0
    config.stabilizer.param2 = DELTA_BOUNDS["param2"][0]
    assert nudge_deltas({"param2": 0.0}, "accept", config)["param2"] == 0.0
    config.stabilizer.param2 = DELTA_BOUNDS["param2"][1]
    assert nudge_deltas({"param2": 0.0}, "accept", config)["param2"] == REWARD_DELTAS["param2"]


def test_effective_params_aplica_deltas():
    config = AstroFrameConfig()
    params = effective_params(config, {"param2": 3.5, "dp": 0.2})
    assert params["param2"] == config.stabilizer.param2 + round(3.5)
    assert params["dp"] == pytest.approx(config.stabilizer.dp + 0.2)
    assert params["max_radius"] == config.stabilizer.max_radius


def test_effective_params_clampa_negativos():
    config = AstroFrameConfig()
    params = effective_params(config, {"param2": -100.0, "occluded_ring": -10.0})
    assert params["param2"] == DELTA_BOUNDS["param2"][0]
    assert params["occluded_ring"] == DELTA_BOUNDS["occluded_ring"][0]


def test_effective_params_kernel_impar_e_por_tipo():
    config = AstroFrameConfig()
    params = effective_params(config, {"gaussian_kernel_size": 2.0})
    assert params["gaussian_kernel_size"] % 2 == 1
    assert isinstance(params["param2"], int)
    assert isinstance(params["dp"], float)


def test_apply_effective_muta_a_config():
    config = AstroFrameConfig()
    apply_effective(config, {"param2": 7.0})
    assert config.stabilizer.param2 == AstroFrameConfig().stabilizer.param2 + 7


# ------------------------------------------------------------------ formas --


def test_same_shape_por_iou():
    assert same_shape(CIRCLE, DiskDetection(102, 100, 50))
    assert not same_shape(CIRCLE, DiskDetection(300, 300, 50))


def test_filter_unjudged_exclui_aceites_e_rejeitadas():
    detected = [CIRCLE, DiskDetection(300, 300, 40), DiskDetection(500, 500, 20)]
    accepted = [DiskDetection(101, 99, 50)]
    rejected = [DiskDetection(301, 299, 40)]
    assert filter_unjudged(detected, accepted, rejected) == [DiskDetection(500, 500, 20)]


def test_persistent_rejected_so_as_que_voltam():
    rejected = [DiskDetection(300, 300, 40), DiskDetection(600, 600, 10)]
    detected = [CIRCLE, DiskDetection(299, 301, 40)]
    assert persistent_rejected(rejected, detected) == [rejected[0]]


def test_best_gt_iou():
    gt = [DiskDetection(300, 300, 40)]
    assert best_gt_iou(CIRCLE, gt) == 0.0
    assert best_gt_iou(DiskDetection(302, 299, 40), gt) == pytest.approx(1.0, abs=0.1)
    assert best_gt_iou(CIRCLE, []) is None


# ------------------------------------------------------------------ estado --


def test_state_roundtrip(tmp_path):
    path = tmp_path / "validator_state.json"
    state = ValidatorState(path)
    state.deltas = {"param2": 2.0}
    state.rewards = 3
    state.punishments = 1
    record = state.record("a.jpg")
    record["accepted"] = [CIRCLE]
    record["done"] = True
    state.save()

    loaded = ValidatorState(path)
    assert loaded.deltas == {"param2": 2.0}
    assert loaded.rewards == 3
    assert loaded.punishments == 1
    assert loaded.is_done("a.jpg")
    assert loaded.samples["a.jpg"]["accepted"] == [CIRCLE]


def test_state_json_invalido_comeca_vazio(tmp_path):
    path = tmp_path / "validator_state.json"
    path.write_text("{nao é json", encoding="utf-8")
    state = ValidatorState(path)
    assert state.deltas == {}
    assert state.samples == {}


def test_state_versao_desconhecida_comeca_vazio(tmp_path):
    path = tmp_path / "validator_state.json"
    path.write_text(json.dumps({"version": 99}), encoding="utf-8")
    state = ValidatorState(path)
    assert state.samples == {}


def test_circle_roundtrip_com_elipse():
    ellipse = DiskDetection(10, 20, 30, 15)
    assert circle_from_dict(circle_to_dict(ellipse)) == ellipse
    assert circle_from_dict(circle_to_dict(CIRCLE)) == CIRCLE


# ----------------------------------------------------------------- sessão --


def test_sessao_apresenta_formas_pendentes(tmp_path):
    session = make_session(tmp_path)
    action = session.apply_detection([CIRCLE, DiskDetection(300, 300, 40)])
    assert action == "present"
    assert session.pending == [CIRCLE, DiskDetection(300, 300, 40)]
    assert session.current == CIRCLE


def test_sessao_aceitar_recompensa(tmp_path):
    session = make_session(tmp_path)
    session.apply_detection([CIRCLE])
    action = session.accept()
    assert action == "redetect"
    assert session.accepted == [CIRCLE]
    assert session.state.rewards == 1
    assert session.state.deltas["param2"] == REWARD_DELTAS["param2"]
    assert session.state.samples["a.jpg"]["done"] is False
    assert session.apply_detection([CIRCLE]) == "complete"


def test_sessao_rejeitar_pune_e_reavalia(tmp_path):
    session = make_session(tmp_path)
    session.apply_detection([CIRCLE, DiskDetection(300, 300, 40)])
    action = session.reject()
    assert action == "present"
    assert session.rejected == [CIRCLE]
    assert session.state.punishments == 1
    assert session.state.deltas["param2"] == PUNISH_DELTAS["param2"]
    assert session.pending == [DiskDetection(300, 300, 40)]
    assert session.reject() == "redetect"
    assert session.state.punishments == 2
    assert session.state.deltas["param2"] == 2 * PUNISH_DELTAS["param2"]


def test_sessao_falso_positivo_teimoso_pune_em_loop(tmp_path):
    session = make_session(tmp_path)
    ghost = DiskDetection(300, 300, 40)
    session.apply_detection([CIRCLE, ghost])
    session.accept()
    session.reject()
    action = session.apply_detection([CIRCLE, ghost])
    assert action == "redetect"
    assert session.state.punishments == 2
    action = session.apply_detection([CIRCLE])
    assert action == "complete"


def test_sessao_sem_formas_completa(tmp_path):
    session = make_session(tmp_path)
    assert session.apply_detection([]) == "complete"


def test_sessao_limite_de_reavaliaçoes(tmp_path):
    session = make_session(tmp_path)
    ghost = DiskDetection(300, 300, 40)
    session.apply_detection([CIRCLE, ghost])
    session.accept()
    session.reject()
    for _ in range(MAX_REEVALS_PER_SAMPLE + 2):
        action = session.apply_detection([CIRCLE, ghost])
    assert action == "complete"
    assert session.reevals == MAX_REEVALS_PER_SAMPLE


def test_sessao_complete_guarda_e_relatorio(tmp_path):
    gt = [CIRCLE]
    session = make_session(tmp_path, gt=gt)
    session.apply_detection([CIRCLE])
    session.accept()
    session.complete()
    assert session.state.is_done("a.jpg")

    result = build_global_report(session.samples, session.store, session.state)
    assert result is not None
    report, lines = result
    assert report.total_manual == 1
    assert report.total_matched == 1
    assert report.recall == 1.0
    assert any("Score global" in line for line in lines)
    assert "a.jpg" in lines[-1]


def test_sessao_sem_guia_relatorio_nao_pontua(tmp_path):
    session = make_session(tmp_path)
    session.apply_detection([CIRCLE])
    session.accept()
    session.complete()
    result = build_global_report(session.samples, session.store, session.state)
    assert result is not None
    _report, lines = result
    assert any("Sem guia" in line for line in lines)


def test_sample_done_text(tmp_path):
    session = make_session(tmp_path, gt=[CIRCLE])
    session.apply_detection([CIRCLE])
    session.accept()
    text = sample_done_text(session)
    assert "1 válida" in text and "guia" in text


def test_detect_config_aplica_deltas_sem_mutar_a_base(tmp_path):
    session = make_session(tmp_path)
    session.apply_detection([CIRCLE])
    session.reject()
    adjusted = session.detect_config()
    assert adjusted.stabilizer.param2 == AstroFrameConfig().stabilizer.param2 + 1
    assert session.config.stabilizer.param2 == AstroFrameConfig().stabilizer.param2


def test_deltas_acumulam_sem_ser_esmagados(tmp_path):
    session = make_session(tmp_path)
    session.apply_detection([CIRCLE])
    for _ in range(10):
        session.apply_detection([CIRCLE])
        session.reject()
    assert session.state.deltas["param2"] == pytest.approx(10 * PUNISH_DELTAS["param2"])


# ------------------------------------------------------------------ séries --


def test_auto_iou_threshold_encolhe_com_as_validaçoes():
    assert auto_iou_threshold(0) == AUTO_IOU_START
    prev = AUTO_IOU_START
    for n in (1, 10, 100, 500, 2000):
        value = auto_iou_threshold(n)
        assert prev < value < AUTO_IOU_END
        prev = value
    assert auto_iou_threshold(10**6) == pytest.approx(AUTO_IOU_END, abs=1e-6)


def test_auto_iou_threshold_nunca_abaixo_de_090():
    assert AUTO_IOU_START >= 0.90
    for start in (0.90, 0.93, 0.95):
        assert auto_iou_threshold(0, start=start) == start
        assert auto_iou_threshold(0, start=start) >= 0.90
        assert auto_iou_threshold(10**6, start=start) == pytest.approx(AUTO_IOU_END, abs=1e-6)


def test_auto_trainer_clampa_iou_minimo_entre_090_e_099(tmp_path):
    store = CalibrationStore(tmp_path / "calibration.json")
    config = AstroFrameConfig()
    state = ValidatorState(tmp_path / "validator_state.json")
    samples = [make_sample("a.jpg")]
    low = AutoTrainer(samples, store, config, state, iou_threshold=0.5)
    assert low.iou_threshold == AUTO_IOU_MIN
    assert low.iou_start == AUTO_IOU_MIN
    high = AutoTrainer(samples, store, config, state, iou_threshold=0.999)
    assert high.iou_threshold == AUTO_IOU_MAX
    assert high.iou_end == AUTO_IOU_MAX


def test_estado_rounds_begin_end_clear(tmp_path):
    state = ValidatorState(tmp_path / "state.json")
    state.deltas = {"param2": 1.0}
    assert state.begin_round("auto") == 1
    assert state.round == 1
    state.rewards += 3
    state.punishments += 2
    state.deltas = {"param2": 4.0, "occluded_ratio": 0.02}
    state.end_round(metrics_from_report(validate_all([("a", [CIRCLE], [CIRCLE])])))
    record = state.rounds[-1]
    assert record["rewards"] == 3
    assert record["punishments"] == 2
    assert record["deltas"]["param2"] == 4.0
    assert record["score"] == pytest.approx(100.0)
    assert state.begin_round("manual") == 2
    assert state.samples == {}
    assert state.deltas["param2"] == 4.0

    loaded = ValidatorState(tmp_path / "state.json")
    assert loaded.round == 2
    assert len(loaded.rounds) == 2
    assert loaded.rounds[0]["rewards"] == 3
    assert loaded.rounds[1]["mode"] == "manual"


def test_rounds_text_só_mostra_séries_fechadas(tmp_path):
    state = ValidatorState(tmp_path / "state.json")
    state.begin_round("auto")
    state.end_round()
    state.begin_round("manual")
    lines = rounds_text(state)
    assert len(lines) == 1
    assert "Série 1 (auto)" in lines[0]


def test_export_trained(tmp_path):
    state = ValidatorState(tmp_path / "state.json")
    state.deltas = {"param2": 3.0, "occluded_ratio": 0.02}
    state.rewards = 5
    state.punishments = 2
    state.round = 2
    config = AstroFrameConfig()
    report = validate_all([("a", [CIRCLE], [CIRCLE])])
    path = export_trained(state, config, tmp_path / "trained.json", report)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["rounds"] == 2
    assert data["stats"]["rewards"] == 5
    assert data["deltas"]["param2"] == 3.0
    assert data["stabilizer"]["param2"] == config.stabilizer.param2 + 3
    assert data["stabilizer"]["occluded_ratio"] == pytest.approx(
        config.stabilizer.occluded_ratio + 0.02
    )
    assert "min_radius" not in data["stabilizer"]
    assert "min_dist" not in data["stabilizer"]
    assert data["stabilizer"]["max_radius"] == config.stabilizer.max_radius
    assert data["score"]["score"] == pytest.approx(100.0)


# ---------------------------------------------------------- treino auto --


def make_auto_trainer(tmp_path, gt, monkeypatch, state=None):
    store = CalibrationStore(tmp_path / "calibration.json")
    store.items["a.jpg"] = CalibrationItem("a.jpg", "image", None, 640, 480, list(gt))
    config = AstroFrameConfig()
    state = state or ValidatorState(tmp_path / "validator_state.json")
    samples = [make_sample("a.jpg")]

    def fake_load(_sample):
        return np.zeros((100, 100, 3), dtype=np.uint8)

    monkeypatch.setattr("validator.load_frame", fake_load)
    trainer = AutoTrainer(samples, store, config, state)
    return trainer, state


def test_auto_trainer_aceita_matches_e_rejeita_ghosts(tmp_path, monkeypatch):
    ghost = DiskDetection(300, 300, 40)
    calls = {"n": 0}

    def fake_detect(_frame, _config):
        calls["n"] += 1
        return [CIRCLE] if calls["n"] > 1 else [CIRCLE, ghost]

    monkeypatch.setattr("validator.find_all_disks", fake_detect)
    trainer, state = make_auto_trainer(tmp_path, [CIRCLE], monkeypatch)
    report = trainer.run_series()
    assert report.samples_done == 1
    assert report.rewards == 1
    assert report.punishments == 1
    assert state.is_done("a.jpg")
    assert report.report is not None and report.report.score == pytest.approx(100.0)


def test_auto_trainer_sem_guia_completa_sem_julgamentos(tmp_path, monkeypatch):
    monkeypatch.setattr("validator.find_all_disks", lambda _f, _c: [CIRCLE])
    trainer, state = make_auto_trainer(tmp_path, [], monkeypatch)
    report = trainer.run_series()
    assert report.rewards == 0
    assert report.punishments == 0
    assert state.is_done("a.jpg")


def test_auto_trainer_respeita_dever_parar(tmp_path, monkeypatch):
    monkeypatch.setattr("validator.find_all_disks", lambda _f, _c: [CIRCLE])
    trainer, _state = make_auto_trainer(tmp_path, [CIRCLE], monkeypatch)
    report = trainer.run_series(should_stop=lambda: True)
    assert report.stopped
    assert report.samples_done == 0


def test_auto_trainer_margem_mais_exigente_com_mais_validaçoes(tmp_path, monkeypatch):
    near_miss = DiskDetection(104, 100, 50)  # IoU ≈ 0.90 com o guia
    assert best_gt_iou(near_miss, [CIRCLE]) >= AUTO_IOU_START
    assert best_gt_iou(near_miss, [CIRCLE]) < AUTO_IOU_END
    monkeypatch.setattr("validator.find_all_disks", lambda _f, _c: [near_miss])

    trainer, state = make_auto_trainer(tmp_path, [CIRCLE], monkeypatch)
    report = trainer.run_series()
    assert report.rewards == 1  # poucas validações → limiar 0.90 → aceite

    state = ValidatorState(tmp_path / "validator_state.json")
    state.reset()
    state.rewards = 500  # muitas validações anteriores → limiar ~0.98 → rejeitado
    trainer, state = make_auto_trainer(tmp_path, [CIRCLE], monkeypatch, state=state)
    report = trainer.run_series()
    assert report.rewards == 0  # quase igual já não passa
    assert report.punishments >= 1  # e é punido (até desaparecer / limite)


def test_auto_trainer_abaixo_de_090_nunca_aceita(tmp_path, monkeypatch):
    below = DiskDetection(110, 100, 50)  # IoU ≈ 0.77 < mínimo 0.90
    monkeypatch.setattr("validator.find_all_disks", lambda _f, _c: [below])

    trainer, state = make_auto_trainer(tmp_path, [CIRCLE], monkeypatch)
    report = trainer.run_series()
    assert report.rewards == 0
    assert report.punishments >= 1


# ------------------------------------------------------- ramos de cobertura --


def test_estado_load_ignora_entradas_de_amostras_malformadas(tmp_path):
    path = tmp_path / "validator_state.json"
    path.write_text(json.dumps({"version": 1, "samples": {"a.jpg": "lixo"}}), encoding="utf-8")
    state = ValidatorState(path)
    state.load()
    assert state.samples == {}


def test_end_round_sem_serie_ativa_nao_faz_nada(tmp_path):
    state = ValidatorState(tmp_path / "validator_state.json")
    state.round = 5
    state.end_round({"score": 50.0})
    assert state.rounds == []


def test_sessao_aplica_rejeicoes_teimosas_punindo(tmp_path):
    session = make_session(tmp_path)
    session.apply_detection([CIRCLE])
    session.reject()
    session.apply_detection([CIRCLE])
    assert session.reevals == 2
    assert session.state.punishments == 2


def test_sessao_julga_sem_pendentes_devolve_completa(tmp_path):
    session = make_session(tmp_path)
    assert session.accept() == "complete"
    assert session.reject() == "complete"


def test_sessao_limite_de_julgamentos_devolve_completa(tmp_path):
    session = make_session(tmp_path)
    shapes = [DiskDetection(x, 50, 5) for x in range(200)]
    session.apply_detection(shapes)
    results = [session.accept() for _ in range(100)]
    assert results[-1] == "complete"
    assert session.judgments == 100


def test_sessao_sem_reavaliaçoes_devolve_completa(tmp_path):
    session = make_session(tmp_path)
    session.reevals = MAX_REEVALS_PER_SAMPLE
    session.apply_detection([CIRCLE])
    assert session.reject() == "complete"


def test_sample_done_text_sem_guia(tmp_path):
    session = make_session(tmp_path)
    session.accepted = [CIRCLE]
    text = sample_done_text(session)
    assert "sem guia" in text
    assert f"{session.sample.label}" in text


def test_auto_trainer_progress_sem_guia_chama_progress(tmp_path, monkeypatch):
    store = CalibrationStore(tmp_path / "calibration.json")
    config = AstroFrameConfig()
    state = ValidatorState(tmp_path / "validator_state.json")
    samples = [make_sample("a.jpg")]
    monkeypatch.setattr("validator.load_frame", lambda _s: np.zeros((100, 100, 3), dtype=np.uint8))
    trainer = AutoTrainer(samples, store, config, state)
    progress_calls = []
    on_detect_calls = []
    report = trainer.run_series(
        progress=lambda *a: progress_calls.append(a),
        on_detect=lambda *a: on_detect_calls.append(a),
    )
    assert progress_calls
    assert on_detect_calls == []
    assert report.samples_done == 1


def test_auto_trainer_on_detect_apos_deteçao_e_reavaliaçao(tmp_path, monkeypatch):
    ghost = DiskDetection(180, 100, 50)  # longe do guia → rejeitado sempre
    monkeypatch.setattr("validator.find_all_disks", lambda _f, _c: [ghost])
    trainer, _state = make_auto_trainer(tmp_path, [CIRCLE], monkeypatch)
    on_detect_calls = []
    report = trainer.run_series(on_detect=lambda *a: on_detect_calls.append(a))
    assert report.samples_done == 1
    assert len(on_detect_calls) >= 2  # deteção inicial + reavaliações
    assert report.punishments >= 1


def test_auto_trainer_erro_de_leitura_registado(tmp_path, monkeypatch):
    trainer, _state = make_auto_trainer(tmp_path, [CIRCLE], monkeypatch)

    def broken_load(_sample):
        raise OSError("ficheiro corrompido")

    monkeypatch.setattr("validator.load_frame", broken_load)
    report = trainer.run_series()
    assert report.samples_done == 0
    assert any("erro ao ler" in error for error in report.errors)


# ------------------------------------------------- CNN de deteção (auto) --


def make_samples_dir(tmp_path: Path, n: int = 2) -> Path:
    """Pasta `samples` com `n` imagens + ground truth (calibration.json)."""
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


def make_patch_pairs(n=6, seed=3):
    rng = np.random.default_rng(seed)
    pos = np.clip(rng.normal(0.6, 0.2, (n, 48, 48)), 0, 1)
    neg = np.clip(rng.normal(0.2, 0.1, (n, 48, 48)), 0, 1)
    return [p.astype(np.float64) for p in pos], [p.astype(np.float64) for p in neg]


def test_estado_v1_para_v2_migra_sem_perda(tmp_path):
    v1 = {"version": 1, "deltas": {"param1": 0.5}, "rewards": 3, "punishments": 1,
          "round": 2, "rounds": [{"round": 1, "mode": "auto"}], "samples": {}}
    path = tmp_path / "state.json"
    path.write_text(json.dumps(v1), encoding="utf-8")
    state = ValidatorState(path)
    assert state.deltas == {"param1": 0.5}
    assert state.rewards == 3 and state.round == 2
    assert state.cnn_positives == 0 and state.cnn_series == []
    state.save()
    state2 = ValidatorState(path)
    assert state2.deltas == {"param1": 0.5}


def test_estado_versao_desconhecida_ignorada(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": 99, "deltas": {"x": 1.0}}), encoding="utf-8")
    state = ValidatorState(path)
    assert state.deltas == {}


def test_estado_guardar_e_ler_campos_cnn(tmp_path):
    state = ValidatorState(tmp_path / "state.json")
    state.cnn_positives = 5
    state.cnn_negatives = 3
    state.cnn_series = [{"round": 1, "accuracy": 0.9, "promoted": True}]
    state.save()
    loaded = ValidatorState(tmp_path / "state.json")
    assert loaded.cnn_positives == 5
    assert loaded.cnn_negatives == 3
    assert loaded.cnn_series[0]["accuracy"] == 0.9


def test_estado_reset_limpa_campos_cnn(tmp_path):
    state = ValidatorState(tmp_path / "state.json")
    state.cnn_positives = 9
    state.reset()
    loaded = ValidatorState(tmp_path / "state.json")
    assert loaded.cnn_positives == 0 and loaded.cnn_series == []


def test_auto_trainer_recolhe_patches_positivos_e_negativos(tmp_path, monkeypatch):
    monkeypatch.setattr("validator.find_all_disks", lambda _f, _c: [CIRCLE])
    store = CalibrationStore(tmp_path / "calibration.json")
    store.items["a.jpg"] = CalibrationItem("a.jpg", "image", None, 640, 480, [CIRCLE])
    config = AstroFrameConfig()
    state = ValidatorState(tmp_path / "validator_state.json")
    samples = [make_sample("a.jpg")]
    monkeypatch.setattr("validator.load_frame", lambda _s: np.zeros((200, 200, 3), dtype=np.uint8))
    trainer = AutoTrainer(samples, store, config, state)
    report = trainer.run_series()
    assert len(trainer.positives) == 1
    assert trainer.positives[0].shape == (48, 48)
    assert len(trainer.negatives) >= 1
    assert report.cnn_positives == 1
    assert report.cnn_negatives == len(trainer.negatives)
    assert state.cnn_positives == 0  # contagens só no fim, via train_classifier_round


def test_auto_trainer_sem_recolha_de_patches(tmp_path, monkeypatch):
    monkeypatch.setattr("validator.find_all_disks", lambda _f, _c: [CIRCLE])
    store = CalibrationStore(tmp_path / "calibration.json")
    store.items["a.jpg"] = CalibrationItem("a.jpg", "image", None, 640, 480, [CIRCLE])
    config = AstroFrameConfig()
    state = ValidatorState(tmp_path / "validator_state.json")
    samples = [make_sample("a.jpg")]
    monkeypatch.setattr("validator.load_frame", lambda _s: np.zeros((100, 100, 3), dtype=np.uint8))
    trainer = AutoTrainer(samples, store, config, state, collect_patches=False)
    report = trainer.run_series()
    assert trainer.positives == [] and trainer.negatives == []
    assert report.cnn_positives == 0 and report.cnn_negatives == 0


def test_auto_trainer_filtra_com_cnn_sem_esvaziar(tmp_path, monkeypatch):
    model, _ = fit_classifier(
        make_patch_pairs()[0], make_patch_pairs()[1], epochs=2, seed=4
    )
    model_path = tmp_path / "filter.npz"
    model.save(model_path)
    monkeypatch.setattr("validator.find_all_disks", lambda _f, _c: [CIRCLE])
    store = CalibrationStore(tmp_path / "calibration.json")
    store.items["a.jpg"] = CalibrationItem("a.jpg", "image", None, 640, 480, [CIRCLE])
    config = AstroFrameConfig()
    state = ValidatorState(tmp_path / "validator_state.json")
    samples = [make_sample("a.jpg")]
    monkeypatch.setattr("validator.load_frame", lambda _s: np.zeros((100, 100, 3), dtype=np.uint8))
    trainer = AutoTrainer(
        samples, store, config, state, cnn_model_path=model_path, cnn_threshold=0.99
    )
    assert trainer._filter is not None
    report = trainer.run_series()
    assert report.samples_done == 1  # filtro nunca esvazia → julgamento normal


def test_auto_trainer_cnn_model_path_inexistente(tmp_path, monkeypatch):
    monkeypatch.setattr("validator.find_all_disks", lambda _f, _c: [CIRCLE])
    trainer, state = make_auto_trainer(tmp_path, [CIRCLE], monkeypatch)
    trainer2 = AutoTrainer(
        trainer.samples, trainer.store, trainer.config, state,
        cnn_model_path=tmp_path / "ausente.npz",
    )
    assert trainer2._filter is None


def test_classifier_accuracy_metricas():
    pos, neg = make_patch_pairs(8)
    model, _ = fit_classifier(pos, neg, epochs=3, seed=5)
    acc = classifier_accuracy(model, pos, neg, 5)
    assert 0.0 <= acc <= 1.0
    assert classifier_accuracy(model, [], [], 5) == 0.0


def test_train_classifier_round_sem_patches_suficientes(tmp_path, monkeypatch):
    db = FeedbackDB(tmp_path / "fb.db")
    state = ValidatorState(tmp_path / "state.json")
    result = train_classifier_round([], [], state, 1, None, db=db)
    assert result == {"skipped": True}
    assert db.logs(component="validator")[0]["message"].startswith("Série 1: sem patches")


def test_train_classifier_round_promove_e_grava(tmp_path):
    db = FeedbackDB(tmp_path / "fb.db")
    state = ValidatorState(tmp_path / "state.json")
    pos, neg = make_patch_pairs(6)
    result = train_classifier_round(pos, neg, state, 1, score=90.0, epochs=3, db=db)
    assert result["skipped"] is False
    assert result["promoted"] is True
    assert validator.CNN_CANONICAL_PATH.exists()
    assert state.cnn_series[0]["round"] == 1
    assert state.cnn_series[0]["score"] == 90.0
    assert db.champion("disk_filter")["path"] == str(result["staged"])


def test_train_classifier_round_mantem_campeao_quando_pior(tmp_path):
    db = FeedbackDB(tmp_path / "fb.db")
    state = ValidatorState(tmp_path / "state.json")
    pos, neg = make_patch_pairs(6, seed=8)
    first = train_classifier_round(pos, neg, state, 1, score=90.0, epochs=3, db=db)
    second = train_classifier_round(pos, neg, state, 2, score=80.0, epochs=3, db=db)
    assert first["promoted"] is True
    assert second["promoted"] is False
    assert second["champion_path"] == str(first["staged"])
    assert state.cnn_series[-1]["promoted"] is False


def test_train_classifier_round_sem_score_usa_accuracy(tmp_path):
    db = FeedbackDB(tmp_path / "fb.db")
    state = ValidatorState(tmp_path / "state.json")
    pos, neg = make_patch_pairs(6, seed=9)
    result = train_classifier_round(pos, neg, state, 1, score=None, epochs=3, db=db)
    assert result["skipped"] is False
    assert db.champion("disk_filter")["metrics"]["score"] == 0.0
    assert db.champion("disk_filter")["metrics"]["accuracy"] > 0.0


def test_run_auto_headless_cnn_ligada(tmp_path, capsys):
    root = make_samples_dir(tmp_path)
    assert run_auto_headless(str(root), series=1, epochs=2) == 0
    out = capsys.readouterr().out
    assert "CNN ligada" in out
    assert "patches CNN recolhidos" in out
    state = ValidatorState(train_dir() / validator.DEFAULT_STATE_NAME)
    assert len(state.cnn_series) == 1


def test_run_auto_headless_cnn_desligada(tmp_path, capsys):
    root = make_samples_dir(tmp_path)
    assert run_auto_headless(str(root), series=1, epochs=2, cnn=False) == 0
    out = capsys.readouterr().out
    assert "CNN desligada" in out
    state = ValidatorState(train_dir() / validator.DEFAULT_STATE_NAME)
    assert state.cnn_series == []


def test_run_auto_headless_sem_amostras(tmp_path):
    root = tmp_path / "samples"
    root.mkdir()
    assert run_auto_headless(str(root), series=1, epochs=2) == 0
    assert not validator.CNN_CANONICAL_PATH.exists()


def test_cli_auto_com_flags_cnn(tmp_path, capsys):
    root = make_samples_dir(tmp_path)
    assert main(["--samples", str(root), "--auto", "--series", "1", "--epochs", "2",
                 "--cnn-off", "--cnn-threshold", "0.4"]) == 0
    assert "CNN desligada" in capsys.readouterr().out


def test_apply_state_weights_ignora_delta_desconhecido(tmp_path):
    state = ValidatorState(tmp_path / "state.json")
    state.deltas = {"param_inexistente": 1.0, "denoise.h": 0.5}
    apply_state_weights(state)
    assert "denoise.h" in state.deltas


def test_random_negatives_frame_pequena_nao_recolhe(tmp_path, monkeypatch):
    monkeypatch.setattr("validator.find_all_disks", lambda _f, _c: [])
    trainer, state = make_auto_trainer(tmp_path, [], monkeypatch)
    trainer._random_negatives(np.zeros((20, 20, 3), dtype=np.uint8), [], "mini")
    assert trainer.negatives == []


def test_random_negatives_descarta_crops_que_tocam_o_guia(tmp_path, monkeypatch):
    monkeypatch.setattr("validator.find_all_disks", lambda _f, _c: [])
    trainer, state = make_auto_trainer(tmp_path, [], monkeypatch)
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    gt = [DiskDetection(cx, cy, 8) for cy in range(8, 96, 9) for cx in range(8, 96, 9)]
    trainer._random_negatives(frame, gt, "densa")
    assert trainer.negatives == []


def test_export_trained_ignora_delta_desconhecido(tmp_path):
    state = ValidatorState(tmp_path / "state.json")
    state.deltas = {"param_inexistente": 1.0, "denoise.h": 0.5}
    path = export_trained(state, AstroFrameConfig(), tmp_path / "out.json")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["deltas"] == {"param_inexistente": 1.0, "denoise.h": 0.5}
