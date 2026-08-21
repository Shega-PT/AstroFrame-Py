"""Testes da interface de calibração (handlers puros + montagem Gradio)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from astroframe.calibration.circles import circles_to_layers
from astroframe.calibration.scan import scan_samples
from astroframe.calibration.store import CalibrationItem, CalibrationStore
from astroframe.config import AstroFrameConfig
from astroframe.core.stabilizer import DiskDetection
from astroframe.ui.calibration_app import (
    _find_sample,
    auto_detect_payload,
    build_calibration_app,
    load_item_payload,
    run,
    save_item_circles,
    validate_all_report,
)


def _make_sample_dir(tmp_path) -> Path:
    root = tmp_path / "samples"
    (root / "images").mkdir(parents=True)
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    cv2.circle(image, (60, 50), 30, (200, 200, 200), -1)
    cv2.imwrite(str(root / "images" / "eclipse.jpg"), image)
    return root


def _sample_key(root: Path) -> str:
    return scan_samples(root)[0].key


def test_find_sample_existente_e_inexistente(tmp_path):
    root = _make_sample_dir(tmp_path)
    samples = scan_samples(root)
    assert _find_sample(samples, "images/eclipse.jpg").kind == "image"
    with pytest.raises(KeyError):
        _find_sample(samples, "desconhecido.jpg")


def test_load_item_payload_sem_chave(tmp_path):
    value, info = load_item_payload(None, str(_make_sample_dir(tmp_path)))
    assert value == {}
    assert "Escolhe uma amostra" in info


def test_load_item_payload_deteccao_automatica(tmp_path, monkeypatch):
    root = _make_sample_dir(tmp_path)
    monkeypatch.setattr(
        "astroframe.ui.calibration_app.find_disks_for_calibration",
        lambda frame, config=None, expected_n=None: [DiskDetection(60, 50, 30)],
    )
    value, info = load_item_payload(_sample_key(root), str(root))
    assert len(value["layers"]) == 1
    assert "deteção automática" in info


def test_load_item_payload_usa_ground_truth_guardado(tmp_path, monkeypatch):
    root = _make_sample_dir(tmp_path)
    store = CalibrationStore(root / "calibration.json")
    store.upsert_item(
        _sample_key(root),
        CalibrationItem("images/eclipse.jpg", "image", None, 120, 100, [DiskDetection(55, 55, 20)]),
    )
    calls = {"n": 0}
    monkeypatch.setattr(
        "astroframe.ui.calibration_app.find_disks_for_calibration",
        lambda frame, config=None, expected_n=None: calls.__setitem__("n", calls["n"] + 1) or [],
    )
    value, info = load_item_payload(_sample_key(root), str(root), store=store)
    assert calls["n"] == 0
    assert len(value["layers"]) == 1
    assert "guardado" in info


def test_load_item_payload_amostra_desconhecida(tmp_path):
    with pytest.raises(KeyError):
        load_item_payload("nao.existe", str(_make_sample_dir(tmp_path)))


def test_auto_detect_payload(tmp_path, monkeypatch):
    root = _make_sample_dir(tmp_path)
    monkeypatch.setattr(
        "astroframe.ui.calibration_app.find_disks_for_calibration",
        lambda frame, config=None, expected_n=None: [DiskDetection(60, 50, 30), DiskDetection(10, 10, 5)],
    )
    value, info = auto_detect_payload(_sample_key(root), str(root))
    assert len(value["layers"]) == 2
    assert "2 círculo(s) detetado(s)" in info


def test_auto_detect_payload_sem_chave(tmp_path):
    value, info = auto_detect_payload(None, str(_make_sample_dir(tmp_path)))
    assert value == {}
    assert "Escolhe uma amostra" in info


def test_save_item_circles_grava_ground_truth(tmp_path):
    root = _make_sample_dir(tmp_path)
    store = CalibrationStore(root / "calibration.json")
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    value = circles_to_layers(frame, [DiskDetection(40, 40, 15)])
    msg = save_item_circles(value, _sample_key(root), str(root), store=store)
    assert "1 círculo(s)" in msg
    stored = store.get_item(_sample_key(root))
    assert stored.circles == [DiskDetection(40, 40, 15)]
    assert stored.width == 120 and stored.height == 100
    assert stored.kind == "image" and stored.frame is None


def test_save_item_circles_erros(tmp_path):
    root = _make_sample_dir(tmp_path)
    store = CalibrationStore(root / "calibration.json")
    assert "Escolhe uma amostra" in save_item_circles(None, None, str(root), store=store)
    assert "Sem imagem" in save_item_circles(None, _sample_key(root), str(root), store=store)
    assert "tenta novamente" in save_item_circles(
        {"background": None, "layers": None}, _sample_key(root), str(root), store=store
    )


def test_validate_all_report_sem_ground_truth(tmp_path, monkeypatch):
    root = _make_sample_dir(tmp_path)

    def _no_disks(frame, config=None, expected_n=None):
        return []

    monkeypatch.setattr(
        "astroframe.ui.calibration_app.find_disks_for_calibration",
        _no_disks,
    )
    rows, summary, suggestions = validate_all_report(str(root))
    assert rows[0][3] == "—"
    assert "Sem ground truth para validar" in summary
    assert "Guardar ajustes" in suggestions


def test_validate_all_report_com_ground_truth(tmp_path, monkeypatch):
    root = _make_sample_dir(tmp_path)
    store = CalibrationStore(root / "calibration.json")
    store.upsert_item(
        _sample_key(root),
        CalibrationItem("images/eclipse.jpg", "image", None, 120, 100, [DiskDetection(60, 50, 30)]),
    )
    monkeypatch.setattr(
        "astroframe.ui.calibration_app.find_disks_for_calibration",
        lambda frame, config=None, expected_n=None: [DiskDetection(61, 50, 29)],
    )
    rows, summary, suggestions = validate_all_report(str(root), store=store)
    assert len(rows) == 1
    assert rows[0][1] == 1 and rows[0][2] == 1 and rows[0][3] == 1
    assert rows[0][6] != "—"
    assert "Score de calibração" in summary
    assert "Sem ajustes sugeridos" in suggestions


def test_build_calibration_app_popula_amostras(tmp_path):
    root = _make_sample_dir(tmp_path)
    app = build_calibration_app(samples_dir=str(root))
    assert app is not None


def test_build_calibration_app_pasta_vazia(tmp_path):
    app = build_calibration_app(samples_dir=str(tmp_path / "vazio"))
    assert app is not None


def test_run_lanca_gradio(monkeypatch, tmp_path):
    launched = {}

    class FakeBlocks:
        def launch(self, **kwargs):
            launched.update(kwargs)
            return None

    monkeypatch.setattr("astroframe.ui.calibration_app.build_calibration_app", lambda **kw: FakeBlocks())
    run(samples_dir=str(tmp_path), port=7899)
    assert launched["server_port"] == 7899


def test_run_com_config_path(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    AstroFrameConfig().to_yaml(config_path)
    built = {}

    class FakeBlocks:
        def launch(self, **kwargs):
            return None

    def fake_build(samples_dir, config):
        built["config"] = config
        return FakeBlocks()

    monkeypatch.setattr("astroframe.ui.calibration_app.build_calibration_app", fake_build)
    run(samples_dir="samples", config_path=str(config_path))
    assert built["config"] is not None
