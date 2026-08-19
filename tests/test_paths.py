"""Testes da estrutura `Logs/` (`astroframe.paths`)."""

from __future__ import annotations

import logging
import shutil

import pytest

from astroframe import paths


def _write(path, content: bytes = b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_data_root_usa_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ASTROFRAME_DATA_DIR", str(tmp_path / "dados"))
    assert paths.data_root() == tmp_path / "dados"


def test_data_root_por_omissao_e_logs_do_projeto(monkeypatch):
    monkeypatch.delenv("ASTROFRAME_DATA_DIR", raising=False)
    assert paths.data_root().name == "Logs"
    assert paths.data_root().is_absolute()


def test_acessores_criam_a_estrutura(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTROFRAME_DATA_DIR", str(tmp_path))
    assert paths.logs_ia_dir() == tmp_path / "logs" / "ia"
    assert paths.logs_system_dir() == tmp_path / "logs" / "system"
    assert paths.train_dir() == tmp_path / "train"
    assert paths.weights_dir() == tmp_path / "weights"
    assert paths.staging_dir() == tmp_path / "weights" / "staging"
    assert paths.feedback_db_path() == tmp_path / "logs" / "system" / "feedback.db"
    for sub in ("logs/ia", "logs/system", "train", "weights", "weights/staging"):
        assert (tmp_path / sub).is_dir()


def test_calibration_json_global_por_omissao(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTROFRAME_DATA_DIR", str(tmp_path))
    _write(tmp_path / "train" / "calibration.json")
    samples = tmp_path / "samples"
    samples.mkdir()
    _write(samples / "calibration.json")
    assert paths.calibration_json(str(samples)) == tmp_path / "train" / "calibration.json"


def test_calibration_json_cai_para_samples_sem_global(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTROFRAME_DATA_DIR", str(tmp_path))
    samples = tmp_path / "samples"
    _write(samples / "calibration.json")
    assert paths.calibration_json(str(samples)) == samples / "calibration.json"


def test_calibration_json_sem_global_nem_local(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTROFRAME_DATA_DIR", str(tmp_path))
    samples = tmp_path / "samples"
    samples.mkdir()
    assert paths.calibration_json(str(samples)) == tmp_path / "train" / "calibration.json"


def test_migrate_legacy_copia_astroframe(monkeypatch, tmp_path):
    monkeypatch.setenv("ASTROFRAME_DATA_DIR", str(tmp_path))
    legacy = tmp_path / "legacy"
    _write(legacy / "disk_filter.npz")
    _write(legacy / "enhancer_cnn.npz")
    _write(legacy / "feedback.db")
    monkeypatch.setattr(paths, "_LEGACY_DIR", legacy)
    paths.migrate_legacy()
    assert (tmp_path / "weights" / "disk_filter.npz").exists()
    assert (tmp_path / "weights" / "enhancer_cnn.npz").exists()
    assert (tmp_path / "logs" / "system" / "feedback.db").exists()


def test_migrate_legacy_nao_duplica_pesos(monkeypatch, tmp_path):
    monkeypatch.setenv("ASTROFRAME_DATA_DIR", str(tmp_path))
    _write(tmp_path / "weights" / "disk_filter.npz")
    legacy = tmp_path / "legacy"
    _write(legacy / "enhancer_cnn.npz")
    _write(legacy / "feedback.db")
    monkeypatch.setattr(paths, "_LEGACY_DIR", legacy)
    paths.migrate_legacy()
    assert not (tmp_path / "weights" / "enhancer_cnn.npz").exists()
    assert not (tmp_path / "logs" / "system" / "feedback.db").exists()


def test_migrate_legacy_copia_calibration_de_samples(monkeypatch, tmp_path):
    monkeypatch.setenv("ASTROFRAME_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(paths, "_LEGACY_DIR", tmp_path / "inexistente")
    samples = paths._REPO_ROOT / "samples"
    original = samples / "calibration.json"
    if not original.exists():
        pytest.skip("sem samples/calibration.json no repositório")
    monkeypatch.setattr(paths, "_REPO_ROOT", tmp_path)
    shutil.copy2(original, tmp_path / "samples_cal_tmp.json")
    (tmp_path / "samples").mkdir(exist_ok=True)
    shutil.copy2(tmp_path / "samples_cal_tmp.json", tmp_path / "samples" / "calibration.json")
    (tmp_path / "samples_cal_tmp.json").unlink()
    paths.migrate_legacy()
    assert (tmp_path / "train" / "calibration.json").exists()


def test_migrate_legacy_sem_origem_nao_falha(monkeypatch, tmp_path):
    monkeypatch.setenv("ASTROFRAME_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(paths, "_LEGACY_DIR", tmp_path / "inexistente")
    paths.migrate_legacy()


def test_setup_logging_escreve_no_ficheiro(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTROFRAME_DATA_DIR", str(tmp_path))
    root = logging.getLogger()
    before = len(root.handlers)
    try:
        paths.setup_logging("teste.log")
        logging.getLogger("astroframe.paths").info("mensagem de teste")
        handler = root.handlers[-1]
        handler.flush()
        content = (tmp_path / "logs" / "system" / "teste.log").read_text(encoding="utf-8")
        assert "mensagem de teste" in content
    finally:
        root.removeHandler(root.handlers[-1])
        root.setLevel(root.level)
    assert len(root.handlers) == before
