"""Testes da configuração externa (YAML)."""

from __future__ import annotations

import logging

import pytest

from astroframe.config import AstroFrameConfig


def test_round_trip_yaml(tmp_path):
    cfg = AstroFrameConfig()
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)
    loaded = AstroFrameConfig.from_yaml(path)
    assert loaded.to_dict() == cfg.to_dict()


def test_override_parcial(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("clahe:\n  clip_limit: 4.5\n", encoding="utf-8")
    loaded = AstroFrameConfig.from_yaml(path)
    assert loaded.clahe.clip_limit == pytest.approx(4.5)
    assert loaded.clahe.tile_grid_size == 8
    assert loaded.unsharp.amount == pytest.approx(0.5)


def test_numero_inteiro_coage_para_float(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("unsharp:\n  sigma: 3\n", encoding="utf-8")
    assert isinstance(AstroFrameConfig.from_yaml(path).unsharp.sigma, float)


def test_min_sharpness_null_mantem_se_none(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("lucky:\n  min_sharpness: null\n", encoding="utf-8")
    assert AstroFrameConfig.from_yaml(path).lucky.min_sharpness is None


def test_campo_opcional_aceita_numero(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("lucky:\n  min_sharpness: 30.5\n", encoding="utf-8")
    assert AstroFrameConfig.from_yaml(path).lucky.min_sharpness == pytest.approx(30.5)


def test_campo_opcional_com_tipo_invalido_avisa(tmp_path, caplog):
    path = tmp_path / "config.yaml"
    path.write_text("lucky:\n  min_sharpness: 'abc'\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        cfg = AstroFrameConfig.from_yaml(path)
    assert cfg.lucky.min_sharpness == "abc"
    assert any("min_sharpness" in record.getMessage() for record in caplog.records)


def test_campo_nao_opcional_null_avisa_e_mantem(tmp_path, caplog):
    path = tmp_path / "config.yaml"
    path.write_text("denoise:\n  h: null\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        cfg = AstroFrameConfig.from_yaml(path)
    assert cfg.denoise.h is None
    assert any("tipo inesperado" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize(
    "yaml_text, key",
    [
        ("clahe:\n  clip_limit: 'abc'\n", "clip_limit"),
        ("xpto: 1\n", "xpto"),
    ],
)
def test_yaml_invalido_avisa_mas_nao_crasha(tmp_path, yaml_text, key, caplog):
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        cfg = AstroFrameConfig.from_yaml(path)
    assert any(key in record.getMessage() for record in caplog.records)
    assert isinstance(cfg, AstroFrameConfig)


# ------------------------------------------------------------ tuning + AI --


def test_tuning_config_por_omissao_desligado():
    cfg = AstroFrameConfig()
    assert cfg.tuning.enabled is False
    assert cfg.tuning.budget_s == 60.0
    assert cfg.tuning.seed == 42
    assert cfg.tuning.anneal is True
    assert cfg.tuning.params is None


def test_ai_config_por_omissao_tudo_desligado():
    cfg = AstroFrameConfig()
    assert cfg.ai.backend == "numpy"
    assert cfg.ai.lstm_trajectory is False
    assert cfg.ai.cnn_enhance is False
    assert cfg.ai.disk_filter == 0.0


def test_yaml_round_trip_com_tuning_e_ai(tmp_path):
    cfg = AstroFrameConfig()
    cfg.tuning.budget_s = 12.5
    cfg.tuning.params = ["clahe.clip_limit", "denoise.h"]
    cfg.ai.lstm_trajectory = True
    cfg.ai.disk_filter = 0.85
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)
    loaded = AstroFrameConfig.from_yaml(path)
    assert loaded.tuning.budget_s == pytest.approx(12.5)
    assert loaded.tuning.params == ["clahe.clip_limit", "denoise.h"]
    assert loaded.ai.lstm_trajectory is True
    assert loaded.ai.disk_filter == pytest.approx(0.85)


def test_override_parcial_de_tuning_e_ai(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "tuning:\n  budget_s: 30\nai:\n  cnn_enhance: true\n", encoding="utf-8"
    )
    loaded = AstroFrameConfig.from_yaml(path)
    assert loaded.tuning.budget_s == pytest.approx(30.0)
    assert loaded.tuning.enabled is False
    assert loaded.ai.cnn_enhance is True
    assert loaded.ai.lstm_trajectory is False
