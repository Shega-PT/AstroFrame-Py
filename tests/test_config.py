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
