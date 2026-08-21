"""Testes do registry unificado de parâmetros (fonte única de limites/passos)."""

from __future__ import annotations

import pytest
from validator import DELTA_BOUNDS, PUNISH_DELTAS, REWARD_DELTAS

from astroframe.ai import params as pparams
from astroframe.ai.feedback import _PARAM_BOUNDS
from astroframe.config import AstroFrameConfig


def test_todos_os_caminhos_sao_validos_na_config():
    cfg = AstroFrameConfig()
    for path in pparams.PARAM_SPECS:
        section, name = path.split(".")
        assert hasattr(getattr(cfg, section), name), path


def test_grupos_cobertos():
    grupos = {spec.group for spec in pparams.specs()}
    assert {"detect", "geometry", "enhance", "stack", "polish", "score", "meta"} <= grupos


def test_spec_por_nome_e_por_caminho_equivalentes():
    spec = pparams.spec("stabilizer.param2")
    assert pparams.spec_by_name("param2") is spec
    assert spec.name == "param2"
    assert spec.path == "stabilizer.param2"


def test_bounds_e_step_consistentes():
    for path, spec in pparams.PARAM_SPECS.items():
        assert pparams.bounds(path) == (spec.low, spec.high)
        assert pparams.step(path) == spec.step
        assert spec.low <= spec.high
        assert spec.step > 0


def test_clamp_float_limita_a_gama():
    assert pparams.clamp_value("clahe.clip_limit", 100.0) == 6.0
    assert pparams.clamp_value("clahe.clip_limit", -10.0) == 0.5
    assert isinstance(pparams.clamp_value("clahe.clip_limit", 3.0), float)


def test_clamp_int_arredonda():
    assert pparams.clamp_value("stabilizer.param2", 12.6) == 13
    assert isinstance(pparams.clamp_value("stabilizer.param2", 12.6), int)


def test_clamp_impar_forca_kernel_impar():
    assert pparams.clamp_value("stabilizer.gaussian_kernel_size", 12) % 2 == 1
    assert pparams.clamp_value("stabilizer.gaussian_kernel_size", 13) % 2 == 1


def test_get_set_param():
    cfg = AstroFrameConfig()
    pparams.set_param(cfg, "unsharp.amount", 1.5)
    assert pparams.get_param(cfg, "unsharp.amount") == pytest.approx(1.5)


def test_apply_deltas_nao_muta_a_original_e_faz_clamp():
    cfg = AstroFrameConfig()
    cfg.clahe.clip_limit = 6.0
    adjusted = pparams.apply_deltas(cfg, {"clahe.clip_limit": 3.0})
    assert cfg.clahe.clip_limit == pytest.approx(6.0)
    assert adjusted.clahe.clip_limit == pytest.approx(6.0)


def test_apply_deltas_ignora_caminhos_desconhecidos():
    cfg = AstroFrameConfig()
    adjusted = pparams.apply_deltas(cfg, {"nao.existe": 1.0, "param2": 5.0})
    assert adjusted == cfg


def test_apply_deltas_arredonda_ints():
    cfg = AstroFrameConfig()
    adjusted = pparams.apply_deltas(cfg, {"stabilizer.param2": 0.5})
    assert isinstance(adjusted.stabilizer.param2, int)


def test_deltas_dict_apenas_dos_alterados():
    cfg = AstroFrameConfig()
    cfg.unsharp.amount = 0.9
    cfg.stabilizer.param2 = 42
    deltas = pparams.deltas_dict(cfg, ["unsharp.amount", "stabilizer.param2", "denoise.h"])
    assert deltas["unsharp.amount"] == pytest.approx(0.4)
    assert deltas["stabilizer.param2"] == 42 - AstroFrameConfig().stabilizer.param2
    assert "denoise.h" not in deltas


def test_deltas_default_do_validator_coincidem_com_o_registry():
    assert PUNISH_DELTAS == pparams.default_punish_deltas()
    assert REWARD_DELTAS == pparams.default_reward_deltas()


def test_limites_do_validator_coincidem_com_o_registry():
    for spec in pparams.specs("detect"):
        low, high = DELTA_BOUNDS[spec.name]
        assert (spec.low, spec.high) == pytest.approx((low, high))


def test_feedback_bounds_coincidem_com_o_registry():
    for path, (low, high) in _PARAM_BOUNDS.items():
        assert pparams.bounds(path) == pytest.approx((low, high))


def test_feedback_params_todos_registados():
    for path in pparams.FEEDBACK_PARAMS:
        assert path in pparams.PARAM_SPECS


def test_int_params_coerentes_com_as_specs():
    for path in pparams.INT_PARAMS:
        assert pparams.PARAM_SPECS[path].dtype is int


def test_specs_detect_tem_exatamente_os_cinco_treinaveis():
    detect = pparams.specs("detect")
    assert {spec.name for spec in detect} == set(PUNISH_DELTAS)
    assert len(detect) == 5
