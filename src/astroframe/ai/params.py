"""Registry unificado dos parâmetros ajustáveis da pipeline.

Fonte única de verdade para os **limites seguros**, passos de otimização e
deltas de treino de todos os parâmetros do AstroFrame. Antes os limites
estavam espalhados (`validator.py` para a deteção, `feedback.py` para a
melhoria); agora o `ai.tuner`, o `validator` e o `feedback` leem todos do
mesmo sítio — impossível divergirem.

Cada parâmetro tem:

- `path` — caminho na config (`"stabilizer.param2"`);
- `low`/`high` — gama segura (clamp: a aprendizagem nunca sai daqui);
- `step` — passo inicial do hill-climbing do tuner;
- `dtype` — `int` ou `float` (ints são arredondados);
- `odd` — True quando o valor tem de ser ímpar (kernels gaussianos);
- `group` — agrupamento: `detect` (os 7 treináveis do validator),
  `geometry`, `enhance`, `stack`, `polish`, `score`, `meta`;
- `costly` — True quando avaliar o parâmetro é caro (denoising);
- `punish`/`reward` — deltas do treino por recompensa/punição do validator
  (só usados no grupo `detect`).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from astroframe.config import AstroFrameConfig

_INT = int
_FLOAT = float


@dataclass(frozen=True)
class ParamSpec:
    """Especificação declarativa de um parâmetro ajustável."""

    path: str
    low: float
    high: float
    step: float
    dtype: type = _FLOAT
    odd: bool = False
    group: str = "enhance"
    costly: bool = False
    punish: float = 0.0
    reward: float = 0.0

    @property
    def name(self) -> str:
        """Nome curto do parâmetro (o campo dentro da secção)."""
        return self.path.rsplit(".", 1)[1]


_SPECS: list[ParamSpec] = [
    # -- deteção: os 7 parâmetros treináveis do validator (pesos do Hough,
    # desfoque e tolerância a discos ocultos). Punish = forma rejeitada;
    # reward = forma válida (relaxa 25% para não ficar demasiado estrito).
    ParamSpec("stabilizer.param2", 1.0, 200.0, 1.0, _INT, group="detect", punish=1.0, reward=-0.25),
    ParamSpec("stabilizer.param1", 5.0, 500.0, 2.0, _INT, group="detect", punish=2.0, reward=-0.5),
    ParamSpec("stabilizer.dp", 0.5, 3.0, 0.05, group="detect", punish=0.05, reward=-0.0125),
    ParamSpec(
        "stabilizer.gaussian_kernel_size",
        1.0,
        31.0,
        2.0,
        _INT,
        odd=True,
        group="detect",
        punish=2.0,
        reward=-0.5,
    ),
    ParamSpec(
        "stabilizer.gaussian_sigma",
        0.5,
        10.0,
        0.25,
        group="detect",
        punish=0.25,
        reward=-0.0625,
    ),
    # -- geometria: derivados da resolução/limites físicos (nunca treinados
    # por recompensa/punição; só limitados ao exportar).
    ParamSpec("stabilizer.max_radius", 50.0, 5000.0, 25.0, _INT, group="geometry"),
    ParamSpec("stabilizer.max_disks", 1.0, 16.0, 1.0, _INT, group="geometry"),
    ParamSpec("stabilizer.jitter_alpha", 0.05, 0.95, 0.05, group="geometry"),
    # -- melhoria: CLAHE + denoising + nitidez + lucky imaging.
    ParamSpec("clahe.clip_limit", 0.5, 6.0, 0.2, group="enhance"),
    ParamSpec("clahe.tile_grid_size", 2.0, 32.0, 2.0, _INT, group="enhance"),
    ParamSpec("denoise.h", 1.0, 20.0, 1.0, group="enhance", costly=True),
    ParamSpec("denoise.template_window_size", 3.0, 15.0, 2.0, _INT, odd=True, group="enhance", costly=True),
    ParamSpec("denoise.search_window_size", 5.0, 35.0, 4.0, _INT, odd=True, group="enhance", costly=True),
    ParamSpec("unsharp.sigma", 0.5, 10.0, 0.5, group="enhance"),
    ParamSpec("unsharp.amount", 0.0, 2.0, 0.1, group="enhance"),
    ParamSpec("lucky.sharpness_percentile", 5.0, 95.0, 5.0, group="enhance"),
    ParamSpec("lucky.gaussian_kernel_size", 3.0, 15.0, 2.0, _INT, odd=True, group="enhance"),
    ParamSpec("lucky.gaussian_sigma", 0.5, 5.0, 0.25, group="enhance"),
    # -- stacking.
    ParamSpec("stacking.n_best", 1.0, 100.0, 5.0, _INT, group="stack"),
    # -- polimento.
    ParamSpec("polish.corona_scale", 1.0, 3.0, 0.05, group="polish"),
    ParamSpec("polish.feather", 0.0, 0.1, 0.005, group="polish"),
    ParamSpec("polish.brightness", 0.0, 0.5, 0.02, group="polish"),
    ParamSpec("polish.reflection_min_radius", 2.0, 100.0, 4.0, _INT, group="polish"),
    # -- pesos da avaliação (0–5 estrelas).
    ParamSpec("score.background_weight", 0.0, 1.0, 0.05, group="score"),
    ParamSpec("score.limb_weight", 0.0, 1.0, 0.05, group="score"),
    ParamSpec("score.noise_weight", 0.0, 1.0, 0.05, group="score"),
    ParamSpec("score.contrast_weight", 0.0, 1.0, 0.05, group="score"),
    ParamSpec("score.reflection_weight", 0.0, 1.0, 0.05, group="score"),
    ParamSpec("score.limb_min_dark", 0.05, 0.5, 0.02, group="score"),
    ParamSpec("score.limb_gain", 1.0, 8.0, 0.25, group="score"),
    ParamSpec("score.edge_radius", 1.0, 1.2, 0.01, group="score"),
    # -- meta-aprendizagem (taxa do feedback por estrelas).
    ParamSpec("feedback.learning_rate", 0.05, 0.9, 0.05, group="meta"),
]

PARAM_SPECS: dict[str, ParamSpec] = {spec.path: spec for spec in _SPECS}

_BY_NAME: dict[str, ParamSpec] = {spec.name: spec for spec in _SPECS}

# Parâmetros que o feedback por estrelas ajusta (os mesmos de sempre).
FEEDBACK_PARAMS = (
    "clahe.clip_limit",
    "denoise.h",
    "unsharp.amount",
    "polish.corona_scale",
    "polish.feather",
)

# Parâmetros de valor inteiro (os restantes são floats).
INT_PARAMS: frozenset[str] = frozenset(path for path, spec in PARAM_SPECS.items() if spec.dtype is _INT)


def specs(group: str | None = None) -> list[ParamSpec]:
    """Parâmetros registados, na ordem de declaração (opcionalmente por grupo)."""
    if group is None:
        return list(_SPECS)
    return [spec for spec in _SPECS if spec.group == group]


def spec(path: str) -> ParamSpec:
    """Especificação de um parâmetro pelo caminho completo (`section.field`)."""
    return PARAM_SPECS[path]


def spec_by_name(name: str) -> ParamSpec:
    """Especificação de um parâmetro pelo nome curto (ex.: `param2`)."""
    return _BY_NAME[name]


def bounds(path: str) -> tuple[float, float]:
    """Gama segura (low, high) do parâmetro."""
    registered = PARAM_SPECS[path]
    return registered.low, registered.high


def step(path: str) -> float:
    """Passo inicial de otimização do parâmetro."""
    return PARAM_SPECS[path].step


def clamp_value(path: str, value: float) -> int | float:
    """Ajusta `value` à gama segura, arredonda ints e força kernels ímpares.

    É o único ponto por onde passa qualquer valor aprendido — a estabilidade
    e a segurança do treino dependem disto nunca falhar.
    """
    registered = PARAM_SPECS[path]
    value = min(registered.high, max(registered.low, value))
    if registered.dtype is _INT:
        value = round(value)
        if registered.odd and int(value) % 2 == 0:
            value = int(value) + 1
        return int(value)
    return float(value)


def get_param(cfg: AstroFrameConfig, path: str) -> float:
    """Valor atual do parâmetro na configuração."""
    section, name = path.split(".")
    return float(getattr(getattr(cfg, section), name))


def set_param(cfg: AstroFrameConfig, path: str, value: float) -> None:
    """Define o valor do parâmetro na configuração (sem clamp)."""
    section, name = path.split(".")
    setattr(getattr(cfg, section), name, value)


def apply_deltas(cfg: AstroFrameConfig, deltas: dict[str, float]) -> AstroFrameConfig:
    """Cópia da configuração com os deltas aprendidos aplicados (com clamp).

    A configuração original nunca é mutada; cada delta é somado ao valor
    atual e limitado à gama segura do registry.
    """
    adjusted = copy.deepcopy(cfg)
    for path, delta in deltas.items():
        if path not in PARAM_SPECS:
            continue
        current = get_param(adjusted, path)
        set_param(adjusted, path, clamp_value(path, current + delta))
    return adjusted


def deltas_dict(cfg: AstroFrameConfig, paths: list[str]) -> dict[str, float]:
    """Deltas dos parâmetros pedidos relativos aos valores por omissão."""
    base = AstroFrameConfig()
    return {
        path: round(get_param(cfg, path) - get_param(base, path), 4)
        for path in paths
        if abs(get_param(cfg, path) - get_param(base, path)) > 1e-9
    }


def default_punish_deltas() -> dict[str, float]:
    """Deltas de punição por omissão (grupo `detect`) — os do validator."""
    return {spec.name: spec.punish for spec in specs("detect")}


def default_reward_deltas() -> dict[str, float]:
    """Deltas de recompensa por omissão (grupo `detect`) — os do validator."""
    return {spec.name: spec.reward for spec in specs("detect")}
