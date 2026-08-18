"""Parâmetros ajustáveis da pipeline AstroFrame.

Todos os valores podem ser sobrescritos por um ficheiro YAML, evitando
valores hardcoded no código. Gere um modelo com `astroframe config-template`.
"""

import logging
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class _TypeMismatch(Exception):
    pass


def _normalize(value: Any, expected: Any) -> Any:
    """Ajusta/harmoniza o valor YAML ao tipo esperado; levanta _TypeMismatch se não bater."""
    if expected in (int, float) and isinstance(value, (int, float)):
        return expected(value)
    if value is None:
        members = getattr(expected, "__args__", None) or (expected,)
        if any(member is type(None) for member in members):
            return None
        raise _TypeMismatch
    members = getattr(expected, "__args__", None)
    if members:  # união de tipos (ex.: float | None)
        for member in members:
            if member is type(None):
                continue
            try:
                return _normalize(value, member)
            except _TypeMismatch:
                continue
        raise _TypeMismatch
    if isinstance(expected, type) and isinstance(value, expected):
        return value
    raise _TypeMismatch


@dataclass
class CLAHEConfig:
    clip_limit: float = 3.0
    tile_grid_size: int = 8


@dataclass
class DenoiseConfig:
    h: float = 5.0
    template_window_size: int = 7
    search_window_size: int = 21


@dataclass
class UnsharpConfig:
    sigma: float = 2.0
    amount: float = 0.5


@dataclass
class StabilizerConfig:
    max_radius: int = 400
    max_disks: int = 8
    dp: float = 1.2
    param1: int = 50
    param2: int = 30
    gaussian_kernel_size: int = 9
    gaussian_sigma: float = 2.0
    contour_fallback: bool = True
    auto_crop: bool = True
    jitter_alpha: float = 0.5
    occluded_ratio: float = 0.75
    occluded_ring: float = 1.25


@dataclass
class LuckyConfig:
    min_sharpness: float | None = None
    sharpness_percentile: float = 25.0
    gaussian_kernel_size: int = 5
    gaussian_sigma: float = 1.5


@dataclass
class StackingConfig:
    n_best: int = 10
    use_median: bool = True


@dataclass
class PolishConfig:
    """Polimento por astros: recorte, diluição com o fundo e remoção de reflexos.

    Cada astro detetado (Sol, Lua, ...) recebe o seu próprio realce — `band`,
    `feather` e `brightness` — e é remontado por blend de máscaras sobrepostas,
    sem costuras. `corona_scale` é a linha de recorte (× raio do astro): entre
    o bordo do astro e essa linha a imagem é **diluída** até ao fundo. O fundo
    é a média do fundo original (`background_fill`) ou preto puro
    (`black_background`); `remove_reflections` elimina círculos-ghost com o
    centro fora do astro maior.
    """

    enabled: bool = True
    corona_scale: float = 1.6
    feather: float = 0.02
    background_fill: bool = True
    black_background: bool = False
    brightness: float = 0.15
    remove_reflections: bool = True
    reflection_min_radius: int = 8


@dataclass
class ScoreConfig:
    """Pesos da avaliação automática (0–5 estrelas) e limiares de recompensa.

    Cada peso multiplica a respetiva métrica (0–1); `limb_gain`/`edge_radius`
    controlam a máscara de brilho usada para medir a redondeza do limbo.
    """

    background_weight: float = 0.30
    limb_weight: float = 0.30
    noise_weight: float = 0.15
    contrast_weight: float = 0.15
    reflection_weight: float = 0.10
    limb_min_dark: float = 0.15
    limb_gain: float = 4.0
    edge_radius: float = 1.05


@dataclass
class FeedbackConfig:
    """Aprendizagem por estrelas: recompensa/punição persistentes.

    `db_path` aponta para a base local (SQLite, `Logs/logs/system/` por
    omissão — caminho relativo resolve contra a raiz de dados) onde cada
    utilização é registada como uma linha (log de ajustes: o que mudou,
    como e porquê). `learning_rate` é a fração do delta aplicado por
    execução e `user_weight` o multiplicador quando o utilizador avalia
    manualmente.
    """

    enabled: bool = True
    db_path: str = "logs/system/feedback.db"
    learning_rate: float = 0.3
    user_weight: float = 2.0
    history_limit: int = 12


@dataclass
class TuningConfig:
    """Auto-tuning (hill-climbing limitado) de todos os parâmetros.

    `budget_s` é o orçamento de tempo do otimizador; `seed` torna a busca
    determinística; `anneal` permite aceitar pioras para escapar de mínimos
    locais (nunca fora das gamas seguras do registry); `proxy_scale`/
    `frames_per_sample` controlam a avaliação rápida (~480p, 3 frames);
    `detection_weight` equilibra deteção vs melhoria no objetivo;
    `params` limita a otimização a um subconjunto de caminhos registados.
    """

    enabled: bool = False
    budget_s: float = 60.0
    seed: int = 42
    anneal: bool = True
    proxy_scale: float = 0.5
    frames_per_sample: int = 3
    detection_weight: float = 0.6
    params: list[str] | None = None


@dataclass
class AIConfig:
    """Redes neurais pequenas (opcionais, todas OFF por omissão).

    O núcleo é NumPy puro; `backend="torch"` ativa a aceleração opcional
    quando o PyTorch está instalado (extra `astroframe[rife]` inclui torch).
    `lstm_trajectory` melhora a anti-trepidação temporal prevendo o
    centroide (extrapolação linear + LSTM opcional); `cnn_enhance` adiciona
    um passo residual de melhoria aprendida ao final do `enhance_image`;
    `disk_filter` (0–1) é o limiar de confiança da CNN para descartar falsos
    positivos da deteção (0.0 = desligado; nunca esvazia a lista detetada).
    """

    backend: str = "numpy"
    lstm_trajectory: bool = False
    cnn_enhance: bool = False
    disk_filter: float = 0.0


@dataclass
class AstroFrameConfig:
    clahe: CLAHEConfig = field(default_factory=CLAHEConfig)
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    unsharp: UnsharpConfig = field(default_factory=UnsharpConfig)
    stabilizer: StabilizerConfig = field(default_factory=StabilizerConfig)
    lucky: LuckyConfig = field(default_factory=LuckyConfig)
    stacking: StackingConfig = field(default_factory=StackingConfig)
    polish: PolishConfig = field(default_factory=PolishConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    ai: AIConfig = field(default_factory=AIConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AstroFrameConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls._build(cls, data)

    @classmethod
    def _build(cls, datacls: type, data: dict[str, Any]):
        known = {f.name for f in fields(datacls)}
        for key in data:
            if key not in known:
                logger.warning("Chave desconhecida no YAML ignorada: %s", key)
        args: dict[str, Any] = {}
        for f in fields(datacls):
            if f.name not in data:
                continue
            value = data[f.name]
            if is_dataclass(f.type) and isinstance(value, dict):
                args[f.name] = cls._build(f.type, value)
            else:
                try:
                    args[f.name] = _normalize(value, f.type)
                except _TypeMismatch:
                    logger.warning(
                        "Campo '%s' com tipo inesperado (%s); a usar o valor tal como está.",
                        f.name,
                        type(value).__name__,
                    )
                    args[f.name] = value
        return datacls(**args)
