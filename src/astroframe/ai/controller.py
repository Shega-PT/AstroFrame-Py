"""Controlador sempre ligado: aplica deltas aprendidos em background thread.

O `Controller` corre como daemon thread e periodicamente aplica os deltas
aprendidos (via FeedbackDB) à configuração ativa — a "memória viva" da IA
entre utilizações. Dois modos:

- **LSTM** (preferido): quando existe um modelo LSTMTuner treinado, usa-o
  para prever o vetor de deltas ideal a partir do histórico recente.
- **FallbackNet** (regra): quando não há modelo treinado, aplica os nudges
  mais recentes do banco (recompensa/punição por estrelas) — correção
  sempre segura, sem dependência de treino.

A thread é daemon (não bloqueia o exit do processo) e pode ser started/stopped
programaticamente. `apply_now()` aplica os deltas imediatamente (startup).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from astroframe.ai.feedback import FeedbackDB, apply_learned
from astroframe.ai.lstm import LSTMTuner
from astroframe.config import AstroFrameConfig

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL = 30.0  # segundos entre polls


class FallbackNet:
    """Motor de correção por regras: aplica nudges recentes sem LSTM.

    Usa os nudges gravados no FeedbackDB (recompensa/punição por estrelas)
    para ajustar a configuração. Seguro sempre — sem dependência de treino.
    """

    def __init__(self, db: FeedbackDB, profile: str):
        self.db = db
        self.profile = profile

    def predict(self, current_config: AstroFrameConfig) -> AstroFrameConfig:
        """Aplica os nudges mais recentes do banco à configuração."""
        return apply_learned(current_config, self.profile, self.db)

    @property
    def available(self) -> bool:
        """FallbackNet está sempre disponível."""
        return True


@dataclass
class ControllerState:
    """Estado interno do controller (para testes e inspeção)."""

    ticks: int = 0
    last_apply_ts: float = 0.0
    deltas_applied: dict = field(default_factory=dict)
    lstm_used: bool = False


class Controller:
    """Controlador sempre ligado: aplica deltas aprendidos em background.

    Parameters
    ----------
    config : AstroFrameConfig
        Configuração ativa (mutada in-place pelos deltas).
    profile : str
        Perfil de aprendizagem (hash de tipo+resolução+câmara+ISO).
    db : FeedbackDB | None
        Banco de aprendizagem (criado por omissão).
    interval : float
        Segundos entre polls (padrão 30s).
    tuner : LSTMTuner | None
        Tuner LSTM para previsão (carregado automaticamente se None).
    """

    def __init__(
        self,
        config: AstroFrameConfig,
        profile: str,
        db: FeedbackDB | None = None,
        interval: float = _DEFAULT_INTERVAL,
        tuner: LSTMTuner | None = None,
    ):
        self.config = config
        self.profile = profile
        self.db = db or FeedbackDB()
        self.interval = interval
        self._tuner = tuner or LSTMTuner.load()
        self._fallback = FallbackNet(self.db, self.profile)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.state = ControllerState()

    @property
    def tuner_available(self) -> bool:
        """True se o tuner LSTM está carregado e pronto."""
        return self._tuner is not None and self._tuner.cell is not None

    @property
    def running(self) -> bool:
        """True se a thread está ativa."""
        return self._thread is not None and self._thread.is_alive()

    def apply_now(self) -> dict[str, float]:
        """Aplica deltas imediatamente (startup ou refresh manual).

        Devolve os deltas aplicados (vazio se nada mudou).
        """
        history = self.db.history(self.profile, limit=12)
        if not history:
            return {}

        if self.tuner_available:
            try:
                deltas = self._tuner.predict_next_delta(history)
                if deltas:
                    from astroframe.ai.params import apply_deltas

                    self.config = apply_deltas(self.config, deltas)
                    self.state.deltas_applied = deltas
                    self.state.lstm_used = True
                    self.state.last_apply_ts = time.time()
                    logger.info("Controller: LSTM aplicou %d deltas", len(deltas))
                    return deltas
            except Exception:
                logger.warning("Controller: LSTM falhou, a usar FallbackNet", exc_info=True)

        # Fallback: aplica nudges recentes via regras
        old_config = self.config
        self.config = self._fallback.predict(self.config)
        self.state.lstm_used = False
        self.state.last_apply_ts = time.time()
        # Calcular deltas aplicados comparando configs
        deltas = self._diff_configs(old_config, self.config)
        if deltas:
            self.state.deltas_applied = deltas
            logger.info("Controller: FallbackNet aplicou %d deltas", len(deltas))
        return deltas

    def _diff_configs(self, old: AstroFrameConfig, new: AstroFrameConfig) -> dict[str, float]:
        """Calcula diferenças entre duas configs (delta por parâmetro)."""
        from astroframe.ai.params import PARAM_SPECS, get_param

        deltas: dict[str, float] = {}
        for path in PARAM_SPECS:
            old_val = get_param(old, path)
            new_val = get_param(new, path)
            if abs(new_val - old_val) > 1e-9:
                deltas[path] = round(new_val - old_val, 6)
        return deltas

    def start(self) -> None:
        """Inicia a thread daemon de polling."""
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="astroframe-controller")
        self._thread.start()
        logger.info("Controller: thread iniciada (intervalo %.1fs)", self.interval)

    def stop(self, timeout: float = 5.0) -> None:
        """Para a thread de polling."""
        if not self.running:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        logger.info("Controller: thread parada")

    def _run_loop(self) -> None:
        """Loop principal da thread: poll + apply周期."""
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                logger.warning("Controller: erro no tick", exc_info=True)
            self._stop_event.wait(self.interval)

    def _tick(self) -> None:
        """Um ciclo de polling: aplica deltas se houver novos runs."""
        self.state.ticks += 1
        self.apply_now()
