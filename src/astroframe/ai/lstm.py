"""Pequena LSTM (NumPy) para aprendizagem temporal e trajetórias.

Duas aplicações, uma única célula LSTM de 1 camada implementada à mão
(forward + backward com backprop-through-time, vetorizado em NumPy — sem
dependências novas):

- **`LSTMTuner`** — aprende do histórico do banco (uma execução = um
  timestep: estrelas + métricas + deltas) e prevê o **vetor de deltas** da
  próxima execução. O auto-tuning (`ai.tuner`) usa essa previsão como ponto
  de partida antes do hill-climbing — convergência mais rápida.
- **`TrajectoryPredictor`** — prevê a posição seguinte do centroide a partir
  dos últimos detetados (anti-trepidação temporal): extrapolação linear como
  base, refinamento LSTM opcional quando existe um modelo treinado.

Segurança: sem histórico suficiente (ou sem modelo) as previsões devolvem
`{}`/`None` — nada muda no comportamento atual. Os pesos são guardados em
`.npz` versionados (`Logs/weights/lstm.npz` por omissão) e o treino é
offline e determinístico (seed fixa).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from astroframe.ai.params import FEEDBACK_PARAMS, PARAM_SPECS, step
from astroframe.paths import weights_dir

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_DEFAULT_MODEL = weights_dir() / "lstm.npz"

# Parâmetros previstos pelo LSTMTuner (os do feedback por estrelas, que são
# os que mais mudam entre execuções). O alvo é o vetor de deltas normalizado.
PREDICT_PARAMS = FEEDBACK_PARAMS


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def torch_available() -> bool:
    """True se o PyTorch estiver instalado (aceleração opcional das redes).

    O núcleo é sempre NumPy (rápido o suficiente para redes pequenas); com
    PyTorch, `ai.backend=torch` permite usar `LSTMCellTorch`/`SmallCNNTorch`.
    """
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


class LSTMCell:
    """Célula LSTM de 1 camada (NumPy) com backprop-through-time manual.

    Gates concatenados num só produto: `gates = W x + U h + b`, com ordem
    i, f, o, g (input, forget, output, candidate).
    """

    def __init__(self, n_in: int, n_hidden: int, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng(0)
        self.n_in = n_in
        self.n_hidden = n_hidden
        scale = 0.1
        self.W = rng.normal(0.0, scale, (4 * n_hidden, n_in))
        self.U = rng.normal(0.0, scale, (4 * n_hidden, n_hidden))
        self.b = np.zeros(4 * n_hidden)

    # ------------------------------------------------------------ forward --

    def forward(self, x_seq: np.ndarray, h0: np.ndarray | None = None) -> tuple[np.ndarray, list[dict]]:
        """Corre a sequência `(T, n_in)` e devolve (saídas `(T, n_hidden)`, cache)."""
        n_hidden = self.n_hidden
        h = np.zeros(n_hidden) if h0 is None else h0.copy()
        c = np.zeros(n_hidden)
        cache: list[dict] = []
        for t in range(len(x_seq)):
            x = x_seq[t]
            gates = self.W @ x + self.U @ h + self.b
            i, f, o, g = np.split(gates, 4)
            i = _sigmoid(i)
            f = _sigmoid(f)
            o = _sigmoid(o)
            g = np.tanh(g)
            c = f * c + i * g
            h = o * np.tanh(c)
            cache.append({"x": x, "i": i, "f": f, "o": o, "g": g, "c": c, "h": h})
        return h, cache

    def forward_full(self, x_seq: np.ndarray) -> np.ndarray:
        """Saídas de todos os timesteps `(T, n_hidden)` (para previsões)."""
        h = np.zeros(self.n_hidden)
        c = np.zeros(self.n_hidden)
        outputs = np.zeros((len(x_seq), self.n_hidden))
        for t in range(len(x_seq)):
            gates = self.W @ x_seq[t] + self.U @ h + self.b
            i, f, o, g = np.split(gates, 4)
            i = _sigmoid(i)
            f = _sigmoid(f)
            o = _sigmoid(o)
            g = np.tanh(g)
            c = f * c + i * g
            h = o * np.tanh(c)
            outputs[t] = h
        return outputs

    # ----------------------------------------------------------- backward --

    def backward(
        self, x_seq: np.ndarray, cache: list[dict], dh_next: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        """Gradientes de W/U/b via BPTT (loss parcial até `dh_next`)."""
        n_hidden = self.n_hidden
        dW = np.zeros_like(self.W)
        dU = np.zeros_like(self.U)
        db = np.zeros_like(self.b)
        if dh_next is None:
            dh_next = np.zeros(n_hidden)
        dc_next = np.zeros(n_hidden)
        for t in range(len(x_seq) - 1, -1, -1):
            step_cache = cache[t]
            x = step_cache["x"]
            i, f, o, g = step_cache["i"], step_cache["f"], step_cache["o"], step_cache["g"]
            c = step_cache["c"]
            c_prev = np.zeros(n_hidden) if t == 0 else cache[t - 1]["c"]
            h_prev = np.zeros(n_hidden) if t == 0 else cache[t - 1]["h"]
            t_c = np.tanh(c)
            do = dh_next * t_c * o * (1.0 - o)
            dc = dh_next * o * (1.0 - t_c * t_c) + dc_next
            df = dc * c_prev * f * (1.0 - f)
            di = dc * g * i * (1.0 - i)
            dg = dc * i * (1.0 - g * g)
            gates_d = np.concatenate([di, df, do, dg])
            db += gates_d
            dW += np.outer(gates_d, x)
            dU += np.outer(gates_d, h_prev)
            dh_next = self.U.T @ gates_d
            dc_next = dc * f
        return {"W": dW, "U": dU, "b": db}

    # --------------------------------------------------------- persistência

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            schema_version=SCHEMA_VERSION,
            n_in=self.n_in,
            n_hidden=self.n_hidden,
            W=self.W,
            U=self.U,
            b=self.b,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> LSTMCell | None:
        path = Path(path)
        if not path.exists():
            return None
        try:
            data = np.load(path)
            if int(data["schema_version"]) != SCHEMA_VERSION:
                logger.warning("Modelo LSTM com versão desconhecida: %s", path)
                return None
            cell = cls(int(data["n_in"]), int(data["n_hidden"]))
            cell.W = data["W"]
            cell.U = data["U"]
            cell.b = data["b"]
            return cell
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("Modelo LSTM ilegível (%s): %s", path, exc)
            return None


def _normalize_delta(path: str, delta: float) -> float:
    """Delta → unidade prevista: delta/(passo×4), limitado a [−1, 1]."""
    return float(np.clip(delta / (step(path) * 4.0), -1.0, 1.0))


def _denormalize_delta(path: str, value: float) -> float:
    """Unidade prevista → delta real (limitado à gama segura)."""
    spec = PARAM_SPECS[path]
    span = step(path) * 4.0
    return float(np.clip(value * span, spec.low - spec.high, spec.high - spec.low))


def run_features(run) -> np.ndarray:
    """Vetor de características de uma execução (9 dims, 0–1 maioritariamente):
    estrelas calculadas/manuais, as 5 métricas e os 2 deltas-chave."""
    metrics = run.metrics or {}
    stars_calc = float(run.stars_calc or 0.0) / 5.0
    stars_user = float(run.stars_user if run.stars_user is not None else run.stars_calc or 0.0) / 5.0
    deltas = run.nudge or {}
    return np.array(
        [
            stars_calc,
            stars_user,
            _clamp01(metrics.get("background", 0.0)),
            _clamp01(metrics.get("limb", 0.0)),
            _clamp01(metrics.get("noise", 0.0)),
            _clamp01(metrics.get("contrast", 0.0)),
            _clamp01(metrics.get("reflections", 0.0)),
            _normalize_delta("denoise.h", deltas.get("denoise.h", 0.0)),
            _normalize_delta("clahe.clip_limit", deltas.get("clahe.clip_limit", 0.0)),
        ],
        dtype=np.float64,
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def target_deltas(run) -> np.ndarray:
    """Alvo do run seguinte: deltas dos parâmetros previstos, normalizados."""
    deltas = run.nudge or {}
    return np.array(
        [_normalize_delta(path, deltas.get(path, 0.0)) for path in PREDICT_PARAMS],
        dtype=np.float64,
    )


def make_sequences(
    history: list, seq_len: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Janelas deslizantes `(X, y)` do histórico: features → deltas seguintes."""
    if len(history) < 2:
        return np.zeros((0, seq_len, 9)), np.zeros((0, len(PREDICT_PARAMS)))
    features = np.array([run_features(run) for run in history])
    targets = np.array([target_deltas(run) for run in history])
    X_list, y_list = [], []
    for end in range(1, len(history)):
        start = max(0, end - seq_len)
        X_list.append(features[start:end])
        y_list.append(targets[end])
    X = np.zeros((len(X_list), seq_len, 9))
    y = np.zeros((len(X_list), len(PREDICT_PARAMS)))
    for i, (x, t) in enumerate(zip(X_list, y_list, strict=False)):
        X[i, -len(x):] = x
        y[i] = t
    return X, y


@dataclass
class FitHistory:
    """Curva do treino (loss final + melhor época)."""

    epochs: int
    final_loss: float
    best_loss: float
    best_epoch: int


class LSTMTuner:
    """Previsão do próximo vetor de deltas a partir do histórico do banco.

    `fit` treina offline (GD completo com validação e early-stop); sem dados
    suficientes ou sem convergência, `predict_next_delta` devolve `{}` e o
    auto-tuning parte da base — falha sempre em silêncio.
    """

    def __init__(self, n_hidden: int = 24, seed: int = 42):
        self.n_hidden = n_hidden
        self.seed = seed
        self.cell: LSTMCell | None = None
        self.head = np.zeros((len(PREDICT_PARAMS), n_hidden))
        self.head_bias = np.zeros(len(PREDICT_PARAMS))
        self._rng = np.random.default_rng(seed)

    def fit(
        self,
        history: list,
        epochs: int = 200,
        lr: float = 0.05,
        seq_len: int = 8,
        val_fraction: float = 0.2,
        patience: int = 6,
    ) -> FitHistory:
        """Treina a previsão no histórico (janelas deslizantes + validação)."""
        X, y = make_sequences(history, seq_len)
        self.cell = LSTMCell(9, self.n_hidden, self._rng)
        self.head = self._rng.normal(0.0, 0.1, (len(PREDICT_PARAMS), self.n_hidden))
        self.head_bias = np.zeros(len(PREDICT_PARAMS))
        if len(X) == 0:
            return FitHistory(epochs=0, final_loss=float("inf"), best_loss=float("inf"), best_epoch=0)
        n_val = max(1, int(len(X) * val_fraction))
        order = self._rng.permutation(len(X))
        val_idx, train_idx = order[:n_val], order[n_val:]
        if len(train_idx) == 0:
            train_idx = val_idx
        best_loss = float("inf")
        best_epoch = 0
        best_params: dict | None = None
        wait = 0
        for epoch in range(epochs):
            loss, grads = self._train_epoch(X[train_idx], y[train_idx])
            val_loss = self._loss(X[val_idx], y[val_idx])
            if val_loss < best_loss - 1e-6:
                best_loss = float(val_loss)
                best_epoch = epoch
                best_params = self._params()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
            self._apply_grads(grads, lr / max(1.0, epoch * 0.02 + 1.0))
        if best_params is not None:
            self._set_params(best_params)
        return FitHistory(
            epochs=epoch + 1,
            final_loss=float(val_loss),
            best_loss=float(best_loss),
            best_epoch=best_epoch,
        )

    def _params(self) -> dict:
        return {
            "W": self.cell.W.copy(),
            "U": self.cell.U.copy(),
            "b": self.cell.b.copy(),
            "head": self.head.copy(),
            "head_bias": self.head_bias.copy(),
        }

    def _set_params(self, params: dict) -> None:
        self.cell.W = params["W"]
        self.cell.U = params["U"]
        self.cell.b = params["b"]
        self.head = params["head"]
        self.head_bias = params["head_bias"]

    def _apply_grads(self, grads: dict, lr: float) -> None:
        self.cell.W -= lr * grads["dW"]
        self.cell.U -= lr * grads["dU"]
        self.cell.b -= lr * grads["db"]
        self.head -= lr * grads["dhead"]
        self.head_bias -= lr * grads["dhead_bias"]

    def _train_epoch(self, X: np.ndarray, y: np.ndarray) -> tuple[float, dict]:
        dW = np.zeros_like(self.cell.W)
        dU = np.zeros_like(self.cell.U)
        db = np.zeros_like(self.cell.b)
        dhead = np.zeros_like(self.head)
        dhead_bias = np.zeros_like(self.head_bias)
        total = 0.0
        for seq, target in zip(X, y, strict=False):
            h, cache = self.cell.forward(seq)
            pred = self.head @ h + self.head_bias
            err = pred - target
            total += float(np.mean(err**2))
            dh = self.head.T @ err
            dhead += np.outer(err, h)
            dhead_bias += err
            grads = self.cell.backward(seq, cache, dh)
            dW += grads["W"]
            dU += grads["U"]
            db += grads["b"]
        n = len(X)
        return total / n, {
            "dW": dW / n,
            "dU": dU / n,
            "db": db / n,
            "dhead": dhead / n,
            "dhead_bias": dhead_bias / n,
        }

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        total = 0.0
        for seq, target in zip(X, y, strict=False):
            h, _ = self.cell.forward(seq)
            pred = self.head @ h + self.head_bias
            total += float(np.mean((pred - target) ** 2))
        return total / len(X) if len(X) else 0.0

    def predict_next_delta(self, history: list, seq_len: int = 8) -> dict[str, float]:
        """Previsão do vetor de deltas para a próxima execução (ou `{}`)."""
        if self.cell is None or len(history) < 2:
            return {}
        recent = history[-seq_len:]
        features = np.array([run_features(run) for run in recent])
        seq = np.zeros((seq_len, 9))
        seq[-len(features):] = features
        h, _ = self.cell.forward(seq)
        pred = self.head @ h + self.head_bias
        return {
            path: _denormalize_delta(path, float(value))
            for path, value in zip(PREDICT_PARAMS, pred, strict=False)
            if abs(float(value)) > 1e-3
        }

    def save(self, path: str | Path | None = None) -> Path:
        """Guarda célula + cabeça num `.npz` versionado."""
        path = Path(path) if path else _DEFAULT_MODEL
        path.parent.mkdir(parents=True, exist_ok=True)
        cell = self.cell or LSTMCell(9, self.n_hidden)
        np.savez(
            path,
            schema_version=SCHEMA_VERSION,
            n_hidden=self.n_hidden,
            W=cell.W,
            U=cell.U,
            b=cell.b,
            head=self.head,
            head_bias=self.head_bias,
        )
        return path

    @classmethod
    def load(cls, path: str | Path | None = None) -> LSTMTuner | None:
        path = Path(path) if path else _DEFAULT_MODEL
        if not path.exists():
            return None
        try:
            data = np.load(path)
            if int(data["schema_version"]) != SCHEMA_VERSION:
                logger.warning("Modelo de previsão com versão desconhecida: %s", path)
                return None
            tuner = cls(int(data["n_hidden"]))
            tuner.cell = LSTMCell(9, tuner.n_hidden)
            tuner.cell.W = data["W"]
            tuner.cell.U = data["U"]
            tuner.cell.b = data["b"]
            tuner.head = data["head"]
            tuner.head_bias = data["head_bias"]
            return tuner
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("Modelo de previsão ilegível (%s): %s", path, exc)
            return None


class TrajectoryPredictor:
    """Previsão do próximo centroide a partir dos últimos detetados.

    Base: extrapolação linear (mínimos quadrados) sobre o histórico de
    posições — robusta e barata. Opcional: refinamento LSTM (célula 2→8)
    treinada offline em trajetórias sintéticas, usada quando `use_lstm`
    está ligado e existe um modelo compatível. Nunca levanta exceções em
    runtime (sem histórico → `None`).
    """

    def __init__(self, maxlen: int = 8, use_lstm: bool = False, model_path: str | Path | None = None):
        self.maxlen = maxlen
        self.use_lstm = bool(use_lstm)
        self.model_path = Path(model_path) if model_path else _DEFAULT_MODEL
        self._history: deque[tuple[float, float]] = deque(maxlen=maxlen)
        self._cell: LSTMCell | None = None
        if self.use_lstm:
            self._cell = LSTMCell.load(self.model_path)

    def push(self, cx: float, cy: float) -> None:
        """Regista uma posição detetada."""
        self._history.append((float(cx), float(cy)))

    def clear(self) -> None:
        self._history.clear()

    def __len__(self) -> int:
        return len(self._history)

    def predict(self) -> tuple[float, float] | None:
        """Próxima posição prevista, ou None sem histórico suficiente."""
        if len(self._history) < 2:
            return None
        if self.use_lstm and self._cell is not None and len(self._history) >= 4:
            predicted = self._predict_lstm()
            if predicted is not None:
                return predicted
        return self._predict_linear()

    def _predict_linear(self) -> tuple[float, float]:
        points = list(self._history)
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        idx = np.arange(len(points), dtype=np.float64)
        n = len(points)
        x_slope, x_intercept = np.polyfit(idx, xs, 1)
        y_slope, y_intercept = np.polyfit(idx, ys, 1)
        return float(x_slope * n + x_intercept), float(y_slope * n + y_intercept)

    def _predict_lstm(self) -> tuple[float, float] | None:
        if self._cell is None or self._cell.n_in != 2:
            return None
        points = list(self._history)
        pairs = [
            (points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]) for i in range(1, len(points))
        ]
        deltas = np.array(pairs)
        seq = np.zeros((8, 2))
        seq[-len(deltas):] = deltas
        h = self._cell.forward_full(seq)[-1]
        last = points[-1]
        return float(last[0] + h[0]), float(last[1] + h[1])


def train_trajectory_model(
    trajectories: list[list[tuple[float, float]]],
    path: str | Path | None = None,
    seed: int = 42,
    epochs: int = 60,
) -> Path:
    """Treina uma célula LSTM (2→8) em trajetórias sintéticas e guarda-a.

    Entrada: sequências de posições; alvo: o deslocamento seguinte. Serve
    para o `TrajectoryPredictor(use_lstm=True)` refinar a extrapolação
    linear nas vídeo-ações com trepidação. Determinístico (seed fixa).
    """
    path = Path(path) if path else _DEFAULT_MODEL
    rng = np.random.default_rng(seed)
    cell = LSTMCell(2, 8, rng)
    head = rng.normal(0.0, 0.1, (2, 8))
    head_bias = np.zeros(2)
    X_list, y_list = [], []
    for trajectory in trajectories:
        pts = [(float(x), float(y)) for x, y in trajectory]
        if len(pts) < 3:
            continue
        deltas = np.array(
            [(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]) for i in range(1, len(pts))]
        )
        for end in range(2, len(deltas) + 1):
            start = max(0, end - 6)
            seq = np.zeros((6, 2))
            seq[- (end - start):] = deltas[start:end]
            X_list.append(seq)
            y_list.append(deltas[end - 1])
    X = np.array(X_list)
    y = np.array(y_list)
    if len(X) == 0:
        raise ValueError("Sem trajetórias válidas para treinar o modelo.")
    lr = 0.05
    for _ in range(epochs):
        dW = np.zeros_like(cell.W)
        dU = np.zeros_like(cell.U)
        db = np.zeros_like(cell.b)
        dhead = np.zeros_like(head)
        dhead_bias = np.zeros_like(head_bias)
        for seq, target in zip(X, y, strict=False):
            h, cache = cell.forward(seq)
            pred = head @ h + head_bias
            err = pred - target
            dh = head.T @ err
            dhead += np.outer(err, h)
            dhead_bias += err
            grads = cell.backward(seq, cache, dh)
            dW += grads["W"]
            dU += grads["U"]
            db += grads["b"]
        n = len(X)
        cell.W -= lr * dW / n
        cell.U -= lr * dU / n
        cell.b -= lr * db / n
        head -= lr * dhead / n
        head_bias -= lr * dhead_bias / n
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        schema_version=SCHEMA_VERSION,
        n_in=2,
        n_hidden=8,
        W=cell.W,
        U=cell.U,
        b=cell.b,
        head=head,
        head_bias=head_bias,
    )
    return path


def trajectory_model_path(path: str | Path | None = None) -> Path:
    """Caminho do modelo de trajetória (por omissão `~/.astroframe/lstm.npz`)."""
    return Path(path) if path else _DEFAULT_MODEL