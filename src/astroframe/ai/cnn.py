"""Pequena CNN (NumPy) para melhoria de imagens e filtragem da deteção.

Corpo partilhado — conv1 (1→8, 3×3, pad 1, ReLU) → conv2 (8→8, 3×3, pad 1,
ReLU) — com duas cabeças:

- **residual** (melhorar imagens): conv3 (8→1, 3×3, pad 1) prevê o **resíduo**
  `r = y − x` entre a entrada e o alvo ideal (o teu próprio pipeline com
  parâmetros afinados). `ResidualEnhancer` treina em pares sintéticos
  (ruído → limpo) e aplica em tiles com overlap — passo final opcional do
  `enhance_image` (`ai.cnn_enhance`).
- **classificadora** (deteção): global average pool → dense 8→2 (softmax)
  decide se um patch centrado é um disco real. `DiskFilter` treina com
  positivos (ground truth de `calibration.json`) e negativos (rejeitados/
  falsos positivos) e filtra os candidatos do Hough no fim de
  `find_all_disks` (`ai.disk_filter`) — **nunca** esvazia a lista detetada.

Tudo NumPy (im2col vetorizado, sem dependências novas); `backend="torch"`
é a aceleração opcional quando o PyTorch está instalado. Os pesos são
guardados em `.npz` versionados (`Logs/weights/enhancer_cnn.npz` e
`Logs/weights/disk_filter.npz`) e o treino é offline e determinístico.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from astroframe.paths import weights_dir

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
PATCH_SIZE = 48
_TILE = 64
_TILE_OVERLAP = 8
_STRIDE = _TILE - _TILE_OVERLAP

# Épocas mínimas antes de o early-stop contar (o treino pequeno começa num
# patamar simétrico e só ganha tração após alguns passos).
_WARMUP = 8

_ENHANCER_MODEL = weights_dir() / "enhancer_cnn.npz"
_FILTER_MODEL = weights_dir() / "disk_filter.npz"


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _softmax(x: np.ndarray) -> np.ndarray:
    ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return ex / np.sum(ex, axis=-1, keepdims=True)


def _im2col_windows(x: np.ndarray, kh: int, kw: int, pad: int) -> np.ndarray:
    """Janelas (N, C, H', W', kh, kw) da entrada `(N, C, H, W)` com padding."""
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    return np.lib.stride_tricks.sliding_window_view(xp, (kh, kw), axis=(2, 3))


def conv2d_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray, pad: int = 1) -> np.ndarray:
    """Convolução 3×3 (stride 1, pad 1) sobre batch `(N, C, H, W)` → `(N, K, H, W)`."""
    windows = _im2col_windows(x, *w.shape[-2:], pad)
    return np.einsum("nchwpq,kcpq->nkhw", windows, w) + b[None, :, None, None]


def conv2d_backward(
    grad_out: np.ndarray, x: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gradientes da convolução: `dw`, `db` e `dx` (para o conv seguinte).

    O `dx` devolvido tem a forma exata de `x` (o gradiente da correlação
    completa é recortado do padding de 1 px de cada lado).
    """
    kh, kw = w.shape[-2:]
    windows = _im2col_windows(x, kh, kw, 1)
    dw = np.einsum("nchwpq,nkhw->kcpq", windows, grad_out)
    db = grad_out.sum(axis=(0, 2, 3))
    dx = _conv_transpose_correlate(grad_out, w)[:, :, 1:-1, 1:-1]
    return dw, db, dx


def _conv_transpose_correlate(grad_out: np.ndarray, w: np.ndarray) -> np.ndarray:
    """`dx` (com padding) da convolução, numa só passada vetorizada.

    O gradiente da correlação `y = x ⋆ w` é a correlação completa de
    `grad_out` com o kernel **invertido** `w[::-1, ::-1]` (o transposto da
    convolução); o resultado tem `H+2` linhas — o recorte `[1:-1]` é feito
    em `conv2d_backward`. Verificado por diferenças finitas nos testes.
    """
    N, K, H, W = grad_out.shape
    _, _, kh, kw = w.shape
    flipped = w[:, :, ::-1, ::-1]
    gp = np.pad(grad_out, ((0, 0), (0, 0), (kh - 1, kh - 1), (kw - 1, kw - 1)))
    windows = np.lib.stride_tricks.sliding_window_view(gp, (kh, kw), axis=(2, 3))
    return np.einsum("nkhwpq,kcpq->nchw", windows, flipped)


@dataclass
class FitReport:
    """Curva do treino (loss final + melhor época)."""

    epochs: int
    final_loss: float
    best_loss: float
    best_epoch: int


class SmallCNN:
    """Rede pequena com cabeça residual ou classificadora (NumPy)."""

    def __init__(self, mode: str = "residual", k: int = 8, seed: int = 42, n_in: int = 1):
        if mode not in ("residual", "classify"):
            raise ValueError(f"Modo desconhecido: {mode!r} (residual | classify)")
        self.mode = mode
        self.k = k
        self.n_in = n_in
        rng = np.random.default_rng(seed)
        self.conv1_w = rng.normal(0.0, 0.08, (k, n_in, 3, 3))
        self.conv1_b = np.zeros(k)
        self.conv2_w = rng.normal(0.0, 0.08, (k, k, 3, 3))
        self.conv2_b = np.zeros(k)
        if mode == "residual":
            self.conv3_w = rng.normal(0.0, 0.08, (1, k, 3, 3))
            self.conv3_b = np.zeros(1)
        else:
            self.head_w = rng.normal(0.0, 0.15, (2, k))
            self.head_b = rng.normal(0.0, 0.05, 2)

    # ------------------------------------------------------------ forward --

    def _features(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """Camadas conv partilhadas → `(c1, c2, cache)`."""
        c1 = _relu(conv2d_forward(x, self.conv1_w, self.conv1_b))
        c2 = _relu(conv2d_forward(c1, self.conv2_w, self.conv2_b))
        return c1, c2, {"x": x, "c1": c1, "c2": c2}

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        """`x` `(N, 1, H, W)` → saída da cabeça + cache."""
        c1, c2, cache = self._features(x)
        if self.mode == "residual":
            r = conv2d_forward(c2, self.conv3_w, self.conv3_b)
            out = x + r
            cache["r"] = r
            return out, cache
        pooled = c2.mean(axis=(2, 3))
        logits = pooled @ self.head_w.T + self.head_b
        probs = _softmax(logits)
        cache["pooled"] = pooled
        return probs, cache

    def predict_class(self, x: np.ndarray) -> np.ndarray:
        """Probabilidade da classe 1 (disco) para `(N, 1, H, W)`."""
        probs, _ = self.forward(x)
        return probs[:, 1]

    # ----------------------------------------------------------- backward --

    def backward_residual(self, grad_out: np.ndarray, cache: dict) -> dict[str, np.ndarray]:
        """Gradientes do modo residual (MSE sobre o resíduo)."""
        x, c1, c2 = cache["x"], cache["c1"], cache["c2"]
        dconv3, db3, dr = conv2d_backward(grad_out, c2, self.conv3_w)
        dc2 = dr
        dc2 = dc2 * (c2 > 0)
        dconv2, db2, dc1 = conv2d_backward(dc2, c1, self.conv2_w)
        dc1 = dc1 * (c1 > 0)
        dconv1, db1, dx = conv2d_backward(dc1, x, self.conv1_w)
        return {
            "conv1_w": dconv1, "conv1_b": db1,
            "conv2_w": dconv2, "conv2_b": db2,
            "conv3_w": dconv3, "conv3_b": db3,
        }

    def backward_classify(self, grad_probs: np.ndarray, cache: dict) -> dict[str, np.ndarray]:
        """Gradientes do modo classificador (cross-entropy)."""
        c1, c2, pooled = cache["c1"], cache["c2"], cache["pooled"]
        dhead = grad_probs.T @ pooled
        dhead_b = grad_probs.sum(axis=0)
        dpooled = grad_probs @ self.head_w
        dc2 = np.zeros_like(c2)
        dc2 += (dpooled / (c2.shape[2] * c2.shape[3]))[:, :, None, None]
        dc2 = dc2 * (c2 > 0)
        dconv2, db2, dc1 = conv2d_backward(dc2, c1, self.conv2_w)
        dc1 = dc1 * (c1 > 0)
        dconv1, db1, _ = conv2d_backward(dc1, cache["x"], self.conv1_w)
        return {
            "conv1_w": dconv1, "conv1_b": db1,
            "conv2_w": dconv2, "conv2_b": db2,
            "head_w": dhead, "head_b": dhead_b,
        }

    # --------------------------------------------------------- persistência

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict = {
            "schema_version": SCHEMA_VERSION,
            "mode": self.mode,
            "k": self.k,
            "n_in": self.n_in,
            "conv1_w": self.conv1_w,
            "conv1_b": self.conv1_b,
            "conv2_w": self.conv2_w,
            "conv2_b": self.conv2_b,
        }
        if self.mode == "residual":
            data["conv3_w"] = self.conv3_w
            data["conv3_b"] = self.conv3_b
        else:
            data["head_w"] = self.head_w
            data["head_b"] = self.head_b
        np.savez(path, **data)
        return path

    @classmethod
    def load(cls, path: str | Path) -> SmallCNN | None:
        path = Path(path)
        if not path.exists():
            return None
        try:
            data = np.load(path)
            if int(data["schema_version"]) != SCHEMA_VERSION:
                logger.warning("Modelo CNN com versão desconhecida: %s", path)
                return None
            model = cls(
                mode=str(data["mode"]), k=int(data["k"]), n_in=int(data["n_in"])
            )
            model.conv1_w = data["conv1_w"]
            model.conv1_b = data["conv1_b"]
            model.conv2_w = data["conv2_w"]
            model.conv2_b = data["conv2_b"]
            if model.mode == "residual":
                model.conv3_w = data["conv3_w"]
                model.conv3_b = data["conv3_b"]
            else:
                model.head_w = data["head_w"]
                model.head_b = data["head_b"]
            return model
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("Modelo CNN ilegível (%s): %s", path, exc)
            return None


def _to_batch(patches: list[np.ndarray], size: int = PATCH_SIZE) -> np.ndarray:
    """Lista de patches gray → batch `(N, 1, size, size)` float 0–1.

    Aceita uint8 (0–255) ou float (0–1); só normaliza por 255 quando a
    entrada é uint8.
    """
    batch = np.zeros((len(patches), 1, size, size), dtype=np.float64)
    for i, patch in enumerate(patches):
        gray = patch if patch.ndim == 2 else np.mean(patch, axis=2)
        resized = gray
        if gray.shape[0] != size or gray.shape[1] != size:
            resized = _resize(gray, size)
        arr = resized.astype(np.float64)
        if arr.size and np.max(arr) > 1.0:
            arr = arr / 255.0
        batch[i, 0] = np.clip(arr, 0.0, 1.0)
    return batch


def _resize(gray: np.ndarray, size: int) -> np.ndarray:
    """Redimensiona com interpolação bilinear simples (sem OpenCV no core?)."""
    h, w = gray.shape
    ys = (np.arange(size) + 0.5) * h / size - 0.5
    xs = (np.arange(size) + 0.5) * w / size - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 2)
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 2)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    top = gray[y0][:, x0] * (1 - fx) + gray[y0][:, x0 + 1] * fx
    bottom = gray[y0 + 1][:, x0] * (1 - fx) + gray[y0 + 1][:, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy


def fit_residual(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    model: SmallCNN | None = None,
    epochs: int = 40,
    lr: float = 0.05,
    batch_size: int = 8,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[SmallCNN, FitReport]:
    """Treina a cabeça residual em pares (x, y): aprende `r = y − x`.

    Loss MSE do resíduo; validação 20% com early-stop (patience 5).
    Devolve (modelo treinado, histórico do treino).
    """
    model = model or SmallCNN(mode="residual", seed=seed)
    rng = np.random.default_rng(seed)
    X = _to_batch([p[0] for p in pairs])
    targets = _to_batch([p[1] for p in pairs])
    residuals = targets - X
    if len(X) == 0:
        raise ValueError("Sem pares de treino para a CNN residual.")
    n_val = max(1, int(len(X) * val_fraction))
    order = rng.permutation(len(X))
    val_idx, train_idx = order[:n_val], order[n_val:]
    if len(train_idx) == 0:
        train_idx = val_idx
    best_loss = float("inf")
    best_epoch = 0
    best_params: dict | None = None
    wait = 0
    for epoch in range(epochs):
        rng.shuffle(train_idx)
        for start in range(0, len(train_idx), batch_size):
            ids = train_idx[start : start + batch_size]
            out, cache = model.forward(X[ids])
            err = out - targets[ids]
            grads = model.backward_residual(err / err.size, cache)
            _apply_grads(model, grads, lr / max(1.0, epoch * 0.02 + 1.0))
        val_loss = _residual_loss(model, X[val_idx], residuals[val_idx])
        if val_loss < best_loss - 1e-6 or epoch < _WARMUP:
            if val_loss < best_loss - 1e-6:
                best_loss = float(val_loss)
                best_epoch = epoch
                best_params = _params_copy(model)
            wait = 0
        else:
            wait += 1
            if wait >= 5:
                break
    if best_params is not None:
        _restore_params(model, best_params)
    return model, FitReport(
        epochs=epoch + 1, final_loss=float(val_loss), best_loss=best_loss, best_epoch=best_epoch
    )


def _residual_loss(model: SmallCNN, X: np.ndarray, residuals: np.ndarray) -> float:
    out, _ = model.forward(X)
    return float(np.mean((out - residuals) ** 2))


def fit_classifier(
    positives: list[np.ndarray],
    negatives: list[np.ndarray],
    model: SmallCNN | None = None,
    epochs: int = 60,
    lr: float = 0.05,
    batch_size: int = 8,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[SmallCNN, FitReport]:
    """Treina a cabeça classificadora: patches de discos vs não-discos."""
    model = model or SmallCNN(mode="classify", seed=seed)
    rng = np.random.default_rng(seed)
    X_pos = _to_batch(positives)
    X_neg = _to_batch(negatives)
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    if len(X) == 0:
        raise ValueError("Sem patches para treinar o classificador.")
    n_val = max(1, int(len(X) * val_fraction))
    order = rng.permutation(len(X))
    val_idx, train_idx = order[:n_val], order[n_val:]
    if len(train_idx) == 0:
        train_idx = val_idx
    best_loss = float("inf")
    best_epoch = 0
    best_params: dict | None = None
    wait = 0
    for epoch in range(epochs):
        rng.shuffle(train_idx)
        for start in range(0, len(train_idx), batch_size):
            ids = train_idx[start : start + batch_size]
            probs, cache = model.forward(X[ids])
            onehot = np.eye(2)[y[ids].astype(int)]
            grad = (probs - onehot) / len(ids)
            grads = model.backward_classify(grad, cache)
            _apply_grads(model, grads, lr / max(1.0, epoch * 0.02 + 1.0))
        val_loss = _classify_loss(model, X[val_idx], y[val_idx])
        if val_loss < best_loss - 1e-6 or epoch < _WARMUP:
            if val_loss < best_loss - 1e-6:
                best_loss = float(val_loss)
                best_epoch = epoch
                best_params = _params_copy(model)
            wait = 0
        else:
            wait += 1
            if wait >= 5:
                break
    if best_params is not None:
        _restore_params(model, best_params)
    return model, FitReport(
        epochs=epoch + 1, final_loss=float(val_loss), best_loss=best_loss, best_epoch=best_epoch
    )


def _classify_loss(model: SmallCNN, X: np.ndarray, y: np.ndarray) -> float:
    probs, _ = model.forward(X)
    onehot = np.eye(2)[y.astype(int)]
    return float(-np.mean(np.sum(onehot * np.log(probs + 1e-12), axis=1)))


def _params_copy(model: SmallCNN) -> dict:
    data = {"conv1_w": model.conv1_w.copy(), "conv1_b": model.conv1_b.copy(),
            "conv2_w": model.conv2_w.copy(), "conv2_b": model.conv2_b.copy()}
    if model.mode == "residual":
        data["conv3_w"] = model.conv3_w.copy()
        data["conv3_b"] = model.conv3_b.copy()
    else:
        data["head_w"] = model.head_w.copy()
        data["head_b"] = model.head_b.copy()
    return data


def _restore_params(model: SmallCNN, params: dict) -> None:
    model.conv1_w = params["conv1_w"]
    model.conv1_b = params["conv1_b"]
    model.conv2_w = params["conv2_w"]
    model.conv2_b = params["conv2_b"]
    if model.mode == "residual":
        model.conv3_w = params["conv3_w"]
        model.conv3_b = params["conv3_b"]
    else:
        model.head_w = params["head_w"]
        model.head_b = params["head_b"]


def _apply_grads(model: SmallCNN, grads: dict, lr: float) -> None:
    model.conv1_w -= lr * grads["conv1_w"]
    model.conv1_b -= lr * grads["conv1_b"]
    model.conv2_w -= lr * grads["conv2_w"]
    model.conv2_b -= lr * grads["conv2_b"]
    if model.mode == "residual":
        model.conv3_w -= lr * grads["conv3_w"]
        model.conv3_b -= lr * grads["conv3_b"]
    else:
        model.head_w -= lr * grads["head_w"]
        model.head_b -= lr * grads["head_b"]


class ResidualEnhancer:
    """Melhoria aprendida: aplica o resíduo da CNN em tiles com overlap.

    `apply` recebe uma imagem BGR e devolve BGR com o resíduo somado ao
    canal L (LAB) — cores originais preservadas. Sem modelo (ou em falha)
    devolve a imagem intacta: nunca piora o resultado.
    """

    def __init__(self, model: SmallCNN | None = None, model_path: str | Path | None = None):
        self.model = model or SmallCNN.load(model_path or _ENHANCER_MODEL)

    @property
    def available(self) -> bool:
        return self.model is not None

    def apply(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.model is None:
            return image_bgr
        import cv2

        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        lightness = lab[:, :, 0].astype(np.float64) / 255.0
        enhanced = self._apply_gray(lightness)
        lab[:, :, 0] = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _apply_gray(self, gray01: np.ndarray) -> np.ndarray:
        h, w = gray01.shape
        pad_h = (_STRIDE - h % _STRIDE) % _STRIDE
        pad_w = (_STRIDE - w % _STRIDE) % _STRIDE
        padded = np.pad(gray01, ((0, pad_h), (0, pad_w)), mode="edge")
        out = np.zeros_like(padded)
        counts = np.zeros_like(padded)
        tiles = []
        positions = []
        for y0 in range(0, padded.shape[0] - _TILE + 1, _STRIDE):
            for x0 in range(0, padded.shape[1] - _TILE + 1, _STRIDE):
                tiles.append(padded[y0 : y0 + _TILE, x0 : x0 + _TILE])
                positions.append((y0, x0))
        if not tiles:
            tile = np.zeros((padded.shape[0], padded.shape[1]))
            tile[:h, :w] = gray01
            batch = _to_batch([tile])
            out_tiles, _ = self.model.forward(batch)
            return out_tiles[0, 0, :h, :w]
        batch = _to_batch(tiles, size=_TILE)
        out_tiles, _ = self.model.forward(batch)
        for (y0, x0), tile in zip(positions, out_tiles[:, 0], strict=True):
            out[y0 : y0 + _TILE, x0 : x0 + _TILE] += tile
            counts[y0 : y0 + _TILE, x0 : x0 + _TILE] += 1.0
        out = np.divide(out, counts, out=np.zeros_like(out), where=counts > 0)
        return out[:h, :w]


def disk_patch(image: np.ndarray, cx: int, cy: int, radius: int) -> np.ndarray:
    """Patch cinza centrado num candidato (2×raio, mínimo 16 px de lado)."""
    gray = image if image.ndim == 2 else np.mean(image, axis=2)
    h, w = gray.shape
    half = max(8, int(radius))
    x0, x1 = max(0, cx - half), min(w, cx + half)
    y0, y1 = max(0, cy - half), min(h, cy + half)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((PATCH_SIZE, PATCH_SIZE))
    return _resize(crop, PATCH_SIZE)


class DiskFilter:
    """Filtro CNN dos candidatos da deteção (discos reais vs falsos positivos).

    `keep` avalia um patch centrado no candidato (recortado a 2×raio e
    reduzido a 48×48): devolve True se P(disco) ≥ limiar. `filter_disks`
    aplica a lista inteira e **nunca** a esvazia quando a deteção raw tinha
    discos — a deteção nunca regride.
    """

    def __init__(self, model: SmallCNN | None = None, model_path: str | Path | None = None):
        self.model = model or SmallCNN.load(model_path or _FILTER_MODEL)

    @property
    def available(self) -> bool:
        return self.model is not None

    def patch(self, image: np.ndarray, cx: int, cy: int, radius: int) -> np.ndarray:
        """Patch gray centrado no candidato (2×raio, mínimo 16 px de lado)."""
        return disk_patch(image, cx, cy, radius)

    def confidence(self, image: np.ndarray, cx: int, cy: int, radius: int) -> float:
        """P(disco) para um candidato (0.5 sem modelo)."""
        if self.model is None:
            return 0.5
        batch = _to_batch([self.patch(image, cx, cy, radius)])
        return float(self.model.predict_class(batch)[0])

    def filter_disks(self, disks, image: np.ndarray, threshold: float) -> list:
        """Filtra os candidatos com confiança < limiar; nunca esvazia a lista."""
        if self.model is None or not disks or threshold <= 0.0:
            return list(disks)
        kept = [d for d in disks if self.confidence(image, d.cx, d.cy, d.radius) >= threshold]
        return kept or list(disks)