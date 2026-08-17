"""Testes da pequena CNN (residual + classificadora) e dos wrappers."""

from __future__ import annotations

import numpy as np
import pytest

from astroframe.ai.cnn import (
    PATCH_SIZE,
    DiskFilter,
    FitReport,
    ResidualEnhancer,
    SmallCNN,
    _to_batch,
    conv2d_backward,
    conv2d_forward,
    fit_classifier,
    fit_residual,
)


def _residual_model(k=2, seed=1) -> SmallCNN:
    return SmallCNN(mode="residual", k=k, seed=seed)


def _classify_model(k=2, seed=2) -> SmallCNN:
    return SmallCNN(mode="classify", k=k, seed=seed)


def _patches(n: int, kind: str, size: int = 32, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    for _i in range(n):
        patch = np.zeros((size, size))
        if kind == "disk":
            r = size // 4
            yy, xx = np.ogrid[:size, :size]
            mask = (xx - size // 2) ** 2 + (yy - size // 2) ** 2 <= r**2
            patch[mask] = 1.0
        else:
            patch = np.clip(rng.normal(0.5, 0.3, (size, size)), 0, 1)
        out.append(patch)
    return out


# ----------------------------------------------------------- convolução --


def test_conv2d_forward_formas():
    x = np.zeros((2, 1, 8, 8))
    w = np.zeros((3, 1, 3, 3))
    b = np.zeros(3)
    out = conv2d_forward(x, w, b)
    assert out.shape == (2, 3, 8, 8)


def test_conv2d_backward_exato_por_diferencas_finitas():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (1, 1, 6, 6))
    w = rng.normal(0, 0.1, (2, 1, 3, 3))
    b = rng.normal(0, 0.1, (2,))
    grad_out = rng.normal(0, 1, (1, 2, 6, 6))

    def loss():
        return float(np.sum(conv2d_forward(x, w, b) * grad_out))

    dw, db, dx = conv2d_backward(grad_out, x, w)
    eps = 1e-6
    for name, P, analytic in (("w", w, dw), ("b", b, db)):
        numeric = np.zeros_like(P)
        for i in range(P.size):
            orig = P.ravel()[i]
            P.ravel()[i] = orig + eps
            l1 = loss()
            P.ravel()[i] = orig - eps
            l2 = loss()
            P.ravel()[i] = orig
            numeric.ravel()[i] = (l1 - l2) / (2 * eps)
        assert np.max(np.abs(numeric - analytic)) < 1e-6, name
    # dx: verificação numérica com perturbação do input
    numeric_dx = np.zeros_like(dx)
    for i in range(x.size):
        orig = x.ravel()[i]
        x.ravel()[i] = orig + eps
        l1 = loss()
        x.ravel()[i] = orig - eps
        l2 = loss()
        x.ravel()[i] = orig
        numeric_dx.ravel()[i] = (l1 - l2) / (2 * eps)
    assert np.max(np.abs(numeric_dx - dx)) < 1e-6


# ------------------------------------------------------------- gradientes --


def _check_gradients(model, x, loss_fn, grad_fn):
    eps = 1e-6
    grads = grad_fn()
    for name, P in (
        ("conv1_w", model.conv1_w),
        ("conv1_b", model.conv1_b),
        ("conv2_w", model.conv2_w),
        ("conv2_b", model.conv2_b),
        *(
            (("conv3_w", model.conv3_w), ("conv3_b", model.conv3_b))
            if model.mode == "residual"
            else (("head_w", model.head_w), ("head_b", model.head_b))
        ),
    ):
        numeric = np.zeros_like(P)
        for i in range(P.size):
            orig = P.ravel()[i]
            P.ravel()[i] = orig + eps
            l1 = loss_fn()
            P.ravel()[i] = orig - eps
            l2 = loss_fn()
            P.ravel()[i] = orig
            numeric.ravel()[i] = (l1 - l2) / (2 * eps)
        assert np.max(np.abs(numeric - grads[name])) < 1e-6, name


def test_gradientes_residual_exatos():
    rng = np.random.default_rng(1)
    model = _residual_model()
    x = rng.normal(0, 1, (1, 1, 8, 8))
    target = rng.normal(0, 1, (1, 1, 8, 8))

    def loss():
        out, _ = model.forward(x)
        return float(np.mean((out - target) ** 2))

    def grad_fn():
        out, cache = model.forward(x)
        err = out - target
        return model.backward_residual(err * 2 / err.size, cache)

    _check_gradients(model, x, loss, grad_fn)


def test_gradientes_classificadora_exatos():
    rng = np.random.default_rng(2)
    model = _classify_model()
    x = rng.normal(0, 1, (1, 1, 6, 6))
    y = np.array([[0.0, 1.0]])

    def loss():
        probs, _ = model.forward(x)
        return float(-np.sum(y * np.log(probs + 1e-12)))

    def grad_fn():
        probs, cache = model.forward(x)
        return model.backward_classify(probs - y, cache)

    _check_gradients(model, x, loss, grad_fn)


def test_modelo_modo_invalido_levanta():
    with pytest.raises(ValueError, match="residual"):
        SmallCNN(mode="xpto")


# ---------------------------------------------------------------- treino --


def test_fit_residual_converge(tmp_path):
    rng = np.random.default_rng(0)
    pairs = []
    for i in range(8):
        clean = _patches(1, "disk", size=32, seed=i)[0]
        noisy = np.clip(clean + rng.normal(0, 0.05, clean.shape), 0, 1)
        pairs.append((noisy, clean))
    model = _residual_model()
    fit_residual([(noisy, clean) for noisy, clean in pairs], model=model, epochs=1, seed=4)[1]
    model2, report = fit_residual(pairs, model=_residual_model(), epochs=10, lr=0.1, seed=4)
    assert isinstance(report, FitReport)
    assert report.epochs >= 1
    assert report.best_loss < 1.0
    path = model2.save(tmp_path / "enh.npz")
    assert path.exists()


def test_fit_residual_sem_pares_levanta():
    with pytest.raises(ValueError, match="Sem pares"):
        fit_residual([])


def test_fit_classifier_separa_discos(tmp_path):
    pos = _patches(10, "disk", seed=1)
    neg = _patches(10, "noise", seed=2)
    model, report = fit_classifier(pos, neg, epochs=16, lr=0.5, seed=5)
    assert report.best_loss < 0.68
    batch = _to_batch(pos[:3] + neg[:3])
    probs = model.predict_class(batch)
    assert probs[:3].mean() - probs[3:].mean() > 0.15
    path = model.save(tmp_path / "filter.npz")
    assert path.exists()


def test_fit_classifier_sem_patches_levanta():
    with pytest.raises(ValueError, match="Sem patches"):
        fit_classifier([], [])


# ---------------------------------------------------------- persistência --


def test_smallcnn_roundtrip_residual(tmp_path):
    model = _residual_model()
    loaded = SmallCNN.load(model.save(tmp_path / "m.npz"))
    assert loaded is not None
    assert loaded.mode == "residual"
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (1, 1, 8, 8))
    assert np.allclose(model.forward(x)[0], loaded.forward(x)[0])


def test_smallcnn_roundtrip_classify(tmp_path):
    model = _classify_model()
    loaded = SmallCNN.load(model.save(tmp_path / "m.npz"))
    assert loaded is not None
    assert loaded.mode == "classify"


def test_smallcnn_load_inexistente_e_corrompido(tmp_path):
    assert SmallCNN.load(tmp_path / "nao_existe.npz") is None
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"lixo")
    assert SmallCNN.load(bad) is None


# -------------------------------------------------------------- wrappers --


def test_residual_enhancer_sem_modelo_devolve_intacta():
    enhancer = ResidualEnhancer(model=None)
    assert enhancer.available is False
    image = (np.random.default_rng(0).normal(0.5, 0.1, (40, 40, 3)) * 255).astype(np.uint8)
    assert np.array_equal(enhancer.apply(image), image)


def test_residual_enhancer_aplica_com_modelo(tmp_path):
    rng = np.random.default_rng(3)
    clean = _patches(1, "disk", size=PATCH_SIZE, seed=3)[0] * 255
    pairs = []
    for _ in range(6):
        noisy = np.clip(clean + rng.normal(0, 20, clean.shape), 0, 255).astype(np.uint8)
        pairs.append((noisy, clean.astype(np.uint8)))
    model, _ = fit_residual(pairs, model=_residual_model(), epochs=4, seed=9)
    enhancer = ResidualEnhancer(model=model)
    assert enhancer.available is True
    bgr = np.repeat(clean.astype(np.uint8)[:, :, None], 3, axis=2)
    out = enhancer.apply(bgr)
    assert out.shape == bgr.shape
    assert out.dtype == np.uint8


def test_disk_filter_sem_modelo(tmp_path):
    filtro = DiskFilter(model=None, model_path=tmp_path / "ausente.npz")
    assert filtro.available is False
    image = np.zeros((64, 64), dtype=np.uint8)
    assert filtro.confidence(image, 32, 32, 8) == 0.5
    disks = [type("D", (), {"cx": 32, "cy": 32, "radius": 8})()]
    assert filtro.filter_disks(disks, image, 0.9) == disks


def test_disk_filter_patch_forma(tmp_path):
    model, _ = fit_classifier(_patches(4, "disk", seed=1), _patches(4, "noise", seed=2), epochs=2, seed=6)
    filtro = DiskFilter(model=model)
    image = np.zeros((100, 100), dtype=np.uint8)
    patch = filtro.patch(image, 50, 50, 10)
    assert patch.shape == (PATCH_SIZE, PATCH_SIZE)
    assert filtro.patch(image, 2, 2, 5).shape == (PATCH_SIZE, PATCH_SIZE)
    assert filtro.confidence(image, 50, 50, 10) >= 0.0


def test_disk_filter_nunca_esvazia_a_lista(tmp_path):
    model, _ = fit_classifier(_patches(4, "disk", seed=1), _patches(4, "noise", seed=2), epochs=2, seed=6)
    filtro = DiskFilter(model=model)
    image = np.zeros((100, 100), dtype=np.uint8)
    disks = [type("D", (), {"cx": 10, "cy": 10, "radius": 8})()]
    kept = filtro.filter_disks(disks, image, 0.99)
    assert len(kept) == 1
    assert len(filtro.filter_disks([], image, 0.9)) == 0
    assert filtro.filter_disks(disks, image, 0.0) == disks


# --------------------------------------------------------------- utilitários --


def test_to_batch_uint8_vs_float():
    uint8 = [np.full((16, 16), 128, dtype=np.uint8)]
    float_img = [np.full((16, 16), 0.5)]
    assert _to_batch(uint8)[0, 0, 0, 0] == pytest.approx(128 / 255)
    assert _to_batch(float_img)[0, 0, 0, 0] == pytest.approx(0.5)


def test_to_batch_redimensiona_e_aceita_rgb():
    rgb = [np.zeros((20, 24, 3), dtype=np.uint8)]
    batch = _to_batch(rgb, size=PATCH_SIZE)
    assert batch.shape == (1, 1, PATCH_SIZE, PATCH_SIZE)


def test_to_batch_clampa():
    batch = _to_batch([np.full((8, 8), 500.0)])
    assert np.max(batch) == 1.0


def test_forward_nao_mutado_entre_chamadas():
    model = _residual_model()
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (1, 1, 8, 8))
    a, _ = model.forward(x)
    b, _ = model.forward(x)
    assert np.array_equal(a, b)