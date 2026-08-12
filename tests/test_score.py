"""Testes da avaliação automática (0–5 estrelas)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from astroframe.ai.score import StarRating, score_from_stars, score_image, stars_text
from astroframe.config import AstroFrameConfig
from astroframe.core.stabilizer import DiskDetection
from tests.helpers import make_disk_image


def _clean_enhanced(height: int = 300, width: int = 360, radius: int = 80) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(image, (width // 2, height // 2), radius, (200,) * 3, -1)
    return image


def _in_frame(detection: DiskDetection) -> DiskDetection:
    return DiskDetection(detection.cx, detection.cy, detection.radius)


def test_score_disco_perfeito_fundo_preto_alta_avaliacao():
    image = _clean_enhanced()
    rating = score_image(image, DiskDetection(180, 150, 80), AstroFrameConfig())
    assert rating.stars >= 4.5
    assert rating.metrics["background"] == 1.0
    assert rating.metrics["limb"] >= 0.9


def test_score_fundo_claro_penaliza_background():
    cfg = AstroFrameConfig()
    image = np.full((300, 360, 3), 80, dtype=np.uint8)
    cv2.circle(image, (180, 150), 80, (200,) * 3, -1)
    clean = score_image(_clean_enhanced(), DiskDetection(180, 150, 80), cfg)
    dirty = score_image(image, DiskDetection(180, 150, 80), cfg)
    assert dirty.metrics["background"] < clean.metrics["background"]
    assert dirty.stars < clean.stars


def test_score_ruido_na_coroa_penaliza_noise():
    rng = np.random.default_rng(7)
    noisy = _clean_enhanced()
    noise = rng.normal(0, 18, noisy.shape)
    noisy = np.clip(noisy.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    det = DiskDetection(180, 150, 80)
    clean = score_image(_clean_enhanced(), det, AstroFrameConfig())
    rained = score_image(noisy, det, AstroFrameConfig())
    assert rained.metrics["noise"] < clean.metrics["noise"]
    assert rained.stars < clean.stars


def test_score_limb_union_vazia_e_aneis_vazios():
    cfg = AstroFrameConfig()
    cfg.polish.corona_scale = 1.0
    rating = score_image(np.zeros((10, 10, 3), dtype=np.uint8), DiskDetection(1000, 1000, 10), cfg)
    assert rating.metrics["limb"] == 0.0
    assert rating.metrics["noise"] == 0.0
    assert rating.metrics["contrast"] == 0.0


def test_score_background_sem_amostra_fora_da_mascara():
    cfg = AstroFrameConfig()
    image = np.full((10, 10, 3), 100, dtype=np.uint8)
    rating = score_image(image, DiskDetection(5, 5, 6), cfg)
    assert rating.metrics["background"] == 1.0


def test_score_reflexos_inferiores_ao_minimo_nao_penalizam(monkeypatch):
    primary = DiskDetection(180, 150, 80)
    monkeypatch.setattr(
        "astroframe.ai.score.find_all_disks",
        lambda img, config=None: [primary, DiskDetection(80, 70, 4)],
    )
    rating = score_image(_clean_enhanced(), primary, AstroFrameConfig())
    assert rating.metrics["reflections"] == 1.0


def test_score_limbo_irregular_penaliza_redondeza():
    image = _clean_enhanced()
    cv2.rectangle(image, (180, 210), (260, 230), (0, 0, 0), -1)
    det = DiskDetection(180, 150, 80)
    perfect = score_image(_clean_enhanced(), det, AstroFrameConfig())
    notched = score_image(image, det, AstroFrameConfig())
    assert notched.metrics["limb"] < perfect.metrics["limb"]
    assert notched.stars < perfect.stars


def test_score_reflexos_penalizam(monkeypatch):
    primary = DiskDetection(180, 150, 80)
    ghost = DiskDetection(80, 70, 25)
    clean_img = _clean_enhanced()
    haunted_img = _clean_enhanced()
    cv2.circle(haunted_img, (80, 70), 25, (150,) * 3, -1)

    def fake_disks(image, config=None):
        if int(image[70, 80].mean()) > 0:
            return [primary, ghost]
        return [primary]

    monkeypatch.setattr("astroframe.ai.score.find_all_disks", fake_disks)
    cfg = AstroFrameConfig()
    clean = score_image(clean_img, primary, cfg)
    haunted = score_image(haunted_img, primary, cfg)
    assert haunted.metrics["reflections"] < clean.metrics["reflections"]
    assert haunted.stars < clean.stars


def test_score_sem_deteccao_nao_falha_e_limita_estrelas():
    rating = score_image(_clean_enhanced(), None, AstroFrameConfig())
    assert 0.0 <= rating.stars <= 5.0
    assert rating.metrics["limb"] == 0.0


def test_score_estrelas_limitadas_entre_0_e_5():
    rating = score_image(np.zeros((50, 50, 3), dtype=np.uint8), None, AstroFrameConfig())
    assert 0.0 <= rating.stars <= 5.0


def test_stars_text_representacoes():
    assert "★" in stars_text(5.0) and "☆" not in stars_text(5.0)
    assert "☆" in stars_text(1.0)
    assert "½" in stars_text(2.5)
    assert "4.4" in stars_text(4.4)
    assert "0.0" in stars_text(0.0)
    assert "Excelente" in stars_text(5.0)
    assert "Mau" in stars_text(1.0)


def test_score_from_stars_clampa():
    assert score_from_stars(9.0).stars == 5.0
    assert score_from_stars(-2.0).stars == 0.0
    assert score_from_stars(3.5).stars == 3.5


def test_star_rating_label_usado():
    rating = StarRating(stars=4.0, metrics={})
    assert rating.label == stars_text(4.0)


@pytest.mark.parametrize("bad", [False, True])
def test_score_imagem_real_do_pipeline_funciona(bad):
    image, cx, cy = make_disk_image(add_noise=bad)
    from astroframe.core.pipeline import process_image

    result = process_image(image)
    detection = result.detection
    assert detection is not None
    score_image(result.enhanced, _in_frame(detection), AstroFrameConfig())


def test_score_deteccao_raio_zero_usa_caminho_sem_deteccao():
    rating = score_image(_clean_enhanced(), DiskDetection(180, 150, 0), AstroFrameConfig())
    assert rating.metrics["limb"] == 0.0
