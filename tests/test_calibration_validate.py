"""Testes da validação: IoU, correspondência, agregados e sugestões."""

from __future__ import annotations

from astroframe.calibration.validate import (
    circle_iou,
    match_circles,
    suggest_parameters,
    validate_all,
    validate_item,
)
from astroframe.config import AstroFrameConfig
from astroframe.core.stabilizer import DiskDetection

MANUAL = DiskDetection(100, 100, 50)
DETECTED = DiskDetection(105, 98, 45)


def test_circle_iou_identicos():
    assert circle_iou(MANUAL, DiskDetection(100, 100, 50)) == 1.0


def test_circle_iou_disjuntos():
    assert circle_iou(MANUAL, DiskDetection(300, 300, 50)) == 0.0


def test_circle_iou_contidos():
    assert abs(circle_iou(DiskDetection(0, 0, 20), DiskDetection(0, 0, 10)) - 0.25) < 1e-9


def test_circle_iou_sobrepostos_entre_zero_e_um():
    iou = circle_iou(DETECTED, MANUAL)
    assert 0.0 < iou < 1.0
    assert abs(circle_iou(DETECTED, MANUAL) - circle_iou(MANUAL, DETECTED)) < 1e-9


def test_match_circles_perfeito():
    manual = [DiskDetection(10, 10, 5), DiskDetection(50, 50, 8)]
    detected = [DiskDetection(11, 9, 5), DiskDetection(49, 51, 8)]
    pairs, unmatched_m, unmatched_d = match_circles(manual, detected)
    assert len(pairs) == 2
    assert not unmatched_m and not unmatched_d


def test_match_circles_com_faltas_e_extras():
    manual = [MANUAL, DiskDetection(300, 300, 40)]
    detected = [DETECTED]
    pairs, unmatched_m, unmatched_d = match_circles(manual, detected)
    assert pairs == [(0, 0)]
    assert unmatched_m == {1}
    assert unmatched_d == set()


def test_match_circles_abaixo_do_limiar_nao_corresponde():
    pairs, unmatched_m, unmatched_d = match_circles(
        [MANUAL], [DiskDetection(200, 100, 50)], iou_threshold=0.5
    )
    assert pairs == []
    assert unmatched_m == {0}
    assert unmatched_d == {0}


def test_match_circles_vazio():
    pairs, unmatched_m, unmatched_d = match_circles([], [])
    assert (pairs, unmatched_m, unmatched_d) == ([], set(), set())


def test_match_circles_conflito_greedy_nao_reutiliza():
    manual = [DiskDetection(100, 100, 50)]
    detected = [
        DiskDetection(102, 99, 49),
        DiskDetection(90, 110, 47),
    ]
    pairs, unmatched_m, unmatched_d = match_circles(manual, detected)
    assert len(pairs) == 1
    assert unmatched_m == set()
    assert unmatched_d == {1}


def test_validate_item():
    report = validate_item("x.jpg", [MANUAL, DiskDetection(300, 300, 40)], [DETECTED])
    assert report.label == "x.jpg"
    assert report.n_manual == 2
    assert report.n_detected == 1
    assert report.n_matched == 1
    assert report.n_false_negatives == 1
    assert report.n_false_positives == 0
    assert report.mean_iou is not None and 0.0 < report.mean_iou < 1.0
    assert report.mean_center_error is not None and report.mean_center_error > 0
    assert report.mean_radius_error_pct is not None


def test_validate_item_sem_correspondencias():
    report = validate_item("vazia.jpg", [MANUAL], [DiskDetection(300, 300, 40)])
    assert report.n_matched == 0
    assert report.mean_iou is None
    assert report.mean_center_error is None
    assert report.mean_radius_error_pct is None


def test_validate_all_agrega():
    report = validate_all(
        [
            ("a.jpg", [MANUAL], [DETECTED]),
            ("b.jpg", [DiskDetection(10, 10, 5)], [DiskDetection(11, 11, 5), DiskDetection(200, 200, 5)]),
        ]
    )
    assert len(report.items) == 2
    assert report.total_manual == 2
    assert report.total_detected == 3
    assert report.total_matched == 2
    assert report.total_false_negatives == 0
    assert report.total_false_positives == 1
    assert report.recall == 1.0
    assert report.precision == 2 / 3
    assert report.has_ground_truth
    assert report.score is not None and 0 < report.score < 100
    assert report.mean_center_error is not None
    assert report.mean_radius_error_pct is not None


def test_validate_all_sem_ground_truth():
    report = validate_all([("a.jpg", [], [DETECTED])])
    assert not report.has_ground_truth
    assert report.score is None
    assert report.recall == 0.0
    assert report.mean_center_error is None


def test_sugestoes_sem_ground_truth():
    report = validate_all([("a.jpg", [], [])])
    suggestions = suggest_parameters(report)
    assert len(suggestions) == 1
    assert "Guardar ajustes" in suggestions[0]


def test_sugestoes_falsos_negativos_e_positivos():
    report = validate_all([("a.jpg", [MANUAL], [DETECTED, DiskDetection(300, 300, 40)])])
    suggestions = suggest_parameters(report)
    text = " ".join(suggestions)
    assert "min_radius" in text and "param2" in text


def test_sugestoes_falsos_negativos_apenas():
    report = validate_all([("a.jpg", [MANUAL, DiskDetection(300, 300, 40)], [DETECTED])])
    suggestions = suggest_parameters(report)
    assert any("não detetado" in s for s in suggestions)


def test_sugestoes_raio_subestimado():
    report = validate_all([("a.jpg", [DiskDetection(100, 100, 50)], [DiskDetection(105, 98, 40)])])
    suggestions = suggest_parameters(report)
    assert any("raio detetado é sistematicamente menor" in s for s in suggestions)


def test_sugestoes_tudo_ok():
    report = validate_all([("a.jpg", [MANUAL], [DiskDetection(101, 100, 50)])])
    suggestions = suggest_parameters(report)
    assert any("Sem ajustes sugeridos" in s for s in suggestions)


def test_sugestoes_respeitam_config_atual():
    report = validate_all([("a.jpg", [MANUAL], [])])
    config = AstroFrameConfig()
    config.stabilizer.param2 = 77
    text = " ".join(suggest_parameters(report, config))
    assert "atualmente 77" in text
