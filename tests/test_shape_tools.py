"""Testes das ferramentas de formas e de gestão de camadas da calibração."""

from __future__ import annotations

import gradio as gr
import numpy as np

from astroframe.calibration.circles import circles_to_layers
from astroframe.calibration.shapes import (
    content_bounds,
    content_center,
    normalize_shape,
    recenter_layer,
    scale_layer,
    shape_layer,
    translate_layer,
)
from astroframe.core.stabilizer import DiskDetection
from astroframe.ui.calibration_app import (
    _selected_index,
    add_shape_layer,
    auto_detect_view,
    auto_detect_with_params,
    clear_layers,
    load_item_view,
    move_layer,
    remove_layer,
    resize_layer,
    shape_choices,
    store_click,
    toggle_ratio,
)
from tests.test_calibration_app import _make_sample_dir, _sample_key

_SIZE = (200, 300)


def _frame() -> np.ndarray:
    return np.zeros((*_SIZE, 3), dtype=np.uint8)


def _editor_value(n_layers: int = 1) -> dict:
    return circles_to_layers(_frame(), [DiskDetection(60, 50, 20) for _ in range(n_layers)])


# ---------------------------------------------------------------- shapes.py


def test_normalize_shape():
    assert normalize_shape("Círculo") == "círculo"
    assert normalize_shape("elipse") == "elipse"
    assert normalize_shape("Elipse") == "elipse"
    assert normalize_shape(None) == "círculo"
    assert normalize_shape("???") == "círculo"


def test_shape_layer_circulo_desenha_anel_e_preenchimento():
    layer = shape_layer("Círculo", 80, 1.0, 150, 100, _SIZE)
    assert layer.shape == (*_SIZE, 4)
    assert layer[..., 3].max() == 255
    assert layer[100, 150, 3] > 0
    assert layer[100, 110, 3] > 0
    assert layer[100, 190, 3] > 0
    assert layer[100, 50, 3] == 0
    assert layer[0, 0, 3] == 0


def test_shape_layer_elipse_respeita_proporcao():
    layer = shape_layer("Elipse", 80, 0.5, 150, 100, _SIZE)
    bounds = content_bounds(layer)
    assert bounds is not None
    x0, y0, x1, y1 = bounds
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    assert abs(width / height - 2.0) <= 0.15
    layer_circle = shape_layer("Círculo", 80, 1.0, 150, 100, _SIZE)
    cb = content_bounds(layer_circle)
    assert abs((cb[2] - cb[0] + 1) - (cb[3] - cb[1] + 1)) <= 1


def test_shape_layer_centro_fora_da_imagem_e_recortado():
    layer = shape_layer("Círculo", 80, 1.0, 500, 500, _SIZE)
    bounds = content_bounds(layer)
    assert bounds is not None
    assert bounds[2] < 300 and bounds[3] < 200


def test_translate_layer_move_conteudo():
    layer = shape_layer("Círculo", 60, 1.0, 100, 80, _SIZE)
    moved = translate_layer(layer, 30, -20)
    before = content_center(layer)
    after = content_center(moved)
    assert after == (before[0] + 30, before[1] - 20)


def test_scale_layer_redimensiona_para_diametro_alvo():
    layer = shape_layer("Círculo", 60, 1.0, 100, 80, _SIZE)
    scaled = scale_layer(layer, 120)
    before = content_bounds(layer)
    after = content_bounds(scaled)
    before_size = max(before[2] - before[0], before[3] - before[1])
    after_size = max(after[2] - after[0], after[3] - after[1])
    assert abs(after_size - 120) <= 3
    assert after_size > before_size
    assert content_center(scaled) == content_center(layer)


def test_scale_layer_camada_vazia_inalterada():
    empty = np.zeros((*_SIZE, 4), dtype=np.uint8)
    np.testing.assert_array_equal(scale_layer(empty, 100), empty)
    assert content_bounds(empty) is None


def test_recenter_layer_move_para_ponto():
    layer = shape_layer("Círculo", 60, 1.0, 100, 80, _SIZE)
    moved = recenter_layer(layer, 250, 150)
    assert content_center(moved) == (250, 150)


def test_content_bounds_rgb_e_camadas_vazias():
    rgb = np.zeros((*_SIZE, 3), dtype=np.uint8)
    rgb[10:20, 5:15] = 200
    x0, y0, x1, y1 = content_bounds(rgb)
    assert (x0, y0, x1, y1) == (5, 10, 14, 19)
    gray2d = np.zeros((*_SIZE,), dtype=np.uint8)
    gray2d[30:35, 40:45] = 255
    assert content_bounds(gray2d) == (40, 30, 44, 34)
    assert content_bounds(np.zeros((0, 0, 4), dtype=np.uint8)) is None
    assert content_center(np.zeros((*_SIZE, 4), dtype=np.uint8)) is None
    np.testing.assert_array_equal(
        recenter_layer(np.zeros((*_SIZE, 4), dtype=np.uint8), 5, 5), np.zeros((*_SIZE, 4), dtype=np.uint8)
    )


def test_translate_layer_sem_deslocamento_devolve_mesma_camada():
    layer = shape_layer("Círculo", 60, 1.0, 100, 80, _SIZE)
    np.testing.assert_array_equal(translate_layer(layer, 0, 0), layer)


# ------------------------------------------------------- handlers (camadas)


def test_shape_choices_e_selecao():
    assert shape_choices(None) == []
    assert shape_choices(_editor_value(2)) == ["Camada 1", "Camada 2"]
    assert _selected_index(_editor_value(3), "Camada 2") == 1
    assert _selected_index(_editor_value(3), None) == 2
    assert _selected_index(_editor_value(3), "lixo") == 2
    assert _selected_index(None, "Camada 1") is None


def test_layers_update_escolhas_e_valor():
    update = add_shape_layer(None, "Círculo", 60, 1.0, None)[1]
    assert update["choices"] == [] and update["value"] is None
    update = add_shape_layer(_editor_value(1), "Círculo", 60, 1.0, None)[1]
    assert update["choices"] == ["Camada 1", "Camada 2"]
    assert update["value"] == "Camada 2"


def test_add_shape_layer_sem_amostra():
    value, update, status = add_shape_layer(None, "Círculo", 60, 1.0, None)
    assert value == {} and update["choices"] == []
    assert "Carrega uma amostra" in status


def test_add_shape_layer_circulo_no_centro_por_omissao():
    value, _, status = add_shape_layer(_editor_value(), "Círculo", 60, 1.0, None)
    assert len(value["layers"]) == 2
    assert "círculo" in status
    assert content_center(value["layers"][1]) == (150, 100)


def test_add_shape_layer_elipse_no_ponto_clicado():
    value, _, status = add_shape_layer(_editor_value(), "Elipse", 80, 0.5, (250, 30))
    assert "elipse" in status
    layer = value["layers"][1]
    assert content_center(layer) == (250, 30)
    bounds = content_bounds(layer)
    assert abs((bounds[2] - bounds[0] + 1) / (bounds[3] - bounds[1] + 1) - 2.0) <= 0.15


def test_remove_layer_selecionada_e_ultima():
    value = _editor_value(3)
    result, _, status = remove_layer(value, "Camada 2")
    assert len(result["layers"]) == 2
    assert "2 eliminada" in status
    result, _, _ = remove_layer(result, None)
    assert len(result["layers"]) == 1
    value, update, status = remove_layer(None, None)
    assert "Sem camadas" in status


def test_clear_layers_mantem_fundo():
    value, update, status = clear_layers(_editor_value(3))
    assert value["layers"] == [] and update["choices"] == []
    assert "limpas" in status
    assert value["background"] is not None


def test_move_layer_para_o_ponto_clicado():
    value = _editor_value(2)
    result, _, status = move_layer(value, "Camada 1", (120, 100))
    assert content_center(result["layers"][0]) == (120, 100)
    assert "movida" in status


def test_move_layer_sem_ponto():
    value = _editor_value()
    result, _, status = move_layer(value, "Camada 1", None)
    assert "Clica na imagem" in status
    assert len(result["layers"]) == 1


def test_clear_layers_sem_amostra():
    value, update, status = clear_layers(None)
    assert value == {} and update["choices"] == []
    assert "amostra" in status


def test_move_layer_sem_camadas():
    value, update, status = move_layer(None, "Camada 1", (10, 10))
    assert value == {} and update["choices"] == []
    assert "Sem camadas" in status


def test_resize_layer_sem_camadas():
    value, _, status = resize_layer(None, "Camada 1", 60)
    assert value == {}
    assert "Sem camadas" in status


def test_resize_layer_para_diametro():
    value = _editor_value()
    result, _, status = resize_layer(value, "Camada 1", 100)
    assert "redimensionada" in status
    bounds = content_bounds(result["layers"][0])
    size = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    assert abs(size - 100) <= 5


def test_store_click_regista_ponto():
    evt = gr.SelectData(None, {"index": [12, 34], "value": None})
    center, text = store_click(evt)
    assert center == (12, 34)
    assert "12" in text and "34" in text


def test_store_click_index_invalido():
    evt = gr.SelectData(None, {"index": None, "value": None})
    center, text = store_click(evt)
    assert center is None
    assert "Clique não registado" in text


def test_store_click_sem_evento():
    center, text = store_click(None)
    assert center is None
    assert "Clica na imagem" in text


def test_toggle_ratio():
    assert toggle_ratio("Círculo")["visible"] is False
    assert toggle_ratio("Elipse")["visible"] is True


# ------------------------------------------------------- deteção automática


def test_load_item_view_devolve_preview_e_dropdown(tmp_path, monkeypatch):
    root = _make_sample_dir(tmp_path)
    monkeypatch.setattr(
        "astroframe.ui.calibration_app.find_disks_for_calibration",
        lambda frame, config=None, expected_n=None: [DiskDetection(60, 50, 30)],
    )
    value, info, preview, update, center, click_text = load_item_view(_sample_key(root), str(root))
    assert preview is not None
    assert update["choices"] == ["Camada 1"]
    assert center is None
    assert "centro da forma" in click_text
    assert "deteção automática" in info


def test_auto_detect_view_sincroniza_dropdown(tmp_path, monkeypatch):
    root = _make_sample_dir(tmp_path)
    monkeypatch.setattr(
        "astroframe.ui.calibration_app.find_disks_for_calibration",
        lambda frame, config=None, expected_n=None: [DiskDetection(60, 50, 30)],
    )
    value, info, update = auto_detect_view(_sample_key(root), str(root))
    assert len(value["layers"]) == 1
    assert update["value"] == "Camada 1"


def test_auto_detect_with_params_aplica_sliders(tmp_path, monkeypatch):
    root = _make_sample_dir(tmp_path)
    seen = {}

    def fake_detect(frame, config=None, expected_n=None):
        seen["param2"] = config.stabilizer.param2
        seen["max_radius"] = config.stabilizer.max_radius
        return [DiskDetection(60, 50, 30)]

    monkeypatch.setattr("astroframe.ui.calibration_app.find_disks_for_calibration", fake_detect)
    value, info, config = auto_detect_with_params(
        _sample_key(root), str(root), None, param2=75, max_radius=500
    )
    assert seen == {"param2": 75, "max_radius": 500}
    assert config.stabilizer.param2 == 75
    assert len(value["layers"]) == 1


def test_build_calibration_app_com_formas(tmp_path):
    from astroframe.ui.calibration_app import build_calibration_app

    app = build_calibration_app(samples_dir=str(_make_sample_dir(tmp_path)))
    assert app is not None


def test_fluxo_completo_editor(tmp_path, monkeypatch):
    """Adicionar → mover → redimensionar → eliminar num editor real."""
    root = _make_sample_dir(tmp_path)
    monkeypatch.setattr(
        "astroframe.ui.calibration_app.find_disks_for_calibration",
        lambda frame, config=None, expected_n=None: [DiskDetection(60, 50, 30)],
    )
    value, _, _, _, _, _ = load_item_view(_sample_key(root), str(root))
    value, _, _ = add_shape_layer(value, "Elipse", 60, 0.5, (20, 20))
    assert len(value["layers"]) == 2
    value, _, _ = move_layer(value, "Camada 2", (60, 50))
    assert content_center(value["layers"][1]) == (60, 50)
    value, _, _ = resize_layer(value, "Camada 2", 50)
    bounds = content_bounds(value["layers"][1])
    assert max(bounds[2] - bounds[0], bounds[3] - bounds[1]) <= 53
    value, _, _ = remove_layer(value, "Camada 2")
    assert len(value["layers"]) == 1
