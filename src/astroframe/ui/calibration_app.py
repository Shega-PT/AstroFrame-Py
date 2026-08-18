"""Interface Gradio de calibração: ajuste manual dos círculos e validação.

Fluxo: escolhe uma amostra da pasta de exemplos (`samples/` por omissão) →
a **deteção automática corre de imediato** (ou carrega o ground truth
guardado) → ajusta com as **ferramentas de formas**: cada astro é uma camada
selecionável que pode ser **movida** (clica na pré-visualização para o novo
centro), **redimensionada** (slider de diâmetro) ou **eliminada**; o seletor
de formas (círculo/elipse) adiciona objetos novos e o pincel serve para
detalhes menores → **Guardar ajustes** grava o ground truth em
`calibration.json` → **Validar todas** compara a deteção automática com o
ground truth em todas as amostras e devolve o relatório (recall, precisão,
IoU, erros) + sugestões de parâmetros. Os sliders de deteção re-correm a
deteção automaticamente ao largar.

Todos os handlers são funções de módulo testáveis; o `build_calibration_app`
é apenas a montagem Gradio.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import gradio as gr

from astroframe.calibration.circles import circles_to_layers, layers_to_circles
from astroframe.calibration.scan import load_frame, scan_samples
from astroframe.calibration.shapes import (
    recenter_layer,
    scale_layer,
    shape_layer,
)
from astroframe.calibration.store import CalibrationItem, CalibrationStore
from astroframe.calibration.validate import suggest_parameters, validate_all
from astroframe.config import AstroFrameConfig
from astroframe.core.stabilizer import find_all_disks
from astroframe.paths import calibration_json
from astroframe.ui.gradio_app import _from_pipeline

logger = logging.getLogger(__name__)

_PCT = 100.0


def _samples_dir_to_root(samples_dir: str) -> Path:
    return Path(samples_dir).expanduser()


def _find_sample(samples: list, key: str):
    """Localiza a `SampleRef` pela chave; levanta KeyError se não existir."""
    for sample in samples:
        if sample.key == key:
            return sample
    raise KeyError(f"Amostra desconhecida: {key}")


def load_item_payload(
    key: str | None,
    samples_dir: str,
    config: AstroFrameConfig | None = None,
    store: CalibrationStore | None = None,
) -> tuple[dict, str]:
    """Carrega o item escolhido e devolve o valor do ImageEditor + informação.

    Círculos guardados (ground truth) têm prioridade; sem ground truth, a
    deteção automática serve de ponto de partida. Devolve (value, info_html).
    """
    config = config or AstroFrameConfig()
    store = store or CalibrationStore(_calibration_path(samples_dir))
    if not key:
        return {}, "Escolhe uma amostra na lista."
    samples = scan_samples(_samples_dir_to_root(samples_dir))
    sample = _find_sample(samples, key)
    frame = load_frame(sample)
    stored = store.get_item(key)
    if stored is not None:
        circles = list(stored.circles)
        info = (
            f"<b>{sample.label}</b> · {frame.shape[1]}×{frame.shape[0]} · "
            f"{len(circles)} círculo(s) guardado(s)."
        )
    else:
        circles = find_all_disks(frame, config)
        info = (
            f"<b>{sample.label}</b> · {frame.shape[1]}×{frame.shape[0]} · "
            f"{len(circles)} círculo(s) da deteção automática (ajusta e guarda)."
        )
    return circles_to_layers(_from_pipeline(frame), circles), info


def auto_detect_payload(
    key: str | None,
    samples_dir: str,
    config: AstroFrameConfig | None = None,
) -> tuple[dict, str]:
    """Re-corre a deteção automática no item e devolve os círculos para edição."""
    config = config or AstroFrameConfig()
    if not key:
        return {}, "Escolhe uma amostra na lista."
    samples = scan_samples(_samples_dir_to_root(samples_dir))
    sample = _find_sample(samples, key)
    frame = load_frame(sample)
    circles = find_all_disks(frame, config)
    info = (
        f"<b>{sample.label}</b> · {len(circles)} círculo(s) detetado(s) — "
        "ajusta à mão e carrega em 'Guardar ajustes'."
    )
    return circles_to_layers(_from_pipeline(frame), circles), info


def save_item_circles(
    editor_value: dict | None,
    key: str | None,
    samples_dir: str,
    store: CalibrationStore | None = None,
) -> str:
    """Extrai os círculos desenhados e grava o ground truth do item."""
    store = store or CalibrationStore(_calibration_path(samples_dir))
    if not key:
        return "Escolhe uma amostra na lista."
    if not editor_value:
        return "Sem imagem no editor — escolhe uma amostra primeiro."
    background = editor_value.get("background")
    layers = editor_value.get("layers")
    if background is None or layers is None:
        return "O editor ainda não devolveu camadas — tenta novamente."
    samples = scan_samples(_samples_dir_to_root(samples_dir))
    sample = _find_sample(samples, key)
    circles = layers_to_circles(layers)
    height, width = background.shape[:2]
    store.upsert_item(
        key,
        CalibrationItem(
            path=sample.path.relative_to(_samples_dir_to_root(samples_dir)).as_posix(),
            kind=sample.kind,
            frame=sample.frame,
            width=width,
            height=height,
            circles=circles,
        ),
    )
    return f"Guardados {len(circles)} círculo(s) para {sample.label}."


def _calibration_path(samples_dir: str) -> Path:
    return calibration_json(samples_dir)


def shape_choices(editor_value: dict | None) -> list[str]:
    """Nomes das camadas atuais (uma por objeto desenhado)."""
    if not editor_value:
        return []
    layers = editor_value.get("layers") or []
    return [f"Camada {i + 1}" for i in range(len(layers))]


def _layers_update(editor_value: dict | None) -> gr.update:
    """Atualização do dropdown de camadas: escolhas + seleção (última camada)."""
    choices = shape_choices(editor_value)
    if not choices:
        return gr.update(choices=[], value=None, interactive=False)
    return gr.update(choices=choices, value=choices[-1], interactive=True)


def _selected_index(editor_value: dict | None, selected: str | None) -> int | None:
    """Índice da camada selecionada no dropdown; sem seleção válida usa a última."""
    layers = (editor_value or {}).get("layers") or []
    if not layers:
        return None
    index = None
    if selected:
        match = re.match(r"Camada (\d+)$", str(selected))
        if match:
            index = int(match.group(1)) - 1
    if index is None or not 0 <= index < len(layers):
        index = len(layers) - 1
    return index


def load_item_view(
    key: str | None,
    samples_dir: str,
    config: AstroFrameConfig | None = None,
    store: CalibrationStore | None = None,
) -> tuple[dict, str, dict | None, gr.update, tuple | None, str]:
    """Versão da UI de `load_item_payload`: também devolve a pré-visualização
    clicável (centro das formas), o dropdown de camadas e o ponto clicado."""
    value, info = load_item_payload(key, samples_dir, config, store)
    preview = value.get("background") if value else None
    return (
        value,
        info,
        preview,
        _layers_update(value),
        None,
        "Clica na imagem para definir o centro da forma.",
    )


def auto_detect_view(
    key: str | None,
    samples_dir: str,
    config: AstroFrameConfig | None = None,
) -> tuple[dict, str, gr.update]:
    """Versão da UI de `auto_detect_payload`: também sincroniza o dropdown."""
    value, info = auto_detect_payload(key, samples_dir, config)
    return value, info, _layers_update(value)


def auto_detect_with_params(
    key: str | None,
    samples_dir: str,
    config: AstroFrameConfig | None = None,
    param2: float | None = None,
    max_radius: float | None = None,
) -> tuple[dict, str, AstroFrameConfig]:
    """Re-corre a deteção com os parâmetros dos sliders e devolve o config
    atualizado (usado também na validação)."""
    config = config or AstroFrameConfig()
    if param2 is not None:
        config.stabilizer.param2 = int(param2)
    if max_radius is not None:
        config.stabilizer.max_radius = int(max_radius)
    value, info = auto_detect_payload(key, samples_dir, config)
    return value, info, config


def store_click(evt: gr.SelectData | None) -> tuple[tuple[int, int] | None, str]:
    """Guarda o ponto clicado na pré-visualização (centro de novas formas)."""
    if evt is None:
        return None, "Clica na imagem para definir o centro da forma."
    index = evt.index
    if not isinstance(index, (list, tuple)) or len(index) < 2:
        return None, "Clique não registado — tenta de novo."
    x, y = int(index[0]), int(index[1])
    return (x, y), f"Centro: ({x}, {y})"


def toggle_ratio(shape: str | None) -> gr.update:
    """Mostra o slider de proporção apenas quando a forma é uma elipse."""
    return gr.update(visible=(shape or "").strip().lower() == "elipse")


def add_shape_layer(
    editor_value: dict | None,
    shape: str | None,
    diameter: float | None,
    ratio: float | None,
    center: tuple | None = None,
) -> tuple[dict, gr.update, str]:
    """Adiciona uma camada com um círculo/elipse centrado no ponto clicado."""
    if not editor_value or editor_value.get("background") is None:
        return {}, _layers_update(None), "Carrega uma amostra primeiro."
    background = editor_value["background"]
    height, width = background.shape[:2]
    cx, cy = center if center else (width // 2, height // 2)
    is_ellipse = (shape or "").strip().lower() == "elipse"
    ratio = max(0.1, float(ratio or 1.0)) if is_ellipse else 1.0
    diameter = max(4.0, float(diameter or 60.0))
    new_layer = shape_layer(shape, diameter, ratio, int(cx), int(cy), (height, width))
    layers = list(editor_value.get("layers") or [])
    layers.append(new_layer)
    value = {"background": background, "layers": layers, "composite": None}
    kind = "elipse" if is_ellipse else "círculo"
    return value, _layers_update(value), f"Adicionada {len(layers)}.ª camada ({kind})."


def remove_layer(
    editor_value: dict | None,
    selected: str | None,
) -> tuple[dict, gr.update, str]:
    """Elimina a camada selecionada (sem seleção válida, a última)."""
    index = _selected_index(editor_value, selected)
    if index is None:
        return {}, _layers_update(None), "Sem camadas para eliminar."
    layers = list(editor_value["layers"])
    layers.pop(index)
    value = {**editor_value, "layers": layers, "composite": None}
    return value, _layers_update(value), f"Camada {index + 1} eliminada."


def clear_layers(editor_value: dict | None) -> tuple[dict, gr.update, str]:
    """Remove todas as camadas (mantém o fundo)."""
    if not editor_value or editor_value.get("background") is None:
        return {}, _layers_update(None), "Carrega uma amostra primeiro."
    value = {"background": editor_value["background"], "layers": [], "composite": None}
    return value, _layers_update(value), "Camadas limpas."


def move_layer(
    editor_value: dict | None,
    selected: str | None,
    center: tuple | None,
) -> tuple[dict, gr.update, str]:
    """Move o conteúdo da camada selecionada para o centro clicado."""
    index = _selected_index(editor_value, selected)
    if index is None:
        return {}, _layers_update(None), "Sem camadas para mover."
    if not center:
        return (
            editor_value or {},
            _layers_update(editor_value),
            "Clica na imagem para escolher a nova posição.",
        )
    layers = list(editor_value["layers"])
    layers[index] = recenter_layer(layers[index], int(center[0]), int(center[1]))
    value = {**editor_value, "layers": layers, "composite": None}
    return value, _layers_update(value), f"Camada {index + 1} movida para ({center[0]}, {center[1]})."


def resize_layer(
    editor_value: dict | None,
    selected: str | None,
    diameter: float | None,
) -> tuple[dict, gr.update, str]:
    """Redimensiona o conteúdo da camada selecionada (eixo maior → diâmetro)."""
    index = _selected_index(editor_value, selected)
    if index is None:
        return {}, _layers_update(None), "Sem camadas para redimensionar."
    diameter = max(4.0, float(diameter or 60.0))
    layers = list(editor_value["layers"])
    layers[index] = scale_layer(layers[index], diameter)
    value = {**editor_value, "layers": layers, "composite": None}
    return value, _layers_update(value), f"Camada {index + 1} redimensionada para {diameter:.0f} px."


def _format(value: float | None, digits: int = 1) -> str:
    return f"{value:.{digits}f}" if value is not None else "—"


def validate_all_report(
    samples_dir: str,
    config: AstroFrameConfig | None = None,
    store: CalibrationStore | None = None,
) -> tuple[list[list], str, str]:
    """Valida todas as amostras: devolve (linhas da tabela, resumo, sugestões)."""
    config = config or AstroFrameConfig()
    samples_dir_path = _samples_dir_to_root(samples_dir)
    store = store or CalibrationStore(_calibration_path(samples_dir))
    samples = scan_samples(samples_dir_path)

    rows: list[list] = []
    graded: list[tuple[str, list, list]] = []
    for sample in samples:
        stored = store.get_item(sample.key)
        detected = find_all_disks(load_frame(sample), config)
        if stored is None or not stored.circles:
            rows.append([sample.label, 0, len(detected), "—", "sem ground truth", ""])
            continue
        report = validate_all([(sample.label, list(stored.circles), detected)]).items[0]
        rows.append(
            [
                sample.label,
                report.n_manual,
                report.n_detected,
                report.n_matched,
                report.n_false_negatives,
                report.n_false_positives,
                _format(report.mean_iou),
                _format(report.mean_center_error),
                _format(report.mean_radius_error_pct),
            ]
        )
        graded.append((sample.label, list(stored.circles), detected))

    report = validate_all(graded)
    if not report.has_ground_truth:
        summary = (
            "<b>Sem ground truth para validar.</b> Desenha os astros em cada amostra "
            "e carrega em 'Guardar ajustes' antes de validar."
        )
    else:
        score = f"{report.score:.1f}/100" if report.score is not None else "—"
        summary = (
            f"<b>Score de calibração: {score}</b> — "
            f"recall <b>{report.recall * _PCT:.1f}%</b> · precisão <b>{report.precision * _PCT:.1f}%</b> · "
            f"IoU médio <b>{report.mean_iou:.2f}</b> · "
            f"erro centro <b>{_format(report.mean_center_error)} px</b> · "
            f"erro raio <b>{_format(report.mean_radius_error_pct)}%</b><br>"
            f"{report.total_matched}/{report.total_manual} círculos manuais correspondidos · "
            f"{report.total_false_negatives} em falta · {report.total_false_positives} deteções extra "
            f"em {len(graded)} amostra(s) com ground truth."
        )
    suggestions = "<br>".join(f"• {item}" for item in suggest_parameters(report, config))
    return rows, summary, suggestions


def build_calibration_app(
    samples_dir: str = "samples",
    config: AstroFrameConfig | None = None,
    store: CalibrationStore | None = None,
) -> gr.Blocks:
    """Monta a interface Gradio de calibração (depende de `samples_dir`)."""
    config = config or AstroFrameConfig()
    store = store or CalibrationStore(_calibration_path(samples_dir))
    samples = scan_samples(_samples_dir_to_root(samples_dir))
    choices = [(sample.label, sample.key) for sample in samples]
    first = choices[0][1] if choices else None
    stab = config.stabilizer

    with gr.Blocks(title="AstroFrame — Calibração") as demo:
        gr.Markdown(
            "# 🔭 AstroFrame — Calibração\n"
            "A deteção automática corre ao carregar cada amostra. Ajusta os objetos: "
            "**camadas** (círculos/elipses) são selecionáveis e podem ser **movidas**, "
            "**redimensionadas** ou **eliminadas**; o **pincel** serve para detalhes "
            "menores (pinta na camada ativa do editor) e a **borracha** remove."
        )
        with gr.Row():
            with gr.Column(scale=3):
                item_selector = gr.Dropdown(
                    choices=choices,
                    value=first,
                    label="Amostra (imagem ou frame de vídeo)",
                )
                editor = gr.ImageEditor(
                    type="numpy",
                    label="Objetos (um por astro)",
                    brush=gr.Brush(colors=["#00ff00"], default_color="#00ff00"),
                    eraser=gr.Eraser(default_size=24),
                )
                item_info = gr.HTML(label="Informação")
                with gr.Row():
                    auto_button = gr.Button("Deteção automática", variant="secondary")
                    save_button = gr.Button("Guardar ajustes", variant="primary")
                status = gr.Textbox(label="Estado", interactive=False)

                with gr.Accordion("Formas e camadas (círculo, elipse)", open=True):
                    with gr.Row():
                        shape_type = gr.Radio(
                            ["Círculo", "Elipse"],
                            value="Círculo",
                            label="Forma",
                        )
                        shape_diameter = gr.Slider(
                            8, 1600, value=120, step=4, label="Diâmetro (px)"
                        )
                        shape_ratio = gr.Slider(
                            0.1, 1.0, value=0.7, step=0.05, label="Proporção da elipse", visible=False
                        )
                    click_info = gr.Textbox(
                        value="Clica na imagem para definir o centro da forma.",
                        label="Centro",
                        interactive=False,
                    )
                    shape_preview = gr.Image(
                        type="numpy",
                        label="Clique para o centro da forma",
                        height=220,
                    )
                    with gr.Row():
                        add_button = gr.Button("Adicionar forma", variant="primary")
                        move_button = gr.Button("Mover selecionada para o ponto")
                        resize_button = gr.Button("Redimensionar selecionada")
                    layers_dd = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Camada (objeto) a editar",
                        interactive=False,
                    )
                    with gr.Row():
                        remove_button = gr.Button("Eliminar selecionada", variant="stop")
                        clear_button = gr.Button("Limpar todas as camadas")

                with gr.Accordion("Parâmetros da deteção automática", open=False):
                    gr.Markdown(
                        "Ajusta um parâmetro e larga-o — a deteção corre automaticamente "
                        "na amostra atual com os novos valores."
                    )
                    with gr.Row():
                        p_param2 = gr.Slider(5, 200, value=stab.param2, step=1, label="Hough param2")
                        p_max_radius = gr.Slider(
                            50, 1000, value=stab.max_radius, step=10, label="Raio máximo (px)"
                        )

            with gr.Column(scale=2):
                validate_button = gr.Button("Validar todas as amostras", variant="primary")
                summary_html = gr.HTML(label="Resumo global")
                table = gr.Dataframe(
                    headers=[
                        "Amostra",
                        "Manuais",
                        "Detetados",
                        "Correspondências",
                        "Em falta",
                        "Extras",
                        "IoU",
                        "Erro centro (px)",
                        "Erro raio (%)",
                    ],
                    label="Por amostra",
                    interactive=False,
                )
                suggestions_html = gr.HTML(label="Sugestões de parâmetros")

        click_center = gr.State()

        item_selector.change(
            load_item_view,
            inputs=[item_selector, gr.State(samples_dir), gr.State(config), gr.State(store)],
            outputs=[editor, item_info, shape_preview, layers_dd, click_center, click_info],
        )
        auto_button.click(
            auto_detect_view,
            inputs=[item_selector, gr.State(samples_dir), gr.State(config)],
            outputs=[editor, item_info, layers_dd],
        )
        save_button.click(
            save_item_circles,
            inputs=[editor, item_selector, gr.State(samples_dir), gr.State(store)],
            outputs=[status],
        )
        shape_type.change(toggle_ratio, inputs=[shape_type], outputs=[shape_ratio])
        shape_preview.select(store_click, outputs=[click_center, click_info])
        add_button.click(
            add_shape_layer,
            inputs=[editor, shape_type, shape_diameter, shape_ratio, click_center],
            outputs=[editor, layers_dd, status],
        )
        move_button.click(
            move_layer,
            inputs=[editor, layers_dd, click_center],
            outputs=[editor, layers_dd, status],
        )
        resize_button.click(
            resize_layer,
            inputs=[editor, layers_dd, shape_diameter],
            outputs=[editor, layers_dd, status],
        )
        remove_button.click(
            remove_layer,
            inputs=[editor, layers_dd],
            outputs=[editor, layers_dd, status],
        )
        clear_button.click(
            clear_layers,
            inputs=[editor],
            outputs=[editor, layers_dd, status],
        )
        for slider in (p_param2, p_max_radius):
            slider.release(
                auto_detect_with_params,
                inputs=[
                    item_selector,
                    gr.State(samples_dir),
                    gr.State(config),
                    p_param2,
                    p_max_radius,
                ],
                outputs=[editor, item_info, gr.State(config)],
            )
        validate_button.click(
            validate_all_report,
            inputs=[gr.State(samples_dir), gr.State(config), gr.State(store)],
            outputs=[table, summary_html, suggestions_html],
        )

    return demo


def run(
    samples_dir: str = "samples",
    config_path: str | None = None,
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    inbrowser: bool = True,
) -> None:
    """Lança a interface de calibração no navegador."""
    config = AstroFrameConfig.from_yaml(config_path) if config_path else AstroFrameConfig()
    build_calibration_app(samples_dir=samples_dir, config=config).launch(
        server_name=host, server_port=port, share=share, inbrowser=inbrowser
    )
