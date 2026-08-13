"""Interface Gradio de calibração: ajuste manual dos círculos e validação.

Fluxo: escolhe uma amostra da pasta de exemplos (`samples/` por omissão) →
os círculos aparecem pré-preenchidos (ground truth guardado; senão a deteção
automática como ponto de partida) → ajusta à mão (arrastar camada = **mover**,
pincel por cima = **adicionar**, borracha = **remover**) → **Guardar ajustes**
grava o ground truth em `calibration.json` → **Validar todas** compara a
deteção automática com o ground truth em todas as amostras e devolve o
relatório (recall, precisão, IoU, erros) + sugestões de parâmetros.

Todos os handlers são funções de módulo testáveis; o `build_calibration_app`
é apenas a montagem Gradio.
"""

from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

from astroframe.calibration.circles import circles_to_layers, layers_to_circles
from astroframe.calibration.scan import load_frame, scan_samples
from astroframe.calibration.store import CalibrationItem, CalibrationStore
from astroframe.calibration.validate import suggest_parameters, validate_all
from astroframe.config import AstroFrameConfig
from astroframe.core.stabilizer import find_all_disks
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
    return _samples_dir_to_root(samples_dir) / "calibration.json"


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
    choices = [sample.label for sample in samples]
    first = choices[0] if choices else None

    with gr.Blocks(title="AstroFrame — Calibração") as demo:
        gr.Markdown(
            "# 🔭 AstroFrame — Calibração\n"
            "Ajusta os círculos (astros) de cada amostra e valida a deteção automática. "
            "Gestos: **arrastar** uma camada move o círculo, **pintar** por cima adiciona, "
            "**borracha** remove."
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
                    label="Círculos (um por astro)",
                    brush=gr.Brush(colors=["#00ff00"], default_color="#00ff00"),
                    eraser=gr.Eraser(default_size=24),
                )
                item_info = gr.HTML(label="Informação")
                with gr.Row():
                    auto_button = gr.Button("Deteção automática", variant="secondary")
                    save_button = gr.Button("Guardar ajustes", variant="primary")
                status = gr.Textbox(label="Estado", interactive=False)
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

        item_selector.change(
            load_item_payload,
            inputs=[item_selector, gr.State(samples_dir), gr.State(config), gr.State(store)],
            outputs=[editor, item_info],
        )
        auto_button.click(
            auto_detect_payload,
            inputs=[item_selector, gr.State(samples_dir), gr.State(config)],
            outputs=[editor, item_info],
        )
        save_button.click(
            save_item_circles,
            inputs=[editor, item_selector, gr.State(samples_dir), gr.State(store)],
            outputs=[status],
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
