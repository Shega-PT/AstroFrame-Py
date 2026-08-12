"""Interface Gradio: processamento de imagens e vídeos com visualização ao vivo.

Dois separadores:

- **Imagem** — Antes/Depois com sliders, zoom na coroa/borda e disco detetado.
- **Vídeo** — ao carregar um vídeo, os metadados (ffprobe/OpenCV/EXIF) são
  lidos e as sugestões de parâmetros são aplicadas automaticamente aos sliders.
  Ao processar, o painel esquerdo mostra o vídeo em tempo real com o círculo
  (bounding box) do disco detetado, e o direito atualiza em frames espaçados o
  resultado final (estabilizado + CLAHE + denoise + nitidez). Opcionalmente,
  exporta o vídeo processado (.mp4, sem áudio — limitação do OpenCV).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from astroframe.config import AstroFrameConfig
from astroframe.core.enhancer import enhance_image
from astroframe.core.pipeline import process_image
from astroframe.core.stabilizer import AntiJitterStabilizer, DiskDetection
from astroframe.meta.extractor import MediaMetadata, extract_metadata
from astroframe.meta.suggest import suggest_config, summary_fields
from astroframe.video.reader import FrameReader

logger = logging.getLogger(__name__)

_PREVIEW_SAMPLES = 8


def _to_pipeline(image: np.ndarray) -> np.ndarray:
    """Gradio entrega arrays RGB; a pipeline usa a convenção BGR (OpenCV)."""
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def _from_pipeline(image: np.ndarray) -> np.ndarray:
    """Converte BGR (OpenCV) para RGB (Gradio) para apresentação."""
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _zoom_crop(image: np.ndarray, zoom: float) -> np.ndarray:
    height, width = image.shape[:2]
    zoom = max(1.0, float(zoom))
    crop_h, crop_w = int(height / zoom), int(width / zoom)
    x0 = (width - crop_w) // 2
    y0 = (height - crop_h) // 2
    return image[y0 : y0 + crop_h, x0 : x0 + crop_w]


def _draw_detection(frame: np.ndarray, detection: DiskDetection | None) -> np.ndarray:
    """Cópia do frame com o círculo (bounding box) do disco detetado."""
    if detection is None:
        return frame.copy()
    height, width = frame.shape[:2]
    if not (0 <= detection.cx < width and 0 <= detection.cy < height):
        return frame.copy()
    marked = frame.copy()
    radius = max(1, min(detection.radius, min(height, width) // 2))
    cv2.circle(marked, (detection.cx, detection.cy), radius, (0, 255, 0), 2)
    return marked


def _preview_every(frame_count: int | None) -> int:
    """Espaçamento entre previews: ~`_PREVIEW_SAMPLES` frames ao longo do vídeo."""
    if not frame_count or frame_count <= _PREVIEW_SAMPLES:
        return 1
    return max(1, frame_count // _PREVIEW_SAMPLES)


def _summary_html(meta: MediaMetadata) -> str:
    """Painel HTML legível com proporção/qualidade e sugestões aplicadas."""
    fields = summary_fields(meta)
    if not fields:
        return "Sem metadados legíveis para este ficheiro."
    rows = "".join(f"<tr><td><b>{key}</b></td><td>{value}</td></tr>" for key, value in fields.items())
    return "<table>" + rows + "</table>"


def inspect_video_upload(
    video_path: str | None, config: AstroFrameConfig | None = None
) -> tuple[str, dict, object, object, object]:
    """Extrai metadados do vídeo carregado e devolve as sugestões para os sliders.

    Devolve (html_resumo, metadados_raw, update_clip, update_denoise, update_unsharp).
    """
    config = config or AstroFrameConfig()
    if not video_path:
        empty = gr.update()
        return "Carrega um vídeo para ver os metadados.", {}, empty, empty, empty
    meta = extract_metadata(video_path)
    suggested = suggest_config(meta)
    return (
        _summary_html(meta),
        meta.raw,
        gr.update(value=suggested.clahe.clip_limit),
        gr.update(value=suggested.denoise.h),
        gr.update(value=suggested.unsharp.amount),
    )


def process_video(
    video_path: str | None,
    export_video: bool,
    clip_limit: float,
    denoise_h: float,
    sharp_amount: float,
    show_disk: bool,
    config: AstroFrameConfig | None = None,
):
    """Processa um vídeo frame a frame, emitindo o estado ao vivo e o preview.

    É um gerador: a cada frame processado entrega
    (live_rgb, preview_rgb, out_video, status, progresso); a última entrega é
    o estado final. O preview (direita) só é atualizado em frames espaçados.
    """
    config = config or AstroFrameConfig()
    if not video_path:
        yield None, None, None, "Carrega um vídeo primeiro.", 0.0
        return

    cfg = replace(
        config,
        clahe=replace(config.clahe, clip_limit=clip_limit),
        denoise=replace(config.denoise, h=denoise_h),
        unsharp=replace(config.unsharp, amount=sharp_amount),
    )

    reader = FrameReader(video_path)
    engine = AntiJitterStabilizer(config=cfg)
    width, height = reader.size
    total = reader.frame_count or 0
    every = _preview_every(total)
    writer = None
    out_path = None
    last_live = None
    last_preview = None
    done = 0
    try:
        if export_video:
            out_path = f"{video_path}.processed.mp4"
            writer = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*"mp4v"), reader.fps or 30.0, (width, height)
            )
            if not writer.isOpened():
                raise OSError(f"Não foi possível criar o vídeo de saída: {out_path}")
        with reader:
            for frame in reader:
                stabilized, detection = engine.stabilize(frame)
                live = _draw_detection(frame, detection) if show_disk else frame.copy()
                state = "sem disco detetado" if detection is None else "disco no centro"
                status = f"Frame {done + 1}/{total or '?'} · {state}"
                if writer is not None and stabilized.shape[:2] == (height, width):
                    writer.write(stabilized)
                if total and done % every == 0:
                    last_preview = enhance_image(stabilized, cfg)
                last_live = live
                done += 1
                yield (
                    _from_pipeline(live),
                    _from_pipeline(last_preview),
                    None,
                    status,
                    done / total if total else 0.0,
                )
        final = f"Concluído — {done} frames processados."
        if writer is not None:
            final += f" Exportado: {out_path}"
        yield (
            _from_pipeline(last_live) if last_live is not None else None,
            _from_pipeline(last_preview) if last_preview is not None else None,
            out_path,
            final,
            1.0,
        )
    finally:
        if writer is not None:
            writer.release()


def process_image_input(
    input_image,
    clip_limit: float,
    denoise_h: float,
    sharp_amount: float,
    zoom: float,
    show_disk: bool,
    config: AstroFrameConfig | None = None,
    progress=None,
):
    """Processa uma imagem do separador Imagem; devolve RGB pronto para o Gradio."""
    if progress is None:
        progress = gr.Progress()
    if input_image is None:
        return None, None, None

    config = config or AstroFrameConfig()
    cfg = replace(
        config,
        clahe=replace(config.clahe, clip_limit=clip_limit),
        denoise=replace(config.denoise, h=denoise_h),
        unsharp=replace(config.unsharp, amount=sharp_amount),
    )
    result = process_image(_to_pipeline(input_image), cfg)

    stabilized = result.stabilized.copy()
    if show_disk and result.detection is not None:
        height, width = stabilized.shape[:2]
        cv2.circle(stabilized, (width // 2, height // 2), result.detection.radius, (0, 255, 0), 2)

    zoomed = _zoom_crop(result.enhanced, zoom)
    return (
        _from_pipeline(stabilized),
        _from_pipeline(result.enhanced),
        _from_pipeline(zoomed),
    )


def build_app(config: AstroFrameConfig | None = None) -> gr.Blocks:
    config = config or AstroFrameConfig()

    with gr.Blocks(title="AstroFrame — Eclipse Auto-Enhancer") as demo:
        gr.Markdown("# 🌒 AstroFrame — Eclipse Auto-Enhancer")
        gr.Markdown("Estabilização geométrica e melhoria automática de fotos e frames de eclipses.")

        with gr.Tabs():
            with gr.Tab("Imagem"):
                with gr.Row():
                    input_image = gr.Image(label="Entrada — imagem (foto/frame original)")
                    stabilized = gr.Image(label="Estabilizado (disco centrado)")
                    processed = gr.Image(label="Processado (CLAHE + denoise + nitidez)")

                with gr.Row():
                    zoomed = gr.Image(label="Zoom na coroa/borda")

                with gr.Accordion("Parâmetros", open=False):
                    clip_limit = gr.Slider(
                        0.5, 6.0, value=config.clahe.clip_limit, step=0.1, label="CLAHE clip limit"
                    )
                    denoise_h = gr.Slider(
                        1.0, 20.0, value=config.denoise.h, step=1.0, label="Força do denoising"
                    )
                    sharp_amount = gr.Slider(
                        0.0, 2.0, value=config.unsharp.amount, step=0.1, label="Nitidez (unsharp)"
                    )
                    zoom = gr.Slider(1.0, 4.0, value=1.0, step=0.5, label="Zoom na coroa/borda")
                    show_disk = gr.Checkbox(True, label="Mostrar disco detetado")

                button = gr.Button("Processar", variant="primary")
                button.click(
                    process_image_input,
                    inputs=[input_image, clip_limit, denoise_h, sharp_amount, zoom, show_disk],
                    outputs=[stabilized, processed, zoomed],
                )

            with gr.Tab("Vídeo"):
                gr.Markdown(
                    "Carregue um vídeo: os **metadados** (ffprobe/OpenCV/EXIF) são lidos e os "
                    "**sliders são pré-preenchidos com sugestões** de otimização (continuam editáveis). "
                    "Ao processar, a **esquerda mostra o vídeo ao vivo** com o círculo do disco detetado "
                    "e a **direita atualiza em frames espaçados** o resultado final com todas as correções."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="Entrada — vídeo (.mp4/.avi/.mov)", sources=["upload"])
                        summary_html = gr.HTML(label="Proporção / qualidade / sugestões")
                        with gr.Accordion("Metadados completos", open=False):
                            meta_json = gr.JSON()
                    with gr.Column(scale=2):
                        with gr.Row():
                            live = gr.Image(label="Ao vivo — frame original + disco detetado")
                            preview = gr.Image(label="Resultado final (frames espaçados)")
                        with gr.Accordion("Processamento", open=False):
                            v_clip_limit = gr.Slider(
                                0.5, 6.0, value=config.clahe.clip_limit, step=0.1, label="CLAHE clip limit"
                            )
                            v_denoise_h = gr.Slider(
                                1.0, 20.0, value=config.denoise.h, step=1.0, label="Força do denoising"
                            )
                            v_sharp_amount = gr.Slider(
                                0.0, 2.0, value=config.unsharp.amount, step=0.1, label="Nitidez (unsharp)"
                            )
                            v_show_disk = gr.Checkbox(True, label="Mostrar disco detetado ao vivo")
                            v_export = gr.Checkbox(False, label="Exportar vídeo processado (.mp4, sem áudio)")
                        status = gr.Textbox(label="Estado", interactive=False)
                        progress_slider = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.0, step=0.01, label="Progresso", visible=False
                        )
                        export_video = gr.Video(label="Vídeo processado (exportação)")
                        video_button = gr.Button("Processar vídeo", variant="primary")

                video_input.upload(
                    inspect_video_upload,
                    inputs=[video_input],
                    outputs=[summary_html, meta_json, v_clip_limit, v_denoise_h, v_sharp_amount],
                )
                video_button.click(
                    process_video,
                    inputs=[
                        video_input,
                        v_export,
                        v_clip_limit,
                        v_denoise_h,
                        v_sharp_amount,
                        v_show_disk,
                    ],
                    outputs=[live, preview, export_video, status, progress_slider],
                )

    return demo


def run(
    config_path: str | None = None,
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    inbrowser: bool = True,
) -> None:
    """Lança a interface Gradio no navegador."""
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config não encontrado: {config_path}")
        config = AstroFrameConfig.from_yaml(path)
    else:
        config = AstroFrameConfig()
    build_app(config).launch(server_name=host, server_port=port, share=share, inbrowser=inbrowser)
