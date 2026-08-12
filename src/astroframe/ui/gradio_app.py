"""Interface mínima Gradio: Antes vs Depois, sliders e zoom na coroa/borda."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from astroframe.config import AstroFrameConfig
from astroframe.core.pipeline import process_image


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


def build_app(config: AstroFrameConfig | None = None) -> gr.Blocks:
    config = config or AstroFrameConfig()

    def process(input_image, clip_limit, denoise_h, sharp_amount, zoom, show_disk):
        if input_image is None:
            return None, None, None

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

    with gr.Blocks(title="AstroFrame — Eclipse Auto-Enhancer") as demo:
        gr.Markdown("# 🌒 AstroFrame — Eclipse Auto-Enhancer")
        gr.Markdown("Estabilização geométrica e melhoria automática de fotos e frames de eclipses.")

        with gr.Row():
            input_image = gr.Image(label="Entrada (foto/frame original)")
            stabilized = gr.Image(label="Estabilizado (disco centrado)")
            processed = gr.Image(label="Processado (CLAHE + denoise + nitidez)")

        with gr.Row():
            zoomed = gr.Image(label="Zoom na coroa/borda")

        with gr.Accordion("Parâmetros", open=False):
            clip_limit = gr.Slider(
                0.5, 6.0, value=config.clahe.clip_limit, step=0.1, label="CLAHE clip limit"
            )
            denoise_h = gr.Slider(1.0, 20.0, value=config.denoise.h, step=1.0, label="Força do denoising")
            sharp_amount = gr.Slider(
                0.0, 2.0, value=config.unsharp.amount, step=0.1, label="Nitidez (unsharp)"
            )
            zoom = gr.Slider(1.0, 4.0, value=1.0, step=0.5, label="Zoom na coroa/borda")
            show_disk = gr.Checkbox(True, label="Mostrar disco detetado")

        button = gr.Button("Processar", variant="primary")
        button.click(
            process,
            inputs=[input_image, clip_limit, denoise_h, sharp_amount, zoom, show_disk],
            outputs=[stabilized, processed, zoomed],
        )

    return demo


def run(
    config_path: str | None = None, host: str = "127.0.0.1", port: int = 7860, share: bool = False
) -> None:
    """Lança a interface Gradio no navegador."""
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config não encontrado: {config_path}")
        config = AstroFrameConfig.from_yaml(path)
    else:
        config = AstroFrameConfig()
    build_app(config).launch(server_name=host, server_port=port, share=share)
