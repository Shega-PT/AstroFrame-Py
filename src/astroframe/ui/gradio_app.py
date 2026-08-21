"""Interface Gradio: processamento de imagens e vídeos com avaliação por estrelas.

Dois separadores:

- **Imagem** — Antes/Depois com sliders, zoom na coroa/borda e disco detetado
  (verde) + reflexos (vermelho); avaliação automática (0–5 estrelas) com
  feedback manual guardado na base local de aprendizagem.
- **Vídeo** — ao carregar um vídeo, os metadados (ffprobe/OpenCV/EXIF) são
  lidos e as sugestões de parâmetros (incluindo o que a IA já aprendeu) são
  aplicadas aos sliders. Ao processar, o painel esquerdo mostra o vídeo em
  tempo real com os discos detetados (verde principal, vermelho reflexos) em
  todos os frames, e o direito atualiza em frames espaçados o resultado final
  (estabilizado + CLAHE + denoise + nitidez + polimento a preto).

A aprendizagem (`ai.feedback`) regista **uma linha por utilização** na base
SQLite `~/.astroframe/feedback.db`: o que foi usado, como correu (métricas +
estrelas), o que se ajustou para a próxima vez e porquê.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from astroframe.ai import params as ai_params
from astroframe.ai.feedback import (
    FeedbackDB,
    apply_learned,
    origin_for,
    profile_for,
    record_run,
)
from astroframe.ai.score import score_image, stars_text
from astroframe.ai.tuner import run_autotune
from astroframe.config import AstroFrameConfig
from astroframe.core.enhancer import enhance_image
from astroframe.core.pipeline import process_image
from astroframe.core.polish import polish_image
from astroframe.core.stabilizer import (
    AntiJitterStabilizer,
    DiskDetection,
    find_all_disks,
    find_disks_for_calibration,
)
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


def _radius_clamped(frame: np.ndarray, radius: int) -> int:
    height, width = frame.shape[:2]
    return max(1, min(radius, min(height, width) // 2))


def _draw_detection(frame: np.ndarray, detection: DiskDetection | None) -> np.ndarray:
    """Cópia do frame com o círculo (bounding box) do disco detetado."""
    if detection is None:
        return frame.copy()
    return _draw_disks(frame, [detection])


def _draw_disks(frame: np.ndarray, disks: list[DiskDetection]) -> np.ndarray:
    """Cópia do frame com TODOS os discos detetados desenhados a verde.

    Na astrofotografia cada disco é um corpo celeste real — não há distinção
    principal/secundário/reflexo na vista ao vivo (os reflexos da lente são
    tratados no polimento, não na deteção).
    """
    height, width = frame.shape[:2]
    marked = frame.copy()
    for disk in disks:
        if not (0 <= disk.cx < width and 0 <= disk.cy < height):
            continue
        cv2.circle(marked, (disk.cx, disk.cy), _radius_clamped(frame, disk.radius), (0, 255, 0), 2)
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


def _learning_db(cfg: AstroFrameConfig, db: FeedbackDB | None) -> FeedbackDB | None:
    """Devolve o banco de aprendizagem, respeitando `feedback.enabled`."""
    if db is not None:
        return db
    if not cfg.feedback.enabled:
        return None
    return FeedbackDB()


def _learning_log_html(profile: str | None, db: FeedbackDB | None) -> str:
    """Log de aprendizagem do perfil (o que mudou, como e porquê)."""
    if not profile:
        return "Sem histórico — processa primeiro uma imagem ou vídeo."
    if db is None:
        return "Aprendizagem desativada (feedback.enabled = false no config)."
    runs = db.history(profile, limit=12)
    if not runs:
        return "Sem registos para este perfil ainda — a primeira utilização cria o log."
    rows = []
    for run in runs:
        stars = stars_text(run.stars_calc)
        manual = f" · utilizador: {stars_text(run.stars_user)}" if run.stars_user is not None else ""
        changes = ", ".join(f"{key} {value:+.3g}" for key, value in run.nudge.items()) or "sem alterações"
        rows.append(
            f"<tr><td>{run.ts}</td><td>{run.kind}</td><td>{stars}{manual}</td>"
            f"<td>{changes}</td><td>{run.rationale}</td></tr>"
        )
    body = "".join(rows)
    return (
        "<table>"
        "<tr><th>Data</th><th>Tipo</th><th>Avaliação</th><th>Ajustes</th><th>Porquê</th></tr>"
        f"{body}</table>"
    )


def _run_state(kind: str, profile: str, cfg: AstroFrameConfig, rating: object, source: str = "") -> dict:
    """Estado do último processamento, guardado para a avaliação manual."""
    return {"kind": kind, "profile": profile, "cfg": cfg, "rating": rating, "source": source}


def inspect_video_upload(
    video_path: str | None, config: AstroFrameConfig | None = None, db: FeedbackDB | None = None
) -> tuple[str, dict, object, object, object, object]:
    """Extrai metadados do vídeo carregado e devolve as sugestões para os sliders.

    As sugestões incluem o que a IA já aprendeu para o perfil do vídeo
    (metadados → `suggest_config`, depois deltas do banco de feedback).
    Devolve (html_resumo, metadados_raw, clip, denoise, unsharp, coroa).
    """
    config = config or AstroFrameConfig()
    if not video_path:
        empty = gr.update()
        return "Carrega um vídeo para ver os metadados.", {}, empty, empty, empty, empty
    meta = extract_metadata(video_path)
    suggested = suggest_config(meta)
    learned = apply_learned(
        suggested,
        profile_for("video", meta.width or 0, meta.height or 0),
        db=_learning_db(config, db),
    )
    if learned is suggested:
        clip, denoise, unsharp, corona = (
            suggested.clahe.clip_limit,
            suggested.denoise.h,
            suggested.unsharp.amount,
            suggested.polish.corona_scale,
        )
    else:
        clip, denoise, unsharp, corona = (
            learned.clahe.clip_limit,
            learned.denoise.h,
            learned.unsharp.amount,
            learned.polish.corona_scale,
        )
    return (
        _summary_html(meta),
        meta.raw,
        gr.update(value=clip),
        gr.update(value=denoise),
        gr.update(value=unsharp),
        gr.update(value=corona),
    )


def process_video(
    video_path: str | None,
    export_video: bool,
    clip_limit: float,
    denoise_h: float,
    sharp_amount: float,
    show_disk: bool,
    corona_scale: float,
    config: AstroFrameConfig | None = None,
    db: FeedbackDB | None = None,
):
    """Processa um vídeo frame a frame, emitindo o estado ao vivo e o preview.

    É um gerador: a cada frame processado entrega
    (live_rgb, preview_rgb, out_video, status, progresso, avaliação, estado,
    log); a última entrega é o estado final. O preview (direita) só é
    atualizado em frames espaçados e já vem polido (fundo preto).
    """
    config = config or AstroFrameConfig()
    if not video_path:
        yield None, None, None, "Carrega um vídeo primeiro.", 0.0, "", None, ""
        return
    db = _learning_db(config, db)

    reader = FrameReader(video_path)
    width, height = reader.size
    profile = profile_for("video", width, height)
    cfg = apply_learned(config, profile, db=db)
    cfg = replace(
        cfg,
        clahe=replace(cfg.clahe, clip_limit=clip_limit),
        denoise=replace(cfg.denoise, h=denoise_h),
        unsharp=replace(cfg.unsharp, amount=sharp_amount),
        polish=replace(cfg.polish, corona_scale=corona_scale),
    )
    total = reader.frame_count or 0
    every = _preview_every(total)
    writer = None
    out_path = None
    last_live = None
    last_preview = None
    last_rating = None
    done = 0
    try:
        if export_video:
            out_path = f"{video_path}.processed.mp4"
            writer = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*"mp4v"), reader.fps or 30.0, (width, height)
            )
            if not writer.isOpened():
                raise OSError(f"Não foi possível criar o vídeo de saída: {out_path}")
        engine = AntiJitterStabilizer(config=cfg)
        with reader:
            for frame in reader:
                stabilized, detection = engine.stabilize(frame)
                disks = engine.last_all_disks
                live = _draw_disks(frame, disks) if show_disk else frame.copy()
                state = "sem disco detetado" if detection is None else "disco no centro"
                status = f"Frame {done + 1}/{total or '?'} · {state}"
                if writer is not None and stabilized.shape[:2] == (height, width):
                    engine_radius = engine.last_detection.radius if engine.last_detection else 0
                    if engine_radius > 0:
                        writer.write(
                            polish_image(
                                stabilized,
                                DiskDetection(width // 2, height // 2, engine_radius),
                                cfg,
                            )
                        )
                    else:
                        writer.write(stabilized)
                if total and done % every == 0:
                    raw = enhance_image(stabilized, cfg)
                    engine_radius = engine.last_detection.radius if engine.last_detection else 0
                    if engine_radius > 0:
                        last_preview = polish_image(
                            raw,
                            DiskDetection(width // 2, height // 2, engine_radius),
                            cfg,
                        )
                        last_rating = score_image(
                            raw,
                            DiskDetection(width // 2, height // 2, engine_radius),
                            cfg,
                        )
                    else:
                        last_preview = raw
                        last_rating = score_image(raw, None, cfg)
                last_live = live
                done += 1
                yield (
                    _from_pipeline(live),
                    _from_pipeline(last_preview),
                    None,
                    status,
                    done / total if total else 0.0,
                    stars_text(last_rating.stars) if last_rating is not None else "",
                    None,
                    "",
                )
        final = f"Concluído — {done} frames processados."
        if writer is not None:
            final += f" Exportado: {out_path}"
        rating_html = stars_text(last_rating.stars) if last_rating is not None else "Avaliação — sem frames."
        state = _run_state("video", profile, cfg, last_rating, source=str(video_path))
        log_html = ""
        if last_rating is not None and db is not None:
            record_run(db, "video", profile, cfg, origin_for(cfg), last_rating, source=str(video_path))
            log_html = _learning_log_html(profile, db)
        yield (
            _from_pipeline(last_live) if last_live is not None else None,
            _from_pipeline(last_preview) if last_preview is not None else None,
            out_path,
            final,
            1.0,
            rating_html,
            state,
            log_html,
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
    corona_scale: float,
    config: AstroFrameConfig | None = None,
    progress=None,
    db: FeedbackDB | None = None,
    expected_disks: int | None = None,
):
    """Processa uma imagem (separador Imagem) e avalia o resultado.

    Devolve (estabilizado, processado, zoom, avaliação_html, estado, log).
    Também regista a utilização no banco de aprendizagem (uma linha nova).

    `expected_disks` (opcional) ativa a **deteção multidisco obrigatória**:
    o número de discos calibrado manualmente na mesma imagem — a deteção
    escala a sensibilidade até os encontrar.
    """
    if progress is None:
        progress = gr.Progress()
    if input_image is None:
        return (None, None, None, "Avaliação — processa primeiro uma imagem.", None, "")

    config = config or AstroFrameConfig()
    db = _learning_db(config, db)
    bgr = _to_pipeline(input_image)
    height, width = bgr.shape[:2]
    profile = profile_for("image", width, height)
    cfg = apply_learned(config, profile, db=db)
    cfg = replace(
        cfg,
        clahe=replace(cfg.clahe, clip_limit=clip_limit),
        denoise=replace(cfg.denoise, h=denoise_h),
        unsharp=replace(cfg.unsharp, amount=sharp_amount),
        polish=replace(cfg.polish, corona_scale=corona_scale),
    )
    result = process_image(bgr, cfg)

    if result.detection is not None:
        rating = score_image(
            result.enhanced_raw,
            DiskDetection(width // 2, height // 2, result.detection.radius),
            cfg,
        )
    else:
        rating = score_image(result.enhanced_raw, None, cfg)
    rating_html = stars_text(rating.stars)

    stabilized = result.stabilized.copy()
    if show_disk and result.detection is not None:
        primary = DiskDetection(width // 2, height // 2, result.detection.radius)
        dx, dy = width // 2 - result.detection.cx, height // 2 - result.detection.cy
        all_disks = find_disks_for_calibration(bgr, cfg, expected_n=expected_disks)
        translated = [
            DiskDetection(d.cx + dx, d.cy + dy, d.radius)
            for d in all_disks
            if not (d.cx == result.detection.cx and d.cy == result.detection.cy)
        ]
        stabilized = _draw_disks(stabilized, [primary, *translated])

    zoomed = _zoom_crop(result.enhanced, zoom)
    state = _run_state("image", profile, cfg, rating, source=f"{width}x{height}")
    log_html = ""
    if db is not None:
        record_run(
            db,
            "image",
            profile,
            cfg,
            origin_for(cfg),
            rating,
            source=f"{width}x{height}",
        )
        log_html = _learning_log_html(profile, db)
    return (
        _from_pipeline(stabilized),
        _from_pipeline(result.enhanced),
        _from_pipeline(zoomed),
        rating_html,
        state,
        log_html,
    )


def manual_feedback(state: dict | None, stars_user: float, db: FeedbackDB | None = None) -> tuple[str, str]:
    """Guarda a avaliação manual (peso reforçado) e devolve o resultado + log."""
    db = db or FeedbackDB()
    if not state:
        return "Processa primeiro (imagem ou vídeo) para poderes avaliar.", _learning_log_html(None, db)
    rating = state.get("rating")
    if rating is None:
        return "Sem avaliação para guardar — processa novamente.", _learning_log_html(
            state.get("profile"), db
        )
    cfg = state.get("cfg") or AstroFrameConfig()
    run = record_run(
        db,
        state.get("kind", "image"),
        state.get("profile", "unknown"),
        cfg,
        state.get("origin") or {},
        rating,
        stars_user=stars_user,
        source=state.get("source", ""),
    )
    return f"Guardado: {run.rationale}", _learning_log_html(state.get("profile"), db)


def run_autotune_tab(
    samples_dir: str,
    budget: float,
    params: str,
    anneal: bool,
    register: bool,
    config: AstroFrameConfig | None = None,
):
    """Auto-tuning a partir do separador: otimiza contra `samples/` e devolve
    o relatório + a configuração otimizada (editável no YAML/JSON exportado)."""
    config = config or AstroFrameConfig()
    if not samples_dir.strip():
        yield "Indica a pasta de amostras (ex.: samples/).", None
        return
    yield "A otimizar… (avaliação em ~480p, cache por parâmetros efetivos)", None
    db = _learning_db(config, None) if register else None
    result = run_autotune(
        samples_dir=samples_dir.strip(),
        config=config,
        budget_s=float(budget),
        anneal=anneal,
        params_filter=params.strip() or None,
        db=db,
    )
    yield "\n".join(result.lines), result.config.to_dict()


def build_app(config: AstroFrameConfig | None = None) -> gr.Blocks:
    config = config or AstroFrameConfig()

    with gr.Blocks(title="AstroFrame — Astro Auto-Enhancer") as demo:
        gr.Markdown("# 🌒 AstroFrame — Astro Auto-Enhancer")
        gr.Markdown(
            "Estabilização geométrica e melhoria automática de fotos e vídeos de astros "
            "(Sol, Lua, planetas, cometas). "
            "Cada utilização é avaliada (0–5 estrelas) e registada no banco local de "
            "aprendizagem — o sistema ajusta-se automaticamente a cada execução."
        )

        with gr.Tabs():
            with gr.Tab("Imagem"):
                with gr.Row():
                    input_image = gr.Image(label="Entrada — imagem (foto/frame original)")
                    stabilized = gr.Image(label="Estabilizado (disco centrado)")
                    processed = gr.Image(label="Processado (estabilizado + CLAHE + denoise + polido)")

                with gr.Row():
                    zoomed = gr.Image(label="Zoom na coroa/borda")

                with gr.Accordion("Parâmetros", open=False):
                    with gr.Accordion("Melhoria (CLAHE + denoising + nitidez)", open=True):
                        gr.Markdown(
                            "<small>⚙️ Ajustes do pipeline de melhoria automática. "
                            "Os pesos são aprendidos pelo sistema a cada utilização.</small>"
                        )
                        clip_limit = gr.Slider(
                            0.5, 6.0, value=config.clahe.clip_limit, step=0.1,
                            label="CLAHE clip limit — contraste adaptativo",
                        )
                        denoise_h = gr.Slider(
                            1.0, 20.0, value=config.denoise.h, step=1.0,
                            label="Denoising (h) — força do filtragem bilateral",
                        )
                        sharp_amount = gr.Slider(
                            0.0, 2.0, value=config.unsharp.amount, step=0.1,
                            label="Nitidez (unsharp mask) — realce de bordas",
                        )
                    with gr.Accordion("Polimento (fundo + coroa)", open=False):
                        gr.Markdown(
                            "<small>⚙️ Controlo do polimento final: remoção de fundo, "
                            "preservação da coroa e gestão de reflexos da lente.</small>"
                        )
                        corona_scale = gr.Slider(
                            1.0, 3.0, value=config.polish.corona_scale, step=0.1,
                            label="Coroa mantida (× raio) — zona preservada além do limbo",
                        )
                    with gr.Accordion("Deteção + visualização", open=False):
                        gr.Markdown(
                            "<small>⚙️ Parâmetros de deteção geométrica e visualização.</small>"
                        )
                        zoom = gr.Slider(1.0, 4.0, value=1.0, step=0.5, label="Zoom na coroa/borda")
                        show_disk = gr.Checkbox(True, label="Mostrar disco detetado")

                    with gr.Row():
                        rating_label = gr.HTML(label="Avaliação automática")
                        stars_manual = gr.Slider(
                            0.0, 5.0, value=3.0, step=0.5, label="Avaliação manual (estrelas)"
                        )
                        feedback_btn = gr.Button("Guardar avaliação manual", variant="secondary")
                    with gr.Accordion("Log de aprendizagem (o que a IA ajustou e porquê)", open=False):
                        image_log_html = gr.HTML()
                        feedback_msg = gr.Textbox(label="Feedback do ajuste", interactive=False)

                image_run_state = gr.State()
                button = gr.Button("Processar", variant="primary")
                button.click(
                    process_image_input,
                    inputs=[input_image, clip_limit, denoise_h, sharp_amount, zoom, show_disk, corona_scale],
                    outputs=[stabilized, processed, zoomed, rating_label, image_run_state, image_log_html],
                )
                feedback_btn.click(
                    manual_feedback,
                    inputs=[image_run_state, stars_manual],
                    outputs=[feedback_msg, image_log_html],
                )

            with gr.Tab("Vídeo"):
                gr.Markdown(
                    "Carregue um vídeo: os **metadados** (ffprobe/OpenCV/EXIF) são lidos e os "
                    "**sliders são pré-preenchidos com sugestões** (héurísticas + o que a IA já "
                    "aprendeu; continuam editáveis). Ao processar, a **esquerda mostra o vídeo ao "
                    "vivo** com os discos detetados (verde = principal, vermelho = reflexos) e a "
                    "**direita atualiza em frames espaçados** o resultado final com todas as correções."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="Entrada — vídeo (.mp4/.avi/.mov)", sources=["upload"])
                        summary_html = gr.HTML(label="Proporção / qualidade / sugestões")
                        with gr.Accordion("Metadados completos", open=False):
                            meta_json = gr.JSON()
                    with gr.Column(scale=2):
                        with gr.Row():
                            live = gr.Image(label="Ao vivo — frame original + discos detetados")
                            preview = gr.Image(label="Resultado final (frames espaçados)")
                        with gr.Accordion("Processamento", open=False):
                            gr.Markdown(
                                "<small>⚙️ Mesmos parâmetros da aba Imagem — valores "
                                "sugestivos dos metadados do vídeo + aprendizagem anterior.</small>"
                            )
                            with gr.Accordion("Melhoria", open=True):
                                v_clip_limit = gr.Slider(
                                    0.5, 6.0, value=config.clahe.clip_limit, step=0.1,
                                    label="CLAHE clip limit",
                                )
                                v_denoise_h = gr.Slider(
                                    1.0, 20.0, value=config.denoise.h, step=1.0,
                                    label="Denoising (h)",
                                )
                                v_sharp_amount = gr.Slider(
                                    0.0, 2.0, value=config.unsharp.amount, step=0.1,
                                    label="Nitidez (unsharp)",
                                )
                            with gr.Accordion("Polimento", open=False):
                                v_corona_scale = gr.Slider(
                                    1.0, 3.0, value=config.polish.corona_scale, step=0.1,
                                    label="Coroa mantida (× raio)",
                                )
                            with gr.Accordion("Deteção + visualização", open=False):
                                v_show_disk = gr.Checkbox(True, label="Mostrar discos detetados ao vivo")
                                v_export = gr.Checkbox(False, label="Exportar vídeo processado (.mp4, sem áudio)")
                        with gr.Row():
                            v_rating_label = gr.HTML(label="Avaliação automática")
                            v_stars_manual = gr.Slider(
                                0.0, 5.0, value=3.0, step=0.5, label="Avaliação manual (estrelas)"
                            )
                            v_feedback_btn = gr.Button("Guardar avaliação manual", variant="secondary")
                        status = gr.Textbox(label="Estado", interactive=False)
                        progress_slider = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.0, step=0.01, label="Progresso", visible=False
                        )
                        export_video = gr.Video(label="Vídeo processado (exportação)")
                        with gr.Accordion("Log de aprendizagem (o que a IA ajustou e porquê)", open=False):
                            v_log_html = gr.HTML()
                            v_feedback_msg = gr.Textbox(label="Feedback do ajuste", interactive=False)
                        video_button = gr.Button("Processar vídeo", variant="primary")
                        video_run_state = gr.State()

                video_input.upload(
                    inspect_video_upload,
                    inputs=[video_input],
                    outputs=[
                        summary_html,
                        meta_json,
                        v_clip_limit,
                        v_denoise_h,
                        v_sharp_amount,
                        v_corona_scale,
                    ],
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
                        v_corona_scale,
                    ],
                    outputs=[
                        live,
                        preview,
                        export_video,
                        status,
                        progress_slider,
                        v_rating_label,
                        video_run_state,
                        v_log_html,
                    ],
                )
                v_feedback_btn.click(
                    manual_feedback,
                    inputs=[video_run_state, v_stars_manual],
                    outputs=[v_feedback_msg, v_log_html],
                )

            with gr.Tab("Auto-tune"):
                gr.Markdown(
                    "Otimiza **todos os parâmetros** da pipeline contra as amostras de `samples/`: "
                    "deteção (vs `calibration.json`) + melhoria (estrelas). A otimização é "
                    "determinística (seed fixa), limitada às gamas seguras e registada no banco "
                    "de aprendizagem (aplicada nas próximas execuções do mesmo perfil)."
                )
                with gr.Row():
                    at_samples = gr.Textbox(value="samples", label="Pasta de amostras")
                    at_budget = gr.Slider(5.0, 300.0, value=60.0, step=5.0, label="Orçamento (segundos)")
                    at_params = gr.Dropdown(
                        choices=list(ai_params.PARAM_SPECS),
                        multiselect=True,
                        allow_custom_value=True,
                        label="Parâmetros (vazio = todos)",
                    )
                    at_anneal = gr.Checkbox(True, label="Recozimento (escapar de mínimos locais)")
                    at_register = gr.Checkbox(True, label="Registar no banco de aprendizagem")
                with gr.Row():
                    at_button = gr.Button("Otimizar", variant="primary")
                    at_reset = gr.Button("Apagar histórico de auto-tuning", variant="secondary")
                at_report = gr.Textbox(label="Relatório", interactive=False, lines=10)
                at_config = gr.JSON(label="Configuração otimizada")
                at_button.click(
                    run_autotune_tab,
                    inputs=[at_samples, at_budget, at_params, at_anneal, at_register],
                    outputs=[at_report, at_config],
                )
                at_reset.click(
                    lambda: f"Histórico de auto-tuning apagado ({FeedbackDB().reset_tuning()} registo(s)).",
                    outputs=[at_report],
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
