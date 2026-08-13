"""Testes da interface Gradio: cores, ao-vivo de vídeo, metadados e sugestões."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from astroframe.config import AstroFrameConfig
from astroframe.core.stabilizer import DiskDetection
from astroframe.meta.extractor import MediaMetadata
from astroframe.ui.gradio_app import (
    _draw_detection,
    _draw_disks,
    _from_pipeline,
    _preview_every,
    _split_disks,
    _summary_html,
    _to_pipeline,
    _zoom_crop,
    inspect_video_upload,
    manual_feedback,
    process_image_input,
    process_video,
    run,
)


@pytest.fixture(autouse=True)
def _feedback_db_tmp(tmp_path, monkeypatch):
    """Isola o banco de aprendizagem por teste (nunca toca em ~/.astroframe)."""
    monkeypatch.setenv("ASTROFRAME_FEEDBACK_DB", str(tmp_path / "feedback-test.db"))


def test_to_pipeline_converte_rgb_para_bgr():
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb[0, 0] = (255, 0, 0)
    bgr = _to_pipeline(rgb)
    np.testing.assert_array_equal(bgr[0, 0], (0, 0, 255))


def test_from_pipeline_converte_bgr_para_rgb():
    bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    bgr[0, 0] = (0, 0, 255)
    rgb = _from_pipeline(bgr)
    np.testing.assert_array_equal(rgb[0, 0], (255, 0, 0))


def test_round_trip_cores():
    image = np.arange(10 * 10 * 3, dtype=np.uint8).reshape(10, 10, 3)
    np.testing.assert_array_equal(_from_pipeline(_to_pipeline(image)), image)


def test_passthrough_escala_de_cinza():
    gray = np.zeros((5, 5), dtype=np.uint8)
    np.testing.assert_array_equal(_to_pipeline(gray), gray)


def test_from_pipeline_escala_de_cinza():
    gray = np.zeros((5, 5), dtype=np.uint8)
    np.testing.assert_array_equal(_from_pipeline(gray), gray)


def test_zoom_crop():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    assert _zoom_crop(image, 1.0).shape[:2] == (100, 200)
    assert _zoom_crop(image, 0.5).shape[:2] == (100, 200)
    assert _zoom_crop(image, 2.0).shape[:2] == (50, 100)


def _write_video(path, frames: list[np.ndarray], fps: float = 10.0) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def _make_disk_frame() -> np.ndarray:
    frame = np.full((64, 80, 3), 15, dtype=np.uint8)
    cv2.circle(frame, (40, 30), 14, (220,) * 3, -1)
    return frame


# ---------------------------------------------------------------------------
# _draw_detection / _draw_disks / _preview_every
# ---------------------------------------------------------------------------


def test_draw_detection_sem_deteccao_devolve_copia():
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    np.testing.assert_array_equal(_draw_detection(frame, None), frame)


def test_draw_detection_desenha_circulo_verde_no_centro():
    frame = np.zeros((50, 60, 3), dtype=np.uint8)
    marked = _draw_detection(frame, DiskDetection(30, 25, 8))
    assert marked.shape == frame.shape
    green = marked[..., 1] > 200
    assert green.any()
    np.testing.assert_array_equal(marked[~green], frame[~green])


def test_draw_detection_fora_dos_limites_devolve_copia():
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    np.testing.assert_array_equal(_draw_detection(frame, DiskDetection(1000, 25, 8)), frame)


def test_draw_disks_desenha_reflexos_a_vermelho():
    frame = np.zeros((50, 60, 3), dtype=np.uint8)
    marked = _draw_disks(frame, DiskDetection(30, 25, 8), [DiskDetection(10, 10, 4)])
    assert (marked[..., 2] > 200).any()
    assert (marked[..., 1] > 200).any()


def test_draw_disks_ignora_discos_fora_dos_limites():
    frame = np.zeros((50, 60, 3), dtype=np.uint8)
    marked = _draw_disks(frame, DiskDetection(30, 25, 8), [DiskDetection(-5, 10, 4)])
    assert not (marked[..., 2] > 200).any()


def test_split_disks_separa_companheiro_de_reflexo():
    primary = DiskDetection(30, 25, 10)
    companion = DiskDetection(32, 27, 5)
    ghost = DiskDetection(50, 45, 6)
    companions, reflections = _split_disks([primary, companion, ghost], primary)
    assert companions == [companion]
    assert reflections == [ghost]
    companions, reflections = _split_disks([companion, ghost], None)
    assert companions == []
    assert len(reflections) == 2


def test_draw_disks_desenha_companheiro_a_amarelo():
    frame = np.zeros((50, 60, 3), dtype=np.uint8)
    marked = _draw_disks(
        frame,
        DiskDetection(30, 25, 10),
        reflections=[DiskDetection(50, 45, 6)],
        companions=[DiskDetection(32, 27, 5)],
    )
    assert (marked[..., 2] > 200).any()
    assert (marked[..., 1] > 200).any()
    red_only = (marked[..., 2] > 200) & (marked[..., 1] <= 200)
    yellow = (marked[..., 1] > 200) & (marked[..., 2] > 200)
    assert red_only.any() and yellow.any()


def test_draw_disks_ignora_companheiro_fora_dos_limites():
    frame = np.zeros((50, 60, 3), dtype=np.uint8)
    marked = _draw_disks(frame, DiskDetection(30, 25, 10), companions=[DiskDetection(-5, 27, 5)])
    yellow = (marked[..., 1] > 200) & (marked[..., 2] > 200)
    assert not yellow.any()


def test_preview_every():
    assert _preview_every(None) == 1
    assert _preview_every(0) == 1
    assert _preview_every(5) == 1
    assert _preview_every(80) == 10
    assert _preview_every(100) == 12


# ---------------------------------------------------------------------------
# metadados + sugestões na interface
# ---------------------------------------------------------------------------


def test_summary_html_com_metadados():
    meta = MediaMetadata(width=1920, height=1080, kind="video")
    html = _summary_html(meta)
    assert "<table>" in html
    assert "1920x1080" in html
    assert "16:9" in html


def test_summary_html_sem_metadados():
    assert "Sem metadados legíveis" in _summary_html(MediaMetadata())


def test_inspect_video_upload_sem_video():
    html, raw, clip, denoise, unsharp, corona = inspect_video_upload(None, db=None)
    assert "Carrega um vídeo" in html
    assert raw == {}


def test_inspect_video_upload_com_video(tmp_path, monkeypatch):
    video = tmp_path / "clip.avi"
    _write_video(video, [_make_disk_frame() for _ in range(3)])
    monkeypatch.setattr("astroframe.meta.extractor._ffprobe", lambda path: None)
    monkeypatch.setenv("ASTROFRAME_FEEDBACK_DB", str(tmp_path / "fb.db"))
    html, raw, clip, denoise, unsharp, corona = inspect_video_upload(str(video))
    assert "<table>" in html
    assert raw == {}
    assert clip["value"] == 3.0  # valores por omissão sem EXIF/bitrate
    assert denoise["value"] == 5.0
    assert unsharp["value"] == 0.5
    assert corona["value"] == 1.6


# ---------------------------------------------------------------------------
# process_video (gerador)
# ---------------------------------------------------------------------------


def test_process_video_sem_ficheiro():
    yields = list(process_video(None, False, 3.0, 5.0, 0.5, True, 1.6, db=None))
    assert yields == [(None, None, None, "Carrega um vídeo primeiro.", 0.0, "", None, "")]


def test_process_video_exporta_mp4(tmp_path):
    video = tmp_path / "clip.avi"
    frames = [_make_disk_frame() for _ in range(5)]
    _write_video(video, frames)

    yields = list(process_video(str(video), True, 3.0, 5.0, 0.5, True, 1.6, db=None))
    per_frame, final = yields[:-1], yields[-1]

    assert len(per_frame) == 5
    for live, preview, out_video, status, progress, rating, state, log in per_frame:
        assert live.shape[:2] == (64, 80) and live.shape[2] == 3
        assert live.dtype == np.uint8
        assert preview is not None and preview.shape[:2] == (64, 80)
        assert out_video is None
        assert status.startswith("Frame")
        assert 0.0 <= progress <= 1.0
        assert "★" in rating
        assert state is None
        assert log == ""

    last_live, last_preview, out_video, status, progress, rating, state, log = final
    assert progress == 1.0
    assert "Concluído" in status
    assert "Exportado" in status
    assert out_video is not None and Path(out_video).exists()
    assert last_live is not None and last_preview is not None
    assert state is not None and state["kind"] == "video"

    capture = cv2.VideoCapture(out_video)
    assert capture.isOpened()
    ok, frame = capture.read()
    capture.release()
    assert ok and frame.shape[:2] == (64, 80)


def test_process_video_sem_exportacao_nao_gera_ficheiro(tmp_path):
    video = tmp_path / "clip.avi"
    _write_video(video, [_make_disk_frame() for _ in range(3)])
    yields = list(process_video(str(video), False, 3.0, 5.0, 0.5, False, 1.6, db=None))
    assert yields[-1][2] is None
    assert "Exportado" not in yields[-1][3]


def test_process_video_mostra_circulo_em_todos_os_frames(tmp_path):
    video = tmp_path / "clip.avi"
    _write_video(video, [_make_disk_frame() for _ in range(4)])
    yields = list(process_video(str(video), False, 3.0, 5.0, 0.5, True, 1.6, db=None))
    for live, *_ in yields[:-1]:
        assert (live[..., 1] > 250).any(), "o bounding verde deve aparecer em todos os frames"


def test_process_video_show_disk_off_sem_circulo(tmp_path):
    video = tmp_path / "clip.avi"
    _write_video(video, [_make_disk_frame() for _ in range(2)])
    yields = list(process_video(str(video), False, 3.0, 5.0, 0.5, False, 1.6, db=None))
    for live, *_ in yields[:-1]:
        assert not (live[..., 1] > 250).any()


def test_process_video_regista_utilizacao_no_banco(tmp_path):
    video = tmp_path / "clip.avi"
    _write_video(video, [_make_disk_frame() for _ in range(3)])
    from astroframe.ai.feedback import FeedbackDB

    db = FeedbackDB(tmp_path / "fb.db")
    yields = list(process_video(str(video), False, 3.0, 5.0, 0.5, True, 1.6, config=None, db=db))
    assert db.count() == 1
    _, _, _, _, _, rating, state, log = yields[-1]
    assert state["kind"] == "video"
    assert db.history(state["profile"], limit=1)[0].stars_calc == pytest.approx(
        state["rating"].stars, abs=0.01
    )


def test_process_video_sem_disco_usa_caminho_sem_polimento(tmp_path):
    video = tmp_path / "black.avi"
    _write_video(video, [np.zeros((64, 80, 3), dtype=np.uint8) for _ in range(3)])
    yields = list(process_video(str(video), True, 3.0, 5.0, 0.5, True, 1.6, db=None))
    for live, preview, _out_video, status, *_ in yields[:-1]:
        assert not (live[..., 1] > 250).any()
        assert "sem disco detetado" in status
        assert preview is not None
    final = yields[-1]
    assert final[2] is not None and Path(final[2]).exists()
    assert final[5]  # avaliação calculada sem deteção
    assert final[6] is not None and final[6]["kind"] == "video"


def test_process_video_ficheiro_inexistente_levanta():
    with pytest.raises(ValueError):
        list(process_video("/nao/existe.avi", False, 3.0, 5.0, 0.5, False, 1.6, db=None))


def test_process_video_escritor_falha_levanta_oserror(tmp_path, monkeypatch):
    class FakeWriter:
        def __init__(self, *args, **kwargs):
            pass

        def isOpened(self):
            return False

        def release(self):
            pass

    video = tmp_path / "clip.avi"
    _write_video(video, [_make_disk_frame() for _ in range(2)])
    monkeypatch.setattr("astroframe.ui.gradio_app.cv2.VideoWriter", FakeWriter)
    with pytest.raises(OSError, match="Não foi possível criar"):
        list(process_video(str(video), True, 3.0, 5.0, 0.5, False, 1.6, db=None))


# ---------------------------------------------------------------------------
# process_image_input (separador Imagem)
# ---------------------------------------------------------------------------


def test_process_image_input_sem_imagem():
    stabilized, processed, zoomed, rating_html, state, log = process_image_input(
        None, 3.0, 5.0, 0.5, 1.0, True, 1.6, db=None
    )
    assert stabilized is None and processed is None and zoomed is None
    assert "processa primeiro" in rating_html
    assert state is None


def test_process_image_input_produz_rgb(tmp_path):
    from astroframe.ai.feedback import FeedbackDB

    db = FeedbackDB(tmp_path / "fb.db")
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    cv2.circle(image, (80, 50), 30, (180,) * 3, -1)
    stabilized, processed, zoomed, rating_html, state, _ = process_image_input(
        image, 3.0, 5.0, 0.5, 1.0, True, 1.6, db=db
    )
    for output in (stabilized, processed, zoomed):
        assert output.ndim == 3 and output.shape[2] == 3
        assert output.shape[:2] == (100, 160)
    assert "★" in rating_html
    assert state is not None and state["kind"] == "image"
    assert db.count() == 1


def test_process_image_input_show_disk_desenha_circulo(tmp_path):
    from astroframe.ai.feedback import FeedbackDB

    db = FeedbackDB(tmp_path / "fb.db")
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    cv2.circle(image, (80, 50), 30, (180,) * 3, -1)
    stabilized, _, _, _, _, _ = process_image_input(image, 3.0, 5.0, 0.5, 1.0, True, 1.6, db=db)
    assert (stabilized[..., 1] > 250).any()
    stabilized_no_disk, _, _, _, _, _ = process_image_input(image, 3.0, 5.0, 0.5, 1.0, False, 1.6, db=db)
    assert not (stabilized_no_disk[..., 1] > 250).any()


def test_process_image_input_sem_disco_avalia_sem_deteccao(tmp_path):
    from astroframe.ai.feedback import FeedbackDB

    db = FeedbackDB(tmp_path / "fb.db")
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    stabilized, _, _, rating_html, state, _ = process_image_input(image, 3.0, 5.0, 0.5, 1.0, True, 1.6, db=db)
    assert "★" in rating_html or "☆" in rating_html
    assert not (stabilized[..., 1] > 250).any()
    assert state is not None


def test_process_image_input_escala_de_cinza(tmp_path):
    from astroframe.ai.feedback import FeedbackDB

    db = FeedbackDB(tmp_path / "fb.db")
    gray = np.full((90, 120), 60, dtype=np.uint8)
    cv2.circle(gray, (60, 45), 25, (200,), -1)
    stabilized, processed, zoomed, *_ = process_image_input(gray, 3.0, 5.0, 0.5, 1.0, False, 1.6, db=db)
    assert stabilized.shape[:2] == (90, 120)


def test_process_image_input_zoom(tmp_path):
    from astroframe.ai.feedback import FeedbackDB

    db = FeedbackDB(tmp_path / "fb.db")
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    cv2.circle(image, (80, 50), 30, (180,) * 3, -1)
    _, _, zoomed, *_ = process_image_input(image, 3.0, 5.0, 0.5, 2.0, False, 1.6, db=db)
    assert zoomed.shape[:2] == (50, 80)


def test_process_image_input_com_aprendizagem_aplicada(tmp_path):
    from astroframe.ai.feedback import FeedbackDB, profile_for, record_run
    from astroframe.ai.score import score_from_stars

    db = FeedbackDB(tmp_path / "fb.db")
    profile = profile_for("image", 160, 100)
    rating = score_from_stars(1.0)
    rating.metrics.update(background=0.1)
    record_run(db, "image", profile, AstroFrameConfig(), {}, rating)
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    cv2.circle(image, (80, 50), 30, (180,) * 3, -1)
    _, _, _, _, state, _ = process_image_input(image, 3.0, 5.0, 0.5, 1.0, False, 1.6, db=db)
    assert state["cfg"].polish.feather > 0.02


def test_process_image_input_feedback_desativado_nao_cria_banco(tmp_path):

    cfg = AstroFrameConfig()
    cfg.feedback.enabled = False
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    cv2.circle(image, (80, 50), 30, (180,) * 3, -1)
    *_, state, log = process_image_input(image, 3.0, 5.0, 0.5, 1.0, False, 1.6, config=cfg, db=None)
    assert log == ""
    assert state is not None


def test_learning_log_html_aprendizagem_desativada():
    from astroframe.ui.gradio_app import _learning_log_html

    assert "desativada" in _learning_log_html("perfil", None)


def test_learning_log_html_sem_perfil():
    from astroframe.ui.gradio_app import _learning_log_html

    assert "Sem histórico" in _learning_log_html(None, None)


def test_inspect_video_upload_aplica_aprendizagem(tmp_path, monkeypatch):
    from astroframe.ai.feedback import FeedbackDB, profile_for, record_run
    from astroframe.ai.score import score_from_stars

    video = tmp_path / "clip.avi"
    _write_video(video, [_make_disk_frame() for _ in range(3)])
    monkeypatch.setattr("astroframe.meta.extractor._ffprobe", lambda path: None)
    db_path = tmp_path / "fb.db"
    monkeypatch.setenv("ASTROFRAME_FEEDBACK_DB", str(db_path))
    db0 = FeedbackDB(db_path)
    profile = profile_for("video", 80, 64)
    rating = score_from_stars(1.0)
    rating.metrics.update(noise=0.1)
    record_run(db0, "video", profile, AstroFrameConfig(), {}, rating)
    html, raw, clip, denoise, unsharp, corona = inspect_video_upload(str(video))
    assert denoise["value"] == pytest.approx(5.3, abs=0.01)


# ---------------------------------------------------------------------------
# manual_feedback (avaliação manual do utilizador)
# ---------------------------------------------------------------------------


def test_manual_feedback_sem_estado(tmp_path):
    from astroframe.ai.feedback import FeedbackDB

    db = FeedbackDB(tmp_path / "fb.db")
    msg, log = manual_feedback(None, 4.0, db=db)
    assert "Processa primeiro" in msg
    assert "Sem histórico" in log


def test_manual_feedback_estado_invalido(tmp_path):
    from astroframe.ai.feedback import FeedbackDB

    db = FeedbackDB(tmp_path / "fb.db")
    msg, _ = manual_feedback({"profile": "x"}, 4.0, db=db)
    assert "Sem avaliação" in msg


def test_manual_feedback_guarda_com_peso_reforcado(tmp_path):
    from astroframe.ai.feedback import FeedbackDB

    db = FeedbackDB(tmp_path / "fb.db")
    state = {
        "kind": "image",
        "profile": "prof-a",
        "cfg": AstroFrameConfig(),
        "rating": None,
        "source": "teste",
    }
    from astroframe.ai.score import score_from_stars

    rating = score_from_stars(1.0)
    rating.metrics.update(noise=0.1)
    # já temos cobertura do estado "Sem avaliação"; aqui o estado válido:
    state["rating"] = rating
    msg, log = manual_feedback(state, 1.0, db=db)
    assert "Guardado" in msg
    assert "<table>" in log
    row = db.history("prof-a", limit=1)[0]
    assert row.stars_user == 1.0


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


def test_run_abre_servidor_com_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    AstroFrameConfig().to_yaml(config_path)
    launched = {}
    monkeypatch.setattr(
        "astroframe.ui.gradio_app.gr.Blocks.launch",
        lambda self, **kwargs: launched.update(kwargs),
    )
    run(config_path=str(config_path), host="127.0.0.1", port=7898, inbrowser=False)
    assert launched["server_port"] == 7898
    assert launched["inbrowser"] is False


def test_run_config_inexistente_levanta():
    with pytest.raises(FileNotFoundError):
        run(config_path="/nao/existe.yaml")


def test_run_sem_config_usa_padroes(monkeypatch):
    launched = {}
    monkeypatch.setattr(
        "astroframe.ui.gradio_app.gr.Blocks.launch",
        lambda self, **kwargs: launched.update(kwargs),
    )
    run(host="127.0.0.1", port=7897, inbrowser=False)
    assert launched["server_port"] == 7897


def test_build_app_com_video_tab():
    from astroframe.ui.gradio_app import build_app

    app = build_app()
    config_text = str(app.get_config_file())
    assert "Vídeo" in config_text
    assert "Imagem" in config_text
