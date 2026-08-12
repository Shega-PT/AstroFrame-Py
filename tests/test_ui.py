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
    _from_pipeline,
    _preview_every,
    _summary_html,
    _to_pipeline,
    _zoom_crop,
    inspect_video_upload,
    process_image_input,
    process_video,
    run,
)


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
# _draw_detection / _preview_every
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
    html, raw, clip, denoise, unsharp = inspect_video_upload(None)
    assert "Carrega um vídeo" in html
    assert raw == {}


def test_inspect_video_upload_com_video(tmp_path, monkeypatch):
    video = tmp_path / "clip.avi"
    _write_video(video, [_make_disk_frame() for _ in range(3)])
    monkeypatch.setattr("astroframe.meta.extractor._ffprobe", lambda path: None)
    html, raw, clip, denoise, unsharp = inspect_video_upload(str(video))
    assert "<table>" in html
    assert raw == {}
    assert clip["value"] == 3.0  # valores por omissão sem EXIF/bitrate
    assert denoise["value"] == 5.0
    assert unsharp["value"] == 0.5


# ---------------------------------------------------------------------------
# process_video (gerador)
# ---------------------------------------------------------------------------


def test_process_video_sem_ficheiro():
    yields = list(process_video(None, False, 3.0, 5.0, 0.5, True))
    assert yields == [(None, None, None, "Carrega um vídeo primeiro.", 0.0)]


def test_process_video_exporta_mp4(tmp_path):
    video = tmp_path / "clip.avi"
    frames = [_make_disk_frame() for _ in range(5)]
    _write_video(video, frames)

    yields = list(process_video(str(video), True, 3.0, 5.0, 0.5, False))
    per_frame, final = yields[:-1], yields[-1]

    assert len(per_frame) == 5
    for live, preview, out_video, status, progress in per_frame:
        assert live.shape[:2] == (64, 80) and live.shape[2] == 3
        assert live.dtype == np.uint8
        assert preview is not None and preview.shape[:2] == (64, 80)
        assert out_video is None
        assert status.startswith("Frame")
        assert 0.0 <= progress <= 1.0

    last_live, last_preview, out_video, status, progress = final
    assert progress == 1.0
    assert "Concluído" in status
    assert "Exportado" in status
    assert out_video is not None and Path(out_video).exists()
    assert last_live is not None and last_preview is not None

    capture = cv2.VideoCapture(out_video)
    assert capture.isOpened()
    ok, frame = capture.read()
    capture.release()
    assert ok and frame.shape[:2] == (64, 80)


def test_process_video_sem_exportacao_nao_gera_ficheiro(tmp_path):
    video = tmp_path / "clip.avi"
    _write_video(video, [_make_disk_frame() for _ in range(3)])
    yields = list(process_video(str(video), False, 3.0, 5.0, 0.5, False))
    assert yields[-1][2] is None
    assert "Exportado" not in yields[-1][3]


def test_process_video_show_disk_mantem_frame_original(tmp_path):
    from astroframe.video.reader import FrameReader

    video = tmp_path / "clip.avi"
    frames = [_make_disk_frame() for _ in range(3)]
    _write_video(video, frames)
    yields = list(process_video(str(video), False, 3.0, 5.0, 0.5, False))
    first_live = yields[0][0]
    expected = cv2.cvtColor(next(iter(FrameReader(str(video)))), cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(first_live, expected)


def test_process_video_ficheiro_inexistente_levanta():
    with pytest.raises(ValueError):
        list(process_video("/nao/existe.avi", False, 3.0, 5.0, 0.5, False))


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
        list(process_video(str(video), True, 3.0, 5.0, 0.5, False))


# ---------------------------------------------------------------------------
# process_image_input (separador Imagem)
# ---------------------------------------------------------------------------


def test_process_image_input_sem_imagem():
    assert process_image_input(None, 3.0, 5.0, 0.5, 1.0, True) == (None, None, None)


def test_process_image_input_produz_rgb():
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    cv2.circle(image, (80, 50), 30, (180,) * 3, -1)
    stabilized, processed, zoomed = process_image_input(image, 3.0, 5.0, 0.5, 1.0, True)
    for output in (stabilized, processed, zoomed):
        assert output.ndim == 3 and output.shape[2] == 3
        assert output.shape[:2] == (100, 160)


def test_process_image_input_show_disk_desenha_circulo():
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    cv2.circle(image, (80, 50), 30, (180,) * 3, -1)
    stabilized, _, _ = process_image_input(image, 3.0, 5.0, 0.5, 1.0, True)
    assert (stabilized[..., 1] > 200).any()
    stabilized_no_disk, _, _ = process_image_input(image, 3.0, 5.0, 0.5, 1.0, False)
    assert not (stabilized_no_disk[..., 1] > 200).any()


def test_process_image_input_escala_de_cinza():
    gray = np.full((90, 120), 60, dtype=np.uint8)
    cv2.circle(gray, (60, 45), 25, (200,), -1)
    stabilized, processed, zoomed = process_image_input(gray, 3.0, 5.0, 0.5, 1.0, False)
    assert stabilized.shape[:2] == (90, 120)


def test_process_image_input_zoom():
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    cv2.circle(image, (80, 50), 30, (180,) * 3, -1)
    _, _, zoomed = process_image_input(image, 3.0, 5.0, 0.5, 2.0, False)
    assert zoomed.shape[:2] == (50, 80)


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
