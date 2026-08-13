"""Testes do varrimento da pasta de exemplos (calibração)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from astroframe.calibration.scan import (
    item_key,
    item_label,
    load_frame,
    sample_video_frames,
    scan_samples,
)
from astroframe.video.reader import FrameReader


def _write_image(path, height=60, width=80):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(frame, (width // 2, height // 2), 20, (200, 200, 200), -1)
    cv2.imwrite(str(path), frame)


def _write_video(path, n_frames=10, height=60, width=80):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (width, height))
    try:
        for i in range(n_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = i * 3
            writer.write(frame)
    finally:
        writer.release()


def test_item_key_imagem_e_video():
    assert item_key("a/b/x.jpg") == "a/b/x.jpg"
    assert item_key("v.mp4", 12) == "v.mp4#12"
    assert item_key("v.mp4", None) == "v.mp4"


def test_item_label_imagem_e_video():
    assert item_label("image", "x.jpg") == "IMG x.jpg"
    assert item_label("video", "v.mp4", 3) == "VID v.mp4 #3"


def test_sample_video_frames_equidistantes():
    assert sample_video_frames(100, 4) == [12, 37, 62, 87]


def test_sample_video_frames_menos_frames_que_n():
    assert sample_video_frames(3, 8) == [0, 1, 2]


def test_sample_video_frames_desconhecido():
    assert sample_video_frames(0, 8) == [0, 1, 2, 3, 4, 5, 6, 7]


def test_sample_video_frames_n_reduzido_a_um():
    assert sample_video_frames(10, 0) == [5]
    assert sample_video_frames(10, -3) == [5]


def test_scan_samples_varre_recursivamente(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "videos").mkdir()
    _write_image(tmp_path / "images" / "sol.jpg")
    _write_video(tmp_path / "videos" / "eclipse.mp4", n_frames=10)
    (tmp_path / "ignorado.txt").write_text("nota", encoding="utf-8")

    samples = scan_samples(tmp_path, frames_per_video=4)

    kinds = [s.kind for s in samples]
    assert kinds == ["image", "video", "video", "video", "video"]
    image = samples[0]
    assert image.kind == "image"
    assert image.frame is None
    assert image.key == "images/sol.jpg"
    assert image.label == "IMG images/sol.jpg"
    assert image.path == tmp_path / "images" / "sol.jpg"
    videos = samples[1:]
    assert [v.frame for v in videos] == sample_video_frames(10, 4) == [1, 3, 6, 8]
    assert videos[0].key == "videos/eclipse.mp4#1"
    assert videos[0].label == "VID videos/eclipse.mp4 #1"


def test_scan_samples_suporta_varias_extensoes(tmp_path):
    _write_image(tmp_path / "a.bmp")
    _write_image(tmp_path / "b.PNG")
    _write_image(tmp_path / "c.webp")
    samples = scan_samples(tmp_path)
    assert [s.key for s in samples] == ["a.bmp", "b.PNG", "c.webp"]


def test_scan_samples_video_ilegivel_ignorado(tmp_path, caplog):
    _write_image(tmp_path / "ok.jpg")
    (tmp_path / "roto.avi").write_bytes(b"nao e um video")
    samples = scan_samples(tmp_path)
    assert len(samples) == 1
    assert samples[0].key == "ok.jpg"


def test_scan_samples_pasta_vazia(tmp_path):
    assert scan_samples(tmp_path) == []


def test_load_frame_imagem(tmp_path):
    _write_image(tmp_path / "foto.png")
    (sample,) = scan_samples(tmp_path)
    frame = load_frame(sample)
    assert frame.shape[:2] == (60, 80)
    assert frame[30, 40].tolist() == [200, 200, 200]


def test_load_frame_video_le_frame_certo(tmp_path):
    _write_video(tmp_path / "clip.mov", n_frames=6)
    samples = scan_samples(tmp_path, frames_per_video=8)
    assert len(samples) == 6
    for sample in samples:
        frame = load_frame(sample)
        assert abs(int(frame[5, 5].mean()) - sample.frame * 3) <= 2


def test_load_frame_imagem_ilegivel_levanta(tmp_path):
    (tmp_path / "rota.jpg").write_bytes(b"x")
    sample = scan_samples(tmp_path)[0]
    with pytest.raises(ValueError):
        load_frame(sample)


def test_frame_at_devolve_frame_no_indice(tmp_path):
    _write_video(tmp_path / "clip.avi", n_frames=5)
    with FrameReader(tmp_path / "clip.avi") as reader:
        frame = reader.frame_at(3)
    assert abs(int(frame[5, 5].mean()) - 9) <= 2


def test_frame_at_clampa_indice_negativo(tmp_path):
    _write_video(tmp_path / "clip.avi", n_frames=5)
    with FrameReader(tmp_path / "clip.avi") as reader:
        frame = reader.frame_at(-2)
    assert abs(int(frame[5, 5].mean())) <= 2


def test_frame_at_indice_para_alem_do_fim_levanta(tmp_path):
    _write_video(tmp_path / "clip.avi", n_frames=5)
    with FrameReader(tmp_path / "clip.avi") as reader:
        with pytest.raises(ValueError):
            reader.frame_at(999)
