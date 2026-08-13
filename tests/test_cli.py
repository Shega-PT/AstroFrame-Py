"""Testes da linha de comando (lote resiliente, vídeo e dispatcher)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from astroframe import __version__
from astroframe.config import AstroFrameConfig
from astroframe.ui.cli import build_parser, main, process_images, process_video
from tests.helpers import make_disk_image


def _write_clip(path: Path, n_frames: int = 5, fps: float = 10.0) -> None:
    height, width = 80, 100
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    try:
        for _ in range(n_frames):
            frame = np.full((height, width, 3), 15, dtype=np.uint8)
            cv2.circle(frame, (width // 2, height // 2), 22, (200,) * 3, -1)
            writer.write(frame)
    finally:
        writer.release()


def test_process_images_continua_com_falhas(tmp_path):
    good = tmp_path / "boa.jpg"
    cv2.imwrite(str(good), make_disk_image()[0])
    bad = tmp_path / "corrompida.jpg"
    bad.write_bytes(b"isto nao e uma imagem")

    out_dir = tmp_path / "out"
    successes, failures = process_images([str(bad), str(good)], str(out_dir), AstroFrameConfig())
    assert (successes, failures) == (1, 1)
    assert (out_dir / "boa_processed.png").exists()


def test_process_images_sem_sucessos_levanta(tmp_path):
    bad = tmp_path / "corrompida.jpg"
    bad.write_bytes(b"lixo")
    with pytest.raises(RuntimeError, match="Nenhum ficheiro"):
        process_images([str(bad)], str(tmp_path / "out"), AstroFrameConfig())


# ---------------------------------------------------------------------------
# process_video
# ---------------------------------------------------------------------------


def test_process_video_modo_enhance(tmp_path):
    video = tmp_path / "clip.avi"
    _write_clip(video)
    out = process_video(str(video), None, AstroFrameConfig(), "enhance", None, False)
    assert Path(out).name == "clip.stabilized.mp4"
    capture = cv2.VideoCapture(out)
    assert capture.isOpened()
    ok, frame = capture.read()
    capture.release()
    assert ok and frame.shape[:2] == (80, 100)


def test_process_video_modo_enhance_fast(tmp_path):
    video = tmp_path / "clip.avi"
    _write_clip(video)
    out = process_video(str(video), str(tmp_path / "fast.mp4"), AstroFrameConfig(), "enhance", None, True)
    assert Path(out).exists()


def test_process_video_modo_stabilize(tmp_path):
    video = tmp_path / "clip.avi"
    _write_clip(video)
    out = process_video(str(video), str(tmp_path / "stab.mp4"), AstroFrameConfig(), "stabilize", None, False)
    assert Path(out).exists()


def test_process_video_modo_stack_fast(tmp_path):
    video = tmp_path / "clip.avi"
    _write_clip(video, n_frames=4)
    out = process_video(str(video), None, AstroFrameConfig(), "stack", 2, True)
    assert out.endswith(".png")
    image = cv2.imread(out)
    assert image is not None and image.shape[:2] == (80, 100)


def test_process_video_ficheiro_invalido_levanta():
    with pytest.raises(ValueError):
        process_video("/nao/existe.avi", None, AstroFrameConfig(), "enhance", None, False)


def test_process_video_escritor_falha_levanta(tmp_path, monkeypatch):
    class FakeWriter:
        def __init__(self, *args, **kwargs):
            pass

        def isOpened(self):
            return False

        def release(self):
            pass

    video = tmp_path / "clip.avi"
    _write_clip(video)
    monkeypatch.setattr("astroframe.ui.cli.cv2.VideoWriter", FakeWriter)
    with pytest.raises(OSError, match="Não foi possível abrir o escritor"):
        process_video(str(video), str(tmp_path / "x.mp4"), AstroFrameConfig(), "enhance", None, False)


# ---------------------------------------------------------------------------
# build_parser / main
# ---------------------------------------------------------------------------


def test_build_parser_tem_subcomandos():
    parser = build_parser()
    names = set()
    for action in parser._actions:
        if isinstance(action.choices, dict):
            names |= set(action.choices)
    assert {"serve", "process", "video", "config-template", "calibrate"} <= names


def test_main_config_template(tmp_path):
    out = tmp_path / "config.yaml"
    assert main(["config-template", "--output", str(out)]) == 0
    assert out.exists()


def test_main_process(tmp_path):
    good = tmp_path / "boa.jpg"
    cv2.imwrite(str(good), make_disk_image()[0])
    assert main(["process", "--input", str(good), "--output-dir", str(tmp_path / "out")]) == 0


def test_main_video_stack_com_stack_n(tmp_path, caplog):
    video = tmp_path / "clip.avi"
    _write_clip(video, n_frames=4)
    assert main(["video", "--input", str(video), "--mode", "stack", "--stack-n", "2"]) == 0
    assert (tmp_path / "clip.png").exists()


def test_main_stack_n_fora_de_stack_avisa(tmp_path, caplog):
    video = tmp_path / "clip.avi"
    _write_clip(video, n_frames=3)
    assert main(["video", "--input", str(video), "--mode", "enhance", "--stack-n", "2"]) == 0


def test_main_serve_lanca_gradio(monkeypatch):
    launched = {}
    monkeypatch.setattr(
        "astroframe.ui.gradio_app.run",
        lambda **kwargs: launched.update(kwargs),
    )
    assert main(["serve", "--port", "7895"]) == 0
    assert launched["port"] == 7895


def test_main_calibrate_lanca_calibration_app(monkeypatch):
    launched = {}
    monkeypatch.setattr(
        "astroframe.ui.calibration_app.run",
        lambda **kwargs: launched.update(kwargs),
    )
    assert main(["calibrate", "--samples", "samples", "--port", "7896"]) == 0
    assert launched["samples_dir"] == "samples"
    assert launched["port"] == 7896


def test_main_config_de_ficheiro(tmp_path):
    config_path = tmp_path / "config.yaml"
    AstroFrameConfig().to_yaml(config_path)
    good = tmp_path / "boa.jpg"
    cv2.imwrite(str(good), make_disk_image()[0])
    assert (
        main(
            [
                "process",
                "--input",
                str(good),
                "--output-dir",
                str(tmp_path / "out"),
                "--config",
                str(config_path),
            ]
        )
        == 0
    )


def test_main_erro_devolve_codigo_1(monkeypatch, tmp_path):
    def falhar(*args, **kwargs):
        raise RuntimeError("falha simulada")

    monkeypatch.setattr("astroframe.ui.cli.process_images", falhar)
    assert main(["process", "--input", "x.jpg", "--output-dir", str(tmp_path / "out")]) == 1


def test_main_sem_argumentos_levanta_systemexit():
    with pytest.raises(SystemExit):
        main([])


def test___main___via_python_m_imprime_versao():
    result = subprocess.run(
        [sys.executable, "-m", "astroframe", "--version"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert __version__ in result.stdout


def test___main___clicli_executa_como_script(monkeypatch):
    import runpy

    monkeypatch.setattr(sys, "argv", ["astroframe", "--version"])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("astroframe.__main__", run_name="__main__")
    assert excinfo.value.code == 0


def test_cli_guard_principal_executa_como_script(monkeypatch):
    import runpy
    import sys as _sys

    monkeypatch.setattr(_sys, "argv", ["astroframe", "--version"])
    _sys.modules.pop("astroframe.ui.cli", None)
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("astroframe.ui.cli", run_name="__main__")
    assert excinfo.value.code == 0
