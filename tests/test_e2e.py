"""Testes fim-a-fim (E2E): fluxos completos pelos pontos de entrada reais.

Correm inteiramente headless (sem janelas, sem interação):
- CLI real em subprocesso (`python -m astroframe ...`) sobre ficheiros reais;
- pipeline completa imagem/vídeo em processo;
- pontos de entrada headless do `validator.py` (`--check`).

Todos os artefactos (Logs/, banco, modelos) são redirecionados para a pasta
temporária via `ASTROFRAME_DATA_DIR`/`ASTROFRAME_FEEDBACK_DB`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from astroframe.config import AstroFrameConfig
from astroframe.core.pipeline import ProcessResult, process_image, process_path
from tests.helpers import make_disk_image

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# utilidades
# ---------------------------------------------------------------------------


def _cli_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["ASTROFRAME_DATA_DIR"] = str(tmp_path / "data")
    env["ASTROFRAME_FEEDBACK_DB"] = str(tmp_path / "feedback.db")
    env["HOME"] = str(tmp_path / "home")
    return env


def _run_cli(args: list[str], tmp_path: Path, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "astroframe", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_cli_env(tmp_path),
    )


def _write_clip(path: Path, n_frames: int = 5, fps: float = 10.0, shift: int = 2) -> None:
    """Vídeo sintético: disco claro que se desloca entre frames (estabilizável)."""
    height, width = 80, 100
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    try:
        for i in range(n_frames):
            frame = np.full((height, width, 3), 15, dtype=np.uint8)
            cv2.circle(frame, (width // 2 + i * shift, height // 2), 22, (200,) * 3, -1)
            writer.write(frame)
    finally:
        writer.release()


def _disk_center(image: np.ndarray) -> tuple[int, int]:
    """Centro do disco mais brilhante (massa do limiar alto)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ys, xs = np.nonzero(gray > 128)
    if not len(xs):
        return (-1, -1)
    return int(xs.mean()), int(ys.mean())


# ---------------------------------------------------------------------------
# CLI real (subprocesso) — fluxos completos
# ---------------------------------------------------------------------------


def test_e2e_cli_process_imagem(tmp_path):
    """`astroframe process`: imagem real na pipeline e PNG exportado válido."""
    image, cx, cy = make_disk_image()
    src = tmp_path / "sol.jpg"
    cv2.imwrite(str(src), image)

    result = _run_cli(["process", "--input", str(src), "--output-dir", str(tmp_path / "out")], tmp_path)
    assert result.returncode == 0, result.stderr

    out = tmp_path / "out" / "sol_processed.png"
    assert out.exists()
    output = cv2.imread(str(out))
    assert output is not None
    assert output.shape[:2] == image.shape[:2]
    assert not np.array_equal(output, image)  # a pipeline mexeu na imagem


def test_e2e_cli_video_enhance(tmp_path):
    """`astroframe video`: vídeo estabilizado/melhorado exportado em MP4."""
    clip = tmp_path / "clip.avi"
    _write_clip(clip)

    result = _run_cli(
        ["video", "--input", str(clip), "--output", str(tmp_path / "out.mp4"), "--mode", "enhance", "--fast"],
        tmp_path,
    )
    assert result.returncode == 0, result.stderr

    capture = cv2.VideoCapture(str(tmp_path / "out.mp4"))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
        ok, frame = capture.read()
        assert ok and frame.shape[:2] == (80, 100)
    finally:
        capture.release()


def test_e2e_cli_video_stabiliza(tmp_path):
    """`astroframe video --mode stabilize`: o disco deixa de derivar no ecrã."""
    clip = tmp_path / "clip.avi"
    _write_clip(clip, n_frames=4, shift=3)

    result = _run_cli(
        [
            "video",
            "--input",
            str(clip),
            "--output",
            str(tmp_path / "stab.mp4"),
            "--mode",
            "stabilize",
            "--fast",
        ],
        tmp_path,
    )
    assert result.returncode == 0, result.stderr

    capture = cv2.VideoCapture(str(tmp_path / "stab.mp4"))
    centers: list[tuple[int, int]] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            centers.append(_disk_center(frame))
    finally:
        capture.release()

    assert len(centers) == 4
    xs = [c[0] for c in centers]
    assert max(xs) - min(xs) <= 2  # a deriva de 3 px/frame foi anulada
    assert all(c[0] > 0 for c in centers)


def test_e2e_cli_video_stack(tmp_path):
    """`astroframe video --mode stack`: PNG com os melhores frames alinhados."""
    clip = tmp_path / "clip.avi"
    _write_clip(clip, n_frames=4)

    result = _run_cli(
        [
            "video",
            "--input",
            str(clip),
            "--output",
            str(tmp_path / "stack.png"),
            "--mode",
            "stack",
            "--stack-n",
            "2",
        ],
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "stack.png").exists()
    stacked = cv2.imread(str(tmp_path / "stack.png"))
    assert stacked is not None and stacked.shape[:2] == (80, 100)


def test_e2e_cli_config_template(tmp_path):
    """`astroframe config-template`: YAML válido gerado e legível de novo."""
    out = tmp_path / "config.yaml"
    result = _run_cli(["config-template", "--output", str(out)], tmp_path)
    assert result.returncode == 0, result.stderr
    assert out.exists()
    config = AstroFrameConfig.from_yaml(str(out))
    assert config.stabilizer.param1 > 0


def test_e2e_cli_autotune_exporta_config(tmp_path):
    """`astroframe autotune`: otimiza contra samples/ e exporta o JSON final."""
    from astroframe.calibration.store import CalibrationItem, CalibrationStore
    from astroframe.core.stabilizer import DiskDetection

    samples = tmp_path / "samples"
    samples.mkdir()
    store = CalibrationStore(samples / "calibration.json")
    for i in range(2):
        image, cx, cy = make_disk_image(height=200, width=200, radius=50)
        cv2.imwrite(str(samples / f"a{i}.jpg"), image)
        store.items[f"a{i}.jpg"] = CalibrationItem(
            f"a{i}.jpg", "image", None, 200, 200, [DiskDetection(cx, cy, 50)]
        )
    store.save()

    export = tmp_path / "tuned.json"
    result = _run_cli(
        [
            "autotune",
            "--samples",
            str(samples),
            "--budget",
            "0.5",
            "--seed",
            "7",
            "--no-anneal",
            "--export",
            str(export),
        ],
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert export.exists()


# ---------------------------------------------------------------------------
# pipeline completa (em processo)
# ---------------------------------------------------------------------------


def test_e2e_pipeline_imagem_completa():
    """Estabilizar -> melhorar -> polir numa imagem, com todas as etapas válidas."""
    image, *_ = make_disk_image(add_noise=True)
    config = AstroFrameConfig()
    height, width = image.shape[:2]

    result = process_image(image, config)

    assert isinstance(result, ProcessResult)
    assert result.original.shape[:2] == image.shape[:2]
    assert result.stabilized.shape == result.original.shape
    assert result.enhanced.shape == result.original.shape
    assert result.enhanced_raw.shape == result.original.shape
    assert result.detection is not None
    # raio original (90) × fator do auto-crop (~1.33); centro dentro da imagem
    assert 45 <= result.detection.radius <= 160
    assert 0 <= result.detection.cx < width and 0 <= result.detection.cy < height

    # a estabilização centrou mesmo o disco na imagem de saída
    sx, sy = _disk_center(result.stabilized)
    assert abs(sx - width // 2) <= 15 and abs(sy - height // 2) <= 15

    # o polimento é aplicado (deteção presente) e muda o resultado
    assert not np.array_equal(result.enhanced, result.enhanced_raw)
    # a melhoria mudou a imagem face à estabilizada
    assert not np.array_equal(result.enhanced_raw, result.stabilized)


def test_e2e_pipeline_process_path(tmp_path):
    """`process_path`: ficheiro -> pipeline -> resultado por etapas."""
    image, *_ = make_disk_image()
    src = tmp_path / "foto.jpg"
    cv2.imwrite(str(src), image)

    result = process_path(src)
    assert result.detection is not None
    assert result.enhanced.dtype == np.uint8
    assert result.enhanced.shape[:2] == (360, 480)


def test_e2e_pipeline_imagem_sem_disco_passa_por_tudo():
    """Sem astro limpo, a pipeline continua e aplica polimento só se detetar algo."""
    rng = np.random.default_rng(3)
    background = rng.normal(50, 15, (240, 320, 3))
    image = np.clip(background, 0, 255).astype(np.uint8)

    result = process_image(image)
    assert result.enhanced.shape == image.shape
    assert result.enhanced.dtype == np.uint8
    # invariante da pipeline: sem deteção não há polimento (e vice-versa)
    assert np.array_equal(result.enhanced, result.enhanced_raw) == (result.detection is None)
    # a melhoria mexeu na imagem (denoise/CLAHE/unsharp sobre o ruído)
    assert not np.array_equal(result.enhanced_raw, result.original)


def test_e2e_pipeline_video_completa(tmp_path):
    """Vídeo completo pela pipeline CLI: tamanho e nº de frames preservados."""
    clip = tmp_path / "clip.avi"
    _write_clip(clip, n_frames=5)

    result = _run_cli(
        [
            "video",
            "--input",
            str(clip),
            "--output",
            str(tmp_path / "final.mp4"),
            "--mode",
            "enhance",
            "--fast",
        ],
        tmp_path,
    )
    assert result.returncode == 0, result.stderr

    capture = cv2.VideoCapture(str(tmp_path / "final.mp4"))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
        assert (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (
            100,
            80,
        )
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            assert frame.dtype == np.uint8
    finally:
        capture.release()


# ---------------------------------------------------------------------------
# pontos de entrada headless (validator / enhancer_trainer)
# ---------------------------------------------------------------------------


def test_e2e_validator_check_via_script(tmp_path):
    """`validator.py --check`: relatório completo contra o ground truth, sem janela."""
    import shutil

    from astroframe.calibration.store import CalibrationItem, CalibrationStore
    from astroframe.core.stabilizer import DiskDetection

    samples = tmp_path / "samples"
    samples.mkdir()
    store = CalibrationStore(samples / "calibration.json")
    for i in range(2):
        image, cx, cy = make_disk_image()
        cv2.imwrite(str(samples / f"v{i}.jpg"), image)
        store.items[f"v{i}.jpg"] = CalibrationItem(
            f"v{i}.jpg", "image", None, 480, 360, [DiskDetection(cx, cy, 90)]
        )
    store.save()

    # o GT canónico vive em Logs/train/calibration.json (o `migrate_legacy`
    # copiaria o GT real do repositório para lá; pré-criamos o ficheiro para o
    # `--check` validar contra O NOSSO ground truth)
    train_dir = tmp_path / "data" / "train"
    train_dir.mkdir(parents=True)
    shutil.copy2(samples / "calibration.json", train_dir / "calibration.json")

    result = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "validator.py"), "--check", "--samples", str(samples)],
        capture_output=True,
        text=True,
        timeout=180,
        env=_cli_env(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert "Score vs guia manual" in result.stdout
    assert "Recall 100%" in result.stdout  # deteção perfeita nas imagens sintéticas


def test_e2e_enhancer_check_via_script(tmp_path):
    """`enhancer_trainer.py --check`: relatório de qualidade por amostra, sem janela."""
    from astroframe.calibration.store import CalibrationItem, CalibrationStore
    from astroframe.core.stabilizer import DiskDetection

    samples = tmp_path / "samples"
    samples.mkdir()
    store = CalibrationStore(samples / "calibration.json")
    for i in range(1):
        image, cx, cy = make_disk_image()
        cv2.imwrite(str(samples / f"e{i}.jpg"), image)
        store.items[f"e{i}.jpg"] = CalibrationItem(
            f"e{i}.jpg", "image", None, 480, 360, [DiskDetection(cx, cy, 90)]
        )
    store.save()

    result = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "enhancer_trainer.py"), "--check", "--samples", str(samples)],
        capture_output=True,
        text=True,
        timeout=180,
        env=_cli_env(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert "sem CNN" in result.stdout or "★" in result.stdout
