"""Testes da linha de comando (lote resiliente)."""

from __future__ import annotations

import cv2
import pytest

from astroframe.config import AstroFrameConfig
from astroframe.ui.cli import process_images
from tests.helpers import make_disk_image


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
