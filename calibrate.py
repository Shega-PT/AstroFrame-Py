"""Ponto de entrada da calibração: interface dedicada ao ajuste dos círculos.

Equivalente ao `main.py`, mas para a **calibração**: carrega as imagens e os
frames de vídeo da pasta `samples/`, permite ajustar os círculos (astros) à
mão e valida a deteção automática contra o ground truth em todas as amostras.

Uso:
    python calibrate.py                # interface em http://127.0.0.1:7860
    python calibrate.py --samples samples --port 7861
    python calibrate.py --config config.yaml --share
"""

from __future__ import annotations

import argparse
import sys

from astroframe.ui.calibration_app import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calibrate",
        description="AstroFrame — calibração: ajuste manual dos círculos e validação da deteção.",
    )
    parser.add_argument(
        "--samples",
        default="samples",
        help="Pasta com as imagens e vídeos de exemplo (varrida recursivamente)",
    )
    parser.add_argument("--config", default=None, help="Caminho para um config.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Gera um link público (Gradio share)")
    parser.add_argument("--no-browser", action="store_true", help="Não abre o navegador automaticamente")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(
            samples_dir=args.samples,
            config_path=args.config,
            host=args.host,
            port=args.port,
            share=args.share,
            inbrowser=not args.no_browser,
        )
    except Exception:
        import logging

        logging.exception("Falha ao arrancar a calibração")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
