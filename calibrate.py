"""Ponto de entrada da calibração: interface dedicada ao ajuste dos círculos.

Equivalente ao `main.py`, mas para a **calibração**: carrega as imagens e os
frames de vídeo da pasta `samples/`, permite ajustar os círculos/elipses
(astros) à mão e valida a deteção automática contra o ground truth em todas
as amostras.

Uso:
    python calibrate.py                    # janela desktop (tkinter)
    python calibrate.py --ui gradio        # interface no navegador
    python calibrate.py --samples samples
    python calibrate.py --config config.yaml

Workflow recomendado (na janela desktop):
    1.ª passagem — deteção desligada: desenhas os astros à mão e guardas.
    2.ª passagem — deteção ligada: a deteção preenche as amostras novas e
    validas/ajustas o que for preciso, voltando a guardar.
"""

from __future__ import annotations

import argparse
import sys

from astroframe.paths import migrate_legacy, setup_logging


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
    parser.add_argument(
        "--ui",
        choices=("tk", "gradio"),
        default="tk",
        help="Interface: janela desktop tkinter (omissão) ou navegador (Gradio)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="(gradio) IP do servidor")
    parser.add_argument("--port", type=int, default=7860, help="(gradio) porta do servidor")
    parser.add_argument("--share", action="store_true", help="(gradio) gera um link público")
    parser.add_argument("--no-browser", action="store_true", help="(gradio) não abre o navegador")
    return parser


def main(argv: list[str] | None = None) -> int:
    migrate_legacy()
    setup_logging("calibrate.log")
    args = build_parser().parse_args(argv)
    try:
        if args.ui == "gradio":
            from astroframe.ui.calibration_app import run as run_gradio

            run_gradio(
                samples_dir=args.samples,
                config_path=args.config,
                host=args.host,
                port=args.port,
                share=args.share,
                inbrowser=not args.no_browser,
            )
        else:
            from astroframe.ui.calibration_tk import run as run_tk

            run_tk(samples_dir=args.samples, config_path=args.config)
    except Exception:
        import logging

        logging.exception("Falha ao arrancar a calibração")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())