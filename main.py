"""Ponto de entrada único: arranca o backend (motor de processamento) e o frontend (Gradio) juntos.

Uso:
    python main.py                # abre a interface em http://127.0.0.1:7860
    python main.py --port 7861    # outra porta
    python main.py --config config.yaml --share
"""

from __future__ import annotations

import argparse
import sys

from astroframe.paths import migrate_legacy, setup_logging
from astroframe.ui.gradio_app import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main",
        description="AstroFrame — abre o frontend (Gradio) ligado ao backend (pipeline).",
    )
    parser.add_argument("--config", default=None, help="Caminho para um config.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Gera um link público (Gradio share)")
    parser.add_argument("--no-browser", action="store_true", help="Não abre o navegador automaticamente")
    return parser


def main(argv: list[str] | None = None) -> int:
    migrate_legacy()
    setup_logging("astroframe.log")
    args = build_parser().parse_args(argv)
    try:
        # O backend corre no mesmo processo do Gradio: cada clique em "Processar"
        # invoca a pipeline (config.py, core/) dentro do servidor.
        run(
            config_path=args.config,
            host=args.host,
            port=args.port,
            share=args.share,
            inbrowser=not args.no_browser,
        )
    except Exception:
        import logging

        logging.exception("Falha ao arrancar a aplicação")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
