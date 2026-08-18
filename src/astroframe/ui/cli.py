"""Linha de comando: lote de fotos, vídeos (estabilizar/melhorar/stack) e config."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

from astroframe import __version__
from astroframe.ai.tuner import DEFAULT_EXPORT_NAME, run_autotune
from astroframe.config import AstroFrameConfig
from astroframe.core.enhancer import enhance_image
from astroframe.core.pipeline import process_path
from astroframe.core.polish import polish_image
from astroframe.core.stabilizer import AntiJitterStabilizer, DiskDetection, center_and_stabilize
from astroframe.paths import setup_logging, train_dir
from astroframe.video.reader import FrameReader
from astroframe.video.select import sharpness
from astroframe.video.stacking import stack_frames

logger = logging.getLogger(__name__)


def _load_config(path: str | None) -> AstroFrameConfig:
    if path:
        return AstroFrameConfig.from_yaml(path)
    return AstroFrameConfig()


def _progress(reader: FrameReader, description: str):
    total = reader.frame_count or None
    return tqdm(reader, total=total, desc=description, unit="frame")


def process_images(paths: list[str], output_dir: str, config: AstroFrameConfig) -> tuple[int, int]:
    """Processa fotos em lote, continuando mesmo se algum ficheiro falhar.

    Devolve (sucessos, falhas).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    successes = failures = 0
    for path in paths:
        try:
            result = process_path(path, config)
        except Exception as exc:
            failures += 1
            logger.error("Falha ao processar %s: %s", path, exc)
            continue
        out_path = output_dir / f"{Path(path).stem}_processed.png"
        cv2.imwrite(str(out_path), result.enhanced)
        successes += 1
        logger.info("Processado %s -> %s", path, out_path)
    if failures:
        logger.warning("Lote concluído: %d processados, %d falhas", successes, failures)
    if successes == 0:
        raise RuntimeError("Nenhum ficheiro foi processado com sucesso")
    return successes, failures


def process_video(
    path: str,
    output: str | None,
    config: AstroFrameConfig,
    mode: str,
    stack_n: int | None,
    fast: bool,
) -> str:
    if mode == "stack":
        if fast:
            logger.info("A opção --fast não tem efeito no modo stack.")
        best: list[tuple[float, object]] = []
        with FrameReader(path) as reader:
            for frame in _progress(reader, "Seleção de frames"):
                best = sorted(
                    best + [(sharpness(frame, config), frame)], key=lambda item: item[0], reverse=True
                )[: stack_n or config.stacking.n_best]
        aligned = [center_and_stabilize(frame, config)[0] for _, frame in best]
        stacked = stack_frames(aligned, config.stacking)
        out_path = output or str(Path(path).with_suffix(".png"))
        cv2.imwrite(out_path, stacked)
        logger.info("Stack de %d frames -> %s", len(aligned), out_path)
        return out_path

    out_path = output or str(Path(path).with_suffix(".stabilized.mp4"))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    logger.warning(
        "Nota: o áudio não é copiado (cv2.VideoWriter). Para preservar o som, "
        "junte a faixa do original ao ficheiro exportado com ffmpeg."
    )
    engine = AntiJitterStabilizer(config=config)
    with FrameReader(path) as reader:
        writer = cv2.VideoWriter(out_path, fourcc, reader.fps or 30.0, reader.size)
        if not writer.isOpened():
            raise OSError(f"Não foi possível abrir o escritor de vídeo: {out_path}")
        try:
            for frame in _progress(reader, "Estabilização"):
                stabilized, _detection = engine.stabilize(frame)
                frame_out = (
                    stabilized
                    if mode == "stabilize"
                    else enhance_image(stabilized, config, use_denoise=not fast)
                )
                if (detection := engine.last_detection) is not None:
                    frame_out = polish_image(
                        frame_out,
                        DiskDetection(reader.size[0] // 2, reader.size[1] // 2, detection.radius),
                        config,
                    )
                writer.write(frame_out)
        finally:
            writer.release()
    logger.info("Vídeo processado -> %s", out_path)
    return out_path


def _run_autotune_cli(args) -> None:
    """Executa o auto-tuning a partir dos argumentos da CLI e imprime o relatório."""
    from astroframe.ai.feedback import FeedbackDB

    config = _load_config(args.config)
    db = FeedbackDB()
    if args.reset:
        removed = db.reset_tuning()
        print(f"Histórico de auto-tuning apagado ({removed} registo(s)).")
    export = args.export or (train_dir() / DEFAULT_EXPORT_NAME)
    print(
        f"AstroFrame — auto-tuning contra {args.samples} (orçamento {args.budget:g}s, "
        f"seed {args.seed}, recozimento {'ligado' if args.anneal else 'desligado'})"
    )
    result = run_autotune(
        samples_dir=args.samples,
        config=config,
        budget_s=args.budget,
        seed=args.seed,
        anneal=args.anneal,
        params_filter=args.params,
        export_path=export,
        profile=args.profile,
        db=db,
    )
    print("\n".join(result.lines))
    print(f"Config otimizada exportada: {export}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="astroframe",
        description=(
            "AstroFrame — estabilização geométrica e melhoria automática de "
            "astrofotografias e astrovídeos."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Inicia a interface Gradio no navegador")
    serve.add_argument("--config", default=None, help="Caminho para um config.yaml")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7860)
    serve.add_argument("--share", action="store_true", help="Gera um link público (Gradio share)")

    process = sub.add_parser("process", help="Processa fotos em lote")
    process.add_argument("--input", nargs="+", required=True, help="Ficheiros de imagem")
    process.add_argument("--output-dir", required=True)
    process.add_argument("--config", default=None)

    video = sub.add_parser("video", help="Processa um vídeo (estabilizar, melhorar ou stack)")
    video.add_argument("--input", required=True)
    video.add_argument("--output", default=None)
    video.add_argument("--config", default=None)
    video.add_argument("--mode", choices=["stabilize", "enhance", "stack"], default="enhance")
    video.add_argument(
        "--fast", action="store_true", help="Omite o denoising (mais rápido; recomendado em vídeos grandes)"
    )
    video.add_argument(
        "--stack-n",
        type=int,
        default=None,
        help="Nº de melhores frames para stacking (padrão: config.stacking.n_best)",
    )

    template = sub.add_parser("config-template", help="Gera um config.yaml com os valores por omissão")
    template.add_argument("--output", default="config.yaml")

    calibrate = sub.add_parser(
        "calibrate", help="Abre a interface de calibração (círculos manuais + validação)"
    )
    calibrate.add_argument("--samples", default="samples", help="Pasta com imagens/vídeos de exemplo")
    calibrate.add_argument("--config", default=None, help="Caminho para um config.yaml")
    calibrate.add_argument("--host", default="127.0.0.1")
    calibrate.add_argument("--port", type=int, default=7860)
    calibrate.add_argument("--share", action="store_true", help="Gera um link público (Gradio share)")

    autotune = sub.add_parser(
        "autotune",
        help="Otimiza todos os parâmetros da pipeline contra as amostras de samples/",
    )
    autotune.add_argument(
        "--samples", default="samples", help="Pasta com as imagens/vídeos de exemplo"
    )
    autotune.add_argument("--config", default=None, help="Caminho para um config.yaml (base)")
    autotune.add_argument(
        "--budget",
        type=float,
        default=60.0,
        help="Orçamento de tempo em segundos para a otimização (omissão: 60)",
    )
    autotune.add_argument("--seed", type=int, default=42, help="Semente determinística (omissão: 42)")
    autotune.add_argument(
        "--no-anneal",
        action="store_false",
        dest="anneal",
        help="Desliga o recozimento (aceitar pioras) — busca mais conservadora",
    )
    autotune.add_argument(
        "--params",
        default=None,
        help="Subconjunto de parâmetros a otimizar (ex.: 'clahe.clip_limit,denoise.h')",
    )
    autotune.add_argument(
        "--profile",
        default="tuning",
        help="Perfil de aprendizagem no banco (omissão: 'tuning')",
    )
    autotune.add_argument(
        "--export",
        default=None,
        help=f"Ficheiro JSON com a configuração otimizada (omissão: Logs/train/{DEFAULT_EXPORT_NAME})",
    )
    autotune.add_argument(
        "--reset",
        action="store_true",
        help="Apaga o histórico de auto-tuning do banco de aprendizagem antes de otimizar",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    setup_logging("astroframe.log")

    try:
        if args.command == "serve":
            from astroframe.ui.gradio_app import run as run_gradio

            run_gradio(config_path=args.config, host=args.host, port=args.port, share=args.share)
        elif args.command == "process":
            process_images(args.input, args.output_dir, _load_config(args.config))
        elif args.command == "video":
            if args.mode != "stack" and args.stack_n is not None:
                logger.warning("--stack-n só tem efeito no modo stack; a ignorar.")
            process_video(
                args.input, args.output, _load_config(args.config), args.mode, args.stack_n, args.fast
            )
        elif args.command == "config-template":
            _load_config(None).to_yaml(args.output)
            print(f"Config escrito em {args.output}")
        elif args.command == "calibrate":
            from astroframe.ui.calibration_app import run as run_calibration

            run_calibration(
                samples_dir=args.samples,
                config_path=args.config,
                host=args.host,
                port=args.port,
                share=args.share,
            )
        elif args.command == "autotune":
            _run_autotune_cli(args)
    except Exception:
        logger.exception("Falha na execução")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
