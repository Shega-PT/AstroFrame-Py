"""enhancer_trainer.py — treino/validação da CNN de edição de imagem (residual).

A rede de edição (melhoria/coloração) é treinada com **pares (x, y)** no
canal L (LAB) da imagem já melhorada pelo pipeline clássico (CLAHE +
denoising + nitidez):

* **manual** (interface): cada amostra é mostrada lado a lado — sem CNN vs
  com CNN — e o utilizador julga: *Válido* guarda o par (entrada, saída com
  CNN); *Rejeitado* guarda (entrada, entrada), ensinando a rede a não mexer.
* **avaliação headless** (`--check`): compara todas as amostras com/sem CNN
  e imprime o relatório de estrelas, sem janela.

O campeão promovido fica em `Logs/weights/enhancer_cnn.npz` (lido pelo
`enhance_image` quando `config.ai.cnn_enhance` está ligado). O estado do
treino e a calibração vivem por omissão em `Logs/train/` e cada ronda deixa
o relatório em `Logs/logs/ia/`. O treino automático (`--auto`) foi removido:
a rede só é treinada com o julgamento manual da interface.
"""

import argparse
import dataclasses
import json
import logging
import queue
import shutil
import sys
import threading
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from astroframe.ai.cnn import SmallCNN, fit_residual
from astroframe.ai.feedback import FeedbackDB
from astroframe.ai.score import score_image
from astroframe.calibration.scan import load_frame, scan_samples
from astroframe.calibration.store import CalibrationStore
from astroframe.config import AstroFrameConfig
from astroframe.core.enhancer import enhance_image
from astroframe.core.stabilizer import DiskDetection, find_disks_for_calibration
from astroframe.paths import (
    calibration_json,
    logs_ia_dir,
    migrate_legacy,
    setup_logging,
    staging_dir,
    train_dir,
    weights_dir,
)

logger = logging.getLogger("enhancer")

ENHANCER_STATE_VERSION = 1
DEFAULT_STATE_NAME = "enhancer_state.json"
DEFAULT_EXPORT_NAME = "enhancer_model.npz"
ENHANCER_CANONICAL_PATH = weights_dir() / "enhancer_cnn.npz"
MODEL_DIR = staging_dir()

TILE = 48
MAX_CROPS_PER_SAMPLE = 8
NOISE_SIGMA = 12.0
BLUR_KSIZE = 5
AUTO_SEED = 42


def _l_channel(image_bgr: np.ndarray) -> np.ndarray:
    """Canal L (LAB) da imagem BGR em float 0–1."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    return lab[:, :, 0].astype(np.float64) / 255.0


def degrade(enhanced_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Degradação sintética: ruído gaussiano + desfoque sobre a melhoria limpa."""
    lab = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2LAB)
    noisy = np.clip(lab.astype(np.float64) + rng.normal(0.0, NOISE_SIGMA, lab.shape), 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(noisy, (BLUR_KSIZE, BLUR_KSIZE), 0)
    return cv2.cvtColor(blurred, cv2.COLOR_LAB2BGR)


def crop_pairs(
    clean_bgr: np.ndarray, degraded_bgr: np.ndarray, rng: np.random.Generator
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pares (x, y) em recortes 48×48 do canal L: x = degradado, y = limpo."""
    clean = _l_channel(clean_bgr)
    degraded = _l_channel(degraded_bgr)
    h, w = clean.shape
    if h < TILE or w < TILE:
        return []
    count = min(MAX_CROPS_PER_SAMPLE, max(1, (h * w) // (TILE * TILE)))
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(count):
        y0 = int(rng.integers(0, h - TILE + 1))
        x0 = int(rng.integers(0, w - TILE + 1))
        pairs.append(
            (degraded[y0 : y0 + TILE, x0 : x0 + TILE].copy(), clean[y0 : y0 + TILE, x0 : x0 + TILE].copy())
        )
    return pairs


def synthetic_pairs(
    sample,
    config: AstroFrameConfig,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pares sintéticos de uma amostra: y = melhoria clássica, x = degradada."""
    frame = load_frame(sample)
    clean_bgr = enhance_image(frame, config, use_denoise=True)
    degraded_bgr = degrade(clean_bgr, rng)
    return crop_pairs(clean_bgr, degraded_bgr, rng)


def evaluate_pairs(model: SmallCNN, pairs: list[tuple[np.ndarray, np.ndarray]], seed: int) -> dict:
    """Avaliação determinística 80/20: `mean_delta` (1 − MSE/255, melhor = 1).

    O resíduo previsto `model(x)` é comparado com o resíduo verdadeiro
    `y − x` na divisão de validação (semente própria, separada do treino).
    """
    if not pairs:
        return {"mean_delta": 0.0, "mse": 1.0}
    X = np.stack([p[0] for p in pairs])[:, None]
    Y = np.stack([p[1] for p in pairs])[:, None]
    residuals = Y - X
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    n_val = max(1, int(len(X) * 0.2))
    val_idx = order[:n_val]
    out, _ = model.forward(X[val_idx])
    mse = float(np.mean((out - residuals[val_idx]) ** 2))
    return {"mean_delta": float(np.clip(1.0 - mse, 0.0, 1.0)), "mse": mse}


def _detection(store: CalibrationStore, sample_key: str) -> DiskDetection | None:
    item = store.items.get(sample_key)
    if item is not None and item.circles:
        return item.circles[0]
    return None


def sample_stars(
    frame: np.ndarray,
    detection: DiskDetection | None,
    config: AstroFrameConfig,
    with_cnn: bool,
    disks: list[DiskDetection] | None = None,
) -> float:
    """Estrelas (0–5) da imagem melhorada, com ou sem a CNN residual.

    `disks` reutiliza as deteções já calculadas pelo chamador — a
    avaliação nunca volta a correr o Hough.
    """
    cfg = dataclasses.replace(config, ai=dataclasses.replace(config.ai, cnn_enhance=with_cnn))
    enhanced = enhance_image(frame, cfg, use_denoise=True)
    return float(score_image(enhanced, detection, config, disks=disks).stars)


class EnhancerState:
    """Estado do treino da CNN de edição (rounds + pares recolhidos)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.round = 0
        self.rounds: list[dict] = []
        self.pairs_positive = 0
        self.pairs_identity = 0
        self.series: list[dict] = []
        self.load()

    def load(self) -> None:
        self.round = 0
        self.rounds = []
        self.pairs_positive = 0
        self.pairs_identity = 0
        self.series = []
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            logger.warning("Estado do enhancer ilegível (%s), a começar vazio: %s", self.path, exc)
            return
        if not isinstance(data, dict) or data.get("version") != ENHANCER_STATE_VERSION:
            logger.warning("Estado do enhancer com versão desconhecida (%s), a começar vazio", self.path)
            return
        self.round = int(data.get("round", 0))
        self.rounds = list(data.get("rounds") or [])
        self.pairs_positive = int(data.get("pairs_positive", 0))
        self.pairs_identity = int(data.get("pairs_identity", 0))
        self.series = list(data.get("series") or [])

    def save(self) -> None:
        data = {
            "version": ENHANCER_STATE_VERSION,
            "round": self.round,
            "rounds": self.rounds,
            "pairs_positive": self.pairs_positive,
            "pairs_identity": self.pairs_identity,
            "series": self.series,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def reset(self) -> None:
        self.round = 0
        self.rounds = []
        self.pairs_positive = 0
        self.pairs_identity = 0
        self.series = []
        self.save()

    def begin_round(self, mode: str) -> int:
        self.round += 1
        self.rounds.append(
            {
                "round": self.round,
                "mode": mode,
                "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "ended": None,
                "pairs_positive": 0,
                "pairs_identity": 0,
                "mean_delta": None,
                "mse": None,
                "stars_cnn": None,
                "stars_plain": None,
                "promoted": False,
            }
        )
        self.save()
        return self.round

    def end_round(self, metrics: dict | None = None) -> None:
        if not self.rounds or self.rounds[-1].get("round") != self.round:
            return
        record = self.rounds[-1]
        record["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
        record["pairs_positive"] = self.pairs_positive
        record["pairs_identity"] = self.pairs_identity
        if metrics:
            for key in ("mean_delta", "mse", "stars_cnn", "stars_plain", "promoted"):
                if metrics.get(key) is not None:
                    record[key] = metrics[key]
        self.save()


@dataclass
class EnhancerReport:
    """Balanço de uma passagem: estrelas com/sem CNN + pares recolhidos."""

    samples_done: int
    samples_total: int
    stars_cnn: float | None
    stars_plain: float | None
    pairs: int
    errors: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        if self.stars_cnn is not None and self.stars_plain is not None:
            stars = f"Melhoria CNN: {self.stars_cnn:.1f}★ vs sem CNN {self.stars_plain:.1f}★"
        elif self.stars_cnn is not None:
            stars = f"Melhoria CNN: {self.stars_cnn:.1f}★ (sem deteção para comparar)"
        else:
            stars = "Melhoria CNN: sem amostras avaliadas"
        lines = [stars, f"Pares recolhidos: {self.pairs}"]
        for error in self.errors:
            lines.append(f"  ! {error}")
        return lines


@dataclass
class AutoSeriesReport:
    """Balanço de uma série automática."""

    series: int
    samples_done: int
    samples_total: int
    pairs_added: int
    mean_delta: float | None
    mse: float | None
    stars_cnn: float | None
    stars_plain: float | None
    errors: list[str] = field(default_factory=list)
    stopped: bool = False


def build_report(
    samples,
    store: CalibrationStore,
    config: AstroFrameConfig,
    pairs_count: int = 0,
    progress: bool = False,
) -> EnhancerReport:
    """Relatório headless: avalia todas as amostras com e sem CNN (sem treinar).

    A deteção corre **uma única vez** por amostra e é reutilizada nas duas
    avaliações; com `progress` imprime o avanço por amostra (essencial em
    pastas grandes/vídeos longos — o `--check` deixa de parecer pendurado).
    """
    done = 0
    stars_cnn: list[float] = []
    stars_plain: list[float] = []
    errors: list[str] = []
    total = len(samples)
    for sample in samples:
        try:
            frame = load_frame(sample)
        except Exception as exc:
            errors.append(f"{sample.label}: erro ao ler ({exc})")
            continue
        item = store.get_item(sample.key)
        expected_n = len(item.circles) if item is not None else 0
        try:
            disks = find_disks_for_calibration(frame, config, expected_n=expected_n)
        except Exception as exc:
            errors.append(f"{sample.label}: erro na deteção ({exc})")
            continue
        detection = _detection(store, sample.key)
        stars_cnn.append(sample_stars(frame, detection, config, with_cnn=True, disks=disks))
        stars_plain.append(sample_stars(frame, detection, config, with_cnn=False, disks=disks))
        done += 1
        if progress:
            print(f"  [{done}/{total}] {sample.label} …", flush=True)
    return EnhancerReport(
        samples_done=done,
        samples_total=total,
        stars_cnn=float(np.mean(stars_cnn)) if stars_cnn else None,
        stars_plain=float(np.mean(stars_plain)) if stars_plain else None,
        pairs=pairs_count,
        errors=errors,
    )


def train_enhancer_round(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    state: EnhancerState,
    round_n: int,
    epochs: int = 40,
    champion_path: str | Path | None = None,
    db: FeedbackDB | None = None,
    seed: int = AUTO_SEED,
) -> dict:
    """Treina/continua a CNN residual com os pares acumulados e compara com o
    campeão registado no banco (`add_model`).

    Warm-start a partir do campeão quando existe; o candidato fica em staging
    (`Logs/weights/staging/enhancer_rN.npz`) e, se for **estritamente melhor**
    na métrica `mean_delta`, é promovido (copia para o caminho canónico
    `Logs/weights/enhancer_cnn.npz`); senão o treino seguinte parte dos pesos
    do campeão. Sem pares suficientes devolve `{"skipped": True}`. O relatório
    de cada ronda fica em `Logs/logs/ia/`.
    """
    if len(pairs) < 2:
        if db is not None:
            db.log("info", "enhancer", f"Série {round_n}: sem pares suficientes para a CNN")
        return {"skipped": True}
    warm = SmallCNN.load(champion_path) if champion_path else None
    model, fit = fit_residual(pairs, model=warm, epochs=epochs, seed=seed)
    eval_results = evaluate_pairs(model, pairs, seed + 1)
    staged = MODEL_DIR / f"enhancer_r{round_n}.npz"
    staged.parent.mkdir(parents=True, exist_ok=True)
    model.save(staged)
    db = db or FeedbackDB()
    result = db.add_model(
        "enhancer",
        staged,
        {"mean_delta": eval_results["mean_delta"], "mse": eval_results["mse"], "best_loss": fit.best_loss},
        dataset_size=len(pairs),
        source="enhancer-auto",
        round=round_n,
        metric_name="mean_delta",
    )
    champion = result["champion"]
    if result["promoted"]:
        shutil.copyfile(staged, ENHANCER_CANONICAL_PATH)
        db.log("info", "enhancer", f"Série {round_n}: novo campeão CNN de edição", eval_results)
    else:
        db.log(
            "info",
            "enhancer",
            f"Série {round_n}: CNN pior que o campeão; próxima série parte do campeão",
            eval_results,
        )
    state.series.append(
        {
            "round": round_n,
            "pairs": len(pairs),
            "mean_delta": eval_results["mean_delta"],
            "mse": eval_results["mse"],
            "promoted": bool(result["promoted"]),
        }
    )
    state.save()
    report_path = logs_ia_dir() / f"enhancer_round_{round_n}.json"
    report_path.write_text(
        json.dumps(
            {
                "round": round_n,
                "pairs": len(pairs),
                **eval_results,
                "best_loss": float(fit.best_loss),
                "promoted": bool(result["promoted"]),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "skipped": False,
        "mean_delta": eval_results["mean_delta"],
        "mse": eval_results["mse"],
        "promoted": bool(result["promoted"]),
        "staged": staged,
        "champion_path": champion["path"] if champion else None,
    }


def run_auto_headless(
    samples_dir: str = "samples",
    config_path: str | None = None,
    state_path: str | Path | None = None,
    series: int = 3,
    export_path: str | Path | None = None,
    epochs: int = 40,
    seed: int = AUTO_SEED,
) -> int:
    """Treino headless mantido declarado (sem entrada na CLI desde a remoção do
    auto-treino): N séries com pares sintéticos e campeão no banco.

    Pode ser chamado programaticamente pelos testes; a interface usa o
    julgamento manual. Entre séries o candidato é comparado com o melhor
    registado no banco e só é promovido se for estritamente melhor.
    """
    config = AstroFrameConfig.from_yaml(config_path) if config_path else AstroFrameConfig()
    state = EnhancerState(state_path or train_dir() / DEFAULT_STATE_NAME)
    samples = scan_samples(samples_dir)
    store = CalibrationStore(calibration_json(samples_dir))
    db = FeedbackDB()

    print(
        f"AstroFrame — treino automático da CNN de edição ({len(samples)} amostras, {series} série(s))",
        flush=True,
    )
    all_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    champion_path: str | Path | None = None
    final_metrics: dict | None = None
    for current in range(1, series + 1):
        state.begin_round("auto")
        pairs_series: list[tuple[np.ndarray, np.ndarray]] = []
        stars_cnn: list[float] = []
        stars_plain: list[float] = []
        errors: list[str] = []
        done = 0
        for sample in samples:
            try:
                rng = np.random.default_rng(seed + zlib.crc32(sample.label.encode("utf-8")))
                frame = load_frame(sample)
                clean_bgr = enhance_image(frame, config, use_denoise=True)
                degraded_bgr = degrade(clean_bgr, rng)
                pairs_series.extend(crop_pairs(clean_bgr, degraded_bgr, rng))
                detection = _detection(store, sample.key)
                item = store.get_item(sample.key)
                expected_n = len(item.circles) if item is not None else 0
                disks = find_disks_for_calibration(frame, config, expected_n=expected_n)
                cfg_cnn = dataclasses.replace(config, ai=dataclasses.replace(config.ai, cnn_enhance=True))
                enhanced_cnn = enhance_image(frame, cfg_cnn, use_denoise=True)
                stars_cnn.append(float(score_image(enhanced_cnn, detection, config, disks=disks).stars))
                stars_plain.append(float(score_image(clean_bgr, detection, config, disks=disks).stars))
                done += 1
            except Exception as exc:
                errors.append(f"{sample.label}: erro ({exc})")
        all_pairs.extend(pairs_series)
        result = train_enhancer_round(
            all_pairs, state, current, epochs=epochs, champion_path=champion_path, db=db, seed=seed
        )
        if result is not None and not result["skipped"]:
            if result["champion_path"] is not None:
                champion_path = result["champion_path"]
            final_metrics = {
                "mean_delta": result["mean_delta"],
                "mse": result["mse"],
                "stars_cnn": float(np.mean(stars_cnn)) if stars_cnn else None,
                "stars_plain": float(np.mean(stars_plain)) if stars_plain else None,
                "promoted": result["promoted"],
            }
            print(
                f"Série {current}: {done}/{len(samples)} amostras · "
                f"+{len(pairs_series)} pares (acumulado {len(all_pairs)}) · "
                f"qualidade {100 * result['mean_delta']:.1f}% · "
                f"{'PROMOVIDA (novo campeão)' if result['promoted'] else 'mantém o campeão'}"
            )
        else:
            final_metrics = None
            print(f"Série {current}: {done}/{len(samples)} amostras · sem pares suficientes")
        for error in errors:
            print(f"  ! {error}")
        state.end_round(final_metrics)

    export_path = Path(export_path) if export_path else train_dir() / DEFAULT_EXPORT_NAME
    if champion_path is not None:
        shutil.copyfile(champion_path, export_path)
        print(f"Modelo CNN de edição exportado: {export_path}")
    else:
        print("Sem modelo campeão para exportar.")
    report = build_report(samples, store, config, pairs_count=len(all_pairs))
    for line in report.lines():
        print(line)
    return 0


def run_check(
    samples_dir: str = "samples",
    config_path: str | None = None,
    state_path: str | Path | None = None,
) -> int:
    """`--check`: avalia todas as amostras com/sem CNN e imprime o relatório."""
    config = AstroFrameConfig.from_yaml(config_path) if config_path else AstroFrameConfig()
    state = EnhancerState(state_path or train_dir() / DEFAULT_STATE_NAME)
    samples = scan_samples(samples_dir)
    store = CalibrationStore(calibration_json(samples_dir))
    print(f"AstroFrame — verificação da melhoria CNN ({len(samples)} amostras)", flush=True)
    report = build_report(samples, store, config, pairs_count=0, progress=True)
    for line in report.lines():
        print(line)
    if state.round:
        print(f"Treinos anteriores: {state.round} série(s), {state.pairs_positive} pares válidos.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="enhancer_trainer",
        description=(
            "AstroFrame — treino/validação da CNN de edição de imagem (residual): "
            "manual lado a lado ou automático com pares sintéticos e campeão no banco."
        ),
    )
    parser.add_argument(
        "--samples",
        default="samples",
        help="Pasta com as imagens e vídeos de exemplo (varrida recursivamente)",
    )
    parser.add_argument("--config", default=None, help="Caminho para um config.yaml")
    parser.add_argument(
        "--state",
        default=None,
        help=f"Ficheiro de estado do treino (omissão: Logs/train/{DEFAULT_STATE_NAME})",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Apaga o progresso e o histórico de séries antes de começar",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Modo sem interface: avalia todas as amostras com/sem CNN e imprime o relatório",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    migrate_legacy()
    setup_logging("enhancer_trainer.log")
    args = build_parser().parse_args(argv)
    state_path = Path(args.state) if args.state else train_dir() / DEFAULT_STATE_NAME
    if args.reset_state and state_path.exists():
        EnhancerState(state_path).reset()
        print(f"Estado do enhancer reposto: {state_path}")
    try:
        if args.check:
            return run_check(args.samples, args.config, state_path)
        return run_gui(samples_dir=args.samples, config_path=args.config, state_path=state_path)
    except Exception as exc:
        logging.exception("Falha ao arrancar o enhancer_trainer")
        print(f"Erro: {exc}", file=sys.stderr)
        return 1


# --------------------------------------------------------------------------
# Interface desktop (tkinter): julgamento manual lado a lado
# --------------------------------------------------------------------------


def run_gui(
    samples_dir: str = "samples",
    config_path: str | None = None,
    state_path: str | Path | None = None,
) -> int:
    config = AstroFrameConfig.from_yaml(config_path) if config_path else AstroFrameConfig()
    state = EnhancerState(state_path or train_dir() / DEFAULT_STATE_NAME)
    samples = scan_samples(samples_dir)
    store = CalibrationStore(calibration_json(samples_dir))
    try:
        import tkinter as tk
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Interface manual indisponível: tkinter em falta. Usa --check.") from exc
    root = tk.Tk()
    root.withdraw()
    app = EnhancerTkApp(samples, store, config, state, root=root)
    root.mainloop()
    if app is not None and hasattr(app, "wait_idle"):
        app.wait_idle()
    return 0


class EnhancerTkApp:
    """Janela de treino manual: lado a lado sem-CNN vs com-CNN + Válido/Rejeitado.

    Válido guarda o par (entrada da CNN, saída com CNN); Rejeitado guarda
    (entrada, entrada) — a rede aprende onde não deve mexer. 'Treinar agora'
    treina a residual com os pares acumulados (warm-start do campeão) e
    compara com o campeão registado no banco.
    """

    def __init__(
        self,
        samples,
        store: CalibrationStore,
        config: AstroFrameConfig,
        state: EnhancerState,
        root=None,
    ):
        try:
            import tkinter as tk

            from PIL import Image, ImageTk
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Interface manual indisponível: tkinter/PIL em falta. Usa --check.") from exc
        self.tk = tk
        self.Image = Image
        self.ImageTk = ImageTk
        self.samples = samples
        self.store = store
        self.config = config
        self.state = state
        self.index = 0
        self.pairs: list[tuple[np.ndarray, np.ndarray]] = []
        self._champion_path: str | Path | None = None
        self._precomputed: list[tuple[np.ndarray, np.ndarray]] = []
        self._auto_stars: tuple[float, float] = (0.0, 0.0)
        self._loading_index = 0
        self._sample_results: queue.Queue | None = None
        self._workers: list[threading.Thread] = []
        self.root = root
        self._build_ui()
        self._load_sample()

    def _build_ui(self) -> None:
        tk = self.tk
        if self.root is None:
            self.root = tk.Tk()
            self.root.withdraw()
        self.top = tk.Toplevel(self.root)
        self.top.title("AstroFrame — Treino da CNN de edição (manual)")
        self.top.geometry("900x560")
        main = tk.Frame(self.top, padx=10, pady=10)
        main.pack(fill=tk.BOTH, expand=True)
        self.info = tk.StringVar(value="")
        tk.Label(main, textvariable=self.info, font=("", 10, "bold")).pack(anchor="w")
        row = tk.Frame(main)
        row.pack(fill=tk.BOTH, expand=True, pady=6)
        tk.Label(row, text="Sem CNN (entrada)", font=("", 9, "bold")).pack(side=tk.LEFT, expand=True)
        tk.Label(row, text="Com CNN (saída)", font=("", 9, "bold")).pack(side=tk.RIGHT, expand=True)
        images = tk.Frame(main)
        images.pack(fill=tk.BOTH, expand=True)
        self.left = tk.Label(images, bg="#1c1c1e")
        self.right = tk.Label(images, bg="#1c1c1e")
        self.left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))
        stars_row = tk.Frame(main)
        stars_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(stars_row, text="Avaliação manual:", font=("", 9)).pack(side=tk.LEFT)
        self.stars_var = tk.DoubleVar(value=3.0)
        self.stars_scale = tk.Scale(
            stars_row,
            from_=0.0,
            to=5.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.stars_var,
            length=200,
        )
        self.stars_scale.pack(side=tk.LEFT, padx=(8, 0))
        self.stars_auto_label = tk.StringVar(value="")
        tk.Label(stars_row, textvariable=self.stars_auto_label, fg="#c90", font=("", 9, "bold")).pack(
            side=tk.LEFT, padx=(12, 0)
        )
        buttons = tk.Frame(main)
        buttons.pack(fill=tk.X, pady=(8, 0))
        tk.Button(buttons, text="◀ Anterior", command=self._prev, bg="#445").pack(side=tk.LEFT)
        tk.Button(
            buttons, text="Válido", command=self._accept, bg="#1e7a3a", fg="white", font=("", 10, "bold")
        ).pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(
            buttons, text="Rejeitado", command=self._reject, bg="#9c2b2b", fg="white", font=("", 10, "bold")
        ).pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(buttons, text="Próximo ▶", command=self._next).pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(buttons, text="Treinar agora", command=self._train_now, bg="#0a7", fg="white").pack(
            side=tk.RIGHT
        )
        self.status = tk.StringVar(value="Pronto.")
        tk.Label(main, textvariable=self.status, fg="#0a7", wraplength=880).pack(anchor="w", pady=(6, 0))

    def _load_sample(self) -> None:
        if not self.samples:
            self.status.set("Sem amostras na pasta indicada.")
            return
        sample = self.samples[self.index]
        self._loading_index = self.index
        self._precomputed = []
        self._auto_stars = (0.0, 0.0)
        self.stars_var.set(3.0)
        self.stars_auto_label.set("")
        self.info.set(f"[{self.index + 1}/{len(self.samples)}] {sample.label} — a processar…")
        self.status.set(f"A processar {sample.label} (deteção + melhoria com/sem CNN)…")
        results: queue.Queue = queue.Queue()

        def work() -> None:
            try:
                frame = load_frame(sample)
                item = self.store.get_item(sample.key)
                expected_n = len(item.circles) if item is not None else 0
                disks = find_disks_for_calibration(frame, self.config, expected_n=expected_n)
                detection = _detection(self.store, sample.key)
                cfg_plain = dataclasses.replace(
                    self.config, ai=dataclasses.replace(self.config.ai, cnn_enhance=False)
                )
                plain_bgr = enhance_image(frame, cfg_plain, use_denoise=True)
                cfg_cnn = dataclasses.replace(
                    self.config, ai=dataclasses.replace(self.config.ai, cnn_enhance=True)
                )
                cnn_bgr = enhance_image(frame, cfg_cnn, use_denoise=True)
                stars_plain = score_image(plain_bgr, detection, self.config, disks=disks).stars
                stars_cnn = score_image(cnn_bgr, detection, self.config, disks=disks).stars
                results.put((plain_bgr, cnn_bgr, sample.label, stars_plain, stars_cnn))
            except Exception as exc:
                results.put(exc)

        thread = threading.Thread(target=work, daemon=True)
        self._load_thread = thread
        thread.start()
        self._sample_results = results
        self._workers.append(thread)
        self.top.after(50, self._poll_sample)

    def _poll_sample(self) -> None:
        try:
            outcome = self._sample_results.get_nowait()
        except queue.Empty:
            self.top.after(50, self._poll_sample)
            return
        if self._loading_index != self.index:
            return
        if isinstance(outcome, Exception):
            self.status.set(f"{self.samples[self.index].label}: erro ao processar ({outcome})")
            return
        plain_bgr, cnn_bgr, label, stars_plain, stars_cnn = outcome
        self._precomputed = [(plain_bgr, cnn_bgr)]
        self._auto_stars = (stars_plain, stars_cnn)
        self.stars_auto_label.set(f"auto: {stars_plain:.1f}★ → {stars_cnn:.1f}★")
        self.info.set(
            f"[{self.index + 1}/{len(self.samples)}] {label} — "
            f"sem CNN {stars_plain:.1f}★ · com CNN {stars_cnn:.1f}★ · "
            f"pares: {len(self.pairs)}"
        )
        self.status.set("Pronto.")
        self._show()

    def _show(self) -> None:
        if not self._precomputed:
            return
        plain_bgr, cnn_bgr = self._precomputed[0]
        self._photo_left = self._photo(plain_bgr)
        self._photo_right = self._photo(cnn_bgr)
        self.left.configure(image=self._photo_left)
        self.right.configure(image=self._photo_right)

    def _photo(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = self.Image.fromarray(rgb)
        max_size = 360
        scale = min(1.0, max_size / max(image.size))
        if scale < 1.0:
            image = image.resize(
                (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                self.Image.Resampling.LANCZOS,
            )
        return self.ImageTk.PhotoImage(image, master=self.top)

    def _record(self, valid: bool) -> None:
        if not self._precomputed:
            self.status.set("Sem imagem para julgar.")
            return
        plain_bgr, cnn_bgr = self._precomputed[0]
        x = _l_channel(plain_bgr)
        y = _l_channel(cnn_bgr) if valid else x.copy()
        self.pairs.append((x, y))
        manual_stars = self.stars_var.get()
        auto_plain, auto_cnn = self._auto_stars
        self.status.set(
            f"Par guardado ({'Válido' if valid else 'Rejeitado'}) — "
            f"auto {auto_plain:.1f}★→{auto_cnn:.1f}★ · manual {manual_stars:.1f}★ — "
            f"{len(self.pairs)} pares acumulados."
        )
        if valid:
            self.state.pairs_positive += 1
        else:
            self.state.pairs_identity += 1
        self.state.save()
        self._next()

    def _accept(self) -> None:
        self._record(True)

    def _reject(self) -> None:
        self._record(False)

    def _prev(self) -> None:
        if self.samples:
            self.index = (self.index - 1) % len(self.samples)
            self._load_sample()

    def _next(self) -> None:
        if self.samples:
            self.index = (self.index + 1) % len(self.samples)
            self._load_sample()

    def _train_now(self) -> None:
        if not self.samples:
            return
        if len(self.pairs) < 2:
            self.status.set("Precisas de pelo menos 2 pares (Válido/Rejeitado) antes de treinar.")
            return
        self.status.set("A treinar a CNN residual…")
        results: queue.Queue[str] = queue.Queue()

        def work() -> None:
            try:
                db = FeedbackDB()
                result = train_enhancer_round(
                    self.pairs,
                    self.state,
                    self.state.round + 1,
                    epochs=40,
                    champion_path=self._champion_path,
                    db=db,
                )
                if result is not None and not result["skipped"]:
                    if result["champion_path"] is not None:
                        self._champion_path = result["champion_path"]
                    text = (
                        f"Treino concluído: qualidade {100 * result['mean_delta']:.1f}% · "
                        f"{'PROMOVIDA (novo campeão)' if result['promoted'] else 'mantém o campeão'}."
                    )
                else:
                    text = "Treino concluído: sem pares suficientes."
            except Exception as exc:
                text = f"Erro no treino: {exc}"
            results.put(text)

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        self._workers.append(thread)
        self._train_results = results
        self.top.after(50, self._poll_train)

    def wait_idle(self, timeout: float = 30.0) -> None:
        """Aguarda as threads de trabalho em curso (encerramento limpo).

        Chamado pelo `run_gui` ao sair do `mainloop` (ex.: nos testes, o
        mainloop fecha após ~60 ms e a carga da amostra pode ainda estar a
        correr); espera apenas pelas threads ativas, sem bloquear para sempre.
        """
        deadline = time.monotonic() + timeout
        for thread in list(self._workers):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(timeout=remaining)

    def _poll_train(self) -> None:
        try:
            text = self._train_results.get_nowait()
        except queue.Empty:
            self.top.after(50, self._poll_train)
            return
        self.status.set(text)


if __name__ == "__main__":
    sys.exit(main())
