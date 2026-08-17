# AstroFrame

Geometric stabilization and automatic enhancement of solar and lunar eclipse photos and videos.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue) ![License MIT](https://img.shields.io/badge/license-MIT-green)

## Features

- **Geometric stabilization** — detects the Sun/Moon disk (`cv2.HoughCircles` + contour fallback + intensity-centroid refinement) and re-aligns each frame to keep the eclipse always at the exact center, without black borders.
- **Automatic enhancement** — CLAHE in the LAB space (without blowing out the brightness), Non-Local Means denoising (useful for high ISO) and unsharp masking to highlight the Moon's limb.
- **Lucky imaging** — discards blurred frames by Laplacian variance, with a threshold estimated statistically from the video itself.
- **Stacking** — combination (median or mean) of the N best frames, aligned by centering, to reduce noise.
- **Temporal anti-jitter** — centroid smoothing (EMA) and reuse of the last valid displacement when a frame has no detection.
- **Multiple disk detection** — besides the main disk, the **reflections** are detected (Hough + contours); the polishing removes the reflections and the live video shows them in red.
- **Polishing and automatic rating** — `polish_image()` adds brightness to the disk while keeping the corona; `score_image()` assigns **stars (0–5)** to the result (noise, contrast, size and corona color).
- **Example-based calibration** — native desktop interface (`python calibrate.py`) that loads the photos and videos from `samples/`, lets you **draw circles/ellipses by hand** (click creates, drag moves, handles resize) in a 1st pass, turn on the **automatic detection** in the 2nd to fill/validate the remaining samples, and compares everything against the ground truth on all samples (recall, precision, IoU, errors + parameter suggestions).
- **Detection validation and training** — `validator.py` (native desktop window) walks the samples, shows the detection with zoom/pan, and learns by **rewarding and punishing 7 detector parameters** shape by shape against the manual guide; **automatic training** (`--auto`) re-detects in series until 100% and exports the **trained weights** for the real system.
- **Feedback learning** — every run is stored in SQLite; besides the automatic rating, you can rate manually (0–5 stars) and AstroFrame **adjusts the sliders automatically** on the next run with the same camera profile, showing the history/log in the interface itself.
- **Gradio interface** — three tabs: **Image** (Before/After, sliders, corona/limb zoom), **Video** (live processing with detected disks, final preview at spaced frames and optional export) and **Auto-tune** (optimizes the parameters against the samples). When loading a video, the **metadata** is read (ffprobe/OpenCV/EXIF) and the **parameters are suggested automatically** (ISO → denoising, resolution → detector radii, bitrate → compression), remaining editable.
- **Auto-tuning** — `astroframe autotune` (or the *Auto-tune* tab) searches all detection/enhancement parameters against the calibration samples (`samples/` + `calibration.json`): proxy evaluation at ~480p (mean IoU between detected and expected disks, with penalties for extra/missing disks) plus deterministic bounded hill climbing with time budget and annealing. The result is registered in the learning database and applied automatically to the next runs of the same profile.
- **Small neural networks (pure NumPy)** — optional LSTM and CNN models: the LSTM learns from the rating history (predicting parameter deltas and the disk trajectory for the anti-jitter) and the CNN learns to remove noise/smearing (residual enhancer) and to score each detection (disk/noise classifier that filters false positives).
- **CLI** — photo batch, videos (stabilize/enhance/stack), auto-tuning, logs and progress bar.

> **AI at a glance (v0.7.0)** — the entire AI layer is **off by default**
> (`[tuning]` and `[ai]`) and degrades silently: the auto-tuning needs a
> folder of samples with ground truth, and the LSTM/CNN models (pure NumPy
> core, PyTorch optional) live in `~/.astroframe/*.npz` — when a model is
> missing or corrupt, the pipeline simply runs without it.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Simple alternative: `pip install -r requirements.txt`. Requires Python 3.10+.

> [!IMPORTANT]
> On recent Debian/Ubuntu, `pip install` on the system Python fails with
> `error: externally-managed-environment` (PEP 668). Always use the virtualenv
> above (`source .venv/bin/activate` before installing), or force with
> `--break-system-packages` at your own risk.

> [!TIP]
> For rich video metadata (codec, bitrate, duration) install the system
> `ffmpeg` — without it, AstroFrame uses only OpenCV (resolution/fps/frames).

## Quick usage

```bash
python main.py                             # web interface (Gradio) — frontend + backend together
python calibrate.py                        # calibration interface (samples/)
python validator.py                        # detection validation/training (samples/)
astroframe serve                          # equivalent via the installed CLI
astroframe process --input photo1.jpg photo2.jpg --output-dir outputs/
astroframe video --input eclipse.mp4 --mode enhance
astroframe video --input eclipse.mp4 --mode stack --stack-n 20
astroframe video --input eclipse.mp4 --mode enhance --fast   # no denoise (faster)
astroframe autotune --samples samples --budget 60            # auto-tunes the parameters (AI)
astroframe config-template                # generates an editable config.yaml
```

`main.py` is the single entry point: it starts the Gradio server that serves
the frontend in the browser and processes the images in the backend (the engine
in `core/` runs in the same process, at each click on **Process**). Options:
`--config`, `--host`, `--port`, `--share` and `--no-browser`.

`validator.py` is the **detection validation/training** (native desktop window;
`--check` for a report without the interface, `--auto` for automatic training):
it compares the detection with the manual guide in `calibration.json`,
**rewards/punishes the parameters** per shape and ends with a report + trained
weights exportable to the real system.

`calibrate.py` is the **calibration interface**: it loads images and video
frames from `samples/`, lets you adjust the circles by hand (drag = move,
brush = add, eraser = remove) and **Validate all** compares the automatic
detection with the ground truth on all samples.

## Documentation

- [docs/EN/API.md](docs/EN/API.md) — reference of the modules `core/`, `video/`, `meta/`, `ai/`, `calibration/` and `config.py` (EN).
- [docs/EN/Usage.md](docs/EN/Usage.md) — practical guide: CLI, YAML configuration field by field, interface, calibration and video workflow (EN).
- [docs/EN/Architecture.md](docs/EN/Architecture.md) — original solution specification (reference, EN).
- [docs/EN/CHANGELOG.md](docs/EN/CHANGELOG.md) — changelog (EN).
- [docs/PT/](docs/PT/) — the same documentation in Portuguese (API, Arquitetura, USO).
- [docs/FR/](docs/FR/) — la même documentation en français (API, Architecture, Usage, CHANGELOG).
- [README.md](README.md) / [README-FR.md](README-FR.md) — this README in Portuguese and French.

## Known limitations

- The exported video **does not contain audio** (`cv2.VideoWriter`); to preserve the sound, merge the original track with ffmpeg:
  `ffmpeg -i original.mp4 -i processed.mp4 -c copy -map 0:a -map 1:v output.mp4`
- RIFE interpolation is optional and requires PyTorch (`pip install -e ".[rife]"`); the model interface varies between versions of the RIFE repositories.
- The denoising is the slowest step (~1 s/frame at 480p); use `--fast` on large videos.

## Development

```bash
pytest                      # 558 tests (pixel tests with synthetic images)
pytest --cov=astroframe     # coverage (100% of the package)
ruff check .                # lint
ruff format .               # formatting
```

CI (GitHub Actions): pytest on Python 3.10/3.12 + ruff, in `.github/workflows/ci.yml`.

## Structure

```
src/astroframe/
├── core/         geometric stabilizer, automatic enhancement and pipeline
├── video/        frame reading, lucky imaging and stacking
├── meta/         metadata reading (ffprobe/OpenCV/EXIF) and parameter suggestions
├── calibration/  example scanning, ground truth and detection validation
├── ui/           Gradio interface (Image/Video/Auto-tune + Calibration) and CLI
└── ai/           auto-tuning, LSTM/CNN learning, feedback, rating and optional RIFE
```

## License

MIT — see [LICENSE](LICENSE).