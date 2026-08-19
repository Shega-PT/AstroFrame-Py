# Changelog

All notable changes to AstroFrame will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versioning [SemVer](https://semver.org/).

## [0.9.0] - 2026-08-19

### Added

- **RIFE interpolation wired into the CLI** — `astroframe video --interp N`
  generates N intermediate frames per frame pair (RIFE via `torch.hub`,
  loaded lazily) and exports the video `(N+1)×` smoother (`fps × (N+1)`);
  if the model fails to load, the CLI warns and continues without
  interpolation.
- **PyTorch and Pillow become mandatory dependencies** — the optional
  `astroframe[rife]` extra is removed; `torch>=2.0` and `Pillow>=10.0` are
  now in `pyproject.toml` and `requirements.txt`.

### Fixed

- **GitHub Actions failures** — Tk tests now run under `xvfb-run -a`;
  `ruff==0.16.2` pinned (the `ruff format --check` failed due to version
  drift); `actions/checkout@v7` and `actions/setup-python@v7` (Node 24,
  removes the Node 20 deprecation warning); the test job installs the CPU
  wheel of PyTorch before the package.
- **Flaky Tk tests under load** — `ImageTk.PhotoImage` objects are now
  created with an explicit `master` (image always in the same Tcl
  interpreter as the canvas) and the trainer tests drain the results queue
  directly instead of relying on `after` timers (2 clean full runs).
- **No-torch tests** — the RIFE wrapper tests simulate the absence of torch
  via `__import__` (still valid with torch installed).

### Removed

- Phantom `ai.backend` field (there was no `torch` implementation of the
  core — `LSTMCellTorch`/`SmallCNNTorch` never existed); the documentation
  (READMEs, PT/EN/FR docs, `torch_available()`) now describes PyTorch as a
  mandatory dependency used only by RIFE.

## [0.8.0] - 2026-08-18

### Added

- **Automatic training with detection CNN** (`validator.py`) — in automatic
  validation (`--auto` with `--cnn`), disk patches are collected every series
  and the `DiskFilter` CNN is retrained between series:
  - `cnn_positives` (guide circles) and `cnn_negatives` (rejected shapes +
    deterministic random crops excluded by IoU) collected per round;
  - the next series judges with the new model (`--cnn-threshold`); the result
    is compared with the **champion** stored in the database — if strictly
    better it is promoted (`disk_filter.npz` updated) and the next series
    warm-starts from the champion's weights;
  - auto-training window with **Train CNN** checkbox and a CNN section in the
    final report; `STATE_VERSION=2` (`cnn_series`, `cnn_positives`,
    `cnn_negatives`) with v1 state-file compatibility;
  - new CLI flags: `--epochs`, `--cnn-off`, `--cnn-threshold`.
- **Residual CNN trainer** (`enhancer_trainer.py`) — standalone tool to train
  and validate the image enhancement:
  - side-by-side GUI (no-CNN vs with-CNN) with **Valid/Rejected** judgement
    (Valid stores input→output pairs; Rejected stores input→input);
  - **Train now** trains the residual with the accumulated pairs (warm-start
    from the champion), compares by `mean_delta` and promotes if better
    (`~/.astroframe/enhancer_cnn.npz`);
  - CLI with `--check`, `--auto` (synthetic-degradation series),
    `--samples/--epochs/--seed/--export/--state/--reset-state/--config`.
- **Wider learning database** (`astroframe/ai/feedback.py`) — new `logs`
  table (per-component event history) and `models` table (NN artifacts with
  metric, champion and series history).

### Security

- Training AI stays off by default (`--cnn-off` in automatic series,
  `ai.disk_filter`/`ai.cnn_enhance` still disabled by default); missing or
  corrupted models degrade silently and the judgement never empties the
  detected list.

### Tests

- **649 tests, 100% coverage** of `src/astroframe/`, `validator.py` and
  `enhancer_trainer.py` — new suites `tests/test_enhancer_trainer.py` (34)
  and `tests/test_enhancer_trainer_ui.py` (10), extended
  `tests/test_validator.py` (69), `tests/test_validator_ui.py` (44) and
  `tests/test_feedback.py` (32); the `_ai_isolado` conftest fixture isolates
  DB, models and canonical paths per test; `ruff check` clean.

### Documentation

- PT/EN/FR docs updated: automatic validation with CNN, residual CNN
  trainer, `logs`/`models` tables and champion logic.

### Data infrastructure (`Logs/`)

- New folder layout at the repository root, replacing the old paths under
  `~/.astroframe/` and `samples/`:
  - `Logs/weights/` — canonical models (`disk_filter.npz`,
    `enhancer_cnn.npz`, `lstm.npz`) with `Logs/weights/staging/` for the
    round candidates;
  - `Logs/train/` — training artifacts by default: `calibration.json`
    (global ground truth, falling back to `samples/calibration.json`),
    `validator_state.json`, `enhancer_state.json`, `trained_config.json`;
  - `Logs/logs/ia/` — network round reports (`disk_filter_round_N.json`,
    `enhancer_round_N.json`);
  - `Logs/logs/system/` — rotating system logs (1 MiB × 3) and
    `feedback.db`;
  - new module `src/astroframe/paths.py` (accessors + `migrate_legacy()`,
    which copies legacy artifacts from `~/.astroframe/` and
    `samples/calibration.json` once, without deleting the source; the
    `ASTROFRAME_DATA_DIR` env var redirects the root) and file logging in
    every entry point (`main`, `calibrate`, `validator`,
    `enhancer_trainer`, CLI); `.gitignore` now ignores `Logs/**` with
    `.gitkeep` exceptions.

### Documentation (astro-centric)

- Documentation and code strings rewritten to be **astro-centric**: the
  protagonists are celestial bodies (Sun, Moon, planets, comets, stars) in
  astrophotographs and astrovideos; phenomena (eclipses, transits,
  occultations) now appear only as contextual examples; "eclipse companion"
  → "secondary disk"; `pyproject.toml` description/keywords, PT/EN/FR
  READMEs, `docs/PT|EN|FR/*` and `samples/README.md` updated.

## [0.7.0] - 2026-08-17

### Added

- **Auto-tuning** (`astroframe/ai/tuner.py` + `astroframe/ai/params.py`):
  - unified **registry of tunable parameters** (`ai.params`) — the single
    source of truth for the safe ranges, optimization steps, dtype, odd
    parity (Gaussian kernels), groups (detect/geometry/enhance/stack/
    polish/score/meta), evaluation cost and validator reward/punish deltas;
    **every learned value passes through the registry clamp**;
  - **proxy evaluation** (`ProxyEval`): the pipeline runs on the calibration
    samples (`samples/` + `calibration.json` ground truth) at ~480p
    (`work_scale` 0.5, never upscaled), measuring the **mean IoU** between
    detected (Hough) and expected disks with **penalties for extra/missing
    disks**, plus star ratings of enhanced frames; cached by effective
    parameters;
  - **bounded hill climbing** (`BoundedHillClimb`): deterministic (fixed
    seed), time-budgeted search (`budget_s`, default 60 s) with momentum
    (step ×1.5 after two consecutive accepts, halved on failures, min
    step/8), optional **annealing** (worse candidates accepted with
    probability exp(−Δ/T), T decaying per pass) and patience; costly
    parameters (denoising) tried only on even passes; can be **pre-seeded
    with LSTM predictions** (`_lstm_seed`) when they improve the proxy;
  - `run_autotune` orchestrates the optimization, exports the tuned
    configuration (`export_trained_config`, default
    `samples/trained_config.json`) and **registers the result in the
    feedback DB** (`tuning` table) — applied automatically to the next runs
    of the same profile via `apply_learned`.
- **CLI `astroframe autotune`** — `--samples DIR`, `--budget N`, `--seed N`,
  `--no-anneal`, `--params p1,p2`, `--profile NAME`, `--export FILE`,
  `--config FILE` and `--reset` (clears the profile's tuning history).
- **"Auto-tune" tab in the Gradio interface** — samples folder, time budget
  (seconds), parameter subset (empty = all), annealing and DB registration
  toggles; shows the progress, the report (parameter · base → adjusted ·
  delta) and the optimized configuration as JSON; button to clear the
  tuning history.
- **LSTM (`astroframe/ai/lstm.py`, pure NumPy)** — one-layer LSTM cell with
  hand-written forward/backward (backprop-through-time, vectorized, no new
  dependencies; `torch_available()` reports the optional PyTorch):
  - `LSTMTuner` — trains **offline** on the feedback history (star ratings +
    metrics, sliding windows, validation and early stop) and predicts the
    **parameter deltas** for the next run; used as the auto-tuning pre-seed;
  - `TrajectoryPredictor` — predicts the **next disk centroid** (linear
    regression as the base + optional LSTM refinement, cell 2→8, trained on
    synthetic trajectories by `train_trajectory_model`); with
    `ai.lstm_trajectory` the anti-jitter **predicts** the centroid instead
    of freezing it in frames without detection;
  - models saved as **versioned `.npz`** (`~/.astroframe/lstm.npz`); a
    corrupt or wrong-version file falls back silently.
- **CNN (`astroframe/ai/cnn.py`, pure NumPy)** — small convolutional network
  (2× conv 3×3 + ReLU + pooling + MLP head), gradients verified by **finite
  differences**, offline deterministic training (fixed seed):
  - `fit_residual` / `ResidualEnhancer` — learns to remove noise/smearing;
    applied **after the unsharp step** of `enhance_image` (L channel of LAB,
    64×64 tiles with overlap) when `ai.cnn_enhance`;
    `~/.astroframe/enhancer_cnn.npz`;
  - `fit_classifier` / `DiskFilter` — disk vs. noise classifier scoring each
    detection (`confidence`); filters false positives when
    `ai.disk_filter > 0.0` and **never empties** the detected list;
    `~/.astroframe/disk_filter.npz`.
- **New configuration sections** — `[tuning]` (`enabled=false`,
  `budget_s=60.0`, `seed=42`, `anneal=true`, `proxy_scale=0.5`,
  `frames_per_sample=3`, `detection_weight=0.6`, `params=null`) and `[ai]`
  (`backend=numpy`, `lstm_trajectory=false`, `cnn_enhance=false`,
  `disk_filter=0.0`).
- **Feedback integration** — `apply_learned` now also sums the auto-tuning
  deltas (`tuning` table) over the star-rating nudges, always clamped via
  the registry: the AI "memory" across runs (with nothing learned the
  configuration returns unchanged).
- **Security**: all AI is **off by default** (`tuning.enabled=false`,
  `ai.*`); a missing or corrupt model degrades silently and never blocks the
  pipeline.

### Tests

- New test modules `tests/test_params.py`, `tests/test_tuner.py`,
  `tests/test_lstm.py`, `tests/test_cnn.py` and `tests/test_ai_coverage.py`
  (proxy with synthetic samples, deterministic hill climbing, finite
  differences on the CNN gradients, silent fallbacks); suite expanded with
  100% package coverage.

### Documentation

- `docs/EN/API.md` — new sections `ai.params`, `ai.tuner`, `ai.lstm` and
  `ai.cnn`; updated `ai.feedback` (tuning table, `apply_learned`),
  `ui.gradio_app` (Auto-tune tab), `ui.cli` (autotune) and `core` hooks.
- `docs/EN/Architecture.md` — new section *4 AI Layer* (registry, proxy +
  hill climbing, LSTM, CNN, feedback integration, security model).
- `docs/EN/Usage.md` — Auto-tuning section (CLI + how it works), Auto-tune
  tab, `[tuning]`/`[ai]` configuration tables and security note.
- `README-EN.md` — new features and the AI-at-a-glance paragraph.

## [0.6.0] - 2026-08-14

### Added

- **Detection validation and training** (`validator.py` at the root) — a
  native desktop interface (tkinter) that walks the samples in `samples/` one
  by one, shows the detection (main + companions) over the image, and lets
  you **accept/reject** each shape against the manual guide
  (`calibration.json`):
  - trainable per-shape weights: **7 parameters** (`param2`, `param1`, `dp`,
    `gaussian_kernel_size`, `gaussian_sigma`, `occluded_ratio`,
    `occluded_ring`) with reward/punish deltas, bounds and history;
  - **automatic training** (`--auto`): re-detection series with self-evaluation
    against the guide (configurable minimum IoU), reward/punish per detected
    or missed shape, doubled punishment for stubborn rejections, until 100%
    of the material is processed;
  - **final report** with score, trained weights and ⓘ tooltips per parameter
    + a **Save** button exporting the trained configuration to
    `trained_config.json` (applicable to the real system);
  - **on-detect preview** — the detection is drawn over the image in real time
    before you are asked to judge;
  - **persistent state** in `validator_state.json` (rounds, series, weight and
    delta history) with `--reset-state` to start over;
  - `--check` mode (report without interface) and an interface with minimum
    IoU sliders, zoom/pan and disk drawing.
- **Desktop calibration editor** (`src/astroframe/ui/calibration_tk.py`) — a
  native window (tkinter) in `calibrate.py` (default) replacing the browser
  editor:
  - click creates a circle/ellipse, drag moves the center, **handles** adjust
    the horizontal/vertical radii, Radius X/Radius Y sliders in real time,
    wheel = zoom, right button = pan, Delete/arrows to remove and nudge
    (Shift = 10);
  - **two passes**: 1st manual (detection off) → ground truth; 2nd with
    **automatic detection on load** to fill/validate;
  - `param2`/max radius sliders re-run the detection on release;
  - **ellipses** supported in the ground truth (`ry` in the JSON) and in
    validation (mask-based IoU + geometric radius for the errors);
  - `calibrate.py --ui gradio` keeps the old browser editor.
- **Detection without explicit `min_radius`/`min_dist`** — radii are now
  derived automatically from the image (resolution, main diameter) and the
  minimum distance is inferred from the detection; calibration suggests only
  the parameters that still exist (`param2`/`param1`).
- **100% test coverage** on all code (`validator.py` + `src/astroframe/`,
  ~435 tests): interface tests with real Tk (hidden window), deterministic
  threads via `monkeypatch`, and infrastructure that works around the
  Python 3.12 GC abort during thread bootstrap with coverage active
  (`gc.disable()` + safe collection on the main thread).

### Fixed

- `RuntimeError: main thread is not in main loop` in automatic training — the
  IoU slider value was read inside the worker thread; it is now captured on
  the main thread before starting the series.
- Intermittent suite abort (`Fatal Python error: Aborted`) when running
  coverage on tests with threads + Tk (GC collecting during a worker thread's
  bootstrap) — cyclic GC disabled for the test session.
- Uninitialized `_pan_start` in the validation editor (drag without a prior
  click raised `AttributeError`).

## [0.5.0] - 2026-08-13

### Added

- **Example-based calibration** (new package `astroframe/calibration/`):
  - `scan_samples` scans the samples folder (`samples/` by default)
    recursively — images (jpg/png/bmp/tif/webp) enter as-is and videos
    (mp4/avi/mov/mkv/m4v) contribute **8 equidistant and deterministic frames**
    (reproducible in validation).
  - `CalibrationStore` stores the **manual ground truth** in
    `samples/calibration.json` (JSON v1, key = relative path + `#frame`).
  - `circles_to_layers` / `layers_to_circles` convert circles into **RGBA
    layers** of the `gr.ImageEditor` (drag = move, brush = add, eraser =
    remove; one circle per connected component).
  - `validate_all` compares the automatic detection (`find_all_disks`) with
    the ground truth on all samples: greedy matching by IoU (≥0.5),
    recall/precision, mean IoU, center errors (px) and signed radius errors
    (%), calibration score 0–100 (recall 0.4 · precision 0.3 · IoU 0.3) and
    **parameter suggestions** (e.g. lower `min_radius` if small disks fail,
    raise `param2` with false detections).
- **Calibration interface** (`astroframe/ui/calibration_app.py`): samples
  dropdown + circle editor + buttons "Automatic detection", "Save
  adjustments" and "Validate all samples" (per-sample table + global summary +
  suggestions).
- **Entry points**: `calibrate.py` at the root (mirror of `main.py`, with
  `--samples/--config/--host/--port/--share/--no-browser`) and the
  `astroframe calibrate` CLI subcommand.
- `FrameReader.frame_at(index)` — direct frame reading by index
  (`CAP_PROP_POS_FRAMES`).
- `samples/README.md` rewritten with the recommended structure
  (images/videos, subfolders by subject: eclipse, moon, sun, planets).

### Documentation (multilingual)

- The documentation became **trilingual**:
  - `docs/PT/` — `API.md`, `Arquitetura.md`, `USO.md` (moved, with the
    calibration section); the root `CHANGELOG.md` remains canonical (PT).
  - `docs/EN/` — `API.md`, `Architecture.md`, `Usage.md`, `CHANGELOG.md`
    translated into English.
  - `docs/FR/` — `API.md`, `Architecture.md`, `Usage.md`, `CHANGELOG.md`
    translated into French.
  - `README-EN.md` and `README-FR.md` at the root (README translations); the
    `README.md` now points to `docs/PT/` and the EN/FR versions.

### Tests

- Suite expanded to **~260 tests with 100% package coverage**
  (`tests/test_calibration_{scan,store,circles,validate,app}.py` +
  `astroframe calibrate` in the CLI + `frame_at`).

## [0.4.0] - 2026-08-13

### Added

- **Eclipse companions (e.g. the Moon entering the Sun)**: `find_all_disks`
  now runs a **second Hough pass with reduced `minDist`** (1/4 of normal) to
  find circles inside the largest body, which the normal pass would discard.
  The interface draws them in **yellow** (largest body in green, lens
  reflections in red), both in the Image and Video tabs.
- **Ghost-circle area filter** (`_is_occluded_artifact`): a circle almost
  fully contained in the largest body is discarded when the contrast with the
  surrounding ring is weak (Sun+Moon edges detected as a single circle);
  compares the **area** overlap (not only the center), resistant to the
  centroid refinement.
- **Per-body polishing** (`core/polish.py` rewritten): each body gets its own
  enhancement (local contrast stretching + brightness, with dark and uniform
  silhouettes — e.g. Moon in eclipse — preserved intact) and the image is
  **recomposed seamlessly** by mask blending with feathering (overlaps =
  smooth average of the enhancements). The cut line (`corona_scale`) feathers
  the ring into the background.
- **Background = mean of the original background** (`polish.background_fill`,
  now by default) instead of pure black; `polish.black_background` opts for
  black again and `polish.brightness` controls the extra brightness of the
  bodies.
- **Disk cap**: `find_all_disks` returns at most 5 disks (`_MAX_DISKS`).
- `find_all_disks` accepts grayscale images `(H, W)`.

### Fixed

- Reflections drawn in red inside the largest body (the
  main/companion/reflection split is now by center relative to the radius of
  the largest body).
- Polishing erasing eclipse companions: only circles with the center
  **outside** the largest body are removed as reflections.

### Documentation

- `docs/PT/USO.md` and `docs/PT/API.md` updated for the per-body polishing,
  the new `PolishConfig` and the two-pass detection; suite with **221 tests
  and 100% coverage**.

## [0.3.0] - 2026-08-13

### Added

- **Multiple disk detection** (`find_all_disks` in `core/stabilizer.py`):
  instead of only the main one, the main disk and its **reflections** are
  detected (Hough + contours, with duplicate fusion and preservation of the
  brightest at each center). The stabilizer keeps using the main one and
  keeps the last detection in frames without a disk (`last_detection`).
- **Polishing** (`core/polish.py`): `polish_image()` applies brightness to
  the main disk (keeping the corona blurred), removes reflections, and is
  used in the preview/final frame and in the exported video.
- **Automatic rating** (`ai/score.py`): `score_image()` computes stars
  (0–5) from noise, contrast, disk size and corona color; the interface
  shows the result in "Automatic rating" (image **and** video).
- **Feedback learning base** (`ai/feedback.py`): every run is recorded
  (camera profile + parameters + metrics + rating); the user can rate
  manually (0–5 stars) and the system **adjusts the sliders
  automatically** in the next runs (milder with good ratings, stronger with
  bad ones; extra denoise for noise, brightness for a weak corona, etc.).
  Learning log with history and reasons in SQLite (`ASTROFRAME_FEEDBACK_DB`
  variable for location).
- **Videos without a disk**: the pipeline stabilization/preview skips the
  polishing and the rating works without detection (previously it failed).
- Suite expanded to **205 tests with 100% package coverage**.

### Fixed

- The polishing **erased a circle at the center of the image**: inner
  near-concentric circles with the main disk (e.g. the Moon silhouette inside
  the Sun) were detected as "reflections" and removed — `polish_image` now
  only removes reflections whose **center is outside the main disk**, and
  `find_all_disks` fuses concentric circles (tolerance of 12% of the radius),
  avoiding duplicates of the same edge in both directions (polishing and
  live drawing).

### Documentation

- `docs/PT/USO.md`: automatic/manual rating, learning log and rewritten video
  section; `docs/PT/API.md` with `find_all_disks`, `polish_image`,
  `score_image` and the new `ai/` package.

## [0.2.0] - 2026-08-12

### Added

- **New live video interface** ("Video" tab): the left panel shows the video
  in real time as it is processed, with the circle (bounding box) of the
  detected disk; the right one updates the final result at spaced frames
  (stabilized + CLAHE + denoise + sharpening). Optional export of the
  processed video (.mp4, no audio). `_best_frame_from_video` was replaced by
  this complete flow.
- **Metadata reading** (new `meta/` package, own MIT implementation):
  video via the ffprobe → OpenCV cascade (codec, bitrate, duration, fps,
  resolution) and image via PIL/EXIF (ISO, exposure, aperture, focal length,
  camera, date); no new pip dependencies.
- **Automatic parameter suggestions** (`meta/suggest.py`): stabilizer radii
  proportional to resolution, `denoise.h` scaled by ISO, denoise reduction in
  heavily compressed bitrate videos; applied to the sliders on video load
  (remain editable).
- "Ratio/quality" panel in the interface (resolution, aspect ratio, fps,
  codec, bitrate, ISO, exposure, camera) + `gr.JSON` with the raw metadata.
- Interface reorganized in tabs ("Image" / "Video"); image processing moved
  to `process_image_input()` (testable module function).
- Half-resolution detection covered and suite expanded to **131 tests with
  100% package coverage** (including RIFE without PyTorch, via a fake
  `torch` module in tests).

## [0.1.2] - 2026-08-12

### Added

- Gradio interface accepts videos (`.mp4/.avi/.mov`): the sharpest frame of
  the video is selected automatically (lucky imaging) and processed as an
  image (`_best_frame_from_video` in `ui.gradio_app`).
- Tests for the sharpest-frame selection from a synthetic video.

## [0.1.1] - 2026-08-12

### Added

- `main.py` at the root: single entry point that starts the frontend
  (Gradio) and the backend (pipeline) together, opening the browser
  automatically (`python main.py [--config|--host|--port|--share|--no-browser]`).
- `inbrowser` parameter in `ui.gradio_app.run()` (opens the browser by
  default).

### Documentation

- README: installation section with a warning about PEP 668 (Debian/Ubuntu)
  and `python main.py` as the first quick-usage command.
- `docs/PT/USO.md`: web interface documented with `python main.py`.
- `docs/PT/API.md`: new `run()` signature with `inbrowser`.
- `.gitignore`: generic patterns for videos (`*.mp4`, `*.MP4`, `*.MOV`, `*.mkv`).

## [0.1.0] - 2026-08-12

### Added

- Complete pipeline: geometric stabilization (HoughCircles + contours),
  automatic enhancement (CLAHE/denoise/unsharp) and orchestration (`core/`).
- Video: frame-by-frame reading, lucky imaging with statistical threshold and
  stacking (`video/`).
- Interfaces: Gradio (Before/After, sliders, zoom) and CLI
  (`astroframe serve|process|video|config-template`).
- External configuration via YAML (`astroframe config-template`), with
  validation and warnings.
- Temporal stabilization (centroid EMA) with reuse of the last displacement
  in frames without detection.
- Automatic cropping after translation (no black borders, without cropping
  the disk) and detection radii relative to the frame resolution.
- Half-resolution detection on large frames (≥1200 px).
- `--fast` mode (omits denoising) for videos.
- Optional RIFE interpolation (`astroframe[rife]`), with lazy import.
- MIT license, GitHub Actions CI (pytest 3.10/3.12 + ruff) and 43 tests.

### Fixed

- Swapped RGB/BGR channels in the Gradio interface (colors now correct).
- CLAHE crash on images smaller than the tile grid.
- Stacking without frame alignment (now centers before stacking).
- Photo batch aborting on the first failure (now continues and summarizes the
  result).
- Invalid keys/types in `config.yaml` being silently accepted (now warn).

### Known

- The exported video does not include audio (use ffmpeg to merge the track).
