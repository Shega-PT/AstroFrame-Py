# AstroFrame User Guide

Practical guide to install, configure and run AstroFrame. The solution
specification is in [Architecture.md](Architecture.md) and the code reference
in [API.md](API.md).

## Table of contents

1. [Installation](#installation)
2. [Web interface (Gradio)](#web-interface-gradio)
3. [Calibration](#calibration)
4. [Detection validation and training](#detection-validation-and-training)
5. [Command line](#command-line)
6. [Configuration (config.yaml)](#configuration-configyaml)
7. [Video workflow](#video-workflow)
8. [Limitations and notes](#limitations-and-notes)

---

## Installation

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Without a virtual environment (`pip install -r requirements.txt` works, but on
Debian/Ubuntu 24+ the venv is mandatory — PEP 668).

## Web interface (Gradio)

The simplest entry point is `main.py` at the repository root: it starts the
frontend (Gradio) and the backend (processing engine in `core/`) in the same
process and opens the browser automatically.

```bash
python main.py
```

Opens at `http://127.0.0.1:7860`. To configure, change the port or get a
public link:

```bash
python main.py --config config.yaml --port 7861 --share
python main.py --no-browser        # does not open the browser (useful on servers)
```

The same server is available via the installed CLI, equivalent to:

```bash
astroframe serve                   # equivalent to python main.py
astroframe serve --config config.yaml --port 7861 --share
```

> `--share` creates a temporary public URL (via the Gradio tunnel) — do not
> use it with sensitive material.

The interface has two tabs:

### Image tab

- **Input** — load the photo/frame (arbitrary image format).
- **Stabilized** — centered disk, with the detected disks drawn:
  **green** = largest body, **yellow** = eclipse companions (e.g. the Moon
  entering the Sun), **red** = lens reflections.
- **Processed** — CLAHE + denoising + sharpening + **per-body polishing**
  (each body enhanced individually and recomposed seamlessly; background =
  mean of the original background; reflections removed).
- **Zoom** — magnification centered on the corona/limb.
- **Parameters** — CLAHE clip limit, denoising strength, sharpening, zoom and
  corona scale kept by the polishing, with initial values from `config.yaml`.
- **Automatic rating** — stars (0–5) computed from noise, contrast, disk size
  and corona color.
- **Manual rating + learning** — slide the number of stars the result deserves
  and click *Save manual rating*: the run is recorded and, next time with the
  same camera profile, the sliders **adjust themselves automatically** (mild
  correction for good ratings, strong for bad ones). The *Learning log* tab
  shows the history and the reason for each adjustment.

### Video tab

1. **Load the video** (`.mp4/.avi/.mov`). At that moment the **metadata** is
   read — ffprobe (if installed; otherwise only OpenCV: resolution/fps/frames)
   for video, EXIF (PIL) for images — and shown in the
   **Ratio / quality / suggestions** panel (resolution, aspect ratio, fps,
   codec, bitrate, ISO, exposure, camera). The **sliders are pre-filled** with
   the optimization suggestions **and with what the AI has learned** (previous
   ratings of the same profile), but remain editable.
2. **Process video** — while the pipeline runs:
   - **Left (live)** — the original frame in real time with the detected
     disks: **green** = largest body, **yellow** = eclipse companions,
     **red** = lens reflections.
   - **Right (final result)** — at well-spaced frames, the result with all
     corrections (stabilized + CLAHE + denoising + sharpening + per-body
     polishing).
   - Status bar with the current frame and progress, and **automatic rating**
     of the final result.
3. **Optional export** — check *"Export processed video (.mp4, no audio)"* to
   write the full video at the end (same pass, without skipping frames).
4. **Manual rating + learning** — as in the Image tab, rate the video result;
   the adjustment applies to the next loads of the same type of video and
   appears in the learning log.

### Learning base (where it is stored)

The runs (parameters used, metrics and ratings) are stored in a SQLite file at
`~/.astroframe/feedback.db`. You can change the location with the environment
variable `ASTROFRAME_FEEDBACK_DB` (for example, to share the learning between
several machines).

## Calibration

AstroFrame includes a **dedicated calibration interface**: it loads the photos
and videos from the [samples/](../../samples/README.md) folder and lets you
**draw the bodies by hand** and validate the automatic detection against the
ground truth on **all** samples.

```bash
python calibrate.py                          # native desktop window (tkinter)
python calibrate.py --ui gradio              # browser interface
python calibrate.py --samples samples
astroframe calibrate --samples samples       # equivalent (installed CLI)
```

### What enters the calibration

- **Images** (jpg/png/bmp/tif/webp) — each one is an item.
- **Videos** (mp4/avi/mov/mkv/m4v) — each contributes 8 frames sampled
  equidistantly and deterministically (reproducible in validation).
- The folder is scanned recursively: organize it by subject as you wish
  (eclipse, moon, sun, planets — subfolders in `samples/images/` and
  `samples/videos/`).

### Workflow

The flow works in **two passes**:

1. **1st pass — manual (detection off by default):**
   1. **Choose the sample** — the panel list shows all items
      (`IMG …` for images, `VID … #frame` for video frames).
   2. **Draw the bodies** — on the canvas:
      - **click empty space** → creates a circle (or ellipse, per the
        selector) at that point;
      - **drag the interior** of the selected shape → moves the center;
      - **drag the right handle** → adjusts the horizontal radius; **top
        handle** → vertical radius (ellipse); the Radius X/Radius Y sliders do
        the same fine-tuning in real time;
      - **mouse wheel** → zoom on the cursor; drag with the right/middle
        button → pan; **Delete** removes the selected shape, arrow keys move
        it 1 px (Shift = 10 px).
   3. **Save (Ctrl+S)** — writes the item's ground truth to
      `samples/calibration.json` (local file, ignored by git).
2. **2nd pass — validation (turn on "Automatic detection on load"):**
   4. **Samples without ground truth** are filled in automatically by the
      detection; saved ones open exactly as you left them. **Adjust** what is
      needed (same gestures) and save again.
   5. **Validate all samples** — runs the automatic detection on everything
      and compares it with the manual ground truth: per sample and globally it
      returns recall, precision, mean IoU, center error (px) and radius error
      (%), false negatives/positives, a **calibration score (0–100)** and
      **parameter suggestions** (e.g. lower `min_radius` if small disks fail,
      raise `param2` if there are false detections).
   6. The parameter sliders re-run the detection on release (with detection
      on), to fine-tune `param2`/radii without leaving the sample.

> Ellipses are stored as objects (with `ry` in the JSON); validation uses
> mask-based IoU when ellipses are present and the geometric radius for the
> errors.

### What the calibration is for

The manual circles are the "right answer" that the system compares with the
automatic detection. With a varied folder (eclipses, Moon, Sun, planets —
large and small disks, high and low contrast), the validation shows where the
detection fails and what to adjust in the `config.yaml` before processing the
real material.

## Detection validation and training

`validator.py` uses that same ground truth to **fine-tune the detection per
shape**: it walks the samples one by one, shows what `find_all_disks` found
(main disk + eclipse companions) over the image, and learns to tell correct
detections from false ones.

```bash
python validator.py                          # desktop window (tkinter)
python validator.py --check                  # report without interface
python validator.py --auto --series 3        # automatic training (3 series)
python validator.py --auto --iou 0.7         # minimum IoU with the guide
python validator.py --reset-state --check    # start over and verify
```

### How it works

1. **Manual round** — on each sample you see the detection and the manual
   guide (`calibration.json`); **Accept/Reject** says whether the shape is
   right.
   - With an **on-detect preview**: the detection is drawn over the image in
     real time before asking for the verdict.
2. **Automatic training (`--auto`)** — no window: each series re-detects the
   samples and **self-evaluates** every shape against the guide (configurable
   minimum IoU with `--iou`). Each correct shape **rewards** the parameters
   that found it; each false or missed shape is **punished** (doubled for
   stubborn rejections). The process ends with 100% of the material processed.
3. **Trainable weights (7)** — `param2`, `param1`, `dp`,
   `gaussian_kernel_size`, `gaussian_sigma`, `occluded_ratio`,
   `occluded_ring`: each has reward/punish deltas, minimum and maximum bounds
   and an application history.
4. **Final report** — detection score, trained weights with **ⓘ tooltips**
   explaining each parameter, and the **Save** button exports the trained
   configuration to `trained_config.json` (in the samples folder), ready for
   the real system.

### State

- Progress is kept in `validator_state.json` (in the samples folder by
  default): rounds, series, weight/delta history and the current minimum IoU.
- `--reset-state` wipes everything (including the history) and starts over;
  alone it opens the interface afterwards, combined with `--check`/`--auto` it
  runs without a window.
- `--state file.json` changes the state location; `--export out.json` changes
  the destination of the saved report.

## Command line

The complete subcommands (`astroframe --help`):

| Command | Description |
|---|---|
| `serve` | Starts the Gradio interface |
| `process` | Processes photos in batch (`--input a.jpg b.jpg --output-dir folder/`) |
| `video` | Processes a video (`--mode stabilize\|enhance\|stack`) |
| `config-template` | Generates `config.yaml` with default values |
| `calibrate` | Opens the calibration interface (`--samples folder/`) |

The detection validation/training is a standalone script (see [Detection
validation and training](#detection-validation-and-training)):
`python validator.py [--check|--auto|--reset-state|--iou N]`.

### Batch photos

```bash
astroframe process --input photo1.jpg photo2.jpg --output-dir outputs/ --config config.yaml
```

- Each file is processed independently: if one is corrupted, the batch
  **continues** and the summary comes at the end (failure count).
- The outputs are PNG with the `_processed.png` suffix.

### Video

```bash
astroframe video --input eclipse.mp4                                  # enhance mode (default)
astroframe video --input eclipse.mp4 --mode stabilize                 # only centers the disk
astroframe video --input eclipse.mp4 --mode stack --stack-n 20        # stacks the 20 best frames
astroframe video --input eclipse.mp4 --mode enhance --fast            # no denoising (fast)
astroframe video --input eclipse.mp4 --output output.mp4              # output file name
```

- **enhance / stabilize** — the video is **stabilized frame by frame** and
  re-exported as MP4 (`<name>_stabilized.mp4` by default) with a progress bar.
  The temporal anti-jitter smooths the centroid (EMA) and keeps the last
  displacement when a frame has no detection.
- **stack** — selects the N sharpest frames (lucky imaging), **centers each
  one** and combines them (median by default) into a single PNG.
- `--fast` omits the slowest step (denoising) and greatly reduces the
  processing time on large videos.

## Configuration (config.yaml)

Generate the template and edit only what is needed (the rest keeps the default
values):

```bash
astroframe config-template --output config.yaml
```

All fields and types:

### `clahe`
| Field | Type | Default | Description |
|---|---|---|---|
| `clip_limit` | float | `3.0` | CLAHE clip limit (higher = more contrast) |
| `tile_grid_size` | int | `8` | Grid size (automatically reduced if the image is smaller) |

### `denoise`
| Field | Type | Default | Description |
|---|---|---|---|
| `h` | float | `5.0` | Denoising strength (raise with high ISO ~ noise σ) |
| `template_window_size` | int | `7` | Non-Local Means template window |
| `search_window_size` | int | `21` | Search window (smaller = faster) |

### `unsharp`
| Field | Type | Default | Description |
|---|---|---|---|
| `sigma` | float | `2.0` | Gaussian blur standard deviation |
| `amount` | float | `0.5` | Sharpening intensity |

### `stabilizer`
| Field | Type | Default | Description |
|---|---|---|---|
| `min_radius` / `max_radius` | int | `30` / `400` | Disk radius limits (adjusted to the frame resolution) |
| `dp`, `min_dist`, `param1`, `param2` | — | `1.2` / `100` / `50` / `30` | `HoughCircles` parameters |
| `gaussian_kernel_size`, `gaussian_sigma` | — | `9` / `2.0` | Pre-detection blur |
| `contour_fallback` | bool | `true` | Contour fallback when Hough fails |
| `auto_crop` | bool | `true` | Removes the black borders of the translation (without cropping the disk) |
| `jitter_alpha` | float | `0.5` | EMA centroid smoothing (1 = no smoothing) |

### `polish`
| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Turns the polishing on/off |
| `corona_scale` | float | `1.6` | Cut line (× body radius): between the edge and the line the image is feathered into the background |
| `feather` | float | `0.02` | Smoothing (fraction of the radius) of the outline and body overlaps |
| `background_fill` | bool | `true` | Background = mean of the original background (outside the cut line) |
| `black_background` | bool | `false` | `true` = pure black background instead of the mean |
| `brightness` | float | `0.15` | Extra brightness added to the bodies (0 = only contrast stretching) |
| `remove_reflections` | bool | `true` | Removes ghost circles (center outside the largest body) |
| `reflection_min_radius` | int | `8` | Minimum radius (px) of a reflection to remove (smaller = star/noise) |

### `feedback` (learning)
| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Stores ratings and applies the learned adjustment to the sliders |
| `db_path` | str | `~/.astroframe/feedback.db` | SQLite database with the run history and adjustments |
| `learning_rate` | float | `0.3` | Fraction of the delta applied per run |
| `user_weight` | float | `2.0` | Multiplier when the user rates manually |
| `history_limit` | int | `12` | Recent runs considered per profile |

### `lucky`
| Field | Type | Default | Description |
|---|---|---|---|
| `min_sharpness` | float\|null | `null` | Fixed sharpness threshold; `null` = estimate from the video |
| `sharpness_percentile` | float | `25.0` | Percentile used in the automatic estimation |
| `gaussian_kernel_size`, `gaussian_sigma` | — | `5` / `1.5` | Blur before the Laplacian |

### `stacking`
| Field | Type | Default | Description |
|---|---|---|---|
| `n_best` | int | `10` | Number of frames to stack (used if `--stack-n` is not given) |
| `use_median` | bool | `true` | `true` = median (robust), `false` = mean |

**Validation:** unknown keys and unexpected types generate warnings in the log
(for example `clip_limit: "abc"`), but never crash the startup.

## Video workflow

1. **Capture** — record the eclipse with a static camera; slow jitter is
   acceptable (the absolute disk stabilization compensates it).
2. **Preselection** (optional): `astroframe video --input clip.mp4 --mode stack --stack-n 30`
   returns a single PNG with the best possible "snapshot".
3. **Full stabilization**: `astroframe video --input clip.mp4 --mode enhance`
   — constant center and improved image. For 1080p/4K videos use `--fast`.
4. **Post-run**: merge the audio with ffmpeg (see limitations).

## Limitations and notes

- **Audio**: the exporter uses OpenCV and **does not copy audio**:
  ```bash
  ffmpeg -i original.mp4 -i processed.mp4 -c copy -map 0:a -map 1:v output.mp4
  ```
- **Slow denoising**: ~1 s/frame at 480p; at 1080p it can reach several
  seconds per frame. `--fast` or reduce `search_window_size`.
- **High-resolution stacking**: frames above 1080p stacked use float32 in
  memory (warning in the log) — reduce `n_best` if it exceeds what is needed.
- **Frames without a disk**: `center_and_stabilize` returns the frame
  unchanged (with a warning); in video, `AntiJitterStabilizer` reuses the last
  valid displacement.
- **RIFE** (interpolation over jumps) is optional and requires PyTorch; see
  [API.md](API.md).
