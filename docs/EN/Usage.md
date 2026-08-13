# AstroFrame User Guide

Practical guide to install, configure and run AstroFrame. The solution
specification is in [Architecture.md](Architecture.md) and the code reference
in [API.md](API.md).

## Table of contents

1. [Installation](#installation)
2. [Web interface (Gradio)](#web-interface-gradio)
3. [Calibration](#calibration)
4. [Command line](#command-line)
5. [Configuration (config.yaml)](#configuration-configyaml)
6. [Video workflow](#video-workflow)
7. [Limitations and notes](#limitations-and-notes)

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
and videos from the [samples/](../../samples/README.md) folder, shows the
detection with circles (circular bounding boxes) that can be **added, removed
and moved manually**, and validates the automatic detection against the ground
truth on **all** samples.

```bash
python calibrate.py                          # interface at http://127.0.0.1:7860
python calibrate.py --samples samples --port 7861
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

1. **Choose the sample** — the dropdown lists all items
   (`IMG …` for images, `VID … #frame` for video frames).
2. **Adjust the circles** — each circle is a layer over the image:
   - **drag** the layer → moves the circle;
   - **brush** over the body → adds (the painting is converted to a circle at
     the center of what you painted);
   - **eraser** → removes.
   - The pre-filled circle is the stored ground truth; without ground truth,
     the **automatic detection** enters as a starting point.
3. **Save adjustments** — writes the item's ground truth to
   `samples/calibration.json` (local file, ignored by git).
4. **Automatic detection** — restores the circles detected by
   `find_all_disks` on the current item (replaces what is in the editor).
5. **Validate all samples** — runs the automatic detection on everything and
   compares it with the manual ground truth: per sample and globally it
   returns recall, precision, mean IoU, center error (px) and radius error
   (%), false negatives/positives, a **calibration score (0–100)** and
   **parameter suggestions** (e.g. lower `min_radius` if small disks fail,
   raise `param2` if there are false detections).

### What the calibration is for

The manual circles are the "right answer" that the system compares with the
automatic detection. With a varied folder (eclipses, Moon, Sun, planets —
large and small disks, high and low contrast), the validation shows where the
detection fails and what to adjust in the `config.yaml` before processing the
real material.

## Command line

The complete subcommands (`astroframe --help`):

| Command | Description |
|---|---|
| `serve` | Starts the Gradio interface |
| `process` | Processes photos in batch (`--input a.jpg b.jpg --output-dir folder/`) |
| `video` | Processes a video (`--mode stabilize\|enhance\|stack`) |
| `config-template` | Generates `config.yaml` with default values |
| `calibrate` | Opens the calibration interface (`--samples folder/`) |

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
