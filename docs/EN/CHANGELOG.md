# Changelog

All notable changes to AstroFrame will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versioning [SemVer](https://semver.org/).

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
