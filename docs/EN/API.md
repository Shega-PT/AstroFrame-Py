# API Reference — AstroFrame

Reference of the modules and contracts of the `astroframe` package. For
practical usage, see [Usage.md](Usage.md).

**Global conventions**

- Images: numpy `np.ndarray` **BGR** (OpenCV convention), `uint8`, `(H, W, 3)`.
  Grayscale `(H, W)` and RGBA `(H, W, 4)` (interpreted as RGBA) inputs are
  normalized to BGR automatically.
- Configuration: `AstroFrameConfig`; any processing function accepts it
  optionally (uses the default values if omitted).

## `astroframe.config`

Dataclasses with the parameters (see
[Usage.md](Usage.md#configuration-configyaml) for the field-by-field table):

- `CLAHEConfig`, `DenoiseConfig`, `UnsharpConfig`
- `StabilizerConfig`, `PolishConfig`, `FeedbackConfig`, `LuckyConfig`, `StackingConfig`
- `AstroFrameConfig` — root with one field per subconfiguration.

Methods of `AstroFrameConfig`:

```python
cfg.to_dict() -> dict
cfg.to_yaml(path) -> None
cfg.from_yaml(path) -> AstroFrameConfig   # classmethod
```

`from_yaml` validates the types and warns (logging) about unknown keys and
unexpected types, without failing.

## `astroframe.core`

### `core.stabilizer`

```python
@dataclass(frozen=True)
class DiskDetection:
    cx: int
    cy: int          # center detected in the source image coordinates
    radius: int      # radius adjusted after crop/rescale

find_all_disks(image, config=None) -> list[DiskDetection]
find_disk_center(image, config=None) -> DiskDetection | None
center_and_stabilize(image, config=None) -> tuple[np.ndarray, DiskDetection | None]
class AntiJitterStabilizer(config=None, alpha=None): ...
```

- `find_all_disks` — **two passes** of `HoughCircles` (the second with
  `minDist` 1/4 of normal, for circles inside the largest body) +
  contour fallback; up to **5 disks**, sorted by decreasing radius.
  Dedup only of circles of the **same edge** (close centers AND near-equal
  radii, tolerance 12% of the radius), and rejection of **ghost circles**:
  a candidate almost fully inside an already accepted disk (≥90% of the
  area) with weak contrast against the ring around it is discarded
  (`_is_occluded_artifact`). Accepts BGR or grayscale.
- `find_disk_center` — first element of `find_all_disks`; HoughCircles +
  contour fallback + intensity-centroid refinement. On frames ≥1200 px the
  detection runs at half resolution.
- `center_and_stabilize` — translates the frame to center the disk and crops
  the black borders (`stabilizer.auto_crop`), returning the adjusted radius.
  Without a detected disk it returns the unchanged image and `None`.
- `AntiJitterStabilizer.stabilize(frame) -> (frame, DiskDetection | None)` —
  internal state: centroid EMA (`jitter_alpha`) and reuse of the last valid
  displacement in frames without detection (`last_detection` — property with
  the last detected disk, used by the video for polishing/preview).

### `core.polish`

```python
polish_image(image, detection, config=None) -> np.ndarray
```

- **Per-body polishing**: detects all disks (`find_all_disks`), separates
  eclipse companions (center inside the largest body) from lens reflections
  (center outside), enhances **each body individually** (`_astro_boost`:
  local contrast stretching + `polish.brightness`; dark and uniform
  silhouettes — like the Moon in eclipse — are preserved intact) and
  **recomposes seamlessly** by mask blending with feathering (`_band_mask` +
  `_astro_region`): the cut line `corona_scale × radius` feathers the ring
  into the background and body overlaps are the smooth average of the
  enhancements. The background is the **mean of the original background**
  (`background_fill`) or pure black (`black_background`); reflections (radius
  ≥ `reflection_min_radius` px) are filled with the background if
  `remove_reflections`. Without detection it returns the unchanged image.

### `core.enhancer`

```python
clahe_enhance(image, config) -> np.ndarray
denoise(image, config) -> np.ndarray
unsharp_mask(image, config) -> np.ndarray
enhance_image(image, config=None, use_denoise=True) -> np.ndarray
```

- Order: CLAHE on the L channel of LAB → `fastNlMeansDenoisingColored` →
  unsharp.
- `use_denoise=False` omits the slowest step (used by `--fast`).

### `core.pipeline`

```python
@dataclass
class ProcessResult:
    original: np.ndarray
    stabilized: np.ndarray
    enhanced: np.ndarray       # stabilized + CLAHE + denoise + unsharp + polishing
    enhanced_raw: np.ndarray   # the same, without polishing (rating basis)
    detection: DiskDetection | None

process_image(image, config=None) -> ProcessResult
process_path(path, config=None) -> ProcessResult   # ValueError if unreadable
```

## `astroframe.video`

### `video.reader`

```python
class FrameReader(path):
    .fps -> float
    .frame_count -> int          # 0 when unknown
    .size -> tuple[int, int]     # (width, height)
    .close() / context manager
    iterable: BGR frames
    # ValueError if the video does not open
```

### `video.select` (lucky imaging)

```python
sharpness(frame, config=None) -> float        # Laplacian variance
estimate_sharpness_threshold(scores, percentile=25.0) -> float
select_sharp_frames(frames, config=None, minimum=None) -> list[(idx, frame, score)]
```

- Threshold order: `minimum` → `config.lucky.min_sharpness` → estimated
  percentile of the sequence itself.

### `video.stacking`

```python
stack_frames(frames, stacking=None) -> np.ndarray   # ValueError if empty or different shapes
select_best(frames, n_best, config=None) -> list[np.ndarray]
```

- `stack_frames`: median (`use_median=True`) or mean; warns about memory
  above 1080p with many frames.

## `astroframe.meta`

Metadata reading and parameter suggestions (own implementation, MIT —
inspired by the same idea of MetadataExplorer, without copied code).

### `meta.extractor`

```python
@dataclass(frozen=True)
class MediaMetadata:
    path: str | None
    kind: str                       # "image" | "video"
    width: int | None
    height: int | None
    aspect_ratio: float | None      # width / height
    fps: float | None
    frame_count: int | None
    duration: float | None          # seconds
    codec: str | None
    bitrate: int | None             # bits/second
    format_name: str | None
    iso: int | None                 # ISO sensitivity (EXIF)
    exposure_time: float | None     # seconds
    focal_length: float | None      # mm
    aperture: float | None          # f-number
    camera_make: str | None
    camera_model: str | None
    captured_at: str | None         # EXIF date/time
    raw: dict                       # everything read (source→key→value)

extract_metadata(path) -> MediaMetadata
```

- Video: **ffprobe** cascade (if installed; codec/bitrate/duration/format) →
  **OpenCV** (resolution/fps/frames — always available).
- Image: EXIF via PIL (ISO, exposure, aperture, focal length, camera, date).
- `ValueError` if the path does not exist; `kind="unknown"` with what can be
  read if neither ffprobe nor OpenCV opens the file.
- `aspect_ratio` rounded to 3 decimals (0.0 → `None`); the presentation text
  (e.g. `5616×3744 · 3:2`) is `aspect_text` in `extractor` (16:9, 3:2, 4:3,
  1:1, square or decimal change).

### `meta.suggest`

```python
suggest_config(meta: MediaMetadata) -> AstroFrameConfig
summary_fields(meta: MediaMetadata) -> dict[str, str]
```

- Heuristics: detection radii proportional to the resolution (`min = 8%` of
  the minor semi-axis, `max = 45%`); `denoise.h` scaled by ISO
  (`2 + ISO/1600*4`, limited to `[2, 15]`, used by default if the config does
  not define it) with `unsharp` 0.4/0.6; in heavily compressed videos
  (< 0.1 bit/pixel) the denoise is reduced ~30% (less risk of "plastifying").
- `summary_fields` returns the dictionary displayed in the "Ratio / quality /
  suggestions" panel of the interface.

## `astroframe.calibration`

Detection calibration against examples (photos/videos), with manual ground
truth.

### `calibration.scan`

```python
@dataclass(frozen=True)
class SampleRef:
    kind: str            # "image" | "video"
    path: Path           # absolute file path
    frame: int | None    # frame index (None for images)
    key: str             # "relative_path#frame" (stable key in the store)
    label: str           # "IMG path" / "VID path #frame" (interface)

scan_samples(root, frames_per_video=8) -> list[SampleRef]
sample_video_frames(frame_count, n=8) -> list[int]
load_frame(sample) -> np.ndarray          # BGR (image or sampled frame)
item_key(relpath, frame=None) -> str
item_label(kind, relpath, frame=None) -> str
```

- **Recursive** scan of the folder; images (jpg/jpeg/png/bmp/tif/tiff/webp)
  enter as-is and videos (mp4/avi/mov/mkv/m4v) contribute N **equidistant and
  deterministic** frames (midpoints — reproducible in validation). Unreadable
  videos are ignored with a warning.
- `load_frame` reads the image via `cv2.imread` or the video frame via
  `FrameReader.frame_at(index)` (new — seeks `CAP_PROP_POS_FRAMES`; errors
  raise `ValueError`).

### `calibration.store`

```python
@dataclass
class CalibrationItem:
    path: str            # path relative to the samples folder
    kind: str
    frame: int | None
    width: int
    height: int
    circles: list[DiskDetection] = []   # manual ground truth

class CalibrationStore(path):           # JSON v1 (samples/calibration.json)
    .load() -> None                     # idempotent; unreadable/version -> empty
    .save() -> None
    .upsert_item(key, item) -> None     # writes immediately
    .get_item(key) -> CalibrationItem | None
```

### `calibration.circles`

```python
circles_to_layers(image_rgb, circles) -> {"background": ..., "layers": [...]}
layers_to_circles(layers) -> list[DiskDetection]
```

- `circles_to_layers` builds the value of the `gr.ImageEditor`: the background
  + **one RGBA layer per circle** (translucent disk + opaque edge). The layers
  are **draggable** in the interface → moving a circle = dragging the layer;
  painting over adds; the eraser removes.
- `layers_to_circles` converts what the user drew into circles — one per
  **connected component** of each layer (two separate paintings on the same
  layer = two circles); accepts layers with alpha or RGB.

### `calibration.validate`

```python
circle_iou(a, b) -> float                                  # intersection/union 0–1
match_circles(manual, detected, iou_threshold=0.5)
    -> (pairs: list[(i, j)], unmatched_manual: set, unmatched_detected: set)

@dataclass
class ItemReport:        # per sample
    label, n_manual, n_detected, n_matched,
    n_false_negatives, n_false_positives,
    mean_iou, mean_center_error, mean_radius_error_pct   # None without pairs

@dataclass
class CalibrationReport: # aggregate
    items, total_* , recall, precision,
    mean_iou, mean_center_error, mean_radius_error_pct,
    score: float | None  # 0–100 = 0.4·recall + 0.3·precision + 0.3·IoU

validate_item(label, manual, detected) -> ItemReport      # signed radius error (%)
validate_all([(label, manual, detected), ...]) -> CalibrationReport
suggest_parameters(report, config=None) -> list[str]      # suggestions
```

- **Greedy matching by decreasing IoU** (threshold 0.5): manual↔detection.

## `astroframe.ui`

### `ui.gradio_app`

```python
build_app(config=None) -> gr.Blocks
run(config_path=None, host="127.0.0.1", port=7860, share=False, inbrowser=True) -> None

def inspect_video_upload(video_path, db=None, config=None) -> tuple[str, dict, dict, dict, dict, dict]
def process_video(video_path, export=False, denoise_h=None, ...) -> Generator[tuple]
def process_image_input(image, clip_limit=None, denoise_h=None, ..., db=None, config=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict, str]
def manual_feedback(state, stars, db=None, config=None) -> tuple[str, str]
```

- The UI converts RGB→BGR on input and BGR→RGB on outputs (functions
  `_to_pipeline` / `_from_pipeline`); the values and the pipeline are shared
  with the CLI.
- Two tabs: **Image** (input, stabilized, processed, zoom, rating, sliders,
  manual rating + learning log) and **Video** (upload, metadata panel,
  pre-filled sliders, live processing with drawn disks, automatic and manual
  rating, optional export).
- `inspect_video_upload` calls `meta.extractor` + `meta.suggest` +
  `apply_learned` (previous ratings of the same camera profile) and returns,
  respectively: the summary HTML (ratio/quality/suggestions), the raw
  metadata and the slider `update()`s.
- `process_video` is a **generator** (consumed by the Gradio
  `gr.Progress.track`); each frame returns:
  `(live_rgb, preview_rgb, out_video_path_or_None, status, progress, rating_html,
  run_state, log_html)` — `live` is the original frame in real time with the
  detected disks (`_draw_disks`: **green** = largest body, **yellow** =
  eclipse companions, **red** = reflections — separated by `_split_disks`,
  which uses the disk center vs. the radius of the largest body), `preview` is
  the final result shown only at spaced frames (`_preview_every`), the other
  fields with `None`/fraction in the middle of the pass. Without a detected
  disk in **any** frame, the final result comes out without polishing and the
  rating is computed without detection (warning in the status). If
  `export=True`, it writes the full polished video (.mp4, `mp4v` codec,
  without audio) and returns the path on the last frame.
- `process_image_input` returns `(stabilized, processed, zoom, rating HTML,
  state, learning log)`; the state (profile, rating, parameters) feeds
  `manual_feedback`, which stores the star rating and reports the learned
  adjustment in the log.
- `run()` accepts `inbrowser` to open the browser automatically; the
  equivalent single entry point is `python main.py` at the repository root.

### `ui.calibration_app`

```python
build_calibration_app(samples_dir="samples", config=None, store=None) -> gr.Blocks
run(samples_dir="samples", config_path=None, host="127.0.0.1", port=7860, share=False, inbrowser=True) -> None

def load_item_payload(key, samples_dir, config=None, store=None) -> (dict, str)
def auto_detect_payload(key, samples_dir, config=None) -> (dict, str)
def save_item_circles(editor_value, key, samples_dir, store=None) -> str
def validate_all_report(samples_dir, config=None, store=None) -> (rows, summary_html, suggestions_html)
```

- Layout: samples dropdown + `gr.ImageEditor` (RGBA layers per circle,
  brush/eraser) + buttons "Automatic detection" / "Save adjustments" /
  "Validate all samples" → per-sample table, global summary (score 0–100,
  recall, precision, IoU, errors) and parameter suggestions.
- `load_item_payload` gives priority to the stored ground truth; without it
  it uses the automatic detection as a starting point. `save_item_circles`
  converts the editor layers into circles and saves them in the store.
  `validate_all_report` walks **all** samples (images + sampled video
  frames).
- Equivalent entry point: `python calibrate.py` at the repository root or
  `astroframe calibrate` (CLI).

### `ui.cli`

```python
main(argv=None) -> int                 # entry point of the `astroframe` script
build_parser() -> argparse.ArgumentParser
process_images(paths, output_dir, config) -> tuple[int, int]   # (successes, failures)
process_video(path, output, config, mode, stack_n, fast) -> str  # output path
```

- Subcommands: `serve`, `process`, `video` (`--mode stabilize|enhance|stack`,
  `--fast`), `config-template`, `calibrate` (calibration interface).
- `process_images` continues after individual failures and raises
  `RuntimeError` if nothing was processed. `mode="stack"` centers the frames
  before stacking. Video export does not copy audio (limited by OpenCV).

## `astroframe.ai` (optional until 0.3: RIFE)

```python
class RifeInterpolator(repo, source="github", model_name="IFNet", device=None):
    .available() -> bool            # stateless: False if PyTorch not installed
    .interpolate(frame_a, frame_b, n_interp=1) -> list[np.ndarray]
```

- Needs `pip install -e ".[rife]"`. Accepts BGR; returns `n_interp`
  intermediate frames in BGR. The model interface depends on the RIFE
  repository used (the internal `_infer` is the point to adjust between
  versions); without PyTorch it raises `RuntimeError` with instructions.

### `ai.score` (automatic rating)

```python
@dataclass
class StarRating:
    stars: float            # 0.0–5.0 (weight of the metrics = 1)
    score: float            # 0.0–1.0 unweighted
    metrics: dict[str, float]  # noise | contrast | size | corona; 0 (bad) to 1 (good)
    explanation: str        # human text with the why

score_image(image, detection=None, config=None) -> StarRating
package_rating(original, stabilized, detection, config=None) -> StarRating
score_from_stars(stars, metrics=None) -> StarRating   # for tests/externalization
```

- `noise` = Laplacian variance (no noise → 1), `contrast` = 99/50 luminance
  percentile ratio, `size` = disk radius vs. frame, `corona` = mean brightness
  of the corona ring (1–2× radius) vs. the disk.
- `score_image` works **without detection** (noise/contrast metrics only).

### `ai.feedback` (learning by rating)

```python
@dataclass(frozen=True)
class ConfigNudge:
    clip_limit: dict       # {multiplier, offset}   e.g.: {'m': 1.0, 'b': 0.0}
    denoise: dict          #                           e.g.: {'h': {'m': 1.0, 'b': 1.5}}
    unsharp: dict          # {'amount': {...}, 'sigma': {'m': 0.7, 'b': 1.2}}
    polish: dict           # {'brightness': {...}}
    explanation: dict[str, str]  # text per parameter (why the adjustment)
    judicial_override: bool      # True: a bad rating imposes the correction immediately
    factor: float          # global scale of the correction (feedback.learning_rate)

@dataclass(frozen=True)
class RunRecord:
    id: int; profile: str; kind: str; params: dict; metrics: dict
    stars: float; source: str; at: str; modified: dict | None

def profile_for(kind, width, height) -> str     # e.g.: "video@5616x3744"
def format_profile(profile) -> str              # "5616×3744 · vídeo" (interface)
def record_run(db, kind, profile, config, params, rating, source="cli") -> RunRecord
def recent_nudges(db, profile, limit=5) -> list[RunRecord]
def apply_learned(config, profile, db=None) -> AstroFrameConfig
def _learning_db(config, db=None) -> FeedbackDB | None   # feedback.enabled?
def _learning_log_html(profile, db) -> str               # history (interface)
class FeedbackDB(path=None):               # SQLite with locking retry (WAL)
    .history(profile, limit=50, base=None) -> list[RunRecord]
    .latest_ids(profile, limit=5) -> list[int]
    .nudges(profile_runs, nudge_params, factor) -> ConfigNudge  # rules
    .store_run(kind, profile, config, params, stars, source, metrics) -> RunRecord
    .apply_nudge(config, nudge) -> AstroFrameConfig
```

- SQLite database at `~/.astroframe/feedback.db` (or `$ASTROFRAME_FEEDBACK_DB`);
  created on first use, with retry on a locked base and `history_limit` per
  profile.
- `apply_learned` returns the original config if there is no history (or the
  `judicial_override`/`factor` is null); rules: good and consistent ratings
  smooth the adjustment (`user_weight`), bad ratings apply extra denoise with
  noise (metrics >`1.0`), weak corona increases the polishing brightness,
  small disk reduces the detector radii; values are limited to valid ranges.
- `FeedbackDB.default_path() -> Path`, `.path -> Path`, `.close()`.
