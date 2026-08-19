# Solution Architecture

The system is divided into three main stages:

- Geometric Stabilization,
- Automatic Image Enhancement,
- Minimal Interface

Since v0.7.0 there is also an **optional AI layer** (auto-tuning and small
neural networks), described in [section 4](#4-ai-layer-auto-tuning-and-small-neural-networks).

---

# 1 Video Tracking and Stabilization (Jitter Cancellation)

Instead of trying to stabilize the background (which is dark or uniform), the algorithm detects the centroid of the Sun/Moon in each frame and moves the image to keep the celestial body always at the exact center of the frame.

Shape Detection: Uses the Hough Circle Transform (`cv2.HoughCircles`) or detection of the largest contours (`cv2.findContours`).

Automatic Reframing: If the camera jumps quickly, the algorithm computes the displacement vector of the Sun's center $(x, y)$ relative to the frame center and re-aligns the image.

Blurred Frame Rejection (Lucky Imaging): Frames captured during very fast camcorder movements are blurred by motion blur. Python can compute the variance of the Laplacian of the image (sharpness level) and skip the most blurred frames.

---

# 2 Automatic Processing and Enhancement (Photos and Videos)

Adaptive Gamma and Contrast (CLAHE): The Contrast Limited Adaptive Histogram Equalization algorithm improves the details of the solar corona or the illumination transition without blowing out the brightness.

Noise Reduction: Application of Non-Local Means Denoising (especially useful for photos with a high ISO).

Unsharp Masking: Highlights the exact edges of the Moon over the solar disk.

---

# 3 Minimal Frontend (Gradio or Streamlit)

The fastest way to build the GUI in Python is to use Gradio. It runs locally in the browser with file selectors, sliders, and side-by-side visualization (Before vs. After).

Pipeline Code Example (Python): here is a functional base implementation using OpenCV for the processing logic and Gradio for the interface.

```Python
import cv2
import numpy as np
import gradio as gr


def auto_enhance_celestial_frame(img):
    """
    Applies adaptive equalization and sharpening focused on astrophotography.
    """
    # Convert to grayscale for edge analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. CLAHE filter to enhance corona/edge contrast without blowing out
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # Apply on the L channel of the LAB color space to preserve the original colors
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_enhanced = clahe.apply(l)
    updated_lab = cv2.merge((l_enhanced, a, b))
    enhanced_bgr = cv2.cvtColor(updated_lab, cv2.COLOR_LAB2BGR)

    # 2. Gentle noise reduction
    denoised = cv2.fastNlMeansDenoisingColored(enhanced_bgr, None, 5, 5, 7, 21)

    # 3. Unsharp masking to highlight the Moon's limb
    gaussian_3 = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    unsharp = cv2.addWeighted(denoised, 1.5, gaussian_3, -0.5, 0)

    return unsharp


def center_and_stabilize(img):
    """
    Locates the Sun/Moon by geometry and centers the image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Search for circular shapes
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=50, param2=30, minRadius=30, maxRadius=400
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Take the largest circle found (Sun)
        x, y, r = circles[0][0]

        # Compute the displacement to the image center
        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2
        dx = center_x - x
        dy = center_y - y

        # Translation matrix
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        stabilized = cv2.warpAffine(img, M, (w, h))
        return stabilized

    return img


def process_image_pipeline(input_image):
    if input_image is None:
        return None
    # 1. Center on the solar disk
    stabilized = center_and_stabilize(input_image)
    # 2. Apply automatic enhancements
    result = auto_enhance_celestial_frame(stabilized)
    return result


# Minimal Interface with Gradio
with gr.Blocks(title="AstroFrame") as demo:
    gr.Markdown(
        "# AstroFrame — geometric stabilization and automatic enhancement of astrophotographs and astrovideos"
    )
    gr.Markdown(
        "Detects celestial bodies (Sun, Moon, planets, comets) and stabilizes and enhances their photos and videos automatically."
    )

    with gr.Row():
        input_img = gr.Image(label="Original Photo/Frame")
        output_img = gr.Image(label="Stabilized and Processed Result")

    btn = gr.Button("Process Image", variant="primary")
    btn.click(fn=process_image_pipeline, inputs=input_img, outputs=output_img)

if __name__ == "__main__":
    demo.launch()
```

---

# 4 AI Layer (Auto-Tuning and Small Neural Networks)

Introduced in v0.7.0. The pipeline parameters are no longer only manual:
a deterministic optimizer searches them against the calibration samples, and
small neural networks (pure NumPy — no new dependencies; PyTorch, mandatory
since v0.9.0, is used only by the RIFE interpolation)
learn from the run history. **Everything is off by default**
(`tuning.enabled=false`, `ai.*`) and degrades silently — a missing or corrupt
model never blocks the pipeline.

## 4.1 Registry of Tunable Parameters (`ai.params`)

Single source of truth for the adjustable parameters of the pipeline
(previously spread across `validator.py` and `feedback.py`). Each `ParamSpec`
declares the safe range (`low`/`high`), the optimization step, the dtype
(ints are rounded), the odd parity (Gaussian kernels must stay odd), the
group (detect / geometry / enhance / stack / polish / score / meta), the
evaluation cost (denoising is expensive) and the validator's reward/punish
deltas. **Every learned value passes through `clamp_value`** — the clamps
are always applied via the registry, so the training can never leave the
safe ranges.

## 4.2 Auto-Tuning (`ai.tuner`)

Two pieces work together:

- **Proxy evaluation** (`ProxyEval`) — the pipeline runs on the calibration
  images (`samples/`; ground truth by default at `Logs/train/calibration.json`,
  with `samples/calibration.json` as fallback) reduced to ~480p
  (maximum work scale 0.5, never upscaled). The detection is compared with
  the expected disks: **mean IoU** between detected (Hough) and ground-truth
  disks, with **penalties for extra/missing disks**; a few enhanced frames
  are also scored with stars. The reports are cached by the effective
  parameter values, so the search only re-evaluates what changed.
- **Bounded hill climbing** (`BoundedHillClimb`) — deterministic (fixed
  seed) search with a **time budget**: per-parameter +step/−step passes,
  momentum (the step doubles after 2 consecutive accepts in the same
  direction, halves on failures, min step/8), optional **annealing**
  (worse candidates accepted with probability exp(−Δ/T), T decaying per
  pass — escapes local minima without leaving the safe ranges) and
  patience. Costly parameters (denoising) are tried only on even passes.
  The search can be **pre-seeded with LSTM predictions** when they improve
  the proxy objective.

The orchestration (`run_autotune`) evaluates, optimizes, exports the tuned
configuration (`export_trained_config`, by default
`Logs/train/trained_config.json`) and **registers the result in the feedback
DB** (`tuning` table). From then on, `apply_learned` applies those deltas
automatically to every run of the same profile. Entry points: the CLI
(`astroframe autotune`) and the *Auto-tune* tab of the Gradio interface.

## 4.3 LSTM (`ai.lstm`)

A single 1-layer LSTM cell implemented by hand in NumPy (forward + backward
with backprop-through-time, vectorized — no torch dependency; `torch_available()`
reports the PyTorch used only by the RIFE interpolation). Two applications:

- **`LSTMTuner`** — trained offline on the feedback history (one run = one
  timestep: star ratings + metrics; sliding windows, validation and early
  stop) and predicts the **parameter deltas** of the next run. It pre-seeds
  the auto-tuning, accelerating convergence. Without enough history it
  returns `{}` and the search starts from the base.
- **`TrajectoryPredictor`** — predicts the **next disk centroid** from the
  last detections: linear regression (least squares) as the base, with an
  optional LSTM refinement (cell 2→8, trained on synthetic trajectories by
  `train_trajectory_model`). It integrates with the pipeline through the
  `AntiJitterStabilizer` (`ai.lstm_trajectory`): in frames without
  detection the centroid is **predicted** instead of frozen.

Models are saved as versioned `.npz` files (`Logs/weights/lstm.npz`); a
corrupt or wrong-version file falls back silently.

## 4.4 CNN (`ai.cnn`)

A small convolutional network in pure NumPy (2× conv 3×3 + ReLU + pooling +
MLP head) with two interchangeable heads, deterministic offline training
(fixed seed) and gradients verified by finite differences:

- **Residual enhancer** (`fit_residual` / `ResidualEnhancer`) — learns the
  residual `r = y − x` from (input, target) pairs, i.e. the removal of
  noise/smearing. Applied in the enhancement pipeline **after the unsharp
  step** (`ai.cnn_enhance`): the residual is added to the L channel of LAB
  in 64×64 tiles with overlap, preserving the colors. Without a model the
  image comes out unchanged.
- **Detection filter** (`fit_classifier` / `DiskFilter`) — a disk vs. noise
  classifier that scores each detection (`confidence`, P(disk)). With
  `ai.disk_filter > 0.0` the detection step drops candidates below the
  threshold — filtering false positives of the Hough transform. It
  **never empties** the detected list: the detection never regresses.

Models: `Logs/weights/enhancer_cnn.npz` and `Logs/weights/disk_filter.npz`.

### Training the NNs

- **Detection CNN** — the `validator.py` automatic training collects disk
  patches each series (`cnn_positives` from guide circles, `cnn_negatives`
  from rejected shapes + deterministic random crops excluded by IoU) and
  retrains `DiskFilter` between series (`--cnn`). The result is compared with
  the **champion** in the `models` table by `score`; if strictly better it is
  promoted to the canonical path and the next series warm-starts from the
  champion's weights.
- **Residual CNN** — `enhancer_trainer.py` (GUI Valid/Rejected or `--auto`)
  accumulates pairs (input, CNN output) / (input, input) and calls
  `train_enhancer_round`, which compares by `mean_delta` (1 − MSE) against
  the champion and promotes if better.
- Both trainers log every round into the `models`/`logs` tables.

## 4.5 Feedback Integration (`ai.feedback`)

`apply_learned(cfg, profile, db)` sums two sources at the start of each run:

- the **star-rating nudges** (the `runs` table, the 5 visual parameters);
- the **auto-tuning deltas** (the `tuning` table, any registered parameter).

Both are clamped through the `ai.params` registry. This is the AI "memory"
across runs: with nothing learned, the configuration is returned unchanged.
`ASTROFRAME_FEEDBACK_DB` overrides the database path
(`Logs/logs/system/feedback.db` by default). Since v0.8.0 the same database also
holds the `models` table (NN artifacts: kind, metric value, promoted flag,
champion status, series) and the `logs` table (per-component event history).

## 4.6 Security Model

- All AI is **off by default** (`tuning.enabled=false`, `ai.lstm_trajectory`,
  `ai.cnn_enhance`, `ai.disk_filter=0.0`).
- A missing or corrupt model **degrades silently** (loads as `None`), and
  the auto-tuning seed falls back to the base configuration.
- The search never leaves the safe ranges of the registry, and the neural
  networks never raise in runtime (no history → `None`/`{}`).

---

# Modules for Expanding the Video Pipeline

To extend this pipeline to the Samsung Digital Camcorder videos with fast movements:

1. Stacking / Lucky Imaging (recommended libraries):

   - Use scikit-image or OpenCV to read the video frame by frame.
   - Compute `cv2.Laplacian(frame, cv2.CV_64F).var()`. If the value is below a predefined threshold, discard the frame (it was captured during a fast camera pan).

2. Motion Interpolation If "Jumps" Occur:

   - If the camera moves too fast and misses some good frames, you can use the RIFE model (Real-Time Intermediate Flow Estimation) via PyTorch to smoothly interpolate frames between manual elevation and direction corrections.
