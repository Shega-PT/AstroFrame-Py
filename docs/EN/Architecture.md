# Solution Architecture

The system is divided into three main stages:

- Geometric Stabilization,
- Automatic Image Enhancement,
- Minimal Interface

---

# 1 Video Tracking and Stabilization (Jitter Cancellation)

Instead of trying to stabilize the background (which is dark or uniform), the algorithm detects the centroid of the Sun/Moon in each frame and moves the image to keep the eclipse always at the exact center of the frame.

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


def auto_enhance_eclipse_frame(img):
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
    result = auto_enhance_eclipse_frame(stabilized)
    return result


# Minimal Interface with Gradio
with gr.Blocks(title="Eclipse Auto-Enhancer") as demo:
    gr.Markdown("# 🌒 Eclipse Auto-Enhancer AI System")
    gr.Markdown("Automatic enhancement and geometric stabilization for eclipse photos and frames.")

    with gr.Row():
        input_img = gr.Image(label="Original Photo/Frame")
        output_img = gr.Image(label="Stabilized and Processed Result")

    btn = gr.Button("Process Image", variant="primary")
    btn.click(fn=process_image_pipeline, inputs=input_img, outputs=output_img)

if __name__ == "__main__":
    demo.launch()
```

---

# Modules for Expanding the Video Pipeline

To extend this pipeline to the Samsung Digital Camcorder videos with fast movements:

1. Stacking / Lucky Imaging (recommended libraries):

   - Use scikit-image or OpenCV to read the video frame by frame.
   - Compute `cv2.Laplacian(frame, cv2.CV_64F).var()`. If the value is below a predefined threshold, discard the frame (it was captured during a fast camera pan).

2. Motion Interpolation If "Jumps" Occur:

   - If the camera moves too fast and misses some good frames, you can use the RIFE model (Real-Time Intermediate Flow Estimation) via PyTorch to smoothly interpolate frames between manual elevation and direction corrections.
