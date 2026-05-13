# preprocess.py — 6-Stage Preprocessing Pipeline
# Stages: Resize+Normalise → CLAHE → Gaussian Denoise →
#          Canny Edge → Histogram Analysis → Fourier Noise Removal

import cv2
import numpy as np
import os
from config import (
    IMG_SIZE, VGG_MEAN_BGR,
    CLAHE_CLIP, CLAHE_GRID,
    GAUSS_KERNEL, GAUSS_SIGMA,
    CANNY_LOW, CANNY_HIGH,
    FOURIER_RADIUS, LAPLACIAN_THRESH,
    OUTPUT_DIR,
)


class SmokePreprocessor:
    """Full 6-stage preprocessing pipeline for smoke density estimation."""

    def __init__(self, save_debug=False):
        """
        Args:
            save_debug (bool): If True, save intermediate stage images to outputs/.
        """
        self.save_debug = save_debug
        self.clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP,
            tileGridSize=CLAHE_GRID
        )
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Stage 1: Resize & Normalise ──────────────────────────────────────────
    def resize_normalise(self, img):
        """Resize to 224×224. Mean subtraction done later in blob creation."""
        img_resized = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        return img_resized

    # ── Stage 2: CLAHE Contrast Enhancement ──────────────────────────────────
    def apply_clahe(self, img):
        """Enhance local contrast on the L* channel in CIE L*a*b* space."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ── Stage 3: Gaussian Denoising ───────────────────────────────────────────
    def gaussian_denoise(self, img):
        """Suppress JPEG grain and sensor noise while preserving plume edges."""
        return cv2.GaussianBlur(img, GAUSS_KERNEL, GAUSS_SIGMA)

    # ── Stage 4: Canny Edge Detection (Diagnostic) ───────────────────────────
    def canny_edges(self, img):
        """Return a Canny edge map for visualisation / diagnostic purposes."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

    # ── Stage 5: Histogram Analysis ───────────────────────────────────────────
    def histogram_analysis(self, img_before, img_after):
        """
        Compute per-channel histograms before and after CLAHE.
        Returns Bhattacharyya distance as quality score.
        Higher distance = more effective enhancement.
        """
        distances = []
        for ch in range(3):
            h1 = cv2.calcHist([img_before], [ch], None, [256], [0, 256])
            h2 = cv2.calcHist([img_after],  [ch], None, [256], [0, 256])
            cv2.normalize(h1, h1)
            cv2.normalize(h2, h2)
            distances.append(
                cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
            )
        return distances   # [dist_B, dist_G, dist_R]

    # ── Stage 6: Fourier Transform Noise Removal ─────────────────────────────
    def fourier_filter(self, img):
        """
        Selectively apply low-pass DFT filter to remove periodic JPEG artefacts.
        Only applied when Laplacian variance of DFT magnitude > LAPLACIAN_THRESH.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        dft  = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        magnitude = 20 * np.log(
            cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1
        )
        lap_var = cv2.Laplacian(magnitude.astype(np.uint8), cv2.CV_64F).var()

        if lap_var <= LAPLACIAN_THRESH:
            return img  # No artefacts detected — skip filter

        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        radius = int(min(rows, cols) * FOURIER_RADIUS)

        mask = np.zeros((rows, cols, 2), dtype=np.uint8)
        cv2.circle(mask, (ccol, crow), radius, (1, 1), thickness=-1)

        dft_filtered = dft_shift * mask
        dft_back = np.fft.ifftshift(dft_filtered)
        img_back = cv2.idft(
            dft_back, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT
        )
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        gray_filtered = img_back.astype(np.uint8)

        # Merge filtered gray back into colour image
        img_filtered = cv2.cvtColor(gray_filtered, cv2.COLOR_GRAY2BGR)
        return img_filtered

    # ── Master Pipeline ───────────────────────────────────────────────────────
    def run(self, img, debug_name=None):
        """
        Apply full 6-stage pipeline. Returns preprocessed image (BGR, 224×224).

        Args:
            img        : Raw BGR image (numpy array)
            debug_name : If save_debug=True, prefix for saved intermediate images.

        Returns:
            preprocessed (np.ndarray): BGR uint8 image ready for VGG16 blob.
            edges        (np.ndarray): Canny edge map for visualisation.
            hist_dist    (list)      : Bhattacharyya distances [B, G, R].
        """
        img_raw = img.copy()

        # Stage 1
        img_s1 = self.resize_normalise(img_raw)

        # Stage 2 — CLAHE
        img_s2 = self.apply_clahe(img_s1)

        # Stage 3 — Gaussian Denoise
        img_s3 = self.gaussian_denoise(img_s2)

        # Stage 4 — Canny edges (diagnostic only, not fed to VGG16)
        edges = self.canny_edges(img_s3)

        # Stage 5 — Histogram analysis (quality check)
        hist_dist = self.histogram_analysis(img_s1, img_s2)

        # Stage 6 — Fourier filter (conditional)
        img_s6 = self.fourier_filter(img_s3)

        if self.save_debug and debug_name:
            self._save_debug_outputs(debug_name, img_s1, img_s2, img_s6, edges)

        return img_s6, edges, hist_dist

    def _save_debug_outputs(self, name, raw, clahe, final, edges):
        prefix = os.path.join(OUTPUT_DIR, name)
        cv2.imwrite(f"{prefix}_1_raw.png",    raw)
        cv2.imwrite(f"{prefix}_2_clahe.png",  clahe)
        cv2.imwrite(f"{prefix}_3_final.png",  final)
        cv2.imwrite(f"{prefix}_4_edges.png",  edges)

    def save_preprocessing_demo(self, img, output_path="outputs/preprocessing_demo.png"):
        """Save a 3x2 grid showing all preprocessing stages side-by-side."""
        import matplotlib.pyplot as plt

        img_s1 = self.resize_normalise(img)
        img_s2 = self.apply_clahe(img_s1)
        img_s3 = self.gaussian_denoise(img_s2)
        edges  = self.canny_edges(img_s3)
        img_s6 = self.fourier_filter(img_s3)

        gray_s3 = cv2.cvtColor(img_s3, cv2.COLOR_BGR2GRAY)
        h1 = cv2.calcHist([img_s1],[0],None,[256],[0,256]).flatten()
        h2 = cv2.calcHist([img_s2],[0],None,[256],[0,256]).flatten()

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle("Preprocessing Pipeline — All 6 Stages", fontsize=14, fontweight="bold")

        axes[0,0].imshow(cv2.cvtColor(img_s1, cv2.COLOR_BGR2RGB))
        axes[0,0].set_title("Stage 1: Resize (224×224)")

        axes[0,1].imshow(cv2.cvtColor(img_s2, cv2.COLOR_BGR2RGB))
        axes[0,1].set_title("Stage 2: CLAHE Enhanced")

        axes[0,2].imshow(cv2.cvtColor(img_s3, cv2.COLOR_BGR2RGB))
        axes[0,2].set_title("Stage 3: Gaussian Denoised")

        axes[1,0].imshow(edges, cmap="gray")
        axes[1,0].set_title("Stage 4: Canny Edge Map")

        axes[1,1].plot(h1, color="blue",   alpha=0.6, label="Before CLAHE")
        axes[1,1].plot(h2, color="orange", alpha=0.8, label="After CLAHE")
        axes[1,1].set_title("Stage 5: Histogram Analysis")
        axes[1,1].legend(fontsize=8)
        axes[1,1].set_xlim([0,255])

        dft  = cv2.dft(gray_s3.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
        mag  = 20*np.log(cv2.magnitude(
            np.fft.fftshift(dft)[:,:,0],
            np.fft.fftshift(dft)[:,:,1]) + 1)
        axes[1,2].imshow(mag, cmap="hot")
        axes[1,2].set_title("Stage 6: DFT Magnitude Spectrum")

        for ax in axes.flat:
            ax.axis("off") if ax != axes[1,1] else None

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {output_path}")
