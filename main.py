# main.py — Real-Time Smoke Density Estimation (Webcam / Video)
# Usage:
#   python main.py                   → webcam (device 0)
#   python main.py --source video.mp4  → video file
#   python main.py --source image.jpg  → single image
#   python main.py --demo              → run on sample images from dataset/

import cv2
import numpy as np
import joblib
import argparse
import os
import time

from config import (
    PCA_PATH, GBR_PATH, OUTPUT_DIR, SDI_TO_PM25_SCALE, AQI_INFO
)
from preprocess import SmokePreprocessor
from feature_extraction import VGG16Extractor


# ── AQI Colour Mapping ────────────────────────────────────────────────────────
AQI_DISPLAY = {
    "good":           {"label": "GOOD",          "color": (0, 200, 0)},
    "moderate":       {"label": "MODERATE",       "color": (0, 200, 255)},
    "unhealthy":      {"label": "UNHEALTHY",      "color": (0, 140, 255)},
    "very_unhealthy": {"label": "VERY UNHEALTHY", "color": (0, 60, 255)},
    "hazardous":      {"label": "HAZARDOUS",      "color": (0, 0, 200)},
}


def sdi_to_aqi(sdi):
    """Convert SDI (0–1) to AQI category string."""
    if   sdi < 0.17: return "good"
    elif sdi < 0.33: return "moderate"
    elif sdi < 0.50: return "unhealthy"
    elif sdi < 0.67: return "very_unhealthy"
    else:            return "hazardous"


class SmokeEstimator:
    """End-to-end inference engine: image → SDI + AQI label."""

    def __init__(self):
        print("[INIT] Loading models...")
        self.preprocessor = SmokePreprocessor()
        self.extractor    = VGG16Extractor()
        self.pca          = joblib.load(PCA_PATH)
        self.gbr          = joblib.load(GBR_PATH)
        print("[INIT] All models loaded.")

    def predict(self, frame):
        """
        Args:
            frame (np.ndarray): Raw BGR image from camera or file.
        Returns:
            sdi   (float)  : Smoke Density Index ∈ [0, 1]
            pm25  (float)  : Estimated PM2.5 (µg/m³)
            aqi   (str)    : AQI category string
            t_ms  (float)  : Inference time in milliseconds
        """
        t0 = time.time()

        img_proc, _, _ = self.preprocessor.run(frame)
        feat     = self.extractor.extract(img_proc)
        feat_pca = self.pca.transform(feat.reshape(1, -1))
        sdi      = float(np.clip(self.gbr.predict(feat_pca)[0], 0.0, 1.0))
        pm25     = sdi * SDI_TO_PM25_SCALE
        aqi      = sdi_to_aqi(sdi)
        t_ms     = (time.time() - t0) * 1000

        return sdi, pm25, aqi, t_ms

    def annotate_frame(self, frame, sdi, pm25, aqi, t_ms):
        """Overlay AQI banner, SDI bar, and metrics on frame."""
        h, w = frame.shape[:2]
        info  = AQI_DISPLAY[aqi]
        color = info["color"]
        label = info["label"]

        # ── Top banner ──
        cv2.rectangle(frame, (0, 0), (w, 60), color, -1)
        cv2.putText(frame,
            f"AQI: {label}   |   SDI: {sdi:.3f}   |   PM2.5: {pm25:.1f} ug/m3",
            (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2
        )

        # ── SDI progress bar ──
        bar_x, bar_y, bar_w, bar_h = 10, h - 35, int(w * 0.6), 18
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h),
                      (50,50,50), -1)
        filled = int(bar_w * sdi)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+filled, bar_y+bar_h),
                      color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h),
                      (200,200,200), 1)
        cv2.putText(frame, f"SDI: {sdi:.2f}",
            (bar_x + bar_w + 10, bar_y + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1
        )

        # ── FPS / latency ──
        fps = 1000 / t_ms if t_ms > 0 else 0
        cv2.putText(frame, f"{fps:.1f} FPS  |  {t_ms:.0f}ms",
            (w - 160, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (200, 200, 200), 1
        )
        return frame


def run_webcam(estimator, source=0):
    """Run real-time inference on webcam or video file."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    print(f"[INFERENCE] Starting stream from {source}. Press Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        sdi, pm25, aqi, t_ms = estimator.predict(frame)
        frame = estimator.annotate_frame(frame, sdi, pm25, aqi, t_ms)
        cv2.imshow("Smoke Density Estimator — Air Quality Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_single_image(estimator, image_path):
    """Run inference on a single image and save annotated result."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    sdi, pm25, aqi, t_ms = estimator.predict(img)
    img = estimator.annotate_frame(img, sdi, pm25, aqi, t_ms)

    name   = os.path.splitext(os.path.basename(image_path))[0]
    outpath = os.path.join(OUTPUT_DIR, f"{name}_result.jpg")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv2.imwrite(outpath, img)

    print(f"[RESULT] SDI={sdi:.3f}  PM2.5={pm25:.1f}  AQI={aqi.upper()}")
    print(f"  Saved → {outpath}")
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_demo(estimator, dataset_dir="dataset"):
    """Run on one sample image from each class, display 5-panel grid."""
    import matplotlib.pyplot as plt
    classes = ["good","moderate","unhealthy","very_unhealthy","hazardous"]
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle("Smoke Density Estimation — Demo Output (One per AQI Class)",
                 fontsize=13, fontweight="bold")

    for ax, cls in zip(axes, classes):
        folder = os.path.join(dataset_dir, cls)
        imgs   = [f for f in os.listdir(folder)
                  if f.lower().endswith((".jpg",".jpeg",".png"))]
        if not imgs:
            ax.set_title(f"{cls}\n(no images)")
            ax.axis("off")
            continue

        img_path = os.path.join(folder, imgs[0])
        img      = cv2.imread(img_path)
        sdi, pm25, aqi, t_ms = estimator.predict(img)
        img = estimator.annotate_frame(img.copy(), sdi, pm25, aqi, t_ms)

        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{aqi.upper()}\nSDI={sdi:.3f}  PM2.5={pm25:.0f}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "demo_grid.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Demo grid saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smoke Density Estimator — VGG16 + sklearn GBR"
    )
    parser.add_argument("--source", type=str, default="0",
                        help="Video source: 0=webcam, path=video/image file")
    parser.add_argument("--demo",   action="store_true",
                        help="Run demo mode on sample dataset images")
    args = parser.parse_args()

    estimator = SmokeEstimator()

    if args.demo:
        run_demo(estimator)
    elif args.source.isdigit():
        run_webcam(estimator, int(args.source))
    elif args.source.lower().endswith((".jpg",".jpeg",".png",".bmp")):
        run_single_image(estimator, args.source)
    else:
        run_webcam(estimator, args.source)   # video file
