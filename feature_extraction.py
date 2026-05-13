# feature_extraction.py — VGG16 Feature Extractor via cv2.dnn
# Loads frozen Caffe model, runs forward pass, extracts fc6 (4096-dim) vector.

import cv2
import numpy as np
import os
from config import PROTOTXT, CAFFEMODEL, IMG_SIZE, VGG_MEAN_BGR, VGG_LAYER


class VGG16Extractor:
    """Loads VGG16 Caffe model via cv2.dnn and extracts fc6 feature vectors."""

    def __init__(self):
        if not os.path.exists(PROTOTXT):
            raise FileNotFoundError(
                f"prototxt not found: {PROTOTXT}\n"
                "  → Run: python download_model.py"
            )
        if not os.path.exists(CAFFEMODEL):
            raise FileNotFoundError(
                f"caffemodel not found: {CAFFEMODEL}\n"
                "  → Run: python download_model.py"
            )

        print(f"[VGG16] Loading model from {CAFFEMODEL} ...")
        self.net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("[VGG16] Model loaded. Feature layer: fc6 → 4096 dims")

    def extract(self, img_preprocessed):
        """
        Extract 4096-dim fc6 feature vector from a single preprocessed image.

        Args:
            img_preprocessed (np.ndarray): BGR uint8 image, already resized to 224×224.

        Returns:
            features (np.ndarray): Shape (4096,), float32
        """
        blob = cv2.dnn.blobFromImage(
            img_preprocessed,
            scalefactor=1.0,
            size=IMG_SIZE,
            mean=VGG_MEAN_BGR,
            swapRB=False,
            crop=False,
        )
        self.net.setInput(blob)
        features = self.net.forward(VGG_LAYER)      # Shape: (1, 4096, 1, 1) or (1, 4096)
        return features.flatten().astype(np.float32)  # Always returns (4096,)

    def extract_batch(self, image_list):
        """
        Extract features from a list of preprocessed images.

        Args:
            image_list (list): List of BGR uint8 numpy arrays (224×224).

        Returns:
            X (np.ndarray): Shape (N, 4096), float32
        """
        X = []
        total = len(image_list)
        for i, img in enumerate(image_list):
            feat = self.extract(img)
            X.append(feat)
            if (i + 1) % 100 == 0 or (i + 1) == total:
                print(f"  Extracted {i+1}/{total} images", end="\r")
        print()
        return np.array(X, dtype=np.float32)


def build_feature_dataset(extractor, preprocessor, labels_csv):
    """
    Read labels.csv, preprocess each image, extract VGG16 features.

    Returns:
        X (np.ndarray): Feature matrix (N, 4096)
        y (np.ndarray): SDI labels (N,)
        paths (list):   Image file paths for reference
    """
    import csv

    rows = []
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    X, y, paths = [], [], []
    total = len(rows)
    failed = 0

    for i, row in enumerate(rows):
        img = cv2.imread(row["filepath"])
        if img is None:
            failed += 1
            continue

        img_proc, _, _ = preprocessor.run(img)
        feat = extractor.extract(img_proc)
        X.append(feat)
        y.append(float(row["sdi"]))
        paths.append(row["filepath"])

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  Processing {i+1}/{total} | failed: {failed}", end="\r")

    print(f"\n[INFO] Feature extraction complete. Valid: {len(X)}, Failed: {failed}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), paths
