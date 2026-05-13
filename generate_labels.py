# generate_labels.py — Scans dataset/ folders and creates labels.csv
# SDI values are sampled uniformly within each class range.
# Run this ONCE before training.

import os
import csv
import random

DATASET_DIR = "dataset"
OUTPUT_CSV  = os.path.join(DATASET_DIR, "labels.csv")
RANDOM_SEED = 42

CLASSES = ["good", "moderate", "unhealthy", "very_unhealthy", "hazardous"]
CLASS_SDI_RANGE = {
    "good":          (0.00, 0.17),
    "moderate":      (0.17, 0.33),
    "unhealthy":     (0.33, 0.50),
    "very_unhealthy":(0.50, 0.67),
    "hazardous":     (0.67, 1.00),
}
SDI_TO_PM25 = 300   # PM2.5 (µg/m³) = SDI × 300
VALID_EXTS  = {".jpg", ".jpeg", ".png", ".bmp"}

random.seed(RANDOM_SEED)

rows = []
for cls in CLASSES:
    folder = os.path.join(DATASET_DIR, cls)
    if not os.path.isdir(folder):
        print(f"[WARN] Folder not found: {folder} — skipping")
        continue

    files = [f for f in os.listdir(folder)
             if os.path.splitext(f)[1].lower() in VALID_EXTS]

    lo, hi = CLASS_SDI_RANGE[cls]
    for fname in files:
        sdi  = round(random.uniform(lo, hi), 4)
        pm25 = round(sdi * SDI_TO_PM25, 2)
        rows.append({
            "filepath": os.path.join(folder, fname),
            "class":    cls,
            "sdi":      sdi,
            "pm25":     pm25,
        })

    print(f"  [{cls:>15}]  {len(files):>5} images   SDI: {lo:.2f}–{hi:.2f}")

random.shuffle(rows)

os.makedirs(DATASET_DIR, exist_ok=True)
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["filepath","class","sdi","pm25"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\n✓ labels.csv written → {OUTPUT_CSV}  ({len(rows)} rows)")
