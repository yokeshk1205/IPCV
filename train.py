# train.py — Full Training Pipeline: Feature Extraction → PCA → GBR
# Run AFTER: python download_model.py && python generate_labels.py
# Usage: python train.py

import os
import numpy as np
import joblib
import csv

from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    LABELS_CSV, MODEL_DIR, OUTPUT_DIR,
    PCA_COMPONENTS, GBR_PARAMS, RANDOM_SEED,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    PCA_PATH, GBR_PATH,
)
from preprocess import SmokePreprocessor
from feature_extraction import VGG16Extractor, build_feature_dataset


def train():
    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Load extractor + preprocessor ───────────────────────────────
    print("\n[STEP 1] Initialising VGG16 Extractor & Preprocessor...")
    extractor    = VGG16Extractor()
    preprocessor = SmokePreprocessor(save_debug=False)

    # ── Step 2: Extract features from all images ─────────────────────────────
    print("\n[STEP 2] Extracting VGG16 fc6 features from dataset...")
    X, y, paths = build_feature_dataset(extractor, preprocessor, LABELS_CSV)
    print(f"  Feature matrix: {X.shape}   |   Labels: {y.shape}")

    # ── Step 3: Train/Val/Test Split ─────────────────────────────────────────
    print("\n[STEP 3] Splitting dataset (70/15/15)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(VAL_RATIO + TEST_RATIO), random_state=RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=RANDOM_SEED
    )
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── Step 4: PCA Dimensionality Reduction ─────────────────────────────────
    print(f"\n[STEP 4] Fitting PCA: 4096 → {PCA_COMPONENTS} dims...")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca   = pca.transform(X_val)
    X_test_pca  = pca.transform(X_test)
    variance = pca.explained_variance_ratio_.sum()
    print(f"  Variance retained: {variance:.4f} ({variance*100:.1f}%)")
    joblib.dump(pca, PCA_PATH)
    print(f"  PCA model saved → {PCA_PATH}")

    # ── Step 5: Train Gradient Boosting Regressor ─────────────────────────────
    print(f"\n[STEP 5] Training GradientBoostingRegressor...")
    print(f"  Params: {GBR_PARAMS}")
    gbr = GradientBoostingRegressor(**GBR_PARAMS)
    gbr.fit(X_train_pca, y_train)
    joblib.dump(gbr, GBR_PATH)
    print(f"  GBR model saved → {GBR_PATH}")

    # ── Step 6: Evaluate on all splits ───────────────────────────────────────
    print("\n[STEP 6] Evaluating on Train / Val / Test sets...")

    results = {}
    for split_name, X_s, y_s in [
        ("Train", X_train_pca, y_train),
        ("Val",   X_val_pca,   y_val),
        ("Test",  X_test_pca,  y_test),
    ]:
        y_pred = gbr.predict(X_s)
        mae  = mean_absolute_error(y_s, y_pred)
        rmse = np.sqrt(mean_squared_error(y_s, y_pred))
        r2   = r2_score(y_s, y_pred)
        results[split_name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
        print(f"  {split_name:5s} → MAE: {mae:.4f}  RMSE: {rmse:.4f}  R²: {r2:.4f}")

    # ── Step 7: Save results summary ──────────────────────────────────────────
    results_path = os.path.join(OUTPUT_DIR, "training_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("Smoke Density Estimation — Training Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset      : {LABELS_CSV}\n")
        f.write(f"Total images : {len(X)}\n")
        f.write(f"Feature dim  : 4096 → {PCA_COMPONENTS} (PCA)\n")
        f.write(f"PCA variance : {variance:.4f}\n\n")
        f.write(f"Regressor    : GradientBoostingRegressor\n")
        f.write(f"Params       : {GBR_PARAMS}\n\n")
        for split_name, metrics in results.items():
            f.write(f"{split_name:5s}  MAE={metrics['MAE']:.4f}  "
                    f"RMSE={metrics['RMSE']:.4f}  R²={metrics['R2']:.4f}\n")

    print(f"\n  Results saved → {results_path}")

    # ── Step 8: Save test predictions for evaluate.py ─────────────────────────
    preds_path = os.path.join(OUTPUT_DIR, "test_predictions.npz")
    np.savez(preds_path, y_true=y_test, y_pred=gbr.predict(X_test_pca))
    print(f"  Test predictions saved → {preds_path}")

    print("\n✓ Training complete. Run: python evaluate.py")


if __name__ == "__main__":
    train()
