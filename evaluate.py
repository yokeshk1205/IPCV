# evaluate.py — Metrics, Confusion Matrix, Scatter Plot, Feature Bar Chart
# Run AFTER: python train.py
# Usage: python evaluate.py

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report
)
from config import OUTPUT_DIR, CLASSES

os.makedirs(OUTPUT_DIR, exist_ok=True)


def sdi_to_class(sdi):
    """Convert continuous SDI → 5-class AQI index."""
    if   sdi < 0.17: return 0  # Good
    elif sdi < 0.33: return 1  # Moderate
    elif sdi < 0.50: return 2  # Unhealthy
    elif sdi < 0.67: return 3  # Very Unhealthy
    else:            return 4  # Hazardous


CLASS_LABELS  = ["Good", "Moderate", "Unhealthy", "Very\nUnhealthy", "Hazardous"]
PALETTE_COLORS= ["#22c55e", "#eab308", "#f97316", "#ef4444", "#7f1d1d"]


def plot_scatter(y_true, y_pred, output_path):
    """Scatter plot: Predicted vs. Actual SDI with regression line."""
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.35, s=25,
               color="#3b82f6", edgecolors="#1e40af", linewidths=0.4)
    ax.plot([0, 1], [0, 1], "r--", lw=1.5, label="Perfect Fit")
    ax.set_xlabel("Actual SDI", fontsize=11)
    ax.set_ylabel("Predicted SDI", fontsize=11)
    ax.set_title(
        f"Predicted vs Actual SDI  —  VGG16 + GBR\n"
        f"R² = {r2:.3f}   MAE = {mae:.3f}   RMSE = {rmse:.3f}",
        fontsize=11
    )
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")
    return r2, mae, rmse


def plot_confusion_matrix(y_true, y_pred, output_path):
    """Seaborn heatmap of 5-class AQI confusion matrix."""
    y_true_cls = [sdi_to_class(s) for s in y_true]
    y_pred_cls = [sdi_to_class(s) for s in y_pred]

    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=list(range(5)))
    acc = np.trace(cm) / cm.sum()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Good","Moderate","Unhealthy","V.Unhlthy","Hazardous"],
        yticklabels=["Good","Moderate","Unhealthy","V.Unhlthy","Hazardous"],
        ax=ax
    )
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("Actual Class",    fontsize=11)
    ax.set_title(
        f"Confusion Matrix — AQI 5-Class  (Accuracy = {acc*100:.1f}%)",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

    # Print classification report
    labels_full = ["Good","Moderate","Unhealthy","V.Unhlthy","Hazardous"]
    print("\n" + classification_report(y_true_cls, y_pred_cls,
                                        target_names=labels_full))
    return acc


def plot_class_distribution(y_true, output_path):
    """Bar chart of actual class distribution in the test set."""
    classes = [sdi_to_class(s) for s in y_true]
    counts  = [classes.count(i) for i in range(5)]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(CLASS_LABELS, counts, color=PALETTE_COLORS, edgecolor="black",
                  linewidth=0.6)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(count), ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("AQI Category", fontsize=11)
    ax.set_ylabel("Image Count",  fontsize=11)
    ax.set_title("Test Set Class Distribution — AQI Categories", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def print_metrics_table(r2, mae, rmse, acc):
    """Print a clean table of all metrics to console."""
    print("\n" + "=" * 50)
    print(" EVALUATION RESULTS — VGG16 + sklearn GBR")
    print("=" * 50)
    print(f"  MAE  (Smoke Density Index) : {mae:.4f}  (~{mae*300:.1f} µg/m³)")
    print(f"  RMSE (Smoke Density Index) : {rmse:.4f}  (~{rmse*300:.1f} µg/m³)")
    print(f"  R²   Score                 : {r2:.4f}")
    print(f"  AQI Category Accuracy      : {acc*100:.1f}%")
    print("=" * 50)


def run_evaluation():
    preds_path = os.path.join(OUTPUT_DIR, "test_predictions.npz")
    if not os.path.exists(preds_path):
        raise FileNotFoundError(
            f"{preds_path} not found. Run: python train.py first."
        )

    data   = np.load(preds_path)
    y_true = data["y_true"]
    y_pred = data["y_pred"]

    print(f"[EVAL] Loaded {len(y_true)} test predictions.")

    r2, mae, rmse = plot_scatter(
        y_true, y_pred,
        os.path.join(OUTPUT_DIR, "scatter_plot.png")
    )
    acc = plot_confusion_matrix(
        y_true, y_pred,
        os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    )
    plot_class_distribution(
        y_true,
        os.path.join(OUTPUT_DIR, "class_distribution.png")
    )
    print_metrics_table(r2, mae, rmse, acc)
    print("\n✓ Evaluation complete. Check outputs/ folder.")


if __name__ == "__main__":
    run_evaluation()
