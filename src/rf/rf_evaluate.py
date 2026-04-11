"""
RF — MRI Alzheimer's Classifier 
=======================================================================
Loads the saved RF model and the held-out test set, then prints
metrics and saves result files to results/.

Usage:
    python3 src/rf_evaluate.py
"""

import os
import sys
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rf_data_loader import load_dataset, split_dataset
from rf_model import flatten_images, IMAGE_SIZE


DATA_DIR     = "data"
RESULTS_DIR  = "results"
RF_SAVE_PATH = os.path.join(RESULTS_DIR, "rf_standalone_model.joblib")


def print_metrics(y_test, y_pred, y_proba, class_names):
    accuracy = (y_pred == y_test).sum() / len(y_test) * 100
    print(f"\n{'='*50}")
    print(f"TEST SET ACCURACY: {accuracy:.2f}%")
    print(f"{'='*50}")

    print("\nPER-CLASS METRICS:")
    print("-" * 50)
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)

    report_path = os.path.join(RESULTS_DIR, "rf_standalone_class_report.txt")
    with open(report_path, "w") as f:
        f.write(f"TEST SET ACCURACY: {accuracy:.2f}%\n\n")
        f.write(report)
    print(f"Saved: {report_path}")

    y_test_bin = label_binarize(y_test, classes=list(range(len(class_names))))
    try:
        auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="macro")
        print(f"Macro AUC (OvR): {auc:.4f}")
    except ValueError as e:
        print(f"AUC skipped: {e}")

    print("\nCONFUSION MATRIX:")
    print("-" * 50)
    cm = confusion_matrix(y_test, y_pred)
    print(f"{'':>20}", end="")
    for name in class_names:
        print(f"{name[:10]:>12}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>20}", end="")
        for val in row:
            print(f"{val:>12}", end="")
        print()
    print("\n(Rows = actual, Columns = predicted)")

    return accuracy, cm


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")
    ax.set_title("RF Standalone — Confusion Matrix (Test Set)", fontsize=13, pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "rf_standalone_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_per_class_accuracy(y_test, y_pred, class_names):
    accuracies = []
    for i in range(len(class_names)):
        mask = y_test == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_test[mask]).sum() / mask.sum() * 100
        else:
            acc = 0
        accuracies.append(acc)

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(class_names, accuracies, color=colors)
    ax.set_title("RF Standalone — Per-Class Accuracy on Test Set", fontsize=14)
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)

    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.1f}%",
            ha="center",
            fontsize=12,
        )

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "rf_standalone_per_class_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_pixel_importance_heatmap(rf):
    importances    = rf.feature_importances_                         # (16384,)
    importance_map = importances.reshape(IMAGE_SIZE, IMAGE_SIZE)     # (128, 128)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(importance_map, cmap="hot", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Gini Importance")
    ax.set_title(
        "RF Pixel Importance Heatmap\n(brighter = more discriminative)",
        fontsize=11,
    )
    ax.axis("off")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "rf_standalone_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def evaluate_model():
    # ── load RF ───────────────────────────────────────────────────────────────
    if not os.path.exists(RF_SAVE_PATH):
        print(f"No saved model found at '{RF_SAVE_PATH}'.")
        print("Run rf_train.py first.")
        return
    rf = joblib.load(RF_SAVE_PATH)
    print(f"Loaded RF model from '{RF_SAVE_PATH}'")

    print("Loading dataset...")
    images, labels, class_names = load_dataset(DATA_DIR)
    _, _, (X_test, y_test) = split_dataset(images, labels)
    print(f"Test samples: {len(X_test)}")

    X_test_flat = flatten_images(X_test)

    print("\nRunning predictions on test set...")
    y_pred  = rf.predict(X_test_flat)
    y_proba = rf.predict_proba(X_test_flat)

    accuracy, cm = print_metrics(y_test, y_pred, y_proba, class_names)

    print("\nGenerating visualizations...")
    plot_confusion_matrix(cm, class_names)
    plot_per_class_accuracy(y_test, y_pred, class_names)
    plot_pixel_importance_heatmap(rf)

    print("\nAll outputs saved to results/")
    return accuracy, class_names, y_pred, y_test


if __name__ == "__main__":
    evaluate_model()