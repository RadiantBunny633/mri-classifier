"""
RF — MRI Alzheimer's Classifier
=====================================================================
Loads data, flattens pixels, trains the Random Forest, and saves
the fitted model to results/.

Usage:
    python3 src/rf/rf_train.py
"""

import os
import sys
import pickle
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rf_data_loader import load_dataset, split_dataset
from rf_model import build_model, flatten_images, train_model


DATA_DIR     = "data"
RESULTS_DIR  = "results"
RF_SAVE_PATH = os.path.join(RESULTS_DIR, "rf_standalone_model.pkl")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading dataset...")
    images, labels, class_names = load_dataset(DATA_DIR)
    print(f"Loaded {len(images)} images | Classes: {class_names}")

    for i, name in enumerate(class_names):
        print(f"  {name}: {(labels == i).sum()} images")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(images, labels)
    print(f"\nSplit -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    print("\nFlattening images to pixel vectors...")
    X_train_flat = flatten_images(X_train)
    X_val_flat   = flatten_images(X_val)

    # Merge train + val to give the RF as much labeled data as possible.
    X_fit = np.vstack([X_train_flat, X_val_flat])
    y_fit = np.concatenate([y_train, y_val])
    print(f"Training feature matrix: {X_fit.shape}")

    print()
    rf = build_model()
    rf = train_model(rf, X_fit, y_fit)

    with open(RF_SAVE_PATH, 'wb') as f:
        pickle.dump(rf, f)
    print(f"\nModel saved: {RF_SAVE_PATH}")
    print("Run rf_evaluate.py to evaluate on the test set.")


if __name__ == "__main__":
    main()