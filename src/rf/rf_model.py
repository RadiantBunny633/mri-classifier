"""
RF — MRI Alzheimer's Classifier 
===========================================================
Defines and trains the Random Forest classifier on flattened
raw pixel vectors.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier


IMAGE_SIZE = 128   # must match preprocessing target_size


def flatten_images(images: np.ndarray) -> np.ndarray:
    """
    Convert (N, H, W) image array to (N, H*W) feature matrix.
    Pixel values are already normalized to [0, 1] by preprocessing.py.

    For 128x128 images this produces 16,384 features per sample.
    """
    return images.reshape(len(images), -1)


def build_model() -> RandomForestClassifier:
    """
    Construct the Random Forest classifier.
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_features="sqrt",
        min_samples_leaf=4,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )


def train_model(
    rf: RandomForestClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RandomForestClassifier:
    """
    Fit the RF on flattened pixel vectors.
    """
    print(f"Training Random Forest...")
    print(f"  Samples:  {X_train.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")
    rf.fit(X_train, y_train)
    print("Training complete.")
    return rf


if __name__ == "__main__":
    X_dummy = np.random.rand(20, IMAGE_SIZE * IMAGE_SIZE).astype(np.float32)
    y_dummy = np.random.randint(0, 4, size=20)

    rf = build_model()
    rf = train_model(rf, X_dummy, y_dummy)
    preds = rf.predict(X_dummy)

    print(f"\nSmoke test passed.")
    print(f"Input shape:  {X_dummy.shape}")
    print(f"Output shape: {preds.shape}")
    print(f"Classes seen: {np.unique(preds)}")