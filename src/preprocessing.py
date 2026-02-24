"""
MRI Preprocessing Pipeline
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_mri_image(filepath):
    """Load a single MRI image."""
    img = Image.open(filepath).convert('L')  # Convert to grayscale
    return np.array(img)


def normalize_intensity(image):
    """Normalize pixel intensities to 0-1 range."""
    img_min = image.min()
    img_max = image.max()
    if img_max - img_min == 0:
        return image
    return (image - img_min) / (img_max - img_min)


def resize_image(image, target_size=(128, 128)):
    """Resize image to consistent dimensions."""
    img = Image.fromarray(image.astype(np.uint8))
    img_resized = img.resize(target_size)
    return np.array(img_resized)


def preprocess_mri(filepath, target_size=(128, 128)):
    """Full preprocessing pipeline for a single MRI image."""
    img = load_mri_image(filepath)
    img = resize_image(img, target_size)
    img = normalize_intensity(img)
    return img


if __name__ == "__main__":
    # Test on sample images
    test_images = [
        "data/mild_99.jpg",
        "data/moderate_8.jpg",
        "data/non_999.jpg",
        "data/verymild_999.jpg"
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            original = load_mri_image(img_path)
            processed = preprocess_mri(img_path)
            print(f"{img_path}:")
            print(f"  Original shape: {original.shape}")
            print(f"  Processed shape: {processed.shape}")
            print(f"  Processed range: [{processed.min():.3f}, {processed.max():.3f}]")
            print()