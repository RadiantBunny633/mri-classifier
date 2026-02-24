import os
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_mri


def load_dataset(data_dir, target_size=(128, 128)):
    """
    Load all images from a directory organized by class.
    Expects structure: data_dir/class_name/image.jpg
    """
    images = []
    labels = []
    class_names = []
    
    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        class_names.append(class_name)
        class_idx = len(class_names) - 1
        
        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(class_path, img_name)
            img = preprocess_mri(img_path, target_size)
            images.append(img)
            labels.append(class_idx)
    
    return np.array(images), np.array(labels), class_names


def split_dataset(images, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train, validation, and test sets.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, 
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=42
    )
    
    val_size = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_size),
        stratify=y_temp,
        random_state=42
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":
    images, labels, class_names = load_dataset("data")
    
    print(f"Loaded {len(images)} images")
    print(f"Classes: {class_names}")
    print(f"Image shape: {images[0].shape}")
    
    for i, name in enumerate(class_names):
        count = (labels == i).sum()
        print(f"  {name}: {count} images")
    
    train, val, test = split_dataset(images, labels)
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train[0])}")
    print(f"  Validation: {len(val[0])}")
    print(f"  Test: {len(test[0])}")