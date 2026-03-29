"""
Evaluation script for MRI Classifier
"""

import torch
import numpy as np
import sys
import os
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_dataset, split_dataset
from model import MRIClassifier


def evaluate_model():
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading dataset...")
    images, labels, class_names = load_dataset("data")
    
    # Split data - we want the TEST set (the 960 we never touched)
    train_data, val_data, test_data = split_dataset(images, labels)
    X_test, y_test = test_data
    
    # Convert to PyTorch tensors
    X_test = torch.FloatTensor(X_test).unsqueeze(1)
    y_test_tensor = torch.LongTensor(y_test)
    
    print(f"Test samples: {len(X_test)}")
    
    # Load trained model
    model = MRIClassifier(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load("results/best_model.pth", map_location=device))
    model.eval()
    print("Loaded trained model from results/best_model.pth")
    
    # Get predictions
    print("\nRunning predictions on test set...")
    with torch.no_grad():
        X_test = X_test.to(device)
        outputs = model(X_test)
        _, predictions = torch.max(outputs, 1)
        predictions = predictions.cpu().numpy()
    
    # Overall accuracy
    accuracy = (predictions == y_test).sum() / len(y_test) * 100
    print(f"\n{'='*50}")
    print(f"TEST SET ACCURACY: {accuracy:.2f}%")
    print(f"{'='*50}")
    
    # Per-class metrics
    print("\nPER-CLASS METRICS:")
    print("-" * 50)
    print(classification_report(y_test, predictions, target_names=class_names))
    
    # Confusion matrix
    print("CONFUSION MATRIX:")
    print("-" * 50)
    cm = confusion_matrix(y_test, predictions)
    
    # Print with labels
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
    
    return accuracy, class_names, predictions, y_test


if __name__ == "__main__":
    evaluate_model()