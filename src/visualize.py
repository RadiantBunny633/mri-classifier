"""
Visualizations for MRI Classifier
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_dataset, split_dataset
from model import MRIClassifier


def plot_class_distribution(labels, class_names):
    """Show how many images per class"""
    counts = [np.sum(labels == i) for i in range(len(class_names))]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, counts, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
    plt.title('Dataset Class Distribution', fontsize=14)
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                 str(count), ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/class_distribution.png')
    plt.close()
    print("Saved: results/class_distribution.png")


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Show what the model confuses"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - Test Set', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    plt.close()
    print("Saved: results/confusion_matrix.png")


def plot_per_class_accuracy(y_true, y_pred, class_names):
    """Bar chart of accuracy per class"""
    accuracies = []
    for i in range(len(class_names)):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_true[mask]).sum() / mask.sum() * 100
            accuracies.append(acc)
        else:
            accuracies.append(0)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, accuracies, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
    plt.title('Per-Class Accuracy on Test Set', fontsize=14)
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105)
    
    # Add percentage labels
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 f'{acc:.1f}%', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/per_class_accuracy.png')
    plt.close()
    print("Saved: results/per_class_accuracy.png")


def plot_sample_predictions(model, X_test, y_true, y_pred, class_names, device):
    """Show sample images with predictions"""
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    fig.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', fontsize=14)
    
    # Get some correct and some incorrect predictions
    correct_mask = y_pred == y_true
    incorrect_mask = ~correct_mask
    
    correct_indices = np.where(correct_mask)[0][:8]
    incorrect_indices = np.where(incorrect_mask)[0][:4]
    
    all_indices = list(correct_indices) + list(incorrect_indices)
    np.random.shuffle(all_indices)
    all_indices = all_indices[:12]
    
    for idx, ax in zip(all_indices, axes.flat):
        img = X_test[idx].squeeze()
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        
        ax.imshow(img, cmap='gray')
        
        color = 'green' if y_true[idx] == y_pred[idx] else 'red'
        ax.set_title(f'True: {true_label}\nPred: {pred_label}', fontsize=9, color=color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/sample_predictions.png')
    plt.close()
    print("Saved: results/sample_predictions.png")


def generate_all_visualizations():
    """Generate all visualizations"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    print("Loading data...")
    images, labels, class_names = load_dataset("data")
    train_data, val_data, test_data = split_dataset(images, labels)
    X_test, y_test = test_data
    
    # Plot class distribution
    print("\nGenerating visualizations...")
    plot_class_distribution(labels, class_names)
    
    # Load model and get predictions
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)
    model = MRIClassifier(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load("results/best_model.pth", map_location=device))
    model.eval()
    
    with torch.no_grad():
        outputs = model(X_test_tensor.to(device))
        _, predictions = torch.max(outputs, 1)
        predictions = predictions.cpu().numpy()
    
    # Generate plots
    plot_confusion_matrix(y_test, predictions, class_names)
    plot_per_class_accuracy(y_test, predictions, class_names)
    plot_sample_predictions(model, X_test, y_test, predictions, class_names, device)
    
    print("\nAll visualizations saved to results/ folder!")


if __name__ == "__main__":
    generate_all_visualizations()