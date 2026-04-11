"""
Visualizations comparing CNN architectures
Generates per-class metrics + confusion matrix for each model individually
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import json
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_dataset, split_dataset
from model_comparison import ShallowCNN, CurrentCNN, DeepCNN


def load_model(model_class, model_path, num_classes, device):
    """Load a saved model from disk"""
    model = model_class(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def get_predictions(model, X_test_tensor, device):
    """Run model on test data and return predictions"""
    with torch.no_grad():
        outputs = model(X_test_tensor.to(device))
        _, predictions = torch.max(outputs, 1)
    return predictions.cpu().numpy()


def plot_model_results(y_true, y_pred, class_names, model_name, save_name, test_accuracy):
    """
    Generate the combined per-class metrics + confusion matrix chart
    matching the style of the existing CNN results visualization.
    """
    # Short labels for display
    short_names = ['Mild\nDemented', 'Moderate\nDemented', 'Non\nDemented', 'Very Mild\nDemented']
    cm_labels = ['Mild', 'Moderate', 'Non', 'Very Mild']

    # Compute per-class precision, recall, f1
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{model_name} — MRI Dementia Classification', fontsize=16, fontweight='bold')

    # ===== LEFT: Per-Class Metrics Grouped Bar Chart =====
    x = np.arange(len(class_names))
    width = 0.25

    bars_p = ax1.bar(x - width, precision, width, label='Precision', color='#1a5276')
    bars_r = ax1.bar(x, recall, width, label='Recall', color='#e67e22')
    bars_f = ax1.bar(x + width, f1, width, label='F1-Score', color='#1e8449')

    # Add value labels
    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 0.005,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_title('Per-Class Metrics', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, fontsize=10)
    ax1.set_ylim(0, 1.12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(axis='y', alpha=0.2)

    # ===== RIGHT: Confusion Matrix =====
    im = ax2.imshow(cm, cmap='Blues')
    ax2.set_title(f'Confusion Matrix (Test Set, n={len(y_true)})', fontsize=13, fontweight='bold')

    ax2.set_xticks(range(len(cm_labels)))
    ax2.set_yticks(range(len(cm_labels)))
    ax2.set_xticklabels(cm_labels, fontsize=10)
    ax2.set_yticklabels(cm_labels, fontsize=10)
    ax2.set_xlabel('Predicted', fontsize=11)
    ax2.set_ylabel('Actual', fontsize=11)

    # Add values in cells
    for i in range(len(cm_labels)):
        for j in range(len(cm_labels)):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax2.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color=color, fontsize=13, fontweight='bold')

    fig.colorbar(im, ax=ax2, shrink=0.8)

    # Add overall accuracy annotation
    fig.text(0.5, -0.02, f'Overall Test Accuracy: {test_accuracy:.2f}%',
             ha='center', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'results/{save_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: results/{save_name}.png")


def plot_all_three_summary(all_results, class_names):
    """Generate a single side-by-side comparison of all three models"""
    short_names = ['Mild\nDemented', 'Moderate\nDemented', 'Non\nDemented', 'Very Mild\nDemented']
    cm_labels = ['Mild', 'Moderate', 'Non', 'Very Mild']

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle('CNN Architecture Comparison — MRI Dementia Classification', fontsize=18, fontweight='bold')

    for col, (model_name, data) in enumerate(all_results.items()):
        precision = data['precision']
        recall = data['recall']
        f1 = data['f1']
        cm = data['cm']
        test_acc = data['test_accuracy']

        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        # Top row: Per-class metrics
        x = np.arange(len(class_names))
        width = 0.25

        bars_p = ax_top.bar(x - width, precision, width, label='Precision', color='#1a5276')
        bars_r = ax_top.bar(x, recall, width, label='Recall', color='#e67e22')
        bars_f = ax_top.bar(x + width, f1, width, label='F1-Score', color='#1e8449')

        for bars in [bars_p, bars_r, bars_f]:
            for bar in bars:
                height = bar.get_height()
                ax_top.text(bar.get_x() + bar.get_width()/2, height + 0.005,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

        ax_top.set_title(f'{model_name}\nTest Accuracy: {test_acc:.2f}%', fontsize=12, fontweight='bold')
        ax_top.set_ylabel('Score' if col == 0 else '', fontsize=10)
        ax_top.set_xticks(x)
        ax_top.set_xticklabels(short_names, fontsize=8)
        ax_top.set_ylim(0, 1.15)
        if col == 2:
            ax_top.legend(loc='upper right', fontsize=8)
        ax_top.grid(axis='y', alpha=0.2)

        # Bottom row: Confusion matrix
        im = ax_bot.imshow(cm, cmap='Blues')
        ax_bot.set_xticks(range(len(cm_labels)))
        ax_bot.set_yticks(range(len(cm_labels)))
        ax_bot.set_xticklabels(cm_labels, fontsize=9)
        ax_bot.set_yticklabels(cm_labels, fontsize=9)
        ax_bot.set_xlabel('Predicted', fontsize=10)
        ax_bot.set_ylabel('Actual' if col == 0 else '', fontsize=10)

        for i in range(len(cm_labels)):
            for j in range(len(cm_labels)):
                color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                ax_bot.text(j, i, str(cm[i, j]), ha='center', va='center',
                            color=color, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/all_models_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/all_models_comparison.png")


def generate_all():
    """Load all three models and generate visualizations"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)

    # Load data
    print("Loading dataset...")
    images, labels, class_names = load_dataset("data")
    train_data, val_data, test_data = split_dataset(images, labels)
    X_test, y_test = test_data

    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)
    num_classes = len(class_names)

    # Define model configs
    models_config = [
        {
            'class': ShallowCNN,
            'path': 'results/shallow_cnn_2_conv.pth',
            'name': 'Shallow CNN (2 conv)',
            'save_name': 'shallow_cnn_results'
        },
        {
            'class': CurrentCNN,
            'path': 'results/current_cnn_3_conv.pth',
            'name': 'Current CNN (3 conv)',
            'save_name': 'current_cnn_results'
        },
        {
            'class': DeepCNN,
            'path': 'results/deep_cnn_4_conv.pth',
            'name': 'Deep CNN (4 conv)',
            'save_name': 'deep_cnn_results'
        },
    ]

    all_results = {}

    for config in models_config:
        print(f"\nEvaluating {config['name']}...")

        # Load model
        model = load_model(config['class'], config['path'], num_classes, device)
        predictions = get_predictions(model, X_test_tensor, device)

        # Compute metrics
        test_accuracy = 100 * (predictions == y_test).sum() / len(y_test)
        precision, recall, f1, support = precision_recall_fscore_support(y_test, predictions, average=None, zero_division=0)
        cm = confusion_matrix(y_test, predictions)

        print(f"  Test Accuracy: {test_accuracy:.2f}%")
        print(classification_report(y_test, predictions, target_names=class_names))

        # Generate individual chart
        plot_model_results(y_test, predictions, class_names, config['name'], config['save_name'], test_accuracy)

        # Store for combined chart
        all_results[config['name']] = {
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cm': cm
        }

    # Generate combined 3-across comparison
    plot_all_three_summary(all_results, class_names)

    print("\nAll comparison visualizations saved to results/")


if __name__ == "__main__":
    generate_all()