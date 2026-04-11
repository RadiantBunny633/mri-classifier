"""
CNN vs Random Forest Comparison Visualization
Loads saved Current CNN (3 conv) and from-scratch RF,
generates side-by-side per-class metrics + confusion matrix charts.
"""

import torch
import numpy as np
import pickle
import sys
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_dataset, split_dataset
from model_comparison import CurrentCNN
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rf'))
from rf.rf_model import flatten_images


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)

    # Load data
    print("Loading dataset...")
    images, labels, class_names = load_dataset("data")
    train_data, val_data, test_data = split_dataset(images, labels)
    X_test, y_test = test_data

    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)
    X_test_flat = flatten_images(X_test)

    # ===== Load CNN =====
    print("\nLoading Current CNN (3 conv)...")
    cnn_model = CurrentCNN(num_classes=len(class_names)).to(device)
    cnn_model.load_state_dict(torch.load("results/current_cnn_3_conv.pth", map_location=device))
    cnn_model.eval()

    with torch.no_grad():
        outputs = cnn_model(X_test_tensor.to(device))
        _, cnn_preds = torch.max(outputs, 1)
    cnn_preds = cnn_preds.cpu().numpy()

    cnn_acc = 100 * (cnn_preds == y_test).sum() / len(y_test)
    cnn_prec, cnn_rec, cnn_f1, _ = precision_recall_fscore_support(y_test, cnn_preds, average=None, zero_division=0)
    cnn_cm = confusion_matrix(y_test, cnn_preds)

    print(f"CNN Test Accuracy: {cnn_acc:.2f}%")
    print(classification_report(y_test, cnn_preds, target_names=class_names))

    # ===== Load RF =====
    print("Loading Random Forest...")
    with open("results/rf_standalone_model.pkl", "rb") as f:
        rf_model = pickle.load(f)

    rf_preds = rf_model.predict(X_test_flat)

    rf_acc = 100 * (rf_preds == y_test).sum() / len(y_test)
    rf_prec, rf_rec, rf_f1, _ = precision_recall_fscore_support(y_test, rf_preds, average=None, zero_division=0)
    rf_cm = confusion_matrix(y_test, rf_preds)

    print(f"RF Test Accuracy: {rf_acc:.2f}%")
    print(classification_report(y_test, rf_preds, target_names=class_names))

    # ===== Generate Comparison Visualization =====
    short_names = ['Mild\nDemented', 'Moderate\nDemented', 'Non\nDemented', 'Very Mild\nDemented']
    cm_labels = ['Mild', 'Moderate', 'Non', 'Very Mild']

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('CNN vs Random Forest — MRI Dementia Classification', fontsize=18, fontweight='bold')

    configs = [
        {
            'precision': cnn_prec, 'recall': cnn_rec, 'f1': cnn_f1,
            'cm': cnn_cm, 'acc': cnn_acc,
            'title': 'Current CNN (3 conv)', 'col': 0
        },
        {
            'precision': rf_prec, 'recall': rf_rec, 'f1': rf_f1,
            'cm': rf_cm, 'acc': rf_acc,
            'title': 'Random Forest (200 trees)', 'col': 1
        },
    ]

    for cfg in configs:
        col = cfg['col']
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        # Top: Per-class metrics
        x = np.arange(len(class_names))
        width = 0.25

        bars_p = ax_top.bar(x - width, cfg['precision'], width, label='Precision', color='#1a5276')
        bars_r = ax_top.bar(x, cfg['recall'], width, label='Recall', color='#e67e22')
        bars_f = ax_top.bar(x + width, cfg['f1'], width, label='F1-Score', color='#1e8449')

        for bars in [bars_p, bars_r, bars_f]:
            for bar in bars:
                height = bar.get_height()
                ax_top.text(bar.get_x() + bar.get_width()/2, height + 0.005,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax_top.set_title(f"{cfg['title']}\nTest Accuracy: {cfg['acc']:.2f}%", fontsize=13, fontweight='bold')
        ax_top.set_ylabel('Score' if col == 0 else '', fontsize=11)
        ax_top.set_xticks(x)
        ax_top.set_xticklabels(short_names, fontsize=10)
        ax_top.set_ylim(0, 1.15)
        ax_top.legend(loc='upper right', fontsize=9)
        ax_top.grid(axis='y', alpha=0.2)

        # Bottom: Confusion matrix
        im = ax_bot.imshow(cfg['cm'], cmap='Blues')
        ax_bot.set_xticks(range(len(cm_labels)))
        ax_bot.set_yticks(range(len(cm_labels)))
        ax_bot.set_xticklabels(cm_labels, fontsize=10)
        ax_bot.set_yticklabels(cm_labels, fontsize=10)
        ax_bot.set_xlabel('Predicted', fontsize=11)
        ax_bot.set_ylabel('Actual' if col == 0 else '', fontsize=11)

        for i in range(len(cm_labels)):
            for j in range(len(cm_labels)):
                color = 'white' if cfg['cm'][i, j] > cfg['cm'].max() / 2 else 'black'
                ax_bot.text(j, i, str(cfg['cm'][i, j]), ha='center', va='center',
                            color=color, fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/cnn_vs_rf_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: results/cnn_vs_rf_comparison.png")

    # ===== Print Summary =====
    print("\n" + "="*50)
    print("CNN vs RF COMPARISON")
    print("="*50)
    print(f"{'Model':<30} {'Test Acc':<15}")
    print("-"*45)
    print(f"{'Current CNN (3 conv)':<30} {cnn_acc:<15.2f}")
    print(f"{'Random Forest (200 trees)':<30} {rf_acc:<15.2f}")
    print("-"*45)

    # Save comparison JSON
    results = {
        'cnn': {
            'test_accuracy': float(cnn_acc),
            'precision': cnn_prec.tolist(),
            'recall': cnn_rec.tolist(),
            'f1': cnn_f1.tolist(),
            'cm': cnn_cm.tolist()
        },
        'rf': {
            'test_accuracy': float(rf_acc),
            'precision': rf_prec.tolist(),
            'recall': rf_rec.tolist(),
            'f1': rf_f1.tolist(),
            'cm': rf_cm.tolist()
        },
        'class_names': class_names
    }
    with open('results/cnn_vs_rf_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved: results/cnn_vs_rf_comparison.json")


if __name__ == "__main__":
    main()