"""
Compare CNN with vs without Dropout
Retrains both models to capture epoch-by-epoch training curves
Generates training curve plots + per-class metrics comparison
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_dataset, split_dataset
from model_comparison import CurrentCNN


class NoDropoutCNN(nn.Module):
    """3 convolutional layers, NO dropout"""
    def __init__(self, num_classes=4, input_size=128):
        super(NoDropoutCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        conv_output_size = 128 * (input_size // 8) * (input_size // 8)

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def train_model_with_history(model, model_name, train_loader, val_loader, device, num_epochs=20):
    """Train a model and return best state + per-epoch history"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_accuracy = 0.0
    best_model_state = None

    history = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': []
    }

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0.0

        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()

        train_accuracy = 100 * train_correct / train_total
        avg_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_images, batch_labels in val_loader:
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()

        val_accuracy = 100 * val_correct / val_total

        # Save history
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_accuracy)
        history['train_loss'].append(avg_loss)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_loss:.4f} "
              f"Train Acc: {train_accuracy:.2f}% "
              f"Val Acc: {val_accuracy:.2f}%")

    return best_model_state, best_val_accuracy, history


def evaluate_model(model, X_test_tensor, y_test, class_names, device):
    """Evaluate a model and return metrics"""
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor.to(device))
        _, predictions = torch.max(outputs, 1)
    predictions = predictions.cpu().numpy()

    test_accuracy = 100 * (predictions == y_test).sum() / len(y_test)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, predictions, average=None, zero_division=0)
    cm = confusion_matrix(y_test, predictions)

    print(f"\nTest Accuracy: {test_accuracy:.2f}%")
    print(classification_report(y_test, predictions, target_names=class_names))

    return {
        'test_accuracy': float(test_accuracy),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist(),
        'cm': cm.tolist(),
        'predictions': predictions.tolist(),
        'y_true': y_test.tolist()
    }


def plot_training_curves(with_history, without_history):
    """Plot training vs validation accuracy curves for both models"""
    epochs = range(1, len(with_history['train_acc']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Training Curves: Dropout vs No Dropout', fontsize=16, fontweight='bold')

    # With Dropout
    ax1.plot(epochs, with_history['train_acc'], 'b-o', markersize=5, label='Train Accuracy', linewidth=2)
    ax1.plot(epochs, with_history['val_acc'], 'r-o', markersize=5, label='Val Accuracy', linewidth=2)
    ax1.fill_between(epochs, with_history['train_acc'], with_history['val_acc'], alpha=0.15, color='gray')
    ax1.set_title('With Dropout (p=0.5)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_ylim(40, 105)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Annotate final values
    ax1.annotate(f"{with_history['train_acc'][-1]:.1f}%",
                 xy=(20, with_history['train_acc'][-1]),
                 xytext=(17, with_history['train_acc'][-1] + 3),
                 fontsize=9, fontweight='bold', color='blue')
    ax1.annotate(f"{with_history['val_acc'][-1]:.1f}%",
                 xy=(20, with_history['val_acc'][-1]),
                 xytext=(17, with_history['val_acc'][-1] - 5),
                 fontsize=9, fontweight='bold', color='red')

    # Without Dropout
    ax2.plot(epochs, without_history['train_acc'], 'b-o', markersize=5, label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, without_history['val_acc'], 'r-o', markersize=5, label='Val Accuracy', linewidth=2)
    ax2.fill_between(epochs, without_history['train_acc'], without_history['val_acc'], alpha=0.15, color='gray')
    ax2.set_title('Without Dropout', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_ylim(40, 105)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Annotate final values
    ax2.annotate(f"{without_history['train_acc'][-1]:.1f}%",
                 xy=(20, without_history['train_acc'][-1]),
                 xytext=(17, without_history['train_acc'][-1] + 3),
                 fontsize=9, fontweight='bold', color='blue')
    ax2.annotate(f"{without_history['val_acc'][-1]:.1f}%",
                 xy=(20, without_history['val_acc'][-1]),
                 xytext=(17, without_history['val_acc'][-1] - 5),
                 fontsize=9, fontweight='bold', color='red')

    plt.tight_layout()
    plt.savefig('results/dropout_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/dropout_training_curves.png")


def plot_dropout_comparison(with_results, without_results, class_names):
    """Generate side-by-side visualization for dropout comparison"""
    short_names = ['Mild\nDemented', 'Moderate\nDemented', 'Non\nDemented', 'Very Mild\nDemented']
    cm_labels = ['Mild', 'Moderate', 'Non', 'Very Mild']

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Dropout Comparison — 3-Layer CNN', fontsize=18, fontweight='bold')

    configs = [
        (with_results, 'With Dropout (p=0.5)', 0),
        (without_results, 'Without Dropout', 1),
    ]

    for data, title, col in configs:
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        precision = np.array(data['precision'])
        recall = np.array(data['recall'])
        f1 = np.array(data['f1'])
        cm = np.array(data['cm'])
        test_acc = data['test_accuracy']

        # Top: Per-class metrics
        x = np.arange(len(class_names))
        width = 0.25

        bars_p = ax_top.bar(x - width, precision, width, label='Precision', color='#1a5276')
        bars_r = ax_top.bar(x, recall, width, label='Recall', color='#e67e22')
        bars_f = ax_top.bar(x + width, f1, width, label='F1-Score', color='#1e8449')

        for bars in [bars_p, bars_r, bars_f]:
            for bar in bars:
                height = bar.get_height()
                ax_top.text(bar.get_x() + bar.get_width()/2, height + 0.005,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax_top.set_title(f'{title}\nTest Accuracy: {test_acc:.2f}%', fontsize=13, fontweight='bold')
        ax_top.set_ylabel('Score' if col == 0 else '', fontsize=11)
        ax_top.set_xticks(x)
        ax_top.set_xticklabels(short_names, fontsize=10)
        ax_top.set_ylim(0, 1.15)
        ax_top.legend(loc='upper right', fontsize=9)
        ax_top.grid(axis='y', alpha=0.2)

        # Bottom: Confusion matrix
        im = ax_bot.imshow(cm, cmap='Blues')
        ax_bot.set_xticks(range(len(cm_labels)))
        ax_bot.set_yticks(range(len(cm_labels)))
        ax_bot.set_xticklabels(cm_labels, fontsize=10)
        ax_bot.set_yticklabels(cm_labels, fontsize=10)
        ax_bot.set_xlabel('Predicted', fontsize=11)
        ax_bot.set_ylabel('Actual' if col == 0 else '', fontsize=11)

        for i in range(len(cm_labels)):
            for j in range(len(cm_labels)):
                color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                ax_bot.text(j, i, str(cm[i, j]), ha='center', va='center',
                            color=color, fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/dropout_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/dropout_comparison.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)
    print(f"Using device: {device}")

    # Load data
    print("Loading dataset...")
    images, labels, class_names = load_dataset("data")
    train_data, val_data, test_data = split_dataset(images, labels)

    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    X_train_t = torch.FloatTensor(X_train).unsqueeze(1)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val).unsqueeze(1)
    y_val_t = torch.LongTensor(y_val)
    X_test_t = torch.FloatTensor(X_test).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=32, shuffle=False)

    num_classes = len(class_names)

    # ===== WITH DROPOUT: Train fresh to capture history =====
    print("\n" + "="*50)
    print("Training: 3-Layer CNN WITH Dropout")
    print("="*50)

    with_dropout_model = CurrentCNN(num_classes=num_classes).to(device)
    with_state, with_val_acc, with_history = train_model_with_history(
        with_dropout_model, "With Dropout", train_loader, val_loader, device, num_epochs=20)

    with_dropout_model.load_state_dict(with_state)
    torch.save(with_state, "results/dropout_with.pth")
    with_results = evaluate_model(with_dropout_model, X_test_t, y_test, class_names, device)

    # ===== WITHOUT DROPOUT: Train fresh =====
    print("\n" + "="*50)
    print("Training: 3-Layer CNN WITHOUT Dropout")
    print("="*50)

    no_dropout_model = NoDropoutCNN(num_classes=num_classes).to(device)
    without_state, without_val_acc, without_history = train_model_with_history(
        no_dropout_model, "Without Dropout", train_loader, val_loader, device, num_epochs=20)

    no_dropout_model.load_state_dict(without_state)
    torch.save(without_state, "results/dropout_without.pth")
    without_results = evaluate_model(no_dropout_model, X_test_t, y_test, class_names, device)

    # ===== COMPARISON =====
    print("\n" + "="*50)
    print("DROPOUT COMPARISON")
    print("="*50)
    print(f"{'Model':<30} {'Test Acc':<15}")
    print("-"*45)
    print(f"{'With Dropout (p=0.5)':<30} {with_results['test_accuracy']:<15.2f}")
    print(f"{'Without Dropout':<30} {without_results['test_accuracy']:<15.2f}")
    print("-"*45)

    # Save results
    comparison = {
        'with_dropout': with_results,
        'without_dropout': without_results,
        'with_history': with_history,
        'without_history': without_history,
        'class_names': class_names
    }
    with open('results/dropout_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    print("Saved results to results/dropout_comparison.json")

    # Generate visualizations
    plot_training_curves(with_history, without_history)
    plot_dropout_comparison(with_results, without_results, class_names)


if __name__ == "__main__":
    main()