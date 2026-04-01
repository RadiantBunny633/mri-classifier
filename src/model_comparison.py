"""
Compare CNN architectures: Shallow vs Current vs Deep
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_dataset, split_dataset


# ============== THREE MODEL ARCHITECTURES ==============

class ShallowCNN(nn.Module):
    """2 convolutional layers"""
    def __init__(self, num_classes=4, input_size=128):
        super(ShallowCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128 -> 64
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32
        )
        
        conv_output_size = 64 * (input_size // 4) * (input_size // 4)
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class CurrentCNN(nn.Module):
    """3 convolutional layers (what we have)"""
    def __init__(self, num_classes=4, input_size=128):
        super(CurrentCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128 -> 64
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
        )
        
        conv_output_size = 128 * (input_size // 8) * (input_size // 8)
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class DeepCNN(nn.Module):
    """4 convolutional layers"""
    def __init__(self, num_classes=4, input_size=128):
        super(DeepCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128 -> 64
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        
        conv_output_size = 256 * (input_size // 16) * (input_size // 16)
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ============== TRAINING FUNCTION ==============

def train_and_evaluate(model, model_name, train_loader, val_loader, test_loader, device, num_epochs=20):
    """Train a model and return results"""
    
    print(f"\n{'='*50}")
    print(f"Training: {model_name}")
    print(f"{'='*50}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_accuracy = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_correct = 0
        train_total = 0
        
        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        train_accuracy = 100 * train_correct / train_total
        
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
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Acc: {train_accuracy:.2f}% Val Acc: {val_accuracy:.2f}%")
    
    # Test
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += batch_labels.size(0)
            test_correct += (predicted == batch_labels).sum().item()
    
    test_accuracy = 100 * test_correct / test_total
    
    print(f"\n{model_name} Results:")
    print(f"  Best Validation Accuracy: {best_val_accuracy:.2f}%")
    print(f"  Test Accuracy: {test_accuracy:.2f}%")
    
    return {
        'name': model_name,
        'val_accuracy': best_val_accuracy,
        'test_accuracy': test_accuracy
    }


# ============== MAIN ==============

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading dataset...")
    images, labels, class_names = load_dataset("data")
    train_data, val_data, test_data = split_dataset(images, labels)
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train).unsqueeze(1)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val).unsqueeze(1)
    y_val = torch.LongTensor(y_val)
    X_test = torch.FloatTensor(X_test).unsqueeze(1)
    y_test = torch.LongTensor(y_test)
    
    # Create data loaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Define models
    models = [
        (ShallowCNN(num_classes=len(class_names)), "Shallow CNN (2 conv)"),
        (CurrentCNN(num_classes=len(class_names)), "Current CNN (3 conv)"),
        (DeepCNN(num_classes=len(class_names)), "Deep CNN (4 conv)")
    ]
    
    # Train and evaluate each
    results = []
    for model, name in models:
        result = train_and_evaluate(model, name, train_loader, val_loader, test_loader, device, num_epochs=20)
        results.append(result)
    
    # Final comparison
    print("\n" + "="*50)
    print("FINAL COMPARISON")
    print("="*50)
    print(f"{'Model':<25} {'Val Acc':<15} {'Test Acc':<15}")
    print("-"*50)
    for r in results:
        print(f"{r['name']:<25} {r['val_accuracy']:<15.2f} {r['test_accuracy']:<15.2f}")
    print("-"*50)
    
    # Find best
    best = max(results, key=lambda x: x['test_accuracy'])
    print(f"\nBest Model: {best['name']} with {best['test_accuracy']:.2f}% test accuracy")


if __name__ == "__main__":
    main()