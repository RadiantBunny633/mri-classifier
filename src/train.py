"""
Training script for MRI Classifier
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
import os
import json

# Add src to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_dataset, split_dataset
from model import MRIClassifier


def train_model(num_epochs=20, batch_size=32, learning_rate=0.001):
    # Check if GPU available (faster training)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)
    print(f"Using device: {device}")
    
    # Load data
    print("Loading dataset...")
    images, labels, class_names = load_dataset("data")
    print(f"Loaded {len(images)} images")
    print(f"Classes: {class_names}")
    
    # Split data
    train_data, val_data, test_data = split_dataset(images, labels)
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    # Convert to PyTorch tensors
    # Add channel dimension (batch, channels, height, width)
    X_train = torch.FloatTensor(X_train).unsqueeze(1)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val).unsqueeze(1)
    y_val = torch.LongTensor(y_val)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create data loaders (handles batching)
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = MRIClassifier(num_classes=len(class_names)).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Track best model
    best_val_accuracy = 0.0
    
    # Training loop
    print("\nStarting training...")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track stats
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        train_accuracy = 100 * train_correct / train_total
        
        # Validation phase
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
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss/len(train_loader):.4f} "
              f"Train Acc: {train_accuracy:.2f}% "
              f"Val Acc: {val_accuracy:.2f}%")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "results/best_model.pth")
            print(f"  -> New best model saved!")
    
    print("-" * 50)
    print(f"Training complete! Best validation accuracy: {best_val_accuracy:.2f}%")
    
    # Evaluate on test set
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predictions = torch.max(outputs, 1)
        predictions = predictions.cpu().numpy()
    
    test_accuracy = 100 * (predictions == y_test).sum() / len(y_test)
    
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # Save results
    results = {
        'test_accuracy': float(test_accuracy),
        'val_accuracy': float(best_val_accuracy),
        'class_names': class_names,
        'predictions': predictions.tolist(),
        'y_true': y_test.tolist()
    }

    with open('results/cnn_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved results to results/cnn_results.json")
    
    return model, class_names


if __name__ == "__main__":
    train_model(num_epochs=20, batch_size=32, learning_rate=0.001)