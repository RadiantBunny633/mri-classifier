import torch
import torch.nn as nn


class MRIClassifier(nn.Module):
    def __init__(self, num_classes=4, input_size=128):
        super(MRIClassifier, self).__init__()
        
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
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


if __name__ == "__main__":
    model = MRIClassifier(num_classes=4)
    dummy_input = torch.randn(2, 1, 128, 128)
    output = model(dummy_input)
    
    print("Model created successfully!")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output represents probabilities for {output.shape[1]} classes")