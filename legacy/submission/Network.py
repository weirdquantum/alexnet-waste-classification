import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=7):
        """
        Customized AlexNet implementation for a seven-class classification task.
        
        Args:
        - num_classes (int): Number of output classes. Default is 7.
        """
        super(AlexNet, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # Conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # Pool1
            
            nn.Conv2d(64, 192, kernel_size=5, padding=2),  # Conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # Pool2
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # Conv3
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # Conv4
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # Pool3
        )
        
        # Adaptive pooling to handle various input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, num_classes)  # Final output layer for 7 classes
        )

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
        - x (Tensor): Input tensor with shape [batch_size, 3, height, width].
        
        Returns:
        - Tensor: Output logits with shape [batch_size, num_classes].
        """
        x = self.features(x)  # Feature extraction
        x = self.avgpool(x)   # Adaptive pooling
        x = torch.flatten(x, 1)  # Flatten to feed into the fully connected layers
        x = self.classifier(x)  # Classification head
        return x


if __name__ == "__main__":
    # Test the model with random input
    model = AlexNet(num_classes=7)  # Set to 7 classes
    input_tensor = torch.randn(1, 3, 192, 192)  # Example input tensor
    output = model(input_tensor)  # Forward pass
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")  # Should output [1, 7]
    print(f"Output: {output}")