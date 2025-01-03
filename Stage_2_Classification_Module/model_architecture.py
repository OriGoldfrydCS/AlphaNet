import torch
import torch.nn as nn

class CNN(nn.Module):
    """
    A deeper Convolutional Neural Network for image classification.

    Architecture:
        - Conv2D -> BatchNorm -> ReLU -> MaxPool
        - Conv2D -> BatchNorm -> ReLU -> MaxPool
        - Conv2D -> BatchNorm -> ReLU -> MaxPool
        - Fully Connected -> Dropout -> Fully Connected -> Dropout -> Output
    """
    def __init__(self):
        """
        Initializes the DeepCNN model with additional convolutional and fully connected layers.
        """
        super(CNN, self).__init__()
        
        # First Convolutional Block: Extract low-level features
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # Output: (32, 28, 28)
        self.bn1 = nn.BatchNorm2d(32)  # Normalize the activations to improve training stability
        
        # Second Convolutional Block: Extract more complex features
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # Output: (64, 28, 28)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample to (64, 14, 14)
        
        # Third Convolutional Block: Deeper feature extraction
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  # Output: (128, 14, 14)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample to (128, 7, 7)
        
        # Fourth Convolutional Block: Further abstraction
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)  # Output: (256, 7, 7)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample to (256, 3, 3)

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=256 * 3 * 3, out_features=512)  # First dense layer
        self.fc2 = nn.Linear(in_features=512, out_features=256)  # Second dense layer
        self.fc3 = nn.Linear(in_features=256, out_features=32)  # Output layer for 32 classes (A-Z), ".",", "?", "!", ";" and "space"
        
        # Activation and Dropout
        self.relu = nn.ReLU()  # Non-linear activation function
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization (50% probability)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
        - x: Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
        - Output logits of shape (batch_size, 32).
        """
        # Convolutional Layers
        x = self.relu(self.bn1(self.conv1(x)))  # First convolutional block
        x = self.pool1(self.relu(self.bn2(self.conv2(x))))  # Second convolutional block
        x = self.pool2(self.relu(self.bn3(self.conv3(x))))  # Third convolutional block
        x = self.pool3(self.relu(self.bn4(self.conv4(x))))  # Fourth convolutional block
        
        # Flatten the feature maps for the fully connected layers
        x = x.view(x.size(0), -1)  # Reshape to (batch_size, 256*3*3)
        
        # Fully Connected Layers
        x = self.relu(self.fc1(x))  # First dense layer
        x = self.dropout(x)  # Apply dropout
        x = self.relu(self.fc2(x))  # Second dense layer
        x = self.dropout(x)  # Apply dropout
        x = self.fc3(x)  # Output layer

        return x
