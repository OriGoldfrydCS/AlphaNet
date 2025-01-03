from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import confusion_matrix, log_loss, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# File paths for the training and testing CSV files
train_file_path = "Stage_2_Classification_Module/data/augmented_train_dataset.csv"
test_file_path = "Stage_2_Classification_Module/data/test_dataset.csv"

# Load the training data (labels + pixel data) from a CSV
# - The first column is the label
# - The rest are flattened 28x28 pixels
train_data = pd.read_csv(train_file_path, header=None)
test_data = pd.read_csv(test_file_path, header=None)

# Extract pixel values and labels from the training and test data.
#   - X_* holds pixel values
#   - y_* holds labels
X_train_full = train_data.iloc[:, 1:].values
X_train_full = X_train_full.reshape(-1, 1, 28, 28).astype('float32')  # Reshape to (N, 1, 28, 28)

y_train_full = train_data.iloc[:, 0].values
X_test = test_data.iloc[:, 1:].values
X_test = X_test.reshape(-1, 1, 28, 28).astype('float32')              # Reshape to (N, 1, 28, 28)

y_test = test_data.iloc[:, 0].values

# Normalize pixel values from [0, 255] to [0, 1]
X_train_full = (X_train_full / 255.0).reshape(-1, 1, 28, 28).astype('float32')

# Reshape to [N, 1, 28, 28]
X_test = (X_test / 255.0).reshape(-1, 1, 28, 28).astype('float32')

# Split the full training data into a smaller training set and a validation set
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# --------------------------------------------------------------------------
# Define a custom Dataset class to handle images and labels
# --------------------------------------------------------------------------
class AtoZDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data (numpy.ndarray): Array of shape (N, 1, 28, 28) with pixel values.
            labels (array-like): 1D array of labels of length N.
            transform (callable, optional): Optional transform to be applied
                                            on a sample (e.g., PIL-based transform).
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """
        Returns the number of samples (N).
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves the sample at index `idx`.

        Returns:
            image (Tensor): The image data (possibly transformed).
            label (Tensor): The class label (long).
        """
        image = self.data[idx]      # Shape: (1, 28, 28)
        label = self.labels[idx]
        
        # Convert numpy array to a PIL image if needed. Otherwise, we just make a PyTorch tensor
        if self.transform:
            image = image.squeeze(0)                # Convert (1, 28, 28) to (28, 28)
            image = transforms.ToPILImage()(image)  # Convert to PIL
            image = self.transform(image)           # Apply transformations
        else:
            # If no transform, just convert to a PyTorch float32 tensor
            image = torch.tensor(image, dtype=torch.float32)

        # Also make the label a long tensor for use with CrossEntropyLoss
        return image, torch.tensor(label, dtype=torch.long)

# Create Datasets and DataLoaders for training, validation, and testing
batch_size = 32
train_dataset = AtoZDataset(X_train, y_train)
val_dataset = AtoZDataset(X_val, y_val)
test_dataset = AtoZDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define a CNN (Convolutional Neural Network)
class CNN(nn.Module):
    """
    A deeper Convolutional Neural Network for image classification.

    Architecture:
    - Convolution + BatchNorm + ReLU -> Convolution + BatchNorm + ReLU -> MaxPool
    - Convolution + BatchNorm + ReLU -> MaxPool
    - Convolution + BatchNorm + ReLU -> MaxPool
    - Flatten
    - Fully Connected -> Dropout -> Fully Connected -> Dropout -> Output
    """
    def __init__(self):
        """
          Initializes the model with multiple convolutional layers + batchnorm and multiple fully connected layers
          """
        super(CNN, self).__init__()
        
        # 1) Convolutional layer from 1 channel -> 32 channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # Output: (32, 28, 28)
        self.bn1 = nn.BatchNorm2d(32)  # Normalize the activations to improve training stability
        
        # 2) Convolutional layer from 32 channels -> 64 channels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # Output: (64, 28, 28)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample to (64, 14, 14)
        
        # 3) Convolutional layer from 64 channels -> 128 channels
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  # Output: (128, 14, 14)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample to (128, 7, 7)
        
        # 4) Convolutional layer from 128 channels -> 256 channels
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)  # Output: (256, 7, 7)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample to (256, 3, 3)

        # After the third pooling, the feature map is 256 x 3 x 3 -> total 2304 units
        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=256 * 3 * 3, out_features=512)     # First dense layer
        self.fc2 = nn.Linear(in_features=512, out_features=256)             # Second dense layer
        self.fc3 = nn.Linear(in_features=256, out_features=32)              # Output layer for 32 classes
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization (50% probability)

    def forward(self, x):
        """
        Forward pass of the CNN.

        Args:
            x (Tensor): Input of shape [batch_size, 1, 28, 28].

        Returns:
            Tensor of shape [batch_size, 32] with unnormalized class scores.
        """
        # 1) Convolution block
        x = self.relu(self.bn1(self.conv1(x)))

        # 2) Convolution block
        x = self.pool1(self.relu(self.bn2(self.conv2(x))))

        # 3) Convolution block
        x = self.pool2(self.relu(self.bn3(self.conv3(x))))

        # 4) Convolution block
        x = self.pool3(self.relu(self.bn4(self.conv4(x))))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)       # Reshape to (batch_size, 256*3*3 = 2304)
        
        # FC layers + Dropout
        x = self.relu(self.fc1(x))          # First dense layer
        x = self.dropout(x)                 # Apply dropout
        x = self.relu(self.fc2(x))          # Second dense layer
        x = self.dropout(x)                 # Apply dropout
        x = self.fc3(x)                     # Output layer

        return x
    
# Instantiate the CNN, loss, optimizer, and learning rate scheduler
model = CNN().to(device)   
print(model)

criterion = nn.CrossEntropyLoss()       # CrossEntropyLoss for multi-class classification
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

# TensorBoard writer
run_name = f"CNN_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
writer = SummaryWriter(f"Stage_2_Classification_Module/runs/runs_CNN_model/{run_name}")

# Training loop parameters
epochs = 100
patience = 10
best_val_loss = float('inf')
early_stop_counter = 0

print("Training the CNN model...")

# --------------------------------------------------------------------------
# 1. Training loop
# --------------------------------------------------------------------------
for epoch in range(epochs):
    model.train()           # Put model in training mode
    running_loss = 0.0      # Cumulative loss for the epoch
    correct_train = 0
    total_train = 0

    # tqdm progress bar to show training status
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
        # Iterate over each batch from train_loader
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()               # Clear gradients
            outputs = model(inputs).float()     # Forward pass
            labels = labels.long()
            loss = criterion(outputs, labels)   # Compute the CrossEntropy loss
            loss.backward()                     # Backpropagation
            optimizer.step()                    # Update model weights

            running_loss += loss.item()         # Accumulate loss
            _, predicted = torch.max(outputs, 1)    # Get predicted classes by finding the index of maximum logit

            # Count correct predictions and total samples
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

            # Count correct predictions and total samples
            pbar.set_postfix({'Train Acc': f'{(correct_train / total_train):.2%}'})
            pbar.update(1)

    # Compute average training accuracy
    train_accuracy = correct_train / total_train

    # --------------------------------------------------------------------------
    # 2. Validation phase
    # --------------------------------------------------------------------------
    model.eval()        # Set model to evaluation mode
    val_loss = 0.0      # Cumulative validation loss
    correct_val = 0     # Count correct predictions
    total_val = 0       # Count total predictions

    # Disable gradients for validation
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)     # Forward pass

            # Compute validation loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Compute validation accuracy
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    val_accuracy = correct_val / total_val
    print(f"Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.2%}")

    # Log metrics to TensorBoard
    writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch)
    writer.add_scalar('Loss/validation', val_loss / len(val_loader), epoch)
    writer.add_scalar('Accuracy/train', train_accuracy, epoch)
    writer.add_scalar('Accuracy/validation', val_accuracy, epoch)

    # Update LR scheduler
    scheduler.step()

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), f"Stage_2_Classification_Module/models/cnn_models/best_model_{run_name}.pth")
    else:
        early_stop_counter += 1
        print(f"early_stop_counter: {early_stop_counter}")
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

# Load best model
model.load_state_dict(torch.load(f"Stage_2_Classification_Module/models/cnn_models/best_model_{run_name}.pth"))

# --------------------------------------------------------------------------
# 3. Test loop
# --------------------------------------------------------------------------
model.eval()
correct = 0
total = 0
test_preds, test_probs, test_labels = [], [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        # Softmax for probabilities, so we can compute log_loss
        probs = outputs.softmax(dim=1).cpu().numpy()
        _, predicted = torch.max(outputs, 1)

        # Track correctness and total
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Accumulate predictions, probabilities, and true labels
        test_preds.extend(predicted.cpu().numpy())
        test_probs.extend(probs)
        test_labels.extend(labels.cpu().numpy())

# Compute final metrics on the test set
test_accuracy = correct / total
precision = precision_score(test_labels, test_preds, average="macro")
recall = recall_score(test_labels, test_preds, average="macro")
f1 = f1_score(test_labels, test_preds, average="macro")
logloss = log_loss(test_labels, test_probs, labels=np.arange(32))
conf_matrix = confusion_matrix(test_labels, test_preds)

print(f"\nCNN Model Performance:")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1-Score: {f1:.4f}")
print(f"Test Log Loss: {logloss:.4f}")

# Generate character labels: A-Z (0-25) and comma, dot, exclamation, question, semicolon (26-30), space (31)
labels = [chr(i) for i in range(65, 91)] + [",", ".", "!", "?", ";", "Space"]

# Confusion Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, 
            annot=True, 
            fmt="d", 
            cmap="Blues", 
            xticklabels=labels, 
            yticklabels=labels)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Close TensorBoard writer
writer.close()
