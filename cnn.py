from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, log_loss, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt 
import seaborn as sns  
import numpy as np

# Load the training and test datasets
train_file_path = "data/augmented_train_dataset.csv"
test_file_path = "data/test_dataset.csv"

train_data = pd.read_csv(train_file_path, header=None)
test_data = pd.read_csv(test_file_path, header=None)

# Extract features and labels
X_train_full = train_data.iloc[:, 1:].values  # Features (flattened images)
y_train_full = train_data.iloc[:, 0].values   # Labels (classes)
X_test = test_data.iloc[:, 1:].values         # Test features
y_test = test_data.iloc[:, 0].values          # Test labels

# Normalize pixel values to range [0, 1]
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0

# Reshape the flattened images into 28x28 format
X_train_full = X_train_full.reshape(-1, 1, 28, 28).astype('float32')
X_test = X_test.reshape(-1, 1, 28, 28).astype('float32')

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Create a custom Dataset class
class AtoZDataset(Dataset):
    """
    Custom dataset for handling image data and labels.

    Args:
        data (numpy.ndarray): Image data.
        labels (numpy.ndarray): Corresponding labels.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the image tensor and label tensor.
    """
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create DataLoaders
batch_size = 128
train_dataset = AtoZDataset(X_train, y_train)
val_dataset = AtoZDataset(X_val, y_val)
test_dataset = AtoZDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the CNN
class CNN(nn.Module):
    """
    Convolutional Neural Network for image classification.

    Architecture:
        - Conv2D -> BatchNorm -> ReLU -> MaxPool
        - Conv2D -> BatchNorm -> ReLU -> MaxPool
        - Fully Connected -> Dropout -> Fully Connected -> Dropout -> Output

    Attributes:
        conv1, conv2: Convolutional layers.
        bn1, bn2: Batch normalization layers.
        pool: Max pooling layer.
        fc1, fc2, fc3: Fully connected layers.
        dropout: Dropout layer.
        relu: ReLU activation.
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 14 * 14, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 26)               # 26 classes for A-Z
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Initialize the model, loss function, optimizer, and TensorBoard writer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

# Add writer to log metrics to TensorBoard with timestamp
writer = SummaryWriter(f"runs/AtoZ_CNN_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

# Early stopping parameters
patience = 10 #5
best_val_loss = float('inf')
early_stop_counter = 0

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate train accuracy
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
            train_accuracy = correct_train / total_train

            pbar.set_postfix({'Train Acc': f'{train_accuracy:.2%}'})
            pbar.update(1)

    # Validation
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Calculate validation accuracy
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    val_accuracy = correct_val / total_val
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.2%}")

    # Log metrics to TensorBoard
    writer.add_scalar('Loss/Train', running_loss / len(train_loader), epoch)
    writer.add_scalar('Loss/Validation', val_loss / len(val_loader), epoch)
    writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
    writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)

    # Update learning rate scheduler
    scheduler.step(val_loss / len(val_loader))

    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), f"best_model.pth")
    else:
        early_stop_counter += 1
        print(f"Early stopping counter: {early_stop_counter}")
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

# Load the best model
model.load_state_dict(torch.load("best_model.pth", weights_only=True))

# Testing loop
model.eval()
correct = 0
total = 0
test_preds = []  
test_probs = [] 
test_labels = []  

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        probs = outputs.softmax(dim=1)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Collect predictions, probabilities, and labels
        test_preds.extend(predicted.cpu().numpy())
        test_probs.extend(probs.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

# Calculate additional metrics
test_accuracy = 100 * correct / total
precision = precision_score(test_labels, test_preds, average="macro")  
recall = recall_score(test_labels, test_preds, average="macro")
f1 = f1_score(test_labels, test_preds, average="macro")  
logloss = log_loss(test_labels, test_probs)  
conf_matrix = confusion_matrix(test_labels, test_preds)  

# Display results
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Test Precision: {precision:.4f}") 
print(f"Test Recall: {recall:.4f}")  
print(f"Test F1-Score: {f1:.4f}")  
print(f"Test Log Loss: {logloss:.4f}") 

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=np.arange(26),
    yticklabels=np.arange(26),
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

writer.add_scalar('Accuracy/Test', test_accuracy)
writer.close()
