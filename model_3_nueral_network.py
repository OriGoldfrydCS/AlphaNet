import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
from sklearn.metrics import confusion_matrix, log_loss, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training dataset
train_file_path = "data/train_dataset.csv"
train_data = pd.read_csv(train_file_path, header=None)

# Load the test dataset
test_file_path = "data/test_dataset.csv"
test_data = pd.read_csv(test_file_path, header=None)

# Extract features and labels for training data
X_train_full = train_data.iloc[:, 1:].values  # Features (flattened images)
y_train_full = train_data.iloc[:, 0].values   # Labels (classes)

# Extract features and labels for test data
X_test = test_data.iloc[:, 1:].values  # Features (flattened images)
y_test = test_data.iloc[:, 0].values   # Labels (classes)

# Normalize pixel values to range [0, 1]
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0

# One-hot encode the labels
label_binarizer = LabelBinarizer()
y_train_full = label_binarizer.fit_transform(y_train_full)
y_test = label_binarizer.transform(y_test)                      # Variable keeping one-hot encoding for Log Loss

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Use class indices for CrossEntropyLoss
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)      # Use class indices for CrossEntropyLoss
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)    # Use class indices for CrossEntropyLoss

# Create DataLoader objects
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, torch.argmax(y_train_tensor, dim=1))
val_dataset = TensorDataset(X_val_tensor, torch.argmax(y_val_tensor, dim=1))
test_dataset = TensorDataset(X_test_tensor, torch.argmax(y_test_tensor, dim=1))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the neural network
class NeuralNetwork(nn.Module):
    """
    A fully connected neural network for multi-class classification.

    Attributes:
        fc1: First fully connected layer.
        fc2: Second fully connected layer.
        fc3: Output fully connected layer.
        relu1: ReLU activation after first layer.
        relu2: ReLU activation after second layer.
    """
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        """
        Initializes the neural network layers.

        Args:
            input_size (int): Number of input features.
            hidden_size1 (int): Number of neurons in the first hidden layer.
            hidden_size2 (int): Number of neurons in the second hidden layer.
            output_size (int): Number of output classes.
        """
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Model parameters
input_size = X_train.shape[1]           # Number of features
hidden_size1 = 128                      # Neurons in the first hidden layer
hidden_size2 = 64                       # Neurons in the second hidden layer
output_size = y_train.shape[1]          # Number of output classes

# Initialize the model, loss function, optimizer, and metrics
model = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)
criterion = nn.CrossEntropyLoss()                                               # Loss function for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)                            # Adam optimizer

# Use torchmetrics for accuracy, precision, and recall
train_accuracy_metric = MulticlassAccuracy(num_classes=output_size, average="micro")
train_precision_metric = MulticlassPrecision(num_classes=output_size, average="micro")
train_recall_metric = MulticlassRecall(num_classes=output_size, average="micro")

val_accuracy_metric = MulticlassAccuracy(num_classes=output_size, average="micro")
val_precision_metric = MulticlassPrecision(num_classes=output_size, average="micro")
val_recall_metric = MulticlassRecall(num_classes=output_size, average="micro")

# Training loop
epochs = 15
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    train_accuracy_metric.reset()
    train_precision_metric.reset()
    train_recall_metric.reset()

    # Training loop with tqdm
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Update metrics
            train_accuracy_metric.update(outputs.softmax(dim=1), labels)
            train_precision_metric.update(outputs.softmax(dim=1), labels)
            train_recall_metric.update(outputs.softmax(dim=1), labels)

            pbar.update(1)

    train_loss = running_loss / len(train_loader)
    train_accuracy = train_accuracy_metric.compute().item() * 100
    train_precision = train_precision_metric.compute().item() * 100
    train_recall = train_recall_metric.compute().item() * 100

    # Validation loop
    model.eval()
    val_loss = 0.0
    val_accuracy_metric.reset()
    val_precision_metric.reset()
    val_recall_metric.reset()

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_accuracy_metric.update(outputs.softmax(dim=1), labels)
            val_precision_metric.update(outputs.softmax(dim=1), labels)
            val_recall_metric.update(outputs.softmax(dim=1), labels)

    val_loss /= len(val_loader)
    val_accuracy = val_accuracy_metric.compute().item() * 100
    val_precision = val_precision_metric.compute().item() * 100
    val_recall = val_recall_metric.compute().item() * 100

    # Print epoch results
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Loss: {val_loss:.4f}%")

# Testing loop
model.eval()
test_accuracy_metric = MulticlassAccuracy(num_classes=output_size, average="micro")
test_precision_metric = MulticlassPrecision(num_classes=output_size, average="micro")
test_recall_metric = MulticlassRecall(num_classes=output_size, average="micro")

test_accuracy_metric.reset()
test_precision_metric.reset()
test_recall_metric.reset()

test_preds = []     # Store predictions for additional metrics
test_probs = []     # Store probabilities for Log Loss
test_labels = []    # Store true labels for additional metrics

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        probs = outputs.softmax(dim=1)      # Softmax probabilities for Log Loss
        preds = torch.argmax(probs, dim=1)  # Predicted class labels

        # Update torchmetrics
        test_accuracy_metric.update(probs, labels)
        test_precision_metric.update(probs, labels)
        test_recall_metric.update(probs, labels)

        # Collect data for additional metrics
        test_preds.extend(preds.cpu().numpy())
        test_probs.extend(probs.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_accuracy = test_accuracy_metric.compute().item() * 100
test_precision = test_precision_metric.compute().item() * 100
test_recall = test_recall_metric.compute().item() * 100
f1 = f1_score(test_labels, test_preds, average="macro")  # F1-Score
logloss = log_loss(test_labels, test_probs)  # Log Loss
conf_matrix = confusion_matrix(test_labels, test_preds)  # Confusion Matrix

# Display results
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Test Precision: {test_precision:.2f}%")
print(f"Test Recall: {test_recall:.2f}%")
print(f"Test F1-Score: {f1:.4f}")
print(f"Test Log Loss: {logloss:.4f}")

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=np.arange(output_size),
    yticklabels=np.arange(output_size),
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()