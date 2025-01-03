import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
      
# TensorBoard setup to log metrics for visualization
run_name = f"FCNN_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
writer = SummaryWriter(f"Stage_2_Classification_Module/runs/runs_FCNN_model/{run_name}")

# Load the training dataset
#   - First column is labels
#   - The rest are flattened pixel values
train_file_path = "Stage_2_Classification_Module/data/train_dataset.csv"
train_data = pd.read_csv(train_file_path, header=None)

# Load the test dataset
test_file_path = "Stage_2_Classification_Module/data/test_dataset.csv"
test_data = pd.read_csv(test_file_path, header=None)

# Separate features (X) and labels (y) in the training data.
#   - X_train_full: all pixel columns
#   - y_train_full: the labels in the first column
X_train_full = train_data.iloc[:, 1:].values        # Features (flattened images)
y_train_full = train_data.iloc[:, 0].values         # Labels (classes)

# Extract features and labels for test data
X_test = test_data.iloc[:, 1:].values               # Features (flattened images)
y_test = test_data.iloc[:, 0].values                # Labels (classes)

# Normalize pixel values from [0, 255] to [0, 1] for both train and test
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0

# Convert labels to one-hot format for computing Log Loss
# LabelBinarizer helps transform integer labels into one-hot vectors
label_binarizer = LabelBinarizer()                              # Initialize label binarizer
y_train_full = label_binarizer.fit_transform(y_train_full)      # One-hot encode training labels
y_test = label_binarizer.transform(y_test)                      # One-hot encode test labels

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Convert the data into PyTorch tensors and move them to GPU (if available)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# Create DataLoader objects for batching and shuffling.
#   - We convert the one-hot labels back to class indices with torch.argmax(..., dim=1)
#   - This is because CrossEntropyLoss expects class indices, not one-hot
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, torch.argmax(y_train_tensor, dim=1))
val_dataset = TensorDataset(X_val_tensor, torch.argmax(y_val_tensor, dim=1))
test_dataset = TensorDataset(X_test_tensor, torch.argmax(y_test_tensor, dim=1))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --------------------------------------------------------------------------
# Define a simple neural network with one hidden layer
#   - input_size: number of features per sample
#   - hidden_size: how many neurons in the hidden layer
#   - output_size: how many classes to predict
# --------------------------------------------------------------------------
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes a neural network with one fully connected hidden layer.
        Args:
            input_size  (int): The number of input features (784 if 28x28).
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Number of classes.
        """
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)      # First fully connected layer from input to hidden layer.
        self.relu1 = nn.ReLU()                             # ReLU activation after the hidden layer
        self.fc2 = nn.Linear(hidden_size, output_size)     # Output layer from hidden layer to output classes

    def forward(self, x):
        """
            Forward pass through the network.
            Returns logits (scores) for each class.
        """
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

# Set up the network, criterion, and optimizer.
#   - input_size = number of columns in X_train
#   - hidden_size = 128
#   - output_size = how many classes, from one-hot dimension
input_size = X_train.shape[1]
hidden_size = 128
output_size = y_train.shape[1]

# Initialize the neural network, loss function, and optimizer
model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Hyperparameters for training
epochs = 100                    # Maximum number of training epochs
patience = 10                   # Early stopping patience
best_val_loss = float('inf')    # Best validation loss so far
early_stop_counter = 0          # Counter for early stopping
start_time = time.time()        # Start timing the training process

# --------------------------------------------------------------------------
# 1. Training loop
# --------------------------------------------------------------------------
for epoch in range(epochs):
    model.train()                           # Set model to training mode
    running_loss = 0.0                      # Cumulative loss for the epoch
    y_true_train, y_pred_train = [], []     # Store true and predicted labels for training data

    # tqdm helps show a progress bar for the training loop
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
        # Iterate over batches of training data
        for inputs, labels in train_loader:
            optimizer.zero_grad()               # Zero gradients from the previous step
            outputs = model(inputs)             # Forward pass
            loss = criterion(outputs, labels)   # Compute loss
            loss.backward()                     # Backpropagation
            optimizer.step()                    # Update model weights
            running_loss += loss.item()         # Accumulate loss

            preds = torch.argmax(outputs, dim=1)        # Get predicted classes by finding the index of maximum logit
            y_true_train.extend(labels.cpu().numpy())   # Store true labels
            y_pred_train.extend(preds.cpu().numpy())    # Store predicted labels

            pbar.update(1)   # Update progress bar

    # Calculate training metrics after the epoch ends
    train_loss = running_loss / len(train_loader)
    train_accuracy = accuracy_score(y_true_train, y_pred_train)
    train_precision = precision_score(y_true_train, y_pred_train, average="macro")
    train_recall = recall_score(y_true_train, y_pred_train, average="macro")
    train_f1 = f1_score(y_true_train, y_pred_train, average="macro")

    # ------------------------------------------------
    # 2. Validation phase (no gradient updates)
    # ------------------------------------------------
    model.eval()                        # Set model to evaluation mode
    val_loss = 0.0                      # Cumulative validation loss
    y_true_val, y_pred_val = [], []     # Store true and predicted labels for validation data

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)             # Forward pass
            loss = criterion(outputs, labels)   # Compute loss
            val_loss += loss.item()             # Accumulate loss

            preds = torch.argmax(outputs, dim=1)        # Get predicted class
            y_true_val.extend(labels.cpu().numpy())     # Store true labels
            y_pred_val.extend(preds.cpu().numpy())      # Store predicted labels

    # Calculate validation metrics
    val_loss /= len(val_loader)
    val_accuracy = accuracy_score(y_true_val, y_pred_val)
    val_precision = precision_score(y_true_val, y_pred_val, average="macro")
    val_recall = recall_score(y_true_val, y_pred_val, average="macro")
    val_f1 = f1_score(y_true_val, y_pred_val, average="macro")

    # ------------------------------------------------
    # Log metrics to TensorBoard
    # ------------------------------------------------
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Accuracy/train", train_accuracy, epoch)
    writer.add_scalar("F1/train", train_f1, epoch)
    writer.add_scalar("Loss/validation", val_loss, epoch)
    writer.add_scalar("Accuracy/validation", val_accuracy, epoch)
    writer.add_scalar("F1/validation", val_f1, epoch)

    # Print epoch results
    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Train F1: {train_f1:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    if early_stop_counter >= patience:
        print(f"\nEarly stopping triggered after {epoch+1} epochs.")
        break

# --------------------------------------------------------------------------
# 3. Testing loop
# --------------------------------------------------------------------------
model.eval()                                            # Set model to evaluation mode
y_true_test, y_pred_test, test_probs = [], [], []       # Store true labels, predictions, and probabilities

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)                                 # Forward pass
        probs = torch.softmax(outputs, dim=1).cpu().numpy()     # Convert logits to probabilities with softmax
        preds = np.argmax(probs, axis=1)                        # Get predicted classes

        y_true_test.extend(labels.cpu().numpy())    # Store true labels
        y_pred_test.extend(preds)                   # Store predictions
        test_probs.extend(probs)                    # Store probabilities

# Normalize probabilities for Log Loss calculation
test_probs = np.array(test_probs)
test_probs = test_probs / test_probs.sum(axis=1, keepdims=True)

# Calculate test metrics
test_accuracy = accuracy_score(y_true_test, y_pred_test)
test_precision = precision_score(y_true_test, y_pred_test, average="macro")
test_recall = recall_score(y_true_test, y_pred_test, average="macro")
test_f1 = f1_score(y_true_test, y_pred_test, average="macro")
test_logloss = log_loss(y_true_test, test_probs)
conf_matrix = confusion_matrix(y_true_test, y_pred_test)

# Save metrics to .txt
training_duration = time.time() - start_time
output_file = f"Stage_2_Classification_Module/runs/runs_neural_network_model/{run_name}.txt"
with open(output_file, "w") as f:
    f.write(f"Run Name: {run_name}\n")
    f.write(f"Date: {datetime.now()}\n")
    f.write(f"Training Duration: {training_duration:.2f} seconds\n")
    f.write(f"Epochs: {epoch+1} out of {epochs}\n\n")

    f.write("Train Metrics:\n")
    f.write(f"  Accuracy: {train_accuracy:.4f}\n")
    f.write(f"  Precision: {train_precision:.4f}\n")
    f.write(f"  Recall: {train_recall:.4f}\n")
    f.write(f"  F1-Score: {train_f1:.4f}\n")
    f.write(f"  Log Loss: {train_loss:.4f}\n\n")

    f.write("Validation Metrics:\n")
    f.write(f"  Accuracy: {val_accuracy:.4f}\n")
    f.write(f"  Precision: {val_precision:.4f}\n")
    f.write(f"  Recall: {val_recall:.4f}\n")
    f.write(f"  F1-Score: {val_f1:.4f}\n")
    f.write(f"  Log Loss: {val_loss:.4f}\n\n")

    f.write("Test Metrics:\n")
    f.write(f"  Accuracy: {test_accuracy:.4f}\n")
    f.write(f"  Precision: {test_precision:.4f}\n")
    f.write(f"  Recall: {test_recall:.4f}\n")
    f.write(f"  F1-Score: {test_f1:.4f}\n")
    f.write(f"  Log Loss: {test_logloss:.4f}\n")

# Display final results
print(f"\nFully-Connected NN Model Performance:")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")
print(f"Test Log Loss: {test_logloss:.4f}")

# Confusion matrix
conf_matrix = confusion_matrix(y_true_test, y_pred_test)
# Generate character labels: A-Z (0-25), punctuation (26-30), and space (31)
labels = [chr(i) for i in range(65, 91)] + [",", ".", "!", "?", ";", "Space"]

# Plot confusion matrix
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