import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time

# Create directories for saving runs and logs
os.makedirs("runs/runs_SM_model", exist_ok=True)

def load_csv_data(file_path):
    """
    Reads a CSV file and extracts features (images) and labels (class IDs).

    Parameters
    ----------
    file_path : str
        The path to the CSV file. The CSV is assumed to have no header and
        the first column is the label, with the remaining columns for pixels.

    Returns
    -------
    images : ndarray
        A 2D NumPy array containing flattened pixel data for each sample.
    labels : ndarray
        A 1D NumPy array with integer class labels.
    """
    data = pd.read_csv(file_path, header=None)     # Read the CSV file without column headers
    labels = data.iloc[:, 0].to_numpy()            # First column is the label
    images = data.iloc[:, 1:].to_numpy()           # Remaining columns are the image data
    return images, labels

# File paths for the training and testing CSV files
train_csv_path = "Stage_2_Classification_Module/data/train_dataset.csv"
test_csv_path = "Stage_2_Classification_Module/data/test_dataset.csv"

print("Loading and preprocessing data...")

# Load and preprocess the datasets
X_train, y_train = load_csv_data(train_csv_path)
X_test, y_test = load_csv_data(test_csv_path)

# Normalize pixel values from [0, 255] to [0, 1]
X_train, X_test = X_train / 255.0, X_test / 255.0

# Convert NumPy arrays to PyTorch tensors so the PyTorch library can use them
#   - float32 for images
#   - long (int64) for labels
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create TensorDatasets, which pair up features (X) and labels (y)
# for use with DataLoader objects
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)      # Combine training features and label
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)         # Combine testing features and labels

# DataLoader handles batching, shuffling, and iterating over the dataset
# - batch_size=64 means 64 samples per batch
# - shuffle=True for the training set to randomize the order
# - shuffle=False for the test set to maintain a consistent order
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class SoftmaxRegression(nn.Module):
    """
    A single-layer linear model for multi-class classification, referred to as Softmax Regression.

    Parameters
    ----------
    input_dim : int
        Number of input features. For a 28x28 image, this is 784.
    output_dim : int
        Number of classes to predict.
    """
    def __init__(self, input_dim, output_dim):
        super(SoftmaxRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)      # Linear layer (input_dim -> output_dim)

    def forward(self, x):
        """
        Forward pass through the linear layer.
        The output of 'self.linear' is then used with CrossEntropyLoss
        to compute softmax and loss under the hood.
        """
        return self.linear(x)

# Initialize the model, loss function, and optimizer
input_dim = 28 * 28                                 # Number of input features (28x28 pixel images)
output_dim = 32                                     # Number of classes
model = SoftmaxRegression(input_dim, output_dim)    # Initiate the model
criterion = nn.CrossEntropyLoss()                   # Loss function (CrossEntropy)
optimizer = optim.SGD(model.parameters(), lr=0.1)   # We use Stochastic Gradient Descent with lr=0.1

# Set up TensorBoard to log training metrics for visualization
run_name = f"SM_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
writer = SummaryWriter(f"Stage_2_Classification_Module/runs/runs_SM_model/{run_name}")

# Training configuration: number of epochs and early stopping patience
epochs = 100                    # Maximum number of training epochs to run
patience = 5                    # Number of epochs to wait before stopping if validation loss doesn't improve
best_val_loss = float('inf')    # Track the best validation loss observed during training
early_stop_counter = 0          # Counter to track epochs without improvement in validation loss

start_time = time.time()        # Record the start time of training

print(f"Training model with early stopping...")

# Training loop
for epoch in range(epochs):
    # --------------------
    # 1. Training phase
    # --------------------
    model.train()           # Put model in training mode
    train_loss = 0          # Accumulate training loss over the batches
    train_correct = 0       # Count of correctly classified training examples
    train_total = 0         # Total training examples seen
    y_true_train = []       # Store true labels
    y_pred_train = []       # Store predicted labels

    # Iterate over batches in the training DataLoader
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()               # Clear the previous gradient
        outputs = model(X_batch)            # Forward pass
        loss = criterion(outputs, y_batch)  # Calculate CrossEntropyLoss
        loss.backward()                     # Backpropagation
        optimizer.step()                    # Update the model parameters

        # Accumulate the loss
        train_loss += loss.item()

        # Get the class with the highest probability
        _, predicted = outputs.max(1)

        # Count how many predictions were correct
        train_correct += (predicted == y_batch).sum().item()    # Count correct predictions

        # Keep track of total samples processed
        train_total += y_batch.size(0)

        # Save labels for precision/recall/f1 calculations
        y_true_train.extend(y_batch.tolist())       # Append true labels
        y_pred_train.extend(predicted.tolist())     # Append predicted labels

    # Average training loss across all batches
    train_accuracy = train_correct / train_total

    # Calculate training accuracy
    train_loss /= len(train_loader)

    # ----------------------
    # 2. Validation phase
    # ----------------------
    model.eval()        # Put model in evaluation mode
    val_loss = 0        # Accumulate validation loss
    val_correct = 0     # Count of correctly classified validation examples
    val_total = 0       # Total number of validation examples
    y_true_val = []     # Store true labels for validation
    y_pred_val = []     # Store predicted labels for validation

    # Disable gradient computation during validation
    with torch.no_grad():
        # Iterate over validation batches
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)                # Forward pass  
            loss = criterion(outputs, y_batch)      # Compute the loss
            val_loss += loss.item()                 # Accumulate loss

            _, predicted = outputs.max(1)           # Get the class with the highest probability
            y_true_val.extend(y_batch.tolist())     # Append true labels
            y_pred_val.extend(predicted.tolist())   # Append predicted labels        

            val_correct += (predicted == y_batch).sum().item()  # Count correct predictions
            val_total += y_batch.size(0)                        # Count total examples

    # Average validation loss across batches
    val_accuracy = val_correct / val_total

    # Validation accuracyc
    val_loss /= len(test_loader)

    # ----------------------------------------------------------------------
    # 3. Log metrics to TensorBoard
    # ----------------------------------------------------------------------
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/validation", val_loss, epoch)
    writer.add_scalar("Accuracy/train", train_accuracy, epoch)
    writer.add_scalar("Accuracy/validation", val_accuracy, epoch)

    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")


    # ----------------------------------------------------------------------
    # 4. Early Stopping: stop if we haven't improved for 'patience' epochs
    # ----------------------------------------------------------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss    # Update best validation loss
        early_stop_counter = 0      # Reset early stopping counter
    else:
        early_stop_counter += 1     # Increment early stopping counter

    # Trigger early stopping if no improvement for 'patience' epochs
    if early_stop_counter >= patience:
        print(f"\nEarly stopping triggered after {epoch+1} epochs.")
        break

# Compute final metrics on the training set
train_metrics = {
    "Accuracy": train_accuracy,
    "Precision": precision_score(y_true_train, y_pred_train, average="macro"),
    "Recall": recall_score(y_true_train, y_pred_train, average="macro"),
    "F1-Score": f1_score(y_true_train, y_pred_train, average="macro"),
    "Log Loss": log_loss(y_true_train, torch.softmax(torch.tensor(model(X_train_tensor).detach().numpy()), dim=1).numpy())
}

#Compute final metrics on the validation set
val_metrics = {
    "Accuracy": val_accuracy,
    "Loss": val_loss,
    "Precision": precision_score(y_true_val, y_pred_val, average="macro"),
    "Recall": recall_score(y_true_val, y_pred_val, average="macro"),
    "F1-Score": f1_score(y_true_val, y_pred_val, average="macro"),
    "Log Loss": log_loss(y_true_val, torch.softmax(torch.tensor(model(X_test_tensor).detach().numpy()), dim=1).numpy())
}

# Final evaluation on test data
y_true_test = []
y_pred_test = []
model.eval()

# Disable gradient computation for testing
with torch.no_grad():
    # Iterate over test batches
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = outputs.max(1)
        y_true_test.extend(y_batch.tolist())
        y_pred_test.extend(predicted.tolist())

# Compute performance metrics on the test data
test_metrics = {
    "Accuracy": accuracy_score(y_true_test, y_pred_test),
    "Precision": precision_score(y_true_test, y_pred_test, average="macro"),
    "Recall": recall_score(y_true_test, y_pred_test, average="macro"),
    "F1-Score": f1_score(y_true_test, y_pred_test, average="macro"),
    "Log Loss": log_loss(y_true_test, torch.softmax(torch.tensor(model(X_test_tensor).detach().numpy()), dim=1).numpy())
}

# Display final results in the console
print(f"\nSoftmax Model Performance:")
print(f"Test Accuracy: {test_metrics['Accuracy']:.4f}")
print(f"Test Precision: {test_metrics['Precision']:.4f}")
print(f"Test Recall: {test_metrics['Recall']:.4f}")
print(f"Test F1-Score: {test_metrics['F1-Score']:.4f}")
print(f"Test Log Loss: {test_metrics['Log Loss']:.4f}")

# Plot confusion matrix for the final test predictions
conf_matrix = confusion_matrix(y_true_test, y_pred_test)
# Generate character labels: A-Z (0-25), punctuation (26-30), and space (31)
labels = [chr(i) for i in range(65, 91)] + [",", ".", "!", "?", ";", "Space"]

# Plot confusion matrix with proper labels
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

# Save metrics to a text file in the run directory
output_file = f"Stage_2_Classification_Module/runs_SM_model/{run_name}.txt"
with open(output_file, "w") as f:
    f.write(f"Run Name: {run_name}\n")
    f.write(f"Date: {datetime.now()}\n")
    f.write(f"Training Duration: {time.time() - start_time:.2f} seconds\n")
    f.write(f"Epochs: {epoch+1} out of {epochs}\n")
    f.write("\nTrain Metrics:\n")
    for key, value in train_metrics.items():
        f.write(f"  {key}: {value:.4f}\n")
    f.write("\nValidation Metrics:\n")
    for key, value in val_metrics.items():
        f.write(f"  {key}: {value:.4f}\n")
    f.write("\nTest Metrics:\n")
    for key, value in test_metrics.items():
        f.write(f"  {key}: {value:.4f}\n")

# Close TensorBoard writer
writer.close()
