import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from datetime import datetime
import time

# Create directories for saving runs and logs
os.makedirs("runs/runs_SM_model", exist_ok=True)

# Function to load data from CSV files
def load_csv_data(file_path):
    """
    Reads data from a CSV file and separates it into images (features) and labels (target values).
    Args:
        file_path (str): Path to the CSV file containing the dataset.
    Returns:
        images (ndarray): Array of image data (features).
        labels (ndarray): Array of labels corresponding to the images.
    """
    data = pd.read_csv(file_path, header=None)      # Read the CSV file without column headers
    labels = data.iloc[:, 0].to_numpy()             # First column is the label
    images = data.iloc[:, 1:].to_numpy()            # Remaining columns are the image data
    return images, labels

# Define paths to training and testing datasets
train_csv_path = "data/train_dataset.csv"
test_csv_path = "data/test_dataset.csv"

print("Loading and preprocessing data...")

# Load and preprocess the datasets
X_train, y_train = load_csv_data(train_csv_path)
X_test, y_test = load_csv_data(test_csv_path)

# Normalize the pixel values of the images to the range [0, 1]
X_train, X_test = X_train / 255.0, X_test / 255.0

# Convert the data into PyTorch tensors for use with the PyTorch library
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create PyTorch datasets for training and testing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)       # Combine training features and label
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)          # Combine testing features and labels

# Create data loaders to handle batching and shuffling of data
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)   # Batch size of 64 for training
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)    # Batch size of 64 for testing

# Define the model: Softmax Regression
class SoftmaxRegression(nn.Module):
    """
    Implements a single-layer linear model with softmax activation for multi-class classification.
    """
    def __init__(self, input_dim, output_dim):
        super(SoftmaxRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  # Linear layer (input_dim -> output_dim)

    def forward(self, x):
        return self.linear(x)       # Forward pass: apply the linear transformation

# Initialize the model, loss function, and optimizer
input_dim = 28 * 28                                 # Number of input features (28x28 pixel images)
output_dim = 26                                     # Number of classes (A-Z)
model = SoftmaxRegression(input_dim, output_dim)    # Instantiate the model
criterion = nn.CrossEntropyLoss()                   # Loss function for multi-class classification
optimizer = optim.SGD(model.parameters(), lr=0.1)   # Stochastic Gradient Descent optimizer with a learning rate of 0.1

# Set up TensorBoard to log training metrics for visualization
run_name = f"AtoZ_SM_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
writer = SummaryWriter(f"runs/runs_SM_model/{run_name}")

# Training configuration: number of epochs and early stopping patience
epochs = 100                    # Maximum number of training epochs to run
patience = 10                   # Number of epochs to wait before stopping if validation loss doesn't improve
best_val_loss = float('inf')    # Track the best validation loss observed during training
early_stop_counter = 0          # Counter to track epochs without improvement in validation loss

start_time = time.time()        # Record the start time of training

print(f"{'='*30}\nTraining model with early stopping...\n{'='*30}")

# Training loop
for epoch in range(epochs):
    # Training phase
    model.train()           # Set the model to training mode
    train_loss = 0          # Cumulative training loss
    train_correct = 0       # Count of correctly classified training examples
    train_total = 0         # Total number of training examples
    y_true_train = []       # Store true labels for training
    y_pred_train = []       # Store predicted labels for training

    # Iterate over training batches
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()               # Clear the gradient buffers
        outputs = model(X_batch)            # Forward pass
        loss = criterion(outputs, y_batch)  # Compute the loss
        loss.backward()                     # Backward pass to compute gradients
        optimizer.step()                    # Update the model parameters

        train_loss += loss.item()               # Accumulate loss
        _, predicted = outputs.max(1)           # Get the class with the highest probability
        train_correct += (predicted == y_batch).sum().item()    # Count correct predictions
        train_total += y_batch.size(0)          # Count total examples
        y_true_train.extend(y_batch.tolist())   # Append true labels
        y_pred_train.extend(predicted.tolist()) # Append predicted labels

    # Calculate training accuracy and average loss
    train_accuracy = train_correct / train_total
    train_loss /= len(train_loader)

    # Validation phase
    model.eval()        # Set the model to evaluation mode
    val_loss = 0        # Cumulative validation loss
    val_correct = 0     # Count of correctly classified validation examples
    val_total = 0       # Total number of validation examples
    y_true_val = []     # Store true labels for validation
    y_pred_val = []     # Store predicted labels for validation

    # Disable gradient computation for validation
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

    # Calculate validation accuracy and average loss
    val_accuracy = val_correct / val_total
    val_loss /= len(test_loader)

    # Log metrics to TensorBoard for visualization
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/validation", val_loss, epoch)
    writer.add_scalar("Accuracy/train", train_accuracy, epoch)
    writer.add_scalar("Accuracy/validation", val_accuracy, epoch)

    # Early stopping logic: check if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss    # Update best validation loss
        early_stop_counter = 0      # Reset early stopping counter
    else:
        early_stop_counter += 1     # Increment early stopping counter

    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    # Trigger early stopping if no improvement for 'patience' epochs
    if early_stop_counter >= patience:
        print(f"{'='*30}\nEarly stopping triggered after {epoch+1} epochs.\n")
        break

# Final train metrics
train_metrics = {
    "Accuracy": train_accuracy,
    "Precision": precision_score(y_true_train, y_pred_train, average="macro"),
    "Recall": recall_score(y_true_train, y_pred_train, average="macro"),
    "F1-Score": f1_score(y_true_train, y_pred_train, average="macro"),
    "Log Loss": log_loss(y_true_train, torch.softmax(torch.tensor(model(X_train_tensor).detach().numpy()), dim=1).numpy())
}

# Final validation metrics
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

# Calculate additional metrics on the test set
test_metrics = {
    "Accuracy": accuracy_score(y_true_test, y_pred_test),
    "Precision": precision_score(y_true_test, y_pred_test, average="macro"),
    "Recall": recall_score(y_true_test, y_pred_test, average="macro"),
    "F1-Score": f1_score(y_true_test, y_pred_test, average="macro"),
    "Log Loss": log_loss(y_true_test, torch.softmax(torch.tensor(model(X_test_tensor).detach().numpy()), dim=1).numpy())
}

# Display results
print(f"{'='*30}\nTest Accuracy: {test_metrics['Accuracy']:.4f}")
print(f"Test Precision: {test_metrics['Precision']:.4f}")
print(f"Test Recall: {test_metrics['Recall']:.4f}")
print(f"Test F1-Score: {test_metrics['F1-Score']:.4f}")
print(f"Test Log Loss: {test_metrics['Log Loss']:.4f}")

# Save metrics to .txt
output_file = f"runs_SM_model/{run_name}.txt"
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

print(f"\nMetrics saved to {output_file}")

# Close TensorBoard writer
writer.close()
