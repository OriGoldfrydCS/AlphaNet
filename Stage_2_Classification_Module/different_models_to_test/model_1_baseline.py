import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Function to load CSV file
def load_csv_file(filepath):
    """
    Reads a CSV file where:
      - The first column holds the labels.
      - The remaining columns hold image pixel values (flattened).

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    images : np.ndarray
        2D NumPy array of pixel data (num_samples x num_pixels).
    labels : np.ndarray
        1D NumPy array of labels (length = num_samples).
    """
    data = pd.read_csv(filepath, header=None)   # Read csv (no header)
    labels = data.iloc[:, 0].to_numpy()         # The first column is the label column
    images = data.iloc[:, 1:].to_numpy()        # The remaining columns are image data
    return images, labels

# Load the train and test datasets from CSV files
train_images, y_train = load_csv_file("Stage_2_Classification_Module/data/train_dataset.csv")
test_images, y_test = load_csv_file("Stage_2_Classification_Module/data/test_dataset.csv")

# Extract unique classes (e.g., 0-25 for A-Z letters)
unique_classes = np.unique(y_train)

# Convert numeric class IDs (0 -> 'A', 1 -> 'B', etc.) to characters
class_labels = [chr(65 + int(cls)) for cls in unique_classes]

# Find the majority class in the training labels
majority_class = np.bincount(y_train).argmax()

# Set up a TensorBoard writer to log metrics
run_name = f"Baseline_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
writer = SummaryWriter(f"Stage_2_Classification_Module/runs/runs_Baseline_model/{run_name}")

# Predict the majority class for all instances
y_pred_majority = np.full(len(y_test), majority_class)

# Calculate baseline metrics
train_loss = -np.log(1 / len(unique_classes))
val_loss = -np.log(1 / len(unique_classes))
train_accuracy = np.mean(y_train == majority_class)
val_accuracy = accuracy_score(y_test, y_pred_majority)
precision = precision_score(y_test, y_pred_majority, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred_majority, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred_majority, average='macro', zero_division=0)

# Simulate 40 epochs of training by logging the metrics each epoch to TensorBoard
# Since this is a dummy baseline, we keep loss and accuracy constant
for epoch in range(40):
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/validation", val_loss, epoch)
    writer.add_scalar("Accuracy/train", train_accuracy, epoch)
    writer.add_scalar("Accuracy/validation", val_accuracy, epoch)

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_majority)

# Print out the performance metrics on the test setc
print("\nBaseline Model Performance:")
print(f"Test Accuracy: {val_accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1-Score: {f1:.4f}")
print(f"Test Log Loss: {val_loss:.4f}")

# Generate character labels for a full set of 32 classes:
# 0-25 -> A-Z, 26->',', 27->'.', 28->'!', 29->'?', 30->';', 31->'Space'
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
