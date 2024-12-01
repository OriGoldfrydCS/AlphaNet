import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix

# Function to load CSV file
def load_csv_file(filepath):
    data = pd.read_csv(filepath, header=None)   # Assuming no header in the CSV
    labels = data.iloc[:, 0].to_numpy()         # First column is the label
    images = data.iloc[:, 1:].to_numpy()        # Remaining columns are the image data
    return images, labels

# Load train and test datasets
train_images, y_train = load_csv_file("data/train_dataset.csv")
test_images, y_test = load_csv_file("data/test_dataset.csv")

# Find the majority class in the training set
(unique, counts) = np.unique(y_train, return_counts=True)

# Print the number of samples for each class
print("Number of samples for each class in the training set:")
for label, count in zip(unique, counts):
    print(f"Class {label}: {count} samples")

# Find the majority class
majority_class = unique[np.argmax(counts)]
print(f"Majority class in the training set: {majority_class}")

# Predict the majority class for all instances in the test set. This baseline model predicts the majority class for every test sample
y_pred_baseline = np.full_like(y_test, majority_class)

# Compute performance metrics
accuracy = accuracy_score(y_test, y_pred_baseline)
precision = precision_score(y_test, y_pred_baseline, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred_baseline, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred_baseline, average='macro', zero_division=0)  # Added

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_baseline) 

# Compute Log Loss
# Log loss measures the uncertainty of the predictions using the predicted probabilities
num_classes = len(unique)                                           # Number of unique classes (26 for letters A-Z)
y_prob_baseline = np.zeros((len(y_test), num_classes))              # Initialize a zero matrix for probabilities
y_prob_baseline[:, np.where(unique == majority_class)[0][0]] = 1    # Set probabilities to 1 for the majority class
logloss = log_loss(y_test, y_prob_baseline, labels=unique)          # Compute log loss based on probabilistic predictions

# Display performance metrics
print("\nBaseline Model Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")  
print(f"Log Loss: {logloss:.4f}")  

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,                 # Annotate each cell with the count             
    fmt="d",                    # Display numbers as integers
    cmap="Blues",
    xticklabels=np.arange(26),  # Set x-axis labels as 0-25
    yticklabels=np.arange(26),  # Set y-axis labels as 0-25
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()