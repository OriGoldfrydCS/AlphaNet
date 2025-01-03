from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, log_loss, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize TensorBoard writer
log_dir = f"ViT_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
writer = SummaryWriter(f"Stage_2_Classification_Module/runs/runs_ViT_model/{log_dir}")

# Load data
train_file_path = "Stage_2_Classification_Module/data/augmented_train_dataset.csv"
test_file_path = "Stage_2_Classification_Module/data/test_dataset.csv"

# CSV file paths for training and test datasets
# The CSV files have no header row; first column is label, next columns are pixels
train_data = pd.read_csv(train_file_path, header=None)
test_data = pd.read_csv(test_file_path, header=None)

# Load training and test data into pandas DataFrames
X_train_full = train_data.iloc[:, 1:].values
y_train_full = train_data.iloc[:, 0].values

# Separate features (X) and labels (y)
# X_* = pixel values; y_* = class labels
X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

# Normalize the pixel values from [0,255] to [0,1], and reshape
# to (N, 1, 28, 28) for single-channel (grayscale) images
X_train_full = (X_train_full / 255.0).reshape(-1, 1, 28, 28).astype('float32')
X_test = (X_test / 255.0).reshape(-1, 1, 28, 28).astype('float32')

#Split the training data into a smaller train set and a validation set.
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

#Custom Dataset class to wrap the NumPy arrays with optional transforms
class AtoZDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        A custom Dataset that holds images (as numpy arrays) and their labels.
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """
        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Fetches a single sample by index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            (image, label): Transformed image and label as Tensors.
        """
        # Retrieve the data (grayscale image) and its label
        image = self.data[idx]      # shape: (1, 28, 28)
        label = self.labels[idx]

        # If it's a NumPy array, convert it to a PIL Image
        if isinstance(image, np.ndarray):
            # Multiply by 255 to restore [0,255] range, then convert to uint8.
            # Squeeze removes the single channel dimension -> (28, 28).
            image = Image.fromarray((image * 255).astype(np.uint8).squeeze(), mode='L')

        # If a transform is provided, apply it (e.g., resizing, normalization)
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


# Define image transforms for training, validation, and testing
# - During training, we can apply data augmentations such as random flip/rotation
# - For validation/test, we generally only resize and normalize
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),              # Resize to match ViT input size
    transforms.RandomHorizontalFlip(),          # Randomly flip image horizontally
    transforms.RandomRotation(10),              # Random rotation up to ±10°
    transforms.ToTensor(),                      # Convert PIL.Image to Tensor [C, H, W]
    transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize grayscale images
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),              # Resize to match ViT input size
    transforms.ToTensor(),                      # Convert PIL.Image to Tensor
    transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize grayscale images
])

# Wrap the data arrays in AtoZDataset objects and create DataLoaders
# shuffle=True for training to randomize sample order.
batch_size = 64
train_dataset = AtoZDataset(X_train, y_train, transform=train_transform)
val_dataset = AtoZDataset(X_val, y_val, transform=val_test_transform)
test_dataset = AtoZDataset(X_test, y_test, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define a Vision Transformer (ViT) model adapted for grayscale images
class ViTGrayscale(nn.Module):
    """
    A Vision Transformer that handles single-channel (grayscale) inputs
    and outputs logits for a specified number of classes
    """
    def __init__(self, num_classes=32):
        """
          Args:
              num_classes (int): How many output classes.
          """
        super(ViTGrayscale, self).__init__()
        # Load a pretrained ViT model from torchvision, specifying the weights.
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        self.vit = models.vit_b_16(weights=weights)

        # Replace the input projection to handle 1 channel instead of 3.
        self.vit.conv_proj = nn.Conv2d(
            in_channels=1,          # (1) The input now has only 1 channel instead of 3 (RGB)
            out_channels=768,       # (2) Keep the output dimension at 768, matching ViT-B-16’s patch embedding size
            kernel_size=16,         # (3) Each patch is 16x16 in spatial size
            stride=16,              # (4) Move the kernel 16 pixels at a time to produce non-overlapping patches
            bias=False              # (5) No bias term here; the linear transformation suffices for patch embedding
        )

        # Replace the final classification (head) to match the number of classes
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
         Forward pass for the Vision Transformer.

         Args:
             x (Tensor): Input of shape [batch_size, 1, 224, 224].

         Returns:
             Tensor: Logits of shape [batch_size, num_classes].
         """
        return self.vit(x)

# Function to initialize the ViT model
def initialize_vit(num_classes=32, pretrained=True):
    model = ViTGrayscale(num_classes=num_classes)
    # If not using the pretrained weights, manually initialize params
    if not pretrained:
        for param in model.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    return model.to(device)

# Create an instance of the ViT model, using pretrained weights, and define
# the optimizer, loss function, and scheduler
model = initialize_vit(num_classes=32, pretrained=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

# Mixed precision training
scaler = torch.cuda.amp.GradScaler()

# Training hyperparameters and early stopping setup
epochs = 100
patience = 10
best_val_loss = float('inf')
early_stop_counter = 0

print("Training Vision Transformer...")

# --------------------------------------------------------------------------
# 1. Training loop
# --------------------------------------------------------------------------
for epoch in range(epochs):
    model.train()               # Put model in training mode
    running_loss = 0.0          # Accumulate total training loss
    correct_train = 0           # Count how many are predicted correctly
    total_train = 0             # Track total samples processed

    # tqdm gives a progress bar for each batch in train_loader
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()       # Clear the gradient from the previous step

            # Forward pass under autocast for mixed precision
            with torch.amp.autocast(device_type='cuda'): 
                outputs = model(inputs)             # Forward pass through ViT
                loss = criterion(outputs, labels)   # Compute cross-entropy loss

            scaler.scale(loss).backward()           # Backprop with scaled gradients for mixed precision.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)      # Clip gradients to avoid exploding gradients

            # Update parameters using the scaled optimizer step
            scaler.step(optimizer)
            scaler.update()

            # Accumulate training metrics
            running_loss += loss.item()                             # Accumulate the loss
            _, predicted = torch.max(outputs, 1)                    # Determine predicted class
            correct_train += (predicted == labels).sum().item()     # Count correct predictions
            total_train += labels.size(0)                           # Increment total samples seen

            # Update progress bar info
            pbar.set_postfix({'Train Acc': f'{(correct_train / total_train):.2%}'})
            pbar.update(1)

    # Compute overall training accuracy and average loss
    train_accuracy = correct_train / total_train
    avg_train_loss = running_loss / len(train_loader)

    # --------------------------------------------------------------------------
    # 2. Validation phase
    # --------------------------------------------------------------------------
    model.eval()        # Switch to evaluation mode
    val_loss = 0.0      # Initialize the cumulative validation loss
    correct_val = 0     # Initialize a counter for correct predictions
    total_val = 0       # Track the total number of validation samples processed

    # Disable gradient calculations during validation
    with torch.no_grad():
        # Loop through all batches in the validation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)                 # Forward pass
            loss = criterion(outputs, labels)       # Compute the validation loss
            val_loss += loss.item()                 # Accumulate (add) the validation loss for this batch
            _, predicted = torch.max(outputs, 1)    # Determine the predicted class
            correct_val += (predicted == labels).sum().item()       # Update the count of correctly predicted samples
            total_val += labels.size(0)             # Add the number of samples in this batch to the total

    # Calculate the overall validation accuracy (correct predictions / total samples)
    val_accuracy = correct_val / total_val
    avg_val_loss = val_loss / len(val_loader)

    # Log metrics to TensorBoard
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('Loss/validation', avg_val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_accuracy, epoch)
    writer.add_scalar('Accuracy/validation', val_accuracy, epoch)
    
    print(f"Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.2%}")

    # Adjust the learning rate according to the schedule
    scheduler.step()

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), f"Stage_2_Classification_Module/models/Vit_models/best_model_{log_dir}.pth")
    else:
        early_stop_counter += 1
        print(f"Early stopping counter: {early_stop_counter}")
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

# Load best model
model.load_state_dict(torch.load(f"Stage_2_Classification_Module/models/Vit_models/best_model_{log_dir}.pth"))

# --------------------------------------------------------------------------
# 3. Test loop
# --------------------------------------------------------------------------
model.eval()        # Put the model in evaluation mode
correct = 0         # Counter for correctly classified samples
total = 0           # Counter for total samples seen
test_preds, test_probs, test_labels = [], [], []         # Lists to store predictions, probabilities, and labels

# No gradient tracking is needed during testing
with torch.no_grad():
    # Loop over all batches in the test DataLoader
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)     # Forward pass

        # Convert the logits into probabilities using softmax along dimension 1 (classes dimension)
        probs = outputs.softmax(dim=1).cpu().numpy()

        # Determine the class with the highest logit for each sample
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)                          # Update the total count of samples
        correct += (predicted == labels).sum().item()    # Count how many predictions matched the actual labels

        test_preds.extend(predicted.cpu().numpy())       # Append class predictions to a list
        test_probs.extend(probs)                         # Append probability distributions
        test_labels.extend(labels.cpu().numpy())         # Append the true labels

# Compute final test metrics: Accuracy, Precision, Recall, F1, LogLoss
test_accuracy = correct / total
precision = precision_score(test_labels, test_preds, average="macro")
recall = recall_score(test_labels, test_preds, average="macro")
f1 = f1_score(test_labels, test_preds, average="macro")
logloss = log_loss(test_labels, test_probs)
conf_matrix = confusion_matrix(test_labels, test_preds)

print(f"\nViT Model Performance:")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1-Score: {f1:.4f}")
print(f"Test Log Loss: {logloss:.4f}")

# Plot confusion matrix
labels = [chr(i) for i in range(65, 91)] + [",", ".", "!", "?", ";", "Space"]

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