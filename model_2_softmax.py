import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns  

# Load data from CSV files
def load_csv_data(file_path):
    data = pd.read_csv(file_path, header=None)
    labels = data.iloc[:, 0].to_numpy()       # First column is the label
    images = data.iloc[:, 1:].to_numpy()      # Remaining columns are the image data
    return images, labels

# Load training and testing datasets
train_csv_path = "data/train_dataset.csv"  
test_csv_path = "data/test_dataset.csv"   

print("Loading training data...")
X_train, y_train = load_csv_data(train_csv_path)

print("Loading testing data...")
X_test, y_test = load_csv_data(test_csv_path)

# Normalize pixel values to [0, 1]
print("Normalizing data...")
X_train = X_train / 255.0
X_test = X_test / 255.0

# Instantiate and train the logistic regression model
print("Training the logistic regression model...")
logreg = LogisticRegression(solver='saga', max_iter=100, verbose=1, n_jobs=-1)
logreg.fit(X_train, y_train)

# Make predictions on the test set
print("Making predictions on the test data...")
y_pred = logreg.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Generate probabilistic predictions for log loss
y_prob = logreg.predict_proba(X_test)  # Get predicted probabilities
logloss = log_loss(y_test, y_prob)

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Display results
print(f'\nLogistic Regression Model Performance:')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}') 
print(f'Log Loss: {logloss:.4f}')  

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,                 # Annotate each cell with the count
    fmt="d",                    # Display numbers as integers
    cmap="Blues",               # Use a blue color map
    xticklabels=np.arange(26),  # Set x-axis labels as 0-25
    yticklabels=np.arange(26),  # Set y-axis labels as 0-25
)
plt.title("Confusion Matrix")   # Title of the heatmap
plt.xlabel("Predicted Labels")  # X-axis represents predicted labels
plt.ylabel("True Labels")       # Y-axis represents true labels
plt.show()