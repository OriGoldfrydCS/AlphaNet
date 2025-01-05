import pandas as pd
import os
from sklearn.model_selection import train_test_split

"""
    This script loads a CSV dataset containing labeled data, splits it into
    training and test sets while keeping the class distribution consistent,
    and then saves those sets to separate CSV files.
"""

def create_train_test_datasets(full_data_path, train_path, test_path, test_size=0.2):
    """
    Loads a full dataset from a CSV file, splits it into training and test sets,
    then saves each split to its own CSV file.

    Parameters
    ----------
    full_data_path : str
        The file path to the original, full dataset CSV.
    train_path : str
        The file path where the training split CSV will be saved.
    test_path : str
        The file path where the test split CSV will be saved.
    test_size : float, optional
        The fraction of the dataset to allocate to the test split, by default 0.2 (20%).

    Returns
    -------
    None
        This function saves the resulting train and test DataFrames to disk
        and prints relevant status messages to the console.
    """
    # Load the dataset from the specified CSV file
    print("Loading the full dataset...")
    df = pd.read_csv(full_data_path, header=None)
    print(f"Dataset loaded with shape: {df.shape}")

    # In this dataset, the first column (index 0) is the label
    label_col = 0
    labels = df[label_col]

    # For debugging: print how many samples exist for each class label before splitting
    print("Class distribution before split:")
    class_counts = labels.value_counts()
    for c, cnt in class_counts.items():
        print(f" Class {c}: {cnt} samples")

    # Split the dataset into train and test while preserving class distribution
    print("Splitting dataset into train and test...")
    train_dataset, test_dataset = train_test_split(
        df, test_size=test_size, random_state=42, stratify=labels
    )

    print(f"Train dataset shape: {train_dataset.shape}")
    print(f"Test dataset shape: {test_dataset.shape}")

    # Make sure the directories for train and test outputs exist (create if needed)
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)

    # Save the training set to a CSV without row indexes or headers
    print(f"Saving train dataset to: {train_path}")
    train_dataset.to_csv(train_path, index=False, header=False)

    # Save the test set similarly
    print(f"Saving test dataset to: {test_path}")
    test_dataset.to_csv(test_path, index=False, header=False)

    print("Train and test datasets have been created and saved successfully.")

if __name__ == "__main__":
    full_data = "Stage_2_Classification_Module/data/original_dataset.csv"
    train_output = "Stage_2_Classification_Module/data/train_dataset.csv"
    test_output = "Stage_2_Classification_Module/data/test_dataset.csv"

    create_train_test_datasets(full_data, train_output, test_output)
