import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from collections import Counter

"""
    This script analyzes the class distribution in a CSV dataset of 32 classes:
    - Letters: A-Z (0-25)
    - Punctuation: ',', '.', '!', '?', ';' (26-30)
    - Space: (31)
    
    Steps:
    1. load_csv() -> Reads the data from a CSV file.
    2. generate_labels() -> Returns a dictionary mapping class indices (0-31) to their characters.
    3. analyze_class_distribution() -> Calculates how many samples are in each class, shows a table (Plotly),
       and plots a bar chart (matplotlib).
    4. main() -> Calls these functions, specifying the file path to the dataset.
"""

# Load the dataset
def load_csv(file_path):
    """
    Reads the dataset from a CSV file and returns it as a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        The path to the CSV file to be loaded.

    Returns
    -------
    pd.DataFrame
        The loaded dataset with no headers. Each row represents one sample.
    """
    data = pd.read_csv(file_path, header=None)
    print(f"Dataset '{file_path}' loaded successfully with {data.shape[0]} samples.")
    return data

# Generate character labels: A-Z (0-25), comma, dot, exclamation, question, semicolon (26-30), space (31)
def generate_labels():
    """
    Generates a dictionary that maps each class index (0-31) to its character:
      - 0-25 -> A-Z
      - 26 -> ,
      - 27 -> .
      - 28 -> !
      - 29 -> ?
      - 30 -> ;
      - 31 ->  (space)

    Returns
    -------
    dict
        Keys: integers (0 through 31)
        Values: corresponding characters (A-Z, punctuation, or space).
    """
    return {
        i: chr(65 + i) if i < 26 else [',', '.', '!', '?', ';', ' '][i - 26]
        for i in range(32)
    }

def analyze_class_distribution(data, dataset_name):
    """
    Counts how many samples appear in each class (0 to 31), shows a summary,
    and plots a table plus a bar chart of class percentages.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset where the first column is the class label (0-31),
        and the rest are pixel/data columns.
    dataset_name : str
        Name of the dataset, used in titles (e.g., "Original Dataset").

    Returns
    -------
    pd.DataFrame
        A DataFrame with four columns:
          - Char: the character representation of the class
          - Class: the integer class label
          - Number of Samples: how many times that class appears
          - Percentage (%): fraction of the total samples in that class
    """
    labels = data.iloc[:, 0]                            # Extract the first column, which holds the labels (0-31)

    # Count how many samples there are for each class label
    class_counts = Counter(labels)
    total_samples = sum(class_counts.values())

    # Ensure that every class from 0 to 31 is present in our class_counts,
    # even if that class doesn't appear in the data (so it won't be missing)
    for cls in range(32):
        class_counts.setdefault(cls, 0)

    # Calculate what percentage of the dataset each class takes
    class_percentages = {cls: round((count / total_samples) * 100, 2) for cls, count in class_counts.items()}

    # Create a mapping from class index to character (A-Z, punctuation, space).
    labels_map = generate_labels()

    # Count how many letters, punctuation marks, and spaces are in the data.
    #  Letters: classes 0-25
    #  Punctuation: classes 26-30
    #  Space: class 31
    num_letters = sum(class_counts[cls] for cls in range(26))
    num_punctuation = sum(class_counts[cls] for cls in range(26, 31))
    num_spaces = class_counts[31]

    # Print a quick summary to the console
    print(f"\nSummary:")
    print(f"Number of letters: {num_letters}")
    print(f"Number of punctuation marks: {num_punctuation}")
    print(f"Number of spaces: {num_spaces}")

    # Build a DataFrame containing class info for displaying in a table and chart
    class_distribution_df = pd.DataFrame({
        "Char": [labels_map[cls] for cls in range(32)],
        "Class": list(range(32)),
        "Number of Samples": [class_counts[cls] for cls in range(32)],
        "Percentage (%)": [class_percentages[cls] for cls in range(32)]
    })

    # Add total row
    total_row = pd.DataFrame({
        "Char": ["Total"],
        "Class": [None],
        "Number of Samples": [total_samples],
        "Percentage (%)": [100.0]
    })
    class_distribution_df = pd.concat([class_distribution_df, total_row], ignore_index=True)

    # Display table using Plotly
    fig_table = go.Figure(data=[
        go.Table(
            header=dict(
                values=["<b>Char</b>", "<b>Class</b>", "<b>Number of Samples</b>", "<b>Percentage (%)</b>"],
                fill_color='paleturquoise',
                align='left',
                font=dict(size=14, color='black')
            ),
            cells=dict(
                values=[
                    class_distribution_df['Char'],
                    class_distribution_df['Class'],
                    class_distribution_df['Number of Samples'],
                    class_distribution_df['Percentage (%)']
                ],
                fill_color='lavender',
                align='left',
                font=dict(size=12)
            )
        )
    ])
    fig_table.update_layout(title=f"Class Distribution Table for {dataset_name}")
    fig_table.show()

    # Plot percentage distribution
    plt.figure(figsize=(12, 6))

    # Exclude the "Total" row (where Class is None)
    valid_classes = class_distribution_df.dropna(subset=['Class']).copy()
    valid_classes['Class'] = valid_classes['Class'].astype(int)

    plt.bar(valid_classes['Class'], valid_classes['Percentage (%)'], color='skyblue', alpha=0.7)
    plt.title(f"Percentage Distribution in {dataset_name}", fontsize=14)
    plt.xlabel("Class Label", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)

    # Use the character representation on the x-axis, rotated a bit to fit
    plt.xticks(ticks=valid_classes['Class'], labels=valid_classes['Char'], rotation=45, fontsize=10)

    # Add grid lines for easier reading
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    return class_distribution_df

def main():
    file_path = 'Stage_2_Classification_Module/data/original_dataset.csv'
    dataset_name = "Original Dataset"

    print(f"\nProcessing {dataset_name}...")
    data = load_csv(file_path)
    analyze_class_distribution(data, dataset_name)

if __name__ == "__main__":
    main()