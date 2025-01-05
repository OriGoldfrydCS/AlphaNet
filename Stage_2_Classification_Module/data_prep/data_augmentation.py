import pandas as pd
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import matplotlib.pyplot as plt
import os

"""
    This script demonstrates how to:
    1. Augment an image dataset by applying random transformations (rotation, scaling, translation, etc.).
    2. Display some images from the dataset.
    3. Save the augmented dataset to a new CSV file.
    
    It contains two main functions:
    - augment_image(): Applies multiple random transformations to a single 28x28 image.
    - display_images(): Shows a grid of images from a pandas DataFrame.
    
    In the main execution block, it:
    - Loads a CSV of training data (labels + pixels).
    - Applies augmentations to each image.
    - Combines the augmented data with the original.
    - Saves the combined data to a new CSV.
    - Optionally displays a selection of augmented images.
"""

def augment_image(label, pixels, size=28):
    """
    Applies random transformations to a single grayscale image.

    Transformations:
    ----------------
    1. Rotation: ±15 degrees
    2. Scaling: 90% to 110%, then resized back to (28,28)
    3. Translation: Up to ±2 pixels in x and y directions
    4. Brightness adjustment: 0.8x to 1.2x
    5. Contrast adjustment: 0.8x to 1.2x
    6. Gaussian blur: Radius between 0 and 1.5
    7. (Potential for) Inversion or other transforms if needed

    Parameters
    ----------
    label : int or str
        The class label associated with this image.
    pixels : array-like
        The flattened 1D pixel array (28 * 28 = 784 values).
    size : int, optional
        Image width/height, default is 28 for 28x28 images.

    Returns
    -------
    list
        A list containing the label, followed by the augmented pixels (still flattened).
        If there's an error, returns the original label + pixels.
    """
    try:
        # Convert the flattened pixel array into a 2D NumPy array of shape (size, size)
        image_array = np.array(pixels).reshape((size, size)).astype('uint8')

        # Convert to a PIL Image
        image = Image.fromarray(image_array)

        # Randomly rotate the image by ±15 degrees
        if random.choice([True, False]):
            image = image.rotate(random.uniform(-15, 15))

        # Randomly scale the image by 90% to 110%, then resize back to (28,28)
        if random.choice([True, False]):
            scale_factor = random.uniform(0.9, 1.1)
            new_size = (int(size * scale_factor), int(size * scale_factor))
            # Resize to the new size first
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            # Resize back to original
            image = image.resize((size, size), Image.Resampling.LANCZOS)

        # Randomly translate (shift) the image by up to ±2 pixels
        if random.choice([True, False]):
            offset_x = random.randint(-2, 2)
            offset_y = random.randint(-2, 2)
            image = image.transform(image.size, Image.AFFINE, (1, 0, offset_x, 0, 1, offset_y))

        # Randomly adjust the brightness of the image (0.8x to 1.2x)
        if random.choice([True, False]):
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))

        # Randomly adjust the contrast of the image (0.8x to 1.2x)
        if random.choice([True, False]):
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))

        # Randomly apply Gaussian blur with a radius between 0 and 1.5
        if random.choice([True, False]):
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(0, 1.5)))

        # Convert the PIL Image back to a flattened NumPy array
        augmented_array = np.array(image).flatten()

        # Return the label plus the augmented pixels
        return [label] + augmented_array.tolist()

    # Print an error if something goes wrong, and return the original data
    except Exception as e:
        print(f"Error processing image for label {label}: {e}")
        return [label] + pixels.tolist()

def display_images(data, rows=10, cols=10, size=28):
    """
    Displays a grid of images from a pandas DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with one row per image. The first column is the label,
        and the next columns are the pixel values (flattened 28x28).
    rows : int, optional
        Number of rows in the display grid (default=10).
    cols : int, optional
        Number of columns in the display grid (default=10).
    size : int, optional
        The image width/height in pixels (default=28).
    """

    # Create a figure with a grid of subplots (rows x cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))

    # Flatten axes so we can index them in a single loop
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        # If we still have images left in 'data', display them
        if i < len(data):
            # Label is in column 0; pixel values are from columns 1
            label = data.iloc[i, 0]  # First column is the label
            pixels = data.iloc[i, 1:].values.reshape((size, size))  
            ax.imshow(pixels, cmap='gray')
            ax.set_title(f"Label: {label}")
            ax.axis('off')
        else:
            # Hide any extra subplot axes if we don't have enough images
            ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_path = 'Stage_2_Classification_Module/data/train_dataset.csv'
    
    # Check if file exists
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{train_path} not found. Please ensure the path is correct.")

    # Read the CSV file into a pandas DataFrame (no header row)
    print("Loading the combined training dataset...")
    train_data = pd.read_csv(train_path, header=None)
    print(f"Original combined train dataset shape: {train_data.shape}")

    # Separate the labels (first column) from the pixel values (remaining columns)
    labels = train_data.iloc[:, 0].values   # First column is the label
    pixels = train_data.iloc[:, 1:].values  # Remaining columns are pixel values

    # Apply augmentation to each row
    print("Applying augmentations to each image...")
    augmented_data = []
    for label, pixel_row in zip(labels, pixels):
        augmented_data.append(augment_image(label, pixel_row))

    # Convert augmented data back to a DataFrame
    augmented_data = pd.DataFrame(augmented_data, columns=train_data.columns)

    # Combine original and augmented data
    combined_data = pd.concat([train_data, augmented_data], ignore_index=True)
    
    # Print the shape of the new augmented dataset
    print(f"Augmented dataset shape: {combined_data.shape}")

    # Save the augmented dataset
    augmented_dataset_path = 'Stage_2_Classification_Module/data/augmented_train_dataset.csv'
    combined_data.to_csv(augmented_dataset_path, index=False, header=False)
    print(f"Augmented dataset saved successfully at {augmented_dataset_path}!")

    # For debugging: display the first 100 augmented images
    # display_images(augmented_data.head(100), rows=10, cols=10, size=28)