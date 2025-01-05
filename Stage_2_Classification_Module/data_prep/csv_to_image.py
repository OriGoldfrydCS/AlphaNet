import os
import pandas as pd
from PIL import Image

"""
    This script reads a CSV file that contains image data (label + pixel values),
    then converts each row of pixel data into a 28x28 image, and saves those images
    subfolders named after their labels.
"""

# Path to the CSV file
input_csv = "Stage_2_Classification_Module/data/augmented_train_dataset"

# Output directory for saving images
output_dir = "Stage_2_Classification_Module/data/images/"

# Create the main output directory if it doesn't exist, so we can save images there
os.makedirs(output_dir, exist_ok=True)

# Read the CSV file into a pandas DataFrame
data = pd.read_csv(input_csv, header=None)

# Loop through each row of the DataFrame
# idx is the row index, row is the actual data in that row
for idx, row in data.iterrows():
    label = int(row.iloc[0])                        # The first column (iloc[0]) is the label
    pixels = row.iloc[1:].values.astype('uint8')    # The remaining columns are the pixel values, which we convert to type uint8

    # Reshape the 1D pixel array (of length 784) into a 2D array of shape 28x28
    image_array = pixels.reshape(28, 28)

    # Create a subfolder named after the label
    # For instance, if label=0, it goes into "Stage_2_Classification_Module/data/images/0
    class_folder = os.path.join(output_dir, str(label))
    os.makedirs(class_folder, exist_ok=True)

    # Convert the NumPy array to a PIL Image
    image = Image.fromarray(image_array)

    # Construct the output file name, e.g., "image_0.png" for the first row (idx=0)
    image_path = os.path.join(class_folder, f"image_{idx}.png")
    image.save(image_path)

    # For debugging: Print progress every 100 rows to keep track of how many images are processed
    # if idx % 100 == 0:
    #     print(f"Processed {idx}/{len(data)} images.")

print("All images saved successfully!")
