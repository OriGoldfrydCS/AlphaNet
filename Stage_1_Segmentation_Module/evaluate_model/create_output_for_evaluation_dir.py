import os
import sys

# Add the parent directory (Stage_1_Segmentation_Module) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Stage_1_Segmentation_Module.split_to_chars import SplitToChars  # Import from parent directory

# Directories
sentence_images_dir = "Stage_1_Segmentation_Module/data/sentences_images/"
segmented_sentences_dir = "Stage_1_Segmentation_Module/evaluate_model/sentences_output_for_evaluate/"

# Create the output directory for segmented sentences
os.makedirs(segmented_sentences_dir, exist_ok=True)

# Initialize the segmenter
segmenter = SplitToChars(padding=10, proportional_padding=False)

# Process each sentence image
for sentence_image_file in os.listdir(sentence_images_dir):
    sentence_image_path = os.path.join(sentence_images_dir, sentence_image_file)

    # Ensure it's a valid image file
    if not sentence_image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    # Create a subfolder in the output directory for the current sentence
    sentence_name = os.path.splitext(sentence_image_file)[0]  # Remove the file extension to get the sentence name
    sentence_output_folder = os.path.join(segmented_sentences_dir, sentence_name)
    os.makedirs(sentence_output_folder, exist_ok=True)

    # Segment the sentence image and save the results
    print(f"Processing sentence: {sentence_name}")
    segmenter.process_image(sentence_image_path, sentence_output_folder)
