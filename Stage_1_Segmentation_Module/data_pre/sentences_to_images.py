import os
import csv
import random
from PIL import Image, ImageDraw, ImageFont

# Define paths and directories for:
#   1) Input text file with sentences
#   2) Output folders for sentence and letter images
#   3) Output CSV file where we log sentence info and paths
input_txt_path = 'Stage_1_Segmentation_Module/data/sentences_dataset_5000.txt'
output_word_images_folder = 'Stage_1_Segmentation_Module/data/sentences_images/'
output_letter_images_folder = 'Stage_1_Segmentation_Module/data/letters_images/'
output_csv_path = 'Stage_1_Segmentation_Module/data/sentences_data_5000.csv'

# Fonts directory
fonts_dir = "C:/Windows/Fonts"

# List of fonts to include
included_fonts = [
    "couri.ttf", "palab.ttf", "verdanab.ttf", "consolai.ttf", "ariali.ttf", 
    "consola.ttf", "arialbi.ttf", "arial.ttf", "l_10646.ttf", "verdanaz.ttf",
    "consolab.ttf", "ahronbd.ttf", "malgunsl.ttf", "AniMeMatrix-MB_EN.ttf", 
    "segoeui.ttf", "seguibl.ttf", "Nirmala.ttf", "lucon.ttf", "ntailu.ttf", 
    "segoeuil.ttf", "taile.ttf", "seguiemj.ttf", "consolaz.ttf", "malgun.ttf", 
    "LeelUIsl.ttf", "seguihis.ttf", "NirmalaS.ttf", "gadugi.ttf", "segoeuiz.ttf", 
    "gisha.ttf", "seguisb.ttf", "LeelawUI.ttf", "phagspa.ttf", "seguisym.ttf", 
    "SegUIVar.ttf", "trebucit.ttf", "segoeuisl.ttf", "mmrtext.ttf", "ebrima.ttf", 
    "lvnmbd.ttf"
]

# Function to get included font paths
def get_included_fonts(fonts_directory, included_fonts):
    """
    Scans the specified fonts directory for valid font files (.ttf, .otf)
    and returns only those matching the filenames listed in 'included_fonts'.

    Args:
        fonts_directory (str): Path to the directory containing font files.
        included_fonts (list): List of font filenames (case-insensitive) to include.

    Returns:
        list: Full paths to the included font files.
    """
    # Gather all .ttf or .otf files in the specified directory
    font_files = [
        os.path.join(fonts_directory, f) for f in os.listdir(fonts_directory)
        if f.lower().endswith(('.ttf', '.otf'))  # Include only valid font files
    ]

    # Filter the above list by checking if the base filename is in 'included_fonts'
    included_font_paths = [
        font for font in font_files
        if os.path.basename(font).lower() in included_fonts
    ]
    return included_font_paths

# Get included font paths
font_paths = get_included_fonts(fonts_dir, included_fonts)

# Ensure output directories exist
os.makedirs(output_word_images_folder, exist_ok=True)
os.makedirs(output_letter_images_folder, exist_ok=True)

# Load sentences from the input text file
with open(input_txt_path, 'r') as file:
    all_sentences = list({line.strip() for line in file})   # Use a set to ensure unique sentences

# Font sizes
word_font_size = 50
letter_font_size = 28

# Prepare and open the CSV file for writing. We'll store:
#   1) The original sentence text
#   2) The path to the rendered sentence image
#   3) The path to the folder containing the letter images
#   4) The font used to render these images
with open(output_csv_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['sentence', 'sentence_image_path', 'letters_folder_path', 'font'])  # CSV header

    # Loop over each sentence in the list of unique sentences
    # and generate:
    #   - A single image containing the entire sentence
    #   - A folder of letter images (28x28) for each character
    #   - A line in the CSV mapping them together
    for idx, sentence in enumerate(all_sentences, start=1):
        # Randomly choose a font for each sentence
        selected_font_path = random.choice(font_paths)
        font = ImageFont.truetype(selected_font_path, word_font_size)

        # Generate a sequential file name
        sentence_name = f"sentence_{idx}"
        sentence_image_path = f"{output_word_images_folder}/{sentence_name}.png"

        # -------------------------------------------------------------------
        # 1) CREATE THE SENTENCE IMAGE
        # -------------------------------------------------------------------
        # Create a dummy image to measure the text's bounding box
        dummy_img = Image.new('L', (1, 1), color=255)
        draw = ImageDraw.Draw(dummy_img)
        text_bbox = draw.textbbox((0, 0), sentence, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Add some padding around the text
        padding = 20  # Add padding around the text
        img_width = text_width + 2 * padding
        img_height = text_height + 2 * padding

        # Create the final image with calculated dimensions
        img = Image.new('L', (img_width, img_height), color=255)  # Grayscale mode, white background
        draw = ImageDraw.Draw(img)

        # Center the text horizontally and vertically
        text_x = (img_width - text_width) // 2
        text_y = (img_height - text_height) // 2 - text_bbox[1]

        # Render the sentence in black (fill=0)
        draw.text((text_x, text_y), sentence, font=font, fill=0)

        # Save the complete sentence image as PNG
        img.save(sentence_image_path)

        # -------------------------------------------------------------------
        # 2) CREATE LETTER IMAGES
        # -------------------------------------------------------------------
        letters_folder_path = f"{output_letter_images_folder}/{sentence_name}"
        os.makedirs(letters_folder_path, exist_ok=True)

        # Create an ImageFont object for individual letters (smaller size)
        letter_font = ImageFont.truetype(selected_font_path, letter_font_size)

        # Loop over each character in the sentence
        for index, char in enumerate(sentence, start=1):
            # Initialize a 28x28 grayscale image with a white background
            letter_img = Image.new('L', (28, 28), color=255)  # 28x28 grayscale image
            draw = ImageDraw.Draw(letter_img)

            # If the character is a space, we just save a blank 28x28 image
            if char == ' ':  # Create an empty image for space
                letter_img.save(f"{letters_folder_path}/{index}.png")
            else:
                # Measure the bounding box for this character
                letter_bbox = draw.textbbox((0, 0), char, font=letter_font)

                # Calculate the x/y positioning to center the character
                letter_x = (28 - (letter_bbox[2] - letter_bbox[0])) // 2
                letter_y = (28 - (letter_bbox[3] - letter_bbox[1])) // 2 - letter_bbox[1]

                # Draw the character in black (fill=0)
                draw.text((letter_x, letter_y), char, font=letter_font, fill=0)

                # Save the letter image as "1.png", "2.png", etc.
                letter_img.save(f"{letters_folder_path}/{index}.png")

        # -------------------------------------------------------------------
        # 3) WRITE TO THE CSV
        # -------------------------------------------------------------------
        # Store the sentence, the path to the sentence image,
        # the folder path with letter images, and the chosen font.
        writer.writerow([sentence,
                         sentence_image_path.replace("\\", "/"),
                         letters_folder_path.replace("\\", "/"),
                         selected_font_path.replace("\\", "/")
        ])

print(f"Sentence images saved in '{output_word_images_folder}'")
print(f"Letter images saved in '{output_letter_images_folder}'")
print(f"CSV file created at '{output_csv_path}'")