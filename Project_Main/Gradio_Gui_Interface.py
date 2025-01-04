import gradio as gr
import os
import sys
import shutil
from PIL import Image, ImageDraw, ImageFont
from Main_Pipeline import main

class GradioInterface:
    def __init__(self):
        """
        Initializes the GradioInterface by setting up directory paths and available fonts.
        Ensures that all necessary directories exist.
        """
        # Define directory paths for various stages of processing
        self.input_dir = "Project_Main/project_dirs/input_gui/"
        self.working_dir = "Project_Main/project_dirs/working_gui_dir/"
        self.output_dir = "Project_Main/project_dirs/output_gui/"
        self.preview_dir = "Project_Main/project_dirs/preview_gui/"
        self.generated_dir = "Project_Main/project_dirs/generated_gui/"

        # Create directories if they do not exist
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.preview_dir, exist_ok=True)
        os.makedirs(self.generated_dir, exist_ok=True)

        # List of included fonts available for generating images from text
        self.included_fonts = [
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

    def resize_image(self, image_path, size=(256, 256)):
        """
        Resizes an image to fit within the specified size while maintaining aspect ratio.
        Pads the image with a white background if necessary and ensures it is in RGB mode.

        Parameters:
        - image_path (str): Path to the original image.
        - size (tuple): Desired size as (width, height).

        Returns:
        - str or None: Path to the resized image in the preview directory or None if an error occurs.
        """
        try:
            with Image.open(image_path) as img:
                # Convert image to RGB to preserve color information
                img = img.convert("RGB")

                # Maintain aspect ratio using the Resampling.LANCZOS filter
                img.thumbnail(size, Image.Resampling.LANCZOS)

                # Create a new image with a white background
                new_img = Image.new("RGB", size, (255, 255, 255))

                # Calculate position to center the thumbnail
                paste_position = (
                    (size[0] - img.width) // 2,
                    (size[1] - img.height) // 2
                )

                # Paste the resized image onto the white background
                new_img.paste(img, paste_position)

                # Save the resized image to the preview directory
                preview_path = os.path.join(self.preview_dir, os.path.basename(image_path))
                new_img.save(preview_path, format="PNG")

                print(f"Resized image saved to: {preview_path}")

                return preview_path
        except Exception as e:
            print(f"Error resizing image {image_path}: {e}")
            return None

    def process_images(self, images):
        """
        Processes one or more images through the main pipeline.
        Copies images to the input directory, runs the pipeline, and gathers the output.

        Parameters:
        - images (list or str or tuple): List of image file paths or a single image file path.

        Returns:
        - tuple: Combined processed text and list of output text file paths.
        """        # Ensure 'images' is always a list
        if not isinstance(images, list):
            images = [images]

        # List to keep track of successfully copied image file paths
        saved_files = []
        for img in images:
            if img:     # Check if the file is not None
                # Handle both string paths and tuples (in case of Gradio returning tuples)
                if isinstance(img, tuple):
                    img_path = img[0]  # Assuming the first element is the filepath
                else:
                    img_path = img

                if isinstance(img_path, str):
                    # Define the destination path in the input directory
                    file_path = os.path.join(self.input_dir, os.path.basename(img_path))
                    try:
                        shutil.copy(img_path, file_path)  # Copy the image to the input directory
                        saved_files.append(file_path)
                        print(f"Copied image to: {file_path}")
                    except Exception as e:
                        print(f"Error copying image {img_path} to {file_path}: {e}")
                else:
                    print(f"Unsupported image format: {img_path}")

        print(f"Processing {len(saved_files)} images...")

        # Run the main pipeline in 'GUI mode' (mode=1)
        main(mode=1)

        # List to hold paths to the generated output text files
        output_files = []
        for file_name in os.listdir(self.output_dir):
            if file_name.endswith(".txt"):      # Only consider text files
                full_path = os.path.join(self.output_dir, file_name)
                output_files.append(full_path)

        # Combine the contents of all output text files into a single string
        output_str = ""
        for txt_file_path in output_files:
            try:
                with open(txt_file_path, "r", encoding="utf-8") as f:
                    file_text = f.read()
                    output_str += f"==> {os.path.basename(txt_file_path)}:\n{file_text}\n\n"
            except Exception as e:
                output_str += f"==> {os.path.basename(txt_file_path)}:\nError reading file: {e}\n\n"

        # Return the combined text and the list of text files
        return output_str, output_files

    def clear_directories(self):
        """
           Clears the input, working, output, preview, and generated directories by forcefully deleting their contents.
           Recreates the directories after deletion to ensure they exist for future operations.
           """
        def force_delete_directory(directory):
            try:
                shutil.rmtree(directory, ignore_errors=False)  # Raises an error if it fails
                print(f"Successfully cleared directory: {directory}")
            except PermissionError as e:
                print(f"Permission Error: {directory} cannot be fully cleared. Reason: {e}")
            except Exception as e:
                print(f"Error clearing directory {directory}. Reason: {e}")

        # Iterate through all directories and clear them
        for directory in [self.input_dir, self.working_dir, self.output_dir, self.preview_dir, self.generated_dir]:
            # Delete the directory and its contents
            force_delete_directory(directory)
            # Recreate the directory to ensure it exists for future use
            os.makedirs(directory, exist_ok=True)

    def clear_uploaded_and_generated_images(self):
        """
         Clears uploaded images from the input directory and generated images from the generated directory.

         Returns:
         - str: Success or error message indicating the result of the operation.
         """
        try:
            # Clear all files in the input directory
            for filename in os.listdir(self.input_dir):
                file_path = os.path.join(self.input_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted uploaded/generated image: {file_path}")

            # Clear all files in the generated directory
            for filename in os.listdir(self.generated_dir):
                file_path = os.path.join(self.generated_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted generated image: {file_path}")

            return "✅ Uploaded and generated images have been cleared successfully!"
        except Exception as e:
            error_msg = f"❌ Error clearing images: {e}"
            print(error_msg)
            return error_msg

    def get_available_fonts(self):
        """
        Retrieves the list of available fonts for generating images from text.

        Returns:
        - list: List of font filenames.
        """
        return self.included_fonts

    def generate_image_from_text(self, text, font_name, image_size=(256, 256), font_size=25):
        """
        Generates an image from the provided text using the specified font.

        Parameters:
        - text (str): The sentence to convert into an image.
        - font_name (str): The filename of the font to use.
        - image_size (tuple): Size of the generated image as (width, height).
        - font_size (int): Size of the font.

        Returns:
        - tuple: (success_flag (bool), image_path or error message (str))
        """
        try:
            # Check if the selected font is in the list of included fonts
            if font_name not in self.included_fonts:
                error_msg = f"Font not allowed: {font_name}"
                print(error_msg)
                return False, error_msg

            # Load the specified TrueType font
            font = ImageFont.truetype(font_name, font_size)

            # Define extra padding to be added on both sides of the text
            extra_padding = 50
            space_width = font.getbbox(" ")[2] * 2

            # Calculate the total width required for the text
            text_width = sum(
                font.getbbox(char)[2] - font.getbbox(char)[0] if char != " " else space_width for char in text)

            # Calculate the total width of the image by adding padding on both sides
            padded_width = text_width + extra_padding * 2

            # Create a new white RGB image with the calculated width and specified height
            img = Image.new('RGB', (padded_width, image_size[1]), color=(255, 255, 255))
            d = ImageDraw.Draw(img)

            # Initialize the starting position for drawing text
            x_text = extra_padding
            y_text = (image_size[1] - font.getbbox(text)[3]) // 2

            # Iterate through each character in the text to draw it on the image
            for char in text:
                if char == " ":
                    x_text += space_width  # Move cursor for space
                else:
                    # Draw the character at the current position
                    d.text((x_text, y_text), char, font=font, fill=(0, 0, 0))
                    # Update the x position based on the width of the character
                    x_text += font.getbbox(char)[2] - font.getbbox(char)[0]

            # Save the image
            image_filename = f"generated_{len(os.listdir(self.generated_dir)) + 1}.png"
            image_path = os.path.join(self.generated_dir, image_filename)
            img.save(image_path, format="PNG")

            return True, image_path

        except Exception as e:
            error_msg = f"Error generating image from text '{text}': {e}"
            print(error_msg)
            return False, error_msg

    def wrap_text(self, text, font, max_width):
        """
         Wraps the input text into multiple lines so that each line does not exceed the specified maximum width.

         Parameters:
         - text (str): The text to wrap.
         - font (ImageFont): The font used to measure text width.
         - max_width (int): The maximum width allowed for each line.

         Returns:
         - list: List of wrapped lines.
         """
        words = text.split()
        lines = []
        current_line = words[0]

        for word in words[1:]:
            test_line = f"{current_line} {word}"
            # Check if the width of the test line exceeds the maximum width
            if font.getbbox(test_line)[2] <= max_width:
                current_line = test_line         # Append the word to the current line
            else:
                lines.append(current_line)       # Add the current line to lines
                current_line = word              # Start a new line with the current word

        lines.append(current_line)               # Add the last line to lines
        return lines

    def generate_word_image(self, text, font_name):
        """
           Generates an image from the provided text and copies it to the input directory for processing.

           Parameters:
           - text (str): The sentence to convert into an image.
           - font_name (str): The filename of the font to use.

           Returns:
           - tuple: (status_message (str), list of image paths or None)
           """
        # If the input text is empty or contains only whitespace, prompt the user
        if not text.strip():
            return "⚠️ Please enter a sentence.", None

        # Convert text to uppercase
        text = text.upper()
        print(f"Converted input text to uppercase: {text}")

        # Generate the image from text using the specified font
        success, result = self.generate_image_from_text(text, font_name)

        if success:
            generated_image_path = result
            destination_path = os.path.join(self.input_dir, os.path.basename(generated_image_path))
            try:
                # Copy the generated image to the input directory for processing
                shutil.copy(generated_image_path, destination_path)
                print(f"Copied generated image to: {destination_path}")
                # Return status and updated gallery
                return "✅ Image generated successfully!", [destination_path]
            except Exception as e:
                error_msg = f"Error copying generated image: {e}"
                print(error_msg)
                return "✅ Image generated successfully!", None
        else:
            # result contains the error message
            return f"❌ Failed to generate image. {result}", None

    def launch(self):
        """
        Sets up and launches the Gradio interface with multiple tabs for uploading, generating, processing images,
        and managing directories. Applies custom CSS for styling and defines the layout and interactions.
        """
        def clear_directories_wrapper():
            """
            Wrapper function to clear all relevant directories and return a success message.
            """
            self.clear_directories()
            return "✅ Directories cleared successfully!"

        def preview_images(file_paths):
            """
            Resizes uploaded images for uniform preview and returns a list of resized image paths.

            Parameters:
            - file_paths (list or str): List of file paths or a single file path.

            Returns:
            - list: List of resized image paths.
            """
            if not isinstance(file_paths, list):
                file_paths = [file_paths]

            preview_images = []
            for fp in file_paths:
                if fp is not None:
                    resized_path = self.resize_image(fp)
                    if resized_path:
                        try:
                            with Image.open(resized_path) as img:
                                preview_images.append(img.copy())   # Append a copy to avoid closed file issues
                                print(f"Added image to preview: {resized_path}")
                        except Exception as e:
                            print(f"Error loading resized image {resized_path}: {e}")

            return preview_images

        def clear_outputs():
            """
            Clears the processed output textbox and the download files component.

            Returns:
            - tuple: Empty string and empty list to reset the UI components.
            """
            return "", []

        def generate_word_image_wrapper(text, font_name):
            """
              Wrapper function to generate a word image and update the gallery.

              Parameters:
              - text (str): The sentence to convert into an image.
              - font_name (str): The filename of the font to use.

              Returns:
              - tuple: Status message and list of image paths.
              """
            status_msg, updated_gallery = self.generate_word_image(text, font_name)
            return status_msg, updated_gallery

        def clean_uploads_and_generated_images():
            """
              Clears uploaded and generated images and returns a status message along with resetting the gallery.

              Returns:
              - tuple: Status message and empty list to reset the gallery.
              """
            message = self.clear_uploaded_and_generated_images()
            return message, []

        # Custom CSS for styling the Gradio interface
        custom_css = """
        <style>
            /* General Styles */
            body {
                background-color: #1e1e1e;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                color: #f0f0f0;
            }
            .header {
                background-color: #3f51b5;
                color: #ffffff;
                padding: 20px;
                text-align: center;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                animation: fadeInDown 1s ease-out;
            }
            @keyframes fadeInDown {
                from {
                    opacity: 0;
                    transform: translateY(-50px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            /* Button Styles */
            .button-primary {
                background-color: #ff5722;
                color: #ffffff;
                border: none;
                padding: 12px 24px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 8px 4px;
                border-radius: 8px;
                cursor: pointer;
                transition: background-color 0.3s, transform 0.2s;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            }
            .button-primary:hover {
                background-color: #e64a19;
                transform: translateY(-2px);
            }
            .button-secondary {
                background-color: #607d8b;
                color: #ffffff;
                border: none;
                padding: 12px 24px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 8px 4px;
                border-radius: 8px;
                cursor: pointer;
                transition: background-color 0.3s, transform 0.2s;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            }
            .button-secondary:hover {
                background-color: #455a64;
                transform: translateY(-2px);
            }
            /* Gallery Styles */
            .gallery img {
                border: 3px solid #ff5722;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                transition: transform 0.3s, box-shadow 0.3s;
            }
            .gallery img:hover {
                transform: scale(1.05);
                box-shadow: 0 8px 12px rgba(0, 0, 0, 0.5);
            }
            /* Textbox Styles */
            textarea {
                background-color: #2c2c2c;
                color: #ffcc80;
                border: 2px solid #ff5722;
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
                resize: vertical;
                transition: border-color 0.3s;
            }
            textarea:hover, textarea:focus {
                border-color: #ff9800;
            }
            /* Tab Title Styles */
            .tab-title {
                color: #ff9800;
                font-size: 24px;
                text-align: center;
                margin-bottom: 20px;
            }
            /* Status Bar Styles */
            #status_bar, #clean_status {
                background-color: #2c2c2c;
                color: #ff9800;
                border: none;
                padding: 10px;
                border-radius: 8px;
                font-size: 16px;
                text-align: center;
            }
            /* Loader Styles */
            .loader {
                border: 8px solid #f3f3f3;
                border-top: 8px solid #ff5722;
                border-radius: 50%;
                width: 60px;
                height: 60px;
                animation: spin 1s linear infinite;
                margin: auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            /* Responsive Design */
            @media (max-width: 768px) {
                .header {
                    padding: 15px;
                }
                .button-primary, .button-secondary {
                    padding: 10px 20px;
                    font-size: 14px;
                }
                .tab-title {
                    font-size: 20px;
                }
            }
            /* Generate Section Styles */
            .generate-section {
                background-color: #2c2c2c;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                margin-top: 20px;
            }
            .generate-section h3 {
                color: #ff5722;
            }
        </style>
        """

        with gr.Blocks() as interface:
            # Inject custom CSS to style the interface
            gr.HTML(custom_css)

            # Header Section
            with gr.Row():
                with gr.Column(scale=1):
                    # Placeholder for an optional logo image
                    pass
                with gr.Column(scale=3):
                    # Display the main header using Markdown
                    gr.Markdown(
                        """
                        <div class="header">
                            <h1>✨ Image to Text Processing System</h1>
                        </div>
                        """,
                        elem_id="header"
                    )

            ########################################################
            #  1) TAB FOR UPLOADING IMAGES & PROCESSING THEM
            ########################################################
            # Title for the Upload & Process tab
            with gr.Tab("Upload & Process"):
                gr.Markdown(
                    """
                    <div class="tab-title">
                        📂 Upload your image(s) and process them
                    </div>
                    """,
                    elem_id="upload-title"
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        # File Upload Section
                        upload_box = gr.File(
                            file_types=[".png"],             # Restrict uploads to PNG files
                            label="📁 Upload Image(s)",      # Label for the upload component
                            type="filepath",                 # Return file paths instead of raw data
                            file_count="multiple",           # Allow multiple file uploads
                            interactive=True                 # Make the component interactive
                        )

                        # Use a Gallery to preview multiple images
                        image_gallery = gr.Gallery(
                            label="🖼️ Images Preview",      # Label for the gallery
                            show_label=True,                # Display the label
                            columns=2,                      # Number of columns in the gallery
                            elem_id="gallery"               # HTML element ID for styling
                        )

                        # Update the Gallery when files change
                        upload_box.change(
                            fn=preview_images,              # Function to call on change
                            inputs=upload_box,              # Input is the upload box
                            outputs=image_gallery           # Output is the image gallery
                        )

                        # Clean Button under Image Preview
                        clean_button = gr.Button(
                            "🧼 Clean Images",               # Button label
                            elem_id="clean-button",          # HTML element ID for styling
                            elem_classes="button-secondary") # CSS class for secondary styling

                        # Textbox to display the status of the clean operation
                        clean_status = gr.Textbox(
                            label="🧹 Clean Status",         # Label for the textbox
                            interactive=False,               # Make the textbox read-only
                            value="",                        # Initial value is empty
                            lines=1,                         # Number of lines in the textbox
                            elem_id="clean_status"           # HTML element ID for styling
                        )

                        # Connect the Clean Images button to its functionality
                        clean_button.click(
                            fn=clean_uploads_and_generated_images,      # Function to call on click
                            inputs=None,                                # No inputs required
                            outputs=[clean_status, image_gallery],      # Outputs to update
                        )

                    with gr.Column(scale=1):
                        # Process Images Button
                        process_button = gr.Button(
                            "🚀 Process Images",              # Button label
                            elem_id="process-button",         # HTML element ID for styling
                            elem_classes="button-primary")    # CSS class for secondary styling

                        # Processed Output Textbox
                        processed_output = gr.Textbox(
                            label="📝 Processed Output",     # Label for the textbox
                            lines=10,                        # Number of lines in the textbox
                            interactive=False,               # Make the textbox read-only
                            placeholder="Processed text will appear here..."        # Placeholder text
                        )

                        # Component to allow downloading of processed text files
                        download_files = gr.Files(
                            label="📥 Download Processed Files"      # Label for the download component
                        )

                        # Connect the Process Images button to its functionality
                        process_button.click(
                            fn=self.process_images,         # Function to call on click
                            inputs=upload_box,              # Input is the uploaded files
                            outputs=[processed_output, download_files],     # Outputs to update
                        )

                        # Clear Outputs Button
                        clear_outputs_button = gr.Button(
                            "🧹 Clear Outputs",                  # Button label
                            elem_id="clear-button",              # HTML element ID for styling
                            elem_classes="button-secondary")     # CSS class for secondary styling

                        # Connect the Clear Outputs button to its functionality
                        clear_outputs_button.click(
                            fn=clear_outputs,                   # Function to call on click
                            inputs=None,                        # No inputs required
                            outputs=[processed_output, download_files],     # Outputs to update
                        )


            ########################################################
            #  2) TAB FOR GENERATING SENTENCE-IMAGES & PROCESSING
            ########################################################
            # Title for the Generate & Process tab
            with gr.Tab("Generate & Process"):
                gr.Markdown(
                    """
                    <div class="tab-title">
                        🖋️ Generate your sentence image(s) and process them
                    </div>
                    """,
                    elem_id="generate-title"
                )

                with gr.Row():
                    # Gallery to preview generated images
                    with gr.Column(scale=2):
                        gen_image_gallery = gr.Gallery(
                            label="🖼️ Images Preview",          # Label for the gallery
                            show_label=True,                    # Display the label
                            columns=2,                          # Number of columns in the gallery
                            elem_id="gallery2"                  # HTML element ID for styling
                        )

                        # ----------------------------
                        # Generate Word Image Section
                        # ----------------------------
                        with gr.Group():
                            # Subheading for the generate section
                            gr.Markdown("### 🖋️ Generate Sentence Image")

                            # Textbox for entering the sentence to generate
                            with gr.Row():
                                word_input = gr.Textbox(
                                    label="✏️ Enter Sentence",      # Label for the textbox
                                    placeholder="Type the sentence you want to convert to an image...", # Placeholder text
                                    lines=1                         # Number of lines in the textbox
                                )

                                # Dropdown menu to select the font for image generation
                                font_dropdown = gr.Dropdown(
                                    label="🎨 Select Font",                  # Label for the dropdown
                                    choices=self.get_available_fonts(),      # List of available fonts
                                    # Default selected font
                                    value=self.get_available_fonts()[0] if self.get_available_fonts() else "arial.ttf",
                                    interactive=True                         # Make the dropdown interactive
                                )

                            # Button to generate the image from text
                            generate_button = gr.Button(
                                "🖨️ Generate Image",            # Button label
                                elem_id="generate-button",      # HTML element ID for styling
                                elem_classes="button-primary")  # CSS class for primary styling

                            # Textbox to display the status of the generate operation
                            generate_status = gr.Textbox(
                                label="ℹ️ Status",      # Label for the textbox
                                interactive=False,      # Make the textbox read-only
                                value="",               # Initial value is empty
                                lines=1                 # Number of lines in the textbox
                            )

                            # Connect the Generate Image button to its functionality
                            generate_button.click(
                                fn=generate_word_image_wrapper,                 # Function to call on click
                                inputs=[word_input, font_dropdown],             # Inputs from the textbox and dropdown
                                outputs=[generate_status, gen_image_gallery],   # Outputs to update
                            )

                        # Clean Images Button specific to the Generate & Process tab
                        clean_button_2 = gr.Button(
                            "🧼 Clean Images",                   # Button label
                            elem_id="clean-button2",             # HTML element ID for styling
                            elem_classes="button-secondary")     # CSS class for secondary styling

                        # Textbox to display the status of the clean operation
                        clean_status_2 = gr.Textbox(
                            label="🧹 Clean Status",             # Label for the textbox
                            interactive=False,                   # Make the textbox read-only
                            value="",                            # Initial value is empty
                            lines=1,                             # Number of lines in the textbox
                            elem_id="clean_status2"              # HTML element ID for styling
                        )

                        # Connect the Clean Images button to its functionality
                        clean_button_2.click(
                            fn=clean_uploads_and_generated_images,          # Function to call on click
                            inputs=None,                                    # No inputs required
                            outputs=[clean_status_2, gen_image_gallery],    # Outputs to update
                        )

                    # Process Images Button for the Generate & Process tab
                    with gr.Column(scale=1):
                        # Process Images Button (for newly generated images)
                        process_button_2 = gr.Button(
                            "🚀 Process Images",             # Button label
                            elem_id="process-button2",       # HTML element ID for styling
                            elem_classes="button-primary")   # CSS class for primary styling

                        # Textbox to display the processed output
                        processed_output_2 = gr.Textbox(
                            label="📝 Processed Output",     # Label for the textbox
                            lines=10,                        # Number of lines in the textbox
                            interactive=False,               # Make the textbox read-only
                            placeholder="Processed text will appear here..."        # Placeholder text
                        )

                        # Component to allow downloading of processed text files
                        download_files_2 = gr.Files(
                            label="📥 Download Processed Files"      # Label for the download component
                        )

                        # Connect the Process Images button to its functionality
                        process_button_2.click(
                            fn=self.process_images,     # Function to call on click
                            inputs=gen_image_gallery,   # Input is the generated image gallery
                            outputs=[processed_output_2, download_files_2],     # Outputs to update
                        )

                        # Clear Outputs Button for the Generate & Process tab
                        clear_outputs_button_2 = gr.Button(
                            "🧹 Clear Outputs",                  # Button label
                            elem_id="clear-button2",             # HTML element ID for styling
                            elem_classes="button-secondary")     # CSS class for secondary styling

                        # Connect the Clear Outputs button to its functionality
                        clear_outputs_button_2.click(
                            fn=clear_outputs,       # Function to call on click
                            inputs=None,            # No inputs required
                            outputs=[processed_output_2, download_files_2],     # Outputs to update
                        )


            ########################################################
            # 3) TAB FOR MANAGING DIRECTORIES
            ########################################################
            # Title for the Manage Directories tab
            with gr.Tab("Manage Directories"):
                gr.Markdown(
                    """
                    <div class="tab-title">
                        🧹 Clear directories to reset the system
                    </div>
                    """,
                    elem_id="manage-title"
                )
                with gr.Row():
                    # Button to clear all directories
                    clear_button = gr.Button(
                        "🗑️ Clear Directories",             # Button label
                        elem_id="manage-clear-button",      # HTML element ID for styling
                        elem_classes="button-secondary")    # CSS class for secondary styling

                    # Textbox to display the status of the directory clear operation
                    status_bar = gr.Textbox(
                        label="✅ Status",                  # Label for the textbox
                        interactive=False,                  # Make the textbox read-only
                        value="",                           # Initial value is empty
                        lines=2,                            # Number of lines in the textbox
                        placeholder="Status messages will appear here...",
                        elem_id="status_bar"                # HTML element ID for styling
                    )

                # Connect the Clear Directories button to its functionality
                clear_button.click(
                    fn=clear_directories_wrapper,           # Function to call on click
                    inputs=None,                            # No inputs required
                    outputs=status_bar,                     # Output to update
                )

            ########################################################
            # 4) TAB FOR ABOUT INFORMATION
            ########################################################
            # Display information about the system using Markdown
            with gr.Tab("About"):
                gr.Markdown(
                    """
                    <div class="tab-title">
                        ℹ️ About This System
                    </div>
                    <div style='text-align: center;'>
                        This application allows you to:
                        <ul style='list-style: none; padding: 0;'>
                            <li>📁 **Upload** image files (.png format).</li>
                            <li>🖋️ **Generate** images from typed sentences with selected fonts.</li>
                            <li>🔍 **Process** these images through a pipeline to extract text.</li>
                            <li>📝 **View** and 📥 **Download** the processed text output (one line per image).</li>
                            <li>🧼 **Clean** uploaded and generated images.</li>
                            <li>🧹 **Clear** directories for resetting the system.</li>
                        </ul>
                        <p><strong>Enjoy using this modern interface! ✨</strong></p>
                    </div>
                    """,
                    elem_id="about-title"
                )

        # Launch the Gradio interface
        interface.launch()

if __name__ == "__main__":
    gui = GradioInterface()
    gui.launch()
