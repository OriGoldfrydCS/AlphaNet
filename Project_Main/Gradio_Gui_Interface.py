import gradio as gr
import os
import sys
import shutil
from PIL import Image, ImageDraw, ImageFont
from Main_Pipeline import main  

class GradioInterface:
    def __init__(self):
        self.input_dir = "Project_Main/project_dirs/input_gui/"
        self.working_dir = "Project_Main/project_dirs/working_gui_dir/"
        self.output_dir = "Project_Main/project_dirs/output_gui/"
        self.preview_dir = "Project_Main/project_dirs/preview_gui/"  
        self.generated_dir = "Project_Main/project_dirs/generated_gui/"  

        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.preview_dir, exist_ok=True)
        os.makedirs(self.generated_dir, exist_ok=True)

    def resize_image(self, image_path, size=(256, 256)):
        """
        Resize an image to fit within the specified size while maintaining aspect ratio.
        Pads the image with a white background if necessary and ensures it is in RGB mode.
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
        """Process one or more images through the pipeline."""
        # Ensure 'images' is always a list
        if not isinstance(images, list):
            images = [images]

        # Save the uploaded images to the input directory
        saved_files = []
        for img in images:
            if img:  # Check if the file is not None
                file_path = os.path.join(self.input_dir, os.path.basename(img))
                try:
                    shutil.copy(img, file_path)  # Use copy for simplicity
                    saved_files.append(file_path)
                    print(f"Copied uploaded image to: {file_path}")  
                except Exception as e:
                    print(f"Error copying uploaded image {img} to {file_path}: {e}")

        print(f"Processing {len(saved_files)} uploaded images...")

        # Run the main pipeline in 'GUI mode' (mode=1)
        main(mode=1)

        # Gather output files
        output_files = []
        for file_name in os.listdir(self.output_dir):
            if file_name.endswith(".txt"):
                full_path = os.path.join(self.output_dir, file_name)
                output_files.append(full_path)

        # Create a multiline string that contains text for each .txt file
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
        """Clear input, working, output, preview, and generated directories using force deletion."""
        def force_delete_directory(directory):
            try:
                shutil.rmtree(directory, ignore_errors=False)  # Raises an error if it fails
                print(f"Successfully cleared directory: {directory}")
            except PermissionError as e:
                print(f"Permission Error: {directory} cannot be fully cleared. Reason: {e}")
            except Exception as e:
                print(f"Error clearing directory {directory}. Reason: {e}")

        for directory in [self.input_dir, self.working_dir, self.output_dir, self.preview_dir, self.generated_dir]:
            # Ensure the directory is recreated after deletion
            force_delete_directory(directory)
            os.makedirs(directory, exist_ok=True)

    def clear_uploaded_and_generated_images(self):
        """Clear uploaded images and generated word images from input_dir and generated_dir."""
        try:
            # Clear input_dir
            for filename in os.listdir(self.input_dir):
                file_path = os.path.join(self.input_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted uploaded/generated image: {file_path}")

            # Clear generated_dir
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
        """Retrieve a list of available font files from C:/Windows/Fonts."""
        fonts_dir = "C:/Windows/Fonts/"
        available_fonts = []
        try:
            for file in os.listdir(fonts_dir):
                if file.lower().endswith(('.ttf', '.otf')):
                    available_fonts.append(file)
        except Exception as e:
            print(f"Error accessing fonts directory: {e}")
        return available_fonts

    def generate_image_from_text(self, text, font_name, image_size=(256, 256), font_size=40):
        """
        Generate an image from the given text using the specified font.
        Returns a tuple: (success: bool, image_path or error_message: str)
        """
        try:
            fonts_dir = "C:/Windows/Fonts/"
            font_path = os.path.join(fonts_dir, font_name)
            
            # Check if the font file exists
            if not os.path.isfile(font_path):
                error_msg = f"Font file not found: {font_path}"
                print(error_msg)
                return False, error_msg
            
            # Load the font
            font = ImageFont.truetype(font_path, font_size)

            # Create a new image with white background
            img = Image.new('RGB', image_size, color=(255, 255, 255))
            d = ImageDraw.Draw(img)

            # Calculate text bounding box
            bbox = d.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)

            # Draw text onto image
            d.text(position, text, font=font, fill=(0, 0, 0))

            # Save the generated image
            image_filename = f"generated_{len(os.listdir(self.generated_dir)) + 1}.png"
            image_path = os.path.join(self.generated_dir, image_filename)
            img.save(image_path, format="PNG")

            print(f"Generated image saved to: {image_path}")  

            return True, image_path
        except Exception as e:
            error_msg = f"Error generating image from text '{text}': {e}"
            print(error_msg) 
            return False, error_msg

    def generate_word_image(self, text, font_name):
        if not text.strip():
            return "⚠️ Please enter a word.", None
        
        # Convert text to uppercase to enforce capital letters
        text = text.upper()
        print(f"Converted input text to uppercase: {text}")  

        success, result = self.generate_image_from_text(text, font_name)
        
        if success:
            generated_image_path = result
            # Copy the generated image to input_dir for processing
            destination_path = os.path.join(self.input_dir, os.path.basename(generated_image_path))
            try:
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
        """Launch the Gradio interface."""

        def clear_directories_wrapper():
            self.clear_directories()
            return "✅ Directories cleared successfully!"

        def preview_images(file_paths):
            """
            Resize images for uniform preview and return PIL Image objects.
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
                                preview_images.append(img.copy())  # Append a copy to avoid closed file issues
                                print(f"Added image to preview: {resized_path}")  
                        except Exception as e:
                            print(f"Error loading resized image {resized_path}: {e}")

            return preview_images

        def clear_outputs():
            """Clear the Processed Output textbox and Download Processed Files."""
            return "", []

        # Wrapper for generating images with uppercase enforcement
        def generate_word_image_wrapper(text, font_name):
            status_msg, updated_gallery = self.generate_word_image(text, font_name)
            return status_msg, updated_gallery

        # Function to handle the Clean Images button
        def clean_uploads_and_generated_images():
            message = self.clear_uploaded_and_generated_images()
            return message, []

        # Custom CSS for styling
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
            # Inject custom CSS
            gr.HTML(custom_css)

            # Header Section
            with gr.Row():
                with gr.Column(scale=1):
                    # Optional: To add a logo image here
                    pass  
                with gr.Column(scale=3):
                    gr.Markdown(
                        """
                        <div class="header">
                            <h1>✨ Image to Text Processing System</h1>
                        </div>
                        """,
                        elem_id="header"
                    )

            # Upload & Process Tab
            with gr.Tab("Upload & Process"):
                gr.Markdown(
                    """
                    <div class="tab-title">
                        📂 Upload your image(s), generate it and process them
                    </div>
                    """,
                    elem_id="upload-title"
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        # File Upload Section
                        upload_box = gr.File(
                            file_types=[".png"], 
                            label="📁 Upload Image(s)", 
                            type="filepath", 
                            file_count="multiple", 
                            interactive=True
                        )

                        # Use a Gallery to preview multiple images
                        image_gallery = gr.Gallery(
                            label="🖼️ Images Preview",
                            show_label=True,
                            columns=2,  # Number of columns in the gallery
                            elem_id="gallery"
                        )

                        # Update the Gallery when files change
                        upload_box.change(
                            fn=preview_images, 
                            inputs=upload_box, 
                            outputs=image_gallery
                        )

                        # ----------------------------
                        # Generate Word Image Section
                        # ----------------------------
                        with gr.Group():
                            gr.Markdown("### 🖋️ Generate Word Image")

                            with gr.Row():
                                word_input = gr.Textbox(
                                    label="✏️ Enter Word",
                                    placeholder="Type the word you want to convert to an image...",
                                    lines=1
                                )

                                font_dropdown = gr.Dropdown(
                                    label="🎨 Select Font",
                                    choices=self.get_available_fonts(),
                                    value=self.get_available_fonts()[0] if self.get_available_fonts() else "arial.ttf",
                                    interactive=True
                                )

                            generate_button = gr.Button("🖨️ Generate Image", elem_id="generate-button", elem_classes="button-primary")
                            generate_status = gr.Textbox(
                                label="ℹ️ Status",
                                interactive=False,
                                value="",
                                lines=1
                            )

                            # Connect Generate Button
                            generate_button.click(
                                fn=generate_word_image_wrapper,
                                inputs=[word_input, font_dropdown],
                                outputs=[generate_status, image_gallery],
                            )

                        # Clean Button under Image Preview
                        clean_button = gr.Button("🧼 Clean Images", elem_id="clean-button", elem_classes="button-secondary")
                        clean_status = gr.Textbox(
                            label="🧹 Clean Status",
                            interactive=False,
                            value="",
                            lines=1,
                            elem_id="clean_status"
                        )

                        clean_button.click(
                            fn=clean_uploads_and_generated_images,
                            inputs=None,
                            outputs=[clean_status, image_gallery],
                        )

                    with gr.Column(scale=1):
                        # Process Images Button
                        process_button = gr.Button("🚀 Process Images", elem_id="process-button", elem_classes="button-primary")
                        # Processed Output Textbox
                        processed_output = gr.Textbox(
                            label="📝 Processed Output", 
                            lines=10, 
                            interactive=False,
                            placeholder="Processed text will appear here..."
                        )

                        # Download Processed Files
                        download_files = gr.Files(
                            label="📥 Download Processed Files"
                        )

                        # Connect Process Button
                        process_button.click(
                            fn=self.process_images,
                            inputs=upload_box,
                            outputs=[processed_output, download_files],
                        )

                        # Clear Outputs Button
                        clear_outputs_button = gr.Button("🧹 Clear Outputs", elem_id="clear-button", elem_classes="button-secondary")
                        clear_outputs_button.click(
                            fn=clear_outputs,
                            inputs=None,
                            outputs=[processed_output, download_files],
                        )

            # Manage Directories Tab
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
                    clear_button = gr.Button("🗑️ Clear Directories", elem_id="manage-clear-button", elem_classes="button-secondary")
                    status_bar = gr.Textbox(
                        label="✅ Status", 
                        interactive=False, 
                        value="", 
                        lines=2,
                        placeholder="Status messages will appear here...",
                        elem_id="status_bar"
                    )
                clear_button.click(
                    fn=clear_directories_wrapper,
                    inputs=None,
                    outputs=status_bar,
                )

            # About Tab
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
                            <li>🖋️ **Generate** images from typed words with selected fonts.</li>
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
