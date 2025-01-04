import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
from torchvision import transforms
from Stage_1_Segmentation_Module.segmentation_stage import SegmentationStage
from Stage_1_Segmentation_Module.split_to_chars import SplitToChars
from Stage_2_Classification_Module.model_architecture import CNN
from Stage_2_Classification_Module.classification_stage import ClassificationStage
from Stage_3_Conversion_Module.vec_to_text_stage import VecToTextStage
from Stage_4_Correction_Module.correction_stage import CorrectionStage
from Stage_5_Output_Module.save_to_file_stage import SaveToFileStage
from evaluator import Evaluation

def main(mode=0):
    """
    Main function to execute the image processing pipeline.

    Parameters:
    - mode (int): Determines the operating mode.
                  0 - Testing mode with logging and evaluation.
                  1 - GUI mode without logging and evaluation.
    """
    # Initialize paths and models based on the selected mode
    if mode == 0:
        # Testing mode: Includes logging and evaluation features
        data_root = "Project_Main/project_dirs/input/"
        working_dir = "Project_Main/project_dirs/working_dir/"
        output_root = "Project_Main/project_dirs/output/"
        logger_path = "Project_Main/project_dirs/logger.txt"
        evaluation_mode = 1
        ground_truth_csv = "Project_Main/project_dirs/sentences_data_5000.csv"

        # Ensure necessary directories exist
        os.makedirs(working_dir, exist_ok=True)
        os.makedirs(output_root, exist_ok=True)

        # Open the logger file in append mode
        logger = open(logger_path, "a")

        # Initialize the evaluator if evaluation mode is enabled
        evaluator = Evaluation(ground_truth_csv) if evaluation_mode else None
    else:
        # GUI mode without logging and evaluation
        data_root = "Project_Main/project_dirs/input_gui/"
        working_dir = "Project_Main/project_dirs/working_gui_dir/"
        output_root = "Project_Main/project_dirs/output_gui/"

        # Ensure necessary directories exist
        os.makedirs(working_dir, exist_ok=True)
        os.makedirs(output_root, exist_ok=True)

        logger = None       # No logging in GUI mode
        evaluator = None    # No evaluation in GUI mode

    # --------------------------
    # Model Initialization
    # --------------------------
    # Path to the pre-trained CNN model for classification
    model_path = "Stage_2_Classification_Module/models/cnn_models/best_model_CNN_2025-01-01.pth"

    # Determine the device to run the model on (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the segmentation stage
    # SplitToChars handles splitting images into individual character images
    segmenter = SplitToChars(padding=10, proportional_padding=False)
    segmentation_stage = SegmentationStage(segmenter, working_dir)

    # Initialize the CNN model for classification
    model = CNN()
    model.load_state_dict(torch.load(model_path, map_location=device))      # Load pre-trained weights
    model.to(device)
    model.eval()                                                            # Set the model to evaluation mode

    # Define the image transformations for the classification stage
    transform = transforms.Compose([
        transforms.Resize((28, 28)),            # Resize images to 28x28 pixels
        transforms.ToTensor(),                  # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,))    # Normalize tensor values
    ])

    # Initialize the classification stage with the CNN model and transformations
    classification_stage = ClassificationStage(model, transform, device)

    # Initialize the conversion stage to convert vectors to text
    vec_to_text_stage = VecToTextStage()

    # Initialize the correction stage, using a language model T5
    correction_stage = CorrectionStage(model_name="t5-large")

    # Initialize the output stage to save the final text results
    save_to_file_stage = SaveToFileStage(output_root)

    # --------------------------
    # Processing Pipeline
    # --------------------------
    # Iterate over each image file in the data root directory
    for image_file in os.listdir(data_root):
        if not image_file.endswith(".png"):
            continue

        # Construct the full path to the image file
        image_path = os.path.join(data_root, image_file)
        sentence_name = os.path.splitext(image_file)[0]

        try:
            # --------------------------
            # Stage 1: Segmentation
            # --------------------------
            # Process the image to segment it into individual character images
            segmented_images = segmentation_stage.process(image_path)

            # --------------------------
            # Stage 2: Classification
            # --------------------------
            # Classify each segmented character image to obtain vector representations
            letter_vector = classification_stage.process(segmented_images)

            # --------------------------
            # Stage 3: Conversion
            # --------------------------
            # Convert the vector representations to recognized text
            recognized_word = vec_to_text_stage.process(letter_vector)

            # --------------------------
            # Stage 4: Correction
            # --------------------------
            # Correct the recognized word using a language model
            corrected_word = correction_stage.process([recognized_word])[0]

            # --------------------------
            # Evaluation (Optional)
            # --------------------------
            if evaluator:
                # Retrieve the ground truth sentence corresponding to the current image
                ground_truth_sentence = evaluator.ground_truth.get(image_file, "")

                # Update evaluation metrics based on the corrected word
                correct, total, mismatches = evaluator.update_metrics(corrected_word, image_path)
                corrected_accuracy = correct / total if total > 0 else 0

                if logger:
                    # Prepare a log entry with the final output and accuracy details
                    log_entry = (
                        f"Final cleaned output: {corrected_word.upper()}\n"
                        f"Ground Truth Sentence: {ground_truth_sentence}\n"
                        f"Accuracy for {image_file}: {corrected_accuracy:.2f}\n"
                    )
                    # Add a separator based on accuracy
                    log_entry += "*" * 55 if corrected_accuracy == 1.0 else "-" * 55
                    log_entry += "\n"

                    # Write the log entry to the logger file
                    logger.write(log_entry)
                    logger.flush()

                # Decide whether to use the corrected word based on accuracy
                final_word = corrected_word if corrected_accuracy > 0.5 else recognized_word
            else:
                # If no evaluation, use the corrected word directly
                final_word = corrected_word

            # --------------------------
            # Stage 5: Save to File
            # --------------------------
            save_to_file_stage.process((final_word, sentence_name))

        # Handle any exceptions that occur during processing
        except Exception as e:
            if logger:
                logger.write(f"Error processing {image_file}: {e}\n")
                logger.flush()
            print(f"Error processing {image_file}: {e}")

    # --------------------------
    # Post-Processing
    # --------------------------
    # Display the evaluation results after processing all images
    if evaluator:
        evaluator.display_results()

    if logger:
        # Close the logger file to free system resources
        logger.close()

if __name__ == "__main__":
    main()
