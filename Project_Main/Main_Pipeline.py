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
    # Initialize paths and model
    if mode == 0:
        # Testing mode
        data_root = "Project_Main/project_dirs/input/"          # Directory containing input word images
        working_dir = "Project_Main/project_dirs/working_dir/"  # Directory to store segmented letter images
        output_root = "Project_Main/project_dirs/output/"       # Directory to save final output text files
        logger = "Project_Main/project_dirs/logger.txt"         # File to log accuracy sentences
        evaluation_mode = 1                                     # Set to 0 to disable evaluation
        ground_truth_csv = "Project_Main/project_dirs/sentences_data_5000.csv"  # Path to CSV file

        # Ensure working and output directories exist
        os.makedirs(working_dir, exist_ok=True)
        os.makedirs(output_root, exist_ok=True)
    else:
        # GUI mode
        data_root = "Project_Main/project_dirs/input_gui/"              # GUI input
        working_dir = "Project_Main/project_dirs/working_gui_dir/"      # GUI working directory
        output_root = "Project_Main/project_dirs/output_gui/"           # GUI output
        logger = ""
        ground_truth_csv = ""

        # Ensure GUI directories exist
        os.makedirs(working_dir, exist_ok=True)
        os.makedirs(output_root, exist_ok=True)

    # Initial configurations
    model_path = "Stage_2_Classification_Module/models/cnn_models/best_model_CNN_2025-01-01.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize pipeline components
    segmenter = SplitToChars(padding=10, proportional_padding=False)
    segmentation_stage = SegmentationStage(segmenter, working_dir)

    model = CNN()                                                       # Load the CNN architecture
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load model weights
    model.to(device)                                                    # Move model to GPU or CPU
    model.eval()                                                        # Set model to evaluation mode

    transform = transforms.Compose([
        transforms.Resize((28, 28)),            # Resize letter images to 28x28
        transforms.ToTensor(),                  # Convert image to tensor
        transforms.Normalize((0.5,), (0.5,))    # Normalize pixel values
    ])

    classification_stage = ClassificationStage(model, transform, device)

    vec_to_text_stage = VecToTextStage()
    correction_stage = CorrectionStage(model_name="t5-large")
    save_to_file_stage = SaveToFileStage(output_root)

    evaluator = Evaluation(ground_truth_csv) if mode == 0 and evaluation_mode else None

    # Open the log file for low accuracy sentences
    with open(logger, "w") as log_file:
        # Execute pipeline for each sentence image in the input folder
        for image_file in os.listdir(data_root):
            if not image_file.endswith(".png"):  # Skip non-PNG files
                continue

            image_path = os.path.join(data_root, image_file)    # Full path to the sentence image
            word_name = os.path.splitext(image_file)[0]         # Extract the word name from the file name

            print(f"Processing word image: {image_path}")

            try:
                # Process the image through segmentation, classification, and correction stages
                segmented_images = segmentation_stage.process(image_path)
                letter_vector = classification_stage.process(segmented_images)
                recognized_word = vec_to_text_stage.process(letter_vector)
                corrected_word = correction_stage.process([recognized_word])[0]
                print(f"Recognized Sentence: {recognized_word}")
                print(f"Corrected Sentence: {corrected_word}")

                if evaluator:
                    # Get ground truth and calculate accuracy
                    ground_truth_sentence = evaluator.ground_truth.get(image_file, "")
                    print(f"Ground Truth Sentence: {ground_truth_sentence}")
                    correct, total, mismatches = evaluator.update_metrics(corrected_word, image_path)
                    corrected_accuracy = correct / total if total > 0 else 0
                    print(f"Accuracy for {image_file}: {corrected_accuracy:.2f}")

                    # Log statistics based on accuracy
                    if corrected_accuracy < 1.0:
                        # Log for low accuracy
                        log_file.write("-------------------------------------------------------\n")
                        # log_file.write(f"Recognized Word: {recognized_word}\n")
                        # log_file.write(f"Corrected Word: {corrected_word}\n")
                        log_file.write(f"Final cleaned output: {corrected_word.upper()}\n")
                        log_file.write(f"Ground Truth Sentence: {ground_truth_sentence}\n")
                        log_file.write(f"Accuracy for {image_file}: {corrected_accuracy:.2f}\n")
                        log_file.write("-------------------------------------------------------\n\n")
                    elif corrected_accuracy == 1.0:
                        # Log for perfect accuracy
                        log_file.write("*******************************************************\n")
                        # log_file.write(f"Recognized Word: {recognized_word}\n")
                        # log_file.write(f"Corrected Word: {corrected_word}\n")
                        log_file.write(f"Final cleaned output: {corrected_word.upper()}\n")
                        log_file.write(f"Ground Truth Sentence: {ground_truth_sentence}\n")
                        log_file.write(f"Accuracy for {image_file}: {corrected_accuracy:.2f}\n")
                        log_file.write("*******************************************************\n\n")

                    # Ensure all data is written immediately
                    log_file.flush()

                    # Decide final word to save based on accuracy
                    if corrected_accuracy <= 0.5:
                        print(f"Low accuracy after correction ({corrected_accuracy:.2f}). Reverting to recognized word.")
                        final_word = recognized_word
                    else:
                        final_word = corrected_word
                else:
                    # No evaluator, just use the corrected word
                    final_word = corrected_word

                # Save the final output to the appropriate directory
                save_to_file_stage.process((final_word, word_name))

            except Exception as e:
                # Handle errors gracefully and log them
                log_file.write(f"Error processing {image_file}: {e}\n")
                log_file.flush()
                print(f"Error processing {image_file}: {e}")

        if evaluator:
            evaluator.display_results()

if __name__ == "__main__":
    main()
