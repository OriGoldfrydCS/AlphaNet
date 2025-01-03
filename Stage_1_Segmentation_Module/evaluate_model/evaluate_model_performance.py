import os
import cv2
import pandas as pd
from skimage.metrics import structural_similarity as ssim

class ModelEvaluator:
    """
    A class to evaluate a segmentation model's performance by comparing
    segmented character images with ground-truth character images using SSIM.

    This class also tracks mismatches, identifies mismatches by font type,
    and computes overall segmentation accuracy.
    """
    def __init__(self, csv_path):
        """
        Initialize the model evaluator with the path to the font information CSV.

        Args:
            csv_path (str):
                Path to a CSV containing columns such as 'letters_folder_path' and 'font'.
                The CSV maps each folder (containing letters) to the font used.
        """
        self.csv_path = csv_path        # Store the path to the CSV file
        self.font_mismatches = {}       # Dictionary to track mismatch counts per font

    def load_font_data(self):
        """
        Load the font information from the CSV file.
        The CSV is expected to have a column 'letters_folder_path' that indicates
        the folder name and a 'font' column that indicates the font type.

        Returns:
            dict:
                A dictionary mapping folder basename -> font type.
                E.g. {'sentence_1': 'arial.ttf', 'sentence_2': 'times.ttf', ...}
        """
        df = pd.read_csv(self.csv_path)
        # Only keep the base folder name as key (remove any preceding paths)
        font_data = {os.path.basename(k): v for k, v in df.set_index('letters_folder_path')['font'].to_dict().items()}
        return font_data

    def calculate_ssim(self, image1, image2, crop_margin=1):
        """
          Calculate the SSIM (Structural Similarity Index) between two images.
          The function can accept file paths or actual image arrays.

          Args:
              image1 (str or np.ndarray):
                  File path to the first image or the image array itself.
              image2 (str or np.ndarray):
                  File path to the second image or the image array itself.
              crop_margin (int, optional):
                  Number of pixels to crop from each edge of the images
                  before resizing and computing SSIM. Defaults to 1.

          Returns:
              float:
                  SSIM score between 0 and 1, where 1 indicates identical images.
          """
        # ----------------
        # Handle image1
        # ----------------
        if isinstance(image1, str):
            if not os.path.exists(image1):
                print(f"Image not found: {image1}")
                return 0
            image1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)       # Read in grayscale

        # ----------------
        # Handle image2
        # ----------------
        if isinstance(image2, str):
            if not os.path.exists(image2):
                print(f"Image not found: {image2}")
                return 0
            image2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)       # Read in grayscale

        # Crop edges by 'crop_margin' to remove potential border artifacts
        h, w = image1.shape
        image1 = image1[crop_margin:h-crop_margin, crop_margin:w-crop_margin]
        h, w = image2.shape
        image2 = image2[crop_margin:h-crop_margin, crop_margin:w-crop_margin]

        # Resize both images to a standard size (28x28) for fair comparison
        image1 = cv2.resize(image1, (28, 28))
        image2 = cv2.resize(image2, (28, 28))

        # Compute SSIM
        score, _ = ssim(image1, image2, full=True)
        return score

    def evaluate(self, segmented_dir, ground_truth_dir, write_mismatches=0, output_file="Stage_1_Segmentation_Module/evaluate_model/mismatches.txt"):
        """
             Evaluate the performance of the segmentation model by comparing the
             segmented character images to ground-truth images.

             Args:
                 segmented_dir (str):
                     Path to the directory containing subfolders of segmented character images.
                     Each subfolder typically corresponds to one word or sentence.
                 ground_truth_dir (str):
                     Path to the directory containing the corresponding ground-truth
                     character images subfolders.
                 output_file (str, optional):
                     Path to the .txt file where mismatch details will be logged.
                     Defaults to "Stage_1_Segmentation_Module/evaluate_model/mismatches.txt".

             Returns:
                 float:
                     The overall accuracy percentage of the segmentation, defined as
                     (correctly_segmented / total_letters) * 100.
             """
        # If requested to write mismatches, ensure the output directory exists
        if write_mismatches == 1:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                f.write("Evaluation started...\n")

        # Initialize counters
        total_sentence = 0
        total_letters = 0
        correctly_segmented = 0
        mismatches = []     # To track mismatches with SSIM scores

        # Load font data
        font_data = self.load_font_data()

        # Evaluate each sentence folder
        for sentences_folder in os.listdir(ground_truth_dir):
            # Compose paths for ground truth and segmented images
            ground_truth_path = os.path.join(ground_truth_dir, sentences_folder)
            segmented_path = os.path.join(segmented_dir, sentences_folder)

            # If there's no corresponding segmented folder, skip
            if not os.path.exists(segmented_path):
                continue

            # List out all image files in the ground truth and segmented folders, sorted by index
            ground_truth_files = sorted(os.listdir(ground_truth_path), key=lambda x: int(os.path.splitext(x)[0]))
            segmented_files = sorted(os.listdir(segmented_path), key=lambda x: int(os.path.splitext(x)[0]))

            total_sentence += 1                             # One more sentence processed
            total_letters += len(ground_truth_files)     # How many letters are expected for this sentence

            gt_index = 0                                 # Index for ground truth
            seg_index = 0                                # Index for segmented files

            # Compare segmented chars with ground truth chars
            while seg_index < len(segmented_files) and gt_index < len(ground_truth_files):
                segmented_file_path = os.path.join(segmented_path, segmented_files[seg_index])
                gt_path = os.path.join(ground_truth_path, ground_truth_files[gt_index])

                # Calculate SSIM
                ssim_score = self.calculate_ssim(segmented_file_path, gt_path)

                #  If SSIM is above a threshold, consider it a match
                if ssim_score > 0.2:
                    correctly_segmented += 1
                    gt_index += 1
                    seg_index += 1
                else:
                    # There's a mismatch => Option to log it
                    font_type = font_data.get(os.path.basename(sentences_folder), "Unknown")
                    if font_type == "Unknown":
                        print(f"Font type not found for: {sentences_folder}")
                    self.font_mismatches[font_type] = self.font_mismatches.get(font_type, 0) + 1
                    mismatches.append((segmented_files[seg_index], sentences_folder, font_type, ground_truth_files[gt_index], ssim_score))
                    gt_index += 1
                    seg_index += 1

            # If writing mismatches, append to the mismatch file if any
            if write_mismatches == 1:
                with open(output_file, "a") as f:
                    if mismatches:
                        f.write("\nMismatches:\n")
                        for seg_file, w_folder, f_type, gt_file, score in mismatches:
                            f.write(
                                f"Segmented file {seg_file} in sentence {w_folder} (Font: {f_type}) "
                                f"did not match ground truth {gt_file}. SSIM: {score:.2f}\n"
                            )
                    else:
                        f.write("\nNo mismatches found.\n")

        # Print mismatches by font type
        print("\nMismatches by Font Type:")
        for font, count in sorted(self.font_mismatches.items(), key=lambda x: -x[1]):
            print(f"Font: {font}, Mismatches: {count}")

        # Identify font with the fewest mismatches
        if self.font_mismatches:
            least_mismatches_font = min(self.font_mismatches.items(), key=lambda x: x[1])
            print(f"\nFont with the Fewest Mismatches: {least_mismatches_font[0]} ({least_mismatches_font[1]} mismatches)")

        # Print summary
        accuracy = (correctly_segmented / total_letters) * 100 if total_letters > 0 else 0
        print("\nEvaluation Results:")
        print(f"Total Sentence Processed: {total_sentence}")
        print(f"Total Letters Expected: {total_letters}")
        print(f"Correctly Segmented Letters: {correctly_segmented}")
        print(f"Accuracy: {accuracy:.2f}%")

        return accuracy


if __name__ == "__main__":
    segmented_dir = "Stage_1_Segmentation_Module/evaluate_model/sentences_output_for_evaluate/"
    ground_truth_dir = "Stage_1_Segmentation_Module/data/letters_images/"
    csv_path = "Stage_1_Segmentation_Module/data/sentences_data_5000.csv"

    evaluator = ModelEvaluator(csv_path)
    print("\nEvaluating segmentation results...")
    evaluator.evaluate(segmented_dir, ground_truth_dir)
