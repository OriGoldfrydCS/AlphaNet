import pandas as pd
import os
from difflib import SequenceMatcher

class Evaluation:
    """
    This class handles the evaluation process for the pipeline using character-level similarity.
    """
    def __init__(self, ground_truth_csv):
        self.overall_correct = 0
        self.overall_total = 0
        self.overall_mismatches = []

        # Load ground truth from CSV file
        self.ground_truth = self._load_ground_truth(ground_truth_csv)

    def _load_ground_truth(self, csv_path):
        """
        Load the ground truth sentences from the CSV file.
        """
        data = pd.read_csv(csv_path)
        return {os.path.basename(row['sentence_image_path']): row['sentence'] for _, row in data.iterrows()}

    def evaluate_accuracy(self, predicted_sentence, ground_truth_sentence):
        """
        Evaluate the accuracy using SequenceMatcher ratio.
        """
        # Compute similarity ratio (ignoring case and extra whitespace)
        predicted_sentence = predicted_sentence.strip().upper()
        ground_truth_sentence = ground_truth_sentence.strip().upper()
        
        similarity = SequenceMatcher(None, predicted_sentence, ground_truth_sentence).ratio()
        
        # Calculate the number of correct letters based on ratio
        correct = int(similarity * len(ground_truth_sentence))
        total = len(ground_truth_sentence)

        # Generate mismatches using a character-wise comparison
        mismatches = []
        for i, (p, g) in enumerate(zip(predicted_sentence, ground_truth_sentence)):
            if p != g:
                mismatches.append({
                    "position": i + 1,
                    "predicted_letter": p,
                    "ground_truth_letter": g,
                    "type": "Mismatch"
                })
                
        return correct, total, mismatches

    def update_metrics(self, predicted_sentence, image_path):
        """
        Update overall metrics after processing each sentence.
        """
        image_name = os.path.basename(image_path)  # Extract the filename (e.g., sentence_1.png)
        ground_truth_sentence = self.ground_truth.get(image_name, "")
        
        # Evaluate using the modified method
        correct, total, mismatches = self.evaluate_accuracy(predicted_sentence, ground_truth_sentence)
        self.overall_correct += correct
        self.overall_total += total
        self.overall_mismatches.extend([(image_path, mismatch) for mismatch in mismatches])
        return correct, total, mismatches

    def display_results(self):
        """
        Display overall evaluation results.
        """
        accuracy = (self.overall_correct / self.overall_total) * 100 if self.overall_total > 0 else 0
        print("\nEvaluation Results:")
        print(f"Overall Accuracy: {accuracy:.2f}%")
        print(f"Total Correct Letters: {self.overall_correct}/{self.overall_total}")
