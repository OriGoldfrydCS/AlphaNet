import pandas as pd
import os

class Evaluation:
    """
    Handles the evaluation process for the pipeline.
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
        Evaluate the accuracy by comparing each predicted letter with the ground truth.
        """
        correct = 0
        total = len(ground_truth_sentence)
        mismatches = []
        used_indices = set()  # Track ground truth indices already matched

        for i, predicted_letter in enumerate(predicted_sentence):
            if i < total and predicted_letter == ground_truth_sentence[i]:
                # Matches the current ground truth letter
                correct += 1
                used_indices.add(i)
            else:
                # Check for forward match
                matched = False
                for j in range(i + 1, total):  # Look ahead in the ground truth sentence
                    if j not in used_indices and predicted_letter == ground_truth_sentence[j]:
                        correct += 1
                        used_indices.add(j)
                        matched = True
                        break

                if not matched:
                    # Record mismatch details
                    current_gt_letter = ground_truth_sentence[i] if i < total else "N/A"
                    next_gt_letters = ground_truth_sentence[i + 1:] if i + 1 < total else "N/A"
                    mismatches.append({
                        "position": i + 1,
                        "predicted_letter": predicted_letter,
                        "current_gt_letter": current_gt_letter,
                        "next_gt_letters": next_gt_letters,
                        "type": "Mismatch"
                    })

        return correct, total, mismatches

    def update_metrics(self, predicted_sentence, image_path):
        """
        Update overall metrics after processing each sentence.
        """
        image_name = os.path.basename(image_path)  # Extract the filename (e.g., sentence_1.png)
        ground_truth_sentence = self.ground_truth.get(image_name, "")
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
