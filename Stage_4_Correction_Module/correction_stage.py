from transformers import T5Tokenizer, T5ForConditionalGeneration
from difflib import SequenceMatcher
import re
import torch


class CorrectionStage:
    """
    This class uses a T5 model to correct recognized sentences.

    It:
    - Loads a T5 model (default is t5-large).
    - Provides a process() method to correct a list of sentences.
    - Cleans the corrected sentences with custom rules.
    - Compares original and corrected sentences to see if the correction is acceptable.
    """

    def __init__(self, model_name="t5-large", max_change_threshold=0.3):
        """
        Sets up the CorrectionStage with a T5 model and a threshold for changes.

        Args:
            model_name (str):
                The name of the T5 model from HuggingFace (default: "t5-large").
            max_change_threshold (float):
                The maximum fraction of changes allowed between the original
                and corrected sentence (default: 0.3).
        """
        print(f"Loading model: {model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)                    # Create a T5 tokenizer using the given model name
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).cuda()  # Create a T5 model for conditional generation
        self.max_change_threshold = max_change_threshold                            # Set the threshold for how much a corrected sentence can differ

    def process(self, sentences):
        """
        Corrects each sentence in the given list using the T5 model.

        Args:
            sentences (list of str): A list of sentences to be corrected.

        Returns:
            list of str: A list of corrected sentences.
        """
        corrected_sentences = []

        # Go through each sentence to generate a correction
        for sentence in sentences:
            try:
                # Prepare input for the T5 model
                input_text = f"Correct the sentence: {sentence}"

                # Convert the prompt to tokens for T5
                inputs = self.tokenizer.encode(input_text, return_tensors="pt").cuda()

                # Let T5 generate a corrected output:
                # - max_length is a bit larger than the original sentence length.
                # - num_beams=5 uses beam search for better results.
                # - repetition_penalty=2.0 penalizes repeated words.
                # - early_stopping=True stops early if all beams finish.
                outputs = self.model.generate(
                    inputs,
                    max_length=len(sentence) + 20,
                    num_beams=5,
                    repetition_penalty=2.0,
                    early_stopping=True,
                )

                # Decode the first (best) result into a string (skipping special tokens, like <s>, </s>)
                corrected_sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

                # Clean the corrected text using our custom rules
                cleaned_sentence = self._clean_output(corrected_sentence)

                # Check how different the cleaned sentence is from the original
                change_ratio = self._calculate_change_ratio(sentence, cleaned_sentence)

                # Check if any words from the original are missing in the corrected version
                missing_words = self._find_missing_words(sentence, cleaned_sentence)

                # If it's too different, or the first word changed, or key words are missing, reject it
                if change_ratio > self.max_change_threshold \
                        or not cleaned_sentence.startswith(sentence.split()[0]) \
                        or missing_words:
                    print(f"Rejected correction due to high change ratio or leading word mismatch: {cleaned_sentence}")

                    # If rejected, just clean the original sentence instead
                    corrected_sentences.append(self._clean_output(sentence))
                else:
                    # If acceptable, store the corrected version
                    corrected_sentences.append(cleaned_sentence)

            # If there's an error, show it and just clean the original sentence instead
            except Exception as e:
                print(f"Error correcting sentence '{sentence}': {e}")
                corrected_sentences.append(self._clean_output(sentence))

        # Return the list of corrected (or rejected) sentences
        return corrected_sentences

    @staticmethod
    def _calculate_change_ratio(original, corrected):
        """
         Calculates how different two sentences are, as a fraction of changed characters.

         Args:
             original (str): The original sentence.
             corrected (str): The corrected sentence.

         Returns:
             float: A number between 0 and 1 that tells us how much was changed.
                    0 means no change, 1 means completely different.
         """
        matcher = SequenceMatcher(None, original, corrected)

        # 1 - ratio means the fraction of characters that differ.
        return 1 - matcher.ratio()

    def _clean_output(self, output):
        """
        Cleans the model's output by applying several regular-expression fixes.

        - Removes prompt text, extra spaces, extra punctuation.
        - Fixes certain words that stick together (e.g., "ISA" -> "IS A").
        - Makes sure the sentence ends with a period, '!', or '?'.
        - Converts everything to uppercase.

        Args:
            output (str): The raw T5 output sentence.

        Returns:
            str: The cleaned, uppercase sentence.
        """

        # A list of (pattern, replacement) for regex substitutions in order
        cleaning_rules = [
            ("Correct the sentence:", ""),              # Remove the prompt text
            ("\.\s+([A-Za-z])", r".\1"),                # Remove extra spaces after a period
            (r"\.\s*\.\s*([A-Za-z])", r"\1"),           # Remove repeated dots
            (r"\.\s*([A-Za-z])", r" \1"),               # Add one space after a period
            (r"\s*\.\s*", "."),                         # Remove spaces around periods
            (r"\s*\.\s*\.\s*", "."),                    # Combine multiple periods into one
            (r"\.\.+", "."),                            # Same as above but covers more cases
            (r"(\b[A-Z]+)\s([A-Z]\b)", r"\1\2"),        # Remove space between uppercase words
            (" H$", "."),                               # If line ends with " H", make it a period
            ("H$", "."),                                # If line ends with just "H", make it a period
            ("\s+", " "),                               # Turn many spaces into one
            ("\s\.", "."),                              # Remove space before a period
            ("\s,", ","),                               # Remove space before a comma
            ("\s;", ";"),                               # Remove space before a semicolon
            ("\s!", "!"),                               # Remove space before an exclamation mark
            ("\s\?", "?"),                              # Remove space before a question mark
            (",\.", "."),                               # If comma then period, just period
            (r"\b([A-Za-z]+)\s([A-Za-z]{1})\b", r"\1\2"),   # Join word + single letter
            (r"\b([A-Za-z]+)\s([A-Za-z]{1})\b", r"\1\2"),   # Join word + single letter
            (r"\b(IS|ARE|WAS|WERE|BE)\s+A\b", r"\1 A"),
        ]

        # Go through each cleaning rule and apply it to the output string
        for pattern, replacement in cleaning_rules:
            output = re.sub(pattern, replacement, output)

        # Fix some specific words that may have been glued together:
        # (ISA -> IS A, AREA -> ARE A, etc.)
        output = re.sub(r"\b(ISA)\b", "IS A", output)
        output = re.sub(r"\b(AREA)\b", "ARE A", output)
        output = re.sub(r"\b(WASA)\b", "WAS A", output)
        output = re.sub(r"\b(WEREA)\b", "WERE A", output)
        output = re.sub(r"\b(ATA)\b", "AT A", output)
        output = re.sub(r"\b(GAVEA)\b", "GAVE A", output)
        output = re.sub(r"\b(NOTA)\b", "NOT A", output)
        output = re.sub(r"\b(BYA)\b", "BY A", output)
        output = re.sub(r"\b(OFA)\b", "OF A", output)
        output = re.sub(r"\b(ONA)\b", "ON A", output)
        output = re.sub(r"\b(ALSOA)\b", "ALSO A", output)

        # If the sentence does not end with ., !, or ?, add a period
        if not output.endswith((".", "!", "?")):
            output += "."

        # Split by spaces to remove any repeated words in sequence
        words = output.split()
        cleaned_words = []
        for word in words:
            # Only add the word if it's not the same as the last added
            if not cleaned_words or word != cleaned_words[-1]:
                cleaned_words.append(word)

        cleaned_output = " ".join(cleaned_words)        # Join the cleaned words back together
        cleaned_output = cleaned_output.upper()         # Convert everything to uppercase

        # print(f"Final cleaned output: '{cleaned_output}'")    # Printing for debugging
        return cleaned_output

    @staticmethod
    def _find_missing_words(original, corrected):
        """
        Finds which words from the original sentence do not appear in the corrected one.

        We ignore small words like "THE", "A", or "AN".

        Args:
            original (str): The original, uncorrected sentence.
            corrected (str): The corrected sentence after cleaning.

        Returns:
            list of str: The words that are in the original but not in the corrected,
                         excluding "THE", "A", and "AN".
        """
        # Split sentences into sets of words
        original_words = set(original.split())
        corrected_words = set(corrected.split())

        # Find words that exist in original but not in corrected
        missing_words = original_words - corrected_words

        # We consider "THE", "A", and "AN" as trivial, so ignore them
        trivial_words = {"THE", "A", "AN"}
        meaningful_missing = [word for word in missing_words if word not in trivial_words]

        return meaningful_missing