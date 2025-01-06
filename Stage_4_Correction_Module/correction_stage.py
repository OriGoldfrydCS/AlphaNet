from difflib import SequenceMatcher
import requests
import re
class CorrectionStage:
    def __init__(self,max_change_threshold=0.2):
        self.max_change_threshold = max_change_threshold
        pass
    def string_similarity(self, a, b):
        """
        Calculate similarity ratio between two strings.
        In order to find the relevent output from the response text.
        In the String.
        """

        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def find_most_similar_sentence(self, original_text, response_text):
        """Find the sentence in response_text most similar to original_text."""
        # Split into sentences (handling multiple potential delimiters)
        sentences = re.split(r'[.!?\n]+', response_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return response_text
        
        # Find the sentence with highest similarity to input
        similarities = [(self.string_similarity(original_text, sent), sent) for sent in sentences]
        most_similar = max(similarities, key=lambda x: x[0])
        
        return most_similar[1]

    def autocorrect_with_ollama(self,text, model="llama2"):
        """
        Use Ollama to autocorrect text and return only the corrected sentence.
        
        Args:
            text (str): Text to correct
            model (str): Ollama model to use
        
        Returns:
            str: Clean corrected text
        """
        url = "http://localhost:11434/api/generate"
        
        prompt = f"""Fix any spelling mistakes in this text:
    {text}"""
        
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            result = response.json()
            
            # Get the raw response
            full_response = result['response']
            
            # Find the most similar sentence to our input
            corrected_text = self.find_most_similar_sentence(text, full_response)
            
            # Remove any remaining quotes
            corrected_text = re.sub(r'^[\'"](.*)[\'"]$', r'\1', corrected_text.strip())
            
            return corrected_text
            
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {e}")
            return text
        except Exception as e:
            print(f"Error during correction: {e}")
            return text

    def process(self, sentences):
        """
        Corrects each sentence in the given list using the T5 model.

        Args:
            sentences (list of str): A list of sentences to be corrected.

        Returns:
            list of str: A list of corrected sentences.
        """
        corrected_sentences = []
        for sentence in sentences:
            corrected = self.autocorrect_with_ollama(sentence)
            # corrected_sentences.append(corrected)

            cleaned_sentence = corrected

            change_ratio = self._calculate_change_ratio(sentence, cleaned_sentence)

            # Check if any words from the original are missing in the corrected version
            # missing_words = self._find_missing_words(sentence, cleaned_sentence)

            # If it's too different, or the first word changed, or key words are missing, reject it
            if change_ratio > self.max_change_threshold \
                    or not cleaned_sentence.startswith(sentence.split()[0]):
                # Print what got us rejected:
                if change_ratio > self.max_change_threshold:
                    print(f"Change ratio: got you in {change_ratio}")
                if not cleaned_sentence.startswith(sentence.split()[0]):
                    print(f"Original sentence: {sentence}")
                    print(f"Cleaned sentence: got you in{cleaned_sentence}")
                print(f"Rejected correction due to high change ratio or leading word mismatch: {cleaned_sentence}")

                # If rejected, just clean the original sentence instead
                corrected_sentences.append(sentence)
            else:
                # If acceptable, store the corrected version
                if cleaned_sentence[-1] not in ".!?":
                    cleaned_sentence += "."
                corrected_sentences.append(cleaned_sentence)
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

    #I don't really need it, see that I when I auto correct I don't use it
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
        ]

        # Go through each cleaning rule and apply it to the output string
        for pattern, replacement in cleaning_rules:
            output = re.sub(pattern, replacement, output)

        # Fix some specific words that may have been glued together:
        # (ISA -> IS A, AREA -> ARE A, etc.)
        output = re.sub(r"\b(AMA)\b", "AM A", output)
        output = re.sub(r"\b(ISA)\b", "IS A", output)
        output = re.sub(r"\b(AREA)\b", "ARE A", output)
        output = re.sub(r"\b(WASA)\b", "WAS A", output)
        output = re.sub(r"\b(WEREA)\b", "WERE A", output)
        output = re.sub(r"\b(HADA)\b", "HAD A", output)
        output = re.sub(r"\b(HAVEA)\b", "HAVE A", output)
        output = re.sub(r"\b(ATA)\b", "AT A", output)
        output = re.sub(r"\b(INA)\b", "IN A", output)
        output = re.sub(r"\b(GAVEA)\b", "GAVE A", output)
        output = re.sub(r"\b(NOTA)\b", "NOT A", output)
        output = re.sub(r"\b(BYA)\b", "BY A", output)
        output = re.sub(r"\b(OFA)\b", "OF A", output)
        output = re.sub(r"\b(ONA)\b", "ON A", output)
        output = re.sub(r"\b(ALSOA)\b", "ALSO A", output)
        output = re.sub(r"\b(IMA)\b", "I AM A", output)
        output = re.sub(r"\b(ASA)\b", "AS A", output)
        output = re.sub(r"\b(LEAVEA)\b", "LEAVE A", output)
        output = re.sub(r"\b(TAKEA)\b", "TAKE A", output)
        output = re.sub(r",\.", ".", output)

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
#self testing code
def main():
    CorrectionStagel = CorrectionStage()

    while True:
        text = input("Enter text to autocorrect (or 'quit' to exit): ")
        
        if text.lower() == 'quit':
            break
            
        print(CorrectionStagel.autocorrect_with_ollama(text))

if __name__ == "__main__":
    main()