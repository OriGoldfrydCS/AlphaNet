from difflib import SequenceMatcher
import requests
import re

class CorrectionStage:
    def __init__(self,max_change_threshold=0.10):
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

    def autocorrect_with_ollama(self, text, model="llama2"):
        """
        Use Ollama to autocorrect text and return only the corrected sentence.
        """
        url = "http://localhost:11434/api/generate"
        
        prompt =  prompt = f"""Fix any spelling mistakes in this text: {text}"""
        
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "cuda": True 
        }
        
        # Check if the text is a single word
        flag = len(text.split()) == 1
            
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            result = response.json()
            
            full_response = result['response']
            if flag:
                match = re.search(r'The correct spelling is "(.*?)"', full_response)
                cleaned_sentence = match.group(1) if match else full_response.strip()
            else:
                cleaned_sentence = full_response.strip()
            corrected_text = self.find_most_similar_sentence(text, cleaned_sentence)
            corrected_text = re.sub(r'^[\'"](.*)[\'"]$', r'\1', corrected_text.strip())
            
            corrected_text = self._clean_output(self.find_most_similar_sentence(text, cleaned_sentence), flag)
            
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
        resistence = self.max_change_threshold  
        corrected_sentences = []
        for sentence in sentences:
            #check how many words there are in the sentence
            words = sentence.split()
            
            #if it's only a single word, we can be less strict
            if len(words) == 1:
                resistence = 0.5

            corrected = self.autocorrect_with_ollama(sentence)

            #technically it shouldn't be upper but because we work on uppercase letters
            #but the model is trained to output things in the right case.
            cleaned_sentence = corrected.upper()
            change_ratio = self._calculate_change_ratio(sentence, cleaned_sentence)
        

            # If it's too different, or the first word changed while it, or key words are missing, reject it
            if change_ratio > resistence:                # Print what got us rejecte
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


    def _clean_output(self, output, flag):
        """
        Cleans the model's output with general corrections only.
        """
        # General corrections only ##########
        # Remove leading and trailing quotation marks
        output = re.sub(r'^THE CORRECT SPELLING IS "', '', output)
        output = re.sub(r'"', '', output)

        # Remove extra spaces
        output = re.sub(r'\s+', ' ', output).strip()
                
        # Ensure sentence ends with a period, exclamation, or question mark
        if flag == 0:
            if not output.endswith((".", "!", "?")):
                output += "."
        
        if flag == 1:
            # Only add a period if the output is more than one word or if it already ends with punctuation
            if " " in output or output.endswith((".", "!", "?")):
                pass  
            else:
                output += "."
                        
        # Convert the entire sentence to uppercase
        output = output.upper()

        return output

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
        # l = [text]
        # print(CorrectionStagel.process(l))
        print(CorrectionStagel.autocorrect_with_ollama(text))
if __name__ == "__main__":
    main()