class VecToTextStage:
    """
    Stage for converting numeric vector to a text word.
    """
    def __init__(self):
        # Map 0-25 to 'A'-'Z'
        self.class_to_char = {i: chr(i + ord('A')) for i in range(26)}
        # Map 26-31 to [",", ".", "!", "?", ";", " "]
        self.class_to_char.update({
            26: ',',
            27: '.',
            28: '!',
            29: '?',
            30: ';',
            31: ' '
        })

    def process(self, vector):
        """
        Convert a numeric vector to a text word.

        Args:
            vector (list[int]): List of numeric labels.

        Returns:
            str: Corresponding text word.
        """
        return ''.join(self.class_to_char[num] for num in vector if num in self.class_to_char)
