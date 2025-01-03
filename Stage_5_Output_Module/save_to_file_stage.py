import os

class SaveToFileStage:
    """
    Stage for saving reconstructed word to a .txt file.
    """
    def __init__(self, output_folder):
        self.output_folder = output_folder

    def process(self, data):
        word, file_name = data
        output_file = os.path.join(self.output_folder, f"{file_name}.txt")
        with open(output_file, "w") as f:
            f.write(word)
        print(f"Saved word to {output_file}")
