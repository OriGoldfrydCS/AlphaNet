import os

class SegmentationStage:
    """
    Stage for segmenting word images into letter images.
    """
    def __init__(self, segmenter, output_folder):
        self.segmenter = segmenter
        self.output_folder = output_folder

    def process(self, image_path):
        word_name = os.path.splitext(os.path.basename(image_path))[0]
        output_subfolder = os.path.join(self.output_folder, word_name)
        os.makedirs(output_subfolder, exist_ok=True)
        return self.segmenter.process_image(image_path, output_subfolder)
