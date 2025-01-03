from PIL import Image
import torch
from torchvision import transforms
import os

class ClassificationStage:
    """
    Stage for classifying letter images into numeric labels (0-25 for A-Z, 0-32 for ".", ",", "?", "!", ";", and space).
    """
    def __init__(self, model, transform, device):
        self.model = model
        self.transform = transform
        self.device = device

    def process(self, letter_images):
        predictions = []
        for img_path in sorted(letter_images, key=lambda x: int(os.path.splitext(os.path.basename(x))[0])):
            image = Image.open(img_path).convert("L")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(image_tensor)
                _, predicted = outputs.max(1)
                predictions.append(predicted.item())
        return predictions
