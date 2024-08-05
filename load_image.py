import requests
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms
import numpy as np

# Define image preprocessing for the models
preprocess_img = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_image_from_file(file_path):
    try:
        image = Image.open(file_path).convert('RGB')
        image = preprocess_img(image)
        image = image.unsqueeze(0)  # Add batch dimension
        return image
    except FileNotFoundError as e:
        print(f"Error loading the image: {e}")
        return None
    except PIL.UnidentifiedImageError as e:
        print(f"Error identifying the image file: {e}")
        return None