import torch
from src.resnet50 import ResNet50
import json
import urllib

# Initialize the model
reference_model = ResNet50()

# Load the saved weights from the torchvision model - to ensure precision correctness
reference_model.load_state_dict(torch.load('src/resnet50_torchvision_weights.pth', weights_only=True))
reference_model.eval()

#run inference for reference_model
def reference_inference(image):
    with torch.no_grad():
        output = reference_model(image)
    return output

