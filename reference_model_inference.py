import torch
from resnet50 import resnet50
import json
import urllib

# Initialize the model
reference_model = resnet50()

# Load the saved weights from the torchvision model - to ensure precision correctness
reference_model.load_state_dict(torch.load('resnet50_torchvision_weights.pth', weights_only=True))
reference_model.eval()

#run inference for reference_model
def reference_inference(image):
    with torch.no_grad():
        output = reference_model(image)
    return output

