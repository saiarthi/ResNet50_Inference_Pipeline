import torch
from torchvision import models
import json
import urllib

# Load ResNet50 torchvision_model weights
torchvision_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
torch.save(torchvision_model.state_dict(), 'resnet50_torchvision_weights.pth')
torchvision_model.eval()
 
#run inference for torchvision_model
def torchvision_inference(image):
    with torch.no_grad():
        output = torchvision_model(image)
    return output

