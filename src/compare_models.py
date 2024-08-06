import torch
from torchvision_model_inference import torchvision_inference
from reference_model_inference import reference_inference
from load_image import load_image_from_file
from comp_pcc import comp_pcc
import requests

#Function to get the class names from ImageNet
def load_imagenet_class_names():
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    response = requests.get(url)
    class_names = response.json()
    return class_names

# Load the ImageNet class names
imagenet_class_names = load_imagenet_class_names()

# path of a specific class image - from Imagenet dataset
image_path = 'photo-1560807707-8cc77767d783.jpeg'

# Load and preprocess the image for both models
image = load_image_from_file(image_path)

if image is not None:
    # Perform inference using both models
    torchvision_model_output = torchvision_inference(image)
    reference_model_output = reference_inference(image)

    # Compare the outputs of the two models
    result, pcc_value = comp_pcc(torchvision_model_output, reference_model_output)

    # Get the predicted class - Model Validation
    _, torchvision_pred_class = torch.max(torchvision_model_output, 1)
    _, reference_pred_class = torch.max(reference_model_output, 1)

    print(f'Torchvision model predicted class index: {torchvision_pred_class.item()}')
    print(f'Torchvision model predicted class name: {imagenet_class_names[torchvision_pred_class.item()]}')
    print(f'Reference model predicted class index: {reference_pred_class.item()}')
    print(f'Reference model predicted class name: {imagenet_class_names[reference_pred_class.item()]}')
    print(f'Comparison result: {result}, PCC: {pcc_value}')

    comparison_result = torch.allclose(torchvision_model_output, reference_model_output, atol=1e-5)
else:
    print("Failed to process the image.")
