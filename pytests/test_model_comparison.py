import pytest
import torch
from src.reference_model_inference import reference_inference
from src.load_image import load_image_from_file
from src.comp_pcc import comp_pcc
from src.torchvision_model_inference import torchvision_inference

# Load and preprocess the image
image_path = 'src/photo-1560807707-8cc77767d783.jpeg'
image = load_image_from_file(image_path)

@pytest.fixture
def model_outputs():
    if image is not None:
        with torch.no_grad():
            torchvision_output = torchvision_inference(image)
            reference_output = reference_inference(image)

        return torchvision_output, reference_output
    else:
        pytest.fail("Failed to process the image")

def test_model_comparison(model_outputs):
    torchvision_output, reference_output = model_outputs
    result, pcc_value = comp_pcc(torchvision_output, reference_output)
    print(f'Comparison result: {result}, {pcc_value}')
    assert result, f"Models do not match.{pcc_value}"

if __name__ == "__main__":
    pytest.main(["-rP"])
