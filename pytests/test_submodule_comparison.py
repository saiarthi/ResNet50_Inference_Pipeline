import pytest
import torch
from torchvision import models
from torchvision.models import resnet50 as torchvision_resnet50
from src.resnet50 import ResNet50, Bottleneck
from src.comp_pcc import comp_pcc

@pytest.fixture
def load_resnet50_bottleneck():
    torch_model = torchvision_resnet50(weights=models.ResNet50_Weights.DEFAULT)
    ref_model = ResNet50()
    ref_model.load_state_dict(torch.load('src/resnet50_torchvision_weights.pth', weights_only=True))
    torch_layer = torch_model.layer1[0]
    ref_layer = ref_model.layer1[0]
    return torch_layer, ref_layer

def test_resnet50_bottleneck(load_resnet50_bottleneck):
    torch_layer, ref_layer = load_resnet50_bottleneck
    torch_layer.eval()
    ref_layer.eval()

    input_tensor = torch.randn(1, 64, 56, 56)

    with torch.no_grad():
        torch_output = torch_layer(input_tensor)
        ref_output = ref_layer(input_tensor)

    result, pcc_value = comp_pcc(torch_output, ref_output)
    assert result, f"ResNet50 Bottleneck submodules do not match. {pcc_value}"
    print(f"ResNet50 Bottleneck submodule {pcc_value}")

if __name__ == "__main__":
    pytest.main()
