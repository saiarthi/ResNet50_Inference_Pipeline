import pytest
import torch
from torchvision.models import resnet50 as torchvision_resnet50, ResNet50_Weights
from src.resnet50 import ResNet50
from src.comp_pcc import comp_pcc

@pytest.fixture(params=[
    (0, 'layer1', 64, 56, 56),
    (1, 'layer1', 256, 56, 56),
    (2, 'layer1', 256, 56, 56),
    (0, 'layer2', 256, 28, 28),
    (1, 'layer2', 512, 28, 28),
    (2, 'layer2', 512, 28, 28),
    (3, 'layer2', 512, 28, 28),
    (0, 'layer3', 512, 14, 14),
    (1, 'layer3', 1024, 14, 14),
    (2, 'layer3', 1024, 14, 14),
    (3, 'layer3', 1024, 14, 14),
    (4, 'layer3', 1024, 14, 14),
    (5, 'layer3', 1024, 14, 14),
    (0, 'layer4', 1024, 7, 7),
    (1, 'layer4', 2048, 7, 7),
    (2, 'layer4', 2048, 7, 7),
])
def load_resnet50_bottlenecks(request):
    idx, layer, in_channels, height, width = request.param
    torch_model = torchvision_resnet50(weights=ResNet50_Weights.DEFAULT)
    ref_model = ResNet50()
    ref_model.load_state_dict(torch.load('src/resnet50_torchvision_weights.pth', weights_only=True))
    torch_layer = getattr(torch_model, layer)[idx]
    ref_layer = getattr(ref_model, layer)[idx]
    return torch_layer, ref_layer, in_channels, height, width

def test_resnet50_bottlenecks(load_resnet50_bottlenecks):
    torch_layer, ref_layer, in_channels, height, width = load_resnet50_bottlenecks

    torch_layer.eval()
    ref_layer.eval()

    input_tensor = torch.randn(1, in_channels, height, width)

    with torch.no_grad():
        torch_output = torch_layer(input_tensor)
        ref_output = ref_layer(input_tensor)

    result, pcc_value = comp_pcc(torch_output, ref_output)
    assert result, f"ResNet50 Bottleneck submodule does not match. {pcc_value}"
    print(f"ResNet50 Bottleneck submodule {pcc_value}")

if __name__ == "__main__":
    pytest.main()
