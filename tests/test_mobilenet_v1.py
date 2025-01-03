import pytest
import torch
from models.mobilenet_v1 import MobileNetV1

@pytest.fixture
def model():
    return MobileNetV1(num_classes=1000)

def test_mobilenet_v1_forward_shape(model):
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    assert output.shape == (2, 1000), "Output shape mismatch"

def test_mobilenet_v1_forward_dtype(model):
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    assert output.dtype == torch.float32, "Output dtype mismatch"

def test_mobilenet_v1_no_nan_values(model):
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    assert not torch.isnan(output).any(), "Output contains NaN values"

def test_mobilenet_v1_eval_mode(model):
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (2, 1000), "Output shape mismatch in eval mode"

def test_mobilenet_v1_train_mode(model):
    model.train()
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    assert output.shape == (2, 1000), "Output shape mismatch in train mode"

def test_mobilenet_v1_train_mode_no_grad(model):
    model.train()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (2, 1000), "Output shape mismatch in train mode with no grad"

if __name__ == "__main__":
    pytest.main([__file__])