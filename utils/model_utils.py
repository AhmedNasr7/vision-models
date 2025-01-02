import torch
from torch import nn
import torch.nn.init as init

def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)  # Xavier Uniform for Linear Layers
        if m.bias is not None:
            init.zeros_(m.bias)         # Initialize Bias to 0
    elif isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # He Initialization
        if m.bias is not None:
            init.zeros_(m.bias)

def param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy(outputs, labels):
  _, preds = torch.max(outputs, dim=1)
  return torch.tensor(torch.sum(preds == labels).item() / len(preds))