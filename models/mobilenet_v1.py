import torch 
from torch import nn
from torch.nn import functional as F
from models.common_blocks import ConvBlock, DepthwiseSeparableConv


  
class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000, middle_stage_layers_size=5):
        super(MobileNetV1, self).__init__()

        middle_stage_layers = [DepthwiseSeparableConv(512, 512, stride=1) for _ in range(middle_stage_layers_size)]


        self.backbone = nn.Sequential(
            ConvBlock(3, 32, stride=2),
            DepthwiseSeparableConv(32, 64, stride=1),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=1),
            DepthwiseSeparableConv(256, 512, stride=2),
            *middle_stage_layers, 
            DepthwiseSeparableConv(512, 1024, stride=2),
            DepthwiseSeparableConv(1024, 1024, stride=1)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    


if __name__ == "__main__":
    model = MobileNetV1(1000)
    x = torch.randn(2, 3, 224, 224)
    print(model(x).shape)




