import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from CustomModules import SelfAttention2D, MultiHeadAttention


class ModifiedResNetAttention(nn.Module):
    def __init__(self, num_classes=100):
        super(ModifiedResNetAttention, self).__init__()
        base = resnet18(weights=None)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()
        self.layer1 = base.layer1
        self.attention = MultiHeadAttention(64, 8)
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.avgpool = base.avgpool
        self.fc = nn.Sequential(
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.attention(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.layer4(x)


        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x