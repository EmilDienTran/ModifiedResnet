import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from CustomModules import SelfAttention2D, MultiHeadAttention
from CustomBlocks import MultiHeadAttentionBlock

class ModifiedResNetAttention(nn.Module):
    def __init__(self, num_classes=100):
        super(ModifiedResNetAttention, self).__init__()
        self.name = 'ModifiedResNetAttention'
        self.inplanes = 64
        base = resnet18(weights=None)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()
        self.layer1 = base.layer1
        self.layer2 = self._make_layer(MultiHeadAttentionBlock, planes=128, blocks=2, attention_heads=8, stride=2)
        self.layer3 = self._make_layer(MultiHeadAttentionBlock, planes=256, blocks=2, attention_heads=8, stride=2)
        self.layer4 = base.layer4

        self.avgpool = base.avgpool
        self.fc = nn.Sequential(
            nn.Linear(512, num_classes),
        )

    def _make_layer(self, block, planes, blocks, attention_heads, stride=1):
        '''
        :param block: The name of the block that the model uses to generate the layer
        :param planes: The amount of neurons per layer - input
        :param blocks: The amount of blocks(How many times is it repeated?)
        :param attention_heads: Multi-head attention heads specific param
        :param stride: Stride of the blocks - enables or disables downsampling

        Note: Standard ResNet18 has [2, 2, 2, 2] blocks per layer.
          Increasing blocks adds depth but increases computation time.
        '''
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, attention_heads=attention_heads))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, attention_heads=attention_heads))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x