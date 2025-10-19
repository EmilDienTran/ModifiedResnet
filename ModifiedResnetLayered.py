import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from CustomModules import LayerFusion
from CustomBlocks import MultiHeadAttentionBlock

class ModifiedResNetLayered(nn.Module):
    def __init__(self, num_classes=100):
        super(ModifiedResNetLayered, self).__init__()
        base = resnet18(weights=None)
        base2 = resnet18(weights=None)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()
        self.layer1 = base.layer1

        #Branch1 - convolutional branch
        self.Convolution_layer2 = base.layer2
        self.Convolution_layer3 = base.layer3

        #Branch2 - Attention Heavy Branch
        self.attention_layer2 = self._make_layer(MultiHeadAttentionBlock, )
        self.attention_layer3 = base2.layer3

        self.fusion = LayerFusion(256)

        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        '''RESNET MODULE 1'''
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        '''Resnet Branch 1'''
        x_branch1 = self.ConvolutionBranch(x)

        '''Resnet Branch 2'''
        x_branch2 = self.AttentionBranch(x)


        x = self.fusion(x_branch1, x_branch2)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


    def ConvolutionBranch(self, x):
        out = self.Convolution_layer2(x)
        out = self.Convolution_layer3(out)
        return out

    def AttentionBranch(self, x):
        out = self.multihead(x)
        out = self.attention_layer2(out)
        out = self.attention(out)
        out = self.attention_layer3(out)
        return out



