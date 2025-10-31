import torch
import torch.nn as nn
import math
from CustomModules import SelfAttention2D
from CustomModules import MultiHeadAttention


class MultiHeadAttentionBlockNorm(nn.Module):
    '''
    A normalized multi-head attention block.
    '''
    def __init__(self, input_dim):
        super(MultiHeadAttentionBlockNorm, self).__init__()
        self.attention = MultiHeadAttention(input_dim)
        self.bn = nn.BatchNorm2d(input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.attention(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SelfAttentionBlockNorm(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionBlockNorm, self).__init__()
        self.attention = SelfAttention2D(input_dim)
        self.bn = nn.BatchNorm2d(input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.attention(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1, downsample=None, attention_heads=2):
        super(MultiHeadAttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(output_dim)
        self.relu1 = nn.ReLU()

        self.cpe = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1, groups=output_dim)
        self.attention = MultiHeadAttention(output_dim, attention_heads)

        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size = 1)
        self.bn2 = nn.BatchNorm2d(output_dim)
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.output_dim = output_dim

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = x + self.cpe(x)
        x = self.attention(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)
        return x

