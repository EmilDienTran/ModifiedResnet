import torch
import torch.nn as nn
import math

class SelfAttention2D(nn.Module):
    '''
    A simple self-attention module.
    This module care about scale, it attempts to attribute attention to the size of the image.
    This functions between each layer, but best where the width and the height - after convolution - isn't too small
    In resnet we 'compress' the image between the layers, and attention requires a big L.
    It is therefore recommended to place attention in layer1 -> layer2
    '''
    def __init__(self, input_dim):
        super(SelfAttention2D, self).__init__()
        self.query = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        self.key = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        self.value = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        #self.dropout = nn.Dropout(0.3)
        self.gamma = nn.Parameter(torch.rand(1))

    def forward(self, x):
        batch, channels, height, width = x.size()

        query = self.query(x).view(batch, -1, height*width).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, height * width)
        value = self.value(x).view(batch, -1, height * width)

        attention_output, _ = self.attention(query, key, value)
        attention_output = attention_output.view(batch, channels, height, width)
        #attention_output = self.dropout(attention_output)
        attention_output = self.gamma * attention_output + x

        return attention_output

    def attention(self, query, key, value):
        scores = torch.matmul(query, key) / math.sqrt(key.size(-1))
        p_attn = torch.softmax(scores, dim=-1).permute(0, 2, 1)
        return torch.matmul(value, p_attn), p_attn


class MultiHeadAttention(nn.Module):
    '''
    multi-head attention module.
    '''
    def __init__(self, input_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        assert input_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.query = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        self.key = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        self.value = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        #self.dropout = nn.Dropout(0.3)
        self.gamma = nn.Parameter(torch.rand(1))


    def forward(self, x):
        batch, channels, height, width = x.size()

        query = self.query(x).view(batch, self.num_heads, height*width, self.head_dim)
        key = self.key(x).view(batch, self.num_heads, height*width, self.head_dim)
        value = self.value(x).view(batch, self.num_heads, height*width, self.head_dim)

        attention_output, _ = self.attention(query, key, value)
        attention_output = attention_output.permute(0, 1, 3, 2).contiguous()
        attention_output = attention_output.view(batch, channels, height, width)
        #attention_output = self.dropout(attention_output)
        attention_output = self.gamma * attention_output + x


        return attention_output


    def attention(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(key.size(-1))
        p_attn = torch.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value), p_attn

class MultiModalFeatureFusion(nn.Module):
    '''
    Multi-modal feature fusion module. It takes in two seperate forms of an image.
    The goal is to train two seperate Resnet50s to learn different important features of an image.
    Then those models are combined - training two seperate models.
    '''
    def __init__(self, input_dim, input_dim2):
        super(MultiModalFeatureFusion, self).__init__()

        self.conv1 = nn.Conv2d(input_dim, 256, kernel_size=1)
        self.conv2 = nn.Conv2d(input_dim2, 256, kernel_size=1)
        self.fusion = nn.Conv2d(512, 256, kernel_size=1)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x = torch.cat((x1, x2), 1)
        x = self.fusion(x)
        return x

class LayerFusion(nn.Module):
    '''
    Layerfusion module. It takes in two seperate branches and combines them with a weighted tensor, allowing for training on branch effect.
    '''
    def __init__(self, input_dim):
        super(LayerFusion, self).__init__()
        self.gamma = nn.Parameter(torch.tensor([0.5]))
        self.bn1 = nn.BatchNorm2d(input_dim)
        self.bn2 = nn.BatchNorm2d(input_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)


    def forward(self, convolution, attention):
        convolution = self.bn1(convolution)
        attention = self.bn2(attention)
        weight_convolution = torch.sigmoid(self.gamma)
        weight_attention = 1 - weight_convolution
        x = weight_convolution * convolution + weight_attention * attention
        x = self.dropout(x)
        x = self.relu(x)
        return x





