import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import torchvision


def conv3x3(input, output):
    return nn.Conv2d(input, output, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.conv = conv3x3(input, output)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            # Transposed Convolution for Up-sampling
            nn.ConvTranspose2d(middle_channels,
                               out_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class UNet11(nn.Module):
    def __init__(self, num_filters=32):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = models.vgg11().features
        
