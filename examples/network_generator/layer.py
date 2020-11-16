import torchex.nn as exnn
import torch.nn as nn
from torch.nn import functional as F
import torch

from typing import Tuple, List
import itertools



class ConvRelu(nn.Module):
    def __init__(self, out_channels, kernel_size, stride):
        super(ConvRelu, self).__init__()
        self.conv = exnn.Conv2d(out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.conv(x))

    
def conv2d(out_channels, kernel_size, stride):
    return ConvRelu(out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride)


class ConcatConv(nn.Module):
    def __init__(self, out_channels, kernel_size, stride):
        super(ConcatConv, self).__init__()
        self.conv = conv2d(out_channels, kernel_size, stride)
        self.relu = nn.ReLU()        

    def forward(self, *x):
        x1 = torch.cat(x, dim=1)
        return self.relu(self.conv(x1))


class FlattenLinear(nn.Module):
    def __init__(self, out_channels):
        super(FlattenLinear, self).__init__()
        self.out_channels = out_channels
        self.linear = exnn.Linear(self.out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        if len(x.shape) == 4:
            B, _, _, _ = x.shape
            x = x.reshape(B, -1)
        return self.relu(self.linear(x))


class ConcatFlatten(nn.Module):
    def __init__(self, out_channels):
        super(ConcatFlatten, self).__init__()
        self.out_channels = out_channels
        self.linear = exnn.Linear(self.out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, *x):
        x1 = torch.cat(x, dim=1)
        if len(x1.shape) == 4:
            B, _, _, _ = x1.shape
            x1 = x1.reshape(B, -1)
        return self.relu(self.linear(x1))


#  https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
def conv2d_output_size(input_size, out_channels, kernel_size, stride, padding=0, dilation=1):
    _, h_in, w_in = input_size
    h_out = int((h_in + 2 * padding - dilation * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    w_out = int((w_in + 2 * padding - dilation * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    return out_channels, h_out, w_out


def conv_output_size(input_size, kernel_size, stride, padding=0, dilation=1):
    output_size = int((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    return output_size


def find_conv_layer(input_size, output_size, kernel_sizes: List[int], strides: List[int]) -> Tuple[int, int]:
    """
    見つからなかったときはraiseしますm(_ _)m
    """
    for k, s in itertools.product(kernel_sizes, strides):
        if conv_output_size(input_size, k, s) == output_size:
            return k, s
    raise ValueError("not found")
