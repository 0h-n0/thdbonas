#!/usr/bin/env python

import copy
from collections import OrderedDict
from enum import Enum, auto

import torch
import torch.nn as nn
import torchex.nn as exnn

from inferno.extensions.layers.reshape import Concatenate


class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()
        self.cat = Concatenate()

    def reshape_1d(self, x):
        B = x.size(0)
        return x.view(B, -1)
        
    def forward(self, x1, x2):
        x1 = self.reshape_1d(x1)
        x2 = self.reshape_1d(x2)        
        return self.cat(x1, x2)
        

class LayerType(Enum):
    CONCAT = auto()
    LINEAR = auto()
    CONV2D = auto()
    FLATTEN = auto()

    @staticmethod
    def _get_linear(out_channels):
        return exnn.Linear(out_channels)
    
    @staticmethod
    def _get_concat():
        return Concat()

    @staticmethod
    def _get_conv2d(out_channels, kernel_size, stride):
        conv2d = nn.Sequential(
            exnn.Conv2d(out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride),
            nn.ReLU())
        
        return conv2d
    
    @staticmethod
    def _get_flatten():
        return exnn.Flatten()


class LayerGenerator:
    ''' create module and vector.
    '''
    def __init__(self):
        self.registered_layers = []
        self.layer_identifed_dict = OrderedDict()
        self.layer_dict[LayerType.CONCAT] = 0
        self._len = None
        

    def register(self, layer_type: LayerType, **params):
        self.registered_layers.append((layer_type, params))
        self.layer_identifed_dict[layer_type] = 0
        for k in params:
            self.layer_identifed_dict[k] = 0

    def __getitem__(self, idx):
        module, vec = self.construct(idx)
        return module, vec

    def construct(self, idx):
        layer_identifed_dict = copy.deepcopy(self.layer_identifed_dict)
        (layer_type, params) = self.registered_layers[idx]
        layer_identifed_dict[layer_type] = 1
        for k, v in params.items():
            layer_identifed_dict[k] = v
        layer_identifed_vec = list(layer_identifed_dict.values())
        
        if LayerType.LINEAR == layer_type:
            return LayerType._get_linear(**params), layer_identifed_vec

    

class NNGenerator:
    def __init__(self):
        pass


if __name__ == '__main__':
    l = LayerGenerator()
    l.register(LayerType.LINEAR, out_channels=100)
    l[0]
    x = torch.randn(3, 3, 3)
    y = torch.randn(3, 3, 3, 10)
    c = Concat()
    y = c(x, y)
    print(y.shape)
    
