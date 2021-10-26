"""
This implementation is modified from the following repository:
https://github.com/poppinace/indexnet_matting

"""

import os
import sys
import math
from time import time
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from hlaspp import ASPP
from lib.nn import SynchronizedBatchNorm2d
from hlindex import HolisticIndexBlock, DepthwiseO2OIndexBlock, DepthwiseM2OIndexBlock
from hldecoder import *
from hlconv import *
from modelsummary import get_model_summary

import units
import units.ConvGRU2 as ConvGRU

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve
try:
    from models.archs.dcn.deform_conv import ModulatedDeformConvPack as DCN
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')

model_urls = {
    'mobilenetv2': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/mobilenet_v2.pth.tar',
}


CORRESP_NAME = {
    # layer0
    "features.0.0.weight": "layer0.0.weight",
    "features.0.1.weight": "layer0.1.weight",
    "features.0.1.bias": "layer0.1.bias",
    "features.0.1.running_mean": "layer0.1.running_mean",
    "features.0.1.running_var": "layer0.1.running_var",
    # layer1
    "features.1.conv.0.weight": "layer1.0.conv.0.weight",
    "features.1.conv.1.weight": "layer1.0.conv.1.weight",
    "features.1.conv.1.bias": "layer1.0.conv.1.bias",
    "features.1.conv.1.running_mean": "layer1.0.conv.1.running_mean",
    "features.1.conv.1.running_var": "layer1.0.conv.1.running_var",
    "features.1.conv.3.weight": "layer1.0.conv.3.weight",
    "features.1.conv.4.weight": "layer1.0.conv.4.weight",
    "features.1.conv.4.bias": "layer1.0.conv.4.bias",
    "features.1.conv.4.running_mean": "layer1.0.conv.4.running_mean",
    "features.1.conv.4.running_var": "layer1.0.conv.4.running_var",
    # layer2
    "features.2.conv.0.weight": "layer2.0.conv.0.weight",
    "features.2.conv.1.weight": "layer2.0.conv.1.weight",
    "features.2.conv.1.bias": "layer2.0.conv.1.bias",
    "features.2.conv.1.running_mean": "layer2.0.conv.1.running_mean",
    "features.2.conv.1.running_var": "layer2.0.conv.1.running_var",
    "features.2.conv.3.weight": "layer2.0.conv.3.weight",
    "features.2.conv.4.weight": "layer2.0.conv.4.weight",
    "features.2.conv.4.bias": "layer2.0.conv.4.bias",
    "features.2.conv.4.running_mean": "layer2.0.conv.4.running_mean",
    "features.2.conv.4.running_var": "layer2.0.conv.4.running_var",
    "features.2.conv.6.weight": "layer2.0.conv.6.weight",
    "features.2.conv.7.weight": "layer2.0.conv.7.weight",
    "features.2.conv.7.bias": "layer2.0.conv.7.bias",
    "features.2.conv.7.running_mean": "layer2.0.conv.7.running_mean",
    "features.2.conv.7.running_var": "layer2.0.conv.7.running_var",

    "features.3.conv.0.weight": "layer2.1.conv.0.weight",
    "features.3.conv.1.weight": "layer2.1.conv.1.weight",
    "features.3.conv.1.bias": "layer2.1.conv.1.bias",
    "features.3.conv.1.running_mean": "layer2.1.conv.1.running_mean",
    "features.3.conv.1.running_var": "layer2.1.conv.1.running_var",
    "features.3.conv.3.weight": "layer2.1.conv.3.weight",
    "features.3.conv.4.weight": "layer2.1.conv.4.weight",
    "features.3.conv.4.bias": "layer2.1.conv.4.bias",
    "features.3.conv.4.running_mean": "layer2.1.conv.4.running_mean",
    "features.3.conv.4.running_var": "layer2.1.conv.4.running_var",
    "features.3.conv.6.weight": "layer2.1.conv.6.weight",
    "features.3.conv.7.weight": "layer2.1.conv.7.weight",
    "features.3.conv.7.bias": "layer2.1.conv.7.bias",
    "features.3.conv.7.running_mean": "layer2.1.conv.7.running_mean",
    "features.3.conv.7.running_var": "layer2.1.conv.7.running_var",
    # layer3
    "features.4.conv.0.weight": "layer3.0.conv.0.weight",
    "features.4.conv.1.weight": "layer3.0.conv.1.weight",
    "features.4.conv.1.bias": "layer3.0.conv.1.bias",
    "features.4.conv.1.running_mean": "layer3.0.conv.1.running_mean",
    "features.4.conv.1.running_var": "layer3.0.conv.1.running_var",
    "features.4.conv.3.weight": "layer3.0.conv.3.weight",
    "features.4.conv.4.weight": "layer3.0.conv.4.weight",
    "features.4.conv.4.bias": "layer3.0.conv.4.bias",
    "features.4.conv.4.running_mean": "layer3.0.conv.4.running_mean",
    "features.4.conv.4.running_var": "layer3.0.conv.4.running_var",
    "features.4.conv.6.weight": "layer3.0.conv.6.weight",
    "features.4.conv.7.weight": "layer3.0.conv.7.weight",
    "features.4.conv.7.bias": "layer3.0.conv.7.bias",
    "features.4.conv.7.running_mean": "layer3.0.conv.7.running_mean",
    "features.4.conv.7.running_var": "layer3.0.conv.7.running_var",

    "features.5.conv.0.weight": "layer3.1.conv.0.weight",
    "features.5.conv.1.weight": "layer3.1.conv.1.weight",
    "features.5.conv.1.bias": "layer3.1.conv.1.bias",
    "features.5.conv.1.running_mean": "layer3.1.conv.1.running_mean",
    "features.5.conv.1.running_var": "layer3.1.conv.1.running_var",
    "features.5.conv.3.weight": "layer3.1.conv.3.weight",
    "features.5.conv.4.weight": "layer3.1.conv.4.weight",
    "features.5.conv.4.bias": "layer3.1.conv.4.bias",
    "features.5.conv.4.running_mean": "layer3.1.conv.4.running_mean",
    "features.5.conv.4.running_var": "layer3.1.conv.4.running_var",
    "features.5.conv.6.weight": "layer3.1.conv.6.weight",
    "features.5.conv.7.weight": "layer3.1.conv.7.weight",
    "features.5.conv.7.bias": "layer3.1.conv.7.bias",
    "features.5.conv.7.running_mean": "layer3.1.conv.7.running_mean",
    "features.5.conv.7.running_var": "layer3.1.conv.7.running_var",

    "features.6.conv.0.weight": "layer3.2.conv.0.weight",
    "features.6.conv.1.weight": "layer3.2.conv.1.weight",
    "features.6.conv.1.bias": "layer3.2.conv.1.bias",
    "features.6.conv.1.running_mean": "layer3.2.conv.1.running_mean",
    "features.6.conv.1.running_var": "layer3.2.conv.1.running_var",
    "features.6.conv.3.weight": "layer3.2.conv.3.weight",
    "features.6.conv.4.weight": "layer3.2.conv.4.weight",
    "features.6.conv.4.bias": "layer3.2.conv.4.bias",
    "features.6.conv.4.running_mean": "layer3.2.conv.4.running_mean",
    "features.6.conv.4.running_var": "layer3.2.conv.4.running_var",
    "features.6.conv.6.weight": "layer3.2.conv.6.weight",
    "features.6.conv.7.weight": "layer3.2.conv.7.weight",
    "features.6.conv.7.bias": "layer3.2.conv.7.bias",
    "features.6.conv.7.running_mean": "layer3.2.conv.7.running_mean",
    "features.6.conv.7.running_var": "layer3.2.conv.7.running_var",
    # layer4
    "features.7.conv.0.weight": "layer4.0.conv.0.weight",
    "features.7.conv.1.weight": "layer4.0.conv.1.weight",
    "features.7.conv.1.bias": "layer4.0.conv.1.bias",
    "features.7.conv.1.running_mean": "layer4.0.conv.1.running_mean",
    "features.7.conv.1.running_var": "layer4.0.conv.1.running_var",
    "features.7.conv.3.weight": "layer4.0.conv.3.weight",
    "features.7.conv.4.weight": "layer4.0.conv.4.weight",
    "features.7.conv.4.bias": "layer4.0.conv.4.bias",
    "features.7.conv.4.running_mean": "layer4.0.conv.4.running_mean",
    "features.7.conv.4.running_var": "layer4.0.conv.4.running_var",
    "features.7.conv.6.weight": "layer4.0.conv.6.weight",
    "features.7.conv.7.weight": "layer4.0.conv.7.weight",
    "features.7.conv.7.bias": "layer4.0.conv.7.bias",
    "features.7.conv.7.running_mean": "layer4.0.conv.7.running_mean",
    "features.7.conv.7.running_var": "layer4.0.conv.7.running_var",

    "features.8.conv.0.weight": "layer4.1.conv.0.weight",
    "features.8.conv.1.weight": "layer4.1.conv.1.weight",
    "features.8.conv.1.bias": "layer4.1.conv.1.bias",
    "features.8.conv.1.running_mean": "layer4.1.conv.1.running_mean",
    "features.8.conv.1.running_var": "layer4.1.conv.1.running_var",
    "features.8.conv.3.weight": "layer4.1.conv.3.weight",
    "features.8.conv.4.weight": "layer4.1.conv.4.weight",
    "features.8.conv.4.bias": "layer4.1.conv.4.bias",
    "features.8.conv.4.running_mean": "layer4.1.conv.4.running_mean",
    "features.8.conv.4.running_var": "layer4.1.conv.4.running_var",
    "features.8.conv.6.weight": "layer4.1.conv.6.weight",
    "features.8.conv.7.weight": "layer4.1.conv.7.weight",
    "features.8.conv.7.bias": "layer4.1.conv.7.bias",
    "features.8.conv.7.running_mean": "layer4.1.conv.7.running_mean",
    "features.8.conv.7.running_var": "layer4.1.conv.7.running_var",

    "features.9.conv.0.weight": "layer4.2.conv.0.weight",
    "features.9.conv.1.weight": "layer4.2.conv.1.weight",
    "features.9.conv.1.bias": "layer4.2.conv.1.bias",
    "features.9.conv.1.running_mean": "layer4.2.conv.1.running_mean",
    "features.9.conv.1.running_var": "layer4.2.conv.1.running_var",
    "features.9.conv.3.weight": "layer4.2.conv.3.weight",
    "features.9.conv.4.weight": "layer4.2.conv.4.weight",
    "features.9.conv.4.bias": "layer4.2.conv.4.bias",
    "features.9.conv.4.running_mean": "layer4.2.conv.4.running_mean",
    "features.9.conv.4.running_var": "layer4.2.conv.4.running_var",
    "features.9.conv.6.weight": "layer4.2.conv.6.weight",
    "features.9.conv.7.weight": "layer4.2.conv.7.weight",
    "features.9.conv.7.bias": "layer4.2.conv.7.bias",
    "features.9.conv.7.running_mean": "layer4.2.conv.7.running_mean",
    "features.9.conv.7.running_var": "layer4.2.conv.7.running_var",

    "features.10.conv.0.weight": "layer4.3.conv.0.weight",
    "features.10.conv.1.weight": "layer4.3.conv.1.weight",
    "features.10.conv.1.bias": "layer4.3.conv.1.bias",
    "features.10.conv.1.running_mean": "layer4.3.conv.1.running_mean",
    "features.10.conv.1.running_var": "layer4.3.conv.1.running_var",
    "features.10.conv.3.weight": "layer4.3.conv.3.weight",
    "features.10.conv.4.weight": "layer4.3.conv.4.weight",
    "features.10.conv.4.bias": "layer4.3.conv.4.bias",
    "features.10.conv.4.running_mean": "layer4.3.conv.4.running_mean",
    "features.10.conv.4.running_var": "layer4.3.conv.4.running_var",
    "features.10.conv.6.weight": "layer4.3.conv.6.weight",
    "features.10.conv.7.weight": "layer4.3.conv.7.weight",
    "features.10.conv.7.bias": "layer4.3.conv.7.bias",
    "features.10.conv.7.running_mean": "layer4.3.conv.7.running_mean",
    "features.10.conv.7.running_var": "layer4.3.conv.7.running_var",
    # layer5
    "features.11.conv.0.weight": "layer5.0.conv.0.weight",
    "features.11.conv.1.weight": "layer5.0.conv.1.weight",
    "features.11.conv.1.bias": "layer5.0.conv.1.bias",
    "features.11.conv.1.running_mean": "layer5.0.conv.1.running_mean",
    "features.11.conv.1.running_var": "layer5.0.conv.1.running_var",
    "features.11.conv.3.weight": "layer5.0.conv.3.weight",
    "features.11.conv.4.weight": "layer5.0.conv.4.weight",
    "features.11.conv.4.bias": "layer5.0.conv.4.bias",
    "features.11.conv.4.running_mean": "layer5.0.conv.4.running_mean",
    "features.11.conv.4.running_var": "layer5.0.conv.4.running_var",
    "features.11.conv.6.weight": "layer5.0.conv.6.weight",
    "features.11.conv.7.weight": "layer5.0.conv.7.weight",
    "features.11.conv.7.bias": "layer5.0.conv.7.bias",
    "features.11.conv.7.running_mean": "layer5.0.conv.7.running_mean",
    "features.11.conv.7.running_var": "layer5.0.conv.7.running_var",

    "features.12.conv.0.weight": "layer5.1.conv.0.weight",
    "features.12.conv.1.weight": "layer5.1.conv.1.weight",
    "features.12.conv.1.bias": "layer5.1.conv.1.bias",
    "features.12.conv.1.running_mean": "layer5.1.conv.1.running_mean",
    "features.12.conv.1.running_var": "layer5.1.conv.1.running_var",
    "features.12.conv.3.weight": "layer5.1.conv.3.weight",
    "features.12.conv.4.weight": "layer5.1.conv.4.weight",
    "features.12.conv.4.bias": "layer5.1.conv.4.bias",
    "features.12.conv.4.running_mean": "layer5.1.conv.4.running_mean",
    "features.12.conv.4.running_var": "layer5.1.conv.4.running_var",
    "features.12.conv.6.weight": "layer5.1.conv.6.weight",
    "features.12.conv.7.weight": "layer5.1.conv.7.weight",
    "features.12.conv.7.bias": "layer5.1.conv.7.bias",
    "features.12.conv.7.running_mean": "layer5.1.conv.7.running_mean",
    "features.12.conv.7.running_var": "layer5.1.conv.7.running_var",

    "features.13.conv.0.weight": "layer5.2.conv.0.weight",
    "features.13.conv.1.weight": "layer5.2.conv.1.weight",
    "features.13.conv.1.bias": "layer5.2.conv.1.bias",
    "features.13.conv.1.running_mean": "layer5.2.conv.1.running_mean",
    "features.13.conv.1.running_var": "layer5.2.conv.1.running_var",
    "features.13.conv.3.weight": "layer5.2.conv.3.weight",
    "features.13.conv.4.weight": "layer5.2.conv.4.weight",
    "features.13.conv.4.bias": "layer5.2.conv.4.bias",
    "features.13.conv.4.running_mean": "layer5.2.conv.4.running_mean",
    "features.13.conv.4.running_var": "layer5.2.conv.4.running_var",
    "features.13.conv.6.weight": "layer5.2.conv.6.weight",
    "features.13.conv.7.weight": "layer5.2.conv.7.weight",
    "features.13.conv.7.bias": "layer5.2.conv.7.bias",
    "features.13.conv.7.running_mean": "layer5.2.conv.7.running_mean",
    "features.13.conv.7.running_var": "layer5.2.conv.7.running_var",
    # layer6
    "features.14.conv.0.weight": "layer6.0.conv.0.weight",
    "features.14.conv.1.weight": "layer6.0.conv.1.weight",
    "features.14.conv.1.bias": "layer6.0.conv.1.bias",
    "features.14.conv.1.running_mean": "layer6.0.conv.1.running_mean",
    "features.14.conv.1.running_var": "layer6.0.conv.1.running_var",
    "features.14.conv.3.weight": "layer6.0.conv.3.weight",
    "features.14.conv.4.weight": "layer6.0.conv.4.weight",
    "features.14.conv.4.bias": "layer6.0.conv.4.bias",
    "features.14.conv.4.running_mean": "layer6.0.conv.4.running_mean",
    "features.14.conv.4.running_var": "layer6.0.conv.4.running_var",
    "features.14.conv.6.weight": "layer6.0.conv.6.weight",
    "features.14.conv.7.weight": "layer6.0.conv.7.weight",
    "features.14.conv.7.bias": "layer6.0.conv.7.bias",
    "features.14.conv.7.running_mean": "layer6.0.conv.7.running_mean",
    "features.14.conv.7.running_var": "layer6.0.conv.7.running_var",

    "features.15.conv.0.weight": "layer6.1.conv.0.weight",
    "features.15.conv.1.weight": "layer6.1.conv.1.weight",
    "features.15.conv.1.bias": "layer6.1.conv.1.bias",
    "features.15.conv.1.running_mean": "layer6.1.conv.1.running_mean",
    "features.15.conv.1.running_var": "layer6.1.conv.1.running_var",
    "features.15.conv.3.weight": "layer6.1.conv.3.weight",
    "features.15.conv.4.weight": "layer6.1.conv.4.weight",
    "features.15.conv.4.bias": "layer6.1.conv.4.bias",
    "features.15.conv.4.running_mean": "layer6.1.conv.4.running_mean",
    "features.15.conv.4.running_var": "layer6.1.conv.4.running_var",
    "features.15.conv.6.weight": "layer6.1.conv.6.weight",
    "features.15.conv.7.weight": "layer6.1.conv.7.weight",
    "features.15.conv.7.bias": "layer6.1.conv.7.bias",
    "features.15.conv.7.running_mean": "layer6.1.conv.7.running_mean",
    "features.15.conv.7.running_var": "layer6.1.conv.7.running_var",

    "features.16.conv.0.weight": "layer6.2.conv.0.weight",
    "features.16.conv.1.weight": "layer6.2.conv.1.weight",
    "features.16.conv.1.bias": "layer6.2.conv.1.bias",
    "features.16.conv.1.running_mean": "layer6.2.conv.1.running_mean",
    "features.16.conv.1.running_var": "layer6.2.conv.1.running_var",
    "features.16.conv.3.weight": "layer6.2.conv.3.weight",
    "features.16.conv.4.weight": "layer6.2.conv.4.weight",
    "features.16.conv.4.bias": "layer6.2.conv.4.bias",
    "features.16.conv.4.running_mean": "layer6.2.conv.4.running_mean",
    "features.16.conv.4.running_var": "layer6.2.conv.4.running_var",
    "features.16.conv.6.weight": "layer6.2.conv.6.weight",
    "features.16.conv.7.weight": "layer6.2.conv.7.weight",
    "features.16.conv.7.bias": "layer6.2.conv.7.bias",
    "features.16.conv.7.running_mean": "layer6.2.conv.7.running_mean",
    "features.16.conv.7.running_var": "layer6.2.conv.7.running_var",
    # layer7
    "features.17.conv.0.weight": "layer7.0.conv.0.weight",
    "features.17.conv.1.weight": "layer7.0.conv.1.weight",
    "features.17.conv.1.bias": "layer7.0.conv.1.bias",
    "features.17.conv.1.running_mean": "layer7.0.conv.1.running_mean",
    "features.17.conv.1.running_var": "layer7.0.conv.1.running_var",
    "features.17.conv.3.weight": "layer7.0.conv.3.weight",
    "features.17.conv.4.weight": "layer7.0.conv.4.weight",
    "features.17.conv.4.bias": "layer7.0.conv.4.bias",
    "features.17.conv.4.running_mean": "layer7.0.conv.4.running_mean",
    "features.17.conv.4.running_var": "layer7.0.conv.4.running_var",
    "features.17.conv.6.weight": "layer7.0.conv.6.weight",
    "features.17.conv.7.weight": "layer7.0.conv.7.weight",
    "features.17.conv.7.bias": "layer7.0.conv.7.bias",
    "features.17.conv.7.running_mean": "layer7.0.conv.7.running_mean",
    "features.17.conv.7.running_var": "layer7.0.conv.7.running_var",
}

def pred(inp, oup, conv_operator, k, batch_norm):
    # the last 1x1 convolutional layer is very important
    hlConv2d = hlconv[conv_operator]
    return nn.Sequential(
        hlConv2d(inp, oup, k, 1, batch_norm),
        nn.Conv2d(oup, oup, k, 1, padding=k//2, bias=False)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, dilation, expand_ratio, batch_norm):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        BatchNorm2d = batch_norm

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.kernel_size = 3
        self.dilation = dilation

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )

    def fixed_padding(self, inputs, kernel_size, dilation):
        kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
        return padded_inputs

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.kernel_size == (3, 3):
                m.dilation = (dilate, dilate)
                m.padding = (dilate, dilate)

    def forward(self, x):
        x_pad = self.fixed_padding(x, self.kernel_size, dilation=self.dilation)
        if self.use_res_connect:
            return x + self.conv(x_pad)
        else:
            return self.conv(x_pad)




#######################################################################################
# RefineNet B2
#######################################################################################
class CRPBlock(nn.Module):
    def __init__(self, inp, oup, n_stages, batch_norm):
        super(CRPBlock, self).__init__()
        BatchNorm2d = batch_norm
        for i in range(n_stages):
            setattr(
                self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                conv_bn(inp if (i == 0) else oup, oup, 1, 1, BatchNorm2d)
            )
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            x = top + x
        return x


#######################################################################################
# IndexNet
#######################################################################################
class hlMobileNetV2UNetDecoderIndexLearning(nn.Module):
    def __init__(
        self, 
        output_stride=32, 
        input_size=320, 
        width_mult=1., 
        conv_operator='std_conv',
        decoder_kernel_size=5,
        apply_aspp=False,
        freeze_bn=False,
        use_nonlinear=False,
        use_context=False,
        indexnet='holistic',
        index_mode='o2o',
        sync_bn=False
        ):
        super(hlMobileNetV2UNetDecoderIndexLearning, self).__init__()

        self.Encoder = Encoder(output_stride=output_stride, 
                               input_size=input_size, 
                               width_mult=width_mult, 
                               conv_operator=conv_operator,
                               decoder_kernel_size=decoder_kernel_size,
                               apply_aspp=apply_aspp,
                               freeze_bn=freeze_bn,
                               use_nonlinear=use_nonlinear,
                               use_context=use_context,
                               indexnet=indexnet,
                               index_mode=index_mode,
                               sync_bn=sync_bn)


        self.AlignedNet = AlignedNet(idim=160, odim=160)
        
        self.ConvGRU = ConvGRU.ConvGRUCell(160, 160, 40, kernel_size=1)
        self.propagate_layers = 3
        self.conv_fusion = nn.Conv2d(160*2, 160, kernel_size=3, padding=1, bias= True)
        self.channel = 160
        self.linear_e = nn.Linear(160, 160,bias = False)
        self.gate = nn.Conv2d(160, 1, kernel_size  = 1, bias = False)
        self.gate_s = nn.Sigmoid()

        self.conv1 = nn.Conv2d(160, 160, kernel_size=3, padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(160)


        self.Decoder = Decoder(output_stride=output_stride, 
                               input_size=input_size, 
                               width_mult=width_mult, 
                               conv_operator=conv_operator,
                               decoder_kernel_size=decoder_kernel_size,
                               apply_aspp=apply_aspp,
                               freeze_bn=freeze_bn,
                               use_nonlinear=use_nonlinear,
                               use_context=use_context,
                               indexnet=indexnet,
                               index_mode=index_mode,
                               sync_bn=sync_bn)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):

                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, pf, ptris, cf, ctris, pf2, ptris2):
        pl, px6, px5, px4, px3, px2, px1, px0 = self.Encoder(pf, ptris)
        pl2, p2x6, p2x5, p2x4, p2x3, p2x2, p2x1, p2x0 = self.Encoder(pf2, ptris2)
        cl, x6, x5, x4, x3, x2, x1, x0 = self.Encoder(cf, ctris)

        input_size = pf.size()[2:]
        batch_num = pf.size()[0]

        exemplars = cl
        querys = pl
        query1s = pl2

        for passing_round in range(self.propagate_layers):


            attention1 = self.conv_fusion(torch.cat([self.AlignedNet(querys, exemplars),
                                     self.AlignedNet(query1s, exemplars)],1)) #message passing with concat operation
            attention2 = self.conv_fusion(torch.cat([self.AlignedNet(exemplars, querys),
                                    self.AlignedNet(query1s, querys)],1))
            attention3 = self.conv_fusion(torch.cat([self.AlignedNet(exemplars, query1s),
                                    self.AlignedNet(querys, query1s)],1))

            h_v1 = self.ConvGRU(attention1, exemplars)
            h_v2 = self.ConvGRU(attention2, querys)
            h_v3 = self.ConvGRU(attention3, query1s)
            
            exemplars = h_v1.clone()
            querys = h_v2.clone()
            query1s = h_v3.clone()
            
            if passing_round == self.propagate_layers -1:
                x1s, x1s_fg = self.Decoder(h_v1, x6, x5, x4, x3, x2, x1, x0)
                x2s, x2s_fg = self.Decoder(h_v2, px6, px5, px4, px3, px2, px1, px0)
                x3s, x3s_fg = self.Decoder(h_v3, p2x6, p2x5, p2x4, p2x3, p2x2, p2x1, p2x0)

        return x2s, x2s_fg, x1s, x1s_fg, x3s, x3s_fg

    def generate_batchwise_data(self, ii, px6, px5, px4, px3, px2, px1, px0):
        px6_ = [px6[0][ii].unsqueeze(0), px6[1]]
        px5_ = px5[ii].unsqueeze(0)
        if px4[1] is None:
            px4_ = [px4[0][ii].unsqueeze(0), px4[1]]
        else:
            px4_ = [px4[0][ii].unsqueeze(0), px4[1][ii].unsqueeze(0)]
        px3_ = [px3[0][ii].unsqueeze(0), px3[1][ii].unsqueeze(0)]
        px2_ = [px2[0][ii].unsqueeze(0), px2[1][ii].unsqueeze(0)]
        px1_ = px1[ii].unsqueeze(0)
        px0_ = [px0[0][ii].unsqueeze(0), px0[1][ii].unsqueeze(0)]
        return px6_, px5_, px4_, px3_, px2_, px1_, px0_


class AlignedNet(nn.Module):
    # Not using location
    def __init__(self, idim, odim):
        super(AlignedNet, self).__init__()

        self.R1_offset_conv1 = nn.Conv2d(idim, odim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.R1_offset_conv2 = nn.Conv2d(odim, odim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.R1_dcnpack = DCN(idim, odim, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
                               extra_offset_mask=True)

        self.RQ_conv = nn.Conv2d(odim*2, odim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        

    def forward(self, R1, Q0):
                
        R1_offset = R1 - Q0
        R1_offset = F.relu(self.R1_offset_conv1(R1_offset))
        R1_offset = F.relu(self.R1_offset_conv2(R1_offset))
        R1_fea = F.relu(self.R1_dcnpack([R1, R1_offset]))

        R1_fea_ = torch.cat([R1_fea, Q0], dim=1)
        R1_fea_ = F.relu(self.RQ_conv(R1_fea_))

        return R1_fea_


class Encoder(nn.Module):
    def __init__(
        self,
        output_stride=32, 
        input_size=320, 
        width_mult=1., 
        conv_operator='std_conv',
        decoder_kernel_size=5,
        apply_aspp=False,
        freeze_bn=False,
        use_nonlinear=False,
        use_context=False,
        indexnet='holistic',
        index_mode='o2o',
        sync_bn=False
        ):
        super(Encoder, self).__init__()
        self.width_mult = width_mult
        self.output_stride = output_stride
        self.index_mode = index_mode

        BatchNorm2d = SynchronizedBatchNorm2d if sync_bn else nn.BatchNorm2d

        block = InvertedResidual
        aspp = ASPP
        

        if indexnet == 'holistic':
            index_block = HolisticIndexBlock
        elif indexnet == 'depthwise':
            if 'o2o' in index_mode:
                index_block = DepthwiseO2OIndexBlock
            elif 'm2o' in index_mode:
                index_block = DepthwiseM2OIndexBlock
            else:
                raise NameError
        else:
            raise NameError

        initial_channel = 32
        current_stride = 1
        rate = 1
        inverted_residual_setting = [
            # expand_ratio, input_chn, output_chn, num_blocks, stride, dilation
            [1, initial_channel, 16, 1, 1, 1],
            [6, 16, 24, 2, 2, 1],
            [6, 24, 32, 3, 2, 1],
            [6, 32, 64, 4, 2, 1],
            [6, 64, 96, 3, 1, 1],
            [6, 96, 160, 3, 2, 1],
            [6, 160, 320, 1, 1, 1],
        ]

        ### encoder ###
        # building the first layer
        # assert input_size % output_stride == 0
        initial_channel = int(initial_channel * width_mult)
        self.layer0 = conv_bn(4, initial_channel, 3, 2, BatchNorm2d)
        self.layer0.apply(partial(self._stride, stride=1)) # set stride = 1
        current_stride *= 2
        # building bottleneck layers
        for i, setting in enumerate(inverted_residual_setting):
            s = setting[4]
            inverted_residual_setting[i][4] = 1 # change stride
            if current_stride == output_stride:
                rate *= s
                inverted_residual_setting[i][5] = rate
            else:
                current_stride *= s
        self.layer1 = self._build_layer(block, inverted_residual_setting[0], BatchNorm2d)
        self.layer2 = self._build_layer(block, inverted_residual_setting[1], BatchNorm2d, downsample=True)
        self.layer3 = self._build_layer(block, inverted_residual_setting[2], BatchNorm2d, downsample=True)
        self.layer4 = self._build_layer(block, inverted_residual_setting[3], BatchNorm2d, downsample=True)
        self.layer5 = self._build_layer(block, inverted_residual_setting[4], BatchNorm2d)
        self.layer6 = self._build_layer(block, inverted_residual_setting[5], BatchNorm2d, downsample=True)
        self.layer7 = self._build_layer(block, inverted_residual_setting[6], BatchNorm2d)

        # freeze encoder batch norm layers
        if freeze_bn:
            self.freeze_bn()
        
        # define index blocks
        if output_stride == 32:

            self.index0 = index_block(32, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)
            self.index2 = index_block(24, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)
            self.index3 = index_block(32, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)
            self.index4 = index_block(64, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)
            self.index6 = index_block(160, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)


        elif output_stride == 16:
            self.index0 = index_block(32, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)
            self.index2 = index_block(24, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)
            self.index3 = index_block(32, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)
            self.index4 = index_block(64, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)
        elif output_stride == 8:
            self.index0 = index_block(32, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)
            self.index2 = index_block(24, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)
            self.index3 = index_block(32, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)
        else:
            raise NotImplementedError
        
        ### context aggregation ###
        if apply_aspp:
            self.dconv_pp = aspp(320, 160, output_stride=output_stride, batch_norm=BatchNorm2d)
        else:
            self.dconv_pp = conv_bn(320, 160, 1, 1, BatchNorm2d)

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


    def _build_layer(self, block, layer_setting, batch_norm, downsample=False):
        t, p, c, n, s, d = layer_setting
        input_channel = int(p * self.width_mult)
        output_channel = int(c * self.width_mult)

        layers = []
        for i in range(n):
            if i == 0:
                d0 = d
                if downsample:
                    d0 = d // 2 if d > 1 else 1
                layers.append(block(input_channel, output_channel, s, d0, expand_ratio=t, batch_norm=batch_norm))
            else:
                layers.append(block(input_channel, output_channel, 1, d, expand_ratio=t, batch_norm=batch_norm))
            input_channel = output_channel

        return nn.Sequential(*layers)

    def _stride(self, m, stride):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.kernel_size == (3, 3):
                m.stride = stride
                return

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, img, tri):

        img = img / 255.
        img -= self.mean
        img /= self.std
        tri /= 255.

        x = torch.cat((img, tri), dim=1)

        # encode
        l0 = self.layer0(x)                                 # 4x320x320

        idx0_en, idx0_de = self.index0(l0)
        l0 = idx0_en * l0
        l0p = 4 * F.avg_pool2d(l0, (2, 2), stride=2)        # 32x160x160

        l1 = self.layer1(l0p)                               # 16x160x160
        l2 = self.layer2(l1)                                # 24x160x160

        idx2_en, idx2_de = self.index2(l2)
        l2 = idx2_en * l2
        l2p = 4 * F.avg_pool2d(l2, (2, 2), stride=2)        # 24x80x80
        
        l3 = self.layer3(l2p)                               # 32x80x80
    
        idx3_en, idx3_de = self.index3(l3)  
        l3 = idx3_en * l3
        l3p = 4 * F.avg_pool2d(l3, (2, 2), stride=2)        # 32x40x40

        l4 = self.layer4(l3p)                               # 64x40x40

        if self.output_stride == 8:
            l4p, idx4_de = l4, None
        else:
            idx4_en, idx4_de = self.index4(l4)
            l4 = idx4_en * l4
            l4p = 4 * F.avg_pool2d(l4, (2, 2), stride=2)        # 64x20x20


        l5 = self.layer5(l4p)                               # 96x20x20
        l6 = self.layer6(l5)                                # 160x20x20

        if self.output_stride == 32:
            idx6_en, idx6_de = self.index6(l6)
            l6 = idx6_en * l6
            l6p = 4 * F.avg_pool2d(l6, (2, 2), stride=2)    # 160x10x10
        elif self.output_stride == 16 or self.output_stride == 8:
            l6p, idx6_de = l6, None

        l7 = self.layer7(l6p)                               # 320x10x10

        # pyramid pooling
        xl = self.dconv_pp(l7)                               # 160x10x10

        return xl, [l6, idx6_de], l5, [l4, idx4_de], [l3, idx3_de], [l2, idx2_de], l1, [l0, idx0_de]


class Decoder(nn.Module):
    def __init__(
        self,
        output_stride=32, 
        input_size=320, 
        width_mult=1., 
        conv_operator='std_conv',
        decoder_kernel_size=5,
        apply_aspp=False,
        freeze_bn=False,
        use_nonlinear=False,
        use_context=False,
        indexnet='holistic',
        index_mode='o2o',
        sync_bn=False
        ):
        super(Decoder, self).__init__()
        decoder_block = IndexedUpsamlping
        BatchNorm2d = SynchronizedBatchNorm2d if sync_bn else nn.BatchNorm2d
        ### decoder ###
        self.decoder_layer6 = decoder_block(160*2, 96, conv_operator=conv_operator, kernel_size=decoder_kernel_size, batch_norm=BatchNorm2d)
        self.decoder_layer5 = decoder_block(96*2, 64, conv_operator=conv_operator, kernel_size=decoder_kernel_size, batch_norm=BatchNorm2d)
        self.decoder_layer4 = decoder_block(64*2, 32, conv_operator=conv_operator, kernel_size=decoder_kernel_size, batch_norm=BatchNorm2d)
        self.decoder_layer3 = decoder_block(32*2, 24, conv_operator=conv_operator, kernel_size=decoder_kernel_size, batch_norm=BatchNorm2d)
        self.decoder_layer2 = decoder_block(24*2, 16, conv_operator=conv_operator, kernel_size=decoder_kernel_size, batch_norm=BatchNorm2d)
        self.decoder_layer1 = decoder_block(16*2, 32, conv_operator=conv_operator, kernel_size=decoder_kernel_size, batch_norm=BatchNorm2d)
        self.decoder_layer0 = decoder_block(32*2, 32, conv_operator=conv_operator, kernel_size=decoder_kernel_size, batch_norm=BatchNorm2d)
        self.pred = pred(32, 1, conv_operator, k=decoder_kernel_size, batch_norm=BatchNorm2d)

        self.decoder_layer6_fg = decoder_block(160*2, 96, conv_operator=conv_operator, kernel_size=decoder_kernel_size, batch_norm=BatchNorm2d)
        self.decoder_layer5_fg = decoder_block(96*2, 64, conv_operator=conv_operator, kernel_size=decoder_kernel_size, batch_norm=BatchNorm2d)
        self.decoder_layer4_fg = decoder_block(64*2, 32, conv_operator=conv_operator, kernel_size=decoder_kernel_size, batch_norm=BatchNorm2d)
        self.decoder_layer3_fg = decoder_block(32*2, 24, conv_operator=conv_operator, kernel_size=decoder_kernel_size, batch_norm=BatchNorm2d)
        self.decoder_layer2_fg = decoder_block(24*2, 16, conv_operator=conv_operator, kernel_size=decoder_kernel_size, batch_norm=BatchNorm2d)
        self.decoder_layer1_fg = decoder_block(16*2, 32, conv_operator=conv_operator, kernel_size=decoder_kernel_size, batch_norm=BatchNorm2d)
        self.decoder_layer0_fg = decoder_block(32*2, 32, conv_operator=conv_operator, kernel_size=decoder_kernel_size, batch_norm=BatchNorm2d)
        self.pred_fg = pred(32, 3, conv_operator, k=3, batch_norm=BatchNorm2d)
        
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, xl, x6, l5, x4, x3, x2, l1, x0):
        l6, idx6_de = x6[0], x6[1]
        l4, idx4_de = x4[0], x4[1]
        l3, idx3_de = x3[0], x3[1]
        l2, idx2_de = x2[0], x2[1]
        l0, idx0_de = x0[0], x0[1]

        # decode
        xl_alpha = self.decoder_layer6(xl, l6, idx6_de)
        xl_fg = self.decoder_layer6_fg(xl, l6, idx6_de)

        xl_alpha = self.decoder_layer5(xl_alpha, l5)
        xl_fg = self.decoder_layer5_fg(xl_fg, l5)

        xl_alpha = self.decoder_layer4(xl_alpha, l4, idx4_de)
        xl_fg = self.decoder_layer4_fg(xl_fg, l4, idx4_de)

        xl_alpha = self.decoder_layer3(xl_alpha, l3, idx3_de)
        xl_fg = self.decoder_layer3_fg(xl_fg, l3, idx3_de)

        xl_alpha = self.decoder_layer2(xl_alpha, l2, idx2_de)
        xl_fg = self.decoder_layer2_fg(xl_fg, l2, idx2_de)

        xl_alpha = self.decoder_layer1(xl_alpha, l1)
        xl_fg = self.decoder_layer1_fg(xl_fg, l1)

        xl_alpha = self.decoder_layer0(xl_alpha, l0, idx0_de)
        xl_fg = self.decoder_layer0_fg(xl_fg, l0, idx0_de)

        xl_alpha = self.pred(xl_alpha)
        xl_fg = self.pred_fg(xl_fg)
        xl_fg *= self.std
        xl_fg += self.mean

        return xl_alpha, xl_fg

def hlmobilenetv2(pretrained=False, decoder='unet_style', **kwargs):
    """Constructs a MobileNet_V2 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if decoder == 'unet_style':
        model = hlMobileNetV2UNetDecoder(**kwargs)
    elif decoder == 'indexnet':
        model = hlMobileNetV2UNetDecoderIndexLearning(**kwargs)
    elif decoder == 'deeplabv3+':
        model = MobileNetV2DeepLabv3Plus(**kwargs)
    elif decoder == 'refinenet':
        model = hlMobileNetV2RefineNet(**kwargs)
    else:
        raise NotImplementedError

    if pretrained:
        corresp_name = CORRESP_NAME
        model_dict = model.state_dict()
        pretrained_dict = load_url(model_urls['mobilenetv2'])

        for name in pretrained_dict:

            if name not in corresp_name:
                continue
            # if corresp_name[name] not in model_dict.keys():
            #     continue

            if 'Encoder.' + corresp_name[name] not in model_dict.keys():
                continue

            if name == "features.0.0.weight":
                # model_weight = model_dict[corresp_name[name]]
                model_weight = model_dict['Encoder.'+corresp_name[name]]
                assert model_weight.shape[1] == 4
                model_weight[:, 0:3, :, :] = pretrained_dict[name]
                model_weight[:, 3, :, :] = torch.tensor(0)
                # model_dict[corresp_name[name]] = model_weight
                model_dict['Encoder.'+corresp_name[name]] = model_weight

            else:
                # model_dict[corresp_name[name]] = pretrained_dict[name]

                model_dict['Encoder.'+corresp_name[name]] = pretrained_dict[name]

        model.load_state_dict(model_dict)

    return model


def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)


if __name__ == "__main__":
    import numpy as np

    net = hlmobilenetv2(
        width_mult=1,
        pretrained=True, 
        freeze_bn=True, 
        sync_bn=False,
        apply_aspp=True,
        output_stride=32,
        conv_operator='std_conv',
        decoder_kernel_size=5,
        decoder='unet_style',
        indexnet='depthwise',
        index_mode='m2o',
        use_nonlinear=True,
        use_context=True,
    )
    net.eval()
    net.cuda()

    dump_x = torch.randn(1, 4, 224, 224).cuda()
    print(get_model_summary(net, dump_x))

    frame_rate = np.zeros((10, 1))
    for i in range(10):
        x = torch.randn(1, 4, 320, 320).cuda()
        torch.cuda.synchronize()
        start = time()
        y = net(x)
        torch.cuda.synchronize()
        end = time()
        running_frame_rate = 1 * float(1 / (end - start))
        frame_rate[i] = running_frame_rate
    print(np.mean(frame_rate))
    print(y.shape)
