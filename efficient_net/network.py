#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.nn import functional as F

from efficient_net.activation import Swish
from efficient_net.config import MBConfig


class ConvBNR(nn.Module):

    def __init__(self,
                 use_bn: bool=True,
                 activation=None,
                 **kwargs: dict):
        super(ConvBNR, self).__init__()
        self.bn = None
        self.activation = None

        if use_bn:
            self.bn = nn.BatchNorm2d(num_features=kwargs['out_channels'],
                                     momentum=kwargs.pop('bn_momentum'),
                                     eps=kwargs.pop('bn_eps'))
        self.conv = nn.Conv2d(**kwargs)
        if activation is not None:
            self.activation = activation()

    def forward(self, inp):
        x = self.conv(inp)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SqueezeExcitation(nn.Module):

    def __init__(self, num_channels: int, activation=Swish):
        super(SqueezeExcitation, self).__init__()
        
        num_reduced_channels = num_channels
        self.conv_sq = nn.Conv2d(
            num_channels, num_reduced_channels, kernel_size=1,
            bias=True, groups=1, padding=0)
        self.conv_ex = nn.Conv2d(
            num_reduced_channels, num_channels, kernel_size=1,
            bias=True, groups=1, padding=0)
        self.activation = activation()
        
    def forward(self, inp):
        x = inp
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.activation(self.conv_sq(x))
        x = torch.sigmoid(self.conv_ex(x))
        x = x * inp
        return x
        

class MBConvX(nn.Module):
    
    def __init__(self, config: MBConfig):
        super(MBConvX, self).__init__()

        self.config = config        
        inner_channels = config.IN_CHANNELS * config.EXPANSION_FACTOR
        bn_momentum = config.BATCH_NORM_MOMENTUM
        bn_eps = config.BATCH_NORM_EPS
        
        ex_attrs = dict(
            in_channels=config.IN_CHANNELS,
            out_channels=inner_channels,
            kernel_size=1, stride=1,
            padding=self.padding(1, 1),
            groups=1, bias=False,
            bn_momentum=bn_momentum,
            bn_eps=bn_eps)
        dw_attrs = dict(
            in_channels=inner_channels,
            out_channels=inner_channels,
            kernel_size=3, stride=1,
            groups=inner_channels, bias=False,
            padding=self.padding(3, 1),
            bn_momentum=bn_momentum,
            bn_eps=bn_eps)
        op_attrs = dict(
            in_channels=inner_channels,
            out_channels=config.OUT_CHANNELS,            
            kernel_size=1, stride=1,
            groups=1, bias=False,
            padding=self.padding(1, 1),
            bn_momentum=bn_momentum,
            bn_eps=bn_eps)
        
        self.conv_ip = ConvBNR(activation=Swish, **ex_attrs)
        self.conv_dw = ConvBNR(activation=Swish, **dw_attrs)
        self.conv_op = ConvBNR(**op_attrs)
        
        if config.HAS_SE:
            self._sqex = SqueezeExcitation(
                num_channels=inner_channels)
            
    def forward(self, inputs):
        x = inputs
        x = self.conv_ip(x)
        x = self.conv_dw(x)
        x = self._sqex(x) if self.config.HAS_SE else x
        x = self.conv_op(x)
        if self.config.identity_skip:
            x = F.dropout(x, p=self.config.DROPOUT_PROB,
                      training=self.config.TRAINING)
            x = x + inputs
        return x

    @staticmethod
    def padding(kernel_size: int, stride: int):
        return max(kernel_size - stride, 0) // 2
    

class EfficientNetBase(nn.Module):
    
    def __init__(self, ):
        pass


if __name__ == '__main__':

    import numpy as np
    x = np.random.random((1, 3, 32, 32)).astype(np.float32)
    y = torch.from_numpy(x)
    
    attrs = dict(in_channels=3, out_channels=32, kernel_size=1,
                 stride=1, bias=False, bn_momentum=0.9, bn_eps=0.001)

    config = MBConfig(IN_CHANNELS=3, OUT_CHANNELS=3, KERNEL_SIZE=1,
                      STRIDES=1, EXPANSION_FACTOR=6)
    print(config)
    # m = ConvBNR(True, torch.nn.ReLU, **attrs)
    
    m = MBConvX(config)
    m(y)
