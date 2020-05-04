#!/usr/bin/env python

import torch
import torch.nn as nn


class ConvBNR(nn.Module):

    def __init__(self,
                 use_bn: bool=True,
                 use_relu: bool=True,
                 **kwargs: dict):
        super(ConvBNR, self).__init__()
        self.bn = None
        self.relu = None

        if use_bn:
            self.bn = nn.BatchNorm2d(num_features=kwargs['out_channels'],
                                     momentum=kwargs.pop('bn_momentum'),
                                     eps=kwargs.pop('bn_eps'))
        self.conv = nn.Conv2d(**kwargs)
        if use_relu:
            self.relu = nn.ReLU()

    def forward(self, inp):
        x = self.conv(inp)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class MBConvX(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(MBConvX, self).__init__()

        # self._use_se = True
        
        inner_channels = in_channels * 6
        
        ex_attrs = dict(
            in_channels=in_channels, out_channels=inner_channels,
            kernel_size=1, stride=1, groups=1, bias=False,
            bn_momentum=0.9, bn_eps=1e-5)
        dw_attrs = dict(
            in_channels=inner_channels, out_channels=inner_channels,
            kernel_size=3, stride=1, groups=inner_channels, bias=False,
            bn_momentum=0.9, bn_eps=1e-5)
        op_attrs = dict(
            in_channels=inner_channels, out_channels=out_channels,
            kernel_size=1, stride=1, groups=1, bias=False,
            bn_momentum=0.9, bn_eps=1e-5)
        
        self.conv_ip = ConvBNR(**ex_attrs)
        self.conv_dw = ConvBNR(**dw_attrs)
        self.conv_op = ConvBNR(use_relu=False, **op_attrs)

    def forward(self, inputs):
        x = inputs

        return x
    

class EfficientNetBase(nn.Module):
    
    def __init__(self, ):
        pass


if __name__ == '__main__':
    e = EfficientNetBase()
    print(vars(e))
    
