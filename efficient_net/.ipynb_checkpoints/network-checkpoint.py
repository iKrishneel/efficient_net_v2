#!/usr/bin/env python

from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import functional as F

from efficient_net.activation import Swish
# from efficient_net.config import MBConfig
from efficient_net import MBConfig


class ConvBNA(nn.Module):

    def __init__(self,
                 use_bn: bool = True,
                 activation=None,
                 **kwargs: dict):
        super(ConvBNA, self).__init__()
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

    def __init__(self,
                 num_channels: int,
                 activation=Swish,
                 ratio: float=1.0):
        super(SqueezeExcitation, self).__init__()

        num_reduced_channels = int(max(num_channels // ratio, 1))
        self.conv_sq = nn.Conv2d(
            num_channels, num_reduced_channels, kernel_size=1,
            bias=True, groups=1, padding=0)
        self.conv_ex = nn.Conv2d(
            num_reduced_channels, num_channels, kernel_size=1,
            bias=True, groups=1, padding=0)
        self.activation = activation()

    def forward(self, inp):
        x = F.adaptive_avg_pool2d(inp, 1)
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
            padding=0,
            groups=1, bias=False,
            bn_momentum=bn_momentum,
            bn_eps=bn_eps)
        dw_attrs = dict(
            in_channels=inner_channels,
            out_channels=inner_channels,
            kernel_size=config.KERNEL_SIZE,
            stride=config.STRIDES,
            groups=inner_channels, bias=False,
            padding=config.padding,
            bn_momentum=bn_momentum,
            bn_eps=bn_eps)
        op_attrs = dict(
            in_channels=inner_channels,
            out_channels=config.OUT_CHANNELS,
            kernel_size=1, stride=1,
            groups=1, bias=False,
            padding=0,
            bn_momentum=bn_momentum,
            bn_eps=bn_eps)

        activation = config.ACTIVATION
        self.conv_ip = ConvBNA(activation=activation, **ex_attrs)
        self.conv_dw = ConvBNA(activation=activation, **dw_attrs)
        self.conv_op = ConvBNA(**op_attrs)

        if config.HAS_SE:
            self._sqex = SqueezeExcitation(
                num_channels=inner_channels,
                ratio=config.REDUCTION_RATIO)

    def forward(self, inputs):
        x = self.conv_ip(inputs)
        x = self.conv_dw(x)
        x = self._sqex(x) if self.config.HAS_SE else x
        x = self.conv_op(x)
        if self.config.identity_skip:
            # todo: replace with drop_connect
            x = F.dropout(x, p=self.config.DROPOUT_PROB,
                          training=self.config.TRAINING)
            x = x + inputs
        return x


class EfficientNetX(nn.Module):

    def __init__(self, model_definition: dict):
        super(EfficientNetX, self).__init__()

        modules = []
        for definition in model_definition:
            operator = definition['operator'].lower()
            config1 = definition['config']
            layers = definition['layers']

            out_channels = config1.OUT_CHANNELS
            for i in range(layers):
                config = deepcopy(config1)
                if i > 0:
                    config.IN_CHANNELS = out_channels
                    config.STRIDES = 1

                module = self._get_module(operator, config)
                modules.append(module)
                out_channels = config.OUT_CHANNELS

        self._network = nn.Sequential(*modules)
        print(self._network)

    def forward(self, inputs):
        x = self._network(inputs)
        # x = torch.sigmoid(x)
        return x

    def _get_module(self, operator, config):
        if operator == 'conv2d':
            module = ConvBNA(
                use_bn=True, activation=None,
                **dict(
                    in_channels=config.IN_CHANNELS,
                    out_channels=config.OUT_CHANNELS,
                    kernel_size=config.KERNEL_SIZE,
                    stride=config.STRIDES,
                    padding=config.padding,
                    bn_momentum=config.BATCH_NORM_MOMENTUM,
                    bn_eps=config.BATCH_NORM_EPS))
        elif operator == 'fc' or operator == 'linear':
            module = nn.Linear(in_features=config.IN_CHANNELS,
                               out_features=config.OUT_CHANNELS,
                               bias=config.HAS_BIAS)
        elif operator == 'mbconv':
            module = MBConvX(config)
        elif operator == 'apool':
            module = nn.AdaptiveAvgPool2d(config.OUT_CHANNELS)
        elif operator == 'dropout':
            module = nn.Dropout(config.DROPOUT_PROB)
        elif operator == 'flatten':
            module = nn.Flatten()
        else:
            raise TypeError('Unknown/unsupported model type')
        return module


if __name__ == '__main__':

    import numpy as np
    x = np.random.random((10, 3, 224, 224)).astype(np.float32)
    y = torch.from_numpy(x)

    from efficient_net import model
    e = EfficientNetX(model.efficient_net_b0)
    x = e(y)
    print(x)
