#!/usr/bin/env python

import torch.nn as nn

from .conv import ConvBNA
from .sequeeze_excitation import SqueezeExcitation as SE


__all__ = ['MBConv', 'FusedMBConv']


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        expansion: int,
        out_channels: int = -1,
        **kwargs: dict,
    ):
        super(MBConv, self).__init__()

        self.out_channels = max(in_channels, out_channels)
        hidden_channels = in_channels * max(expansion, 1)
        reduction = kwargs.get('reduction', 4)
        knxn = kwargs.get('knxn', 3)
        bias = kwargs.get('bias', False)
        self.stride = kwargs.get('stride', 1)

        self.conv1 = ConvBNA(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            bias=bias,
        )
        self.conv2 = ConvBNA(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            groups=hidden_channels,
            kernel_size=knxn,
            padding=1,
            stride=self.stride,
            bias=bias,
        )

        self.se = (
            SE(
                in_channels=hidden_channels,
                reduction=reduction,
                out_channels=in_channels,
            )
            if reduction > 0
            else None
        )

        self.conv3 = ConvBNA(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
            activation=None,
        )

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.conv2(x)
        if self.se is not None:
            x = self.se(x)
        x = self.conv3(x)
        if x.shape[1:] == inp.shape[1:]:
            return x + inp
        return x


class FusedMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        expansion: int,
        out_channels: int = -1,
        **kwargs: dict,
    ):
        super(FusedMBConv, self).__init__()

        self.out_channels = max(in_channels, out_channels)
        hidden_channels = in_channels * max(expansion, 1)
        reduction = kwargs.get('reduction', 4)
        knxn = kwargs.get('knxn', 3)
        bias = kwargs.get('bias', False)
        self.stride = kwargs.get('stride', 1)

        self.conv1 = ConvBNA(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=knxn,
            padding=1,
            stride=self.stride,
            bias=bias,
        )
        self.se = (
            SE(
                in_channels=hidden_channels,
                reduction=reduction,
                out_channels=in_channels,
            )
            if reduction > 0
            else None
        )
        self.conv2 = ConvBNA(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            activation=None,
            bias=bias,
        )

    def forward(self, inp):
        x = self.conv1(inp)
        if self.se is not None:
            x = self.se(x)
        x = self.conv2(x)
        if x.shape[1:] == inp.shape[1:]:
            return x + inp
        return x
