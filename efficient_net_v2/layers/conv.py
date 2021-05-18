#!/usr/bin/env python

import torch.nn as nn


class ConvBNA(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_bn: bool = True,
        **kwargs: dict,
    ):
        super(ConvBNA, self).__init__()

        momentum = kwargs.pop('momentum', 0.1)
        eps = kwargs.pop('eps', 1e-5)
        self.activation = kwargs.pop('activation', nn.ReLU(inplace=True))
        self.stride = kwargs.get('stride', 1)
        self.out_channels = out_channels

        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, **kwargs
        )
        self.bn = (
            nn.BatchNorm2d(
                num_features=out_channels, momentum=momentum, eps=eps
            )
            if use_bn
            else None
        )

    def forward(self, inp):
        x = self.conv(inp)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
