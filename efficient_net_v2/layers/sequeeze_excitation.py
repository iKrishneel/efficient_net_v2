#!/usr/bin/env python

import torch.nn as nn
from torch.nn import functional as F

from ..utils import make_divisible


class SqueezeExcitation(nn.Module):

    def __init__(
            self, in_channels: int, reduction: int = 4,
            out_channels: int = -1, **kwargs: dict
    ):
        super(SqueezeExcitation, self).__init__()
        assert in_channels > 0

        num_reduced_channels = make_divisible(
            max(out_channels, 8) // reduction, 8
        )
        
        self.fc1 = nn.Conv2d(
            in_channels, num_reduced_channels, kernel_size=1
        )
        self.fc2 = nn.Conv2d(
            num_reduced_channels, in_channels, kernel_size=1
        )
        self.activation = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        x = F.adaptive_avg_pool2d(inp, 1)
        x = self.activation(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x + inp


if __name__ == '__main__':

    s = SqueezeExcitation(10)
    print(s)
