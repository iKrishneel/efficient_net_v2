#!/usr/bin/env python

import pytest
import torch

from efficient_net_v2.layers import FusedMBConv, MBConv


def test_fused_mbconv():
    in_c = 64
    t = 4
    m = FusedMBConv(in_c, expansion=t, reduction=0)
    print(m)
    assert m.se.fc1.in_channels == in_c * t

    x = torch.randn((1, in_c, 32, 32), dtype=torch.float32, requires_grad=False)
    r = m(x)
    assert x.shape == r.shape


def test_mbconv():
    in_c = 272
    t = 6
    m = MBConv(in_c, t)
    print(m)
    assert m.se.fc1.in_channels == in_c * t

    x = torch.randn((1, in_c, 32, 32), dtype=torch.float32, requires_grad=False)
    r = m(x)
    assert x.shape == r.shape


if __name__ == '__main__':

    # test_mbconv()
    test_fused_mbconv()
