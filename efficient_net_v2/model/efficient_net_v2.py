#!/usr/bin/env python

from copy import deepcopy

import torch.nn as nn
from yacs.config import CfgNode as CN

from ..layers import ConvBNA, MBConv, FusedMBConv


class EfficientNetV2(nn.Module):
    def __init__(self, cfg: CN, in_channels: int = 3):
        super(EfficientNetV2, self).__init__()

        # input_shape = cfg.get('INPUTS').get('SHAPE')
        backbone = cfg['BACKBONE']
        # assert len(input_shape) == 3
        # in_channels = input_shape[0]

        layers, in_channels = self.build(backbone, in_channels)
        self.backbone = nn.Sequential(*layers)

        try:
            head = cfg['HEAD']
            layers, in_channels = self.build(head, in_channels)
            self.head = nn.Sequential(*layers)
        except KeyError:
            self.head = None

        self.out_channels = in_channels

    def build(self, nodes, in_channels):
        layers = []
        for index, (stage, node) in enumerate(nodes.items()):
            for i in range(node.pop('LAYERS', 1)):
                stride = node.get('STRIDE', 1) if i == 0 else 1
                assert stride
                layers.append(self.create_layer(node, in_channels, stride))
                in_channels = node.get('CHANNELS')

        return layers, in_channels

    def create_layer(self, node: CN, in_channels: int, stride: int):
        node = deepcopy(node)

        ops = node.pop('OPS')
        out_channels = node.pop('CHANNELS', None)
        kernel_size = node.get('KERNEL')
        expansion = node.get('EXPANSION')
        se = node.get('SE', 0)
        padding = node.get('PADDING', 0)

        if ops == 'conv':
            layer = ConvBNA(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        elif ops == 'mbconv':
            layer = MBConv(
                in_channels=in_channels,
                expansion=expansion,
                out_channels=out_channels,
                knxn=kernel_size,
                stride=stride,
                reduction=se,
            )
        elif ops == 'fused_mbconv':
            layer = FusedMBConv(
                in_channels=in_channels,
                expansion=expansion,
                out_channels=out_channels,
                knxn=kernel_size,
                stride=stride,
                reduction=se,
            )
        else:
            layer = getattr(nn, ops)
            if not issubclass(layer, nn.Module):
                raise ValueError(f'Unknown layer type {ops}')
            layer = layer(**node)

        return layer

    def forward(self, x):
        x = self.backbone(x)
        if self.head is not None:
            x = self.head(x)
        return x
