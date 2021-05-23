#!/usr/bin/env python

from typing import List

import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import (
    BACKBONE_REGISTRY, Backbone, FPN, ShapeSpec
)

from .efficient_net_v2 import EfficientNetV2
from ..layers import ConvBNA, FusedMBConv, MBConv
from ..config.detectron2_config import get_cfg


class EfficientNet(EfficientNetV2, Backbone):
    def __init__(self, cfg, out_features: List[str] = None):
        super(EfficientNet, self).__init__(cfg)

        self.out_features = (
            ['s6'] if out_features is None else out_features
        )

        self.strides = {}
        self.channels = {}
        for key in self.out_features:
            index = self.stage_indices[key]
            stride, channel = 1, 0
            for child in self.backbone[:index].children():
                if isinstance(child, (ConvBNA, FusedMBConv, MBConv)):
                    stride, channel = (
                        stride * child.stride, child.out_channels
                    )
            self.strides[key] = stride
            self.channels[key] = channel

        assert len(self.channels) > 0

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self.channels[name], stride=self.strides[name]
            ) for name in self.out_features
        }

    def forward(self, x):
        features = self.stage_forward(x)
        return {
            name: features[name] for name in self.out_features
        }

    def freeze(self, at: int = 0):
        # TODO: freeze at selected layer
        for parameters in self.parameters():
            parameters.requires_grad = False
        return self


@BACKBONE_REGISTRY.register()
def build_effnet_backbone(cfg=None, input_shape=None):
    config = get_cfg()
    return EfficientNet(config, ['s2', 's3', 's4', 's5', 's6'])  # .freeze(0)


@BACKBONE_REGISTRY.register()
def build_effnet_fpn_backbone(cfg=None, input_shape=None):
    backbone = build_effnet_backbone()
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS

    return FPN(
        bottom_up=backbone,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )


class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]
