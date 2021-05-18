#!/usr/bin/env python

from typing import List

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

from .efficient_net_v2 import EfficientNetV2
from ..layers import ConvBNA, FusedMBConv, MBConv
from ..config.detectron2_config import get_cfg


class EfficientNet(EfficientNetV2, Backbone):
    def __init__(self, cfg, out_features: List[str] = None):
        super(EfficientNet, self).__init__(cfg)

        self.out_features = ['stage7'] if out_features is None else out_features

        self._stride, self._channels = 1, 0
        for child in self.backbone.children():
            self._stride, self._channels = (
                (self._stride * child.stride, child.out_channels)
                if isinstance(child, (ConvBNA, FusedMBConv, MBConv))
                else (self._stride, self._channels)
            )

        assert self._channels > 0

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self._channels, stride=self._stride)
            for name in self.out_features
        }

    def forward(self, x):
        return {'stage7': super().forward(x)}


@BACKBONE_REGISTRY.register()
def build_effnet_backbone(cfg=None, input_shape=None):
    config = get_cfg()
    return EfficientNet(
        config,
    )
