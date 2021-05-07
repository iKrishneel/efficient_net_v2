#!/usr/bin/env python

from yacs.config import CfgNode as CN


_C = CN()

# stages of the network
_C.BACKBONE = CN()

# stage 0
_C.BACKBONE.S0 = CN()
_C.BACKBONE.S0.OPS = 'conv'
_C.BACKBONE.S0.KERNEL = 3
_C.BACKBONE.S0.STRIDE = 2
_C.BACKBONE.S0.CHANNELS = 24
_C.BACKBONE.S0.LAYERS = 1
_C.BACKBONE.S0.PADDING = 1

# stage 1
_C.BACKBONE.S1 = CN()
_C.BACKBONE.S1.OPS = 'fused_mbconv'
_C.BACKBONE.S1.KERNEL = 3
_C.BACKBONE.S1.STRIDE = 1
_C.BACKBONE.S1.EXPANSION = 1
# _C.BACKBONE.S1.SE = 1
_C.BACKBONE.S1.CHANNELS = 24
_C.BACKBONE.S1.LAYERS = 2

# stage 2
_C.BACKBONE.S2 = CN()
_C.BACKBONE.S2.OPS = 'fused_mbconv'
_C.BACKBONE.S2.KERNEL = 3
_C.BACKBONE.S2.STRIDE = 2
_C.BACKBONE.S2.EXPANSION = 4
# _C.BACKBONE.S2.SE = 1
_C.BACKBONE.S2.CHANNELS = 48
_C.BACKBONE.S2.LAYERS = 4

# stage 3
_C.BACKBONE.S3 = CN()
_C.BACKBONE.S3.OPS = 'fused_mbconv'
_C.BACKBONE.S3.KERNEL = 3
_C.BACKBONE.S3.STRIDE = 2
_C.BACKBONE.S3.EXPANSION = 4
# _C.BACKBONE.S3.SE = 1
_C.BACKBONE.S3.CHANNELS = 64
_C.BACKBONE.S3.LAYERS = 4

# stage 4
_C.BACKBONE.S4 = CN()
_C.BACKBONE.S4.OPS = 'mbconv'
_C.BACKBONE.S4.KERNEL = 3
_C.BACKBONE.S4.STRIDE = 2
_C.BACKBONE.S4.EXPANSION = 4
_C.BACKBONE.S4.SE = 4
_C.BACKBONE.S4.CHANNELS = 128
_C.BACKBONE.S4.LAYERS = 6

# stage 5
_C.BACKBONE.S5 = CN()
_C.BACKBONE.S5.OPS = 'mbconv'
_C.BACKBONE.S5.KERNEL = 3
_C.BACKBONE.S5.STRIDE = 1
_C.BACKBONE.S5.EXPANSION = 6
_C.BACKBONE.S5.SE = 4
_C.BACKBONE.S5.CHANNELS = 160
_C.BACKBONE.S5.LAYERS = 9

# stage 6
_C.BACKBONE.S6 = CN()
_C.BACKBONE.S6.OPS = 'mbconv'
_C.BACKBONE.S6.KERNEL = 3
_C.BACKBONE.S6.STRIDE = 2
_C.BACKBONE.S6.EXPANSION = 6
_C.BACKBONE.S6.SE = 4
_C.BACKBONE.S6.CHANNELS = 272
_C.BACKBONE.S6.LAYERS = 15


def get_cfg():
    return _C.clone()