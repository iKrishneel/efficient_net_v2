#!/usr/bin/env python

from yacs.config import CfgNode as CN


_C = CN()

# input
_C.INPUTS = CN()
_C.INPUTS.SHAPE = (3, 224, 224)

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

# head
_C.HEAD = CN()

# conv
_C.HEAD.CONV = CN()
_C.HEAD.CONV.OPS = 'conv'
_C.HEAD.CONV.KERNEL = 1
_C.HEAD.CONV.STRIDE = 1
_C.HEAD.CONV.CHANNELS = 1792
_C.HEAD.CONV.LAYERS = 1

# average pool
_C.HEAD.AVG = CN()
_C.HEAD.AVG.OPS = 'AdaptiveAvgPool2d'
_C.HEAD.AVG.output_size = (1, 1)
_C.HEAD.AVG.LAYERS = 1

# flatten
_C.HEAD.FLAT = CN()
_C.HEAD.FLAT.OPS = 'Flatten'
_C.HEAD.FLAT.start_dim = 1
_C.HEAD.FLAT.end_dim = -1

# linear
_C.HEAD.FC = CN()
_C.HEAD.FC.OPS = 'Linear'
_C.HEAD.FC.in_features = 1792
_C.HEAD.FC.out_features = 20


# datasets
_C.DATASETS = CN()
_C.DATASETS.NUM_WORKER = 8

# solver
_C.SOLVER = CN()
_C.SOLVER.BATCH_SIZE = 10
_C.SOLVER.LR = 0.02
_C.SOLVER.LR_DECAY = 0.1
_C.SOLVER.LR_DECAY_STEP = 10
_C.SOLVER.EPOCHS = 100


_C.OUTPUT_DIR = './'

def get_cfg():
    return _C.clone()
