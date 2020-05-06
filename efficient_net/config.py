#!/usr/bin/env python

from dataclasses import dataclass
from efficient_net.activation import Swish


@dataclass
class MBConfig(object):

    IN_CHANNELS: int = 1

    OUT_CHANNELS: int = 1

    STRIDES: int = 1

    KERNEL_SIZE: int = 1

    EXPANSION_FACTOR: int = 0

    HAS_BIAS: bool = False

    ID_SKIP: bool = True

    BATCH_NORM_MOMENTUM: float = 0.9

    BATCH_NORM_EPS: float = 1E-5

    HAS_SE: bool = True

    DROPOUT_PROB: float = 0.0

    ACTIVATION = Swish

    TRAINING: bool = True

    @property
    def identity_skip(self):
        return self.ID_SKIP and \
            self.IN_CHANNELS == self.OUT_CHANNELS

    @property
    def padding(self):
        return max(self.KERNEL_SIZE + 1 - self.STRIDES, 0) // 2


class ENConfig(object):

    # MBConv config
    pass
