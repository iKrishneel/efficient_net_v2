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

    BATCH_NORM_MOMENTUM: float = 0.01

    BATCH_NORM_EPS: float = 1E-3

    HAS_SE: bool = True

    REDUCTION_RATIO: int = 16

    DROPOUT_PROB: float = 0.2

    ACTIVATION = Swish

    TRAINING: bool = True

    @property
    def identity_skip(self):
        return self.ID_SKIP and \
            self.IN_CHANNELS == self.OUT_CHANNELS

    @property
    def padding(self):
        return max(self.KERNEL_SIZE + 1 - self.STRIDES, 0) // 2


class TrainConfig(object):

    MODEL: str = 'model.efficient_net_b0'
    
    DATASET: str = None

    LOG_DIR: str = 'logs'

    EPOCHS: int = 100

    ITER_PER_EPOCH: int = 1000
    
    # optimizer params
    LR: float = 0.256

    LR_DECAY: float = 0.97

    LR_DECAY_EPOCH: int = 2

    MOMENTUM: float = 0.9
    
    WEIGHT_DECAY: float = 1e-5

    # input
    INPUT_SHAPE: list = [3, 224, 224]

    DEVICE_ID: int = 0
    
    
    def display(self):
        """Display configuration values
        Prints all configuration values to the cout.
        """
        print("\nOptimization Configurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
