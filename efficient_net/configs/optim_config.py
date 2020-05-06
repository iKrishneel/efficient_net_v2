#!/usr/bin/env python

from efficient_net import model


class TrainConfig(object):

    # MODEL: str = 'model.efficient_net_b0'
    MODEL: dict = model.efficient_net_b0
    
    DATASET: str = None

    LOG_DIR: str = 'logs/'

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
            if a == 'MODEL':
                continue
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
