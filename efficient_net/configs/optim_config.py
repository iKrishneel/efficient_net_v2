#!/usr/bin/env python

from efficient_net import model


class TrainConfig(object):

    # MODEL: str = 'model.efficient_net_b0'
    MODEL: dict = model.efficient_net_b0
    
    DATASET: str = None

    LOG_DIR: str = 'logs/'

    BATCH_SIZE: int = 1
        
    VAL_BATCH_SIZE: int = 1
    
    EPOCHS: int = 200

    ITER_PER_EPOCH: int = 1000
    
    # optimizer params
    LR: float = 0.01

    LR_DECAY: float = 0.57

    LR_DECAY_EPOCH: int = 20

    MOMENTUM: float = 0.9
    
    WEIGHT_DECAY: float = 1e-4
        
    SCHEDULE_LR: bool = False

    # input
    INPUT_SHAPE: list = [3, 224, 224]

    DEVICE_ID: int = 0

    SNAPSHOT_NAME: str = 'efficient_net'

    SNAPSHOT_EPOCH: int = 1

    WEIGHT_FILE: str = None
    
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
