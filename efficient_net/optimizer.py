#!/usr/bin/env python

import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn

from efficient_net.network import EfficientNetX as EfficientNet
from efficient_net.config import TrainConfig
from efficient_net import appname


def get_name(prefix=''):
    return prefix + '_' + datetime.now().strftime("%Y%m%d%H%M%S")


class Optimizer(EfficientNet):

    def __init__(self, config: TrainConfig):
        config.display()
        
        # logging directory setup
        if not os.path.isdir(config.LOG_DIR):
            os.mkdir(config.LOG_DIR)
        self._log_dir = os.path.join(config.LOG_DIR, get_name(appname()))
        os.mkdir(self._log_dir)

        # device
        device = torch.device(f'cuda:{config.DEVICE_ID}' \
                              if torch.cuda.is_available() else 'cpu')
        
        self._criterion = nn.CrossEntropyLoss()
        """
        self._optimizer = torch.optim.RMSprop(
            # params=
            lr=config.LR,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY,)"""
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train', action='store_true', default=True)
    parser.add_argument(
        '--dataset', type=str, required=True)
    parser.add_argument(
        '--log_dir', type=str, required=False, default='logs')
    args = parser.parse_args()

    config = TrainConfig()
    config.DATASET = args.dataset
    config.LOG_DIR = args.log_dir

    
    # o = Optimizer(config)
    
    model_def = config.MODEL
    module, defi = model_def.split('.')
    import importlib
    module = importlib.import_module('model')
    v = getattr(module, defi)
    print(v)
