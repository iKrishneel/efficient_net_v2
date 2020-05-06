#!/usr/bin/env python

from .configs.enet_config import MBConfig
from .configs.optim_config import TrainConfig

__APPNAME__ = 'efnet'


def appname():
    return __APPNAME__
