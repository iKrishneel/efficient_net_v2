#!/usr/bin/env python

import pytest
import torch

from efficient_net_v2.config.config import get_cfg
from efficient_net_v2.model import EfficientNetV2


@pytest.fixture(scope='function')
def cfg():
    return get_cfg()


@pytest.fixture(scope='function')
def data(cfg):
    in_shape = cfg.get('INPUTS').get('SHAPE')
    return torch.randn(in_shape, dtype=torch.float32).unsqueeze(0)
    

def test_net(cfg, data):

    model = EfficientNetV2(cfg)
    print(model)

    r = model(data)

    print(r.shape)


if __name__ == '__main__':

    c = get_cfg()
    in_shape = c.get('INPUTS').get('SHAPE')
    d = torch.randn(in_shape, dtype=torch.float32).unsqueeze(0)

    test_net(c, d)
