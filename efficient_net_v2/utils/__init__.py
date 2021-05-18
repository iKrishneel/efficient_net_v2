#!/usr/bin/env python


def make_divisible(v, divisor=8, min_value=None):
    """
    The channel number of each layer should be divisable by 8.
    The function is taken from
    github.com/rwightman/pytorch-image-models/master/timm/models/layers/helpers.py
    """
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
