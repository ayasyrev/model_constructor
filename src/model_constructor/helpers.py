from collections import OrderedDict

from torch import nn


def nn_seq(list_of_tuples: list[tuple[str, nn.Module]]) -> nn.Sequential:
    """return nn.Sequential from OrderedDict from list of tuples"""
    return nn.Sequential(OrderedDict(list_of_tuples))
