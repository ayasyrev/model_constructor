from collections import OrderedDict
from typing import Iterable

from torch import nn


def nn_seq(list_of_tuples: Iterable[tuple[str, nn.Module]]) -> nn.Sequential:
    """return nn.Sequential from OrderedDict from list of tuples"""
    return nn.Sequential(OrderedDict(list_of_tuples))  #
