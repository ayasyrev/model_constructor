from typing import List

from torch import nn

from .helpers import nnModule
from .xresnet import McXResNet


class McMxResNet(McXResNet):
    stem_sizes: List[int] = [3, 32, 64, 64]
    act_fn: nnModule = nn.Mish


class McMxResNet34(McMxResNet):
    layers: List[int] = [3, 4, 6, 3]


class McMxResNet50(McMxResNet34):
    expansion: int = 4
    block_sizes: List[int] = [256, 512, 1024, 2048]
