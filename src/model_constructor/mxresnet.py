from functools import partial
from typing import List, Optional

from torch import nn

from .blocks import BottleneckBlock
from .helpers import nnModule
from .xresnet import McXResNet


class McMxResNet(McXResNet):
    stem_sizes: List[int] = [3, 32, 64, 64]
    act_fn: nnModule = nn.Mish
    pool: Optional[nnModule] = partial(nn.AvgPool2d, kernel_size=2, ceil_mode=True)



class McMxResNet34(McMxResNet):
    layers: List[int] = [3, 4, 6, 3]


class McMxResNet50(McMxResNet34):
    block: nnModule = BottleneckBlock
    expansion: int = 4
    block_sizes: List[int] = [256, 512, 1024, 2048]
