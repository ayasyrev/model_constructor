from torch import nn

from .xresnet import XResNet


class MxResNet(XResNet):
    stem_sizes: list[int] = [3, 32, 64, 64]
    act_fn: type[nn.Module] = nn.Mish


class MxResNet34(MxResNet):
    layers: list[int] = [3, 4, 6, 3]


class MxResNet50(MxResNet34):
    expansion: int = 4
    block_sizes: list[int] = [256, 512, 1024, 2048]
