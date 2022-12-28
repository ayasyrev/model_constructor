from torch import nn

from .xresnet import XResNet, XResNet34, XResNet50


class MxResNet(XResNet):
    stem_sizes: list[int] = [3, 32, 64, 64]
    act_fn: type[nn.Module] = nn.Mish

class MxResNet34(XResNet34):
    stem_sizes: list[int] = [3, 32, 64, 64]
    act_fn: type[nn.Module] = nn.Mish


class MxResNet50(XResNet50):
    stem_sizes: list[int] = [3, 32, 64, 64]
    act_fn: type[nn.Module] = nn.Mish
