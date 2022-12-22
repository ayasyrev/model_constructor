# YaResBlock - former NewResBlock.
# Yet another ResNet.

from collections import OrderedDict
from typing import Callable, List, Type, Union

import torch.nn as nn
from torch.nn import Mish

from .layers import ConvBnAct, get_act
from .model_constructor import ModelConstructor

__all__ = [
    "YaResBlock",
    "YaResNet34",
    "YaResNet50",
]


class YaResBlock(nn.Module):
    """YaResBlock. Reduce by pool instead of stride 2"""

    def __init__(
        self,
        expansion: int,
        in_channels: int,
        mid_channels: int,
        stride: int = 1,
        conv_layer=ConvBnAct,
        act_fn: Type[nn.Module] = nn.ReLU,
        zero_bn: bool = True,
        bn_1st: bool = True,
        groups: int = 1,
        dw: bool = False,
        div_groups: Union[None, int] = None,
        pool: Union[Callable[[], nn.Module], None] = None,
        se: Union[Type[nn.Module], None] = None,
        sa: Union[Type[nn.Module], None] = None,
    ):
        super().__init__()
        # pool defined at ModelConstructor.
        out_channels, in_channels = mid_channels * expansion, in_channels * expansion
        if div_groups is not None:  # check if groups != 1 and div_groups
            groups = int(mid_channels / div_groups)

        if stride != 1:
            if pool is None:
                self.reduce = conv_layer(in_channels, in_channels, 1, stride=2)
                # warnings.warn("pool not passed")  # need to warn?
            else:
                self.reduce = pool()
        else:
            self.reduce = None
        if expansion == 1:
            layers = [
                (
                    "conv_0",
                    conv_layer(
                        in_channels,
                        mid_channels,
                        3,
                        stride=1,
                        act_fn=act_fn,
                        bn_1st=bn_1st,
                        groups=in_channels if dw else groups,
                    ),
                ),
                (
                    "conv_1",
                    conv_layer(
                        mid_channels,
                        out_channels,
                        3,
                        zero_bn=zero_bn,
                        act_fn=False,
                        bn_1st=bn_1st,
                        groups=mid_channels if dw else groups,
                    ),
                ),
            ]
        else:
            layers = [
                (
                    "conv_0",
                    conv_layer(
                        in_channels,
                        mid_channels,
                        1,
                        act_fn=act_fn,
                        bn_1st=bn_1st,
                    ),
                ),
                (
                    "conv_1",
                    conv_layer(
                        mid_channels,
                        mid_channels,
                        3,
                        stride=1,
                        act_fn=act_fn,
                        bn_1st=bn_1st,
                        groups=mid_channels if dw else groups,
                    ),
                ),
                (
                    "conv_2",
                    conv_layer(
                        mid_channels,
                        out_channels,
                        1,
                        zero_bn=zero_bn,
                        act_fn=False,
                        bn_1st=bn_1st,
                    ),
                ),  # noqa E501
            ]
        if se:
            layers.append(("se", se(out_channels)))
        if sa:
            layers.append(("sa", sa(out_channels)))
        self.convs = nn.Sequential(OrderedDict(layers))
        if in_channels != out_channels:
            self.id_conv = conv_layer(
                in_channels,
                out_channels,
                1,
                stride=1,
                act_fn=False,
            )
        else:
            self.id_conv = None
        self.merge = get_act(act_fn)

    def forward(self, x):
        if self.reduce:
            x = self.reduce(x)
        identity = self.id_conv(x) if self.id_conv is not None else x
        return self.merge(self.convs(x) + identity)


class YaResNet34(ModelConstructor):
    block: type[nn.Module] = YaResBlock
    expansion: int = 1
    layers: list[int] = [3, 4, 6, 3]
    stem_sizes: list[int] = [3, 32, 64, 64]
    act_fn: Type[nn.Module] = Mish


class YaResNet50(YaResNet34):
    expansion: int = 4
