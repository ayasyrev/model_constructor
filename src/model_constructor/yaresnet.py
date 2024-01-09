# YaResBlock - former NewResBlock.
# Yet another ResNet.

from functools import partial
from typing import List, Optional, Type

import torch
from torch import nn

from model_constructor.helpers import MakeModule, nn_seq, nnModule

from .layers import ConvBnAct, get_act
from .model_constructor import ListStrMod, ModelConstructor
from .xresnet import xresnet_stem

__all__ = [
    "YaBasicBlock",
    "YaBottleneckBlock",
    "McYaResNet",
    "McYaResNet34",
    "McYaResNet50",
]


class YaBasicBlock(nn.Module):
    """Ya Basic block.
    Reduce by pool instead of stride 2"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        conv_layer: Type[ConvBnAct] = ConvBnAct,
        act_fn: nnModule = nn.ReLU,
        zero_bn: bool = True,
        bn_1st: bool = True,
        groups: int = 1,
        dw: bool = False,
        div_groups: Optional[int] = None,
        pool: Optional[nnModule] = None,
        se: Optional[nnModule] = None,
        sa: Optional[nnModule] = None,
    ):
        super().__init__()
        # pool defined at ModelConstructor.
        if div_groups is not None:  # check if groups != 1 and div_groups
            groups = int(out_channels / div_groups)

        if stride != 1:
            if pool is None:
                self.reduce = conv_layer(in_channels, in_channels, 1, stride=2)
                # warnings.warn("pool not passed")  # need to warn?
            else:
                self.reduce = pool()
        else:
            self.reduce = None

        layers: ListStrMod = [
            (
                "conv_0",
                conv_layer(
                    in_channels,
                    out_channels,
                    3,
                    act_fn=act_fn,
                    bn_1st=bn_1st,
                    groups=in_channels if dw else groups,
                ),
            ),
            (
                "conv_1",
                conv_layer(
                    out_channels,
                    out_channels,
                    3,
                    zero_bn=zero_bn,
                    act_fn=False,
                    bn_1st=bn_1st,
                    groups=out_channels if dw else groups,
                ),
            ),
        ]
        if se:
            layers.append(("se", se(out_channels)))
        if sa:
            layers.append(("sa", sa(out_channels)))
        self.convs = nn_seq(layers)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reduce:
            x = self.reduce(x)
        identity = self.id_conv(x) if self.id_conv is not None else x
        return self.merge(self.convs(x) + identity)


class YaBottleneckBlock(nn.Module):
    """Ya Bottleneck block.
    Reduce by pool instead of stride 2"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 4,
        conv_layer: Type[ConvBnAct] = ConvBnAct,
        act_fn: nnModule = nn.ReLU,
        zero_bn: bool = True,
        bn_1st: bool = True,
        groups: int = 1,
        dw: bool = False,
        div_groups: Optional[int] = None,
        pool: Optional[nnModule] = None,
        se: Optional[nnModule] = None,
        sa: Optional[nnModule] = None,
    ):
        super().__init__()
        # pool defined at ModelConstructor.
        mid_channels = out_channels // expansion
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

        layers: ListStrMod = [
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
            ),
        ]
        if se:
            layers.append(("se", se(out_channels)))
        if sa:
            layers.append(("sa", sa(out_channels)))
        self.convs = nn_seq(layers)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reduce:
            x = self.reduce(x)
        identity = self.id_conv(x) if self.id_conv is not None else x
        return self.merge(self.convs(x) + identity)


class McYaResNet(ModelConstructor):
    make_stem: MakeModule = xresnet_stem
    stem_sizes: List[int] = [32, 64, 64]
    block: nnModule = YaBasicBlock
    act_fn: nnModule = nn.Mish
    pool: Optional[nnModule] = partial(nn.AvgPool2d, kernel_size=2, ceil_mode=True)


class McYaResNet34(McYaResNet):
    layers: List[int] = [3, 4, 6, 3]


class McYaResNet26(McYaResNet):
    block: nnModule = YaBottleneckBlock
    block_sizes: List[int] = [256, 512, 1024, 2048]


class McYaResNet50(McYaResNet26):
    layers: List[int] = [3, 4, 6, 3]
