from typing import Callable, Optional

import torch
from torch import nn

from .helpers import ListStrMod, nn_seq
from .layers import ConvBnAct, get_act


class BasicBlock(nn.Module):
    """Basic Resnet block.
    Configurable - can use pool to reduce at identity path, change act etc."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        conv_layer: type[ConvBnAct] = ConvBnAct,
        act_fn: type[nn.Module] = nn.ReLU,
        zero_bn: bool = True,
        bn_1st: bool = True,
        groups: int = 1,
        dw: bool = False,
        div_groups: Optional[int] = None,
        pool: Optional[Callable[[], nn.Module]] = None,
        se: Optional[nn.Module] = None,
        sa: Optional[nn.Module] = None,
    ):
        super().__init__()
        # pool defined at ModelConstructor.
        if div_groups is not None:  # check if groups != 1 and div_groups
            groups = int(out_channels / div_groups)
        layers: ListStrMod = [
            (
                "conv_0",
                conv_layer(
                    in_channels,
                    out_channels,
                    3,
                    stride=stride,
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
        if stride != 1 or in_channels != out_channels:
            id_layers: ListStrMod = []
            if (
                stride != 1 and pool is not None
            ):  # if pool - reduce by pool else stride 2 art id_conv
                id_layers.append(("pool", pool()))
            if in_channels != out_channels or (stride != 1 and pool is None):
                id_layers.append(
                    (
                        "id_conv",
                        conv_layer(
                            in_channels,
                            out_channels,
                            1,
                            stride=1 if pool else stride,
                            act_fn=False,
                        ),
                    )
                )
            self.id_conv = nn_seq(id_layers)
        else:
            self.id_conv = None
        self.act_fn = get_act(act_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        identity = self.id_conv(x) if self.id_conv is not None else x
        return self.act_fn(self.convs(x) + identity)


class BottleneckBlock(nn.Module):
    """Bottleneck Resnet block.
    Configurable - can use pool to reduce at identity path, change act etc."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 4,
        conv_layer: type[ConvBnAct] = ConvBnAct,
        act_fn: type[nn.Module] = nn.ReLU,
        zero_bn: bool = True,
        bn_1st: bool = True,
        groups: int = 1,
        dw: bool = False,
        div_groups: Optional[int] = None,
        pool: Optional[Callable[[], nn.Module]] = None,
        se: Optional[nn.Module] = None,
        sa: Optional[nn.Module] = None,
    ):
        super().__init__()
        # pool defined at ModelConstructor.
        mid_channels = out_channels // expansion
        if div_groups is not None:  # check if groups != 1 and div_groups
            groups = int(mid_channels / div_groups)
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
                    stride=stride,
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
        if stride != 1 or in_channels != out_channels:
            id_layers: ListStrMod = []
            if (
                stride != 1 and pool is not None
            ):  # if pool - reduce by pool else stride 2 art id_conv
                id_layers.append(("pool", pool()))
            if in_channels != out_channels or (stride != 1 and pool is None):
                id_layers.append(
                    (
                        "id_conv",
                        conv_layer(
                            in_channels,
                            out_channels,
                            1,
                            stride=1 if pool else stride,
                            act_fn=False,
                        ),
                    )
                )
            self.id_conv = nn_seq(id_layers)
        else:
            self.id_conv = None
        self.act_fn = get_act(act_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        identity = self.id_conv(x) if self.id_conv is not None else x
        return self.act_fn(self.convs(x) + identity)
