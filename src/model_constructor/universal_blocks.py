from typing import Callable, Optional

import torch
from torch import nn

from .helpers import ModSeq, nn_seq
from .layers import ConvBnAct, get_act
from .model_constructor import ListStrMod, ModelCfg, ModelConstructor

__all__ = [
    "XResBlock",
    "XResNet34",
    "XResNet50",
    "YaResNet",
    "YaResNet34",
    "YaResNet50",
]


class XResBlock(nn.Module):
    """Universal XResnet block. Basic block if expansion is 1, otherwise is Bottleneck."""

    def __init__(
        self,
        expansion: int,
        in_channels: int,
        mid_channels: int,
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
        out_channels, in_channels = mid_channels * expansion, in_channels * expansion
        if div_groups is not None:  # check if groups != 1 and div_groups
            groups = int(mid_channels / div_groups)
        if expansion == 1:
            layers: ListStrMod = [
                (
                    "conv_0",
                    conv_layer(
                        in_channels,
                        mid_channels,
                        3,
                        stride=stride,  # type: ignore
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
                ),  # noqa E501
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

    def forward(self, x: torch.Tensor):  # type: ignore
        identity = self.id_conv(x) if self.id_conv is not None else x
        return self.act_fn(self.convs(x) + identity)


class YaResBlock(nn.Module):
    """YaResBlock. Reduce by pool instead of stride 2.
    Universal model, as XResNet.
    If `expansion=1` - `Basic` block, else - `Bottleneck`"""

    def __init__(
        self,
        expansion: int,
        in_channels: int,
        mid_channels: int,
        stride: int = 1,
        conv_layer: type[ConvBnAct] = ConvBnAct,
        act_fn: type[nn.Module] = nn.ReLU,
        zero_bn: bool = True,
        bn_1st: bool = True,
        groups: int = 1,
        dw: bool = False,
        div_groups: Optional[int] = None,
        pool: Optional[Callable[[], nn.Module]] = None,
        se: Optional[type[nn.Module]] = None,
        sa: Optional[type[nn.Module]] = None,
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
            layers: ListStrMod = [
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
            layers.append(("se", se(out_channels)))  # type: ignore
        if sa:
            layers.append(("sa", sa(out_channels)))  # type: ignore
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

    def forward(self, x: torch.Tensor):  # type: ignore
        if self.reduce:
            x = self.reduce(x)
        identity = self.id_conv(x) if self.id_conv is not None else x
        return self.merge(self.convs(x) + identity)


def make_stem(cfg: ModelCfg) -> nn.Sequential:  # type: ignore
    """Create xResnet stem -> 3 conv 3*3 instead of 1 conv 7*7"""
    len_stem = len(cfg.stem_sizes)
    stem: list[tuple[str, nn.Module]] = [
        (
            f"conv_{i}",
            cfg.conv_layer(
                cfg.stem_sizes[i - 1] if i else cfg.in_chans,  # type: ignore
                cfg.stem_sizes[i],
                stride=2 if i == cfg.stem_stride_on else 1,
                bn_layer=(not cfg.stem_bn_end) if i == (len_stem - 1) else True,
                act_fn=cfg.act_fn,
                bn_1st=cfg.bn_1st,
            ),
        )
        for i in range(len_stem)
    ]
    if cfg.stem_pool:
        stem.append(("stem_pool", cfg.stem_pool()))
    if cfg.stem_bn_end:
        stem.append(("norm", cfg.norm(cfg.stem_sizes[-1])))  # type: ignore
    return nn_seq(stem)


def make_layer(cfg: ModelCfg, layer_num: int) -> nn.Sequential:  # type: ignore
    """Create layer (stage)"""
    # if no pool on stem - stride = 2 for first layer block in body
    stride = 1 if cfg.stem_pool and layer_num == 0 else 2
    num_blocks = cfg.layers[layer_num]
    block_chs = [cfg.stem_sizes[-1] // cfg.expansion] + cfg.block_sizes
    return nn_seq(
        (
            f"bl_{block_num}",
            cfg.block(
                cfg.expansion,  # type: ignore
                block_chs[layer_num] if block_num == 0 else block_chs[layer_num + 1],
                block_chs[layer_num + 1],
                stride if block_num == 0 else 1,
                sa=cfg.sa if (block_num == num_blocks - 1) and layer_num == 0 else None,
                conv_layer=cfg.conv_layer,
                act_fn=cfg.act_fn,
                pool=cfg.pool,
                zero_bn=cfg.zero_bn,
                bn_1st=cfg.bn_1st,
                groups=cfg.groups,
                div_groups=cfg.div_groups,
                dw=cfg.dw,
                se=cfg.se,
            ),
        )
        for block_num in range(num_blocks)
    )


def make_body(cfg: ModelCfg) -> nn.Sequential:  # type: ignore
    """Create model body."""
    return nn_seq(
        (f"l_{layer_num}", cfg.make_layer(cfg, layer_num))  # type: ignore
        for layer_num in range(len(cfg.layers))
    )


def make_head(cfg: ModelCfg) -> nn.Sequential:  # type: ignore
    """Create head."""
    head = [
        ("pool", nn.AdaptiveAvgPool2d(1)),
        ("flat", nn.Flatten()),
        ("fc", nn.Linear(cfg.block_sizes[-1] * cfg.expansion, cfg.num_classes)),
    ]
    return nn_seq(head)


class XResNet(ModelConstructor):
    """Base Xresnet constructor."""

    make_stem: Callable[[ModelCfg], ModSeq] = make_stem
    make_layer: Callable[[ModelCfg, int], ModSeq] = make_layer
    make_body: Callable[[ModelCfg], ModSeq] = make_body
    make_head: Callable[[ModelCfg], ModSeq] = make_head
    block: type[nn.Module] = XResBlock


class XResNet34(XResNet):
    layers: list[int] = [3, 4, 6, 3]


class XResNet50(XResNet34):
    expansion: int = 4


class YaResNet(XResNet):
    """Base Yaresnet constructor.
    YaResBlock, Mish activation, custom stem.
    """

    block: type[nn.Module] = YaResBlock
    stem_sizes: list[int] = [3, 32, 64, 64]
    act_fn: type[nn.Module] = nn.Mish


class YaResNet34(YaResNet):
    layers: list[int] = [3, 4, 6, 3]


class YaResNet50(YaResNet34):
    expansion: int = 4
