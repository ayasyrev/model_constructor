from collections import OrderedDict
from typing import Callable, List, Optional, Type, Union

import torch.nn as nn
from pydantic import BaseModel

from .layers import ConvBnAct, SEModule, SimpleSelfAttention

__all__ = [
    "init_cnn",
    "act_fn",
    "ResBlock",
    "ModelConstructor",
    # "xresnet34",
    # "xresnet50",
]


act_fn = nn.ReLU(inplace=True)


class ResBlock(nn.Module):
    """Universal Resnet block. Basic block if expansion is 1, otherwise is Bottleneck."""

    def __init__(
        self,
        expansion: int,
        in_channels: int,
        mid_channels: int,
        stride: int = 1,
        conv_layer=ConvBnAct,
        act_fn: nn.Module = act_fn,
        zero_bn: bool = True,
        bn_1st: bool = True,
        groups: int = 1,
        dw: bool = False,
        div_groups: Union[None, int] = None,
        pool: Union[nn.Module, None] = None,
        se: Union[nn.Module, None] = None,
        sa: Union[nn.Module, None] = None,
    ):
        super().__init__()
        # pool defined at ModelConstructor.
        out_channels, in_channels = mid_channels * expansion, in_channels * expansion
        if div_groups is not None:  # check if groups != 1 and div_groups
            groups = int(mid_channels / div_groups)
        if expansion == 1:
            layers = [
                ("conv_0", conv_layer(
                    in_channels,
                    mid_channels,
                    3,
                    stride=stride,  # type: ignore
                    act_fn=act_fn,
                    bn_1st=bn_1st,
                    groups=in_channels if dw else groups,
                ),),
                ("conv_1", conv_layer(
                    mid_channels,
                    out_channels,
                    3,
                    zero_bn=zero_bn,
                    act_fn=False,
                    bn_1st=bn_1st,
                    groups=mid_channels if dw else groups,
                ),),
            ]
        else:
            layers = [
                ("conv_0", conv_layer(
                    in_channels,
                    mid_channels,
                    1,
                    act_fn=act_fn,
                    bn_1st=bn_1st,
                ),),
                ("conv_1", conv_layer(
                    mid_channels,
                    mid_channels,
                    3,
                    stride=stride,
                    act_fn=act_fn,
                    bn_1st=bn_1st,
                    groups=mid_channels if dw else groups,
                ),),
                ("conv_2", conv_layer(
                    mid_channels,
                    out_channels,
                    1,
                    zero_bn=zero_bn,
                    act_fn=False,
                    bn_1st=bn_1st,
                ),),  # noqa E501
            ]
        if se:
            layers.append(("se", se(out_channels)))
        if sa:
            layers.append(("sa", sa(out_channels)))
        self.convs = nn.Sequential(OrderedDict(layers))
        if stride != 1 or in_channels != out_channels:
            id_layers = []
            if stride != 1 and pool is not None:  # if pool - reduce by pool else stride 2 art id_conv
                id_layers.append(("pool", pool))
            if in_channels != out_channels or (stride != 1 and pool is None):
                id_layers += [("id_conv", conv_layer(
                    in_channels,
                    out_channels,
                    1,
                    stride=1 if pool else stride,
                    act_fn=False,
                ),)]
            self.id_conv = nn.Sequential(OrderedDict(id_layers))
        else:
            self.id_conv = None
        self.act_fn = act_fn

    def forward(self, x):
        identity = self.id_conv(x) if self.id_conv is not None else x
        return self.act_fn(self.convs(x) + identity)


class CfgMC(BaseModel):
    """Model constructor Config. As default - xresnet18"""

    name: str = "MC"
    in_chans: int = 3
    num_classes: int = 1000
    block: Type[nn.Module] = ResBlock
    conv_layer: Type[nn.Module] = ConvBnAct
    block_sizes: List[int] = [64, 128, 256, 512]
    layers: List[int] = [2, 2, 2, 2]
    norm: Type[nn.Module] = nn.BatchNorm2d
    act_fn: nn.Module = nn.ReLU(inplace=True)
    pool: nn.Module = nn.AvgPool2d(2, ceil_mode=True)
    expansion: int = 1
    groups: int = 1
    dw: bool = False
    div_groups: Union[int, None] = None
    sa: Union[bool, int, Type[nn.Module]] = False
    se: Union[bool, int, Type[nn.Module]] = False
    se_module: Union[bool, None] = None
    se_reduction: Union[int, None] = None
    bn_1st: bool = True
    zero_bn: bool = True
    stem_stride_on: int = 0
    stem_sizes: List[int] = [32, 32, 64]
    stem_pool: Union[nn.Module, None] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # type: ignore
    stem_bn_end: bool = False
    init_cnn: Optional[Callable[[nn.Module], None]] = None
    make_stem: Optional[Callable] = None
    make_layer: Optional[Callable] = None
    make_body: Optional[Callable] = None
    make_head: Optional[Callable] = None

    class Config:
        arbitrary_types_allowed = True


def init_cnn(module: nn.Module):
    "Init module - kaiming_normal for Conv2d and 0 for biases."
    if getattr(module, "bias", None) is not None:
        nn.init.constant_(module.bias, 0)  # type: ignore
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight)
    for layer in module.children():
        init_cnn(layer)


def make_stem(self: CfgMC) -> nn.Sequential:
    stem: List[tuple[str, nn.Module]] = [
        (f"conv_{i}", self.conv_layer(
            self.stem_sizes[i],  # type: ignore
            self.stem_sizes[i + 1],
            stride=2 if i == self.stem_stride_on else 1,
            bn_layer=(not self.stem_bn_end)
            if i == (len(self.stem_sizes) - 2)
            else True,
            act_fn=self.act_fn,
            bn_1st=self.bn_1st,
        ),)
        for i in range(len(self.stem_sizes) - 1)
    ]
    if self.stem_pool:
        stem.append(("stem_pool", self.stem_pool))
    if self.stem_bn_end:
        stem.append(("norm", self.norm(self.stem_sizes[-1])))  # type: ignore
    return nn.Sequential(OrderedDict(stem))


def make_layer(cfg: CfgMC, layer_num: int) -> nn.Sequential:
    #  expansion, in_channels, out_channels, blocks, stride, sa):
    # if no pool on stem - stride = 2 for first layer block in body
    stride = 1 if cfg.stem_pool and layer_num == 0 else 2
    num_blocks = cfg.layers[layer_num]
    block_chs = [cfg.stem_sizes[-1] // cfg.expansion] + cfg.block_sizes
    return nn.Sequential(
        OrderedDict(
            [
                (
                    f"bl_{block_num}",
                    cfg.block(
                        cfg.expansion,  # type: ignore
                        block_chs[layer_num] if block_num == 0 else block_chs[layer_num + 1],
                        block_chs[layer_num + 1],
                        stride if block_num == 0 else 1,
                        sa=cfg.sa
                        if (block_num == num_blocks - 1) and layer_num == 0
                        else None,
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
            ]
        )
    )


def make_body(cfg: CfgMC) -> nn.Sequential:
    return nn.Sequential(
        OrderedDict(
            [
                (
                    f"l_{layer_num}",
                    cfg.make_layer(cfg, layer_num)  # type: ignore
                )
                for layer_num in range(len(cfg.layers))
            ]
        )
    )


def make_head(cfg: CfgMC) -> nn.Sequential:
    head = [
        ("pool", nn.AdaptiveAvgPool2d(1)),
        ("flat", nn.Flatten()),
        ("fc", nn.Linear(cfg.block_sizes[-1] * cfg.expansion, cfg.num_classes)),
    ]
    return nn.Sequential(OrderedDict(head))


class ModelConstructor(CfgMC):
    """Model constructor. As default - xresnet18"""

    def __init__(self, **data):
        super().__init__(**data)
        if self.init_cnn is None:
            self.init_cnn = init_cnn
        if self.make_stem is None:
            self.make_stem = make_stem
        if self.make_layer is None:
            self.make_layer = make_layer
        if self.make_body is None:
            self.make_body = make_body
        if self.make_head is None:
            self.make_head = make_head

        if self.stem_sizes[0] != self.in_chans:
            self.stem_sizes = [self.in_chans] + self.stem_sizes
        if self.se and isinstance(self.se, (bool, int)):  # if se=1 or se=True
            self.se = SEModule
        if self.sa and isinstance(self.sa, (bool, int)):  # if sa=1 or sa=True
            self.sa = SimpleSelfAttention  # default: ks=1, sym=sym
        if self.se_module or self.se_reduction:  # pragma: no cover
            print(
                "Deprecated. Pass se_module as se argument, se_reduction as arg to se."
            )  # add deprecation warning.

    @property
    def stem(self):
        return self.make_stem(self)  # type: ignore

    @property
    def head(self):
        return self.make_head(self)  # type: ignore

    @property
    def body(self):
        return self.make_body(self)  # type: ignore

    @classmethod
    def from_cfg(cls, cfg: CfgMC):
        return cls(**cfg.dict())

    def __call__(self):
        model = nn.Sequential(
            OrderedDict([("stem", self.stem), ("body", self.body), ("head", self.head)])
        )
        self.init_cnn(model)  # type: ignore
        model.extra_repr = lambda: f"{self.name}"
        return model

    def __repr__(self):
        return (
            f"{self.name} constructor\n"
            f"  in_chans: {self.in_chans}, num_classes: {self.num_classes}\n"
            f"  expansion: {self.expansion}, groups: {self.groups}, dw: {self.dw}, div_groups: {self.div_groups}\n"
            f"  sa: {self.sa}, se: {self.se}\n"
            f"  stem sizes: {self.stem_sizes}, stride on {self.stem_stride_on}\n"
            f"  body sizes {self.block_sizes}\n"
            f"  layers: {self.layers}"
        )


class XResNet34(ModelConstructor):
    name: str = "xresnet34"
    layers: list[int] = [3, 4, 6, 3]


class XResNet50(XResNet34):
    name: str = "xresnet50"
    expansion: int = 4
