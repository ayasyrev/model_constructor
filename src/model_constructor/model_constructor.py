from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, Optional, Type, TypeVar, Union

import torch.nn as nn
from pydantic import BaseModel, root_validator

from .layers import ConvBnAct, SEModule, SimpleSelfAttention, get_act

__all__ = [
    "init_cnn",
    "ResBlock",
    "ModelConstructor",
    "XResNet34",
    "XResNet50",
]


TModelCfg = TypeVar("TModelCfg", bound="ModelCfg")


def init_cnn(module: nn.Module):
    "Init module - kaiming_normal for Conv2d and 0 for biases."
    if getattr(module, "bias", None) is not None:
        nn.init.constant_(module.bias, 0)  # type: ignore
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight)
    for layer in module.children():
        init_cnn(layer)


class ResBlock(nn.Module):
    """Universal Resnet block. Basic block if expansion is 1, otherwise is Bottleneck."""

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
        self.convs = nn.Sequential(OrderedDict(layers))
        if stride != 1 or in_channels != out_channels:
            id_layers = []
            if (
                stride != 1 and pool is not None
            ):  # if pool - reduce by pool else stride 2 art id_conv
                id_layers.append(("pool", pool()))
            if in_channels != out_channels or (stride != 1 and pool is None):
                id_layers += [
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
                ]
            self.id_conv = nn.Sequential(OrderedDict(id_layers))
        else:
            self.id_conv = None
        self.act_fn = get_act(act_fn)  # type: ignore

    def forward(self, x):
        identity = self.id_conv(x) if self.id_conv is not None else x
        return self.act_fn(self.convs(x) + identity)


def make_stem(cfg: TModelCfg) -> nn.Sequential:  # type: ignore
    len_stem = len(cfg.stem_sizes)
    stem: List[tuple[str, nn.Module]] = [
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
    return nn.Sequential(OrderedDict(stem))


def make_layer(cfg: TModelCfg, layer_num: int) -> nn.Sequential:  # type: ignore
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
                        block_chs[layer_num]
                        if block_num == 0
                        else block_chs[layer_num + 1],
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


def make_body(cfg: TModelCfg) -> nn.Sequential:  # type: ignore
    return nn.Sequential(
        OrderedDict(
            [
                (f"l_{layer_num}", cfg.make_layer(cfg, layer_num))  # type: ignore
                for layer_num in range(len(cfg.layers))
            ]
        )
    )


def make_head(cfg: TModelCfg) -> nn.Sequential:  # type: ignore
    head = [
        ("pool", nn.AdaptiveAvgPool2d(1)),
        ("flat", nn.Flatten()),
        ("fc", nn.Linear(cfg.block_sizes[-1] * cfg.expansion, cfg.num_classes)),
    ]
    return nn.Sequential(OrderedDict(head))


class ModelCfg(BaseModel):
    """Model constructor Config. As default - xresnet18"""

    name: Optional[str] = None
    in_chans: int = 3
    num_classes: int = 1000
    block: Type[nn.Module] = ResBlock
    conv_layer: Type[nn.Module] = ConvBnAct
    block_sizes: List[int] = [64, 128, 256, 512]
    layers: List[int] = [2, 2, 2, 2]
    norm: Type[nn.Module] = nn.BatchNorm2d
    act_fn: Type[nn.Module] = nn.ReLU
    pool: Callable[[Any], nn.Module] = partial(
        nn.AvgPool2d, kernel_size=2, ceil_mode=True
    )
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
    stem_pool: Union[Callable[[], nn.Module], None] = partial(
        nn.MaxPool2d, kernel_size=3, stride=2, padding=1
    )
    stem_bn_end: bool = False
    init_cnn: Callable[[nn.Module], None] = init_cnn
    make_stem: Callable[[TModelCfg], Union[nn.Module, nn.Sequential]] = make_stem  # type: ignore
    make_layer: Callable[[TModelCfg, int], Union[nn.Module, nn.Sequential]] = make_layer  # type: ignore
    make_body: Callable[[TModelCfg], Union[nn.Module, nn.Sequential]] = make_body  # type: ignore
    make_head: Callable[[TModelCfg], Union[nn.Module, nn.Sequential]] = make_head  # type: ignore

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    def _get_str_value(self, field: str) -> str:
        value = getattr(self, field)
        if isinstance(value, type):
            value = value.__name__
        elif isinstance(value, partial):
            value = f"{value.func.__name__} {value.keywords}"
        elif callable(value):
            value = value.__name__
        return value

    def __repr__(self) -> str:
        return f"{self.__repr_name__()}(\n  {self.__repr_str__(chr(10) + '  ')})"

    def __repr_args__(self):
        return [
            (field, str_value)
            for field in self.__fields__
            if (str_value := self._get_str_value(field))
        ]


class ModelConstructor(ModelCfg):
    """Model constructor. As default - xresnet18"""

    @root_validator
    def post_init(cls, values):  # pylint: disable=E0213
        if values["se"] and isinstance(values["se"], (bool, int)):  # if se=1 or se=True
            values["se"] = SEModule
        if values["sa"] and isinstance(values["sa"], (bool, int)):  # if sa=1 or sa=True
            values["sa"] = SimpleSelfAttention  # default: ks=1, sym=sym
        if values["se_module"] or values["se_reduction"]:  # pragma: no cover
            print(
                "Deprecated. Pass se_module as se argument, se_reduction as arg to se."
            )  # add deprecation warning.
        return values

    @property
    def stem(self):
        return self.make_stem(self)  # pylint: disable=too-many-function-args

    @property
    def head(self):
        return self.make_head(self)  # pylint: disable=too-many-function-args

    @property
    def body(self):
        return self.make_body(self)  # pylint: disable=too-many-function-args

    @classmethod
    def from_cfg(cls, cfg: ModelCfg):
        return cls(**cfg.dict())

    def __call__(self):
        model_name = self.name or self.__class__.__name__
        named_sequential = type(model_name, (nn.Sequential,), {})
        model = named_sequential(
            OrderedDict([("stem", self.stem), ("body", self.body), ("head", self.head)])
        )
        self.init_cnn(model)  # pylint: disable=too-many-function-args
        extra_repr = self._get_extra_repr()
        if extra_repr:
            model.extra_repr = lambda: extra_repr
        return model

    def _get_extra_repr(self) -> str:
        return " ".join(
            f"{field}: {self._get_str_value(field)},"
            for field in self.__fields_set__
            if field != "name"
        )[:-1]

    def __repr__(self):
        se_repr = self.se.__name__ if self.se else "False"  # type: ignore
        model_name = self.name or self.__class__.__name__
        return (
            f"{model_name}\n"
            f"  in_chans: {self.in_chans}, num_classes: {self.num_classes}\n"
            f"  expansion: {self.expansion}, groups: {self.groups}, dw: {self.dw}, div_groups: {self.div_groups}\n"
            f"  act_fn: {self.act_fn.__name__}, sa: {self.sa}, se: {se_repr}\n"
            f"  stem sizes: {self.stem_sizes}, stride on {self.stem_stride_on}\n"
            f"  body sizes {self.block_sizes}\n"
            f"  layers: {self.layers}"
        )

    def print_cfg(self) -> None:
        """Print full config"""
        print(f"{self.__repr_name__()}(\n  {self.__repr_str__(chr(10) + '  ')})")


class XResNet34(ModelConstructor):
    layers: list[int] = [3, 4, 6, 3]


class XResNet50(XResNet34):
    expansion: int = 4
