from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Optional, TypeVar, Union

import torch
from pydantic import BaseModel, field_validator
from torch import nn

from .helpers import nn_seq
from .layers import ConvBnAct, SEModule, SimpleSelfAttention, get_act

__all__ = [
    "init_cnn",
    # "ResBlock",
    "ModelConstructor",
    "ResNet34",
    "ResNet50",
]


TModelCfg = TypeVar("TModelCfg", bound="ModelCfg")

ListStrMod = list[tuple[str, nn.Module]]


def init_cnn(module: nn.Module) -> None:
    "Init module - kaiming_normal for Conv2d and 0 for biases."
    if getattr(module, "bias", None) is not None:
        nn.init.constant_(module.bias, 0)  # type: ignore
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight)
    for layer in module.children():
        init_cnn(layer)


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
        div_groups: Union[None, int] = None,
        pool: Union[Callable[[], nn.Module], None] = None,
        se: Union[nn.Module, None] = None,
        sa: Union[nn.Module, None] = None,
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
        div_groups: Union[None, int] = None,
        pool: Union[Callable[[], nn.Module], None] = None,
        se: Union[nn.Module, None] = None,
        sa: Union[nn.Module, None] = None,
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


def make_stem(cfg: TModelCfg) -> nn.Sequential:  # type: ignore
    """Create Resnet stem."""
    stem: ListStrMod = [
        (
            "conv_1",
            cfg.conv_layer(
                cfg.in_chans,  # type: ignore
                cfg.stem_sizes[-1],
                kernel_size=7,
                stride=2,
                padding=3,
                bn_layer=not cfg.stem_bn_end,
                act_fn=cfg.act_fn,
                bn_1st=cfg.bn_1st,
            ),
        )
    ]
    if cfg.stem_pool:
        stem.append(("stem_pool", cfg.stem_pool()))
    if cfg.stem_bn_end:
        stem.append(("norm", cfg.norm(cfg.stem_sizes[-1])))  # type: ignore
    return nn_seq(stem)


def make_layer(cfg: TModelCfg, layer_num: int) -> nn.Sequential:  # type: ignore
    """Create layer (stage)"""
    # if no pool on stem - stride = 2 for first layer block in body
    stride = 1 if cfg.stem_pool and layer_num == 0 else 2
    num_blocks = cfg.layers[layer_num]
    block_chs = [cfg.stem_sizes[-1]] + cfg.block_sizes
    return nn_seq(
        (
            f"bl_{block_num}",
            cfg.block(
                block_chs[layer_num]  # type: ignore
                if block_num == 0
                else block_chs[layer_num + 1],
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


def make_body(cfg: TModelCfg) -> nn.Sequential:  # type: ignore
    """Create model body."""
    return nn_seq(
        (f"l_{layer_num}", cfg.make_layer(cfg, layer_num))  # type: ignore
        for layer_num in range(len(cfg.layers))
    )


def make_head(cfg: TModelCfg) -> nn.Sequential:  # type: ignore
    """Create head."""
    head = [
        ("pool", nn.AdaptiveAvgPool2d(1)),
        ("flat", nn.Flatten()),
        ("fc", nn.Linear(cfg.block_sizes[-1], cfg.num_classes)),
    ]
    return nn_seq(head)


class ModelCfg(BaseModel, arbitrary_types_allowed=True, extra="forbid"):
    """Model constructor Config. As default - xresnet18"""

    name: Optional[str] = None
    in_chans: int = 3
    num_classes: int = 1000
    block: type[nn.Module] = BasicBlock
    conv_layer: type[nn.Module] = ConvBnAct
    block_sizes: list[int] = [64, 128, 256, 512]
    layers: list[int] = [2, 2, 2, 2]
    norm: type[nn.Module] = nn.BatchNorm2d
    act_fn: type[nn.Module] = nn.ReLU
    pool: Optional[Callable[[Any], nn.Module]] = None
    expansion: int = 1
    groups: int = 1
    dw: bool = False
    div_groups: Union[int, None] = None
    sa: Union[bool, type[nn.Module]] = False
    se: Union[bool, type[nn.Module]] = False
    se_module: Union[bool, None] = None
    se_reduction: Union[int, None] = None
    bn_1st: bool = True
    zero_bn: bool = True
    stem_stride_on: int = 0
    stem_sizes: list[int] = [64]
    stem_pool: Union[Callable[[], nn.Module], None] = partial(
        nn.MaxPool2d, kernel_size=3, stride=2, padding=1
    )
    stem_bn_end: bool = False
    init_cnn: Callable[[nn.Module], None] = init_cnn
    make_stem: Callable[[TModelCfg], Union[nn.Module, nn.Sequential]] = make_stem  # type: ignore
    make_layer: Callable[[TModelCfg, int], Union[nn.Module, nn.Sequential]] = make_layer  # type: ignore
    make_body: Callable[[TModelCfg], Union[nn.Module, nn.Sequential]] = make_body  # type: ignore
    make_head: Callable[[TModelCfg], Union[nn.Module, nn.Sequential]] = make_head  # type: ignore

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

    def __repr_args__(self) -> list[tuple[str, str]]:
        return [
            (field, str_value)
            for field in self.model_fields
            if (str_value := self._get_str_value(field))
        ]

    def __repr_changed_args__(self) -> list[str]:
        """Return list repr for changed fields"""
        return [
            f"{field}: {self._get_str_value(field)}"
            for field in self.model_fields
            if field != "name"
        ]

    def print_cfg(self) -> None:
        """Print full config"""
        print(f"{self.__repr_name__()}(\n  {self.__repr_str__(chr(10) + '  ')})")

    def print_changed(self) -> None:
        """Print changed fields."""
        changed_fields = self.__repr_changed_args__()
        if changed_fields:
            print("Changed fields:")
            for i in changed_fields:
                print("  ", i)
        else:
            print("Nothing changed")


class ModelConstructor(ModelCfg):
    """Model constructor. As default - resnet18"""

    @field_validator("se")
    def set_se(  # pylint: disable=no-self-argument
        cls, value: Union[bool, type[nn.Module]]
    ) -> Union[bool, type[nn.Module]]:
        if value:
            if isinstance(value, (int, bool)):
                return SEModule
        return value

    @field_validator("sa")
    def set_sa(  # pylint: disable=no-self-argument
        cls, value: Union[bool, type[nn.Module]]
    ) -> Union[bool, type[nn.Module]]:
        if value:
            if isinstance(value, (int, bool)):
                return SimpleSelfAttention  # default: ks=1, sym=sym
        return value

    @field_validator("se_module", "se_reduction")  # pragma: no cover
    def deprecation_warning(  # pylint: disable=no-self-argument
        cls, value: Union[bool, int, None]
    ) -> Union[bool, int, None]:
        print("Deprecated. Pass se_module as se argument, se_reduction as arg to se.")
        return value

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
        return cls(**cfg.model_dump())

    @classmethod
    def create_model(
        cls, cfg: Union[ModelCfg, None] = None, **kwargs: dict[str, Any]
    ) -> nn.Sequential:
        if cfg:
            return cls(**cfg.model_dump())()
        return cls(**kwargs)()

    def __call__(self) -> nn.Sequential:
        """Create model."""
        model_name = self.name or self.__class__.__name__
        named_sequential = type(
            model_name, (nn.Sequential,), {}
        )  # create type named as model
        model = named_sequential(
            OrderedDict([("stem", self.stem), ("body", self.body), ("head", self.head)])  # type: ignore
        )
        self.init_cnn(model)  # pylint: disable=too-many-function-args
        extra_repr = self.__repr_changed_args__()
        if extra_repr:
            model.extra_repr = lambda: ", ".join(extra_repr)
        return model

    def __repr__(self) -> str:
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


class ResNet34(ModelConstructor):
    layers: list[int] = [3, 4, 6, 3]


class ResNet50(ResNet34):
    block: type[nn.Module] = BottleneckBlock
    block_sizes: list[int] = [256, 512, 1024, 2048]
