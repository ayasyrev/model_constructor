from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Optional, Union

from pydantic import field_validator
from torch import nn

from .blocks import BasicBlock, BottleneckBlock
from .helpers import Cfg, ListStrMod, ModSeq, init_cnn, nn_seq
from .layers import ConvBnAct, SEModule, SimpleSelfAttention

__all__ = [
    "init_cnn",
    "ModelConstructor",
    "ResNet34",
    "ResNet50",
]


class ModelCfg(Cfg, arbitrary_types_allowed=True, extra="forbid"):
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
    div_groups: Optional[int] = None
    sa: Union[bool, type[nn.Module]] = False
    se: Union[bool, type[nn.Module]] = False
    se_module: Optional[bool] = None
    se_reduction: Optional[int] = None
    bn_1st: bool = True
    zero_bn: bool = True
    stem_stride_on: int = 0
    stem_sizes: list[int] = [64]
    stem_pool: Union[Callable[[], nn.Module], None] = partial(
        nn.MaxPool2d, kernel_size=3, stride=2, padding=1
    )
    stem_bn_end: bool = False

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


def make_stem(cfg: ModelCfg) -> nn.Sequential:  # type: ignore
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


def make_layer(cfg: ModelCfg, layer_num: int) -> nn.Sequential:  # type: ignore
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
        ("fc", nn.Linear(cfg.block_sizes[-1], cfg.num_classes)),
    ]
    return nn_seq(head)


class ModelConstructor(ModelCfg):
    """Model constructor. As default - resnet18"""

    init_cnn: Callable[[nn.Module], None] = init_cnn
    make_stem: Callable[[ModelCfg], ModSeq] = make_stem  # type: ignore
    make_layer: Callable[[ModelCfg, int], ModSeq] = make_layer  # type: ignore
    make_body: Callable[[ModelCfg], ModSeq] = make_body  # type: ignore
    make_head: Callable[[ModelCfg], ModSeq] = make_head  # type: ignore

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
        cls, cfg: Optional[ModelCfg] = None, **kwargs: dict[str, Any]
    ) -> nn.Sequential:
        if cfg:
            return cls(**cfg.model_dump())()
        return cls(**kwargs)()  # type: ignore

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


class ResNet34(ModelConstructor):
    layers: list[int] = [3, 4, 6, 3]


class ResNet50(ResNet34):
    block: type[nn.Module] = BottleneckBlock
    block_sizes: list[int] = [256, 512, 1024, 2048]
