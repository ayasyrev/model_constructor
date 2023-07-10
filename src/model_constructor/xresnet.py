from functools import partial
from typing import Any, Callable, List, Optional, Type

from torch import nn

from .blocks import BottleneckBlock
from .helpers import ListStrMod, nn_seq, ModSeq
from .model_constructor import ModelCfg, ModelConstructor

__all__ = [
    "XResNet",
    "XResNet34",
    "XResNet50",
]


def xresnet_stem(cfg: ModelCfg) -> nn.Sequential:
    """Create xResnet stem -> 3 conv 3*3 instead 1 conv 7*7"""
    len_stem = len(cfg.stem_sizes)
    stem: ListStrMod = [
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


class XResNet(ModelConstructor):
    make_stem: Callable[[ModelCfg], ModSeq] = xresnet_stem
    stem_sizes: List[int] = [32, 32, 64]
    pool: Optional[Callable[[Any], nn.Module]] = partial(
        nn.AvgPool2d, kernel_size=2, ceil_mode=True
    )


class XResNet34(XResNet):
    layers: List[int] = [3, 4, 6, 3]


class XResNet50(XResNet34):
    block: Type[nn.Module] = BottleneckBlock
    block_sizes: List[int] = [256, 512, 1024, 2048]
