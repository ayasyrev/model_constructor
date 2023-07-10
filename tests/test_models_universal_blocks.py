from typing import Type
import pytest
import torch
from torch import nn

from model_constructor.universal_blocks import (
    ModelConstructor,
    XResNet,
    XResNet34,
    XResNet50,
    YaResNet,
    YaResNet34,
    YaResNet50,
)

bs_test = 2
img_size = 16
xb = torch.rand(bs_test, 3, img_size, img_size)

mc_list = [
    XResNet,
    XResNet34,
    XResNet50,
    YaResNet,
    YaResNet34,
    YaResNet50,
]
act_fn_list = [nn.ReLU, nn.Mish, nn.GELU]


@pytest.mark.parametrize("model_constructor", mc_list)
@pytest.mark.parametrize("act_fn", act_fn_list)
def test_mc(model_constructor: Type[ModelConstructor], act_fn: Type[nn.Module]):
    """test models"""
    mc = model_constructor(act_fn=act_fn)
    model = mc()
    pred = model(xb)
    assert pred.shape == torch.Size([bs_test, 1000])


def test_stem_bn_end():
    """test stem"""
    mc = XResNet()
    assert not mc.stem_bn_end
    mc.stem_bn_end = True
    stem = mc.stem
    assert isinstance(stem[-1], nn.BatchNorm2d)
    stem_out = stem(xb)
    assert stem_out.shape == torch.Size([bs_test, 64, 4, 4])
