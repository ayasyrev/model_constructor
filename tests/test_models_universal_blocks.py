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
def test_mc(model_constructor: type[ModelConstructor], act_fn: type[nn.Module]):
    """test models"""
    mc = model_constructor(act_fn=act_fn)
    model = mc()
    pred = model(xb)
    assert pred.shape == torch.Size([bs_test, 1000])
