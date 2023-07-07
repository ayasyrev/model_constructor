import pytest
import torch
from torch import nn

from model_constructor.model_constructor import ModelConstructor, ResNet34, ResNet50
from model_constructor.mxresnet import MxResNet, MxResNet34, MxResNet50
from model_constructor.xresnet import XResNet, XResNet34, XResNet50
from model_constructor.yaresnet import YaResNet, YaResNet34, YaResNet50

bs_test = 2
img_size = 16
xb = torch.rand(bs_test, 3, img_size, img_size)

mc_list = [
    ModelConstructor,
    ResNet34,
    ResNet50,
    XResNet,
    XResNet34,
    XResNet50,
    YaResNet,
    YaResNet34,
    YaResNet50,
    MxResNet,
    MxResNet34,
    MxResNet50,
]
act_fn_list = [
    nn.ReLU,
    nn.Mish,
    nn.GELU,
]


@pytest.mark.parametrize("model_constructor", mc_list)
@pytest.mark.parametrize("act_fn", act_fn_list)
def test_mc(model_constructor: type[ModelConstructor], act_fn: type[nn.Module]):
    """test models"""
    mc = model_constructor(act_fn=act_fn)
    model = mc()
    pred = model(xb)
    assert pred.shape == torch.Size([bs_test, 1000])


def test_xresnet_stem():
    """test xresnet stem"""
    mc = XResNet()
    assert mc.stem_bn_end == False
    mc.stem_bn_end = True
    stem = mc.stem
    assert isinstance(stem[-1], nn.BatchNorm2d)
    stem_out = stem(xb)
    assert stem_out.shape == torch.Size([bs_test, 64, 4, 4])
