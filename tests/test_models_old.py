import pytest
import torch
import torch.nn as nn

from model_constructor.xresnet import xresnet18, xresnet34, xresnet50
from model_constructor.mxresnet import mxresnet34, mxresnet50

bs_test = 2
img_size = 16
xb = torch.rand(bs_test, 3, img_size, img_size)

num_classes = 10

models_list = [xresnet18, xresnet34, xresnet50]
act_fn_list = [nn.ReLU, nn.Mish, nn.GELU]
mx_list = [mxresnet34, mxresnet50]


@pytest.mark.parametrize("model", models_list)
def test_model(model):
    """test models"""
    mod = model(num_classes=num_classes)
    pred = mod(xb)
    assert pred.shape == torch.Size([bs_test, num_classes])


@pytest.mark.parametrize("model_constructor", mx_list)
@pytest.mark.parametrize("act_fn", act_fn_list)
def test_model_mx(model_constructor, act_fn):
    """test models"""
    mc = model_constructor(c_out=num_classes)
    assert mc.c_out == num_classes
    mc.act_fn = act_fn()
    model = mc()
    pred = model(xb)
    assert pred.shape == torch.Size([bs_test, num_classes])
