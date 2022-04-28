import torch
import torch.nn as nn

from model_constructor.layers import ConvBnAct, Flatten, Noop, SEModule, SEModuleConv, SimpleSelfAttention, noop


bs_test = 4


params = dict(
    kernel_size=[3, 1],
    stride=[1, 2],
    padding=[None, 1],
    bias=[False, True],
    groups=[1, 2],
    # # act_fn=act_fn,
    pre_act=[False, True],
    bn_layer=[True, False],
    bn_1st=[True, False],
    zero_bn=[False, True],
    # SA
    sym=[False, True],
    # SE
    se_module=[SEModule, SEModuleConv],
    reduction=[16, 2],
    rd_channels=[None, 2],
    rd_max=[False, True],
    use_bias=[True, False],
)


def value_name(value) -> str:
    name = getattr(value, "__name__", None)
    if name is not None:
        return name
    if isinstance(value, nn.Module):
        return value._get_name()
    else:
        return value


def ids_fn(key, value):
    return [f"{key[:2]}_{value_name(v)}" for v in value]


def pytest_generate_tests(metafunc):
    for key, value in params.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key, value, ids=ids_fn(key, value))


def test_Flatten():
    """test Flatten"""
    flatten = Flatten()
    channels = 4
    xb = torch.randn(bs_test, channels, channels)
    out = flatten(xb)
    assert out.shape == torch.Size([bs_test, channels * channels])


def test_noop():
    """test Noop, noop"""
    xb = torch.randn(bs_test)
    xb_copy = xb.clone().detach()
    out = noop(xb)
    assert out is xb
    assert all(out.eq(xb_copy))
    noop_module = Noop()
    out = noop_module(xb)
    assert out is xb
    assert all(out.eq(xb_copy))


def test_ConvBnAct(kernel_size, stride, bias, groups, pre_act, bn_layer, bn_1st, zero_bn):
    """test ConvBnAct"""
    in_channels = out_channels = 4
    channel_size = 4
    block = ConvBnAct(
        in_channels, out_channels, kernel_size, stride,
        padding=None, bias=bias, groups=groups,
        pre_act=pre_act, bn_layer=bn_layer, bn_1st=bn_1st, zero_bn=zero_bn)
    xb = torch.randn(bs_test, in_channels, channel_size, channel_size)
    out = block(xb)
    out_size = channel_size
    if stride == 2:
        out_size = channel_size // stride
    assert out.shape == torch.Size([bs_test, out_channels, out_size, out_size])


def test_SimpleSelfAttention(sym):
    """test SimpleSelfAttention"""
    in_channels = 4
    kernel_size = 1  # ? can be 3? if so check sym hack.
    channel_size = 4
    sa = SimpleSelfAttention(in_channels, kernel_size, sym)
    xb = torch.randn(bs_test, in_channels, channel_size, channel_size)
    out = sa(xb)
    assert out.shape == torch.Size([bs_test, in_channels, channel_size, channel_size])


def test_SE(se_module, reduction, rd_channels, rd_max, use_bias):
    """test SE"""
    in_channels = 8
    channel_size = 4
    se = se_module(in_channels, reduction, rd_channels, rd_max, use_bias=use_bias)
    xb = torch.randn(bs_test, in_channels, channel_size, channel_size)
    out = se(xb)
    assert out.shape == torch.Size([bs_test, in_channels, channel_size, channel_size])
