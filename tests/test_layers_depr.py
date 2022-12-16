# old (deprecated layers)
import torch
import torch.nn as nn

from model_constructor.layers import ConvLayer, SEBlock, SEBlockConv


bs_test = 4


params = dict(
    # SE
    se_module=[SEBlock, SEBlockConv],
    reduction=[16, 2],
    rd_channels=[None, 2],
    rd_max=[False, True],
    use_bias=[True, False],
    # ConvLayer
    nf=[8, 16],
    ks=[3, 1],
    stride=[1, 2],
    act=[True, False],
    bn_layer=[True, False],
    bn_1st=[True, False],
    zero_bn=[False, True],
    bias=[False, True],
    groups=[1, 2],
)


def value_name(value) -> str:  # pragma: no cover
    name = getattr(value, "__name__", None)
    if name is not None:
        return name
    if isinstance(value, nn.Module):
        return value._get_name()  # pylint: disable=W0212
    return value


def ids_fn(key, value):
    return [f"{key[:2]}_{value_name(v)}" for v in value]


def pytest_generate_tests(metafunc):
    for key, value in params.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key, value, ids=ids_fn(key, value))


def test_SE(se_module, reduction, use_bias):
    """test SE"""
    in_channels = 8
    channel_size = 4
    se = se_module(in_channels, reduction)
    se.use_bias = use_bias
    xb = torch.randn(bs_test, in_channels, channel_size, channel_size)
    out = se(xb)
    assert out.shape == torch.Size([bs_test, in_channels, channel_size, channel_size])


def test_ConvLayer(nf, ks, stride, bn_layer, bn_1st, zero_bn, bias, groups):
    """test ConvLayer"""
    ni = 8
    channel_size = 4
    block = ConvLayer(
        ni,
        nf,
        ks,
        stride,
        bn_layer=bn_layer,
        bn_1st=bn_1st,
        zero_bn=zero_bn,
        bias=bias,
        groups=groups,
    )
    xb = torch.randn(bs_test, ni, channel_size, channel_size)
    out = block(xb)
    # out_ch = nf
    out_size = channel_size
    if stride == 2:
        out_size = channel_size // stride
    assert out.shape == torch.Size([bs_test, nf, out_size, out_size])
