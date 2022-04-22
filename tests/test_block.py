# import pytest
import torch
import torch.nn as nn
from model_constructor.layers import SEModule, SimpleSelfAttention

from model_constructor.model_constructor import ResBlock
from model_constructor.yaresnet import YaResBlock

bs_test = 4
img_size = 16


params = {
    "Block": [ResBlock, YaResBlock],
    "expansion": [1, 2],
    "mid_channels": [8, 16],
    "stride": [1, 2],
    "pool": [None, nn.AvgPool2d(2, ceil_mode=True)],
    "se": [None, SEModule],
    "sa": [None, SimpleSelfAttention],
}


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


def test_block(Block, expansion, mid_channels, stride, pool, se, sa):
    """test block"""
    in_channels = 8
    out_channels = mid_channels * expansion
    block = Block(expansion, in_channels, mid_channels, stride, pool=pool, se=se, sa=sa)
    xb = torch.randn(bs_test, in_channels * expansion, img_size, img_size)
    y = block(xb)
    out_size = img_size if stride == 1 else img_size // stride
    assert y.shape == torch.Size([bs_test, out_channels, out_size, out_size])
