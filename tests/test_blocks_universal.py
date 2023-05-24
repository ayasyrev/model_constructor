# import pytest
from functools import partial

import torch
from torch import nn

from model_constructor.layers import SEModule, SimpleSelfAttention
from model_constructor.universal_blocks import XResBlock, YaResBlock

from .parameters import ids_fn

bs_test = 4
img_size = 16


params = dict(
    Block=[XResBlock, YaResBlock],
    expansion=[1, 2],
    mid_channels=[8, 16],
    stride=[1, 2],
    div_groups=[None, 2],
    pool=[None, partial(nn.AvgPool2d, kernel_size=2, ceil_mode=True)],
    se=[None, SEModule],
    sa=[None, SimpleSelfAttention],
)


def pytest_generate_tests(metafunc):
    for key, value in params.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key, value, ids=ids_fn(key, value))


def test_block(Block, expansion, mid_channels, stride, div_groups, pool, se, sa):
    """test block"""
    in_channels = 8
    out_channels = mid_channels * expansion
    block = Block(
        expansion,
        in_channels,
        mid_channels,
        stride,
        div_groups=div_groups,
        pool=pool,
        se=se,
        sa=sa,
    )
    xb = torch.randn(bs_test, in_channels * expansion, img_size, img_size)
    out = block(xb)
    out_size = img_size if stride == 1 else img_size // stride
    assert out.shape == torch.Size([bs_test, out_channels, out_size, out_size])


def test_block_dw(Block, expansion, mid_channels, stride):
    """test block, dw=1"""
    in_channels = 8
    out_channels = mid_channels * expansion
    block = Block(
        expansion,
        in_channels,
        mid_channels,
        stride,
        dw=1,
    )
    xb = torch.randn(bs_test, in_channels * expansion, img_size, img_size)
    out = block(xb)
    out_size = img_size if stride == 1 else img_size // stride
    assert out.shape == torch.Size([bs_test, out_channels, out_size, out_size])
