import torch
import torch.nn as nn

from model_constructor.net import Net, NewResBlock, ResBlock
# from model_constructor.layers import SEModule, SimpleSelfAttention


bs_test = 4


params = dict(
    block=[ResBlock, NewResBlock],
    expansion=[1, 2],
    groups=[1, 2],
    dw=[0, 1],
    div_groups=[None, 2],
    sa=[0, 1],
    se=[0, 1],
    bn_1st=[True, False],
    zero_bn=[True, False],
    stem_bn_end=[True, False],
    stem_stride_on=[0, 1]
)


def value_name(value) -> str:  # pragma: no cover
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


def test_Net(
    block, expansion,
    groups,
    # dw, div_groups,
):
    """test Net"""
    c_in = 3
    img_size = 16
    c_out = 8
    name = "Test name"

    mc = Net(
        name, c_in, c_out, block,
        expansion=expansion,
        stem_sizes=[8, 16],
        block_sizes=[16, 32, 64, 128],
        groups=groups,
        # dw=dw,
        # div_groups=div_groups,
        # bn_1st=bn_1st, zero_bn=zero_bn,
        # stem_bn_end=stem_bn_end,
    )
    assert f"{name} constructor" in str(mc)
    model = mc()
    xb = torch.randn(bs_test, c_in, img_size, img_size)
    pred = model(xb)
    assert pred.shape == torch.Size([bs_test, c_out])


def test_Net_div_gr(
    block, expansion,
    div_groups,
):
    """test Net"""
    c_in = 3
    img_size = 16
    c_out = 8
    name = "Test name"

    mc = Net(
        name, c_in, c_out, block,
        expansion=expansion,
        stem_sizes=[8, 16],
        block_sizes=[16, 32, 64, 128],
        div_groups=div_groups,
    )
    assert f"{name} constructor" in str(mc)
    model = mc()
    xb = torch.randn(bs_test, c_in, img_size, img_size)
    pred = model(xb)
    assert pred.shape == torch.Size([bs_test, c_out])


def test_Net_dw(
    block, expansion,
    dw
):
    """test Net"""
    c_in = 3
    img_size = 16
    c_out = 8
    name = "Test name"

    mc = Net(
        name, c_in, c_out, block,
        expansion=expansion,
        stem_sizes=[8, 16],
        block_sizes=[16, 32, 64, 128],
        dw=dw
    )
    assert f"{name} constructor" in str(mc)
    model = mc()
    xb = torch.randn(bs_test, c_in, img_size, img_size)
    pred = model(xb)
    assert pred.shape == torch.Size([bs_test, c_out])


def test_Net_2(
    block, expansion,
    bn_1st, zero_bn,
):
    """test Net"""
    c_in = 3
    img_size = 16
    c_out = 8
    name = "Test name"

    mc = Net(
        name, c_in, c_out, block,
        expansion=expansion,
        stem_sizes=[8, 16],
        block_sizes=[16, 32, 64, 128],
        bn_1st=bn_1st, zero_bn=zero_bn,
    )
    assert f"{name} constructor" in str(mc)
    model = mc()
    xb = torch.randn(bs_test, c_in, img_size, img_size)
    pred = model(xb)
    assert pred.shape == torch.Size([bs_test, c_out])


def test_Net_stem(
    stem_bn_end,
    stem_stride_on
):
    """test Net"""
    c_in = 3
    img_size = 16
    c_out = 8
    name = "Test name"

    mc = Net(
        name, c_in, c_out,
        stem_sizes=[8, 16],
        block_sizes=[16, 32, 64, 128],
        stem_bn_end=stem_bn_end,
        stem_stride_on=stem_stride_on
    )
    assert f"{name} constructor" in str(mc)
    model = mc()
    xb = torch.randn(bs_test, c_in, img_size, img_size)
    pred = model(xb)
    assert pred.shape == torch.Size([bs_test, c_out])