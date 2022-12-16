import torch

from model_constructor.convmixer import ConvMixer, ConvMixerOriginal

from .parameters import ids_fn

bs_test = 4
img_size = 16


params = dict(
    bn_1st=[True, False],
    pre_act=[True, False],
)


def pytest_generate_tests(metafunc):
    for key, value in params.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key, value, ids=ids_fn(key, value))


def test_ConvMixer(bn_1st, pre_act):
    """test ConvMixer"""
    model = ConvMixer(dim=64, depth=4, bn_1st=bn_1st, pre_act=pre_act)
    xb = torch.randn(bs_test, 3, img_size, img_size)
    pred = model(xb)
    assert pred.shape == torch.Size([bs_test, 1000])


def test_ConvMixerOriginal():
    """test ConvMixerOriginal"""
    model = ConvMixerOriginal(dim=64, depth=4)
    xb = torch.randn(bs_test, 3, img_size, img_size)
    pred = model(xb)
    assert pred.shape == torch.Size([bs_test, 1000])
