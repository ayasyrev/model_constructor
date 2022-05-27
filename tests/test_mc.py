import torch

from model_constructor import ModelConstructor
from model_constructor.layers import SEModule, SEModuleConv, SimpleSelfAttention


bs_test = 4


def test_MC():
    """test ModelConstructor"""
    img_size = 16
    mc = ModelConstructor()
    assert "MC constructor" in str(mc)
    model = mc()
    xb = torch.randn(bs_test, 3, img_size, img_size)
    pred = model(xb)
    assert pred.shape == torch.Size([bs_test, 1000])
    num_classes = 10
    mc.num_classes = num_classes
    mc.se = SEModule
    mc.sa = SimpleSelfAttention
    mc.stem_bn_end = True
    model = mc()
    pred = model(xb)
    assert pred.shape == torch.Size([bs_test, num_classes])
    mc = ModelConstructor(sa=1, se=1, num_classes=num_classes)
    assert mc.se is SEModule
    assert mc.sa is SimpleSelfAttention
    model = mc()
    pred = model(xb)
    assert pred.shape == torch.Size([bs_test, num_classes])
    mc = ModelConstructor(sa=SimpleSelfAttention, se=SEModuleConv, num_classes=num_classes)
    assert mc.se is SEModuleConv
    assert mc.sa is SimpleSelfAttention
    model = mc()
    pred = model(xb)
    assert pred.shape == torch.Size([bs_test, num_classes])
