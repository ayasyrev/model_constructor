from functools import partial

import torch

from model_constructor.layers import (SEModule, SEModuleConv,
                                      SimpleSelfAttention)
from model_constructor.model_constructor import (BottleneckBlock,
                                                 ModelConstructor)

bs_test = 4
in_chans = 3
img_size = 16
xb = torch.randn(bs_test, in_chans, img_size, img_size)


def test_MC():
    """test ModelConstructor"""
    mc = ModelConstructor()
    assert "name=" not in str(mc)
    mc.name = "MC"
    assert "name='MC'" in str(mc)
    model = mc()
    pred = model(xb)
    assert pred.shape == torch.Size([bs_test, 1000])
    # with basic block it nonsense, just check
    mc.expansion = 2
    model = mc()
    pred = model(xb)
    assert pred.shape == torch.Size([bs_test, 1000])
    mc.expansion = 1
    num_classes = 10
    mc.num_classes = num_classes
    mc.se = SEModule
    mc.sa = SimpleSelfAttention
    mc.stem_bn_end = True
    model = mc()
    pred = model(xb)
    assert pred.shape == torch.Size([bs_test, num_classes])
    # sa & se bool, but check that it work anyway
    mc = ModelConstructor(sa=1, se=1, num_classes=num_classes)  # type: ignore
    assert mc.se is SEModule
    assert mc.sa is SimpleSelfAttention
    model = mc()
    pred = model(xb)
    assert pred.shape == torch.Size([bs_test, num_classes])
    mc = ModelConstructor(
        sa=SimpleSelfAttention, se=SEModuleConv, num_classes=num_classes
    )
    assert mc.se is SEModuleConv
    assert mc.sa is SimpleSelfAttention
    model = mc()
    pred = model(xb)
    assert pred.shape == torch.Size([bs_test, num_classes])
    # create model from class. Default config, num_classes 1000. ??- how to change
    model = ModelConstructor.create_model()
    pred = model(xb)
    assert pred.shape == torch.Size([bs_test, 1000])

    model = ModelConstructor.create_model(num_classes=num_classes)
    pred = model(xb)
    assert pred.shape == torch.Size([bs_test, num_classes])


def test_MC_bottleneck():
    """test ModelConstructor w/ bottleneck block"""
    mc = ModelConstructor(block=BottleneckBlock)
    model = mc()
    pred = model(xb)
    assert pred.shape == torch.Size([bs_test, 1000])
    assert model.body.l_0.bl_0.convs.conv_0.conv.in_channels == 64
    assert model.body.l_0.bl_0.convs.conv_0.conv.out_channels == 16
    assert model.body.l_0.bl_1.convs.conv_0.conv.in_channels == 64
    mc = ModelConstructor()
    mc.block = BottleneckBlock
    mc.num_classes = 10
    model = mc()
    pred = model(xb)
    assert pred.shape == torch.Size([bs_test, 10])
    assert model.body.l_0.bl_0.convs.conv_0.conv.in_channels == 64
    assert model.body.l_0.bl_0.convs.conv_0.conv.out_channels == 16
    assert model.body.l_0.bl_1.convs.conv_0.conv.in_channels == 64

    mc.block_sizes = [256, 512, 1024, 2048]
    model = mc()
    pred = model(xb)
    assert pred.shape == torch.Size([bs_test, 10])
    assert model.body.l_0.bl_0.convs.conv_0.conv.in_channels == 64
    assert model.body.l_0.bl_0.convs.conv_0.conv.out_channels == 64
    assert model.body.l_0.bl_1.convs.conv_0.conv.in_channels == 256

    mc.block = partial(BottleneckBlock, expansion=2)
    model = mc()
    pred = model(xb)
    assert pred.shape == torch.Size([bs_test, 10])
    assert model.body.l_0.bl_0.convs.conv_0.conv.in_channels == 64
    assert model.body.l_0.bl_0.convs.conv_0.conv.out_channels == 128
    assert model.body.l_0.bl_1.convs.conv_0.conv.in_channels == 256
