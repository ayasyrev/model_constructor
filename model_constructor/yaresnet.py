# YaResBlock - former NewResBlock.
# Yet another ResNet.

import torch.nn as nn
from functools import partial
from collections import OrderedDict
from .layers import SEModule, ConvLayer, act_fn, noop, SimpleSelfAttention
from .net import Net
from torch.nn import Mish


__all__ = ['YaResBlock', 'yaresnet_parameters', 'yaresnet34', 'yaresnet50']


class YaResBlock(nn.Module):
    '''YaResBlock. Reduce by pool instead of stride 2'''

    def __init__(self, expansion, in_channels, mid_channels, stride=1,
                 conv_layer=ConvLayer, act_fn=act_fn, zero_bn=True, bn_1st=True,
                 pool=nn.AvgPool2d(2, ceil_mode=True), sa=False, sym=False,
                 groups=1, dw=False, div_groups=None,
                 se_module=SEModule, se=False, se_reduction=16
                 ):
        super().__init__()
        out_channels, in_channels = mid_channels * expansion, in_channels * expansion
        if div_groups is not None:  # check if grops != 1 and div_groups
            groups = int(mid_channels / div_groups)
        self.reduce = noop if stride == 1 else pool
        layers = [("conv_0", conv_layer(in_channels, mid_channels, 3, stride=1,
                                        act_fn=act_fn, bn_1st=bn_1st, groups=in_channels if dw else groups)),
                  ("conv_1", conv_layer(mid_channels, out_channels, 3, zero_bn=zero_bn,
                                        act=False, bn_1st=bn_1st, groups=mid_channels if dw else groups))
                  ] if expansion == 1 else [
                      ("conv_0", conv_layer(in_channels, mid_channels, 1, act_fn=act_fn, bn_1st=bn_1st)),
                      ("conv_1", conv_layer(mid_channels, mid_channels, 3, stride=1, act_fn=act_fn, bn_1st=bn_1st,
                                            groups=mid_channels if dw else groups)),
                      ("conv_2", conv_layer(mid_channels, out_channels, 1, zero_bn=zero_bn, act=False, bn_1st=bn_1st))
        ]
        if se:
            layers.append(('se', se_module(out_channels, se_reduction)))
        if sa:
            layers.append(('sa', SimpleSelfAttention(out_channels, ks=1, sym=sym)))
        self.convs = nn.Sequential(OrderedDict(layers))
        self.idconv = noop if in_channels == out_channels else conv_layer(in_channels, out_channels, 1, act=False)
        self.merge = act_fn

    def forward(self, x):
        o = self.reduce(x)
        return self.merge(self.convs(o) + self.idconv(o))


yaresnet_parameters = {'block': YaResBlock, 'stem_sizes': [3, 32, 64, 64], 'act_fn': Mish(), 'stem_stride_on': 1}
yaresnet34 = partial(Net, name='YaResnet34', expansion=1, layers=[3, 4, 6, 3], **yaresnet_parameters)
yaresnet50 = partial(Net, name='YaResnet50', expansion=4, layers=[3, 4, 6, 3], **yaresnet_parameters)
