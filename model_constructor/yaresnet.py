# YaResBlock - former NewResBlock.
# Yet another ResNet.

import torch.nn as nn
from functools import partial
from collections import OrderedDict
from .layers import SEBlock, ConvLayer, act_fn, noop, SimpleSelfAttention
from .net import Net
from torch.nn import Mish


__all__ = ['YaResBlock', 'yaresnet_parameters', 'yaresnet34', 'yaresnet50']


class YaResBlock(nn.Module):
    '''YaResBlock. Reduce by pool instead of stride 2'''
    se_block = SEBlock

    def __init__(self, expansion, ni, nh, stride=1,
                 conv_layer=ConvLayer, act_fn=act_fn, zero_bn=True, bn_1st=True,
                 pool=nn.AvgPool2d(2, ceil_mode=True), sa=False, sym=False, se=False,
                 groups=1, dw=False):
        super().__init__()
        nf, ni = nh * expansion, ni * expansion
        # if groups != 1:
        #     groups = int(nh / groups)
        self.reduce = noop if stride == 1 else pool
        layers = [("conv_0", conv_layer(ni, nh, 3, stride=1, act_fn=act_fn, bn_1st=bn_1st,
                                        groups=nh if dw else groups)),
                  ("conv_1", conv_layer(nh, nf, 3, zero_bn=zero_bn, act=False, bn_1st=bn_1st))
                  ] if expansion == 1 else [
                      ("conv_0", conv_layer(ni, nh, 1, act_fn=act_fn, bn_1st=bn_1st)),
                      ("conv_1", conv_layer(nh, nh, 3, stride=1, act_fn=act_fn, bn_1st=bn_1st,
                                            groups=nh if dw else groups)),
                      ("conv_2", conv_layer(nh, nf, 1, zero_bn=zero_bn, act=False, bn_1st=bn_1st))
        ]
        if se:
            layers.append(('se', self.se_block(nf)))
        if sa:
            layers.append(('sa', SimpleSelfAttention(nf, ks=1, sym=sym)))
        self.convs = nn.Sequential(OrderedDict(layers))
        self.idconv = noop if ni == nf else conv_layer(ni, nf, 1, act=False)
        self.merge = act_fn

    def forward(self, x):
        o = self.reduce(x)
        return self.merge(self.convs(o) + self.idconv(o))


yaresnet_parameters = {'block': YaResBlock, 'stem_sizes': [3, 32, 64, 64], 'act_fn': Mish(), 'stem_stride_on': 1}
yaresnet34 = partial(Net, name='YaResnet34', expansion=1, layers=[3, 4, 6, 3], **yaresnet_parameters)
yaresnet50 = partial(Net, name='YaResnet50', expansion=4, layers=[3, 4, 6, 3], **yaresnet_parameters)
