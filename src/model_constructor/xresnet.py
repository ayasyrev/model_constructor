from collections import OrderedDict

import torch.nn as nn

from .base_constructor import Net
from .layers import ConvLayer, Noop, act

__all__ = ['DownsampleLayer', 'XResBlock', 'xresnet18', 'xresnet34', 'xresnet50']


class DownsampleLayer(nn.Sequential):
    """Downsample layer for Xresnet Resblock"""

    def __init__(self, conv_layer, ni, nf, stride, act,
                 pool=nn.AvgPool2d(2, ceil_mode=True), pool_1st=True,
                 **kwargs):
        layers = [] if stride == 1 else [('pool', pool)]
        layers += [] if ni == nf else [('idconv', conv_layer(ni, nf, 1, act=act, **kwargs))]
        if not pool_1st:
            layers.reverse()
        super().__init__(OrderedDict(layers))


class XResBlock(nn.Module):
    '''XResnet block'''

    def __init__(self, ni, nh, expansion=1, stride=1, zero_bn=True,
                 conv_layer=ConvLayer, act_fn=act, **kwargs):
        super().__init__()
        nf, ni = nh * expansion, ni * expansion
        layers = [('conv_0', conv_layer(ni, nh, 3, stride=stride, act_fn=act_fn, **kwargs)),
                  ('conv_1', conv_layer(nh, nf, 3, zero_bn=zero_bn, act=False, act_fn=act_fn, **kwargs))
                  ] if expansion == 1 else [
                      ('conv_0', conv_layer(ni, nh, 1, act_fn=act_fn, **kwargs)),
                      ('conv_1', conv_layer(nh, nh, 3, stride=stride, act_fn=act_fn, **kwargs)),
                      ('conv_2', conv_layer(nh, nf, 1, zero_bn=zero_bn, act=False, act_fn=act_fn, **kwargs))
        ]
        self.convs = nn.Sequential(OrderedDict(layers))
        self.identity = DownsampleLayer(conv_layer, ni, nf, stride,
                                        act=False, act_fn=act_fn, **kwargs) if ni != nf or stride == 2 else Noop()
        self.merge = Noop()
        self.act_fn = act_fn

    def forward(self, x):
        return self.act_fn(self.merge(self.convs(x) + self.identity(x)))


def xresnet18(**kwargs):
    """Constructs xresnet18 model. """
    return Net(stem_sizes=[32, 32], block=XResBlock, blocks=[2, 2, 2, 2], expansion=1, **kwargs)


def xresnet34(**kwargs):
    """Constructs xresnet34 model. """
    return Net(stem_sizes=[32, 32], block=XResBlock, blocks=[3, 4, 6, 3], expansion=1, **kwargs)


def xresnet50(**kwargs):
    """Constructs xresnet50 model. """
    return Net(stem_sizes=[32, 32], block=XResBlock, blocks=[3, 4, 6, 3], expansion=4, **kwargs)
