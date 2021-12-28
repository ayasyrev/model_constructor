from collections import OrderedDict
from functools import partial
from typing import Callable, Union

import torch.nn as nn

from .layers import ConvBnAct, SEModule, SimpleSelfAttention


__all__ = ['init_cnn', 'act_fn', 'ResBlock', 'ModelConstructor', 'xresnet34', 'xresnet50']


act_fn = nn.ReLU(inplace=True)


def init_cnn(module: nn.Module):
    "Init module - kaiming_normal for Conv2d and 0 for biases."
    if getattr(module, 'bias', None) is not None:
        nn.init.constant_(module.bias, 0)
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight)
    for layer in module.children():
        init_cnn(layer)


class ResBlock(nn.Module):
    '''Resnet block'''

    def __init__(self, expansion, in_channels, mid_channels, stride=1,
                 conv_layer=ConvBnAct, act_fn=act_fn, zero_bn=True, bn_1st=True,
                 groups=1, dw=False, div_groups=None,
                 pool=None,  # pool defined at ModelConstuctor.
                 se=None, sa=None
                 ):
        super().__init__()
        out_channels, in_channels = mid_channels * expansion, in_channels * expansion
        if div_groups is not None:  # check if grops != 1 and div_groups
            groups = int(mid_channels / div_groups)
        if expansion == 1:
            layers = [("conv_0", conv_layer(in_channels, mid_channels, 3, stride=stride,
                                            act_fn=act_fn, bn_1st=bn_1st, groups=in_channels if dw else groups)),
                      ("conv_1", conv_layer(mid_channels, out_channels, 3, zero_bn=zero_bn,
                                            act_fn=False, bn_1st=bn_1st, groups=mid_channels if dw else groups))
                      ]
        else:
            layers = [("conv_0", conv_layer(in_channels, mid_channels, 1, act_fn=act_fn, bn_1st=bn_1st)),
                      ("conv_1", conv_layer(mid_channels, mid_channels, 3, stride=stride, act_fn=act_fn, bn_1st=bn_1st,
                                            groups=mid_channels if dw else groups)),
                      ("conv_2", conv_layer(mid_channels, out_channels, 1, zero_bn=zero_bn, act_fn=False, bn_1st=bn_1st))  # noqa E501
                      ]
        if se:
            layers.append(('se', se(out_channels)))
        if sa:
            layers.append(('sa', sa(out_channels)))
        self.convs = nn.Sequential(OrderedDict(layers))
        id_layers = []
        if stride != 1 and pool:
            id_layers.append(("pool", pool))
        id_layers += [] if in_channels == out_channels else [("id_conv", conv_layer(in_channels, out_channels, 1,
                                                                                    stride=1 if pool else stride,
                                                                                    act_fn=False))]
        self.id_conv = None if id_layers == [] else nn.Sequential(OrderedDict(id_layers))
        self.act_fn = act_fn

    def forward(self, x):
        identity = self.id_conv(x) if self.id_conv is not None else x
        return self.act_fn(self.convs(x) + identity)


def _make_stem(self):
    stem = [(f"conv_{i}", self.conv_layer(self.stem_sizes[i], self.stem_sizes[i + 1],
                                          stride=2 if i == self.stem_stride_on else 1,
                                          bn_layer=(not self.stem_bn_end) if i == (len(self.stem_sizes) - 2) else True,
                                          act_fn=self.act_fn, bn_1st=self.bn_1st))
            for i in range(len(self.stem_sizes) - 1)]
    if self.stem_pool:
        stem.append(('stem_pool', self.stem_pool))
    if self.stem_bn_end:
        stem.append(('norm', self.norm(self.stem_sizes[-1])))
    return nn.Sequential(OrderedDict(stem))


def _make_layer(self, expansion, in_channels, out_channels, blocks, stride, sa):
    layers = [(f"bl_{i}", self.block(expansion, in_channels if i == 0 else out_channels, out_channels,
                                     stride if i == 0 else 1, sa=sa if i == blocks - 1 else None,
                                     conv_layer=self.conv_layer, act_fn=self.act_fn, pool=self.pool,
                                     zero_bn=self.zero_bn, bn_1st=self.bn_1st,
                                     groups=self.groups, div_groups=self.div_groups,
                                     dw=self.dw, se=self.se))
              for i in range(blocks)]
    return nn.Sequential(OrderedDict(layers))


def _make_body(self):
    stride = 1 if self.stem_pool else 1  # if no pool on stem - stride = 2 for first block in body
    blocks = [(f"l_{i}", self._make_layer(self, self.expansion,
                                          in_channels=self.block_sizes[i], out_channels=self.block_sizes[i + 1],
                                          blocks=l, stride=stride if i == 0 else 2,
                                          sa=self.sa if i == 0 else None))
              for i, l in enumerate(self.layers)]
    return nn.Sequential(OrderedDict(blocks))


def _make_head(self):
    head = [('pool', nn.AdaptiveAvgPool2d(1)),
            ('flat', nn.Flatten()),
            ('fc', nn.Linear(self.block_sizes[-1] * self.expansion, self.c_out))]
    return nn.Sequential(OrderedDict(head))


class ModelConstructor():
    """Model constructor. As default - xresnet18"""
    def __init__(self, name='MC', c_in=3, c_out=1000,
                 block=ResBlock, conv_layer=ConvBnAct,
                 block_sizes=[64, 128, 256, 512], layers=[2, 2, 2, 2],
                 norm=nn.BatchNorm2d,
                 act_fn=nn.ReLU(inplace=True),
                 pool=nn.AvgPool2d(2, ceil_mode=True),
                 expansion=1, groups=1, dw=False, div_groups=None,
                 sa=False,
                 se: Union[bool, Callable] = False,  # se can be bool or nn.Module
                 se_module=None, se_reduction=None,  # deprecated. Leaved for worning and checks.
                 bn_1st=True,
                 zero_bn=True,
                 stem_stride_on=0,
                 stem_sizes=[32, 32, 64],
                 stem_pool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # if stem_pool is False - no pool at stem
                 stem_bn_end=False,
                 _init_cnn=init_cnn,
                 _make_stem=_make_stem,
                 _make_layer=_make_layer,
                 _make_body=_make_body,
                 _make_head=_make_head,
                 ):
        super().__init__()

        params = locals()
        del params['self']
        self.__dict__ = params

        self._block_sizes = params['block_sizes']
        if self.stem_sizes[0] != self.c_in:
            self.stem_sizes = [self.c_in] + self.stem_sizes
        if self.se:  # TODO add check issubclass or isinstance of nn.Module
            if type(self.se) == bool:
                self.se = SEModule  # if se=1
        if self.sa:
            if type(self.sa) == bool:
                self.sa = SimpleSelfAttention  # default: ks=1, sym=sym
        if self.se_module or se_reduction:
            print("Deprecated. Pass se_module as se argument, se_reduction as arg to se.")  # add deprecation worning.

    @property
    def block_sizes(self):
        return [self.stem_sizes[-1] // self.expansion] + self._block_sizes

    @property
    def stem(self):
        return self._make_stem(self)

    @property
    def head(self):
        return self._make_head(self)

    @property
    def body(self):
        return self._make_body(self)

    def __call__(self):
        model = nn.Sequential(OrderedDict([
            ('stem', self.stem),
            ('body', self.body),
            ('head', self.head)]))
        self._init_cnn(model)
        model.extra_repr = lambda: f"{self.name}"
        return model

    def __repr__(self):
        return (f"{self.name} constructor\n"
                f"  c_in: {self.c_in}, c_out: {self.c_out}\n"
                f"  expansion: {self.expansion}, groups: {self.groups}, dw: {self.dw}, div_groups: {self.div_groups}\n"
                f"  sa: {self.sa}, se: {self.se}\n"
                f"  stem sizes: {self.stem_sizes}, stride on {self.stem_stride_on}\n"
                f"  body sizes {self._block_sizes}\n"
                f"  layers: {self.layers}")


xresnet34 = partial(ModelConstructor, name='xresnet34', expansion=1, layers=[3, 4, 6, 3])
xresnet50 = partial(ModelConstructor, name='xresnet34', expansion=4, layers=[3, 4, 6, 3])
