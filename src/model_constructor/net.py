from collections import OrderedDict
from functools import partial

import torch.nn as nn

from .layers import ConvLayer, Flatten, SEBlock, SimpleSelfAttention, noop


__all__ = ['init_cnn', 'act_fn', 'ResBlock', 'NewResBlock', 'Net', 'xresnet34', 'xresnet50']


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
    se_block = SEBlock

    def __init__(self, expansion, ni, nh, stride=1,
                 conv_layer=ConvLayer, act_fn=act_fn, zero_bn=True, bn_1st=True,
                 pool=nn.AvgPool2d(2, ceil_mode=True), sa=False, sym=False, se=False, se_reduction=16,
                 groups=1, dw=False, div_groups=None):
        super().__init__()
        nf, ni = nh * expansion, ni * expansion
        if div_groups is not None:  # check if groups != 1 and div_groups
            groups = int(nh / div_groups)
        if expansion == 1:
            layers = [("conv_0", conv_layer(ni, nh, 3, stride=stride, act_fn=act_fn, bn_1st=bn_1st,
                                            groups=ni if dw else groups)),
                      ("conv_1", conv_layer(nh, nf, 3, zero_bn=zero_bn, act=False, bn_1st=bn_1st))
                      ]
        else:
            layers = [("conv_0", conv_layer(ni, nh, 1, act_fn=act_fn, bn_1st=bn_1st)),
                      ("conv_1", conv_layer(nh, nh, 3, stride=stride, act_fn=act_fn, bn_1st=bn_1st,
                                            groups=nh if dw else groups)),
                      ("conv_2", conv_layer(nh, nf, 1, zero_bn=zero_bn, act=False, bn_1st=bn_1st))
                      ]
        if se:
            layers.append(('se', self.se_block(nf, se_reduction)))
        if sa:
            layers.append(('sa', SimpleSelfAttention(nf, ks=1, sym=sym)))
        self.convs = nn.Sequential(OrderedDict(layers))
        self.pool = noop if stride == 1 else pool
        self.idconv = noop if ni == nf else conv_layer(ni, nf, 1, act=False)
        self.act_fn = act_fn

    def forward(self, x):
        return self.act_fn(self.convs(x) + self.idconv(self.pool(x)))


# NewResBlock now is YaResBlock - Yet Another ResNet Block! It is now at model_constructor.yaresnet.


class NewResBlock(nn.Module):  # todo: deprecation worning.
    '''YaResnet block.
    This is first impl, deprecated, use yaresnet module.
    '''
    se_block = SEBlock

    def __init__(self, expansion, ni, nh, stride=1,
                 conv_layer=ConvLayer, act_fn=act_fn, zero_bn=True, bn_1st=True,
                 pool=nn.AvgPool2d(2, ceil_mode=True), sa=False, sym=False, se=False, se_reduction=16,
                 groups=1, dw=False, div_groups=None):
        super().__init__()
        nf, ni = nh * expansion, ni * expansion
        if div_groups is not None:  # check if groups != 1 and div_groups
            groups = int(nh / div_groups)
        self.reduce = noop if stride == 1 else pool
        if expansion == 1:
            layers = [("conv_0", conv_layer(ni, nh, 3, stride=1, act_fn=act_fn, bn_1st=bn_1st,
                                            groups=ni if dw else groups)),
                      ("conv_1", conv_layer(nh, nf, 3, zero_bn=zero_bn, act=False, bn_1st=bn_1st))
                      ]
        else:
            layers = [("conv_0", conv_layer(ni, nh, 1, act_fn=act_fn, bn_1st=bn_1st)),
                      ("conv_1", conv_layer(nh, nh, 3, stride=1, act_fn=act_fn, bn_1st=bn_1st,
                                            groups=nh if dw else groups)),
                      ("conv_2", conv_layer(nh, nf, 1, zero_bn=zero_bn, act=False, bn_1st=bn_1st))
                      ]
        if se:
            layers.append(('se', self.se_block(nf, se_reduction)))
        if sa:
            layers.append(('sa', SimpleSelfAttention(nf, ks=1, sym=sym)))
        self.convs = nn.Sequential(OrderedDict(layers))
        self.idconv = noop if ni == nf else conv_layer(ni, nf, 1, act=False)
        self.merge = act_fn

    def forward(self, x):
        o = self.reduce(x)
        return self.merge(self.convs(o) + self.idconv(o))


def _make_stem(self):
    stem = [(f"conv_{i}", self.conv_layer(self.stem_sizes[i], self.stem_sizes[i + 1],
                                          stride=2 if i == self.stem_stride_on else 1,
                                          bn_layer=(not self.stem_bn_end) if i == (len(self.stem_sizes) - 2) else True,
                                          act_fn=self.act_fn, bn_1st=self.bn_1st))
            for i in range(len(self.stem_sizes) - 1)]
    stem.append(('stem_pool', self.stem_pool))
    if self.stem_bn_end:
        stem.append(('norm', self.norm(self.stem_sizes[-1])))
    return nn.Sequential(OrderedDict(stem))


def _make_layer(self, expansion, ni, nf, blocks, stride, sa):
    layers = [(f"bl_{i}", self.block(expansion, ni if i == 0 else nf, nf,
                                     stride if i == 0 else 1, sa=sa if i == blocks - 1 else False,
                                     conv_layer=self.conv_layer, act_fn=self.act_fn, pool=self.pool,
                                     zero_bn=self.zero_bn, bn_1st=self.bn_1st,
                                     groups=self.groups, div_groups=self.div_groups,
                                     dw=self.dw, se=self.se))
              for i in range(blocks)]
    return nn.Sequential(OrderedDict(layers))


def _make_body(self):
    blocks = [(f"l_{i}", self._make_layer(self, self.expansion,
                                          ni=self.block_sizes[i], nf=self.block_sizes[i + 1],
                                          blocks=l, stride=1 if i == 0 else 2,
                                          sa=self.sa if i == 0 else False))
              for i, l in enumerate(self.layers)]
    return nn.Sequential(OrderedDict(blocks))


def _make_head(self):
    head = [('pool', nn.AdaptiveAvgPool2d(1)),
            ('flat', Flatten()),
            ('fc', nn.Linear(self.block_sizes[-1] * self.expansion, self.c_out))]
    return nn.Sequential(OrderedDict(head))


class Net():  # todo: deprecation worning.
    """Model constructor. As default - xresnet18.
    First version, still here for compatibility. Use ModelConstructor instead.
    """
    def __init__(self, name='Net', c_in=3, c_out=1000,
                 block=ResBlock, conv_layer=ConvLayer,
                 block_sizes=[64, 128, 256, 512], layers=[2, 2, 2, 2],
                 norm=nn.BatchNorm2d,
                 act_fn=nn.ReLU(inplace=True),
                 pool=nn.AvgPool2d(2, ceil_mode=True),
                 expansion=1, groups=1, dw=False, div_groups=None,
                 sa=False, se=False, se_reduction=16,
                 bn_1st=True,
                 zero_bn=True,
                 stem_stride_on=0,
                 stem_sizes=[32, 32, 64],
                 stem_pool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
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

    @property
    def block_sizes(self):
        return [self.stem_sizes[-1] // self.expansion] + self._block_sizes + [256] * (len(self.layers) - 4)

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
        model.extra_repr = lambda: f"model {self.name}"
        return model

    def __repr__(self):
        return (f"{self.name} constructor\n"
                f"  c_in: {self.c_in}, c_out: {self.c_out}\n"
                f"  expansion: {self.expansion}, groups: {self.groups}, dw: {self.dw}, div_groups: {self.div_groups}\n"
                f"  sa: {self.sa}, se: {self.se}\n"
                f"  stem sizes: {self.stem_sizes}, stride on {self.stem_stride_on}\n"
                f"  body sizes {self._block_sizes}\n"
                f"  layers: {self.layers}")


xresnet34 = partial(Net, name='xresnet34', expansion=1, layers=[3, 4, 6, 3])
xresnet50 = partial(Net, name='xresnet34', expansion=4, layers=[3, 4, 6, 3])
