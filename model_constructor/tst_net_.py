#
# intermediate vers - at this stage worked!
__all__ = ['Flatten', 'init_cnn', 'conv', 'noop', 'act_fn', 'conv_layer', 'ResBlock', 'XResNet', 'xresnet', 'me',
           'xresnet18_deep', 'xresnet34_deep', 'xresnet50_deep', 'ConvL']

# Cell
import torch.nn as nn
# import torch
from collections import OrderedDict
from .layers import *

# Cell
import torch,math,sys
import torch.utils.model_zoo as model_zoo
from functools import partial
from fastai.torch_core import Module

# Cell
act_fn = nn.ReLU(inplace=True)

# in layers
class Flatten(Module):
    def forward(self, x): return x.view(x.size(0), -1)

def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d,nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)

def conv(ni, nf, ks=3, stride=1, bias=False):
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)
# in layers
def noop(x): return x

# Cell

def conv_layer(ni, nf, ks=3, stride=1, zero_bn=False, act=True):
    bn = nn.BatchNorm2d(nf)
    nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
    layers = [conv(ni, nf, ks, stride=stride), bn]
    if act: layers.append(act_fn)
    return nn.Sequential(*layers)

# Cell
class ResBlock(Module):
    def __init__(self, expansion, ni, nh, stride=1):
        nf,ni = nh*expansion,ni*expansion
        layers  = [conv_layer(ni, nh, 3, stride=stride),
                   conv_layer(nh, nf, 3, zero_bn=True, act=False)
        ] if expansion == 1 else [
                   conv_layer(ni, nh, 1),
                   conv_layer(nh, nh, 3, stride=stride),
                   conv_layer(nh, nf, 1, zero_bn=True, act=False)
        ]
        self.convs = nn.Sequential(*layers)
        # TODO: check whether act=True works better
        self.idconv = noop if ni==nf else conv_layer(ni, nf, 1, act=False)
        self.pool = noop if stride==1 else nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x): return act_fn(self.convs(x) + self.idconv(self.pool(x)))

# Cell
# v5
class XResNet():
    def __init__(self, expansion=1, layers=[2,2,2,2], c_in=3, c_out=1000):
        super().__init__()
        self.c_in, self.c_out,self.expansion,self.layers = c_in,c_out,expansion,layers
        self.stem_sizes = [c_in,32,32,64]
        self.block_szs = [64//expansion,64,128,256,512] +[256]*(len(layers)-4)
        self._conv_layer = conv_layer
        self.block = ResBlock
        self.norm = nn.BatchNorm2d
        self.act_fn=nn.ReLU(inplace=True)
#     @property
#     def conv_layer(self): return self._conv_layer
    def conv_layer(self, ni, nf, ks=3, stride=1, zero_bn=False, act=True):
        bn = self.norm(nf)
        nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
        layers = [conv(ni, nf, ks, stride=stride), bn]
        if act: layers.append(self.act_fn)
        return nn.Sequential(*layers)

    @property
    def stem(self):
        return self._make_stem()
    @property
    def head(self):
        return self._make_head()
    @property
    def body(self):
        return self._make_body()

    def _make_stem(self):
        stem = []
        for i in range(len(self.stem_sizes)-1):
            stem.append((f"conv_{i}", self.conv_layer(self.stem_sizes[i], self.stem_sizes[i+1], stride=2 if i==0 else 1)))
        stem.append(('stem_pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)))
        return nn.Sequential(OrderedDict(stem))

    def _make_head(self):
        head = [('pool', nn.AdaptiveAvgPool2d(1)),
                ('flat', Flatten()),
                ('fc',   nn.Linear(self.block_szs[-1]*self.expansion, self.c_out))]
        return nn.Sequential(OrderedDict(head))

    def _make_body(self):
        blocks = [(f"l_{i}", self._make_layer(self.expansion,
                        self.block_szs[i], self.block_szs[i+1], l, 1 if i==0 else 2))
                  for i,l in enumerate(self.layers)]
        return nn.Sequential(OrderedDict(blocks))

    def _make_layer(self, expansion, ni, nf, blocks, stride):
        return nn.Sequential(
            *[self.block(expansion, ni if i==0 else nf, nf, stride if i==0 else 1)
              for i in range(blocks)])
    def __call__(self):
        model = nn.Sequential(OrderedDict([
#             ('stem', nn.Sequential(*self.stem)),
            ('stem', self.stem),
            ('body', self.body),
            ('head', self.head)
        ]))
        init_cnn(model)
        return model

# Cell
def xresnet(expansion, n_layers, name, c_out=1000, pretrained=False, **kwargs):
    model = XResNet(expansion, n_layers, c_out=c_out, **kwargs)
#     return model
    return model()

# Cell
me = sys.modules[__name__]
for n,e,l in [
    [ 18 , 1, [2,2,2 ,2] ],
    [ 34 , 1, [3,4,6 ,3] ],
    [ 50 , 4, [3,4,6 ,3] ],
    [ 101, 4, [3,4,23,3] ],
    [ 152, 4, [3,8,36,3] ],
]:
    name = f'xresnet{n}'
    setattr(me, name, partial(xresnet, expansion=e, n_layers=l, name=name))

xresnet18_deep = partial(xresnet, expansion=1, n_layers=[2, 2,  2, 2,1,1], name='xresnet18_deep')
xresnet34_deep = partial(xresnet, expansion=1, n_layers=[3, 4,  6, 3,1,1], name='xresnet34_deep')
xresnet50_deep = partial(xresnet, expansion=4, n_layers=[3, 4,  6, 3,1,1], name='xresnet50_deep')


# Cell
class ConvL():
    def __init__(self, ni, nf, ks=3, stride=1, zero_bn=False, act=True):
        super().__init__()
        bn = nn.BatchNorm2d(nf)
        nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
        self.layers = [conv(ni, nf, ks, stride=stride), bn]
        if act: self.layers.append(act_fn)

    def __call__(self):
        return nn.Sequential(*self.layers)