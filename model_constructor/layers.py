import torch.nn as nn
import torch
from torch.nn.utils import spectral_norm
from collections import OrderedDict


__all__ = ['Flatten', 'noop', 'Noop', 'ConvLayer', 'act_fn',
           'conv1d', 'SimpleSelfAttention', 'SEBlock', 'SEBlockConv']


class Flatten(nn.Module):
    '''flat x to vector'''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def noop(x):
    '''Dummy func. Return input'''
    return x


class Noop(nn.Module):
    '''Dummy module'''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


act_fn = nn.ReLU(inplace=True)


class ConvLayer(nn.Sequential):
    """Basic conv layers block"""
    Conv2d = nn.Conv2d

    def __init__(self, ni, nf, ks=3, stride=1,
                 act=True, act_fn=act_fn,
                 bn_layer=True, bn_1st=True, zero_bn=False,
                 padding=None, bias=False, groups=1, **kwargs):

        if padding is None:
            padding = ks // 2
        layers = [('conv', self.Conv2d(ni, nf, ks, stride=stride,
                                       padding=padding, bias=bias, groups=groups))]
        act_bn = [('act_fn', act_fn)] if act else []
        if bn_layer:
            bn = nn.BatchNorm2d(nf)
            nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
            act_bn += [('bn', bn)]
        if bn_1st:
            act_bn.reverse()
        layers += act_bn
        super().__init__(OrderedDict(layers))

# Cell
# SA module from mxresnet at fastai. todo - add persons!!!
# Unmodified from https://github.com/fastai/fastai/blob/5c51f9eabf76853a89a9bc5741804d2ed4407e49/fastai/layers.py


def conv1d(ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = False):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias:
        conv.bias.data.zero_()
    return spectral_norm(conv)


class SimpleSelfAttention(nn.Module):
    '''SimpleSelfAttention module.  # noqa W291
    Adapted from SelfAttention layer at  
    https://github.com/fastai/fastai/blob/5c51f9eabf76853a89a9bc5741804d2ed4407e49/fastai/layers.py  
    Inspired by https://arxiv.org/pdf/1805.08318.pdf  
    '''

    def __init__(self, n_in: int, ks=1, sym=False):
        super().__init__()
        self.conv = conv1d(n_in, n_in, ks, padding=ks // 2, bias=False)
        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.sym = sym
        self.n_in = n_in

    def forward(self, x):
        if self.sym:
            # symmetry hack by https://github.com/mgrankin
            c = self.conv.weight.view(self.n_in, self.n_in)
            c = (c + c.t()) / 2
            self.conv.weight = c.view(self.n_in, self.n_in, 1)
        size = x.size()
        x = x.view(*size[:2], -1)   # (C,N)
        # changed the order of mutiplication to avoid O(N^2) complexity
        # (x*xT)*(W*x) instead of (x*(xT*(W*x)))
        convx = self.conv(x)   # (C,C) * (C,N) = (C,N)   => O(NC^2)
        xxT = torch.bmm(x, x.permute(0, 2, 1).contiguous())   # (C,N) * (N,C) = (C,C)   => O(NC^2)
        o = torch.bmm(xxT, convx)   # (C,C) * (C,N) = (C,N)   => O(NC^2)
        o = self.gamma * o + x
        return o.view(*size).contiguous()


class SEBlock(nn.Module):
    "se block"
    se_layer = nn.Linear
    act_fn = nn.ReLU(inplace=True)

    def __init__(self, c, r=16):
        super().__init__()
        ch = c // r
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            OrderedDict([('fc_reduce', self.se_layer(c, ch, bias=False)),
                         ('se_act', self.act_fn),
                         ('fc_expand', self.se_layer(ch, c, bias=False)),
                         ('sigmoid', nn.Sigmoid())
                         ]))

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class SEBlockConv(nn.Module):
    "se block with conv on excitation"
    se_layer = nn.Conv2d
    act_fn = nn.ReLU(inplace=True)

    def __init__(self, c, r=16):
        super().__init__()
#         c_in = math.ceil(c//r/8)*8
        c_in = c // r
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            OrderedDict([
                ('conv_reduce', self.se_layer(c, c_in, 1, bias=False)),
                ('se_act', self.act_fn),
                ('conv_expand', self.se_layer(c_in, c, 1, bias=False)),
                ('sigmoid', nn.Sigmoid())
            ]))

    def forward(self, x):
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y.expand_as(x)
