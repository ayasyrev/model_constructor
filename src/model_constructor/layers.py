from collections import OrderedDict
from typing import List, Optional, Type, Union

import torch
import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm

__all__ = [
    "Flatten",
    "noop",
    "Noop",
    "ConvLayer",
    "act",
    "conv1d",
    "SimpleSelfAttention",
    "SEBlock",
    "SEBlockConv",
]


class Flatten(nn.Module):
    """flat x to vector"""

    def forward(self, x):
        return x.view(x.size(0), -1)


def noop(x):
    """Dummy func. Return input"""
    return x


class Noop(nn.Module):
    """Dummy module"""

    def forward(self, x):
        return x


act = nn.ReLU(inplace=True)


def get_act(act_fn: Type[nn.Module], inplace: bool = True) -> nn.Module:
    """Return obj of act_fn, inplace if possible."""
    try:
        res = act_fn(inplace=inplace)  # type: ignore
    except TypeError:
        res = act_fn()
    return res


class ConvBnAct(nn.Sequential):
    """Basic Conv + Bn + Act block"""

    convolution_module = nn.Conv2d  # can be changed in models like twist.
    batchnorm_module = nn.BatchNorm2d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = False,
        groups: int = 1,
        act_fn: Union[Type[nn.Module], bool] = nn.ReLU,
        pre_act: bool = False,
        bn_layer: bool = True,
        bn_1st: bool = True,
        zero_bn: bool = False,
    ):

        if padding is None:
            padding = kernel_size // 2
        layers: List[tuple[str, nn.Module]] = [
            (
                "conv",
                self.convolution_module(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                    groups=groups,
                ),
            )
        ]  # if no bn - bias True?
        if bn_layer:
            bn = self.batchnorm_module(out_channels)
            nn.init.constant_(bn.weight, 0.0 if zero_bn else 1.0)
            layers.append(("bn", bn))
        if act_fn:  # act_fn either nn.Module subclass or False
            if pre_act:
                act_position = 0
            elif not bn_1st:
                act_position = 1
            else:
                act_position = len(layers)
            layers.insert(act_position, ("act_fn", get_act(act_fn)))  # type: ignore
        super().__init__(OrderedDict(layers))


# NOTE First version. Leaved for backwards compatibility with old blocks, models.
class ConvLayer(nn.Sequential):
    """Basic conv layers block"""

    Conv2d = nn.Conv2d

    def __init__(
        self,
        ni,
        nf,
        ks=3,
        stride=1,
        act=True,
        act_fn=act,
        bn_layer=True,
        bn_1st=True,
        zero_bn=False,
        padding=None,
        bias=False,
        groups=1,
        **kwargs
    ):

        if padding is None:
            padding = ks // 2
        layers = [
            (
                "conv",
                self.Conv2d(
                    ni, nf, ks, stride=stride, padding=padding, bias=bias, groups=groups
                ),
            )
        ]
        act_bn = [("act_fn", act_fn)] if act else []
        if bn_layer:
            bn = nn.BatchNorm2d(nf)
            nn.init.constant_(bn.weight, 0.0 if zero_bn else 1.0)
            act_bn += [("bn", bn)]
        if bn_1st:
            act_bn.reverse()
        layers += act_bn
        super().__init__(OrderedDict(layers))


# Cell
# SA module from mxresnet at fastai. todo - add persons!!!
# Unmodified from https://github.com/fastai/fastai/blob/5c51f9eabf76853a89a9bc5741804d2ed4407e49/fastai/layers.py


def conv1d(
    ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = False
):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias:
        conv.bias.data.zero_()  # type: ignore
    return spectral_norm(conv)


class SimpleSelfAttention(nn.Module):
    """SimpleSelfAttention module.  # noqa W291
    Adapted from SelfAttention layer at
    https://github.com/fastai/fastai/blob/5c51f9eabf76853a89a9bc5741804d2ed4407e49/fastai/layers.py
    Inspired by https://arxiv.org/pdf/1805.08318.pdf
    """

    def __init__(self, n_in: int, ks=1, sym=False, use_bias=False):
        super().__init__()
        self.conv = conv1d(n_in, n_in, ks, padding=ks // 2, bias=use_bias)
        self.gamma = torch.nn.Parameter(torch.tensor([0.0]))  # type: ignore
        self.sym = sym
        self.n_in = n_in

    def forward(self, x):
        if self.sym:  # check ks=3
            # symmetry hack by https://github.com/mgrankin
            c = self.conv.weight.view(self.n_in, self.n_in)
            c = (c + c.t()) / 2
            self.conv.weight = c.view(self.n_in, self.n_in, 1)
        size = x.size()
        x = x.view(*size[:2], -1)  # (C,N)
        # changed the order of multiplication to avoid O(N^2) complexity
        # (x*xT)*(W*x) instead of (x*(xT*(W*x)))
        convx = self.conv(x)  # (C,C) * (C,N) = (C,N)   => O(NC^2)
        xxT = torch.bmm(
            x, x.permute(0, 2, 1).contiguous()
        )  # (C,N) * (N,C) = (C,C)   => O(NC^2)
        o = torch.bmm(xxT, convx)  # (C,C) * (C,N) = (C,N)   => O(NC^2)
        o = self.gamma * o + x
        return o.view(*size).contiguous()


class SEBlock(nn.Module):  # todo: deprecation warning.
    "se block"
    se_layer = nn.Linear
    act_fn = nn.ReLU(inplace=True)
    use_bias = True

    def __init__(self, c, r=16):
        super().__init__()
        ch = max(c // r, 1)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            OrderedDict(
                [
                    ("fc_reduce", self.se_layer(c, ch, bias=self.use_bias)),
                    ("se_act", self.act_fn),
                    ("fc_expand", self.se_layer(ch, c, bias=self.use_bias)),
                    ("sigmoid", nn.Sigmoid()),
                ]
            )
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class SEBlockConv(nn.Module):  # todo: deprecation warning.
    "se block with conv on excitation"
    se_layer = nn.Conv2d
    act_fn = nn.ReLU(inplace=True)
    use_bias = True

    def __init__(self, c, r=16):
        super().__init__()
        #         c_in = math.ceil(c//r/8)*8
        c_in = max(c // r, 1)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            OrderedDict(
                [
                    ("conv_reduce", self.se_layer(c, c_in, 1, bias=self.use_bias)),
                    ("se_act", self.act_fn),
                    ("conv_expand", self.se_layer(c_in, c, 1, bias=self.use_bias)),
                    ("sigmoid", nn.Sigmoid()),
                ]
            )
        )

    def forward(self, x):
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y.expand_as(x)


class SEModule(nn.Module):
    "se block"

    def __init__(
        self,
        channels,
        reduction=16,
        rd_channels=None,
        rd_max=False,
        se_layer=nn.Linear,
        act_fn=nn.ReLU(inplace=True),
        use_bias=True,
        gate=nn.Sigmoid,
    ):
        super().__init__()
        reducted = max(channels // reduction, 1)  # preserve zero-element tensors
        if rd_channels is None:
            rd_channels = reducted
        else:
            if rd_max:
                rd_channels = max(rd_channels, reducted)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            OrderedDict(
                [
                    ("reduce", se_layer(channels, rd_channels, bias=use_bias)),
                    ("se_act", act_fn),
                    ("expand", se_layer(rd_channels, channels, bias=use_bias)),
                    ("se_gate", gate()),
                ]
            )
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class SEModuleConv(nn.Module):
    "se block with conv on excitation"

    def __init__(
        self,
        channels,
        reduction=16,
        rd_channels=None,
        rd_max=False,
        se_layer=nn.Conv2d,
        act_fn=nn.ReLU(inplace=True),
        use_bias=True,
        gate=nn.Sigmoid,
    ):
        super().__init__()
        #       rd_channels = math.ceil(channels//reduction/8)*8
        reducted = max(channels // reduction, 1)  # preserve zero-element tensors
        if rd_channels is None:
            rd_channels = reducted
        else:
            if rd_max:
                rd_channels = max(rd_channels, reducted)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            OrderedDict(
                [
                    ("reduce", se_layer(channels, rd_channels, 1, bias=use_bias)),
                    ("se_act", act_fn),
                    ("expand", se_layer(rd_channels, channels, 1, bias=use_bias)),
                    ("gate", gate()),
                ]
            )
        )

    def forward(self, x):
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y.expand_as(x)
