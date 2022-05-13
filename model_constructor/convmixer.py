# Implementation of ConvMixer for the ICLR 2022 submission "Patches Are All You Need?".
# https://openreview.net/forum?id=TVHS5Y4dNvM
# Adopted from https://github.com/tmp-iclr/convmixer
# Home for convmixer: https://github.com/locuslab/convmixer
from collections import OrderedDict
from typing import Callable
import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


# As original version, act_fn as argument.
def ConvMixerOriginal(dim, depth,
                      kernel_size=9, patch_size=7, n_classes=1000,
                      act_fn=nn.GELU()):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        act_fn,
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                act_fn,
                nn.BatchNorm2d(dim)
            )),
            nn.Conv2d(dim, dim, kernel_size=1),
            act_fn,
            nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )


class ConvLayer(nn.Sequential):
    """Basic conv layers block"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 act_fn=nn.GELU(), padding=0, groups=1,
                 bn_1st=False, pre_act=False):

        conv_layer = [('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                                         padding=padding, groups=groups))]
        act_bn = [
            ('act_fn', act_fn),
            ('bn', nn.BatchNorm2d(out_channels))
        ]
        if bn_1st:
            act_bn.reverse()
        if pre_act:
            act_bn.insert(1, conv_layer[0])
            layers = act_bn
        else:
            layers = conv_layer + act_bn
        super().__init__(OrderedDict(layers))


class ConvMixer(nn.Sequential):

    def __init__(self, dim: int, depth: int,
                 kernel_size: int = 9, patch_size: int = 7, n_classes: int = 1000,
                 act_fn: nn.Module = nn.GELU(),
                 stem: nn.Module = None,
                 bn_1st: bool = False, pre_act: bool = False,
                 init_func: Callable = None):
        """ConvMixer constructor.
        Adopted from https://github.com/tmp-iclr/convmixer

        Args:
            dim (int): Dimention of model.
            depth (int): Depth of model.
            kernel_size (int, optional): Kernel size. Defaults to 9.
            patch_size (int, optional): Patch size. Defaults to 7.
            n_classes (int, optional): Number of classes. Defaults to 1000.
            act_fn (nn.Module, optional): Activation function. Defaults to nn.GELU().
            stem (nn.Module, optional): You can path different first layer..
            stem_ks (int, optional): If stem_ch not 0 - kernel size for adittional layer. Defaults to 1.
            bn_1st (bool, optional): If True - BatchNorm befor activation function. Defaults to False.
            pre_act (bool, optional): If True - activatin function befor convolution layer. Defaults to False.
            init_func (Callable, optional): External function for init model.

        """
        if pre_act:
            bn_1st = False
        if stem is None:
            stem = ConvLayer(3, dim, kernel_size=patch_size, stride=patch_size, act_fn=act_fn, bn_1st=bn_1st)

        super().__init__(
            stem,
            *[nn.Sequential(
                Residual(
                    ConvLayer(dim, dim, kernel_size, act_fn=act_fn,
                              groups=dim, padding="same", bn_1st=bn_1st, pre_act=pre_act)),
                ConvLayer(dim, dim, kernel_size=1, act_fn=act_fn, bn_1st=bn_1st, pre_act=pre_act))
              for i in range(depth)],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dim, n_classes))
        if init_func is not None:  # pragma: no cover
            init_func(self)
