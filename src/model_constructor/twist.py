from collections import OrderedDict
from .layers import ConvLayer, noop, act, SimpleSelfAttention

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


__all__ = ['ConvTwist', 'ConvLayerTwist', 'NewResBlockTwist', 'ResBlockTwist']


class ConvTwist(nn.Module):
    '''Replacement for Conv2d (kernelsize 3x3)'''
    permute = True
    twist = False
    use_groups = True
    groups_ch = 8

    def __init__(self, ni, nf,
                 ks=3, stride=1, padding=1, bias=False,
                 groups=1, iters=1, init_max=0.7, **kwargs):
        super().__init__()
        self.same = ni == nf and stride == 1
        self.groups = ni // self.groups_ch if self.use_groups else 1
        self.conv = nn.Conv2d(ni, nf, kernel_size=3, stride=stride, padding=1, bias=False, groups=self.groups)
        if self.twist:
            std = self.conv.weight.std().item()
            self.coeff_Ax = nn.Parameter(torch.empty((nf, ni // groups)).normal_(0, std), requires_grad=True)
            self.coeff_Ay = nn.Parameter(torch.empty((nf, ni // groups)).normal_(0, std), requires_grad=True)
        self.iters = iters
        self.stride = stride
        self.DD = self.derivatives()

    def derivatives(self):
        I = torch.Tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).view(1, 1, 3, 3)   # noqa E741
        D_x = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 10
        D_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view(1, 1, 3, 3) / 10

        def convolution(K1, K2):
            return F.conv2d(K1, K2.flip(2).flip(3), padding=2)
        D_xx = convolution(I + D_x, I + D_x).view(5, 5)
        D_yy = convolution(I + D_y, I + D_y).view(5, 5)
        D_xy = convolution(I + D_x, I + D_y).view(5, 5)
        return {'x': D_x, 'y': D_y, 'xx': D_xx, 'yy': D_yy, 'xy': D_xy}

    def kernel(self, coeff_x, coeff_y):
        D_x = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).to(coeff_x.device)
        D_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).to(coeff_x.device)
        return coeff_x[:, :, None, None] * D_x + coeff_y[:, :, None, None] * D_y

    def full_kernel(self, kernel):  # permuting the groups
        if self.groups == 1:
            return kernel
        n = self.groups
        a, b, _, _ = kernel.size()
        a = a // n
        KK = torch.zeros((a * n, b * n, 3, 3)).to(kernel.device)
        for i in range(n):
            if i % 4 == 0:
                KK[a * i:a * (i + 1), b * (i + 3):b * (i + 4)] = kernel[a * i:a * (i + 1)]
            else:
                KK[a * i:a * (i + 1), b * (i - 1):b * i] = kernel[a * i:a * (i + 1)]
        return KK

    def _conv(self, inpt, kernel=None):
        if kernel is None:
            kernel = self.conv.weight
        if self.permute is False:
            return F.conv2d(inpt, kernel, padding=1, stride=self.stride, groups=self.groups)
        else:
            return F.conv2d(inpt, self.full_kernel(kernel), padding=1, stride=self.stride, groups=1)

    def symmetrize(self, conv_wt):
        if self.same:
            n = conv_wt.size()[1]
            for i in range(self.groups):
                conv_wt.data[n * i:n * (i + 1)] = (conv_wt[n * i:n * (i + 1)]
                                                   + torch.transpose(conv_wt[n * i:n * (i + 1)], 0, 1)) / 2  # noqa E503

    def forward(self, inpt):
        out = self._conv(inpt)
        if self.twist is False:
            return out
        _, _, h, w = out.size()
        XX = torch.from_numpy(np.indices((1, 1, h, w))[3] * 2 / w - 1).type(out.dtype).to(out.device)
        YY = torch.from_numpy(np.indices((1, 1, h, w))[2] * 2 / h - 1).type(out.dtype).to(out.device)
        kernel_x = self.kernel(self.coeff_Ax, self.coeff_Ay)
        self.symmetrize(kernel_x)
        kernel_y = kernel_x.transpose(2, 3).flip(3)  # make conv_y a 90 degree rotation of conv_x
        out = out + XX * self._conv(inpt, kernel_x) + YY * self._conv(inpt, kernel_y)
        if self.same and self.iters > 1:
            out = inpt + out / self.iters
            for _ in range(self.iters - 1):
                out = out + (self._conv(out) + XX * self._conv(out, kernel_x)
                                             + YY * self._conv(out, kernel_y)) / self.iters  # noqa E727
            out = out - inpt
        return out

    def extra_repr(self):
        return f"twist: {self.twist}, permute: {self.permute}, same: {self.same}, groups: {self.groups}"


class ConvLayerTwist(ConvLayer):  # replace Conv2d by Twist
    '''Conv layer with ConvTwist'''
    Conv2d = ConvTwist


class NewResBlockTwist(nn.Module):
    '''Resnet block with ConvTwist.
    Reduce by pool instead of stride 2.
    Now YaResBlock.'''

    def __init__(self, expansion, ni, nh, stride=1,
                 conv_layer=ConvLayer, act_fn=act, bn_1st=True,
                 pool=nn.AvgPool2d(2, ceil_mode=True), sa=False, sym=False, zero_bn=True, **kwargs):
        super().__init__()
        nf, ni = nh * expansion, ni * expansion
        self.reduce = noop if stride == 1 else pool
        layers = [("conv_0", conv_layer(ni, nh, 3, act_fn=act_fn, bn_1st=bn_1st)),
                  ("conv_1", conv_layer(nh, nf, 3, zero_bn=zero_bn, act=False, bn_1st=bn_1st))
                  ] if expansion == 1 else [
                      ("conv_0", conv_layer(ni, nh, 1, act_fn=act_fn, bn_1st=bn_1st)),
                      ("conv_1_twist", ConvLayerTwist(nh, nh, 3, act_fn=act_fn, bn_1st=bn_1st)),
                      ("conv_2", conv_layer(nh, nf, 1, zero_bn=zero_bn, act=False, bn_1st=bn_1st))
        ]
        if sa:
            layers.append(('sa', SimpleSelfAttention(nf, ks=1, sym=sym)))
        self.convs = nn.Sequential(OrderedDict(layers))
        self.idconv = noop if ni == nf else conv_layer(ni, nf, 1, act=False, bn_1st=bn_1st)
        self.merge = act_fn

    def forward(self, x):
        o = self.reduce(x)
        return self.merge(self.convs(o) + self.idconv(o))


class ResBlockTwist(nn.Module):
    '''Resnet block with ConvTwist'''

    def __init__(self, expansion, ni, nh, stride=1,
                 conv_layer=ConvLayer, act_fn=act, zero_bn=True, bn_1st=True,
                 pool=nn.AvgPool2d(2, ceil_mode=True), sa=False, sym=False, **kwargs):
        super().__init__()
        nf, ni = nh * expansion, ni * expansion
        layers = [("conv_0", conv_layer(ni, nh, 3, stride=stride, act_fn=act_fn, bn_1st=bn_1st)),
                  ("conv_1", conv_layer(nh, nf, 3, zero_bn=zero_bn, act=False, bn_1st=bn_1st))
                  ] if expansion == 1 else [
                      ("conv_0", conv_layer(ni, nh, 1, act_fn=act_fn, bn_1st=bn_1st)),
                      ("conv_1_twist", ConvLayerTwist(nh, nh, 3, stride=stride, act_fn=act_fn, bn_1st=bn_1st)),
                      ("conv_2", conv_layer(nh, nf, 1, zero_bn=zero_bn, act=False, bn_1st=bn_1st))
        ]
        if sa:
            layers.append(('sa', SimpleSelfAttention(nf, ks=1, sym=sym)))
        self.convs = nn.Sequential(OrderedDict(layers))
        self.pool = noop if stride == 1 else pool
        self.idconv = noop if ni == nf else conv_layer(ni, nf, 1, act=False)
        self.act_fn = act_fn

    def forward(self, x):
        return self.act_fn(self.convs(x) + self.idconv(self.pool(x)))
