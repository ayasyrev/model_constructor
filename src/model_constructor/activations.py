# forked from https://github.com/rwightman/pytorch-image-models/timm/models/layers/activations.py
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import Mish


__all__ = ['mish', 'Mish', 'mish_jit', 'MishJit', 'mish_jit_fwd', 'mish_jit_bwd', 'MishJitAutoFn', 'mish_me', 'MishMe',
           'hard_mish_jit', 'HardMishJit', 'hard_mish_jit_fwd', 'hard_mish_jit_bwd', 'HardMishJitAutoFn',
           'hard_mish_me', 'HardMishMe']


def mish(x, inplace: bool = False):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    NOTE: I don't have a working inplace variant
    """
    return x.mul(F.softplus(x).tanh())


# from torch v 1.9 Mish is in pytorch.

# class Mish(nn.Module):
#     """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681"""
#     def __init__(self, inplace: bool = False):
#         """NOTE: inplace variant not working """
#         super(Mish, self).__init__()

#     def forward(self, x):
#         return mish(x)


@torch.jit.script
def mish_jit(x, _inplace: bool = False):
    """Jit version of Mish.
    Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    """
    return x.mul(F.softplus(x).tanh())


class MishJit(nn.Module):
    def __init__(self, inplace: bool = False):
        """Jit version of Mish.
        Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681"""
        super(MishJit, self).__init__()

    def forward(self, x):
        return mish_jit(x)


@torch.jit.script
def mish_jit_fwd(x):
    # return x.mul(torch.tanh(F.softplus(x)))
    return x.mul(F.softplus(x).tanh())


@torch.jit.script
def mish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    x_tanh_sp = F.softplus(x).tanh()
    return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))


class MishJitAutoFn(torch.autograd.Function):
    """ Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    A memory efficient, jit scripted variant of Mish"""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return mish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return mish_jit_bwd(x, grad_output)


def mish_me(x, inplace=False):
    return MishJitAutoFn.apply(x)


class MishMe(nn.Module):
    """ Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    A memory efficient, jit scripted variant of Mish"""
    def __init__(self, inplace: bool = False):
        super(MishMe, self).__init__()

    def forward(self, x):
        return MishJitAutoFn.apply(x)


@torch.jit.script
def hard_mish_jit(x, inplace: bool = False):
    """ Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    return 0.5 * x * (x + 2).clamp(min=0, max=2)


class HardMishJit(nn.Module):
    """ Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    def __init__(self, inplace: bool = False):
        super(HardMishJit, self).__init__()

    def forward(self, x):
        return hard_mish_jit(x)


@torch.jit.script
def hard_mish_jit_fwd(x):
    return 0.5 * x * (x + 2).clamp(min=0, max=2)


@torch.jit.script
def hard_mish_jit_bwd(x, grad_output):
    m = torch.ones_like(x) * (x >= -2.)
    m = torch.where((x >= -2.) & (x <= 0.), x + 1., m)
    return grad_output * m


class HardMishJitAutoFn(torch.autograd.Function):
    """ A memory efficient, jit scripted variant of Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return hard_mish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return hard_mish_jit_bwd(x, grad_output)


def hard_mish_me(x, inplace: bool = False):
    return HardMishJitAutoFn.apply(x)


class HardMishMe(nn.Module):
    """ A memory efficient, jit scripted variant of Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    def __init__(self, inplace: bool = False):
        super(HardMishMe, self).__init__()

    def forward(self, x):
        return HardMishJitAutoFn.apply(x)
