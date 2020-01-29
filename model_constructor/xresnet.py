# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/03_xresnet.ipynb (unless otherwise specified).

__all__ = ['xresnet18', 'xresnet34', 'xresnet50']

# Cell
import torch.nn as nn
import torch
from collections import OrderedDict

# Cell
from .constructor import *
from .layers import *

# Cell
def xresnet18(**kwargs):
    """Constructs a xresnet-18 model. """
    return Net(stem_sizes=[32,32], block=XResBlock, blocks=[2, 2, 2, 2], expansion=1, **kwargs)
def xresnet34(**kwargs):
    """Constructs axresnet-34 model. """
    return Net(stem_sizes=[32,32], block=XResBlock, blocks=[3, 4, 6, 3], expansion=1, **kwargs)
def xresnet50(**kwargs):
    """Constructs axresnet-34 model. """
    return Net(stem_sizes=[32,32],block=XResBlock, blocks=[3, 4, 6, 3], expansion=4, **kwargs)