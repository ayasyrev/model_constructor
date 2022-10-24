from collections import OrderedDict
from functools import partial
from typing import Callable, List, Type, Union

import torch.nn as nn

from .layers import ConvBnAct, SEModule, SimpleSelfAttention


__all__ = ['init_cnn', 'act_fn', 'ResBlock', 'ModelConstructor', 'xresnet34', 'xresnet50']


act_fn = nn.ReLU(inplace=True)


def init_cnn(module: nn.Module):
    "Init module - kaiming_normal for Conv2d and 0 for biases."
    if getattr(module, 'bias', None) is not None:
        nn.init.constant_(module.bias, 0)  # type: ignore
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight)
    for layer in module.children():
        init_cnn(layer)


class ResBlock(nn.Module):
    '''Universal Resnet block. Basic block if expansion is 1, otherwise is Bottleneck.'''

    def __init__(
        self,
        expansion: int,
        in_channels: int,
        mid_channels: int,
        stride: int = 1,
        conv_layer=ConvBnAct,
        act_fn: nn.Module = act_fn,
        zero_bn: bool = True,
        bn_1st: bool = True,
        groups: int = 1,
        dw: bool = False,
        div_groups: Union[None, int] = None,
        pool: Union[nn.Module, None] = None,
        se: Union[nn.Module, None] = None,
        sa: Union[nn.Module, None] = None,
    ):
        super().__init__()
        # pool defined at ModelConstructor.
        out_channels, in_channels = mid_channels * expansion, in_channels * expansion
        if div_groups is not None:  # check if groups != 1 and div_groups
            groups = int(mid_channels / div_groups)
        if expansion == 1:
            layers = [("conv_0", conv_layer(in_channels, mid_channels, 3, stride=stride,  # type: ignore
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
        if stride != 1 or in_channels != out_channels:
            id_layers = []
            if stride != 1 and pool is not None:  # if pool - reduce by pool else stride 2 art id_conv
                id_layers.append(("pool", pool))
            if in_channels != out_channels or (stride != 1 and pool is None):
                id_layers += [("id_conv", conv_layer(
                    in_channels, out_channels, 1,
                    stride=1 if pool else stride,
                    act_fn=False))]
            self.id_conv = nn.Sequential(OrderedDict(id_layers))
        else:
            self.id_conv = None
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


def _make_layer(self, layer_num: int) -> nn.Module:
    #  expansion, in_channels, out_channels, blocks, stride, sa):
    # if no pool on stem - stride = 2 for first layer block in body
    stride = 1 if self.stem_pool and layer_num == 0 else 2
    num_blocks = self.layers[layer_num]
    return nn.Sequential(OrderedDict([
        (f"bl_{block_num}", self.block(
            self.expansion,
            self.block_sizes[layer_num] if block_num == 0 else self.block_sizes[layer_num + 1],
            self.block_sizes[layer_num + 1],
            stride if block_num == 0 else 1,
            sa=self.sa if (block_num == num_blocks - 1) and layer_num == 0 else None,
            conv_layer=self.conv_layer,
            act_fn=self.act_fn,
            pool=self.pool,
            zero_bn=self.zero_bn, bn_1st=self.bn_1st,
            groups=self.groups, div_groups=self.div_groups,
            dw=self.dw, se=self.se
        ))
        for block_num in range(num_blocks)
    ]))


def _make_body(self):
    return nn.Sequential(OrderedDict([
        (f"l_{layer_num}", self._make_layer(self, layer_num))
        for layer_num in range(len(self.layers))
    ]))


def _make_head(self):
    head = [('pool', nn.AdaptiveAvgPool2d(1)),
            ('flat', nn.Flatten()),
            ('fc', nn.Linear(self.block_sizes[-1] * self.expansion, self.num_classes))]
    return nn.Sequential(OrderedDict(head))


class ModelConstructor():
    """Model constructor. As default - xresnet18"""
    def __init__(
        self,
        name: str = 'MC',
        in_chans: int = 3,
        num_classes: int = 1000,
        block=ResBlock,
        conv_layer=ConvBnAct,
        block_sizes: List[int] = [64, 128, 256, 512],
        layers: List[int] = [2, 2, 2, 2],
        norm: Type[nn.Module] = nn.BatchNorm2d,
        act_fn: nn.Module = nn.ReLU(inplace=True),
        pool: nn.Module = nn.AvgPool2d(2, ceil_mode=True),
        expansion: int = 1,
        groups: int = 1,
        dw: bool = False,
        div_groups: Union[int, None] = None,
        sa: Union[bool, int, Type[nn.Module]] = False,
        se: Union[bool, int, Type[nn.Module]] = False,
        se_module=None,
        se_reduction=None,
        bn_1st: bool = True,
        zero_bn: bool = True,
        stem_stride_on: int = 0,
        stem_sizes: List[int] = [32, 32, 64],
        stem_pool: Union[Type[nn.Module], None] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # type: ignore
        stem_bn_end: bool = False,
        _init_cnn: Callable = init_cnn,
        _make_stem: Callable = _make_stem,
        _make_layer: Callable = _make_layer,
        _make_body: Callable = _make_body,
        _make_head: Callable = _make_head,
    ):
        super().__init__()
        # se can be bool, int (0, 1) or nn.Module
        # se_module - deprecated. Leaved for warning and checks.
        # if stem_pool is False - no pool at stem

        self.name = name
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.block = block
        self.conv_layer = conv_layer
        self._block_sizes = block_sizes
        self.layers = layers
        self.norm = norm
        self.act_fn = act_fn
        self.pool = pool
        self.expansion = expansion
        self.groups = groups
        self.dw = dw
        self.div_groups = div_groups
        # se_module
        # se_reduction
        self.bn_1st = bn_1st
        self.zero_bn = zero_bn
        self.stem_stride_on = stem_stride_on
        self.stem_pool = stem_pool
        self.stem_bn_end = stem_bn_end
        self._init_cnn = _init_cnn
        self._make_stem = _make_stem
        self._make_layer = _make_layer
        self._make_body = _make_body
        self._make_head = _make_head

        # params = locals()
        # del params['self']
        # self.__dict__ = params

        # self._block_sizes = params['block_sizes']
        self.stem_sizes = stem_sizes
        if self.stem_sizes[0] != self.in_chans:
            self.stem_sizes = [self.in_chans] + self.stem_sizes
        self.se = se
        if self.se:
            if type(self.se) in (bool, int):  # if se=1 or se=True
                self.se = SEModule
            else:
                self.se = se  # TODO add check issubclass or isinstance of nn.Module
        self.sa = sa
        if self.sa:  # if sa=1 or sa=True
            if type(self.sa) in (bool, int):
                self.sa = SimpleSelfAttention  # default: ks=1, sym=sym
            else:
                self.sa = sa
        if se_module or se_reduction:  # pragma: no cover
            print("Deprecated. Pass se_module as se argument, se_reduction as arg to se.")  # add deprecation warning.

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
                f"  in_chans: {self.in_chans}, num_classes: {self.num_classes}\n"
                f"  expansion: {self.expansion}, groups: {self.groups}, dw: {self.dw}, div_groups: {self.div_groups}\n"
                f"  sa: {self.sa}, se: {self.se}\n"
                f"  stem sizes: {self.stem_sizes}, stride on {self.stem_stride_on}\n"
                f"  body sizes {self._block_sizes}\n"
                f"  layers: {self.layers}")


xresnet34 = partial(ModelConstructor, name='xresnet34', expansion=1, layers=[3, 4, 6, 3])
xresnet50 = partial(ModelConstructor, name='xresnet34', expansion=4, layers=[3, 4, 6, 3])
