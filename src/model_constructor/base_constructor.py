import torch.nn as nn
from collections import OrderedDict
from .layers import ConvLayer, Noop, Flatten


__all__ = ['act_fn', 'Stem', 'DownsampleBlock', 'BasicBlock', 'Bottleneck', 'BasicLayer', 'Body', 'Head', 'init_model',
           'Net']


act_fn = nn.ReLU(inplace=True)


class Stem(nn.Sequential):
    """Base stem"""

    def __init__(self, c_in=3, stem_sizes=[], stem_out=64,
                 conv_layer=ConvLayer, stride_on=0,
                 stem_bn_last=False, bn_1st=True,
                 stem_use_pool=True, stem_pool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                 **kwargs
                 ):
        self.sizes = [c_in] + stem_sizes + [stem_out]
        num_layers = len(self.sizes) - 1
        stem = [(f"conv_{i}", conv_layer(self.sizes[i], self.sizes[i + 1],
                stride=2 if i == stride_on else 1, act=True,
                bn_layer=not stem_bn_last if i == num_layers - 1 else True,
                bn_1st=bn_1st, **kwargs))
                for i in range(num_layers)]
        if stem_use_pool:
            stem += [('pool', stem_pool)]
        if stem_bn_last:
            stem.append(('bn', nn.BatchNorm2d(stem_out)))
        super().__init__(OrderedDict(stem))

    def extra_repr(self):
        return f"sizes: {self.sizes}"


def DownsampleBlock(conv_layer, ni, nf, ks, stride, act=False, **kwargs):
    '''Base downsample for res-like blocks'''
    return conv_layer(ni, nf, ks, stride, act, **kwargs)


class BasicBlock(nn.Module):
    """Basic block (simplified) as in pytorch resnet"""
    def __init__(self, ni, nf, expansion=1, stride=1, zero_bn=False,
                 conv_layer=ConvLayer, act_fn=act_fn,
                 downsample_block=DownsampleBlock, **kwargs):
        super().__init__()
        self.downsample = not ni == nf or stride == 2
        self.conv = nn.Sequential(OrderedDict([
            ('conv_0', conv_layer(ni, nf, stride=stride, act_fn=act_fn, **kwargs)),
            ('conv_1', conv_layer(nf, nf, zero_bn=zero_bn, act=False, act_fn=act_fn, **kwargs))]))
        if self.downsample:
            self.downsample = downsample_block(conv_layer, ni, nf, ks=1, stride=stride, act=False, **kwargs)
        self.merge = Noop()
        self.act_conn = act_fn

    def forward(self, x):
        identity = x
        out = self.conv(x)
        if self.downsample:
            identity = self.downsample(x)
        return self.act_conn(self.merge(out + identity))


class Bottleneck(nn.Module):
    '''Bottleneck block for resnet models'''
    def __init__(self, ni, nh, expansion=4, stride=1, zero_bn=False,
                 conv_layer=ConvLayer, act_fn=act_fn,
                 downsample_block=DownsampleBlock, **kwargs):
        super().__init__()
        self.downsample = not ni == nh or stride == 2
        ni = ni * expansion
        nf = nh * expansion
        self.conv = nn.Sequential(OrderedDict([
            ('conv_0', conv_layer(ni, nh, ks=1,            act_fn=act_fn, **kwargs)),   # noqa: E241
            ('conv_1', conv_layer(nh, nh, stride=stride,   act_fn=act_fn, **kwargs)),   # noqa: E241
            ('conv_2', conv_layer(nh, nf, ks=1, zero_bn=zero_bn, act=False, act_fn=act_fn, **kwargs))]))
        if self.downsample:
            self.downsample = downsample_block(conv_layer, ni, nf, ks=1,
                                               stride=stride, act=False, act_fn=act_fn, **kwargs)
        self.merge = Noop()
        self.act_conn = act_fn

    def forward(self, x):
        identity = x
        out = self.conv(x)
        if self.downsample:
            identity = self.downsample(x)
        return self.act_conn(self.merge(out + identity))


class BasicLayer(nn.Sequential):
    '''Layer from blocks'''
    def __init__(self, block, blocks, ni, nf, expansion, stride, sa=False, **kwargs):
        self.ni = ni
        self.nf = nf
        self.blocks = blocks
        self.expansion = expansion
        super().__init__(OrderedDict(
            [(f'block_{i}', block(ni if i == 0 else nf, nf, expansion,
                                  stride if i == 0 else 1,
                                  sa=sa if i == blocks - 1 else False,
                                  **kwargs))
             for i in range(blocks)]))

    def extra_repr(self):
        return f'from {self.ni * self.expansion} to {self.nf}, {self.blocks} blocks, expansion {self.expansion}.'


class Body(nn.Sequential):
    '''Constructor for body'''
    def __init__(self, block,
                 body_in=64, body_out=512,
                 bodylayer=BasicLayer, expansion=1,
                 layer_szs=[64, 128, 256, ], blocks=[2, 2, 2, 2],
                 sa=False, **kwargs):
        layer_szs = [body_in // expansion] + layer_szs + [body_out]
        num_layers = len(layer_szs) - 1
        layers = [(f"layer_{i}", bodylayer(block, blocks[i],
                                           layer_szs[i], layer_szs[i + 1], expansion,
                                           stride=1 if i == 0 else 2,
                                           sa=sa if i == 0 else False,
                                           **kwargs))
                  for i in range(num_layers)]
        super().__init__(OrderedDict(layers))


class Head(nn.Sequential):
    '''base head'''
    def __init__(self, ni, nf, **kwargs):
        super().__init__(OrderedDict(
            [('pool', nn.AdaptiveAvgPool2d((1, 1))),
             ('flat', Flatten()),
             ('fc', nn.Linear(ni, nf)),
             ]))


def init_model(model, nonlinearity='leaky_relu'):
    '''Init model'''
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=nonlinearity)


class Net(nn.Sequential):
    '''Constructor for model'''
    def __init__(self, stem=Stem,
                 body=Body, block=BasicBlock, sa=False,
                 layer_szs=[64, 128, 256, ], blocks=[2, 2, 2, 2],
                 head=Head,
                 c_in=3, num_classes=1000,
                 body_in=64, body_out=512, expansion=1,
                 init_fn=init_model, **kwargs):
        self.init_model = init_fn
        super().__init__(OrderedDict(
            [('stem', stem(c_in=c_in, stem_out=body_in, **kwargs)),
             ('body', body(block, body_in, body_out,
                           layer_szs=layer_szs, blocks=blocks, expansion=expansion,
                           sa=sa, **kwargs)),
             ('head', head(body_out * expansion, num_classes, **kwargs))
             ]))
        self.init_model(self)
