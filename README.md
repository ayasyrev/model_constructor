# model_constructor
> Constructor to create pytorch model.


_

## Install

`pip install model-constructor`

## How to use

model = Net()

```python
model = Net()
```

# Resnet as example

Lets create resnet18 and resnet34 (default Net() is resnet18()).

```python
resnet18 = Net(block=BasicBlock, blocks=[2, 2, 2, 2])
```

```python
resnet34 = Net(block=BasicBlock, blocks=[3, 4, 6, 3])
```

# Predefined Resnet models - 18, 34, 50.

```python
from model_constructor.resnet import *
```

```python
model = resnet34(num_classes=10)
```

```python
model = resnet50(num_classes=10)
```

# Predefined Xresnet from fastai 1.

This ie simplified version from fastai v1. I did refactoring for better understand and experiment with models. For example, it's very simple to change activation funtions, different stems, batchnorm and activation order etc. In v2 much powerfull realisation.

```python
from model_constructor.xresnet import *
```

```python
model = xresnet50()
```

# Some examples.

We can experiment with models by changing some parts of model. Here only base functionality, but it can be easily extanded.

Here is some examples:
    

## Custom stem

Stem with 3 conv layers

```python
model = Net(stem=partial(Stem, stem_sizes=[32, 32]))
```

```python
model.stem
```




    Stem(
      sizes: [3, 32, 32, 64]
      (conv0): ConvLayer(
        (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): ReLU(inplace=True)
      )
      (conv1): ConvLayer(
        (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): ReLU(inplace=True)
      )
      (conv2): ConvLayer(
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): ReLU(inplace=True)
      )
      (pool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    )



```python
model = Net(stem_sizes=[32, 64])
```

```python
model.stem
```




    Stem(
      sizes: [3, 32, 64, 64]
      (conv0): ConvLayer(
        (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): ReLU(inplace=True)
      )
      (conv1): ConvLayer(
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): ReLU(inplace=True)
      )
      (conv2): ConvLayer(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): ReLU(inplace=True)
      )
      (pool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    )



## Activation function before Normalization

```python
model = Net(bn_1st=False)
```

```python
model.stem
```




    Stem(
      sizes: [3, 64]
      (conv0): ConvLayer(
        (conv): Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (act_fn): ReLU(inplace=True)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (pool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    )




## Change activation function

```python
new_act_fn = nn.LeakyReLU(inplace=True)
```

```python
model = Net(act_fn=new_act_fn)
```

```python
model.stem
```




    Stem(
      sizes: [3, 64]
      (conv0): ConvLayer(
        (conv): Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): LeakyReLU(negative_slope=0.01, inplace=True)
      )
      (pool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    )



```python
model.body.layer_0.block_0
```




    BasicBlock(
      (conv): Sequential(
        (conv_0): ConvLayer(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): LeakyReLU(negative_slope=0.01, inplace=True)
        )
        (conv_1): ConvLayer(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (merge): Noop()
      (act_conn): LeakyReLU(negative_slope=0.01, inplace=True)
    )


