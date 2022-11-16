# Layers

> Basic layers for constructor.

## ConvBnAct - nn.module


```python

conv_layer = ConvBnAct(32, 32)
conv_layer
```
??? done "output"  
    <pre>ConvBnAct(
      (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act_fn): ReLU(inplace=True)
    )




```python

conv_layer = ConvBnAct(32, 32, kernel_size=1)
conv_layer
```
??? done "output"  
    <pre>ConvBnAct(
      (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act_fn): ReLU(inplace=True)
    )




```python

conv_layer = ConvBnAct(32, 32, stride=2)
conv_layer
```
??? done "output"  
    <pre>ConvBnAct(
      (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act_fn): ReLU(inplace=True)
    )




```python

conv_layer = ConvBnAct(32, 32, groups=32)
conv_layer
```
??? done "output"  
    <pre>ConvBnAct(
      (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
      (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act_fn): ReLU(inplace=True)
    )




```python

conv_layer = ConvBnAct(32, 64, act_fn=False)
conv_layer
```
??? done "output"  
    <pre>ConvBnAct(
      (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )




```python

conv_layer = ConvBnAct(32, 64, bn_layer=False)
conv_layer
```
??? done "output"  
    <pre>ConvBnAct(
      (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (act_fn): ReLU(inplace=True)
    )




```python

conv_layer = ConvBnAct(32, 64, pre_act=True)
conv_layer
```
??? done "output"  
    <pre>ConvBnAct(
      (act_fn): ReLU(inplace=True)
      (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )




```python
conv_layer[0]
```
???+ done "output"  
    <pre>ReLU(inplace=True)




```python

conv_layer = ConvBnAct(32, 64, bn_1st=True, pre_act=True)
conv_layer
```
??? done "output"  
    <pre>ConvBnAct(
      (act_fn): ReLU(inplace=True)
      (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )




```python

conv_layer = ConvBnAct(32, 64, bn_1st=True)
conv_layer
```
??? done "output"  
    <pre>ConvBnAct(
      (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act_fn): ReLU(inplace=True)
    )




```python

conv_layer = ConvBnAct(32, 64, bn_1st=True, act_fn=nn.LeakyReLU())
conv_layer
```
??? done "output"  
    <pre>ConvBnAct(
      (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act_fn): LeakyReLU(negative_slope=0.01)
    )



## SimpleSelfAttention

SA module from mxresnet at fastai.


```python
sa = SimpleSelfAttention(32)
sa
```
???+ done "output"  
    <pre>SimpleSelfAttention(
      (conv): Conv1d(32, 32, kernel_size=(1,), stride=(1,), bias=False)
    )



## SEModule


```python
se_block = SEModule(128)
se_block
```
???+ done "output"  
    <pre>SEModule(
      (squeeze): AdaptiveAvgPool2d(output_size=1)
      (excitation): Sequential(
        (reduce): Linear(in_features=128, out_features=8, bias=True)
        (se_act): ReLU(inplace=True)
        (expand): Linear(in_features=8, out_features=128, bias=True)
        (se_gate): Sigmoid()
      )
    )




```python
se_block = SEModule(128, rd_channels=32)
se_block
```
???+ done "output"  
    <pre>SEModule(
      (squeeze): AdaptiveAvgPool2d(output_size=1)
      (excitation): Sequential(
        (reduce): Linear(in_features=128, out_features=32, bias=True)
        (se_act): ReLU(inplace=True)
        (expand): Linear(in_features=32, out_features=128, bias=True)
        (se_gate): Sigmoid()
      )
    )



## SEModuleConv


```python
se_block = SEModuleConv(128)
se_block
```
???+ done "output"  
    <pre>SEModuleConv(
      (squeeze): AdaptiveAvgPool2d(output_size=1)
      (excitation): Sequential(
        (reduce): Conv2d(128, 8, kernel_size=(1, 1), stride=(1, 1))
        (se_act): ReLU(inplace=True)
        (expand): Conv2d(8, 128, kernel_size=(1, 1), stride=(1, 1))
        (gate): Sigmoid()
      )
    )




```python
se_block = SEModuleConv(128, reduction=32)
se_block
```
???+ done "output"  
    <pre>SEModuleConv(
      (squeeze): AdaptiveAvgPool2d(output_size=1)
      (excitation): Sequential(
        (reduce): Conv2d(128, 4, kernel_size=(1, 1), stride=(1, 1))
        (se_act): ReLU(inplace=True)
        (expand): Conv2d(4, 128, kernel_size=(1, 1), stride=(1, 1))
        (gate): Sigmoid()
      )
    )




```python
se_block = SEModuleConv(128, rd_channels=32)
se_block
```
???+ done "output"  
    <pre>SEModuleConv(
      (squeeze): AdaptiveAvgPool2d(output_size=1)
      (excitation): Sequential(
        (reduce): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
        (se_act): ReLU(inplace=True)
        (expand): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        (gate): Sigmoid()
      )
    )




```python
se_block = SEModuleConv(128, reduction=4, rd_channels=16, rd_max=True)
se_block
```
???+ done "output"  
    <pre>SEModuleConv(
      (squeeze): AdaptiveAvgPool2d(output_size=1)
      (excitation): Sequential(
        (reduce): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
        (se_act): ReLU(inplace=True)
        (expand): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        (gate): Sigmoid()
      )
    )





model_constructor
by ayasyrev
