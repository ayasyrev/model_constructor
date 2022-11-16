# Model constructor.

> Create and tune pytorch model.

## ResBlock


```python
block = ResBlock(1,64,64)
block
```
???+ done "output"  
    <pre>ResBlock(
      (convs): Sequential(
        (conv_0): ConvBnAct(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): ReLU(inplace=True)
        )
        (conv_1): ConvBnAct(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (act_fn): ReLU(inplace=True)
    )




```python
block = ResBlock(4,64,64,dw=True)
block
```
???+ done "output"  
    <pre>ResBlock(
      (convs): Sequential(
        (conv_0): ConvBnAct(
          (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): ReLU(inplace=True)
        )
        (conv_1): ConvBnAct(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): ReLU(inplace=True)
        )
        (conv_2): ConvBnAct(
          (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (act_fn): ReLU(inplace=True)
    )




```python
block = ResBlock(4,64,64,groups=4)
block
```
???+ done "output"  
    <pre>ResBlock(
      (convs): Sequential(
        (conv_0): ConvBnAct(
          (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): ReLU(inplace=True)
        )
        (conv_1): ConvBnAct(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): ReLU(inplace=True)
        )
        (conv_2): ConvBnAct(
          (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (act_fn): ReLU(inplace=True)
    )




```python
block = ResBlock(2,64,64,act_fn=nn.LeakyReLU(), bn_1st=False)
block
```
???+ done "output"  
    <pre>ResBlock(
      (convs): Sequential(
        (conv_0): ConvBnAct(
          (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (act_fn): LeakyReLU(negative_slope=0.01)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv_1): ConvBnAct(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (act_fn): LeakyReLU(negative_slope=0.01)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv_2): ConvBnAct(
          (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (act_fn): LeakyReLU(negative_slope=0.01)
    )




```python
block = ResBlock(2, 32, 64, dw=True)
block
```
???+ done "output"  
    <pre>ResBlock(
      (convs): Sequential(
        (conv_0): ConvBnAct(
          (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): ReLU(inplace=True)
        )
        (conv_1): ConvBnAct(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): ReLU(inplace=True)
        )
        (conv_2): ConvBnAct(
          (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (id_conv): Sequential(
        (id_conv): ConvBnAct(
          (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (act_fn): ReLU(inplace=True)
    )




```python
pool = nn.AvgPool2d(2, ceil_mode=True)
```


```python
block = ResBlock(2, 32, 64, stride=2, dw=True, pool=pool)
block
```
???+ done "output"  
    <pre>ResBlock(
      (convs): Sequential(
        (conv_0): ConvBnAct(
          (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): ReLU(inplace=True)
        )
        (conv_1): ConvBnAct(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=64, bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): ReLU(inplace=True)
        )
        (conv_2): ConvBnAct(
          (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (id_conv): Sequential(
        (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
        (id_conv): ConvBnAct(
          (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (act_fn): ReLU(inplace=True)
    )




```python
from model_constructor.layers import SEModule, SimpleSelfAttention
```


```python

block = ResBlock(2, 32, 64, stride=2, dw=True, pool=pool, se=SEModule)
block
```
??? done "output"  
    <pre>ResBlock(
      (convs): Sequential(
        (conv_0): ConvBnAct(
          (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): ReLU(inplace=True)
        )
        (conv_1): ConvBnAct(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=64, bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): ReLU(inplace=True)
        )
        (conv_2): ConvBnAct(
          (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (se): SEModule(
          (squeeze): AdaptiveAvgPool2d(output_size=1)
          (excitation): Sequential(
            (fc_reduce): Linear(in_features=128, out_features=8, bias=True)
            (se_act): ReLU(inplace=True)
            (fc_expand): Linear(in_features=8, out_features=128, bias=True)
            (se_gate): Sigmoid()
          )
        )
      )
      (id_conv): Sequential(
        (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
        (id_conv): ConvBnAct(
          (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (act_fn): ReLU(inplace=True)
    )




```python

block = ResBlock(2, 32, 64, stride=2, dw=True, pool=pool, se=SEModule, sa=SimpleSelfAttention)
block
```
??? done "output"  
    <pre>ResBlock(
      (convs): Sequential(
        (conv_0): ConvBnAct(
          (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): ReLU(inplace=True)
        )
        (conv_1): ConvBnAct(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=64, bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): ReLU(inplace=True)
        )
        (conv_2): ConvBnAct(
          (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (se): SEModule(
          (squeeze): AdaptiveAvgPool2d(output_size=1)
          (excitation): Sequential(
            (fc_reduce): Linear(in_features=128, out_features=8, bias=True)
            (se_act): ReLU(inplace=True)
            (fc_expand): Linear(in_features=8, out_features=128, bias=True)
            (se_gate): Sigmoid()
          )
        )
        (sa): SimpleSelfAttention(
          (conv): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
        )
      )
      (id_conv): Sequential(
        (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
        (id_conv): ConvBnAct(
          (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (act_fn): ReLU(inplace=True)
    )



## Model Constructor.


```python
mc  = ModelConstructor()
mc
```
???+ done "output"  
    <pre>MC constructor
      in_chans: 3, num_classes: 1000
      expansion: 1, groups: 1, dw: False, div_groups: None
      sa: False, se: False
      stem sizes: [3, 32, 32, 64], stride on 0
      body sizes [64, 128, 256, 512]
      layers: [2, 2, 2, 2]




```python

mc.stem
```
??? done "output"  
    <pre>Sequential(
      (conv_0): ConvBnAct(
        (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): ReLU(inplace=True)
      )
      (conv_1): ConvBnAct(
        (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): ReLU(inplace=True)
      )
      (conv_2): ConvBnAct(
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): ReLU(inplace=True)
      )
      (stem_pool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    )




```python

mc.stem_stride_on = 1
mc.stem
```
??? done "output"  
    <pre>Sequential(
      (conv_0): ConvBnAct(
        (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): ReLU(inplace=True)
      )
      (conv_1): ConvBnAct(
        (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): ReLU(inplace=True)
      )
      (conv_2): ConvBnAct(
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): ReLU(inplace=True)
      )
      (stem_pool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    )




```python
mc.bn_1st = False
```


```python
mc.act_fn = nn.LeakyReLU(inplace=True)
```


```python
mc.sa = SimpleSelfAttention
mc.se = SEModule
```


```python

mc.body.l_0
```
??? done "output"  
    <pre>Sequential(
      (bl_0): ResBlock(
        (convs): Sequential(
          (conv_0): ConvBnAct(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (act_fn): LeakyReLU(negative_slope=0.01, inplace=True)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (conv_1): ConvBnAct(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (se): SEModule(
            (squeeze): AdaptiveAvgPool2d(output_size=1)
            (excitation): Sequential(
              (reduce): Linear(in_features=64, out_features=4, bias=True)
              (se_act): ReLU(inplace=True)
              (expand): Linear(in_features=4, out_features=64, bias=True)
              (se_gate): Sigmoid()
            )
          )
        )
        (act_fn): LeakyReLU(negative_slope=0.01, inplace=True)
      )
      (bl_1): ResBlock(
        (convs): Sequential(
          (conv_0): ConvBnAct(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (act_fn): LeakyReLU(negative_slope=0.01, inplace=True)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (conv_1): ConvBnAct(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (se): SEModule(
            (squeeze): AdaptiveAvgPool2d(output_size=1)
            (excitation): Sequential(
              (reduce): Linear(in_features=64, out_features=4, bias=True)
              (se_act): ReLU(inplace=True)
              (expand): Linear(in_features=4, out_features=64, bias=True)
              (se_gate): Sigmoid()
            )
          )
          (sa): SimpleSelfAttention(
            (conv): Conv1d(64, 64, kernel_size=(1,), stride=(1,), bias=False)
          )
        )
        (act_fn): LeakyReLU(negative_slope=0.01, inplace=True)
      )
    )





model_constructor
by ayasyrev
