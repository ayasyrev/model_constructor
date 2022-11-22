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
??? done "output"  
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

lock = ResBlock(2, 32, 64, dw=True)
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
            (reduce): Linear(in_features=128, out_features=8, bias=True)
            (se_act): ReLU(inplace=True)
            (expand): Linear(in_features=8, out_features=128, bias=True)
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
            (reduce): Linear(in_features=128, out_features=8, bias=True)
            (se_act): ReLU(inplace=True)
            (expand): Linear(in_features=8, out_features=128, bias=True)
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



## Stem, Body, Layer, Head

Helper functions to create stem, body and head of model from config.  
Returns `nn.Sequential`.  


```python
from model_constructor.model_constructor import CfgMC, make_stem, make_body, make_layer, make_head
from rich import print
```


```python
cfg = CfgMC()
print(cfg)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">CfgMC</span><span style="font-weight: bold">(</span>
    <span style="color: #808000; text-decoration-color: #808000">name</span>=<span style="color: #008000; text-decoration-color: #008000">'MC'</span>,
    <span style="color: #808000; text-decoration-color: #808000">in_chans</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>,
    <span style="color: #808000; text-decoration-color: #808000">num_classes</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1000</span>,
    <span style="color: #808000; text-decoration-color: #808000">block</span>=<span style="font-weight: bold">&lt;</span><span style="color: #ff00ff; text-decoration-color: #ff00ff; font-weight: bold">class</span><span style="color: #000000; text-decoration-color: #000000"> </span><span style="color: #008000; text-decoration-color: #008000">'model_constructor.model_constructor.ResBlock'</span><span style="font-weight: bold">&gt;</span>,
    <span style="color: #808000; text-decoration-color: #808000">conv_layer</span>=<span style="font-weight: bold">&lt;</span><span style="color: #ff00ff; text-decoration-color: #ff00ff; font-weight: bold">class</span><span style="color: #000000; text-decoration-color: #000000"> </span><span style="color: #008000; text-decoration-color: #008000">'model_constructor.layers.ConvBnAct'</span><span style="font-weight: bold">&gt;</span>,
    <span style="color: #808000; text-decoration-color: #808000">block_sizes</span>=<span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">64</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">128</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">256</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">512</span><span style="font-weight: bold">]</span>,
    <span style="color: #808000; text-decoration-color: #808000">layers</span>=<span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span><span style="font-weight: bold">]</span>,
    <span style="color: #808000; text-decoration-color: #808000">norm</span>=<span style="font-weight: bold">&lt;</span><span style="color: #ff00ff; text-decoration-color: #ff00ff; font-weight: bold">class</span><span style="color: #000000; text-decoration-color: #000000"> </span><span style="color: #008000; text-decoration-color: #008000">'torch.nn.modules.batchnorm.BatchNorm2d'</span><span style="font-weight: bold">&gt;</span>,
    <span style="color: #808000; text-decoration-color: #808000">act_fn</span>=<span style="color: #800080; text-decoration-color: #800080; font-weight: bold">ReLU</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">inplace</span>=<span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span><span style="font-weight: bold">)</span>,
    <span style="color: #808000; text-decoration-color: #808000">pool</span>=<span style="color: #800080; text-decoration-color: #800080; font-weight: bold">AvgPool2d</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">kernel_size</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #808000; text-decoration-color: #808000">stride</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #808000; text-decoration-color: #808000">padding</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">)</span>,
    <span style="color: #808000; text-decoration-color: #808000">expansion</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>,
    <span style="color: #808000; text-decoration-color: #808000">groups</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>,
    <span style="color: #808000; text-decoration-color: #808000">dw</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,
    <span style="color: #808000; text-decoration-color: #808000">div_groups</span>=<span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>,
    <span style="color: #808000; text-decoration-color: #808000">sa</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,
    <span style="color: #808000; text-decoration-color: #808000">se</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,
    <span style="color: #808000; text-decoration-color: #808000">bn_1st</span>=<span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span>,
    <span style="color: #808000; text-decoration-color: #808000">zero_bn</span>=<span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span>,
    <span style="color: #808000; text-decoration-color: #808000">stem_stride_on</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>,
    <span style="color: #808000; text-decoration-color: #808000">stem_sizes</span>=<span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">64</span><span style="font-weight: bold">]</span>,
    <span style="color: #808000; text-decoration-color: #808000">stem_pool</span>=<span style="color: #800080; text-decoration-color: #800080; font-weight: bold">MaxPool2d</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">kernel_size</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>, <span style="color: #808000; text-decoration-color: #808000">stride</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #808000; text-decoration-color: #808000">padding</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>, <span style="color: #808000; text-decoration-color: #808000">dilation</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>, <span style="color: #808000; text-decoration-color: #808000">ceil_mode</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span><span style="font-weight: bold">)</span>,
    <span style="color: #808000; text-decoration-color: #808000">stem_bn_end</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>
<span style="font-weight: bold">)</span>
</pre>



### Stem


```python

stem = make_stem(cfg)
stem
```
??? done "output"  
    <pre>Sequential(
      (conv_0): ConvBnAct(
        (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): ReLU(inplace=True)
      )
      (conv_1): ConvBnAct(
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): ReLU(inplace=True)
      )
      (stem_pool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    )



### Layer

`make_layer` need `layer_num` argument - number of layer.


```python

layer = make_layer(cfg, layer_num=0)
layer
```
??? done "output"  
    <pre>Sequential(
      (bl_0): ResBlock(
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
      (bl_1): ResBlock(
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
    )



### Body

`make_body` needs `cfg._make_layer` initialized. As default - `make_layer`.


```python

cfg._make_layer = make_layer
body = make_body(cfg)
body
```
??? done "output"  
    <pre>Sequential(
      (l_0): Sequential(
        (bl_0): ResBlock(
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
        (bl_1): ResBlock(
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
      )
      (l_1): Sequential(
        (bl_0): ResBlock(
          (convs): Sequential(
            (conv_0): ConvBnAct(
              (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1): ConvBnAct(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
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
        (bl_1): ResBlock(
          (convs): Sequential(
            (conv_0): ConvBnAct(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1): ConvBnAct(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (act_fn): ReLU(inplace=True)
        )
      )
      (l_2): Sequential(
        (bl_0): ResBlock(
          (convs): Sequential(
            (conv_0): ConvBnAct(
              (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1): ConvBnAct(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (id_conv): Sequential(
            (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
            (id_conv): ConvBnAct(
              (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (act_fn): ReLU(inplace=True)
        )
        (bl_1): ResBlock(
          (convs): Sequential(
            (conv_0): ConvBnAct(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1): ConvBnAct(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (act_fn): ReLU(inplace=True)
        )
      )
      (l_3): Sequential(
        (bl_0): ResBlock(
          (convs): Sequential(
            (conv_0): ConvBnAct(
              (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1): ConvBnAct(
              (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (id_conv): Sequential(
            (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
            (id_conv): ConvBnAct(
              (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (act_fn): ReLU(inplace=True)
        )
        (bl_1): ResBlock(
          (convs): Sequential(
            (conv_0): ConvBnAct(
              (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1): ConvBnAct(
              (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (act_fn): ReLU(inplace=True)
        )
      )
    )



## Head


```python

head = make_head(cfg)
head
```
??? done "output"  
    <pre>Sequential(
      (pool): AdaptiveAvgPool2d(output_size=1)
      (flat): Flatten(start_dim=1, end_dim=-1)
      (fc): Linear(in_features=512, out_features=1000, bias=True)
    )



## Model Constructor.


```python
mc  = ModelConstructor()
mc.print_cfg()
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