# Model constructor.

> Create and tune pytorch model.

## ResBlock


```python
block = ResBlock(1, 64, 64)
block
```
<details open> <summary>output</summary>  
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
    )<pre>
</details>




```python
block = ResBlock(4, 64, 64, dw=True)
block
```
<details open> <summary>output</summary>  
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
    )<pre>
</details>




```python
block = ResBlock(4, 64, 64,groups=4)
block
```
<details open> <summary>output</summary>  
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
    )<pre>
</details>




```python

block = ResBlock(2, 64, 64,act_fn=nn.LeakyReLU(), bn_1st=False)
block
```
<details> <summary>output</summary>  
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
    )<pre>
</details>




```python

lock = ResBlock(2, 32, 64, dw=True)
block
```
<details> <summary>output</summary>  
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
    )<pre>
</details>




```python
pool = nn.AvgPool2d(2, ceil_mode=True)
```


```python

block = ResBlock(2, 32, 64, stride=2, dw=True, pool=pool)
block
```
<details> <summary>output</summary>  
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
    )<pre>
</details>




```python
from model_constructor.layers import SEModule, SimpleSelfAttention
```


```python

block = ResBlock(2, 32, 64, stride=2, dw=True, pool=pool, se=SEModule)
block
```
<details> <summary>output</summary>  
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
    )<pre>
</details>




```python

block = ResBlock(2, 32, 64, stride=2, dw=True, pool=pool, se=SEModule, sa=SimpleSelfAttention)
block
```
<details> <summary>output</summary>  
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
    )<pre>
</details>



## Stem, Body, Layer, Head

Helper functions to create stem, body and head of model from config.  
Returns `nn.Sequential`.  


```python
from model_constructor.model_constructor import ModelCfg, make_stem, make_body, make_layer, make_head
from rich import print
```


```python
cfg = ModelCfg()
print(cfg.extra_repr())
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">name: MC
in_chans: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>
num_classes: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1000</span>
block: <span style="font-weight: bold">&lt;</span><span style="color: #ff00ff; text-decoration-color: #ff00ff; font-weight: bold">class</span><span style="color: #000000; text-decoration-color: #000000"> </span><span style="color: #008000; text-decoration-color: #008000">'model_constructor.model_constructor.ResBlock'</span><span style="font-weight: bold">&gt;</span>
conv_layer: <span style="font-weight: bold">&lt;</span><span style="color: #ff00ff; text-decoration-color: #ff00ff; font-weight: bold">class</span><span style="color: #000000; text-decoration-color: #000000"> </span><span style="color: #008000; text-decoration-color: #008000">'model_constructor.layers.ConvBnAct'</span><span style="font-weight: bold">&gt;</span>
block_sizes: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">64</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">128</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">256</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">512</span><span style="font-weight: bold">]</span>
layers: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span><span style="font-weight: bold">]</span>
norm: <span style="font-weight: bold">&lt;</span><span style="color: #ff00ff; text-decoration-color: #ff00ff; font-weight: bold">class</span><span style="color: #000000; text-decoration-color: #000000"> </span><span style="color: #008000; text-decoration-color: #008000">'torch.nn.modules.batchnorm.BatchNorm2d'</span><span style="font-weight: bold">&gt;</span>
act_fn: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">ReLU</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">inplace</span>=<span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span><span style="font-weight: bold">)</span>
pool: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">AvgPool2d</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">kernel_size</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #808000; text-decoration-color: #808000">stride</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #808000; text-decoration-color: #808000">padding</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">)</span>
expansion: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>
groups: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>
dw: <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>
sa: <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>
se: <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>
bn_1st: <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span>
zero_bn: <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span>
stem_stride_on: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>
stem_sizes: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">64</span><span style="font-weight: bold">]</span>
stem_pool: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">MaxPool2d</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">kernel_size</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>, <span style="color: #808000; text-decoration-color: #808000">stride</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #808000; text-decoration-color: #808000">padding</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>, <span style="color: #808000; text-decoration-color: #808000">dilation</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>, <span style="color: #808000; text-decoration-color: #808000">ceil_mode</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span><span style="font-weight: bold">)</span>
stem_bn_end: <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>

</pre>



### Stem


```python

stem = make_stem(cfg)
stem
```
<details> <summary>output</summary>  
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
    )<pre>
</details>



### Layer

`make_layer` need `layer_num` argument - number of layer.


```python

layer = make_layer(cfg, layer_num=0)
layer
```
<details> <summary>output</summary>  
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
    )<pre>
</details>



### Body

`make_body` needs `cfg._make_layer` initialized. As default - `make_layer`.


```python

cfg.make_layer = make_layer
body = make_body(cfg)
body
```
<details> <summary>output</summary>  
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
    )<pre>
</details>



## Head


```python

head = make_head(cfg)
head
```
<details> <summary>output</summary>  
    <pre>Sequential(
      (pool): AdaptiveAvgPool2d(output_size=1)
      (flat): Flatten(start_dim=1, end_dim=-1)
      (fc): Linear(in_features=512, out_features=1000, bias=True)
    )<pre>
</details>



## Model Constructor.


```python
mc  = ModelConstructor()
mc
```
<details open> <summary>output</summary>  
    <pre>MC constructor
      in_chans: 3, num_classes: 1000
      expansion: 1, groups: 1, dw: False, div_groups: None
      sa: False, se: False
      stem sizes: [3, 32, 32, 64], stride on 0
      body sizes [64, 128, 256, 512]
      layers: [2, 2, 2, 2]<pre>
</details>




```python

mc.stem
```
<details> <summary>output</summary>  
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
    )<pre>
</details>




```python

mc.stem_stride_on = 1
mc.stem
```
<details> <summary>output</summary>  
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
    )<pre>
</details>




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
<details> <summary>output</summary>  
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
    )<pre>
</details>





model_constructor
by ayasyrev
