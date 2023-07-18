# YaResNet.

> Yet Another ResNet model.


```python
from model_constructor.yaresnet import YaBasicBlock, YaBottleneckBlock
```

## YaBasicBlock, YaBottleneckBlock


```python

bl = YaBasicBlock(64, 64)
bl
```
<details> <summary>output</summary>  
    <pre>YaBasicBlock(
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
      (merge): ReLU(inplace=True)
    )</pre>
</details>




```python
pool = partial(nn.AvgPool2d, kernel_size=2, ceil_mode=True)
```


```python

bl = YaBottleneckBlock(64, 128, stride=2, pool=pool, act_fn=nn.LeakyReLU, bn_1st=False)
bl
```
<details> <summary>output</summary>  
    <pre>YaBottleneckBlock(
      (reduce): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (convs): Sequential(
        (conv_0): ConvBnAct(
          (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (act_fn): LeakyReLU(negative_slope=0.01, inplace=True)
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv_1): ConvBnAct(
          (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (act_fn): LeakyReLU(negative_slope=0.01, inplace=True)
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv_2): ConvBnAct(
          (conv): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (id_conv): ConvBnAct(
        (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (merge): LeakyReLU(negative_slope=0.01, inplace=True)
    )</pre>
</details>




```python

bl = YaBottleneckBlock(
    64,
    128,
    expansion=2,
    stride=2,
    pool=pool,
    act_fn=nn.LeakyReLU,
    bn_1st=False,
    groups=4,
)
bl
```
<details> <summary>output</summary>  
    <pre>YaBottleneckBlock(
      (reduce): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (convs): Sequential(
        (conv_0): ConvBnAct(
          (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (act_fn): LeakyReLU(negative_slope=0.01, inplace=True)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv_1): ConvBnAct(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
          (act_fn): LeakyReLU(negative_slope=0.01, inplace=True)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv_2): ConvBnAct(
          (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (id_conv): ConvBnAct(
        (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (merge): LeakyReLU(negative_slope=0.01, inplace=True)
    )</pre>
</details>




```python

bl = YaBottleneckBlock(
    64,
    128,
    stride=2,
    pool=pool,
    act_fn=nn.LeakyReLU,
    bn_1st=False,
    div_groups=4,
)
bl
```
<details> <summary>output</summary>  
    <pre>YaBottleneckBlock(
      (reduce): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (convs): Sequential(
        (conv_0): ConvBnAct(
          (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (act_fn): LeakyReLU(negative_slope=0.01, inplace=True)
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv_1): ConvBnAct(
          (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=8, bias=False)
          (act_fn): LeakyReLU(negative_slope=0.01, inplace=True)
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv_2): ConvBnAct(
          (conv): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (id_conv): ConvBnAct(
        (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (merge): LeakyReLU(negative_slope=0.01, inplace=True)
    )</pre>
</details>




```python

bl = YaBasicBlock(
    64,
    128,
    stride=2,
    pool=pool,
    act_fn=nn.Mish,
    bn_1st=False,
    dw=True,
)
bl
```
<details> <summary>output</summary>  
    <pre>YaBasicBlock(
      (reduce): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (convs): Sequential(
        (conv_0): ConvBnAct(
          (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
          (act_fn): Mish(inplace=True)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv_1): ConvBnAct(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (id_conv): ConvBnAct(
        (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (merge): Mish(inplace=True)
    )</pre>
</details>



### Se, Sa


```python
from model_constructor.layers import SimpleSelfAttention, SEModule
```


```python

bl = YaBottleneckBlock(
    64,
    128,
    stride=2,
    pool=pool,
    act_fn=nn.GELU,
    dw=True,
    se=SEModule,
)
bl
```
<details> <summary>output</summary>  
    <pre>YaBottleneckBlock(
      (reduce): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (convs): Sequential(
        (conv_0): ConvBnAct(
          (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): GELU(approximate='none')
        )
        (conv_1): ConvBnAct(
          (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): GELU(approximate='none')
        )
        (conv_2): ConvBnAct(
          (conv): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
      (id_conv): ConvBnAct(
        (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (merge): GELU(approximate='none')
    )</pre>
</details>




```python

bl = YaBottleneckBlock(
    64,
    128,
    stride=2,
    pool=pool,
    act_fn=nn.LeakyReLU,
    dw=True,
    sa=SimpleSelfAttention,
)
bl
```
<details> <summary>output</summary>  
    <pre>YaBottleneckBlock(
      (reduce): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (convs): Sequential(
        (conv_0): ConvBnAct(
          (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): LeakyReLU(negative_slope=0.01, inplace=True)
        )
        (conv_1): ConvBnAct(
          (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): LeakyReLU(negative_slope=0.01, inplace=True)
        )
        (conv_2): ConvBnAct(
          (conv): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (sa): SimpleSelfAttention(
          (conv): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
        )
      )
      (id_conv): ConvBnAct(
        (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (merge): LeakyReLU(negative_slope=0.01, inplace=True)
    )</pre>
</details>




```python

bl = YaBottleneckBlock(
    64,
    128,
    expansion=4,
    stride=2,
    pool=pool,
    act_fn=nn.LeakyReLU,
    dw=True,
    se=SEModule,
    sa=SimpleSelfAttention)
bl
```
<details> <summary>output</summary>  
    <pre>YaBottleneckBlock(
      (reduce): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (convs): Sequential(
        (conv_0): ConvBnAct(
          (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): LeakyReLU(negative_slope=0.01, inplace=True)
        )
        (conv_1): ConvBnAct(
          (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): LeakyReLU(negative_slope=0.01, inplace=True)
        )
        (conv_2): ConvBnAct(
          (conv): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
      (id_conv): ConvBnAct(
        (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (merge): LeakyReLU(negative_slope=0.01, inplace=True)
    )</pre>
</details>



## YaResNet constructor.


```python
from model_constructor import ModelConstructor
from model_constructor.xresnet import xresnet_stem


yaresnet  = ModelConstructor(
    block=YaBasicBlock,
    make_stem=xresnet_stem,
    stem_sizes=[3, 32, 64, 64],
    name='YaResNet',
)
```


```python
yaresnet
```
<details open> <summary>output</summary>  
    <pre>YaResNet
      in_chans: 3, num_classes: 1000
      expansion: 1, groups: 1, dw: False, div_groups: None
      act_fn: ReLU, sa: False, se: False
      stem sizes: [3, 32, 64, 64], stride on 0
      body sizes [64, 128, 256, 512]
      layers: [2, 2, 2, 2]</pre>
</details>




```python
yaresnet.print_changed_fields()
```
<details open> <summary>output</summary>  
    <pre>Changed fields:
    name: YaResNet
    block: YaBasicBlock
    stem_sizes: [3, 32, 64, 64]
    make_stem: xresnet_stem
    </pre>
</details>


```python

yaresnet.stem
```
<details> <summary>output</summary>  
    <pre>Sequential(
      (conv_0): ConvBnAct(
        (conv): Conv2d(3, 3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): ReLU(inplace=True)
      )
      (conv_1): ConvBnAct(
        (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): ReLU(inplace=True)
      )
      (conv_2): ConvBnAct(
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): ReLU(inplace=True)
      )
      (conv_3): ConvBnAct(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): ReLU(inplace=True)
      )
      (stem_pool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    )</pre>
</details>




```python

yaresnet.body
```
<details> <summary>output</summary>  
    <pre>Sequential(
      (l_0): Sequential(
        (bl_0): YaBasicBlock(
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
          (merge): ReLU(inplace=True)
        )
        (bl_1): YaBasicBlock(
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
          (merge): ReLU(inplace=True)
        )
      )
      (l_1): Sequential(
        (bl_0): YaBasicBlock(
          (reduce): ConvBnAct(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (convs): Sequential(
            (conv_0): ConvBnAct(
              (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1): ConvBnAct(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (id_conv): ConvBnAct(
            (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (merge): ReLU(inplace=True)
        )
        (bl_1): YaBasicBlock(
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
          (merge): ReLU(inplace=True)
        )
      )
      (l_2): Sequential(
        (bl_0): YaBasicBlock(
          (reduce): ConvBnAct(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (convs): Sequential(
            (conv_0): ConvBnAct(
              (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1): ConvBnAct(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (id_conv): ConvBnAct(
            (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (merge): ReLU(inplace=True)
        )
        (bl_1): YaBasicBlock(
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
          (merge): ReLU(inplace=True)
        )
      )
      (l_3): Sequential(
        (bl_0): YaBasicBlock(
          (reduce): ConvBnAct(
            (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (convs): Sequential(
            (conv_0): ConvBnAct(
              (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1): ConvBnAct(
              (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (id_conv): ConvBnAct(
            (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (merge): ReLU(inplace=True)
        )
        (bl_1): YaBasicBlock(
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
          (merge): ReLU(inplace=True)
        )
      )
    )</pre>
</details>




```python

yaresnet.head
```
<details> <summary>output</summary>  
    <pre>Sequential(
      (pool): AdaptiveAvgPool2d(output_size=1)
      (flat): Flatten(start_dim=1, end_dim=-1)
      (fc): Linear(in_features=512, out_features=1000, bias=True)
    )</pre>
</details>



Lots of experiments showed that it worth trying Mish activation function.


```python

yaresnet.act_fn = torch.nn.Mish
yaresnet()
```
<details> <summary>output</summary>  
    <pre>YaResNet(
      block: YaBasicBlock, act_fn: Mish, stem_sizes: [3, 32, 64, 64], make_stem: xresnet_stem
      (stem): Sequential(
        (conv_0): ConvBnAct(
          (conv): Conv2d(3, 3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): Mish(inplace=True)
        )
        (conv_1): ConvBnAct(
          (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): Mish(inplace=True)
        )
        (conv_2): ConvBnAct(
          (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): Mish(inplace=True)
        )
        (conv_3): ConvBnAct(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): Mish(inplace=True)
        )
        (stem_pool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      )
      (body): Sequential(
        (l_0): Sequential(
          (bl_0): YaBasicBlock(
            (convs): Sequential(
              (conv_0): ConvBnAct(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): Mish(inplace=True)
              )
              (conv_1): ConvBnAct(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (merge): Mish(inplace=True)
          )
          (bl_1): YaBasicBlock(
            (convs): Sequential(
              (conv_0): ConvBnAct(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): Mish(inplace=True)
              )
              (conv_1): ConvBnAct(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (merge): Mish(inplace=True)
          )
        )
        (l_1): Sequential(
          (bl_0): YaBasicBlock(
            (reduce): ConvBnAct(
              (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (convs): Sequential(
              (conv_0): ConvBnAct(
                (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): Mish(inplace=True)
              )
              (conv_1): ConvBnAct(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (id_conv): ConvBnAct(
              (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (merge): Mish(inplace=True)
          )
          (bl_1): YaBasicBlock(
            (convs): Sequential(
              (conv_0): ConvBnAct(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): Mish(inplace=True)
              )
              (conv_1): ConvBnAct(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (merge): Mish(inplace=True)
          )
        )
        (l_2): Sequential(
          (bl_0): YaBasicBlock(
            (reduce): ConvBnAct(
              (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (convs): Sequential(
              (conv_0): ConvBnAct(
                (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): Mish(inplace=True)
              )
              (conv_1): ConvBnAct(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (id_conv): ConvBnAct(
              (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (merge): Mish(inplace=True)
          )
          (bl_1): YaBasicBlock(
            (convs): Sequential(
              (conv_0): ConvBnAct(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): Mish(inplace=True)
              )
              (conv_1): ConvBnAct(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (merge): Mish(inplace=True)
          )
        )
        (l_3): Sequential(
          (bl_0): YaBasicBlock(
            (reduce): ConvBnAct(
              (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (convs): Sequential(
              (conv_0): ConvBnAct(
                (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): Mish(inplace=True)
              )
              (conv_1): ConvBnAct(
                (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (id_conv): ConvBnAct(
              (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (merge): Mish(inplace=True)
          )
          (bl_1): YaBasicBlock(
            (convs): Sequential(
              (conv_0): ConvBnAct(
                (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): Mish(inplace=True)
              )
              (conv_1): ConvBnAct(
                (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (merge): Mish(inplace=True)
          )
        )
      )
      (head): Sequential(
        (pool): AdaptiveAvgPool2d(output_size=1)
        (flat): Flatten(start_dim=1, end_dim=-1)
        (fc): Linear(in_features=512, out_features=1000, bias=True)
      )
    )</pre>
</details>




```python
yaresnet.se = SEModule
```


```python

yaresnet.body.l_0.bl_0
```

# YaResnet34, YaResnet50

We has `Resnet34` and `Resnet50` like models predefined, we can impoer it as:  
`from model_constructor.yaresnet import YaResNet34, YaResNet50`  
But lets create it.


```python
class YaResNet(ModelConstructor):
    make_stem: Callable[[ModelCfg], ModSeq] = xresnet_stem
    stem_sizes: list[int] = [32, 64, 64]
    block: type[nn.Module] = YaBasicBlock
    act_fn: type[nn.Module] = nn.Mish
    pool: Optional[Callable[[Any], nn.Module]] = partial(
        nn.AvgPool2d, kernel_size=2, ceil_mode=True
)
```


```python
class YaResNet34(YaResNet):
    layers: list[int] = [3, 4, 6, 3]
```


```python
yaresnet34 = YaResNet34()
yaresnet34
```
<details open> <summary>output</summary>  
    <pre>YaResNet34
      in_chans: 3, num_classes: 1000
      expansion: 1, groups: 1, dw: False, div_groups: None
      act_fn: Mish, sa: False, se: False
      stem sizes: [32, 64, 64], stride on 0
      body sizes [64, 128, 256, 512]
      layers: [3, 4, 6, 3]</pre>
</details>




```python
class YaResNet50(YaResNet34):
    block: type[nn.Module] = YaBottleneckBlock
    block_sizes: list[int] = [256, 512, 1024, 2048]
```


```python
yaresnet50 = YaResNet50()
yaresnet50
```
<details open> <summary>output</summary>  
    <pre>YaResNet50
      in_chans: 3, num_classes: 1000
      expansion: 1, groups: 1, dw: False, div_groups: None
      act_fn: Mish, sa: False, se: False
      stem sizes: [32, 64, 64], stride on 0
      body sizes [256, 512, 1024, 2048]
      layers: [3, 4, 6, 3]</pre>
</details>



model_constructor
by ayasyrev
