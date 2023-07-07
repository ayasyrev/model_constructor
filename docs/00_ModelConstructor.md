# Model constructor.

> Create and tune pytorch model.

# ModelConstructor and ModelCfg

Main part of model_constructor - ModelConstructor and ModelCfg.


```python
from model_constructor.model_constructor import ModelCfg, ModelConstructor
```

ModelCfg is base for model config, ModelConstructor got all we need to create model. And it subclassed from ModelCfg all config plus methods for create model.

So we can create config and than constructor or model from it.  
Or create constructor or model from MOdelConstructor.

Lets create base config.


```python
cfg = ModelCfg()
cfg.print_cfg()
```
<details open> <summary>output</summary>  
    <pre>ModelCfg
      in_chans: 3, num_classes: 1000
      expansion: 1, groups: 1, dw: False, div_groups: None
      act_fn: ReLU, sa: False, se: False
      stem sizes: [64], stride on 0
      body sizes [64, 128, 256, 512]
      layers: [2, 2, 2, 2]
    </pre>
</details>

Now we can create model directly from config or throw creating constructor.


```python
model = ModelConstructor.create_model(cfg)
```


```python
model_constructor = ModelConstructor.from_cfg(cfg)
model = model_constructor()
```

# Instantiate config or constructor.

When initialize config or constructor, we can use string interpolations of nn.Modules instead of class. By default we search at torch.nn.


```python
cfg = ModelCfg(act_fn="torch.nn.Mish")
print(cfg.act_fn)
```
<details open> <summary>output</summary>  
    <pre>class 'torch.nn.modules.activation.Mish'>
    </pre>
</details>


```python
cfg = ModelCfg(act_fn="nn.SELU")
print(cfg.act_fn)
```
<details open> <summary>output</summary>  
    <pre>class 'torch.nn.modules.activation.SELU'
    </pre>
</details>


```python
cfg = ModelCfg(
    act_fn="Mish",
    block="model_constructor.yaresnet.YaBasicBlock",
)
print(cfg.act_fn)
print(cfg.block)
```
<details open> <summary>output</summary>  
    <pre>class 'torch.nn.modules.activation.Mish'

    class 'model_constructor.yaresnet.YaBasicBlock'
    </pre>
</details>

# Stem, Body, Head.

By default constructor create `nn.Sequential` model with `stem`, `body` and `head`. We can check it at constructor stage.


```python
model = ModelConstructor.create_model()
```


```python
for name, mod in model.named_children():
    print(name)
```
<details open> <summary>output</summary>  
    <pre>stem
    body
    head
    </pre>
</details>

Constructor create `stem`, `body` and `head` with `make_stem`, `make_body` and `make_head` methods. They are defined separately as functions with ModelCfg as argument.  
And we can change it on the fly as:  
`mc.make_stem = custom_stem`  
`mc.make_body = custom_body`  
`mc.make_head = custom_head`  
Or at initializations as:  
`mc = ModelConstructor(make_stem=custom_stem)`



```python
from model_constructor.model_constructor import make_stem, make_body, make_head, make_layer
```

## Stem


```python

stem = make_stem(cfg)
stem
```
<details> <summary>output</summary>  
    </pre>Sequential(
      (conv_1): ConvBnAct(
        (conv): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): Mish(inplace=True)
      )
      (stem_pool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    )</pre>
</details>



### Layer

`make_layer` need `layer_num` argument - number of layer.


```python

layer = make_layer(cfg, layer_num=0)
layer
```
<details> <summary>output</summary>  
    </pre>Sequential(
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
    )</pre>
</details>



### Body

`make_body` needs `cfg.make_layer` initialized. As default - `make_layer`.  It can be changed.


```python

# cfg.make_layer = make_layer
# body = make_body(cfg)
# body
```

## Head


```python

head = make_head(cfg)
head
```
<details> <summary>output</summary>  
    </pre>Sequential(
      (pool): AdaptiveAvgPool2d(output_size=1)
      (flat): Flatten(start_dim=1, end_dim=-1)
      (fc): Linear(in_features=512, out_features=1000, bias=True)
    )</pre>
</details>



## Model Constructor.


```python
mc  = ModelConstructor()
mc
```
<details open> <summary>output</summary>  
    <pre>ModelConstructor
      in_chans: 3, num_classes: 1000
      expansion: 1, groups: 1, dw: False, div_groups: None
      act_fn: ReLU, sa: False, se: False
      stem sizes: [64], stride on 0
      body sizes [64, 128, 256, 512]
      layers: [2, 2, 2, 2]</pre>
</details>




```python

mc.stem
```
<details> <summary>output</summary>  
    </pre>Sequential(
      (conv_1): ConvBnAct(
        (conv): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): ReLU(inplace=True)
      )
      (stem_pool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    )</pre>
</details>




```python
mc.bn_1st = False
```


```python
mc.act_fn = nn.LeakyReLU
```


```python
mc.sa = SimpleSelfAttention
mc.se = SEModule
```


```python

mc.body.l_0
```
<details> <summary>output</summary>  
    </pre>Sequential(
      (bl_0): BasicBlock(
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
      (bl_1): BasicBlock(
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
    )</pre>
</details>



model_constructor
by ayasyrev
