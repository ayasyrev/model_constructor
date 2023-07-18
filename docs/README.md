# model_constructor

> Constructor to create pytorch model. 

## Install

`pip install model-constructor`

Or install from repo:

`pip install git+https://github.com/ayasyrev/model_constructor.git`

## How to use

First import constructor class, then create model constructor object.

Now you can change every part of model.


```python
from model_constructor import ModelConstructor
```


```python
mc = ModelConstructor()
```

Check base parameters:


```python
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



Check all parameters with `print_cfg` method:


```python
mc.print_cfg()
```
<details open> <summary>output</summary>  
    <pre>ModelConstructor(
      in_chans=3
      num_classes=1000
      block='BasicBlock'
      conv_layer='ConvBnAct'
      block_sizes=[64, 128, 256, 512]
      layers=[2, 2, 2, 2]
      norm='BatchNorm2d'
      act_fn='ReLU'
      expansion=1
      groups=1
      bn_1st=True
      zero_bn=True
      stem_sizes=[64]
      stem_pool="MaxPool2d {'kernel_size': 3, 'stride': 2, 'padding': 1}"
      init_cnn='init_cnn'
      make_stem='make_stem'
      make_layer='make_layer'
      make_body='make_body'
      make_head='make_head')
    </pre>
</details>

Now we have model constructor, default setting as resnet18. And we can get model after call it.


```python

model = mc()
model
```
<details> <summary>output</summary>  
    <pre>ModelConstructor(
      (stem): Sequential(
        (conv_1): ConvBnAct(
          (conv): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): ReLU(inplace=True)
        )
        (stem_pool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      )
      (body): Sequential(
        (l_0): Sequential(
          (bl_0): BasicBlock(
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
          (bl_1): BasicBlock(
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
          (bl_0): BasicBlock(
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
              (id_conv): ConvBnAct(
                (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act_fn): ReLU(inplace=True)
          )
          (bl_1): BasicBlock(
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
          (bl_0): BasicBlock(
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
              (id_conv): ConvBnAct(
                (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act_fn): ReLU(inplace=True)
          )
          (bl_1): BasicBlock(
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
          (bl_0): BasicBlock(
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
              (id_conv): ConvBnAct(
                (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act_fn): ReLU(inplace=True)
          )
          (bl_1): BasicBlock(
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
      (head): Sequential(
        (pool): AdaptiveAvgPool2d(output_size=1)
        (flat): Flatten(start_dim=1, end_dim=-1)
        (fc): Linear(in_features=512, out_features=1000, bias=True)
      )
    )</pre>
</details>



If you want to change model, just change constructor parameters.  
Lets create resnet50.


```python
mc.expansion = 4
mc.layers = [3,4,6,3]
```

We can check, what we changed (compare to default constructor).


```python
mc.changed_fields
```
<details open> <summary>output</summary>  
    <pre>{'layers': [3, 4, 6, 3], 'expansion': 4}</pre>
</details>




```python
mc.print_changed_fields()
```
<details open> <summary>output</summary>  
    <pre>Changed fields:
    layers: [3, 4, 6, 3]
    expansion: 4
    </pre>
</details>

We can compare changed with defaults.


```python
mc.print_changed_fields(show_default=True)
```
<details open> <summary>output</summary>  
    <pre>Changed fields:
    layers: [3, 4, 6, 3] | [2, 2, 2, 2]
    expansion: 4 | 1
    </pre>
</details>

Now we can look at model parts - stem, body, head.  


```python

mc.body
```
<details> <summary>output</summary>  
    <pre>Sequential(
      (l_0): Sequential(
        (bl_0): BasicBlock(
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
        (bl_1): BasicBlock(
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
        (bl_2): BasicBlock(
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
        (bl_0): BasicBlock(
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
            (id_conv): ConvBnAct(
              (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (act_fn): ReLU(inplace=True)
        )
        (bl_1): BasicBlock(
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
        (bl_2): BasicBlock(
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
        (bl_3): BasicBlock(
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
        (bl_0): BasicBlock(
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
            (id_conv): ConvBnAct(
              (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (act_fn): ReLU(inplace=True)
        )
        (bl_1): BasicBlock(
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
        (bl_2): BasicBlock(
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
        (bl_3): BasicBlock(
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
        (bl_4): BasicBlock(
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
        (bl_5): BasicBlock(
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
        (bl_0): BasicBlock(
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
            (id_conv): ConvBnAct(
              (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (act_fn): ReLU(inplace=True)
        )
        (bl_1): BasicBlock(
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
        (bl_2): BasicBlock(
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
    )</pre>
</details>



## Create constructor from config.

Alternative we can create config first and than create constructor from it. 


```python
from model_constructor import ModelCfg
```


```python
cfg = ModelCfg(
    num_classes=10,
    act_fn=nn.Mish,
)
print(cfg)
```
<details open> <summary>output</summary>  
    <pre>ModelCfg(
      in_chans=3
      num_classes=10
      block='BasicBlock'
      conv_layer='ConvBnAct'
      block_sizes=[64, 128, 256, 512]
      layers=[2, 2, 2, 2]
      norm='BatchNorm2d'
      act_fn='Mish'
      expansion=1
      groups=1
      bn_1st=True
      zero_bn=True
      stem_sizes=[64]
      stem_pool="MaxPool2d {'kernel_size': 3, 'stride': 2, 'padding': 1}")
    </pre>
</details>

When creating config or constructor we can use string annotation for nn.Modules - it useful when creating model from config files.


```python
cfg = ModelCfg(
    num_classes=10,
    act_fn="nn.SELU",
)
print(cfg.act_fn)
```
<details open> <summary>output</summary>  
    <pre>class 'torch.nn.modules.activation.SELU'</pre>
</details>

Now we can create constructor from config:


```python
mc = ModelConstructor.from_cfg(cfg)
mc
```
<details open> <summary>output</summary>  
    <pre>ModelConstructor
      in_chans: 3, num_classes: 10
      expansion: 1, groups: 1, dw: False, div_groups: None
      act_fn: SELU, sa: <class 'model_constructor.layers.SimpleSelfAttention'>, se: SEModule
      stem sizes: [64], stride on 0
      body sizes [64, 128, 256, 512]
      layers: [2, 2, 2, 2]</pre>
</details>



## More modification.

Main purpose of this module - fast and easy modify model.
And here is the link to more modification to beat Imagenette leaderboard with add MaxBlurPool and modification to ResBlock [notebook](https://github.com/ayasyrev/imagenette_experiments/blob/master/ResnetTrick_create_model_fit.ipynb)  

But now lets create model as mxresnet50 from [fastai forums tread](https://forums.fast.ai/t/how-we-beat-the-5-epoch-imagewoof-leaderboard-score-some-new-techniques-to-consider)  


Lets create mxresnet constructor.


```python
mc = ModelConstructor(name='MxResNet')
```

Then lets modify stem.


```python
from model_constructor.xresnet import xresnet_stem
```


```python
mc.make_stem = xresnet_stem
mc.stem_sizes = [3,32,64,64]
```

Now lets change activation function to Mish.
Here is link to [forum discussion](https://forums.fast.ai/t/meet-mish-new-activation-function-possible-successor-to-relu)    
We'v got Mish is in model_constructor.activations, but from pytorch 1.9 take it from torch:


```python
from torch.nn import Mish
```


```python
mc.act_fn = Mish
```


```python
mc
```
<details open> <summary>output</summary>  
    <pre>MxResNet
      in_chans: 3, num_classes: 1000
      expansion: 1, groups: 1, dw: False, div_groups: None
      act_fn: Mish, sa: False, se: False
      stem sizes: [3, 32, 64, 64], stride on 0
      body sizes [64, 128, 256, 512]
      layers: [2, 2, 2, 2]</pre>
</details>




```python
mc.print_changed_fields()
```
<details open> <summary>output</summary>  
    <pre>Changed fields:
    name: MxResNet
    act_fn: Mish
    stem_sizes: [3, 32, 64, 64]
    make_stem: xresnet_stem
    </pre>
</details>

Here is model:  


```python

mc()
```
<details> <summary>output</summary>  
    <pre>MxResNet(
      act_fn: Mish, stem_sizes: [3, 32, 64, 64], make_stem: xresnet_stem
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
          (bl_0): BasicBlock(
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
            (act_fn): Mish(inplace=True)
          )
          (bl_1): BasicBlock(
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
            (act_fn): Mish(inplace=True)
          )
        )
        (l_1): Sequential(
          (bl_0): BasicBlock(
            (convs): Sequential(
              (conv_0): ConvBnAct(
                (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): Mish(inplace=True)
              )
              (conv_1): ConvBnAct(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (id_conv): Sequential(
              (id_conv): ConvBnAct(
                (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act_fn): Mish(inplace=True)
          )
          (bl_1): BasicBlock(
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
            (act_fn): Mish(inplace=True)
          )
        )
        (l_2): Sequential(
          (bl_0): BasicBlock(
            (convs): Sequential(
              (conv_0): ConvBnAct(
                (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): Mish(inplace=True)
              )
              (conv_1): ConvBnAct(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (id_conv): Sequential(
              (id_conv): ConvBnAct(
                (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act_fn): Mish(inplace=True)
          )
          (bl_1): BasicBlock(
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
            (act_fn): Mish(inplace=True)
          )
        )
        (l_3): Sequential(
          (bl_0): BasicBlock(
            (convs): Sequential(
              (conv_0): ConvBnAct(
                (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): Mish(inplace=True)
              )
              (conv_1): ConvBnAct(
                (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (id_conv): Sequential(
              (id_conv): ConvBnAct(
                (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act_fn): Mish(inplace=True)
          )
          (bl_1): BasicBlock(
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
            (act_fn): Mish(inplace=True)
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



## MXResNet50

Now lets make MxResNet50


```python
mc.expansion = 4
mc.layers = [3,4,6,3]
mc.name = "mxresnet50"
```


```python
mc.print_changed_fields()
```
<details open> <summary>output</summary>  
    <pre>Changed fields:
    name: mxresnet50
    layers: [3, 4, 6, 3]
    act_fn: Mish
    expansion: 4
    stem_sizes: [3, 32, 64, 64]
    make_stem: xresnet_stem
    </pre>
</details>

Now we have mxresnet50 constructor.  
We can inspect every parts of it.  
And after call it we got model.


```python
mc
```
<details open> <summary>output</summary>  
    <pre>mxresnet50
      in_chans: 3, num_classes: 1000
      expansion: 4, groups: 1, dw: False, div_groups: None
      act_fn: Mish, sa: False, se: False
      stem sizes: [3, 32, 64, 64], stride on 0
      body sizes [64, 128, 256, 512]
      layers: [3, 4, 6, 3]</pre>
</details>




```python

mc.stem.conv_1
```
<details> <summary>output</summary>  
    <pre>ConvBnAct(
      (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act_fn): Mish(inplace=True)
    )</pre>
</details>




```python

mc.body.l_0.bl_0
```
<details> <summary>output</summary>  
    <pre>BasicBlock(
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
      (act_fn): Mish(inplace=True)
    )</pre>
</details>



We can get model direct way:


```python
mc = ModelConstructor(
    name="MxResNet",
    act_fn=Mish,
    layers=[3,4,6,3],
    expansion=4,
    make_stem=xresnet_stem,
    stem_sizes=[32,64,64]
)
model = mc()
```

Another way:


```python
model = ModelConstructor.create_model(
    name="MxResNet",
    act_fn=Mish,
    layers=[3,4,6,3],
    expansion=4,
    make_stem=xresnet_stem,
    stem_sizes=[32,64,64]
)
```

## YaResNet

Now lets change Resblock to YaResBlock (Yet another ResNet, former NewResBlock) is in lib from version 0.1.0


```python
from model_constructor.yaresnet import YaBasicBlock
```


```python
mc = ModelConstructor(name="YaResNet")
mc.block = YaBasicBlock
```

Or in one line:


```python
mc = ModelConstructor(name="YaResNet", block=YaBasicBlock)
```

That all. Now we have YaResNet constructor


```python

mc.print_cfg()
```
<details> <summary>output</summary>  
    <pre>ModelConstructor(
      name='YaResNet'
      in_chans=3
      num_classes=1000
      block='YaBasicBlock'
      conv_layer='ConvBnAct'
      block_sizes=[64, 128, 256, 512]
      layers=[2, 2, 2, 2]
      norm='BatchNorm2d'
      act_fn='ReLU'
      expansion=1
      groups=1
      bn_1st=True
      zero_bn=True
      stem_sizes=[64]
      stem_pool="MaxPool2d {'kernel_size': 3, 'stride': 2, 'padding': 1}"
      init_cnn='init_cnn'
      make_stem='make_stem'
      make_layer='make_layer'
      make_body='make_body'
      make_head='make_head')
    </pre>
</details>

Let see what we have.


```python

mc.body.l_1.bl_0
```
<details> <summary>output</summary>  
    <pre>YaBasicBlock(
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
    )</pre>
</details>



Lets create `xResnet34` like model constructor:


```python
from typing import Callable

from model_constructor.helpers import ModSeq


class YaResnet34(ModelConstructor):
    block: type[nn.Module] = YaBasicBlock
    layers: list[int] = [3, 4, 6, 3]
    make_stem: Callable[[ModelCfg], ModSeq] = xresnet_stem
```


```python
mc = YaResnet34()
mc.print_cfg()
```
<details open> <summary>output</summary>  
    <pre>YaResnet34(
      in_chans=3
      num_classes=1000
      block='YaBasicBlock'
      conv_layer='ConvBnAct'
      block_sizes=[64, 128, 256, 512]
      layers=[3, 4, 6, 3]
      norm='BatchNorm2d'
      act_fn='ReLU'
      expansion=1
      groups=1
      bn_1st=True
      zero_bn=True
      stem_sizes=[64]
      stem_pool="MaxPool2d {'kernel_size': 3, 'stride': 2, 'padding': 1}"
      init_cnn='init_cnn'
      make_stem='xresnet_stem'
      make_layer='make_layer'
      make_body='make_body'
      make_head='make_head')
    </pre>
</details>

And `xResnet50` like model can be inherited from `YaResnet34`:


```python
class YaResnet50(YaResnet34):
    expansion: int = 4
```


```python
mc = YaResnet50()
mc
```
<details open> <summary>output</summary>  
    <pre>YaResnet50
      in_chans: 3, num_classes: 1000
      expansion: 4, groups: 1, dw: False, div_groups: None
      act_fn: ReLU, sa: False, se: False
      stem sizes: [64], stride on 0
      body sizes [64, 128, 256, 512]
      layers: [3, 4, 6, 3]</pre>
</details>


