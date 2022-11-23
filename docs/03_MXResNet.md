# MXResNet.

> MXResNet model.

MXResNet model, [forum discussion](https://forums.fast.ai/t/how-we-beat-the-5-epoch-imagewoof-leaderboard-score-some-new-techniques-to-consider)

## MXResNet constructor.


```python
mxresnet  = ModelConstructor(
    name='MXResNet',
    stem_sizes=[3, 32, 64, 64],
    act_fn=torch.nn.Mish(),
)
```


```python
mxresnet
```
<details open> <summary>output</summary>  
    <pre>MXResNet constructor
      in_chans: 3, num_classes: 1000
      expansion: 1, groups: 1, dw: False, div_groups: None
      sa: False, se: False
      stem sizes: [3, 32, 64, 64], stride on 0
      body sizes [64, 128, 256, 512]
      layers: [2, 2, 2, 2]<pre>
</details>




```python
mxresnet.block_sizes, mxresnet.layers
```
<details open> <summary>output</summary>  
    <pre>([64, 128, 256, 512], [2, 2, 2, 2])<pre>
</details>




```python

mxresnet.stem
```
<details> <summary>output</summary>  
    <pre>Sequential(
      (conv_0): ConvBnAct(
        (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): Mish()
      )
      (conv_1): ConvBnAct(
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): Mish()
      )
      (conv_2): ConvBnAct(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): Mish()
      )
      (stem_pool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    )<pre>
</details>




```python

mxresnet.body
```
<details> <summary>output</summary>  
    <pre>Sequential(
      (l_0): Sequential(
        (bl_0): ResBlock(
          (convs): Sequential(
            (conv_0): ConvBnAct(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
            )
            (conv_1): ConvBnAct(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (act_fn): Mish()
        )
        (bl_1): ResBlock(
          (convs): Sequential(
            (conv_0): ConvBnAct(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
            )
            (conv_1): ConvBnAct(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (act_fn): Mish()
        )
      )
      (l_1): Sequential(
        (bl_0): ResBlock(
          (convs): Sequential(
            (conv_0): ConvBnAct(
              (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
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
          (act_fn): Mish()
        )
        (bl_1): ResBlock(
          (convs): Sequential(
            (conv_0): ConvBnAct(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
            )
            (conv_1): ConvBnAct(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (act_fn): Mish()
        )
      )
      (l_2): Sequential(
        (bl_0): ResBlock(
          (convs): Sequential(
            (conv_0): ConvBnAct(
              (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
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
          (act_fn): Mish()
        )
        (bl_1): ResBlock(
          (convs): Sequential(
            (conv_0): ConvBnAct(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
            )
            (conv_1): ConvBnAct(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (act_fn): Mish()
        )
      )
      (l_3): Sequential(
        (bl_0): ResBlock(
          (convs): Sequential(
            (conv_0): ConvBnAct(
              (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
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
          (act_fn): Mish()
        )
        (bl_1): ResBlock(
          (convs): Sequential(
            (conv_0): ConvBnAct(
              (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
            )
            (conv_1): ConvBnAct(
              (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (act_fn): Mish()
        )
      )
    )<pre>
</details>




```python

mxresnet.body
```
<details> <summary>output</summary>  
    <pre>Sequential(
      (l_0): Sequential(
        (bl_0): ResBlock(
          (convs): Sequential(
            (conv_0): ConvBnAct(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
            )
            (conv_1): ConvBnAct(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (act_fn): Mish()
        )
        (bl_1): ResBlock(
          (convs): Sequential(
            (conv_0): ConvBnAct(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
            )
            (conv_1): ConvBnAct(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (act_fn): Mish()
        )
      )
      (l_1): Sequential(
        (bl_0): ResBlock(
          (convs): Sequential(
            (conv_0): ConvBnAct(
              (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
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
          (act_fn): Mish()
        )
        (bl_1): ResBlock(
          (convs): Sequential(
            (conv_0): ConvBnAct(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
            )
            (conv_1): ConvBnAct(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (act_fn): Mish()
        )
      )
      (l_2): Sequential(
        (bl_0): ResBlock(
          (convs): Sequential(
            (conv_0): ConvBnAct(
              (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
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
          (act_fn): Mish()
        )
        (bl_1): ResBlock(
          (convs): Sequential(
            (conv_0): ConvBnAct(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
            )
            (conv_1): ConvBnAct(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (act_fn): Mish()
        )
      )
      (l_3): Sequential(
        (bl_0): ResBlock(
          (convs): Sequential(
            (conv_0): ConvBnAct(
              (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
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
          (act_fn): Mish()
        )
        (bl_1): ResBlock(
          (convs): Sequential(
            (conv_0): ConvBnAct(
              (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
            )
            (conv_1): ConvBnAct(
              (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (act_fn): Mish()
        )
      )
    )<pre>
</details>




```python

mxresnet.head
```
<details> <summary>output</summary>  
    <pre>Sequential(
      (pool): AdaptiveAvgPool2d(output_size=1)
      (flat): Flatten(start_dim=1, end_dim=-1)
      (fc): Linear(in_features=512, out_features=1000, bias=True)
    )<pre>
</details>



## MxResNet constructors

Lets create constructor class for MxResnet.


```python
class MXResnet34(ModelConstructor):
    name: str = "MXResnet34"
    expansion: int = 1
    layers: List[int] = [3, 4, 6, 3]
    stem_sizes: List[int] = [3, 32, 64, 64]
    act_fn: torch.nn.Module = torch.nn.Mish()
```

MXResnet50 inherit from MXResnet34.


```python
class MXResnet50(MXResnet34):
    name: str = "MXResnet50"
    expansion: int = 4
```

Now we can create constructor from class adn change model parameters during initialization or after.


```python
mc = MXResnet34(num_classes=10)
mc
```
<details open> <summary>output</summary>  
    <pre>MXResnet34 constructor
      in_chans: 3, num_classes: 10
      expansion: 1, groups: 1, dw: False, div_groups: None
      sa: False, se: False
      stem sizes: [3, 32, 64, 64], stride on 0
      body sizes [64, 128, 256, 512]
      layers: [3, 4, 6, 3]<pre>
</details>




```python
mc = MXResnet50()
mc
```
<details open> <summary>output</summary>  
    <pre>MXResnet50 constructor
      in_chans: 3, num_classes: 1000
      expansion: 4, groups: 1, dw: False, div_groups: None
      sa: False, se: False
      stem sizes: [3, 32, 64, 64], stride on 0
      body sizes [64, 128, 256, 512]
      layers: [3, 4, 6, 3]<pre>
</details>



To create model - call model constructor object.


```python
model = mc()
```



model_constructor
by ayasyrev
