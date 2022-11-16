# MXResNet.

> MXResNet model.

MXResNet model, [forum discussion](https://forums.fast.ai/t/how-we-beat-the-5-epoch-imagewoof-leaderboard-score-some-new-techniques-to-consider)

## MXResNet constructor.


```python
mxresnet  = Net(stem_sizes = [3, 32, 64, 64], name='MXResNet', act_fn=Mish())
```


```python
mxresnet
```
???+ done "output"  
    <pre>MXResNet constructor
      c_in: 3, c_out: 1000
      expansion: 1, groups: 1, dw: False, div_groups: None
      sa: False, se: False
      stem sizes: [3, 32, 64, 64], stride on 0
      body sizes [64, 128, 256, 512]
      layers: [2, 2, 2, 2]




```python
mxresnet.block_sizes, mxresnet.layers
```
???+ done "output"  
    <pre>([64, 64, 128, 256, 512], [2, 2, 2, 2])




```python

mxresnet.stem
```
??? done "output"  
    <pre>Sequential(
      (conv_0): ConvLayer(
        (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): Mish()
      )
      (conv_1): ConvLayer(
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): Mish()
      )
      (conv_2): ConvLayer(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): Mish()
      )
      (stem_pool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    )




```python

mxresnet.body
```
??? done "output"  
    <pre>Sequential(
      (l_0): Sequential(
        (bl_0): ResBlock(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
            )
            (conv_1): ConvLayer(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (act_fn): Mish()
        )
        (bl_1): ResBlock(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
            )
            (conv_1): ConvLayer(
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
            (conv_0): ConvLayer(
              (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
            )
            (conv_1): ConvLayer(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
          (idconv): ConvLayer(
            (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (act_fn): Mish()
        )
        (bl_1): ResBlock(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
            )
            (conv_1): ConvLayer(
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
            (conv_0): ConvLayer(
              (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
            )
            (conv_1): ConvLayer(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
          (idconv): ConvLayer(
            (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (act_fn): Mish()
        )
        (bl_1): ResBlock(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
            )
            (conv_1): ConvLayer(
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
            (conv_0): ConvLayer(
              (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
            )
            (conv_1): ConvLayer(
              (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
          (idconv): ConvLayer(
            (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (act_fn): Mish()
        )
        (bl_1): ResBlock(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
            )
            (conv_1): ConvLayer(
              (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (act_fn): Mish()
        )
      )
    )




```python

mxresnet.body
```
??? done "output"  
    <pre>Sequential(
      (l_0): Sequential(
        (bl_0): ResBlock(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
            )
            (conv_1): ConvLayer(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (act_fn): Mish()
        )
        (bl_1): ResBlock(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
            )
            (conv_1): ConvLayer(
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
            (conv_0): ConvLayer(
              (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
            )
            (conv_1): ConvLayer(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
          (idconv): ConvLayer(
            (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (act_fn): Mish()
        )
        (bl_1): ResBlock(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
            )
            (conv_1): ConvLayer(
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
            (conv_0): ConvLayer(
              (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
            )
            (conv_1): ConvLayer(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
          (idconv): ConvLayer(
            (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (act_fn): Mish()
        )
        (bl_1): ResBlock(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
            )
            (conv_1): ConvLayer(
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
            (conv_0): ConvLayer(
              (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
            )
            (conv_1): ConvLayer(
              (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
          (idconv): ConvLayer(
            (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (act_fn): Mish()
        )
        (bl_1): ResBlock(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): Mish()
            )
            (conv_1): ConvLayer(
              (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (act_fn): Mish()
        )
      )
    )




```python

mxresnet.head
```
??? done "output"  
    <pre>Sequential(
      (pool): AdaptiveAvgPool2d(output_size=1)
      (flat): Flatten()
      (fc): Linear(in_features=512, out_features=1000, bias=True)
    )



## MxResNet constructors


```python
mxresnet_parameters = {'stem_sizes': [3, 32, 64, 64], 'act_fn': Mish()}
mxresnet34 = partial(Net, name='MXResnet32', expansion=1, layers=[3, 4, 6, 3], **mxresnet_parameters)
mxresnet50 = partial(Net, name='MXResnet50', expansion=4, layers=[3, 4, 6, 3], **mxresnet_parameters)
```


```python
model = mxresnet50(c_out=10)
```


```python
model
```
???+ done "output"  
    <pre>MXResnet50 constructor
      c_in: 3, c_out: 10
      expansion: 4, groups: 1, dw: False, div_groups: None
      sa: False, se: False
      stem sizes: [3, 32, 64, 64], stride on 0
      body sizes [64, 128, 256, 512]
      layers: [3, 4, 6, 3]




```python
model.c_out, model.layers
```
???+ done "output"  
    <pre>(10, [3, 4, 6, 3])





model_constructor
by ayasyrev
