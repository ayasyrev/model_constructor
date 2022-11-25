# Twist.

> Create and tune models with Twist layers.

## ConvTwist


```python
from model_constructor.twist import ConvTwist
```


```python
ConvTwist(64,64)
```
<details open> <summary>output</summary>  
    <pre>ConvTwist(
      twist: False, permute: True, same: True, groups: 8
      (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=8, bias=False)
    )<pre>
</details>




```python
ConvTwist.twist, ConvTwist.permute
```
<details open> <summary>output</summary>  
    <pre>(False, True)<pre>
</details>




```python
ConvTwist.use_groups, ConvTwist.groups_ch
```
<details open> <summary>output</summary>  
    <pre>(True, 8)<pre>
</details>




```python
ConvTwist(64,64)
```
<details open> <summary>output</summary>  
    <pre>ConvTwist(
      twist: False, permute: True, same: True, groups: 8
      (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=8, bias=False)
    )<pre>
</details>




```python
ConvTwist.twist = True
ConvTwist.permute = False
ConvTwist(64,64)
```
<details open> <summary>output</summary>  
    <pre>ConvTwist(
      twist: True, permute: False, same: True, groups: 8
      (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=8, bias=False)
    )<pre>
</details>



## ConvLayerTwist


```python
class ConvLayerTwist(ConvLayer):  # replace Conv2d by Twist
    Conv2d = ConvTwist
```


```python
ConvLayerTwist(64,64, stride=1)
```
<details open> <summary>output</summary>  
    <pre>ConvLayerTwist(
      (conv): ConvTwist(
        twist: True, permute: False, same: True, groups: 8
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=8, bias=False)
      )
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act_fn): ReLU(inplace=True)
    )<pre>
</details>




```python
ConvLayer.Conv2d
```
<details open> <summary>output</summary>  
    <pre>torch.nn.modules.conv.Conv2d<pre>
</details>




```python
ConvLayerTwist.Conv2d
```
<details open> <summary>output</summary>  
    <pre>model_constructor.twist.ConvTwist<pre>
</details>




```python
conv_layer = ConvLayerTwist(32, 64)
conv_layer
```
<details open> <summary>output</summary>  
    <pre>ConvLayerTwist(
      (conv): ConvTwist(
        twist: True, permute: False, same: False, groups: 4
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
      )
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act_fn): ReLU(inplace=True)
    )<pre>
</details>




```python
ConvTwist.twist = False
conv_layer = ConvLayerTwist(32, 64)
conv_layer
```
<details open> <summary>output</summary>  
    <pre>ConvLayerTwist(
      (conv): ConvTwist(
        twist: False, permute: False, same: False, groups: 4
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
      )
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act_fn): ReLU(inplace=True)
    )<pre>
</details>




```python
conv_layer = ConvLayerTwist(32, 64, act=False)
conv_layer
```
<details open> <summary>output</summary>  
    <pre>ConvLayerTwist(
      (conv): ConvTwist(
        twist: False, permute: False, same: False, groups: 4
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
      )
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )<pre>
</details>




```python
conv_layer = ConvLayerTwist(32, 64, bn_layer=False)
conv_layer
```
<details open> <summary>output</summary>  
    <pre>ConvLayerTwist(
      (conv): ConvTwist(
        twist: False, permute: False, same: False, groups: 4
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
      )
      (act_fn): ReLU(inplace=True)
    )<pre>
</details>




```python
conv_layer = ConvLayerTwist(32, 64, bn_1st=True)
conv_layer
```
<details open> <summary>output</summary>  
    <pre>ConvLayerTwist(
      (conv): ConvTwist(
        twist: False, permute: False, same: False, groups: 4
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
      )
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act_fn): ReLU(inplace=True)
    )<pre>
</details>




```python
conv_layer = ConvLayerTwist(32, 64, bn_1st=True, act_fn=nn.LeakyReLU())
conv_layer
```
<details open> <summary>output</summary>  
    <pre>ConvLayerTwist(
      (conv): ConvTwist(
        twist: False, permute: False, same: False, groups: 4
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
      )
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act_fn): LeakyReLU(negative_slope=0.01)
    )<pre>
</details>




```python
conv_layer = ConvLayerTwist(32, 64, ks=1)
conv_layer
```
<details open> <summary>output</summary>  
    <pre>ConvLayerTwist(
      (conv): ConvTwist(
        twist: False, permute: False, same: False, groups: 4
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
      )
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act_fn): ReLU(inplace=True)
    )<pre>
</details>




```python
conv_layer = ConvLayerTwist(32, 64, ks=1, stride=2)
conv_layer
```
<details open> <summary>output</summary>  
    <pre>ConvLayerTwist(
      (conv): ConvTwist(
        twist: False, permute: False, same: False, groups: 4
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=4, bias=False)
      )
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act_fn): ReLU(inplace=True)
    )<pre>
</details>




```python
conv_layer = ConvLayerTwist(32, 64, stride=2)
conv_layer
```
<details open> <summary>output</summary>  
    <pre>ConvLayerTwist(
      (conv): ConvTwist(
        twist: False, permute: False, same: False, groups: 4
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=4, bias=False)
      )
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act_fn): ReLU(inplace=True)
    )<pre>
</details>




```python
ConvTwist.groups_ch = 4
conv_layer = ConvLayerTwist(32, 64, stride=2)
conv_layer
```
<details open> <summary>output</summary>  
    <pre>ConvLayerTwist(
      (conv): ConvTwist(
        twist: False, permute: False, same: False, groups: 8
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=8, bias=False)
      )
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act_fn): ReLU(inplace=True)
    )<pre>
</details>



## NewResBlockTwist


```python
from model_constructor.twist import NewResBlockTwist
```


```python

bl = NewResBlockTwist(4,64,64,sa=True)
bl
```
<details> <summary>output</summary>  
    <pre>NewResBlockTwist(
      (convs): Sequential(
        (conv_0): ConvLayer(
          (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): ReLU(inplace=True)
        )
        (conv_1_twist): ConvLayerTwist(
          (conv): ConvTwist(
            twist: False, permute: False, same: True, groups: 16
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
          )
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): ReLU(inplace=True)
        )
        (conv_2): ConvLayer(
          (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (sa): SimpleSelfAttention(
          (conv): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
        )
      )
      (merge): ReLU(inplace=True)
    )<pre>
</details>




```python

bl = NewResBlockTwist(4,64,64,stride=2)
bl
```
<details> <summary>output</summary>  
    <pre>NewResBlockTwist(
      (reduce): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (convs): Sequential(
        (conv_0): ConvLayer(
          (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): ReLU(inplace=True)
        )
        (conv_1_twist): ConvLayerTwist(
          (conv): ConvTwist(
            twist: False, permute: False, same: True, groups: 16
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
          )
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): ReLU(inplace=True)
        )
        (conv_2): ConvLayer(
          (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (merge): ReLU(inplace=True)
    )<pre>
</details>




```python

bl = NewResBlockTwist(4,64,128,stride=2)
bl
```
<details> <summary>output</summary>  
    <pre>NewResBlockTwist(
      (reduce): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (convs): Sequential(
        (conv_0): ConvLayer(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): ReLU(inplace=True)
        )
        (conv_1_twist): ConvLayerTwist(
          (conv): ConvTwist(
            twist: False, permute: False, same: True, groups: 32
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          )
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): ReLU(inplace=True)
        )
        (conv_2): ConvLayer(
          (conv): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (idconv): ConvLayer(
        (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (merge): ReLU(inplace=True)
    )<pre>
</details>




```python

bl = NewResBlockTwist(4,64,128,stride=2,act_fn=nn.LeakyReLU(), bn_1st=False)
bl
```
<details> <summary>output</summary>  
    <pre>NewResBlockTwist(
      (reduce): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (convs): Sequential(
        (conv_0): ConvLayer(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (act_fn): LeakyReLU(negative_slope=0.01)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv_1_twist): ConvLayerTwist(
          (conv): ConvTwist(
            twist: False, permute: False, same: True, groups: 32
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          )
          (act_fn): LeakyReLU(negative_slope=0.01)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (conv_2): ConvLayer(
          (conv): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (idconv): ConvLayer(
        (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (merge): LeakyReLU(negative_slope=0.01)
    )<pre>
</details>



## ResBlockTwist


```python
from model_constructor.twist import ResBlockTwist
```


```python

bl = ResBlockTwist(4,64,64,sa=True)
bl
```
<details> <summary>output</summary>  
    <pre>ResBlockTwist(
      (convs): Sequential(
        (conv_0): ConvLayer(
          (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): ReLU(inplace=True)
        )
        (conv_1_twist): ConvLayerTwist(
          (conv): ConvTwist(
            twist: False, permute: False, same: True, groups: 16
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
          )
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): ReLU(inplace=True)
        )
        (conv_2): ConvLayer(
          (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (sa): SimpleSelfAttention(
          (conv): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
        )
      )
      (act_fn): ReLU(inplace=True)
    )<pre>
</details>




```python

bl = ResBlockTwist(4,64,64,stride=2)
bl
```
<details> <summary>output</summary>  
    <pre>ResBlockTwist(
      (convs): Sequential(
        (conv_0): ConvLayer(
          (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): ReLU(inplace=True)
        )
        (conv_1_twist): ConvLayerTwist(
          (conv): ConvTwist(
            twist: False, permute: False, same: False, groups: 16
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=16, bias=False)
          )
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): ReLU(inplace=True)
        )
        (conv_2): ConvLayer(
          (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (act_fn): ReLU(inplace=True)
    )<pre>
</details>




```python

bl = ResBlockTwist(4,64,128,stride=2)
bl
```
<details> <summary>output</summary>  
    <pre>ResBlockTwist(
      (convs): Sequential(
        (conv_0): ConvLayer(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): ReLU(inplace=True)
        )
        (conv_1_twist): ConvLayerTwist(
          (conv): ConvTwist(
            twist: False, permute: False, same: False, groups: 32
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=32, bias=False)
          )
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): ReLU(inplace=True)
        )
        (conv_2): ConvLayer(
          (conv): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (idconv): ConvLayer(
        (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (act_fn): ReLU(inplace=True)
    )<pre>
</details>



## Model


```python
model  = Net(expansion=4, layers=[3,4,6,3])
```


```python
model.block = NewResBlockTwist
```


```python

model.body
```
<details> <summary>output</summary>  
    <pre>Sequential(
      (l_0): Sequential(
        (bl_0): NewResBlockTwist(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1_twist): ConvLayerTwist(
              (conv): ConvTwist(
                twist: False, permute: False, same: True, groups: 16
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
              )
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_2): ConvLayer(
              (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (idconv): ConvLayer(
            (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (merge): ReLU(inplace=True)
        )
        (bl_1): NewResBlockTwist(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1_twist): ConvLayerTwist(
              (conv): ConvTwist(
                twist: False, permute: False, same: True, groups: 16
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
              )
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_2): ConvLayer(
              (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (merge): ReLU(inplace=True)
        )
        (bl_2): NewResBlockTwist(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1_twist): ConvLayerTwist(
              (conv): ConvTwist(
                twist: False, permute: False, same: True, groups: 16
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
              )
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_2): ConvLayer(
              (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (merge): ReLU(inplace=True)
        )
      )
      (l_1): Sequential(
        (bl_0): NewResBlockTwist(
          (reduce): AvgPool2d(kernel_size=2, stride=2, padding=0)
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1_twist): ConvLayerTwist(
              (conv): ConvTwist(
                twist: False, permute: False, same: True, groups: 32
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
              )
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_2): ConvLayer(
              (conv): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (idconv): ConvLayer(
            (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (merge): ReLU(inplace=True)
        )
        (bl_1): NewResBlockTwist(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1_twist): ConvLayerTwist(
              (conv): ConvTwist(
                twist: False, permute: False, same: True, groups: 32
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
              )
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_2): ConvLayer(
              (conv): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (merge): ReLU(inplace=True)
        )
        (bl_2): NewResBlockTwist(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1_twist): ConvLayerTwist(
              (conv): ConvTwist(
                twist: False, permute: False, same: True, groups: 32
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
              )
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_2): ConvLayer(
              (conv): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (merge): ReLU(inplace=True)
        )
        (bl_3): NewResBlockTwist(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1_twist): ConvLayerTwist(
              (conv): ConvTwist(
                twist: False, permute: False, same: True, groups: 32
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
              )
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_2): ConvLayer(
              (conv): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (merge): ReLU(inplace=True)
        )
      )
      (l_2): Sequential(
        (bl_0): NewResBlockTwist(
          (reduce): AvgPool2d(kernel_size=2, stride=2, padding=0)
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1_twist): ConvLayerTwist(
              (conv): ConvTwist(
                twist: False, permute: False, same: True, groups: 64
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
              )
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_2): ConvLayer(
              (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (idconv): ConvLayer(
            (conv): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (merge): ReLU(inplace=True)
        )
        (bl_1): NewResBlockTwist(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1_twist): ConvLayerTwist(
              (conv): ConvTwist(
                twist: False, permute: False, same: True, groups: 64
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
              )
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_2): ConvLayer(
              (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (merge): ReLU(inplace=True)
        )
        (bl_2): NewResBlockTwist(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1_twist): ConvLayerTwist(
              (conv): ConvTwist(
                twist: False, permute: False, same: True, groups: 64
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
              )
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_2): ConvLayer(
              (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (merge): ReLU(inplace=True)
        )
        (bl_3): NewResBlockTwist(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1_twist): ConvLayerTwist(
              (conv): ConvTwist(
                twist: False, permute: False, same: True, groups: 64
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
              )
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_2): ConvLayer(
              (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (merge): ReLU(inplace=True)
        )
        (bl_4): NewResBlockTwist(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1_twist): ConvLayerTwist(
              (conv): ConvTwist(
                twist: False, permute: False, same: True, groups: 64
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
              )
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_2): ConvLayer(
              (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (merge): ReLU(inplace=True)
        )
        (bl_5): NewResBlockTwist(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1_twist): ConvLayerTwist(
              (conv): ConvTwist(
                twist: False, permute: False, same: True, groups: 64
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
              )
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_2): ConvLayer(
              (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (merge): ReLU(inplace=True)
        )
      )
      (l_3): Sequential(
        (bl_0): NewResBlockTwist(
          (reduce): AvgPool2d(kernel_size=2, stride=2, padding=0)
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1_twist): ConvLayerTwist(
              (conv): ConvTwist(
                twist: False, permute: False, same: True, groups: 128
                (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
              )
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_2): ConvLayer(
              (conv): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (idconv): ConvLayer(
            (conv): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (merge): ReLU(inplace=True)
        )
        (bl_1): NewResBlockTwist(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1_twist): ConvLayerTwist(
              (conv): ConvTwist(
                twist: False, permute: False, same: True, groups: 128
                (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
              )
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_2): ConvLayer(
              (conv): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (merge): ReLU(inplace=True)
        )
        (bl_2): NewResBlockTwist(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1_twist): ConvLayerTwist(
              (conv): ConvTwist(
                twist: False, permute: False, same: True, groups: 128
                (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
              )
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_2): ConvLayer(
              (conv): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (merge): ReLU(inplace=True)
        )
      )
    )<pre>
</details>




```python
model.block = ResBlockTwist
```


```python
m = model()
```


```python

m
```
<details> <summary>output</summary>  
    <pre>Sequential(
      model Net
      (stem): Sequential(
        (conv_0): ConvLayer(
          (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): ReLU(inplace=True)
        )
        (conv_1): ConvLayer(
          (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): ReLU(inplace=True)
        )
        (conv_2): ConvLayer(
          (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): ReLU(inplace=True)
        )
        (stem_pool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      )
      (body): Sequential(
        (l_0): Sequential(
          (bl_0): ResBlockTwist(
            (convs): Sequential(
              (conv_0): ConvLayer(
                (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_1_twist): ConvLayerTwist(
                (conv): ConvTwist(
                  twist: False, permute: False, same: True, groups: 16
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
                )
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_2): ConvLayer(
                (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (idconv): ConvLayer(
              (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (act_fn): ReLU(inplace=True)
          )
          (bl_1): ResBlockTwist(
            (convs): Sequential(
              (conv_0): ConvLayer(
                (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_1_twist): ConvLayerTwist(
                (conv): ConvTwist(
                  twist: False, permute: False, same: True, groups: 16
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
                )
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_2): ConvLayer(
                (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act_fn): ReLU(inplace=True)
          )
          (bl_2): ResBlockTwist(
            (convs): Sequential(
              (conv_0): ConvLayer(
                (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_1_twist): ConvLayerTwist(
                (conv): ConvTwist(
                  twist: False, permute: False, same: True, groups: 16
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
                )
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_2): ConvLayer(
                (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act_fn): ReLU(inplace=True)
          )
        )
        (l_1): Sequential(
          (bl_0): ResBlockTwist(
            (convs): Sequential(
              (conv_0): ConvLayer(
                (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_1_twist): ConvLayerTwist(
                (conv): ConvTwist(
                  twist: False, permute: False, same: False, groups: 32
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=32, bias=False)
                )
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_2): ConvLayer(
                (conv): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
            (idconv): ConvLayer(
              (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (act_fn): ReLU(inplace=True)
          )
          (bl_1): ResBlockTwist(
            (convs): Sequential(
              (conv_0): ConvLayer(
                (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_1_twist): ConvLayerTwist(
                (conv): ConvTwist(
                  twist: False, permute: False, same: True, groups: 32
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
                )
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_2): ConvLayer(
                (conv): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act_fn): ReLU(inplace=True)
          )
          (bl_2): ResBlockTwist(
            (convs): Sequential(
              (conv_0): ConvLayer(
                (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_1_twist): ConvLayerTwist(
                (conv): ConvTwist(
                  twist: False, permute: False, same: True, groups: 32
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
                )
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_2): ConvLayer(
                (conv): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act_fn): ReLU(inplace=True)
          )
          (bl_3): ResBlockTwist(
            (convs): Sequential(
              (conv_0): ConvLayer(
                (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_1_twist): ConvLayerTwist(
                (conv): ConvTwist(
                  twist: False, permute: False, same: True, groups: 32
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
                )
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_2): ConvLayer(
                (conv): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act_fn): ReLU(inplace=True)
          )
        )
        (l_2): Sequential(
          (bl_0): ResBlockTwist(
            (convs): Sequential(
              (conv_0): ConvLayer(
                (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_1_twist): ConvLayerTwist(
                (conv): ConvTwist(
                  twist: False, permute: False, same: False, groups: 64
                  (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=64, bias=False)
                )
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_2): ConvLayer(
                (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
            (idconv): ConvLayer(
              (conv): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (act_fn): ReLU(inplace=True)
          )
          (bl_1): ResBlockTwist(
            (convs): Sequential(
              (conv_0): ConvLayer(
                (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_1_twist): ConvLayerTwist(
                (conv): ConvTwist(
                  twist: False, permute: False, same: True, groups: 64
                  (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
                )
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_2): ConvLayer(
                (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act_fn): ReLU(inplace=True)
          )
          (bl_2): ResBlockTwist(
            (convs): Sequential(
              (conv_0): ConvLayer(
                (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_1_twist): ConvLayerTwist(
                (conv): ConvTwist(
                  twist: False, permute: False, same: True, groups: 64
                  (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
                )
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_2): ConvLayer(
                (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act_fn): ReLU(inplace=True)
          )
          (bl_3): ResBlockTwist(
            (convs): Sequential(
              (conv_0): ConvLayer(
                (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_1_twist): ConvLayerTwist(
                (conv): ConvTwist(
                  twist: False, permute: False, same: True, groups: 64
                  (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
                )
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_2): ConvLayer(
                (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act_fn): ReLU(inplace=True)
          )
          (bl_4): ResBlockTwist(
            (convs): Sequential(
              (conv_0): ConvLayer(
                (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_1_twist): ConvLayerTwist(
                (conv): ConvTwist(
                  twist: False, permute: False, same: True, groups: 64
                  (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
                )
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_2): ConvLayer(
                (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act_fn): ReLU(inplace=True)
          )
          (bl_5): ResBlockTwist(
            (convs): Sequential(
              (conv_0): ConvLayer(
                (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_1_twist): ConvLayerTwist(
                (conv): ConvTwist(
                  twist: False, permute: False, same: True, groups: 64
                  (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
                )
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_2): ConvLayer(
                (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act_fn): ReLU(inplace=True)
          )
        )
        (l_3): Sequential(
          (bl_0): ResBlockTwist(
            (convs): Sequential(
              (conv_0): ConvLayer(
                (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_1_twist): ConvLayerTwist(
                (conv): ConvTwist(
                  twist: False, permute: False, same: False, groups: 128
                  (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=128, bias=False)
                )
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_2): ConvLayer(
                (conv): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
            (idconv): ConvLayer(
              (conv): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (act_fn): ReLU(inplace=True)
          )
          (bl_1): ResBlockTwist(
            (convs): Sequential(
              (conv_0): ConvLayer(
                (conv): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_1_twist): ConvLayerTwist(
                (conv): ConvTwist(
                  twist: False, permute: False, same: True, groups: 128
                  (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
                )
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_2): ConvLayer(
                (conv): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act_fn): ReLU(inplace=True)
          )
          (bl_2): ResBlockTwist(
            (convs): Sequential(
              (conv_0): ConvLayer(
                (conv): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_1_twist): ConvLayerTwist(
                (conv): ConvTwist(
                  twist: False, permute: False, same: True, groups: 128
                  (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
                )
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_2): ConvLayer(
                (conv): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act_fn): ReLU(inplace=True)
          )
        )
      )
      (head): Sequential(
        (pool): AdaptiveAvgPool2d(output_size=1)
        (flat): Flatten()
        (fc): Linear(in_features=2048, out_features=1000, bias=True)
      )
    )<pre>
</details>




```python

m.stem
```
<details> <summary>output</summary>  
    <pre>Sequential(
      (conv_0): ConvLayer(
        (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): ReLU(inplace=True)
      )
      (conv_1): ConvLayer(
        (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): ReLU(inplace=True)
      )
      (conv_2): ConvLayer(
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): ReLU(inplace=True)
      )
      (stem_pool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    )<pre>
</details>




```python

m.head
```
<details> <summary>output</summary>  
    <pre>Sequential(
      (pool): AdaptiveAvgPool2d(output_size=1)
      (flat): Flatten()
      (fc): Linear(in_features=2048, out_features=1000, bias=True)
    )<pre>
</details>




```python

m.body.l_0
```
<details> <summary>output</summary>  
    <pre>Sequential(
      (bl_0): ResBlockTwist(
        (convs): Sequential(
          (conv_0): ConvLayer(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_1_twist): ConvLayerTwist(
            (conv): ConvTwist(
              twist: False, permute: False, same: True, groups: 16
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
            )
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_2): ConvLayer(
            (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (idconv): ConvLayer(
          (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (act_fn): ReLU(inplace=True)
      )
      (bl_1): ResBlockTwist(
        (convs): Sequential(
          (conv_0): ConvLayer(
            (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_1_twist): ConvLayerTwist(
            (conv): ConvTwist(
              twist: False, permute: False, same: True, groups: 16
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
            )
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_2): ConvLayer(
            (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (act_fn): ReLU(inplace=True)
      )
      (bl_2): ResBlockTwist(
        (convs): Sequential(
          (conv_0): ConvLayer(
            (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_1_twist): ConvLayerTwist(
            (conv): ConvTwist(
              twist: False, permute: False, same: True, groups: 16
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
            )
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_2): ConvLayer(
            (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (act_fn): ReLU(inplace=True)
      )
    )<pre>
</details>




```python

m.body.l_1
```
<details> <summary>output</summary>  
    <pre>Sequential(
      (bl_0): ResBlockTwist(
        (convs): Sequential(
          (conv_0): ConvLayer(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_1_twist): ConvLayerTwist(
            (conv): ConvTwist(
              twist: False, permute: False, same: False, groups: 32
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=32, bias=False)
            )
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_2): ConvLayer(
            (conv): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
        (idconv): ConvLayer(
          (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (act_fn): ReLU(inplace=True)
      )
      (bl_1): ResBlockTwist(
        (convs): Sequential(
          (conv_0): ConvLayer(
            (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_1_twist): ConvLayerTwist(
            (conv): ConvTwist(
              twist: False, permute: False, same: True, groups: 32
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            )
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_2): ConvLayer(
            (conv): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (act_fn): ReLU(inplace=True)
      )
      (bl_2): ResBlockTwist(
        (convs): Sequential(
          (conv_0): ConvLayer(
            (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_1_twist): ConvLayerTwist(
            (conv): ConvTwist(
              twist: False, permute: False, same: True, groups: 32
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            )
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_2): ConvLayer(
            (conv): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (act_fn): ReLU(inplace=True)
      )
      (bl_3): ResBlockTwist(
        (convs): Sequential(
          (conv_0): ConvLayer(
            (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_1_twist): ConvLayerTwist(
            (conv): ConvTwist(
              twist: False, permute: False, same: True, groups: 32
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            )
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_2): ConvLayer(
            (conv): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (act_fn): ReLU(inplace=True)
      )
    )<pre>
</details>




```python

m.body.l_2
```
<details> <summary>output</summary>  
    <pre>Sequential(
      (bl_0): ResBlockTwist(
        (convs): Sequential(
          (conv_0): ConvLayer(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_1_twist): ConvLayerTwist(
            (conv): ConvTwist(
              twist: False, permute: False, same: False, groups: 64
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=64, bias=False)
            )
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_2): ConvLayer(
            (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
        (idconv): ConvLayer(
          (conv): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (act_fn): ReLU(inplace=True)
      )
      (bl_1): ResBlockTwist(
        (convs): Sequential(
          (conv_0): ConvLayer(
            (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_1_twist): ConvLayerTwist(
            (conv): ConvTwist(
              twist: False, permute: False, same: True, groups: 64
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
            )
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_2): ConvLayer(
            (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (act_fn): ReLU(inplace=True)
      )
      (bl_2): ResBlockTwist(
        (convs): Sequential(
          (conv_0): ConvLayer(
            (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_1_twist): ConvLayerTwist(
            (conv): ConvTwist(
              twist: False, permute: False, same: True, groups: 64
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
            )
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_2): ConvLayer(
            (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (act_fn): ReLU(inplace=True)
      )
      (bl_3): ResBlockTwist(
        (convs): Sequential(
          (conv_0): ConvLayer(
            (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_1_twist): ConvLayerTwist(
            (conv): ConvTwist(
              twist: False, permute: False, same: True, groups: 64
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
            )
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_2): ConvLayer(
            (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (act_fn): ReLU(inplace=True)
      )
      (bl_4): ResBlockTwist(
        (convs): Sequential(
          (conv_0): ConvLayer(
            (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_1_twist): ConvLayerTwist(
            (conv): ConvTwist(
              twist: False, permute: False, same: True, groups: 64
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
            )
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_2): ConvLayer(
            (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (act_fn): ReLU(inplace=True)
      )
      (bl_5): ResBlockTwist(
        (convs): Sequential(
          (conv_0): ConvLayer(
            (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_1_twist): ConvLayerTwist(
            (conv): ConvTwist(
              twist: False, permute: False, same: True, groups: 64
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
            )
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_2): ConvLayer(
            (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (act_fn): ReLU(inplace=True)
      )
    )<pre>
</details>




```python

m.body.l_3
```
<details> <summary>output</summary>  
    <pre>Sequential(
      (bl_0): ResBlockTwist(
        (convs): Sequential(
          (conv_0): ConvLayer(
            (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_1_twist): ConvLayerTwist(
            (conv): ConvTwist(
              twist: False, permute: False, same: False, groups: 128
              (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=128, bias=False)
            )
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_2): ConvLayer(
            (conv): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
        (idconv): ConvLayer(
          (conv): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (act_fn): ReLU(inplace=True)
      )
      (bl_1): ResBlockTwist(
        (convs): Sequential(
          (conv_0): ConvLayer(
            (conv): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_1_twist): ConvLayerTwist(
            (conv): ConvTwist(
              twist: False, permute: False, same: True, groups: 128
              (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
            )
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_2): ConvLayer(
            (conv): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (act_fn): ReLU(inplace=True)
      )
      (bl_2): ResBlockTwist(
        (convs): Sequential(
          (conv_0): ConvLayer(
            (conv): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_1_twist): ConvLayerTwist(
            (conv): ConvTwist(
              twist: False, permute: False, same: True, groups: 128
              (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
            )
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_fn): ReLU(inplace=True)
          )
          (conv_2): ConvLayer(
            (conv): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (act_fn): ReLU(inplace=True)
      )
    )<pre>
</details>



model_constructor
by ayasyrev
