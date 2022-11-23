# ConvMixer

> ConvMixer model.

Implementation of ConvMixer.  
ConvMixer - ICLR 2022 submission ["Patches Are All You Need?".](https://openreview.net/forum?id=TVHS5Y4dNvM)  
Adopted from [https://github.com/tmp-iclr/convmixer](https://github.com/tmp-iclr/convmixer)  
Home for convmixer: [https://github.com/locuslab/convmixer](https://github.com/locuslab/convmixer)

Purpose of this implementation - possibilities for tune this model.  
For example - play with activation function, initialization etc.  

## Import and create model

Base class for model - ConvMixer, return pytorch Sequential model.  


```python
from model_constructor import ConvMixer
```

Now we can create convmixer model:


```python
convmixer_1024_20 = ConvMixer(dim=1024, depth=20)
```


```python

convmixer_1024_20
```
<details> <summary>output</summary>  
    <pre>ConvMixer(
      (0): ConvLayer(
        (conv): Conv2d(3, 1024, kernel_size=(7, 7), stride=(7, 7))
        (act_fn): GELU()
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): Sequential(
        (0): Residual(
          (fn): ConvLayer(
            (conv): Conv2d(1024, 1024, kernel_size=(9, 9), stride=(1, 1), padding=same, groups=1024)
            (act_fn): GELU()
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): ConvLayer(
          (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
          (act_fn): GELU()
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (2): Sequential(
        (0): Residual(
          (fn): ConvLayer(
            (conv): Conv2d(1024, 1024, kernel_size=(9, 9), stride=(1, 1), padding=same, groups=1024)
            (act_fn): GELU()
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): ConvLayer(
          (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
          (act_fn): GELU()
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (3): Sequential(
        (0): Residual(
          (fn): ConvLayer(
            (conv): Conv2d(1024, 1024, kernel_size=(9, 9), stride=(1, 1), padding=same, groups=1024)
            (act_fn): GELU()
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): ConvLayer(
          (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
          (act_fn): GELU()
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (4): Sequential(
        (0): Residual(
          (fn): ConvLayer(
            (conv): Conv2d(1024, 1024, kernel_size=(9, 9), stride=(1, 1), padding=same, groups=1024)
            (act_fn): GELU()
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): ConvLayer(
          (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
          (act_fn): GELU()
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (5): Sequential(
        (0): Residual(
          (fn): ConvLayer(
            (conv): Conv2d(1024, 1024, kernel_size=(9, 9), stride=(1, 1), padding=same, groups=1024)
            (act_fn): GELU()
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): ConvLayer(
          (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
          (act_fn): GELU()
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (6): Sequential(
        (0): Residual(
          (fn): ConvLayer(
            (conv): Conv2d(1024, 1024, kernel_size=(9, 9), stride=(1, 1), padding=same, groups=1024)
            (act_fn): GELU()
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): ConvLayer(
          (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
          (act_fn): GELU()
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (7): Sequential(
        (0): Residual(
          (fn): ConvLayer(
            (conv): Conv2d(1024, 1024, kernel_size=(9, 9), stride=(1, 1), padding=same, groups=1024)
            (act_fn): GELU()
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): ConvLayer(
          (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
          (act_fn): GELU()
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (8): Sequential(
        (0): Residual(
          (fn): ConvLayer(
            (conv): Conv2d(1024, 1024, kernel_size=(9, 9), stride=(1, 1), padding=same, groups=1024)
            (act_fn): GELU()
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): ConvLayer(
          (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
          (act_fn): GELU()
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (9): Sequential(
        (0): Residual(
          (fn): ConvLayer(
            (conv): Conv2d(1024, 1024, kernel_size=(9, 9), stride=(1, 1), padding=same, groups=1024)
            (act_fn): GELU()
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): ConvLayer(
          (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
          (act_fn): GELU()
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (10): Sequential(
        (0): Residual(
          (fn): ConvLayer(
            (conv): Conv2d(1024, 1024, kernel_size=(9, 9), stride=(1, 1), padding=same, groups=1024)
            (act_fn): GELU()
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): ConvLayer(
          (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
          (act_fn): GELU()
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (11): Sequential(
        (0): Residual(
          (fn): ConvLayer(
            (conv): Conv2d(1024, 1024, kernel_size=(9, 9), stride=(1, 1), padding=same, groups=1024)
            (act_fn): GELU()
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): ConvLayer(
          (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
          (act_fn): GELU()
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (12): Sequential(
        (0): Residual(
          (fn): ConvLayer(
            (conv): Conv2d(1024, 1024, kernel_size=(9, 9), stride=(1, 1), padding=same, groups=1024)
            (act_fn): GELU()
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): ConvLayer(
          (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
          (act_fn): GELU()
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (13): Sequential(
        (0): Residual(
          (fn): ConvLayer(
            (conv): Conv2d(1024, 1024, kernel_size=(9, 9), stride=(1, 1), padding=same, groups=1024)
            (act_fn): GELU()
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): ConvLayer(
          (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
          (act_fn): GELU()
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (14): Sequential(
        (0): Residual(
          (fn): ConvLayer(
            (conv): Conv2d(1024, 1024, kernel_size=(9, 9), stride=(1, 1), padding=same, groups=1024)
            (act_fn): GELU()
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): ConvLayer(
          (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
          (act_fn): GELU()
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (15): Sequential(
        (0): Residual(
          (fn): ConvLayer(
            (conv): Conv2d(1024, 1024, kernel_size=(9, 9), stride=(1, 1), padding=same, groups=1024)
            (act_fn): GELU()
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): ConvLayer(
          (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
          (act_fn): GELU()
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (16): Sequential(
        (0): Residual(
          (fn): ConvLayer(
            (conv): Conv2d(1024, 1024, kernel_size=(9, 9), stride=(1, 1), padding=same, groups=1024)
            (act_fn): GELU()
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): ConvLayer(
          (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
          (act_fn): GELU()
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (17): Sequential(
        (0): Residual(
          (fn): ConvLayer(
            (conv): Conv2d(1024, 1024, kernel_size=(9, 9), stride=(1, 1), padding=same, groups=1024)
            (act_fn): GELU()
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): ConvLayer(
          (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
          (act_fn): GELU()
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (18): Sequential(
        (0): Residual(
          (fn): ConvLayer(
            (conv): Conv2d(1024, 1024, kernel_size=(9, 9), stride=(1, 1), padding=same, groups=1024)
            (act_fn): GELU()
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): ConvLayer(
          (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
          (act_fn): GELU()
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (19): Sequential(
        (0): Residual(
          (fn): ConvLayer(
            (conv): Conv2d(1024, 1024, kernel_size=(9, 9), stride=(1, 1), padding=same, groups=1024)
            (act_fn): GELU()
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): ConvLayer(
          (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
          (act_fn): GELU()
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (20): Sequential(
        (0): Residual(
          (fn): ConvLayer(
            (conv): Conv2d(1024, 1024, kernel_size=(9, 9), stride=(1, 1), padding=same, groups=1024)
            (act_fn): GELU()
            (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): ConvLayer(
          (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
          (act_fn): GELU()
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (21): AdaptiveAvgPool2d(output_size=(1, 1))
      (22): Flatten(start_dim=1, end_dim=-1)
      (23): Linear(in_features=1024, out_features=1000, bias=True)
    )<pre>
</details>



## Change activation function.

Lets create model with Mish (import it from torch) instead of GELU.


```python
convmixer_1024_20 = ConvMixer(dim=1024, depth=20, act_fn=Mish())
```


```python

convmixer_1024_20[0]
```
<details> <summary>output</summary>  
    <pre>ConvLayer(
      (conv): Conv2d(3, 1024, kernel_size=(7, 7), stride=(7, 7))
      (act_fn): Mish()
      (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )<pre>
</details>




```python

convmixer_1024_20[1]
```
<details> <summary>output</summary>  
    <pre>Sequential(
      (0): Residual(
        (fn): ConvLayer(
          (conv): Conv2d(1024, 1024, kernel_size=(9, 9), stride=(1, 1), padding=same, groups=1024)
          (act_fn): Mish()
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): ConvLayer(
        (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
        (act_fn): Mish()
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )<pre>
</details>



## Pre activation

Activation function before convolution.


```python
convmixer_1024_20 = ConvMixer(dim=1024, depth=20, act_fn=Mish(), pre_act=True)
```


```python

convmixer_1024_20[0]
```
<details> <summary>output</summary>  
    <pre>ConvLayer(
      (conv): Conv2d(3, 1024, kernel_size=(7, 7), stride=(7, 7))
      (act_fn): Mish()
      (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )<pre>
</details>




```python

convmixer_1024_20[1]
```
<details> <summary>output</summary>  
    <pre>Sequential(
      (0): Residual(
        (fn): ConvLayer(
          (act_fn): Mish()
          (conv): Conv2d(1024, 1024, kernel_size=(9, 9), stride=(1, 1), padding=same, groups=1024)
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): ConvLayer(
        (act_fn): Mish()
        (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )<pre>
</details>



## BatchNorm before activation.


```python
convmixer_1024_20 = ConvMixer(dim=1024, depth=20, act_fn=Mish(), bn_1st=True)
```


```python

convmixer_1024_20[0]
```
<details> <summary>output</summary>  
    <pre>ConvLayer(
      (conv): Conv2d(3, 1024, kernel_size=(7, 7), stride=(7, 7))
      (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act_fn): Mish()
    )<pre>
</details>




```python

convmixer_1024_20[1]
```
<details> <summary>output</summary>  
    <pre>Sequential(
      (0): Residual(
        (fn): ConvLayer(
          (conv): Conv2d(1024, 1024, kernel_size=(9, 9), stride=(1, 1), padding=same, groups=1024)
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): Mish()
        )
      )
      (1): ConvLayer(
        (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): Mish()
      )
    )<pre>
</details>


