# model_constructor
> Constructor to create pytorch model.


_

## Install

`pip install model-constructor`

Or instll from repo:

`pip install git+https://github.com/ayasyrev/model_constructor.git`

## How to use

It can be used two ways.  
Classic - create model from function with parameters.   
And by creating constructor object, then modify it and then create model.
    

# Class Net

First import constructor class, then create model constructor oject.

```python
from model_constructor.net import *
```

```python
model = Net()
```

```python
model
```




     constr Net



Now we have model consructor, defoult setting as xresnet18. And we can get model after call it.

```python
model.c_in
```




    3



```python
model.c_out
```




    1000



```python
model.stem_sizes
```




    [3, 32, 32, 64]



```python
model.layers
```




    [2, 2, 2, 2]



```python
model.expansion
```




    1



```python
model()
```




    Sequential(
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
          (bl_0): ResBlock(
            (convs): Sequential(
              (conv_0): ConvLayer(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_1): ConvLayer(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (act_fn): ReLU(inplace=True)
          )
          (bl_1): ResBlock(
            (convs): Sequential(
              (conv_0): ConvLayer(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_1): ConvLayer(
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
              (conv_0): ConvLayer(
                (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
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
            (act_fn): ReLU(inplace=True)
          )
          (bl_1): ResBlock(
            (convs): Sequential(
              (conv_0): ConvLayer(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_1): ConvLayer(
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
              (conv_0): ConvLayer(
                (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
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
            (act_fn): ReLU(inplace=True)
          )
          (bl_1): ResBlock(
            (convs): Sequential(
              (conv_0): ConvLayer(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_1): ConvLayer(
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
              (conv_0): ConvLayer(
                (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
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
            (act_fn): ReLU(inplace=True)
          )
          (bl_1): ResBlock(
            (convs): Sequential(
              (conv_0): ConvLayer(
                (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act_fn): ReLU(inplace=True)
              )
              (conv_1): ConvLayer(
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
        (flat): Flatten()
        (fc): Linear(in_features=512, out_features=1000, bias=True)
      )
    )



If you want to change model, just change constructor parameters.  
Lets create xresnet50.

```python
model.expansion = 4
model.layers = [3,4,6,3]
```

Now we can look at model body and if we call constructor - we have pytorch model!

```python
model.body
```




    Sequential(
      (l_0): Sequential(
        (bl_0): ResBlock(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1): ConvLayer(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
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
        (bl_1): ResBlock(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1): ConvLayer(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
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
        (bl_2): ResBlock(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1): ConvLayer(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
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
        (bl_0): ResBlock(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1): ConvLayer(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
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
        (bl_1): ResBlock(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1): ConvLayer(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
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
        (bl_2): ResBlock(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1): ConvLayer(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
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
        (bl_3): ResBlock(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1): ConvLayer(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
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
        (bl_0): ResBlock(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1): ConvLayer(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
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
        (bl_1): ResBlock(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1): ConvLayer(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
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
        (bl_2): ResBlock(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1): ConvLayer(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
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
        (bl_3): ResBlock(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1): ConvLayer(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
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
        (bl_4): ResBlock(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1): ConvLayer(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
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
        (bl_5): ResBlock(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1): ConvLayer(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
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
        (bl_0): ResBlock(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1): ConvLayer(
              (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
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
        (bl_1): ResBlock(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1): ConvLayer(
              (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
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
        (bl_2): ResBlock(
          (convs): Sequential(
            (conv_0): ConvLayer(
              (conv): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act_fn): ReLU(inplace=True)
            )
            (conv_1): ConvLayer(
              (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
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



```python
model.block_szs
```




    [16, 64, 128, 256, 512]



## More modification.

Main purpose of this module - fast and easy modify model.
And here is the link to more modification to beat Imagenette leaderboard with add MaxBlurPool and modification to ResBlock https://github.com/ayasyrev/imagenette_experiments/blob/master/ResnetTrick_create_model_fit.ipynb  

But now lets create model as mxresnet50 from fastai forums tread https://forums.fast.ai/t/how-we-beat-the-5-epoch-imagewoof-leaderboard-score-some-new-techniques-to-consider  


Lets create mxresnet constructor.

```python
mxresnet = Net()
```

Then lets modify stem.

```python
mxresnet.stem_sizes = [3,32,64,64]
```

Now lets change activation function to Mish.
Here is link to forum disscussion https://forums.fast.ai/t/meet-mish-new-activation-function-possible-successor-to-relu

```python
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):  
        return x *( torch.tanh(F.softplus(x))) 
```

```python
mxresnet.expansion = 4
mxresnet.layers = [3,4,6,3]
mxresnet.act_fn = Mish()
mxresnet.name = 'mxresnet50'
```

Now we have mxresnet50 constructor.  
We can inspect some parts of it.  
And after call it we got model.

```python
mxresnet
```




     constr mxresnet50



```python
mxresnet.stem.conv_1
```




    ConvLayer(
      (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act_fn): Mish()
    )



```python
mxresnet.body.l_0.bl_0
```




    ResBlock(
      (convs): Sequential(
        (conv_0): ConvLayer(
          (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): Mish()
        )
        (conv_1): ConvLayer(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): Mish()
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
      (act_fn): Mish()
    )



Now lets change Resblock. NewResBlock (stiil not own name yet) is in lib from version 0.1.0

```python
mxresnet.block = NewResBlock
```

That all. Let see what we have.

```python
mxresnet.body.l_1.bl_0
```




    NewResBlock(
      (reduce): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (convs): Sequential(
        (conv_0): ConvLayer(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): Mish()
        )
        (conv_1): ConvLayer(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): Mish()
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
      (merge): Mish()
    )



# Classic way

Usual way to get model - call constructor with parametrs.

```python
from model_constructor.constructor import *
```

Default is resnet18.

```python
model = Net()
```

You cant modify model after call constructor, so define model with parameters.   
For example, resnet34:

```python
resnet34 = Net(block=BasicBlock, blocks=[3, 4, 6, 3])
```

## Predefined Resnet models - 18, 34, 50.

```python
from model_constructor.resnet import *
```

```python
model = resnet34(num_classes=10)
```

```python
model = resnet50(num_classes=10)
```

## Predefined Xresnet from fastai 1.

This ie simplified version from fastai v1. I did refactoring for better understand and experiment with models. For example, it's very simple to change activation funtions, different stems, batchnorm and activation order etc. In v2 much powerfull realisation.

```python
from model_constructor.xresnet import *
```

```python
model = xresnet50()
```

## Some examples.

We can experiment with models by changing some parts of model. Here only base functionality, but it can be easily extanded.

Here is some examples:
    

### Custom stem

Stem with 3 conv layers

```python
model = Net(stem=partial(Stem, stem_sizes=[32, 32]))
```

```python
model.stem
```




    Stem(
      sizes: [3, 32, 32, 64]
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
      (pool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    )



```python
model = Net(stem_sizes=[32, 64])
```

```python
model.stem
```




    Stem(
      sizes: [3, 32, 64, 64]
      (conv_0): ConvLayer(
        (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): ReLU(inplace=True)
      )
      (conv_1): ConvLayer(
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): ReLU(inplace=True)
      )
      (conv_2): ConvLayer(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): ReLU(inplace=True)
      )
      (pool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    )



### Activation function before Normalization

```python
model = Net(bn_1st=False)
```

```python
model.stem
```




    Stem(
      sizes: [3, 64]
      (conv_0): ConvLayer(
        (conv): Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (act_fn): ReLU(inplace=True)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (pool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    )




### Change activation function

```python
new_act_fn = nn.LeakyReLU(inplace=True)
```

```python
model = Net(act_fn=new_act_fn)
```

```python
model.stem
```




    Stem(
      sizes: [3, 64]
      (conv_0): ConvLayer(
        (conv): Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_fn): LeakyReLU(negative_slope=0.01, inplace=True)
      )
      (pool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    )



```python
model.body.layer_0.block_0
```




    BasicBlock(
      (conv): Sequential(
        (conv_0): ConvLayer(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act_fn): LeakyReLU(negative_slope=0.01, inplace=True)
        )
        (conv_1): ConvLayer(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (merge): Noop()
      (act_conn): LeakyReLU(negative_slope=0.01, inplace=True)
    )


