# OpenVX MIGraphX Extension Library

`vx_amd_migraphx` is an OpenVX AMD extension module which has one node (`com.amd.amd_migraphx_node`). This node enables importing the <a href="https://github.com/ROCmSoftwarePlatform/AMDMIGraphX#amd-migraphx" target="_blank"> AMD's MIGraphx </a> library into an OpenVX graph for inference.

## Build Instructions

### Pre-requisites

* AMD OpenVX&trade; library
* <a href="https://github.com/ROCmSoftwarePlatform/AMDMIGraphX#amd-migraphx" target="_blank"> AMD MIGraphX </a>

This module is built by default when building the MIVisionX.

### Example 1: vision inference example with the MNIST

Following is an example gdf to perform inference using the `vx_amd_migraphx` extension. The model used is a CNN pre-trained on the MNIST dataset.

```
import vx_amd_migraphx
import vx_nn

data input = image:28,28,U008
read input image_4.jpg
data a = scalar:FLOAT32,0.00392157
data b = scalar:FLOAT32,0.0
data reverse_channel_order = scalar:BOOL,0
data image_tensor = tensor:4,{28,28,1,1},VX_TYPE_FLOAT32,0
node com.amd.nn_extension.convert_image_to_tensor input image_tensor a b reverse_channel_order

data model = scalar:STRING,"mnist-8.onnx"
data output_tensor = tensor:2,{10,1},VX_TYPE_FLOAT32,0

node com.amd.amd_migraphx_node model image_tensor output_tensor
write output_tensor out_mnist.f32
```

For additional examples for using the `vx_amd_migraphx` extension, please see [amd_migraphx_tests](https://github.com/ROCm/MIVisionX/tree/master/tests/amd_migraphx_tests/) section.

**NOTE:** OpenVX and the OpenVX logo are trademarks of the Khronos Group Inc.
