# OpenVX Custom Node Extension Library

`vx_amd_custom` is an OpenVX AMD custom node extension module. This module currently has a single extension node named`com.amd.custom_extension.custom_layer`.
This layer takes an input tensor and produces an output tensor using one of the custom functions specified by the user

More details of the usage and implementation of a new custion function can be found in
    - [Creating Custom](./CustomNode.md)

## Build Instructions
It is built with MIVisionX package.

### Pre-requisites

* AMD OpenVX&trade; library
* ROCM installed system with AMD GPU
* [ROCm](https://rocmdocs.amd.com/en/latest/)
    
### Example 1: Using custom extension with example "Copy" function and CPU backend

To show the usage of custom extension, an example function to "Copy" is implemented in custom_lib module.
The follwing is the gdf to test it using runvx utility
``` 
import vx_amd_custom

# read and initialize input tensor
data input_1 = tensor:4,{3,1,1,1},FLOAT32,0

# please create a binary file to store 3 float values of input tensor and read the values into the tensor data
read input_1 input_tensor_1.bin

data output = tensor:4,{3,1,1,1},FLOAT32,0

data function = scalar:UINT32,0     # function 0 corresponds to default (Copy)
data backend = scalar:UINT32,0      # (0)CPU (1)GPU backend
node com.amd.custom_extension.custom_layer input_1 function backend NULL output
write output out_tensor_1.bin

```
* To run the gdf using runvx use the command "runvx example.gdf"
* After running the gdf using the runvx utility, you can see the out_tensor_1.bin will have the same data as input tensor

### Example 2: Using custom extension with example "Copy" function and GPU backend

To show the usage of custom extension an example function to "Copy" is implemented in custom_lib module.
The follwing is the gdf to test it using runvx utility

``` 
import vx_amd_custom

# read and initialize input tensor
data input_1 = tensor:4,{3,1,1,1},FLOAT32,0

# please create a binary file to store 3 float values of input tensor
read input_1 input_tensor_1.bin

data output = tensor:4,{3,1,1,1},FLOAT32,0

data function = scalar:UINT32,0     #function 0 corresponds to default (Copy)
data backend = scalar:UINT32,1      # (0) CPU (1) GPU
node com.amd.custom_extension.custom_layer input_1 function backend NULL output
write output out_tensor_1.bin

```
* To run the gdf using runvx use the command "runvx -affinity:GPU example.gdf"
* After running the gdf using the runvx utility, you can see the out_tensor_1.bin will have the same data as input tensor

**NOTE:** OpenVX and the OpenVX logo are trademarks of the Khronos Group Inc.
