# MIGraphX Driver

## Description
The MIGraphX driver is a tool that allows you to utilize many of the core functions of MIGraphX without having to write your own program. 

## How to Use this Example

The MIGraphX driver is installed with MIGraphX and can be found in `/opt/rocm/bin/migraphx-driver`, or in `AMDMIGraphX/build/bin/migraphx-driver` after building the source code. 

See below for a comprehensive list of commands and option arguments, as well as some usage examples.

### Commands
| Command | Description                                                                |
| ------- | -------------------------------------------------------------------------- |
| op      | When followed by the option --list or -l, prints all operators of MIGraphX |
| params  | Prints the input and output parameter shapes                               |
| run     | Compiles, allocates parameters, evaluates, and prints input graph          |
| read    | Loads and prints input graph                                               |
| compile | Compiles and prints input graph                                            |
| verify  | Runs reference and GPU implementations and checks outputs for consistency  |
| perf    | Compiles and runs input graph then prints performance report               |

### Options
| Option                                   | Description                                               |
| ---------------------------------------- | --------------------------------------------------------- |
| --help \| -h                             | Show help                                                 |
| --test                                   | Test MIGraphX with Single Layer GEMM Model                |
| --onnx                                   | Load file as onnx graph                                   |
| --tf                                     | Load file as a tensorflow graph                           |
| --migraphx                               | Load file as a migraphx graph                             |
| --migraphx-json                          | Load file as a migraphx JSON graph                        |
| --batch                                  | For a static model, set batch size. For a dynamic batch model, sets the batch size at runtime.|
| --nhwc                                   | Treat tensorflow format as nhwc                           |
| --nchw                                   | Treat tensorflow format as nchw                           |
| --skip-unknown-operators                 | Skip unknown operators when parsing and continue to parse |
| --trim \| -t                             | Trim instructions from the end                            |
| --optimize \| -O                         | Optimize when reading                                     |
| --graphviz \| -g                         | Print out a graphviz representation                       |
| --brief                                  | Make the output brief                                     |
| --cpp                                    | Print out the program as cpp program                      |
| --json                                   | Print out program as json                                 |
| --text                                   | Print out program in text format                          |
| --binary                                 | Print out program in binary format                        |
| --output \| -o                           | Output to file                                            |
| --fill0                                  | Fill parameter with 0s                                    |
| --fill1                                  | Fill parameter with 1s                                    |
| --input-dim                              | Set static dimensions of a parameter                      |
| --dyn-input-dim                          | Set dynamic dimensions of a parameter                     |
| --default-dyn-dim                        | Set default dynamic dimension                             |
| --gpu                                    | Compile on the gpu                                        |
| --cpu                                    | Compile on the cpu                                        |
| --ref                                    | Compile on the reference implementation                   |
| --enable-offload-copy                    | Enable implicit offload copying                           |
| --disable-fast-math                      | Disable fast math optimization                            |
| --exhaustive-tune                        | Enable exhaustive search to find fastest kernel           |
| --fp16                                   | Quantize for fp16                                         |
| --int8                                   | Quantize for int8                                         |
| --fp8                                    | Quantize for Float8E4M3FNUZ type                          |
| --rms-tol                                | Tolerance for the RMS error (Default: 0.001)              |
| --atol                                   | Tolerance for elementwise absolute difference (Default: 0.001) |
| --rtol                                   | Tolerance for elementwise relative difference (Default: 0.001) |
| --per-instruction \| -i                  | Verify each instruction                                   |
| --reduce \| -r                           | Reduce program and verify                                 |
| --iterations \| -n                       | Number of iterations to run for perf report               |
| --list \| -l                             | List all the operators of MIGraphX                        |

## Usage Examples
The examples below supply a simple MNIST ConvNet as the input graph. Models of higher complexity will have considerably larger outputs in most cases.

##### Example: op
```
$ /opt/rocm/bin/migraphx-driver op --list
```

<details>
<summary>View output</summary>

```
@literal
@param
@return
abs
acos
acosh
add
argmax
argmin
as_shape
asin
asinh
atan
atanh
batch_norm_inference
broadcast
capture
ceil
check_context::migraphx::gpu::context
clip
concat
contiguous
convert
convolution
cos
cosh
deconvolution
div
dot
elu
equal
erf
exp
flatten
floor
gather
gpu::abs
gpu::acos
gpu::acosh
gpu::add
gpu::add_clip
gpu::add_gelu
gpu::add_gelu_new
gpu::add_relu
gpu::add_tanh
gpu::argmax
gpu::argmin
gpu::asin
gpu::asinh
gpu::atan
gpu::atanh
gpu::batch_norm_inference
gpu::ceil
gpu::clip
gpu::concat
gpu::contiguous
gpu::conv_bias
gpu::conv_bias_relu
gpu::convert
gpu::convolution
gpu::cos
gpu::cosh
gpu::deconv
gpu::div
gpu::elu
gpu::equal
gpu::erf
gpu::exp
gpu::floor
gpu::gather
gpu::gelu
gpu::gelu_new
gpu::gemm
gpu::greater
gpu::layernorm
gpu::leaky_relu
gpu::less
gpu::log
gpu::logsoftmax
gpu::lrn
gpu::max
gpu::min
gpu::mul
gpu::mul_add
gpu::mul_add_relu
gpu::pad
gpu::pooling
gpu::pow
gpu::prelu
gpu::quant_convolution
gpu::quant_gemm
gpu::recip
gpu::record_event
gpu::reduce_max
gpu::reduce_mean
gpu::reduce_min
gpu::reduce_prod
gpu::reduce_sum
gpu::relu
gpu::rnn_var_sl_last_output
gpu::rnn_var_sl_shift_output
gpu::rnn_var_sl_shift_sequence
gpu::round
gpu::rsqrt
gpu::set_stream
gpu::sigmoid
gpu::sign
gpu::sin
gpu::sinh
gpu::softmax
gpu::sqdiff
gpu::sqrt
gpu::sub
gpu::tan
gpu::tanh
gpu::triadd
gpu::triadd_clip
gpu::triadd_relu
gpu::triadd_sigmoid
gpu::triadd_tanh
gpu::wait_event
greater
gru
hip::allocate
hip::copy
hip::copy_from_gpu
hip::copy_to_gpu
hip::hip_allocate_memory
hip::hip_copy_literal
identity
im2col
leaky_relu
less
load
log
logsoftmax
lrn
lstm
max
min
mul
multibroadcast
neg
outline
pad
pooling
pow
prelu
quant_convolution
quant_dot
recip
reduce_max
reduce_mean
reduce_min
reduce_prod
reduce_sum
ref::batch_norm_inference
ref::convolution
ref::deconvolution
ref::dot
ref::elu
ref::im2col
ref::leaky_relu
ref::logsoftmax
ref::lrn
ref::op
ref::pad
ref::pooling_average
ref::pooling_max
ref::quant_convolution
ref::rnn_var_sl_last_output
ref::softmax
relu
reshape
rnn
rnn_last_cell_output
rnn_last_hs_output
rnn_var_sl_last_output
rnn_var_sl_shift_output
rnn_var_sl_shift_sequence
round
rsqrt
scalar
sigmoid
sign
sin
sinh
slice
softmax
sqdiff
sqrt
squeeze
sub
tan
tanh
transpose
undefined
unknown:
unsqueeze
```

</details>
<br/><br/>

##### Example: params
```
$ /opt/rocm/bin/migraphx-driver params simple_graph.pb 
```

<details>
<summary>View output</summary>

```
Reading: simple_graph.pb
x: float_type, {1, 28, 28}, {784, 28, 1}
```

</details>
<br/><br/>

##### Example: run (onnx file input)
```
$ /opt/rocm/bin/migraphx-driver run --onnx simple_graph.onnx
```

<details>
<summary>View output</summary>

```
Compiling ... 
Reading: simple_graph.onnx
@0 = check_context::migraphx::gpu::context -> float_type, {}, {}
@1 = hip::hip_allocate_memory[shape=float_type, {256}, {1},id=scratch] -> float_type, {256}, {1}
@2 = hip::hip_copy_literal[id=@literal:1] -> float_type, {784, 128}, {128, 1}
x:0 = @param:x:0 -> float_type, {1, 28, 28}, {784, 28, 1}
@3 = reshape[dims={-1, 784}](x:0) -> float_type, {1, 784}, {784, 1}
@4 = load[offset=0,end=512](@1) -> float_type, {1, 128}, {128, 1}
@5 = gpu::gemm[alpha=1,beta=0](@3,@2,@4) -> float_type, {1, 128}, {128, 1}
@6 = hip::hip_copy_literal[id=@literal:0] -> float_type, {128}, {1}
@7 = hip::hip_copy_literal[id=@literal:2] -> float_type, {10}, {1}
@8 = hip::hip_copy_literal[id=@literal:3] -> float_type, {128, 10}, {10, 1}
@9 = multibroadcast[output_lens={1, 128}](@6) -> float_type, {1, 128}, {0, 1}
@10 = load[offset=512,end=1024](@1) -> float_type, {1, 128}, {128, 1}
@11 = gpu::add_relu(@5,@9,@10) -> float_type, {1, 128}, {128, 1}
@12 = load[offset=0,end=40](@1) -> float_type, {1, 10}, {10, 1}
@13 = gpu::gemm[alpha=1,beta=0](@11,@8,@12) -> float_type, {1, 10}, {10, 1}
@14 = multibroadcast[output_lens={1, 10}](@7) -> float_type, {1, 10}, {0, 1}
@15 = load[offset=40,end=80](@1) -> float_type, {1, 10}, {10, 1}
@16 = gpu::add(@13,@14,@15) -> float_type, {1, 10}, {10, 1}
#output_0 = @param:#output_0 -> float_type, {1, 10}, {10, 1}
@17 = gpu::softmax[axis=1](@16,#output_0) -> float_type, {1, 10}, {10, 1}
@18 = @return(@17)

Allocating params ... 
@0 = check_context::migraphx::gpu::context -> float_type, {}, {}
@1 = hip::hip_allocate_memory[shape=float_type, {256}, {1},id=scratch] -> float_type, {256}, {1}
@2 = hip::hip_copy_literal[id=@literal:1] -> float_type, {784, 128}, {128, 1}
x:0 = @param:x:0 -> float_type, {1, 28, 28}, {784, 28, 1}
@3 = reshape[dims={-1, 784}](x:0) -> float_type, {1, 784}, {784, 1}
@4 = load[offset=0,end=512](@1) -> float_type, {1, 128}, {128, 1}
@5 = gpu::gemm[alpha=1,beta=0](@3,@2,@4) -> float_type, {1, 128}, {128, 1}
@6 = hip::hip_copy_literal[id=@literal:0] -> float_type, {128}, {1}
@7 = hip::hip_copy_literal[id=@literal:2] -> float_type, {10}, {1}
@8 = hip::hip_copy_literal[id=@literal:3] -> float_type, {128, 10}, {10, 1}
@9 = multibroadcast[output_lens={1, 128}](@6) -> float_type, {1, 128}, {0, 1}
@10 = load[offset=512,end=1024](@1) -> float_type, {1, 128}, {128, 1}
@11 = gpu::add_relu(@5,@9,@10) -> float_type, {1, 128}, {128, 1}
@12 = load[offset=0,end=40](@1) -> float_type, {1, 10}, {10, 1}
@13 = gpu::gemm[alpha=1,beta=0](@11,@8,@12) -> float_type, {1, 10}, {10, 1}
@14 = multibroadcast[output_lens={1, 10}](@7) -> float_type, {1, 10}, {0, 1}
@15 = load[offset=40,end=80](@1) -> float_type, {1, 10}, {10, 1}
@16 = gpu::add(@13,@14,@15) -> float_type, {1, 10}, {10, 1}
#output_0 = @param:#output_0 -> float_type, {1, 10}, {10, 1}
@17 = gpu::softmax[axis=1](@16,#output_0) -> float_type, {1, 10}, {10, 1}
@18 = @return(@17)
```

</details>
<br/><br/>

##### Example: read
```
$ /opt/rocm/bin/migraphx-driver read simple_graph.pb 
```

<details>
<summary>View output</summary>

```
Reading: simple_graph.pb
@0 = @literal{0.0136018, -0.0839988, 0.0375392, 0.0613085, -0.125795, 0.176185, 0.0761055, 0.0093384, -0.110057, -0.170587} -> float_type, {10}, {1}
@1 = @literal{ ... } -> float_type, {128, 10}, {10, 1}
@2 = @literal{ ... } -> float_type, {128}, {1}
@3 = @literal{ ... } -> float_type, {784, 128}, {128, 1}
@4 = @literal{-1, 784} -> int32_type, {2}, {1}
x = @param:x -> float_type, {1, 28, 28}, {784, 28, 1}
@5 = reshape[dims={-1, 784}](x) -> float_type, {1, 784}, {784, 1}
@6 = identity(@3) -> float_type, {784, 128}, {128, 1}
@7 = dot[alpha=1,beta=1](@5,@6) -> float_type, {1, 128}, {128, 1}
@8 = identity(@2) -> float_type, {128}, {1}
@9 = broadcast[axis=1,dims={1, 128}](@8) -> float_type, {1, 128}, {0, 1}
@10 = add(@7,@9) -> float_type, {1, 128}, {128, 1}
@11 = relu(@10) -> float_type, {1, 128}, {128, 1}
@12 = identity(@1) -> float_type, {128, 10}, {10, 1}
@13 = dot[alpha=1,beta=1](@11,@12) -> float_type, {1, 10}, {10, 1}
@14 = identity(@0) -> float_type, {10}, {1}
@15 = broadcast[axis=1,dims={1, 10}](@14) -> float_type, {1, 10}, {0, 1}
@16 = add(@13,@15) -> float_type, {1, 10}, {10, 1}
@17 = softmax[axis=1](@16) -> float_type, {1, 10}, {10, 1}
@18 = identity(@17) -> float_type, {1, 10}, {10, 1}
```

</details>
<br/><br/>

##### Example: compile (on GPU, quantized for fp16)
```
$ /opt/rocm/bin/migraphx-driver compile --gpu --fp16 simple_graph.pb
```

<details>
<summary>View output</summary>

```
Compiling ... 
Reading: simple_graph.pb
@0 = check_context::migraphx::gpu::context -> float_type, {}, {}
@1 = hip::hip_allocate_memory[shape=float_type, {456}, {1},id=scratch] -> float_type, {456}, {1}
@2 = hip::hip_copy_literal[id=@literal:0] -> half_type, {784, 128}, {128, 1}
@3 = load[offset=256,end=1824](@1) -> half_type, {1, 28, 28}, {784, 28, 1}
x = @param:x -> float_type, {1, 28, 28}, {784, 28, 1}
@4 = gpu::convert[target_type=1](x,@3) -> half_type, {1, 28, 28}, {784, 28, 1}
@5 = reshape[dims={-1, 784}](@4) -> half_type, {1, 784}, {784, 1}
@6 = load[offset=0,end=256](@1) -> half_type, {1, 128}, {128, 1}
@7 = gpu::gemm[alpha=1,beta=0](@5,@2,@6) -> half_type, {1, 128}, {128, 1}
@8 = hip::hip_copy_literal[id=@literal:2] -> half_type, {128, 10}, {10, 1}
@9 = hip::hip_copy_literal[id=@literal:1] -> half_type, {128}, {1}
@10 = hip::hip_copy_literal[id=@literal:3] -> half_type, {10}, {1}
@11 = load[offset=256,end=512](@1) -> half_type, {1, 128}, {128, 1}
@12 = broadcast[axis=1,dims={1, 128}](@9) -> half_type, {1, 128}, {0, 1}
@13 = gpu::add_relu(@7,@12,@11) -> half_type, {1, 128}, {128, 1}
@14 = load[offset=0,end=20](@1) -> half_type, {1, 10}, {10, 1}
@15 = gpu::gemm[alpha=1,beta=0](@13,@8,@14) -> half_type, {1, 10}, {10, 1}
@16 = broadcast[axis=1,dims={1, 10}](@10) -> half_type, {1, 10}, {0, 1}
@17 = load[offset=20,end=40](@1) -> half_type, {1, 10}, {10, 1}
@18 = gpu::add(@15,@16,@17) -> half_type, {1, 10}, {10, 1}
@19 = load[offset=0,end=20](@1) -> half_type, {1, 10}, {10, 1}
@20 = gpu::softmax[axis=1](@18,@19) -> half_type, {1, 10}, {10, 1}
output = @param:output -> float_type, {1, 10}, {10, 1}
@21 = gpu::convert[target_type=2](@20,output) -> float_type, {1, 10}, {10, 1}
```

</details>
<br/><br/>

##### Example: verify
```
$ /opt/rocm/bin/migraphx-driver verify simple_graph.pb
```

<details>
<summary>View output</summary>

```
Reading: simple_graph.pb
@0 = @literal{0.0136018, -0.0839988, 0.0375392, 0.0613085, -0.125795, 0.176185, 0.0761055, 0.0093384, -0.110057, -0.170587} -> float_type, {10}, {1}
@1 = @literal{ ... } -> float_type, {128, 10}, {10, 1}
@2 = @literal{ ... } -> float_type, {128}, {1}
@3 = @literal{ ... } -> float_type, {784, 128}, {128, 1}
@4 = @literal{-1, 784} -> int32_type, {2}, {1}
x = @param:x -> float_type, {1, 28, 28}, {784, 28, 1}
@5 = reshape[dims={-1, 784}](x) -> float_type, {1, 784}, {784, 1}
@6 = identity(@3) -> float_type, {784, 128}, {128, 1}
@7 = dot[alpha=1,beta=1](@5,@6) -> float_type, {1, 128}, {128, 1}
@8 = identity(@2) -> float_type, {128}, {1}
@9 = broadcast[axis=1,dims={1, 128}](@8) -> float_type, {1, 128}, {0, 1}
@10 = add(@7,@9) -> float_type, {1, 128}, {128, 1}
@11 = relu(@10) -> float_type, {1, 128}, {128, 1}
@12 = identity(@1) -> float_type, {128, 10}, {10, 1}
@13 = dot[alpha=1,beta=1](@11,@12) -> float_type, {1, 10}, {10, 1}
@14 = identity(@0) -> float_type, {10}, {1}
@15 = broadcast[axis=1,dims={1, 10}](@14) -> float_type, {1, 10}, {0, 1}
@16 = add(@13,@15) -> float_type, {1, 10}, {10, 1}
@17 = softmax[axis=1](@16) -> float_type, {1, 10}, {10, 1}
@18 = identity(@17) -> float_type, {1, 10}, {10, 1}

@0 = @literal{0.0136018, -0.0839988, 0.0375392, 0.0613085, -0.125795, 0.176185, 0.0761055, 0.0093384, -0.110057, -0.170587} -> float_type, {10}, {1}
@1 = @literal{ ... } -> float_type, {128, 10}, {10, 1}
@2 = @literal{ ... } -> float_type, {128}, {1}
@3 = @literal{ ... } -> float_type, {784, 128}, {128, 1}
@4 = @literal{-1, 784} -> int32_type, {2}, {1}
x = @param:x -> float_type, {1, 28, 28}, {784, 28, 1}
@5 = reshape[dims={-1, 784}](x) -> float_type, {1, 784}, {784, 1}
@6 = identity(@3) -> float_type, {784, 128}, {128, 1}
@7 = dot[alpha=1,beta=1](@5,@6) -> float_type, {1, 128}, {128, 1}
@8 = identity(@2) -> float_type, {128}, {1}
@9 = broadcast[axis=1,dims={1, 128}](@8) -> float_type, {1, 128}, {0, 1}
@10 = add(@7,@9) -> float_type, {1, 128}, {128, 1}
@11 = relu(@10) -> float_type, {1, 128}, {128, 1}
@12 = identity(@1) -> float_type, {128, 10}, {10, 1}
@13 = dot[alpha=1,beta=1](@11,@12) -> float_type, {1, 10}, {10, 1}
@14 = identity(@0) -> float_type, {10}, {1}
@15 = broadcast[axis=1,dims={1, 10}](@14) -> float_type, {1, 10}, {0, 1}
@16 = add(@13,@15) -> float_type, {1, 10}, {10, 1}
@17 = softmax[axis=1](@16) -> float_type, {1, 10}, {10, 1}
@18 = identity(@17) -> float_type, {1, 10}, {10, 1}

@0 = @literal{0.0136018, -0.0839988, 0.0375392, 0.0613085, -0.125795, 0.176185, 0.0761055, 0.0093384, -0.110057, -0.170587} -> float_type, {10}, {1}
@1 = @literal{ ... } -> float_type, {128, 10}, {10, 1}
@2 = @literal{ ... } -> float_type, {128}, {1}
@3 = @literal{ ... } -> float_type, {784, 128}, {128, 1}
x = @param:x -> float_type, {1, 28, 28}, {784, 28, 1}
@4 = ref::reshape[dims={-1, 784}](x) -> float_type, {1, 784}, {784, 1}
@5 = ref::identity(@3) -> float_type, {784, 128}, {128, 1}
@6 = ref::dot[alpha=1,beta=1](@4,@5) -> float_type, {1, 128}, {128, 1}
@7 = ref::identity(@2) -> float_type, {128}, {1}
@8 = ref::broadcast[axis=1,dims={1, 128}](@7) -> float_type, {1, 128}, {0, 1}
@9 = ref::contiguous(@8) -> float_type, {1, 128}, {128, 1}
@10 = ref::add(@6,@9) -> float_type, {1, 128}, {128, 1}
@11 = ref::relu(@10) -> float_type, {1, 128}, {128, 1}
@12 = ref::identity(@1) -> float_type, {128, 10}, {10, 1}
@13 = ref::dot[alpha=1,beta=1](@11,@12) -> float_type, {1, 10}, {10, 1}
@14 = ref::identity(@0) -> float_type, {10}, {1}
@15 = ref::broadcast[axis=1,dims={1, 10}](@14) -> float_type, {1, 10}, {0, 1}
@16 = ref::contiguous(@15) -> float_type, {1, 10}, {10, 1}
@17 = ref::add(@13,@16) -> float_type, {1, 10}, {10, 1}
@18 = ref::softmax[axis=1](@17) -> float_type, {1, 10}, {10, 1}
@19 = ref::identity(@18) -> float_type, {1, 10}, {10, 1}

@0 = check_context::migraphx::gpu::context -> float_type, {}, {}
@1 = hip::hip_allocate_memory[shape=float_type, {256}, {1},id=scratch] -> float_type, {256}, {1}
@2 = hip::hip_copy_literal[id=@literal:3] -> float_type, {784, 128}, {128, 1}
x = @param:x -> float_type, {1, 28, 28}, {784, 28, 1}
@3 = load[offset=0,end=512](@1) -> float_type, {1, 128}, {128, 1}
@4 = reshape[dims={-1, 784}](x) -> float_type, {1, 784}, {784, 1}
@5 = gpu::gemm[alpha=1,beta=0](@4,@2,@3) -> float_type, {1, 128}, {128, 1}
@6 = hip::hip_copy_literal[id=@literal:1] -> float_type, {128, 10}, {10, 1}
@7 = hip::hip_copy_literal[id=@literal:2] -> float_type, {128}, {1}
@8 = hip::hip_copy_literal[id=@literal:0] -> float_type, {10}, {1}
@9 = load[offset=512,end=1024](@1) -> float_type, {1, 128}, {128, 1}
@10 = broadcast[axis=1,dims={1, 128}](@7) -> float_type, {1, 128}, {0, 1}
@11 = gpu::add_relu(@5,@10,@9) -> float_type, {1, 128}, {128, 1}
@12 = load[offset=40,end=80](@1) -> float_type, {1, 10}, {10, 1}
@13 = gpu::gemm[alpha=1,beta=0](@11,@6,@12) -> float_type, {1, 10}, {10, 1}
@14 = load[offset=0,end=40](@1) -> float_type, {1, 10}, {10, 1}
@15 = broadcast[axis=1,dims={1, 10}](@8) -> float_type, {1, 10}, {0, 1}
@16 = gpu::add(@13,@15,@14) -> float_type, {1, 10}, {10, 1}
output = @param:output -> float_type, {1, 10}, {10, 1}
@17 = gpu::softmax[axis=1](@16,output) -> float_type, {1, 10}, {10, 1}
```

</details>
<br/><br/>

##### Example: perf
```
$ /opt/rocm/bin/migraphx-driver perf simple_graph.pb
```

<details>
<summary>View output</summary>

```
Compiling ... 
Reading: simple_graph.pb
@0 = check_context::migraphx::gpu::context -> float_type, {}, {}
@1 = hip::hip_allocate_memory[shape=float_type, {256}, {1},id=scratch] -> float_type, {256}, {1}
@2 = hip::hip_copy_literal[id=@literal:3] -> float_type, {784, 128}, {128, 1}
@3 = load[offset=0,end=512](@1) -> float_type, {1, 128}, {128, 1}
x = @param:x -> float_type, {1, 28, 28}, {784, 28, 1}
@4 = reshape[dims={-1, 784}](x) -> float_type, {1, 784}, {784, 1}
@5 = gpu::gemm[alpha=1,beta=0](@4,@2,@3) -> float_type, {1, 128}, {128, 1}
@6 = hip::hip_copy_literal[id=@literal:1] -> float_type, {128, 10}, {10, 1}
@7 = hip::hip_copy_literal[id=@literal:0] -> float_type, {10}, {1}
@8 = hip::hip_copy_literal[id=@literal:2] -> float_type, {128}, {1}
@9 = broadcast[axis=1,dims={1, 128}](@8) -> float_type, {1, 128}, {0, 1}
@10 = load[offset=512,end=1024](@1) -> float_type, {1, 128}, {128, 1}
@11 = gpu::add_relu(@5,@9,@10) -> float_type, {1, 128}, {128, 1}
@12 = load[offset=0,end=40](@1) -> float_type, {1, 10}, {10, 1}
@13 = gpu::gemm[alpha=1,beta=0](@11,@6,@12) -> float_type, {1, 10}, {10, 1}
@14 = broadcast[axis=1,dims={1, 10}](@7) -> float_type, {1, 10}, {0, 1}
@15 = load[offset=40,end=80](@1) -> float_type, {1, 10}, {10, 1}
@16 = gpu::add(@13,@14,@15) -> float_type, {1, 10}, {10, 1}
output = @param:output -> float_type, {1, 10}, {10, 1}
@17 = gpu::softmax[axis=1](@16,output) -> float_type, {1, 10}, {10, 1}

Allocating params ... 
Running performance report ... 
@0 = check_context::migraphx::gpu::context -> float_type, {}, {}: 0.00057782ms, 1%
@1 = hip::hip_allocate_memory[shape=float_type, {256}, {1},id=scratch] -> float_type, {256}, {1}: 0.000295ms, 1%
@2 = hip::hip_copy_literal[id=@literal:3] -> float_type, {784, 128}, {128, 1}: 0.00027942ms, 1%
@3 = load[offset=0,end=512](@1) -> float_type, {1, 128}, {128, 1}: 0.000232ms, 1%
x = @param:x -> float_type, {1, 28, 28}, {784, 28, 1}: 0.0003206ms, 1%
@4 = reshape[dims={-1, 784}](x) -> float_type, {1, 784}, {784, 1}: 0.00033842ms, 1%
@5 = gpu::gemm[alpha=1,beta=0](@4,@2,@3) -> float_type, {1, 128}, {128, 1}: 0.212592ms, 52%
@6 = hip::hip_copy_literal[id=@literal:1] -> float_type, {128, 10}, {10, 1}: 0.00085822ms, 1%
@7 = hip::hip_copy_literal[id=@literal:0] -> float_type, {10}, {1}: 0.000382ms, 1%
@8 = hip::hip_copy_literal[id=@literal:2] -> float_type, {128}, {1}: 0.0003486ms, 1%
@9 = broadcast[axis=1,dims={1, 128}](@8) -> float_type, {1, 128}, {0, 1}: 0.000299ms, 1%
@10 = load[offset=512,end=1024](@1) -> float_type, {1, 128}, {128, 1}: 0.000234ms, 1%
@11 = gpu::add_relu(@5,@9,@10) -> float_type, {1, 128}, {128, 1}: 0.0416597ms, 11%
@12 = load[offset=0,end=40](@1) -> float_type, {1, 10}, {10, 1}: 0.0007548ms, 1%
@13 = gpu::gemm[alpha=1,beta=0](@11,@6,@12) -> float_type, {1, 10}, {10, 1}: 0.0733071ms, 18%
@14 = broadcast[axis=1,dims={1, 10}](@7) -> float_type, {1, 10}, {0, 1}: 0.00088142ms, 1%
@15 = load[offset=40,end=80](@1) -> float_type, {1, 10}, {10, 1}: 0.000408ms, 1%
@16 = gpu::add(@13,@14,@15) -> float_type, {1, 10}, {10, 1}: 0.0410144ms, 10%
output = @param:output -> float_type, {1, 10}, {10, 1}: 0.0010222ms, 1%
@17 = gpu::softmax[axis=1](@16,output) -> float_type, {1, 10}, {10, 1}: 0.0385636ms, 10%

Summary:
gpu::gemm: 0.285899ms, 69%
gpu::add_relu: 0.0416597ms, 11%
gpu::add: 0.0410144ms, 10%
gpu::softmax: 0.0385636ms, 10%
hip::hip_copy_literal: 0.00186824ms, 1%
load: 0.0016288ms, 1%
@param: 0.0013428ms, 1%
broadcast: 0.00118042ms, 1%
check_context::migraphx::gpu::context: 0.00057782ms, 1%
reshape: 0.00033842ms, 1%
hip::hip_allocate_memory: 0.000295ms, 1%

Rate: 2866.1/sec
Total time: 0.348906ms
Total instructions time: 0.414369ms
Overhead time: 0.00348144ms, -0.0654627ms
Overhead: 1%, -19%
```

</details>
<br/><br/>
