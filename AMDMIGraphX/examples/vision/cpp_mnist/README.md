# Performing Inference Using C++ API

## Description
This example demonstrates how to perform inference using the MIGraphX C++ API. The model used is a convolutional network pre-trained on the MNIST dataset, and inference is performed on a random digit selected from the test set. 

## Content
- [Basic Setup](#Basic-Setup)
- [Quantization](#Quantization)
- [Compilation](#Compilation)
- [Preparing Input Data](#Preparing-Input-Data)
- [Evaluating Inputs and Handling Outputs](#Evaluating-Inputs-and-Handling-Outputs)
- [**Running this Example**](#Running-this-Example)

## Basic Setup
Before running inference, we must first instantiate a network graph and select a compilation target. See [this example](../cpp_parse_load_save) for more information about working with MIGraphX program objects. 
```
migraphx::program prog;
migraphx::onnx_options onnx_opts;
prog = parse_onnx("../mnist-8.onnx", onnx_opts);

std::string target_str;
if(CPU)
    target_str = "cpu";
else if(GPU)
    target_str = "gpu";
else
    target_str = "ref";
migraphx::target targ = migraphx::target(target_str.c_str());
```

## Quantization
Optionally, graph programs may be quantized to fp16 or int8 precision to improve performance and memory usage. 

##### Floating Point 16-bit Precision
To quantize using fp16, we simply add the following line:
```
migraphx::quantize_fp16(prog);
```

##### Integer 8-bit Precision
Int8 quantization requires calibration to accurately map ranges of floating point values onto integer values. 

To calibrate prior to inference, one or more inputs can be supplied as follows:
```
std::vector<float> calib_dig;
// ... read in data

migraphx::quantize_int8_options quant_opts;
migraphx::program_parameters quant_params;
auto param_shapes = prog.get_parameter_shapes();
for(auto&& name : param_shapes.names())
{
    quant_params.add(name, migraphx::argument(param_shapes[name], calib_dig.data()));
}

quant_opts.add_calibration_data(quant_params);
migraphx::quantize_int8(prog, targ, quant_opts);
```

## Compilation 
Network graphs saved in e.g. ONNX or protobuf format are not target-specific. In order to run inference, we must compile the graph into a target-specific program. 

Two options may be turned on when compiling:
- `set_offload_copy(bool value)`: For targets with offloaded memory (such as the gpu), this will insert instructions during compilation to copy the input parameters to the offloaded memory and to copy the final result from the offloaded memory back to main memory. Default value is `false` for offload_copy.
- `set_fast_math(bool value)`: Optimize math functions to use faster approximate versions. There may be slight accuracy degredation when enabled. Default value is `true` for fast_math. 

The following snippet assumes `targ` has been set as "gpu", and will compile the program without the fast_math optimization.
```
migraphx::compile_options comp_opts;
comp_opts.set_offload_copy();
prog.compile(targ, comp_opts);
``` 

To compile a program with the default options, we simply call:
```
prog.compile(targ);
```

The targets "ref" and "cpu" both compile the program to run on the CPU. The target "ref" is primarily used for correctness checking. The target "cpu" is under ongoing development and has more optimizations enabled. Additionally, the "cpu" target requires MIGraphX to be built with the `-DMIGRAPHX_ENABLE_CPU=On` flag. Specifically, 
```
CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIGRAPHX_ENABLE_CPU=On ..
```

## Preparing Input Data
Now that we have a compiled program, the last step to perform infernce is to prepare the input data as program parameters. 
The first step is to read in the data and store it in a `std::vector<float>` we will in this case call `digit`. 
Next, we create a program parameter containing the data stored in `digit`:
```
migraphx::program_parameters prog_params;
auto param_shapes = prog.get_parameter_shapes();
for(auto&& name : param_shapes.names())
{
    prog_params.add(name, migraphx::argument(param_shapes[name], digit.data()));
}
```

## Evaluating Inputs and Handling Outputs
Now that everything is in place, the final step to run inference is to call:
```
auto outputs = prog.eval(prog_params);
```

The output layer(s) will be returned and stored in `outputs`. Our network for this example returns a single output layer with the shape (1, 10). The index of the largest value in this output layer corresponds to the digit that the model has predicted. 
```
auto shape   = outputs[0].get_shape();
auto lengths = shape.lengths();
auto num_results = std::accumulate(lengths.begin(), lengths.end(), 1, std::multiplies<size_t>();
float* results = reinterpret_cast<float*>(outputs[0].data());
float* max     = std::max_element(results, results + num_results);
int answer     = max - results;
```

Other networks may require alternative processing of outputs. 


## Running this Example
This directory contains everything that is needed to perform inference on an MNIST digit. To create the executable:
```
$ mkdir build
$ cd build
$ CXX=/opt/rocm/llvm/bin/clang++ cmake ..
$ make
```
There will now be an executable named `mnist_inference` in the `build` directory. This can be run with or without options. Executing without any options will produce the following output:
```
Usage: ./mnist_inference [options]
options:
         -c, --cpu      Compile for CPU
         -g, --gpu      Compile for GPU
         -f, --fp16     FP16 Quantization
         -i, --int8     Int8 Quantization
               --cal    Int8 Calibration ON
         -p, --print    Print Graph at Each Stage


Parsing ONNX model...

Compiling program for ref...

Model input: 
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@%=@@@@@@@@@
@@@@@@@@@@@@@0+.   +@@@@@@@@
@@@@@@@@@@@0+   ..  0@@@@@@@
@@@@@@@@@@+    .00  #@@@@@@@
@@@@@@@@@%    .0@0  #@@@@@@@
@@@@@@@@@-  .*0@@%  #@@@@@@@
@@@@@@@@@0+#@@@@@%  #@@@@@@@
@@@@@@@@@@@@@@@@@*  #@@@@@@@
@@@@@@@@@@@@@====- -@@@@@@@@
@@@@@@@@@@@#-     .0@@@@@@@@
@@@@@@@@@#.  .*    =@@@@@@@@
@@@@@@@@%  =#@@.    %@@@@@@@
@@@@@@@+  -@@@-  +*  -#00@@@
@@@@@@+  =@@#- .#@@#*   .@@@
@@@@@=  %@#*  =0@@@@@%--0@@@
@@@@@   ..   =@@@@@@@@@@@@@@
@@@@@.    *=0@@@@@@@@@@@@@@@
@@@@@@%+=@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@

Model evaluating input...
Inference complete
Inference time: 0.022ms

Randomly chosen digit: 2
Result from inference: 2

CORRECT

```

*Note: the actual digit selected and printed will not necessarily be the same as shown above. 