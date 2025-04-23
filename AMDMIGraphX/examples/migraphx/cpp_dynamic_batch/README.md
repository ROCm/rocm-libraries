# Running ONNX model with dynamic batch


## Description
This examples demonstrates how to run a graph program with dynamic batch using the MIGraphX C++ API. 


## Creating dynamic dimension objects
`dynamic_dimension` objects are used in MIGraphX to specify a range of dimension values from a minimum value to a maximum value and optimal values that the tensor can be at model evaluation time.
A dynamic shape is defined by a list of `dynamic_dimensions` while a static shape only has fixed dimension values.
For example, a `dynamic_dimension` with `{min:1, max:10, optimals:{1, 4, 10}}` means that the dimension can be any value from 1 through 10 with the optimal values being 1, 4, and 10.
Supplied optimal values may allow MIGraphX to optimize the program for those specific shapes.

A fixed `dynamic_dimension` can be specified by setting the `min` and `max` to the same value (ex. `{min:3, max:3}`).
A dynamic shape specified solely by fixed `dynamic_dimension` objects will be converted to a static shape during parsing.
This can be useful for setting a static shape using the `set_dyn_input_parameter_shape()` method discussed later in this document.


## Parsing
ONNX graphs [ONNX](https://onnx.ai/get-started.html) can be parsed by MIGraphX to create a runnable program with dynamic batch sizes.
The dynamic batch range must be specified by a `dynamic_dimension` object.

One method to set the `dynamic_dimension` object works for ONNX files that only have symbolic variables for the batch dimensions:
```
migraphx::program p;
migraphx::onnx_options options;
options.set_default_dyn_dim_value(migraphx::dynamic_dimension{1, 4, {2, 4}});
p = parse_onnx(input_file, options);
```

Another option that can run any ONNX model with dynamic batch sizes uses the dynamic input map where the entire shape of the input parameter is supplied:
```
migraphx::program p;
migraphx::onnx_options options;
migraphx::dynamic_dimensions dyn_dims = {migraphx::dynamic_dimension{1, 4, {2, 4}},
                                         migraphx::dynamic_dimension{3, 3},
                                         migraphx::dynamic_dimension{4, 4},
                                         migraphx::dynamic_dimension{4, 4}};
options.set_dyn_input_parameter_shape("input", dyn_dims);
p = parse_onnx(input_file, options);
```

## Compiling
Currently the MIGraphX C/C++ API requires that `offload_copy` be enabled for compiling dynamic batch programs.
Here is a snippet of compiling a model with `offload_copy` enabled:
```
migraphx::compile_options c_options;
c_options.set_offload_copy();
p.compile(migraphx::target("gpu"), c_options);
```
where `p` is the `migraphx::program`.


## Saving and Loading
A dynamic batch MIGraphX program can be saved and loaded to/from a MXR file the same way as a fully static shape program.


## Executing the dynamic batch model
The compiled dynamic batch model can be executed the same way as a static model by supplying the input data as `arguments` in a `program_parameters` object.


## Running the Example
Your ROCm installation could be installed in a location other than the one specified in the CMakeLists.txt.
You can set `LD_LIBRARY_PATH` or `CMAKE_PREFIX_PATH` to that location so that this program can still build.

The provided example is [`dynamic_batch.cpp`](./dynamic_batch.cpp)
To compile and run the example from this directory:
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```
There will now be an executable named `dynamic_batch` with the following usage:
```
$ ./dynamic_batch
```
