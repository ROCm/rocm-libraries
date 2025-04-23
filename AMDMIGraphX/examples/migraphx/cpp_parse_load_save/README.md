# Parsing, Loading, and Saving MIGraphX Programs

## Description
This examples demonstrates how to parse, load, and save a graph program using the MIGraphX C++ API. 

## Parsing
Computation graphs that have been saved in a compatible serialized format, such as [ONNX](https://onnx.ai/get-started.html), can be read in by MIGraphX to create a runable program. 

```
migraphx::program p;
unsigned batch = 1; //Or read in as argument
migraphx::onnx_options options;
options.set_default_dim_value(batch);
p = parse_onnx(input_file, options);
```

## Saving
An instantiated migraphx::program object can then be serialized to MessagePack (.mxr) format and saved so that it can be loaded for future uses. 

A program can be saved with either of the following:
```
migraphx::program p = ... <migraphx::program>;
migraphx::save(p, output_file); 
```

```
migraphx::program p = ... <migraphx::program>;
migraphx::file_options options;
options.set_file_format("msgpack");
migraphx::save(p, output_file, options);
```

## Loading
Similarly, graphs that have been previously parsed, and possibly compiled, and then saved in either MessagePack or JSON format can be loaded at later time. 

MessagePack is the default format, and can be loaded with either:
```
migraphx::program p;
p = migraphx::load(input_file);
```

```
migraphx::program p;
migraphx::file_options options;
options.set_file_format("msgpack");
p = migraphx::load(input_file, options);
```
To load a program that has been saved in JSON format:
```
migraphx::program p;
migraphx::file_options options;
options.set_file_format("json");
p = migraphx::load(input_file, options);
```


## Running the Example
The provided example [`parse_load_save.cpp`](./parse_load_save.cpp) has these features implemented to allow for comparing outputs. 

To compile and run the example from this directory:
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```
There will now be an executable named `parse_load_save` with the following usage:
```
$ ./parse_load_save <input_file> [options]
options:
	--parse onnx
	--load  json/msgpack
	--save  <output_file>
```

The program will then attempt to parse or load the graph file, print out its internal graph structure if successful, and optionally save the program to a given file name.
