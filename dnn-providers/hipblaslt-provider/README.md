# hipBLASLt Provider Plugin
The hipBLASLt provider plugin is a wrapping around hipBLASLt that provides engines to solve certain hipDNN graphs.

:construction: **This project is under active development** :construction:

## Building

### Building with the superbuild
Build hipDNN and hipblaslt-provider together from the rocm-libraries root using the superbuild. See [Superbuild](../../projects/hipdnn/docs/Superbuild.md) for details.

```bash
cmake --preset hipdnn
cmake --build --preset default
```

### Building as a standalone plugin
To build the plugin standalone, first install hipDNN and hipBLASLt on the system and then follow these steps:

1. Navigate to the `dnn-providers/hipblaslt-provider` directory.
1. Make a build directory using `mkdir build && cd build`.
1. Configure the build using `cmake -DCMAKE_CXX_COMPILER=<path to amdclang>/clang++ ..`.
1. Finally, run `ninja` to build the plugin.

## Operation support

The list of supported operations is described in [Operation Support](docs/OperationSupport.md) documentation.
