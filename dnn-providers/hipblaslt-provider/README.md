# hipBLASLt Provider Plugin
A plugin wrapping hipBLASLt in order to provide engines to solve some hipDNN graphs.

:construction: **This project is under active development** :construction:

## Building
This plugin is built as a standalone plugin. To build the plugin you need to have installed hipDNN and hipBLASLt on the system first.

1. Navigate to the `dnn-providers/hipblaslt-provider` directory.
1. Make a build directory, `mkdir build && cd build`.
1. Run `cmake -DCMAKE_CXX_COMPILER=<path to amdclang>/clang++ ..` to configure the build.
1. Run `ninja` to build the plugin.
