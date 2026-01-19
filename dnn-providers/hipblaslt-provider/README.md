# hipBLASLt Provider Plugin
The hipBLASLt provider plugin is a wrapping around hipBLASLt that provides engines to solve certain hipDNN graphs.

:construction: **This project is under active development** :construction:

## Building
This plugin is built as a standalone plugin. To build the plugin, first install hipDNN and hipBLASLt on the system and then follow these steps:

1. Navigate to the `dnn-providers/hipblaslt-provider` directory.
1. Make a build directory using `mkdir build && cd build`.
1. Configure the build using `cmake -DCMAKE_CXX_COMPILER=<path to amdclang>/clang++ ..`.
1. Finally, run `ninja` to build the plugin.
