# HIP kernel provider plugin
A plugin that provides engines for HIP kernels to solve certain hipDNN graphs.

:construction: **This project is under active development** :construction:

## Building
This plugin should be built as a standalone plugin. To build the plugin, first install hipDNN on the system and then follow these steps:

1. Navigate to the `dnn-providers/hip-kernel-provider` directory.
2. Make a build directory using `mkdir build && cd build`.
3. Configure the build using `cmake -DCMAKE_CXX_COMPILER=<path to amdclang>/clang++ ..`.
4. Finally, run `ninja` to build the plugin.
