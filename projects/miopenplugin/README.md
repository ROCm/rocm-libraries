# MIOpen Legacy Plugin
A plugin wrapping MIOpen in order to provide engines to solve some hipDNN graphs.

## Building
This plugin can be built as part of the MIOpen project or as a standalone plugin.

### Building as a standalone plugin
In order to build the plugin standalone, you will need to have installed hipDNN and MIOpen on the system first.

1. Navigate to the `miopenplugin` directory.
1. Make a build directory, `mkdir build && cd build`.
1. Run `cmake -DCMAKE_CXX_COMPILER=<path to amdclang>/clang++ ..` to configure the build.
1. Run `ninja` to build the plugin.
