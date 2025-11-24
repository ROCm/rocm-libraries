# MIOpen Legacy Plugin
A plugin wrapping MIOpen in order to provide engines to solve some hipDNN graphs.

## Building
This plugin can be built as part of the MIOpen project or as a standalone plugin.

### Building as part of hipDNN
1. Follow the build instructions for hipDNN defined in [Building hipDNN](../../docs/Building.md).
2. Currently the plugin build is defaulted to off.  Eventually it won't be a hipdnn build option.
  - To enable the plugin build as part of hipdnn, set `HIP_DNN_BUILD_PLUGINS=ON` in the CMake configuration.  `cmake -DHIP_DNN_BUILD_PLUGINS=ON ..`

### Building as a standalone plugin
In order to build the plugin standalone, you will need to have installed hipDNN and MIOpen on the system first.

1. Navigate to the `plugins/miopen_legacy_plugin` directory.
2. Make a build directory, `mkdir build && cd build`.
3. Run `cmake -DCMAKE_CXX_COMPILER=<path to amdclang>/clang++ ..` to configure the build.
4. Run `ninja` to build the plugin.
