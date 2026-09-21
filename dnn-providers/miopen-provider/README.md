# MIOpen Provider Plugin
A plugin wrapping MIOpen in order to provide engines to solve some hipDNN graphs.

## Building

### Building with the superbuild
Build hipDNN and miopen-provider together from the rocm-libraries root using the superbuild. See [Superbuild](../../projects/hipdnn/docs/Superbuild.md) for details.

```bash
cmake --preset hipdnn
cmake --build --preset default
```

### Building as a standalone plugin
In order to build the plugin standalone, you will need to have installed hipDNN and MIOpen on the system first.

1. Navigate to the `dnn-providers/miopen-provider` directory.
1. Make a build directory, `mkdir build && cd build`.
1. Run `cmake -DCMAKE_CXX_COMPILER=<path to amdclang>/clang++ ..` to configure the build.
1. Run `ninja` to build the plugin.
