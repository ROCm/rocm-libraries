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

The [hipDNN developer image](../../projects/hipdnn/dockerfiles/README.md) supplies the third-party libraries. With the compatible hipDNN SDKs and MIOpen available, the following image recipe needs no manual dependency download, install, or fetch flag.

1. Navigate to the `dnn-providers/miopen-provider` directory.
1. Make a build directory, `mkdir build && cd build`.
1. Run `cmake -DCMAKE_CXX_COMPILER=<path to amdclang>/clang++ ..` to configure the build.
1. Run `ninja` to build the plugin.

Outside the image, this provider resolves third-party libraries like every other hipDNN component; see [Third-Party Libraries](../../projects/hipdnn/docs/Building.md#third-party-libraries) for the installed-package and fetch options. Configure from this provider's build directory:

```bash
cmake -G Ninja -DCMAKE_CXX_COMPILER=/path/to/amdclang/clang++ \
    -DCMAKE_PREFIX_PATH="/path/to/hipdnn-install;/path/to/rocm;/path/to/dependencies" ..
```

Beyond the shared third-party set, the prefixes must provide HIP, MIOpen, `hipdnn_data_sdk`, `hipdnn_flatbuffers_sdk`, `hipdnn_plugin_sdk`, and their transitive dependencies. The fetch opt-in supplies none of those.

## Project policies

This plugin is part of the hipDNN project. Shared project documentation and
policies are maintained in hipDNN:

- [hipDNN Overview](../../projects/hipdnn/README.md)
- [Contributing Guidelines](../../projects/hipdnn/CONTRIBUTING.md)
- [Security Policy](../../projects/hipdnn/SECURITY.md)
