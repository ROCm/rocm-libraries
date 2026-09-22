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

The [hipDNN developer image](../../projects/hipdnn/dockerfiles/README.md) automatically provides GoogleTest/GoogleMock and spdlog in `/usr/local`. With the compatible hipDNN SDKs and MIOpen available, the following image recipe needs no manual dependency download, install, or fetch flag.

1. Navigate to the `dnn-providers/miopen-provider` directory.
1. Make a build directory, `mkdir build && cd build`.
1. Run `cmake -DCMAKE_CXX_COMPILER=<path to amdclang>/clang++ ..` to configure the build.
1. Run `ninja` to build the plugin.

Outside the image, choose one configure command from this provider's build directory:

```bash
# Installed packages only, using absolute paths.
cmake -G Ninja -DCMAKE_CXX_COMPILER=/path/to/amdclang/clang++ \
    -DCMAKE_PREFIX_PATH="/path/to/hipdnn-install;/path/to/rocm;/path/to/dependencies" \
    -DALLOW_FETCH_DEPS=OFF ..

# Alternatively, allow this standalone provider to fetch missing GoogleTest.
cmake -G Ninja -DCMAKE_CXX_COMPILER=/path/to/amdclang/clang++ \
    -DCMAKE_PREFIX_PATH="/path/to/hipdnn-install;/path/to/rocm;/path/to/dependencies" \
    -DALLOW_FETCH_DEPS=ON ..
```

The combined prefixes must provide HIP, MIOpen, `hipdnn_data_sdk`, `hipdnn_flatbuffers_sdk`, `hipdnn_plugin_sdk`, and their transitive dependencies (including FlatBuffers and nlohmann_json when enabled), plus GoogleTest including GoogleMock for tests. `GTest_DIR` can identify an installed GoogleTest package instead of adding its prefix. The `ON` alternative only fetches GoogleTest, not hipDNN, HIP, MIOpen, or other prerequisites. Each standalone configure needs its own fetch setting; a previous hipDNN configure does not supply it.
