# {Plugin Name} Plugin

{Brief one-line description of what the plugin does and the technology or backend it provides.}

## Overview

{Expanded description of the plugin's purpose, the operations it supports, and how it implements them (e.g., wrapping an existing library, custom HIP kernels via HIPRTC, etc.).}

## Prerequisites

| Dependency | Purpose | Notes |
|---|---|---|
| CMake >= 3.25 | Build system | |
| C++17 compiler | GCC/G++ or MSVC | |
| ROCm (HIP SDK + HIPRTC) | GPU kernel compilation and execution | Required at runtime |
| hipDNN | Plugin SDK, data SDK | SDK packages from ROCm install or built from source |
| GoogleTest including GoogleMock | Unit testing and mocking frameworks | Supplied by the hipDNN developer image, or provide its CMake package or opt in to fetching with `-DALLOW_FETCH_DEPS=ON` |
| {Additional dependencies specific to your plugin} | | |

## Building

### Building as a standalone plugin

To build the plugin standalone, first install hipDNN and any required dependencies on the system.

The hipDNN developer image supplies GoogleTest/GoogleMock and spdlog automatically in `/usr/local`, which CMake searches by default. No manual dependency download, install, or fetch flag is needed for those packages inside the image.

1. Navigate to the `{plugin-directory}` directory.
2. Configure the build using `cmake -B build`.
3. Run `cmake --build build` to build the plugin.
4. Run `ctest --test-dir build` to run the tests.

Outside the image, choose one configure command, using absolute paths:

```bash
# Installed packages only.
cmake -B build -DALLOW_FETCH_DEPS=OFF \
    -DCMAKE_PREFIX_PATH="/path/to/hipdnn-install;/path/to/rocm;/path/to/dependencies"

# Alternatively, allow this plugin to fetch missing GoogleTest.
cmake -B build -DALLOW_FETCH_DEPS=ON \
    -DCMAKE_PREFIX_PATH="/path/to/hipdnn-install;/path/to/rocm;/path/to/dependencies"
```

The combined prefixes must provide HIP/HIPRTC, the hipDNN SDK packages used by the plugin and sample, their transitive dependencies (including FlatBuffers and nlohmann_json when enabled), and GoogleTest including GoogleMock. `GTest_DIR` can identify an installed GoogleTest package instead of adding its prefix. On Windows, use Windows absolute paths in the same semicolon-separated list and select your generator. The fetch alternative does not supply hipDNN, HIP, or other SDK dependencies. This independent configure needs its own `ALLOW_FETCH_DEPS`; a previous hipDNN or samples configure does not supply it.

### CMake Options

| Option | Default | Description |
|---|---|---|
| `{PLUGINNAME}_BUILD_UNIT_TESTS` | `ON` | Build unit tests |
| `{PLUGINNAME}_BUILD_SAMPLE` | `ON` | Build sample application |
| `ALLOW_FETCH_DEPS` | `OFF` | Allow fetching GoogleTest when its CMake package is unavailable |
| `EXAMPLE_PROVIDER_GTEST_VERSION` | `1.17.0` | GoogleTest fetch fallback version; installed packages or explicitly supplied sources take precedence |

The copied plugin is standalone: its GoogleTest fallback default is local to the template, not imported from a monorepo file. An installed package or explicitly supplied `FETCHCONTENT_SOURCE_DIR_GOOGLETEST` tree can be used with fetching disabled.

## Architecture

The plugin follows the standard hipDNN plugin architecture. It builds as a shared library providing a hipDNN [kernel engine plugin](https://github.com/ROCm/hipDNN/blob/develop/docs/PluginDevelopment.md#creating-a-kernel-engine-plugin) API.

Five macros in `{PluginName}PluginPublic.cpp` configure the plugin entry points:

- `HIPDNN_PLUGIN_NAME` -- display name string
- `HIPDNN_PLUGIN_VERSION` -- version string
- `HIPDNN_PLUGIN_CONTAINER_TYPE` -- fully qualified Container class name
- `HIPDNN_PLUGIN_HANDLE_TYPE` -- fully qualified Handle struct name
- `HIPDNN_PLUGIN_CONTEXT_TYPE` -- fully qualified Context struct name

### Type Hierarchy

```
Container
├── Owns EngineManager<Handle, Settings, Context>
├── Creates engines defined via getEngineDefinitions()
│   └── Engine ({PLUGIN_ENGINE_NAME})
│       └── PlanBuilder
│           └── Plan
└── copyEngineIds() -- returns registered engine IDs to hipDNN

Handle
├── Holds shared_ptr<Container>
├── setStream(hipStream_t)
└── getEngineManager()
```

{Add plugin-specific architecture details, including descriptions of engines, plan builders, and any additional infrastructure.}

## Operation Support

{List or describe the operations this plugin supports. Link to detailed operation support documentation if available.}

## Testing

After building, run the test suites:

```bash
# Unit tests
./bin/hipdnn_{plugin_name}_unit_tests
```

{Describe any additional testing details, test categories, or GPU requirements.}
