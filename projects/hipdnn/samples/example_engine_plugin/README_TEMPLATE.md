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
| {Additional dependencies specific to your plugin} | | |

## Building

### Building as a standalone plugin

To build the plugin standalone, first install hipDNN and any required dependencies on the system.

1. Navigate to the `{plugin-directory}` directory.
2. Configure the build using `cmake -B build`.
3. Run `cmake --build build` to build the plugin.
4. Run `ctest --test-dir build` to run the tests.

### CMake Options

| Option | Default | Description |
|---|---|---|
| `{PLUGINNAME}_BUILD_UNIT_TESTS` | `ON` | Build unit tests |
| `{PLUGINNAME}_BUILD_SAMPLE` | `ON` | Build sample application |

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
./bin/{plugin_name}_tests
```

{Describe any additional testing details, test categories, or GPU requirements.}
