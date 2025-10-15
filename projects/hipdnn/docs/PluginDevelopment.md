# Plugin Development

## Table of Contents

- [Overview](#overview)
- [Plugin Types](#plugin-types)
- [hipDNN-SDK Library](#hipdnn-sdk-library)
- [Plugin API](#plugin-api)
- [Creating a Kernel Engine Plugin](#creating-a-kernel-engine-plugin)
  - [Steps Overview](#steps-overview)
  - [Implementation Details & Best Practices](#implementation-details)
  - [Key Files Reference](#key-files-reference)
- [Plugin Architecture](#plugin-architecture)
- [Example: MIOpen Legacy Plugin](#example-miopen-legacy-plugin)

---

## Overview

hipDNN supports a plugin architecture that allows for modular extensions to the framework. Plugins are designed to be separate projects that extend hipDNN's capabilities without being part of the core repository. The backend discovers and manages these plugins, leveraging them for different aspects of deep learning routines. This architecture provides flexibility in implementation choices and enables optimizations for specific hardware or use cases.

## Plugin Types

hipDNN will support three types of plugins, each serving a specific purpose:

### 1. Engine Heuristic and Selection Plugins (`hipdnn_plugins/heuristics/`)
These plugins help determine the best execution strategy for a given operation or graph. They analyze the computation requirements and available resources to select optimal implementations.

### 2. Benchmarking and Tuning Plugins (`hipdnn_plugins/benchmarking/`)
These plugins focus on performance optimization by benchmarking different implementations and tuning parameters for specific hardware configurations.

### 3. Kernel Engine Plugins (`hipdnn_plugins/engines/`)
These plugins provide the actual kernel implementations for operations. They contain the compute kernels that execute on the target hardware (GPUs, accelerators, etc.).

> [!IMPORTANT]
> 🕒 **Current Status**: Only kernel engine plugins are presently supported in hipDNN. Support for engine heuristic/selection and benchmarking/tuning plugins will be added in future releases. See the [Roadmap](./Roadmap.md#plugins) for future development plans.

## hipDNN-SDK Library

The hipDNN-SDK API is a Header-Only C++ library which provides the requirements needed to create a plugin that hipDNN can consume. It includes:

- Plugin interface definitions
- Data structures for graph representation
- Utilities for serialization/deserialization
- Base classes for engine implementation

For adding new operations to the SDK (schemas, nodes, attributes), see the [How-To Guide](./HowTo.md#adding-a-new-operation-to-existing-plugins).

## Plugin API

The plugin API defines how kernel engine plugins interact with hipDNN:

- **Graph Processing**: Topologically sorted graphs are passed in a serialized format to plugins using FlatBuffers
- **SDK Data Objects**: Plugins use SDK data objects to deserialize and process graphs
- **Capability Reporting**: Plugins analyze graphs and report whether they can execute them
- **Execution Interface**: Plugins provide execution methods for supported operations

## Creating a Kernel Engine Plugin

This section focuses on developing kernel engine plugins; currently the only supported plugin type.

### Prerequisites

Before creating a plugin, ensure you have **built and installed hipDNN**. Plugins depend on the hipDNN SDK headers and libraries. See the [Quick Start Guide](./Building.md#quick-start-guide) for build and installation instructions.

### Steps Overview

1. **Create Plugin Structure**
   - Create a new project/repository for your plugin
   - Implement the plugin interface defined in [`sdk/include/hipdnn_sdk/plugin/EnginePluginApi.h`](../sdk/include/hipdnn_sdk/plugin/EnginePluginApi.h)
   - See [MIOpen Legacy Plugin](../plugins/miopen_legacy_plugin/) as a reference implementation (currently included but will become a separate project)

2. **Implement Plugin API Functions**

   The underlying implementation below the plugin API level is entirely at the developer's discretion. While the following architectural components are recommended for code organization and maintainability; the only true requirement is to implement the exported API functions defined in `engine_plugin_api.h`. However, the common architectural pattern consists of:
   - **Engine Manager**: Manages available engines and their capabilities
   - **Engine**: Implements graph execution for specific operations (each engine must have a globally unique `int64_t` ID)
   - **Execution Plans**: Define how operations are executed

3. **Build and Deploy Plugin**
   - Configure CMake to build the plugin as a shared library
   - Install to the appropriate plugin directory where hipDNN can discover it at runtime

### Implementation Details

The **Engine Manager** is responsible for:
- Creating and managing engine instances
- Reporting supported operations
- Handling resource allocation
- Managing device-specific contexts

For **Engine Implementations**:
- Each engine must have a unique inter-plugin `int64_t` identifier
- Implement the `execute()` method for graph execution
- Provide `get_supported_operations()` to report capabilities
- Handle operation-specific kernel launches
- Manage memory transfers and synchronization

> [!TIP]
> 💡 An engine ID is an integer unique to all loaded plugins. These IDs are used by the backend to identify and select specific engines for execution. You may want to reference other loaded plugins to accrue a set of unused engine IDs.

**Execution plans** for kernel engines:
- Map hipDNN operations to backend-specific kernel implementations
- Define memory layouts and data transformations
- Specify kernel launch configurations
- Handle device-specific optimizations

In general, the **best practices** consist of:

1. Organizing kernels by operation type
2. Efficiently manage device memory allocations and transfers
3. Validate inputs and provide meaningful error messages and logs via the sdk
4. Properly manage compute streams for asynchronous execution
5. Profile kernels and optimize for target hardware
6. Validate and document supported operations, hardware requirements, and limitations
7. Include unit tests and integration tests

### Key Files Reference

- **Plugin API Interface**: [`sdk/include/hipdnn_sdk/plugin/EnginePluginApi.h`](../sdk/include/hipdnn_sdk/plugin/EnginePluginApi.h)
- **Example Plugin Implementation**: [`plugins/miopen_legacy_plugin/MiopenLegacyPlugin.cpp`](../plugins/miopen_legacy_plugin/MiopenLegacyPlugin.cpp)
- **Example Engine Manager**: [`plugins/miopen_legacy_plugin/EngineManager.hpp`](../plugins/miopen_legacy_plugin/EngineManager.hpp)
- **Example Engine Implementation**: [`plugins/miopen_legacy_plugin/engines/MiopenEngine.cpp`](../plugins/miopen_legacy_plugin/engines/MiopenEngine.cpp)

## Plugin Architecture

### Directory Structure for Kernel Engine Plugins

Your plugin should be structured as an independent project:

```
your_kernel_plugin_project/
├── CMakeLists.txt
├── YourPlugin.cpp           # Main plugin entry point
├── EngineManager.cpp        # Engine management
├── engines/
│   ├── EngineInterface.hpp  # Engine interface
│   ├── YourEngine.cpp       # Engine implementation
│   └── plans/                # Internal plans
├── tests/                    # Plugin-specific tests
└── integration_tests/        # End-to-end integration tests
```

### Build Configuration
Your plugin's CMakeLists.txt should:
- Build as a shared library
- Link against hipDNN SDK
- Set appropriate install paths
- Link to required compute libraries (ie. HIP)

#### Using hipDNN SDK in External Plugins

When building an external plugin, the hipDNN SDK provides CMake variables to help you install your plugin in the correct location:

- **Absolute path** (`HIPDNN_FULL_INSTALL_PLUGIN_ENGINE_DIR`): 
  - Hardcoded at CMake configure time
  - This is intended for **developer-use only**
  
- **Relative path** (`HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR`):
  - **Recommended for installations**
  - Automatically prepends the `CMAKE_INSTALL_PREFIX` of the consumer 
  - Remains correct when setting the prefix during the CMake install command 

```cmake
find_package(hipdnn_sdk CONFIG REQUIRED) # or hipdnn_frontend which includes hipdnn_sdk

# Example: Configure your plugin to install to the correct location
install(
    TARGETS your_plugin_name
    LIBRARY DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}
)
```

This ensures your plugin will be installed to the same directory structure that hipDNN expects for plugin discovery.

#### Build and Install Directory Structure

The hipDNN build system maintains consistent directory structures for plugins:

**Build directory structure:**
```
build/lib/
└── hipdnn_plugins/
    └── engines/
        └── your_plugin.so
```

**Install directory structure:**
```
/opt/rocm/lib/
└── hipdnn_plugins/
    └── engines/
        └── your_plugin.so
```

External plugins should follow this same structure to ensure compatibility.

## Plugin Loading

hipDNN supports dynamic plugin loading with configurable search paths.

### Default Plugin Loading

By default, hipDNN loads plugins from:
```
./hipdnn_plugins/engines/
```

This path is relative to the backend shared library location, typically:
```
/opt/rocm/lib/
```

**Default structure example:**
```
/opt/rocm/lib/
└── hipdnn_plugins/
    └── engines/
        ├── miopen_legacy_plugin.so
        └── other_plugin.so
```

### Environment Variable Override

You can override the default plugin directory using the `HIPDNN_PLUGIN_DIR` environment variable. This is particularly useful for testing and development:

```bash
# Load plugins from a custom directory
export HIPDNN_PLUGIN_DIR=/path/to/test/plugins

# Example: Load test plugins during testing
export HIPDNN_PLUGIN_DIR=/home/user/hipDNN/build/lib/test_plugins
```

When `HIPDNN_PLUGIN_DIR` is set, hipDNN will **only** load plugins from the specified directory and supplementary custom paths, ignoring the default location. This allows complete control over which plugins are loaded, which is essential for:
- Running tests with test-specific plugins
- Development and debugging of new plugins
- Isolating production plugins from test plugins

### Custom Plugin Paths

Prior to creating a hipDNN handle, you can specify custom plugin paths using the `hipdnnSetEnginePluginPaths_ext` function:

```c
hipdnnStatus_t hipdnnSetEnginePluginPaths_ext(
    size_t num_paths,
    const char* const* plugin_paths,
    hipdnnPluginLoadingMode_ext_t loading_mode
);
```

#### Path Resolution

Custom paths can be:
- **Relative paths**: Resolved from the backend shared library location
- **Absolute paths**: Used as specified

#### Loading Modes

| Mode | Description |
|------|-------------|
| `HIPDNN_PLUGIN_LOADING_ADDITIVE` | Adds new paths to the existing plugin search paths |
| `HIPDNN_PLUGIN_LOADING_ABSOLUTE` | Only loads from the specified paths |

#### Example Usage

```c
// Add custom plugin directories
const char* custom_paths[] = {
    "/home/user/my_plugins",        // Absolute path
    "./local_plugins",              // Relative to backend shared library
    "/opt/custom/hipdnn/plugins"
};

hipdnnSetEnginePluginPaths_ext(
    3,                              // Number of paths
    custom_paths,                   // Array of path strings
    HIPDNN_PLUGIN_LOADING_ADDITIVE  // Add to existing paths
);
```

Plugins are loaded according to the selected path schema during hipDNN handle creation. Changing paths after handle creation has no effect until another handle is created.

### Querying Loaded Plugins

After creating a hipDNN handle, you can query which engine plugins were successfully loaded using the `hipdnnGetLoadedEnginePluginPaths_ext` function:

```c
hipdnnStatus_t hipdnnGetLoadedEnginePluginPaths_ext(
    hipdnnHandle_t handle,
    size_t* num_plugin_paths,
    char** plugin_paths,
    size_t* max_string_len
);
```

This function uses a two-call pattern:

1. **First call** - Query the number of plugins and required buffer size:
    ```cpp
    size_t num_plugins = 0;
    size_t max_len = 0;

    hipdnnGetLoadedEnginePluginPaths_ext(handle, &num_plugins, nullptr, &max_len);
    ```

2. **Second call** - Retrieve the actual plugin paths:
    ```cpp
    hipdnnGetLoadedEnginePluginPaths_ext(handle, &num_plugins, nullptr, &max_len);

    std::vector<std::vector<char>> buffers(num_plugins, std::vector<char>(max_len));
    std::vector<char*> ptrs;
    ptrs.reserve(num_plugins);
    for(size_t i = 0; i < num_plugins; ++i) ptrs.push_back(buffers[i].data());

    hipdnnGetLoadedEnginePluginPaths_ext(handle, &num_plugins, ptrs.data(), &max_len);

    for(size_t i = 0; i < num_plugins; ++i)
    {
        std::cout << "Loaded plugin: " << buffers[i].data() << '\n';
    }
    ```

## How to Test Plugins

> [!IMPORTANT]
> Testing is crucial for ensuring plugin reliability and correctness. Plugins should include both unit tests and integration tests to validate their functionality.

### Test Structure

Following the [Testing Strategy](./testing/TestingStrategy.md), plugins should organize tests as follows:

```
your_kernel_plugin_project/
├── tests/                    # Unit tests
│   ├── TestEngine.cpp
│   ├── TestKernels.cpp
│   └── TestUtilties.cpp
└── integration_tests/        # End-to-end integration tests
    ├── Operation1Test.cpp
    └── Operation2Test.cpp
```

### Unit Tests

Unit tests focus on the internal implementation of your plugin components:

- **Location**: `plugins/<plugin_name>/tests/`
- **Purpose**: Test individual components in isolation (engines, utilities, kernel logic)
- **Requirements**:
  - Must be fast-running
  - GPU operations must be marked with `SKIP_IF_NO_DEVICE()` macro
  - Use mocking/stubbing for dependencies where appropriate
  - Should work on both Windows and Linux

### Integration Tests

Integration tests validate end-to-end functionality of your plugin:

- **Location**: `plugins/<plugin_name>/integration_tests/`
- **Purpose**: Validate correctness of graph execution and accuracy of results
- **Requirements**:
  - Test complete operation graphs
  - Validate against reference implementations
  - Test different data types, layouts, dimensions, and edge-cases for each
  - Enable tests for all supported ASICs
  - GPU typically required for meaningful validation

For a comprehensive example of an integration test, see: [`plugins/miopen_legacy_plugin/integration_tests/BatchnormFwdInferenceIntegrationTest.cpp`](../plugins/miopen_legacy_plugin/integration_tests/BatchnormFwdInferenceIntegrationTest.cpp)

Moreover, see our [general testing requirements](./testing/TestingStrategy.md#general-testing-requirements).

## Example: [MIOpen Legacy Plugin](../plugins/miopen_legacy_plugin/)

The MIOpen Legacy Plugin is a complete example of a kernel engine plugin. It demonstrates how a plugin integrates with hipDNN and delegates execution to a backend. Furthermore, it incorporates the recommended structure and best practices for kernel engine plugins.

At a high level, it:
- Initializes and manages the GPU context using MIOpen handles
- Translates hipDNN tensors into MIOpen tensor descriptors
- Dispatches MIOpen kernels to execute operations
- Coordinates streams and handles synchronization

### Structure
- **Main Plugin**: [`MiopenLegacyPlugin.cpp`](../plugins/miopen_legacy_plugin/MiopenLegacyPlugin.cpp) - Entry point and plugin registration
- **Engine Manager**: [`EngineManager.hpp`](../plugins/miopen_legacy_plugin/EngineManager.hpp) - Manages MIOpen engines
- **MIOpen Engine**: [`MiopenEngine.cpp`](../plugins/miopen_legacy_plugin/engines/MiopenEngine.cpp) - Implements graph execution using MIOpen kernels
