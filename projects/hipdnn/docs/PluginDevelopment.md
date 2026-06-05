# Plugin Development

## Table of Contents

- [Overview](#overview)
- [Plugin Types](#plugin-types)
- [SDK Libraries](#sdk-libraries)
- [Plugin API](#plugin-api)
- [Engine IDs](#engine-ids)
- [Creating a Kernel Engine Plugin](#creating-a-kernel-engine-plugin)
  - [Prerequisites](#prerequisites)
  - [Steps Overview](#steps-overview)
  - [Implementation Details & Best Practices](#implementation-details)
  - [Providing Knobs](#providing-knobs)
    - [Plugin SDK Knob Utilities](#plugin-sdk-knob-utilities)
    - [Registering Knobs with the Plugin SDK](#registering-knobs-with-the-plugin-sdk)
    - [Accessing Knob Settings in Your Plugin](#accessing-knob-settings-in-your-plugin)
  - [Key Files Reference](#key-files-reference)
- [Plugin Architecture](#plugin-architecture)
- [Plugin Loading](#plugin-loading)
- [How to Test Plugins](#how-to-test-plugins)
- [Example: MIOpen Provider Plugin](#example-miopen-provider-plugin)
- [Troubleshooting](#troubleshooting)

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

## SDK Libraries

hipDNN provides several C++ SDK libraries for plugin development:

### Data SDK (`data_sdk`)

The Data SDK contains shared types and utilities used across hipDNN. It includes:

- Type helpers (e.g., `half`, `bfloat16`)
- Tensor and memory utilities
- The engine name registry (`EngineNames.hpp`)

### FlatBuffers SDK (`flatbuffers_sdk`)

The FlatBuffers SDK contains the FlatBuffers schemas, generated headers, and graph wrapper classes. It includes:

- FlatBuffers schema definitions for graphs, nodes, and attributes
- Generated headers under `hipdnn_flatbuffers_sdk/data_objects/`
- Wrapper classes (`GraphWrapper`, `NodeWrapper`, `IEngineConfig`) for working with serialized graphs

For adding new operations to the FlatBuffers SDK (schemas, nodes, attributes), see the [How-To Guide](./HowTo.md#adding-a-new-operation-to-existing-plugins).

### Plugin SDK (`plugin_sdk`)

The Plugin SDK contains the plugin API and utilities needed to create a plugin that hipDNN can consume. It includes:

- Plugin interface definitions
- Base classes for engine implementation
- Utilities for plugin development

### Test SDK (`test_sdk`)

The Test SDK provides utilities for testing plugins. It includes:

- CPU reference implementations for validation (convolution, batchnorm, etc.)
- Test utilities (tolerances, seeds, logging)
- Mock objects for unit testing
- FlatBuffer test utilities

## Plugin API

The plugin API defines how kernel engine plugins interact with hipDNN:

- **Graph Processing**: Topologically sorted graphs are passed in a serialized format to plugins using FlatBuffers
- **Data SDK Objects**: Plugins use Data SDK objects to deserialize and process graphs
- **Capability Reporting**: Plugins analyze graphs and report whether they can execute them
- **Execution Interface**: Plugins provide execution methods for supported operations

## Engine IDs

hipDNN uses a deterministic hash-based system for managing engine IDs. This system converts human-readable engine names to unique `int64_t` identifiers.

### How It Works

1. **Engine Names**: Define human-readable string names for your engines (e.g., "MIOPEN_PLUGIN", "MY_CUSTOM_ENGINE")
2. **Hash Function**: The `hipdnn_plugin_sdk::engine_names::engineNameToId()` function converts names to IDs using a FNV-1a hash algorithm
3. **Registration**: Engine names are registered in the Plugin SDK header for discoverability

### Using Engine IDs

```cpp
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

// Option 1: Use a registered engine ID
const int64_t engineId = hipdnn_data_sdk::utilities::MIOPEN_ENGINE_ID;

// Option 2: Generate ID from custom name
const int64_t customEngineId = hipdnn_data_sdk::utilities::engineNameToId("MY_CUSTOM_ENGINE");

// In your engine implementation
class MyEngine {
    int64_t _id;
public:
    MyEngine(const char* engineName)
        : _id(hipdnn_data_sdk::utilities::engineNameToId(engineName)) {
        // Engine is now initialized with a unique ID
    }
};
```

### Registering New Engine Names

To add your engine name to the official registry:

1. **Choose a Unique Name**:
   - Use UPPER_CASE with underscores
   - Make the name match the value.

2. **Add to Registry**: Submit a PR to add your engine name to [`data_sdk/include/hipdnn_data_sdk/utilities/EngineNames.hpp`](../data_sdk/include/hipdnn_data_sdk/utilities/EngineNames.hpp):
   ```cpp
   HIPDNN_REGISTER_ENGINE(MY_NEW_ENGINE)
   ```

3. **Test Locally First**: You can use unregistered names during development - they'll generate a warning but work correctly

### Benefits

- **Deterministic**: Same name always produces same ID
- **No Collisions**: Hash algorithm minimizes collision risk
- **Human-Readable**: Debug logs can show meaningful engine names
- **Forward Compatible**: New engines can be used without registry updates

> [!TIP]
> 💡 The engine ID system ensures globally unique identifiers across all plugins. You can query registered engines using `hipdnn_data_sdk::utilities::getAllEngineNames()` and check for name collisions using the provided test utilities.


## Creating a Kernel Engine Plugin

This section focuses on developing kernel engine plugins; currently the only supported plugin type.

### Prerequisites

Before creating a plugin, ensure you have **built and installed hipDNN**. Plugins depend on the hipDNN Data SDK and Plugin SDK headers. See the [Quick Start Guide](./Building.md#quick-start-guide) for build and installation instructions.

### Steps Overview

1. **Create Plugin Structure**
   - Create a new project/repository for your plugin
   - Implement the plugin interface defined in [`plugin_sdk/include/hipdnn_plugin_sdk/EnginePluginApi.h`](../plugin_sdk/include/hipdnn_plugin_sdk/EnginePluginApi.h)
   - See [MIOpen Provider Plugin](../../../dnn-providers/miopen-provider/) as a reference implementation.

2. **Implement Plugin API Functions**

   The underlying implementation below the plugin API level is entirely at the developer's discretion. While the following architectural components are recommended for code organization and maintainability, the only true requirement is to implement the exported API functions defined in `engine_plugin_api.h`. However, the common architectural pattern consists of:
   - **Engine Manager**: Manages available engines and their capabilities
   - **Engine**: Implements graph execution for specific operations (each engine must have a globally unique `int64_t` ID)
   - **Execution Plans**: Define how operations are executed
   - **Engine Name & ID**: Name your engine and place it in the [EngineNames](../data_sdk/include/hipdnn_data_sdk/utilities/EngineNames.hpp) registry

3. **Build and Deploy Plugin**
   - Configure CMake to build the plugin as a shared library
   - Install to the appropriate plugin directory where hipDNN can discover it at runtime

### Implementation Details

The Plugin SDK provides template interfaces that simplify plugin development. While you can implement the raw C API directly, using these interfaces is recommended for code organization and maintainability.

#### Using Plugin SDK Interfaces

**Template Parameters**: All SDK interfaces are templated on three types:
- `THandle` - Your plugin's handle type (wraps backend resources)
- `TSettings` - Your plugin's settings type (stores knob values)
- `TContext` - Your plugin's context type (holds the execution plan)

The **EngineManager** ([`EngineManager.hpp`](../plugin_sdk/include/hipdnn_plugin_sdk/EngineManager.hpp)) is responsible for:
- Creating and managing `IEngine` instances
- Reporting supported operations via `getApplicableEngineIds()`
- Delegating to the appropriate engine based on engine ID
- Handling resource allocation

For **Engine Implementations** ([`IEngine.hpp`](../plugin_sdk/include/hipdnn_plugin_sdk/interfaces/IEngine.hpp)):
- Each engine must have a unique inter-plugin `int64_t` identifier
- Implement `isApplicable()` to report graph support
- Implement `getDetails()` to return engine metadata and knobs
- Delegate plan building to `IPlanBuilder` implementations

For **Plan Builder Implementations** ([`IPlanBuilder.hpp`](../plugin_sdk/include/hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp)):
- Implement `getCustomKnobs()` to expose configurable parameters
- Implement `initializeExecutionSettings()` to apply knob values
- Implement `buildPlan()` to create executable plans

**Execution plans** ([`IPlan.hpp`](../plugin_sdk/include/hipdnn_plugin_sdk/interfaces/IPlan.hpp)) for kernel engines:
- Map hipDNN operations to backend-specific kernel implementations
- Define memory layouts and data transformations
- Specify kernel launch configurations
- Handle device-specific optimizations

#### Best Practices

1. Organizing kernels by operation type
2. Efficiently managing device memory allocations and transfers
3. Validating inputs and provide meaningful error messages and logs via the sdk
4. Properly managing compute streams for asynchronous execution
5. Profiling kernels and optimize for target hardware
6. Validating and documenting supported operations, hardware requirements, and limitations
7. Including unit tests and integration tests

### Providing Knobs

Knobs allow plugin developers to expose configurable runtime parameters to end-users. This enables performance tuning, feature toggles, and algorithmic choices without requiring code changes or recompilation.

#### What are Knobs?

**Knobs** are runtime-configurable parameters that:
- Control engine behavior (e.g., enable benchmarking, select algorithms)
- Tune performance parameters (e.g., tile sizes, workspace limits)
- Allow users to make trade-offs (e.g., memory vs. performance)

Each knob has:
- **Unique identifier**: String-based ID following namespace conventions
- **Type**: Integer (int64), Float (double), or String
- **Default value**: Used when not explicitly set by the user
- **Constraints**: Valid ranges or explicit allowed values
- **Description**: Human-readable explanation

#### Knob Naming Conventions

Follow a hierarchical naming scheme to avoid conflicts:

```
<plugin_name>.<category>.<knob_name>
```

**Examples**:
```
miopen.conv.tile_size
miopen.global.benchmarking
rocblas.gemm.algorithm
custom_plugin.matmul.block_size
```

> [!IMPORTANT]
> The `global.*` namespace is **reserved** for standard knobs. Custom plugins must use their own namespace prefix.

**Standard global knobs** available for all plugins:
- `global.benchmarking` - Enable kernel benchmarking
- `global.workspace_size_limit` - Maximum workspace memory (operation-specific)

These constants are defined in [`GlobalKnobDefines.hpp`](../plugin_sdk/include/hipdnn_plugin_sdk/GlobalKnobDefines.hpp):

```cpp
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>

// Use the constants instead of string literals
hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME        // "global.benchmarking"
hipdnn_plugin_sdk::WORKSPACE_SIZE_LIMIT_KNOB_NAME // "global.workspace_size_limit"
```

#### Plugin SDK Knob Utilities

The Plugin SDK provides helper classes that simplify knob implementation:

| Class | Purpose |
|-------|---------|
| [`KnobFactory`](../plugin_sdk/include/hipdnn_plugin_sdk/KnobFactory.hpp) | Create knob definitions with constraints |
| [`KnobSettingFactory`](../plugin_sdk/include/hipdnn_plugin_sdk/KnobSettingFactory.hpp) | Create knob settings (for testing) |
| [`GlobalKnobDefines`](../plugin_sdk/include/hipdnn_plugin_sdk/GlobalKnobDefines.hpp) | Constants for standard global knobs |

The [`IPlanBuilder`](../plugin_sdk/include/hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp) interface includes a `getCustomKnobs()` method that plan builders implement to expose their knobs.

#### Registering Knobs with the Plugin SDK

Plugins expose knobs through the `IPlanBuilder::getCustomKnobs()` method. When hipDNN queries your engine for its details, the engine collects knobs from all applicable plan builders and includes them in the response.

```cpp
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>

std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> MyPlanBuilder::getCustomKnobs(
    const HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> knobs;

    if(!isApplicable(handle, opGraph))
    {
        return knobs;
    }

    hipdnn_flatbuffers_sdk::data_objects::KnobT sampleKnob;
    sampleKnob.knob_id = "myplugin.myoperation.myknob";
    sampleKnob.description = "Sample Knob Description";

    hipdnn_flatbuffers_sdk::data_objects::IntValueT defaultValue;
    defaultValue.value = 1;
    sampleKnob.default_value.Set(defaultValue);

    hipdnn_flatbuffers_sdk::data_objects::IntConstraintT constraint;
    constraint.min_value = 0;
    constraint.max_value = 4;
    constraint.step = 1;
    sampleKnob.constraint.Set(constraint);

    knobs.push_back(std::move(sampleKnob));

    return knobs;
}
```

#### Accessing Knob Settings in Your Plugin

When a user creates an execution plan with knob settings, those settings are passed in the `EngineConfig` to your plan builder. Your plugin should:

1. **Extract knob settings** in `initializeExecutionSettings()` or `buildPlan()`
2. **Validate settings** against your knob constraints
3. **Store settings** in your execution settings or plan
4. **Apply settings** during plan execution

**Using the IPlanBuilder Interface**

The recommended approach is to implement the [`IPlanBuilder`](../plugin_sdk/include/hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp) interface. This interface provides a clean separation between knob definition (`getCustomKnobs()`) and knob consumption (`initializeExecutionSettings()`, `buildPlan()`):

```cpp
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>

class MyPlanBuilder : public hipdnn_plugin_sdk::IPlanBuilder<MyHandle, MySettings, MyContext>
{
public:
    // Define the knobs this plan builder supports
    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> getCustomKnobs(
        const MyHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override
    {
        // Return knob definitions (see "Registering Knobs" section above)
        return {};
    }

    // Initialize execution settings from knob values
    void initializeExecutionSettings(
        const MyHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        MySettings& executionSettings) const override
    {
        // Extract and apply knob settings to execution settings
        for (const auto& setting : engineConfig.knobSettings()) {
            const std::string& knobId = setting.knobIdStr();

            if (knobId == hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME) {
                executionSettings.benchmarkingEnabled = (setting.intValue() == 1);
            }
            else if (knobId == "myplugin.tile_size") {
                executionSettings.tileSize = setting.intValue();
            }
        }
    }

    // Build the plan using the configured settings
    void buildPlan(
        const MyHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        MyContext& executionContext) const override
    {
        // Build plan using engineConfig knob settings
    }
};
```

#### Validation Best Practices

1. **Validate during finalization**: The frontend validates against constraints, but plugins should also validate logical combinations:
   ```cpp
   // Example: Knob A can only be 1 if Knob B is 0
   if (knobA == 1 && knobB != 0) {
       return error("Knob A requires Knob B to be 0");
   }
   ```

2. **Provide defaults**: Always provide sensible default values so knobs are optional.

3. **Document constraints**: Clearly document valid ranges and value combinations in your knob descriptions.

4. **Log usage**: Log when non-default knob values are used to aid debugging.

#### Custom vs. Global Knobs

**Custom Knobs** (recommended for most use cases):
- Use your plugin namespace (e.g., `miopen.*`)
- Plugin-specific behavior
- Full control over semantics

**Global Knobs** (use existing ones when applicable):
- `global.benchmarking`: Enable benchmarking for kernel selection
- `global.workspace_size_limit`: Limit workspace memory (when applicable)

#### Deprecating Knobs

When a knob is no longer needed, mark it as deprecated rather than removing it. All knobs have a deprecated flag on them that can be set.

2. **Update documentation** to indicate the deprecation and recommend alternatives.

3. **Remove only during major version updates** to maintain backward compatibility.

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
- Link against hipDNN Data SDK and Plugin SDK
- Set appropriate install paths
- Link to required compute libraries (ie. HIP)

#### Using hipDNN SDKs in External Plugins

When building an external plugin, the hipDNN Data SDK provides CMake variables to help you install your plugin in the correct location:

- **Absolute path** (`HIPDNN_FULL_INSTALL_PLUGIN_ENGINE_DIR`):
  - Computed at `find_package()` time relative to the installed hipDNN location
  - This is intended for **developer use only**

- **Relative path** (`HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR`):
  - **Recommended for installations**
  - Automatically prepends the `CMAKE_INSTALL_PREFIX` of the consumer
  - Remains correct when setting the prefix during the CMake install command

```cmake
find_package(hipdnn_data_sdk CONFIG REQUIRED) # or hipdnn_frontend which includes hipdnn_data_sdk

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
        ├── miopen_plugin.so
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

### Plugin Symbol Resolution
All plugins are loaded with `RTLD_NOW` to ensure that all symbols are resolved at load time. This means
that all dependencies must be satisfied when the plugin is loaded. To avoid symbol conflicts, all plugins must be built with with `-fvisibility=hidden` to limit symbol exposure.

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
  - Tests are divided into two categories described by the prefix argument passed to INSTANTIATE_TEST_SUITE_P
    - **Smoke** - These tests are designed to test features using the smallest possible shape and run quickly (combined smoke test run time must be under 5 mins)
    - **Full** - These tests can contain regression shapes, large shapes, or slow shapes

For a comprehensive example of an integration test, see: [`dnn-providers/miopen-provider/integration_tests/IntegrationGpuBatchnormForwardInference.cpp`](../../../dnn-providers/miopen-provider/integration_tests/IntegrationGpuBatchnormForwardInference.cpp)

Moreover, see our [general testing requirements](./testing/TestingStrategy.md#general-testing-requirements).

## Example: [MIOpen Provider Plugin](../../../dnn-providers/miopen-provider/)

The MIOpen Provider Plugin is a complete example of a kernel engine plugin. It demonstrates how a plugin integrates with hipDNN and delegates execution to a backend. Furthermore, it incorporates the recommended structure and best practices for kernel engine plugins.

At a high level, it:
- Initializes and manages the GPU context using MIOpen handles
- Translates hipDNN tensors into MIOpen tensor descriptors
- Dispatches MIOpen kernels to execute operations
- Coordinates streams and handles synchronization

### Structure
- **Main Plugin**: [`MiopenPlugin.cpp`](../../../dnn-providers/miopen-provider/MiopenPlugin.cpp) - Entry point and plugin registration
- **Engine Manager**: [`EngineManager.hpp`](../../../dnn-providers/miopen-provider/EngineManager.hpp) - Manages MIOpen engines
- **MIOpen Engine**: [`MiopenEngine.cpp`](../../../dnn-providers/miopen-provider/engines/MiopenEngine.cpp) - Implements graph execution using MIOpen kernels

## Troubleshooting

### Plugin Loading Failures

When a plugin fails to load or initialize, hipDNN logs an error and continues loading other plugins. Common issues include:

#### Plugin Handle Creation Fails

If you see errors like "Failed to create handle for plugin 'PluginName'", this typically indicates:
- Missing dependencies that the plugin requires at runtime
- GPU initialization failures (e.g., no compatible device found)
- Plugin internal initialization errors

**Solution**: Check that all plugin dependencies are satisfied and a compatible GPU device is available.

#### Null Handle Returned

If you see "Plugin 'PluginName' returned null handle", the plugin's `hipdnnEnginePluginCreate` function returned a null pointer without throwing an exception.

**Solution**: Review the plugin's handle creation logic to ensure it either returns a valid handle or throws an exception with a meaningful error message.

### Symbol Collisions Between Plugins

#### Symptoms

When multiple plugins are loaded and one or more plugins don't properly hide their symbols, you may encounter:

- Handle collision errors: "Plugin 'PluginName' returned a handle that collides with another plugin"
- Unexpected behavior where one plugin's functions are called instead of another's
- Crashes or undefined behavior during plugin operations

This occurs because dynamically loaded shared libraries can inadvertently share symbols, causing one plugin's function to override another's.

#### Example Error Log

```
[ERROR] Plugin 'my_plugin' returned a handle that collides with another plugin.
        This may indicate a symbol collision between plugins.
        Ensure all plugins are built with -fvisibility=hidden.
```

#### Solution

All plugins **must** be built with symbol visibility hidden to prevent symbol collisions:

1. **CMake Configuration**: Add the following to your plugin's CMakeLists.txt:
   ```cmake
   set(CMAKE_CXX_VISIBILITY_PRESET hidden)
   set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
   ```

2. **Compiler Flags**: Alternatively, add `-fvisibility=hidden` to your compiler flags:
   ```cmake
   target_compile_options(your_plugin PRIVATE -fvisibility=hidden)
   ```

3. **Explicit Symbol Export**: Only export the required plugin API symbols. The plugin SDK macros handle this automatically when visibility is hidden by default.

#### Verification

To verify your plugin has proper symbol visibility:

```bash
# List exported symbols (should only show plugin API functions)
nm -gD your_plugin.so | grep " T "

# Expected output should only contain:
# hipdnnEnginePluginCreate
# hipdnnEnginePluginDestroy
# hipdnnEnginePluginGetAllEngineIds
# ... (other plugin API functions)
```

If you see many internal symbols exported, your visibility settings are incorrect.
