# RFC 0003: Engine ID Design

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Design Goals](#design-goals)
4. [Proposed Solution](#proposed-solution)
5. [Implementation Details](#implementation-details)
6. [Engine Name Registration](#engine-name-registration)
7. [Error Handling](#error-handling)
8. [Examples](#examples)
9. [Future Improvments](#future-improvments)

## Executive Summary

This RFC proposes a simple and effective design for managing engine IDs in the hipDNN plugin ecosystem. The solution uses a deterministic hash function to convert human-readable engine names to `int64_t` IDs.

### Key Benefits
- **Human-Readable**: Developers can use meaningful string names for engines
- **Deterministic**: Same name always produces the same ID via hash function
- **Simple Implementation**: No complex registry or runtime management required
- **Backward Compatible**: Plugin API remains `int64_t`, no breaking changes
- **Flexible**: Supports custom plugins without requiring pre-registration

## Problem Statement

The current implementation has several critical limitations:

1. **Hardcoded IDs**: Engine IDs are manually set in code (e.g., `engineId = 1`)
2. **Collision Risk**: Multiple plugin authors may inadvertently select the same ID
3. **Poor Discoverability**: No mechanism to identify which plugin/engine an ID represents
4. **Limited Debugging**: Difficult to track which engines are loaded and active
5. **No Capability Documentation**: No standard way to document what operations an engine supports

### Current Implementation Issues

```cpp
// MiopenPlugin.cpp (line 120)
auto allEngineIds = std::vector<int64_t>({1});  // Hardcoded!

// MiopenContainer.cpp (line 21)
int64_t engineId = 1;  // Same hardcoded value!
```

## Design Goals

1. **Maintain API Compatibility**: Keep engine ID as `int64_t` in the backend/plugin API
2. **Provide Human-Readable Interface**: Allow use of string names in frontend
3. **Ensure Deterministic IDs**: Same name always produces same ID
4. **Support Forward Compatibility**: Allow unknown engine names for newer engines
5. **Enable Documentation**: Standardize how to document engine capabilities
6. **Keep It Simple**: Minimal complexity for implementation, developers only need to worry about a name

## Proposed Solution

### Overview

The solution consists of two main components:

1. **Shared Header**: Central definition of known engine names
2. **Hash Function**: Deterministic conversion from name to `int64_t`

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Frontend                              │
│  - Accepts engine names (strings) or IDs (int64_t)      │
│  - Converts names to IDs using hash function            │
│  - Passes int64_t to backend                            │
└──────────────────────┬──────────────────────────────────┘
                       │ int64_t
┌──────────────────────▼──────────────────────────────────┐
│                   Backend                               │
│  - Only deals with int64_t engine IDs                   │
│  - No knowledge of string names                         │
│  - Detects duplicate IDs and warns                      │
└──────────────────────┬──────────────────────────────────┘
                       │ int64_t
┌──────────────────────▼──────────────────────────────────┐
│                   Plugins                               │
│  - Use hash function to generate their engine IDs       │
│  - Return int64_t IDs via API                           │
│  - Document their engine names and capabilities         │
└─────────────────────────────────────────────────────────┘
```

## Implementation Details

### Engine Names Header

Create a shared header file in the plugin SDK that only requires a single line change to add a new engine name.  Below are 3 possible options that enable this functionality.  Ideally we would have the ability to create automation to detect duplicates/collisions when new names are added without needing to modify tests.

#### Selected Header format: Simple Macros with Static Registration
Easy to understand macro approach that runtime creates a set of all registered engine names for future automation

Note: This shows the anticipated format but it is subject to change as we implement and test it out.

```cpp
#pragma once

#include <set>
#include <string_view>

namespace hipdnn_plugin_sdk::engine_names {

// Forward declare the registration set
inline std::set<std::string_view>& getAllEngineNames() {
    static std::set<std::string_view> allEngines;
    return allEngines;
}

inline std::unordered_map<int64_t, std::string_view>& getEngineIdToNameMap() {
    static std::unordered_map<int64_t, std::string_view> engineIdToNameMap;
    return engineIdToNameMap;
}

// Registration helper class
struct EngineRegistrar {
    EngineRegistrar(std::string_view name) {
        getAllEngineNames().insert(name);
        getEngineIdToNameMap()[hipdnn_data_sdk::engineNameToId(name.data())] = name;
    }
};

// Macro that defines engine and automatically registers it
#define HIPDNN_REGISTER_ENGINE(name) \
    inline constexpr const char* name##_NAME = value; \
    inline const int64_t name##_ID = hipdnn_data_sdk::utilities::engineNameToId(#name); \
    inline const hipdnn_data_sdk::utilities::EngineRegistrar name##_registrar{#name};

// Define all engines using the macro, name and value should match.
HIPDNN_REGISTER_ENGINE(MIOPEN_PLUGIN)
HIPDNN_REGISTER_ENGINE(VENDOR_FAST_CONV)
HIPDNN_REGISTER_ENGINE(CPU_REFERENCE_ENGINE)
HIPDNN_REGISTER_ENGINE(EXAMPLE_PLUGIN_RENAME_THIS)

} // namespace hipdnn_plugin_sdk::engine_names

```

#### Other Header Options Considered

##### Simple Strings
Simple implementation with string constants.  Possible issues with creating automation to detect duplicates/hash collisions automatically

```cpp
#pragma once

namespace hipdnn_plugin_sdk::engine_names {

// Built-in AMD engines
constexpr const char* MIOPEN_LEGACY = "MIOPEN_PLUGIN";

// Vendor engines
constexpr const char* VENDOR_EXAMPLE = "VENDOR_FAST_CONV";

} // namespace hipdnn_plugin_sdk::engine_names
```

##### X-Macro Pattern
More complex macro option that offers compile time arrays of all engine names.  Additionally developers only need to list their engine name once as stringifiation can take care of making the `const char*` definitions.

A downside to this approach is the complexity.

```cpp
#pragma once

#include <array>
#include <string_view>
#include <unordered_set>

namespace hipdnn_plugin_sdk::engine_names {

// X-Macro list of all engines, add all new engines here.
#define HIPDNN_ENGINE_LIST(X) \
    X(MIOPEN_ENGINE) \
    X(VENDOR_FAST_CONV_ENGINE)

// Generate const char* definitions using stringification
#define DEFINE_ENGINE_NAME(name) \
    inline constexpr const char* name = #name;

HIPDNN_ENGINE_LIST(DEFINE_ENGINE_NAME)
#undef DEFINE_ENGINE_NAME

// Count engines at compile time
#define COUNT_ENGINE(name) +1
inline constexpr size_t engineCount = 0 HIPDNN_ENGINE_LIST(COUNT_ENGINE);
#undef COUNT_ENGINE

// Generate compile-time array of all engine names
#define ADD_TO_ARRAY(name) #name,
inline constexpr std::array<const char*, engineCount> allEngineNames = {{
    HIPDNN_ENGINE_LIST(ADD_TO_ARRAY)
}};
#undef ADD_TO_ARRAY

// Runtime set for easy lookup (initialized once)
inline const std::unordered_set<std::string_view>& getAllEngineNamesSet() {
    static std::unordered_set<std::string_view> engineSet = []() {
        std::unordered_set<std::string_view> set;
        for (const auto& name : allEngineNames) {
            set.insert(name);
        }
        return set;
    }();
    return engineSet;
}

} // namespace hipdnn_plugin_sdk::engine_names

```

### Hash Function

Implement a deterministic hash function to convert names to IDs:

"engine name string" ---> [Hash Function] ---> int64_t engine ID

Hash function is implemented in the data_sdk so that it can be used anywhere

```cpp
#pragma once
#include <string>
#include <cstdint>

namespace hipdnn_data_sdk {
inline int64_t engineNameToId(const char* engineName) {
    // Implementation of this hash function is an implementation detail and is
    // up to the developer who implements it.  The developer is expected to create a robust
    // test suite to ensure the function is deterministic and is made to have minimal collision possibilities.
}
} // namespace hipdnn_data_sdk
```

### Plugin Implementation

Plugins will use the hash function internally to convert their engine names to ids so they can communicate via the existing API.  The EnginePluginApi.h only addresses engines by `int64_t`, so no API changes are needed.

### Frontend Implementation

The frontend accepts both names and IDs:

```cpp
// Frontend API extension
namespace hipdnn_frontend::graph {

class Graph : public INode
{
public:
    // Existing API - accepts int64_t
    void set_preferred_engine_id_ext(std::optional<int64_t> engineId)

    // New overload - accepts string name
    void set_preferred_engine_id_ext(const std::optional<std::string> engineName) {
        int64_t engineId = hipdnn::engineNameToId(engineName);

        // Log for debugging
        HIPDNN_LOG_DEBUG("Engine name '{}' mapped to ID: 0x{:016X}",
                        engineName, engineId);

        // Forward to the int64_t version
        setPreferredEngine(engineId);
    }
};

} // namespace hipdnn_frontend::graph
```

### Backend Duplicate Detection

The backend when loading plugins can check for any duplicate engine IDs.  It will throw
an error when duplicate IDs are found along with the names of the plugins that caused the conflict.


## Engine Name Registration

### Process for Adding New Engine Names

1. **Choose a Unique Name**: Select a descriptive, unique engine name
   - Good: `"VENDOR_FAST_CONV_V2"`
   - Bad: `"ENGINE_1"`, `"FAST"`

2. **Test Locally**: Use the name in your plugin without modifying the header, the code will throw a warning but its safe to ignore

3. **Submit PR**: Add your engine name to `EngineNames.hpp`
   ```cpp
   constexpr const char* MY_VENDOR_ENGINE = "MY_VENDOR_FAST_CONV";
   ```

4. **Document Capabilities**: Include documentation per the standards below

### Guidelines

- Engine names should be UPPER_CASE with underscores
- Include vendor/organization prefix to make it easier to identify author
- Be descriptive about the engine's purpose
- Once merged, names should not be changed (breaks compatibility)

### Forward Compatibility

If a plugin name is not known (not in the shared header), the hash function can still generate a unique ID.  The usage of a unknown plugin name will generate a warning but thats it.  This allows newer plugins to be used without needing to update the shared header.

## Error Handling

### Duplicate IDs

Add a unit test to the repo that checks for duplicate engine IDs. This test should be setup to such that it doesnt need updates in order to test newly added engines.  This requires we pick a engine name registration option that allows us to query all known engine names at runtime without manually updating a list.

## Examples

### Example 1: Custom Plugin Development

```cpp
#include "hipdnn_plugin_sdk/EngineNames.hpp"

// MyCustomPlugin.cpp
class MyCustomPlugin {
    void initialize() {
        // Use the custom engine name defined in the shared header
        const std::string engineName = hipdnn::engine_names::MY_CUSTOM_ENGINE;

        // Generate ID from name
        int64_t engineId = hipdnn::data_sdk::engineNameToId(engineName);

        // Log for debugging
        HIPDNN_LOG_INFO("Initializing engine '{}' with ID: 0x{:016X}",
                       engineName, engineId);

        // Register engine
        auto engine = std::make_unique<MyCustomEngine>(engineId);
        registerEngine(std::move(engine));
    }
};
```

### Example 2: Frontend Usage

```cpp
// Application using the frontend
void setupGraph() {
    hipdnn::frontend::Graph graph;

    // Option 1: Use string name directly
    graph.set_preferred_engine_id_ext("MIOPEN_LEGACY");

    // Option 2: Use constant from header
    graph.set_preferred_engine_id_ext(hipdnn::engine_names::MIOPEN_LEGACY);

    // Option 3: Use custom engine name (not in header)
    graph.set_preferred_engine_id_ext("MY_CUSTOM_ENGINE_V2");  // Works, with warning

    // Option 4: Still support int64_t for compatibility
    graph.set_preferred_engine_id_ext(0x123456789ABCDEF0LL);
}
```

## Future Improvments

### Benchmarking Support

In the future we would like to be able to compare plugins against themselves over time in order to track performance and regressions.  To facilitate this we can add a compile time override that appends a prefix to the engine name when generating the ID.  This will allow multiple unique IDs for the same engine name when benchmarking without needing to modify the engine name registration header.  The engine under test just needs to be compiled with the `HIPDNN_BENCHMARK_MODE` macro defined.

```cpp
// Update EngineNames.hpp
// Note: This will look differnent based on which option is chosen for engine name registration header
...

// Allow compile-time override for testing
#ifdef HIPDNN_BENCHMARK_MODE
    #define HIPDNN_ENGINE_PREFIX "BENCHMARK_"
#else
    #define HIPDNN_ENGINE_PREFIX ""
#endif

// Helper to concatenate prefix with name
#define HIPDNN_CONCAT_PREFIX(prefix, name) prefix #name

// Macro that defines engine and automatically registers it
// Applies benchmark prefix if defined
#define HIPDNN_REGISTER_ENGINE(name) \
    inline constexpr const char* name = HIPDNN_CONCAT_PREFIX(HIPDNN_ENGINE_PREFIX, name); \
    inline const EngineRegistrar name##_registrar{HIPDNN_CONCAT_PREFIX(HIPDNN_ENGINE_PREFIX, name)};

...
```
