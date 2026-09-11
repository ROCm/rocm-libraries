# RFC 0003: Engine ID Design

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Design Goals](#design-goals)
4. [Proposed Solution](#proposed-solution)
5. [Implementation Details](#implementation-details)
6. [Plugin-Supplied Engine Names](#plugin-supplied-engine-names)
7. [Engine Name Registration](#engine-name-registration)
8. [Error Handling](#error-handling)
9. [Examples](#examples)
10. [Future Improvements](#future-improvements)

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

The solution consists of three main components:

1. **Shared Header**: Central definition of known engine names
2. **Hash Function**: Deterministic conversion from name to `int64_t`
3. **Plugin Entry Point**: An optional ABI call through which a plugin names engines the
   shared header does not know about, described in
   [Plugin-Supplied Engine Names](#plugin-supplied-engine-names)

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
│  - Routes and dispatches by int64_t engine IDs          │
│  - Requires a declared name to hash to its ID           │
│  - Drops duplicate IDs and logs an error                │
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

Create a shared header file in the data SDK, so that the frontend, the backend and plugins can all reach it, that only requires a single line change to add a new engine name.  Below are 3 possible options that enable this functionality.  Ideally we would have the ability to create automation to detect duplicates/collisions when new names are added without needing to modify tests.

#### Selected Header format: Simple Macros with Static Registration
Easy to understand macro approach that runtime creates a set of all registered engine names for future automation

As shipped this lives in `data_sdk/include/hipdnn_data_sdk/utilities/EngineNames.hpp`. The
sketch below is condensed: the real header keeps the registries behind accessors in a
`detail` namespace and reports which two names collided.

```cpp
#pragma once

#include <set>
#include <string_view>
#include <unordered_map>

namespace hipdnn_data_sdk::utilities {

// Forward declare the registration set
inline std::set<std::string_view>& getAllEngineNames() {
    static std::set<std::string_view> allEngines;
    return allEngines;
}

inline std::unordered_map<int64_t, std::string_view>& getEngineIdToNameMap() {
    static std::unordered_map<int64_t, std::string_view> engineIdToNameMap;
    return engineIdToNameMap;
}

// Registration helper class. Throws if the ID is already claimed, so a duplicate
// registration or a hash collision between two in-tree engines fails at static
// initialization rather than at load.
struct EngineRegistrar {
    EngineRegistrar(std::string_view name) {
        getAllEngineNames().insert(name);
        getEngineIdToNameMap()[engineNameToId(name)] = name;
    }
};

// Macro that defines engine and automatically registers it.
#define HIPDNN_REGISTER_ENGINE(name) \
    inline constexpr const char* name##_NAME = #name; \
    inline const int64_t name##_ID = hipdnn_data_sdk::utilities::engineNameToId(#name); \
    inline const hipdnn_data_sdk::utilities::EngineRegistrar name##_registrar{#name};

// Define all engines using the macro.
HIPDNN_REGISTER_ENGINE(MIOPEN_ENGINE)
HIPDNN_REGISTER_ENGINE(HIPBLASLT_ENGINE)
HIPDNN_REGISTER_ENGINE(ROCKE_ENGINE)

} // namespace hipdnn_data_sdk::utilities

```

The shipped macro also accepts an optional second argument, for the case where the engine
string cannot be spelled as a C++ identifier or is meant to differ from it:

```cpp
HIPDNN_REGISTER_ENGINE(VENDOR_FAST_CONV, "Vendor.FastConv/v2")
```

Every in-tree engine currently uses the one-argument form, where the two coincide. Either
form generates the same three names: `<name>_NAME`, `<name>_ID`, and a registrar object. All
three matter — the ID constant is what a plugin returns, and the registrar is what makes the
name visible to collision detection and to reverse lookup.

#### Other Header Options Considered

##### Simple Strings
Simple implementation with string constants.  Possible issues with creating automation to detect duplicates/hash collisions automatically

```cpp
#pragma once

namespace hipdnn_data_sdk::utilities {

// Built-in AMD engines
constexpr const char* MIOPEN_ENGINE_NAME = "MIOPEN_ENGINE";

// Vendor engines
constexpr const char* VENDOR_EXAMPLE_NAME = "VENDOR_FAST_CONV";

} // namespace hipdnn_data_sdk::utilities
```

##### X-Macro Pattern
More complex macro option that offers compile time arrays of all engine names.  Additionally developers only need to list their engine name once as stringifiation can take care of making the `const char*` definitions.

A downside to this approach is the complexity.

```cpp
#pragma once

#include <array>
#include <string_view>
#include <unordered_set>

namespace hipdnn_data_sdk::utilities {

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

} // namespace hipdnn_data_sdk::utilities

```

### Hash Function

Implement a deterministic hash function to convert names to IDs:

"engine name string" ---> [Hash Function] ---> int64_t engine ID

Hash function is implemented in the data_sdk so that it can be used anywhere

```cpp
#pragma once
#include <string>
#include <cstdint>

namespace hipdnn_data_sdk::utilities {
inline int64_t engineNameToId(std::string_view engineName) {
    // Implementation of this hash function is an implementation detail and is
    // up to the developer who implements it.  The developer is expected to create a robust
    // test suite to ensure the function is deterministic and is made to have minimal collision possibilities.
}
} // namespace hipdnn_data_sdk::utilities
```

Callers that need to accept either spelling of an engine — a name or a decimal/hexadecimal
ID — should use `engineNameOrIdToId` instead, which parses an ID spelling and otherwise
falls through to `engineNameToId`. That is what the frontend, the fallback ordering
environment variable and the heuristics config file all resolve through, so any spelling
`hipdnn_list_engines` prints can be pasted back into any of them.

### Plugin Implementation

Plugins use the hash function internally to convert their engine names to ids so they can communicate via the existing API.  Dispatch is unchanged: `hipdnnEnginePluginGetAllEngineIds`, `hipdnnEnginePluginGetApplicableEngineIds` and `hipdnnEnginePluginGetEngineDetails` still address engines only by `int64_t`.

Two additive channels carry a name alongside that, and they are not interchangeable:

| Channel | Available at | Confers a name? |
|---|---|---|
| `hipdnnEnginePluginGetEngineName` (EnginePluginApi.h) | Load, before any graph exists | Yes |
| `EngineDetails.name` (engine_details.fbs) | Only once a graph has been supplied | No |

The entry point is authoritative because it is the only one load-time admission can reach.
Admission is graph-blind, so it cannot see `EngineDetails.name`; a name arriving only through
the schema would reach a user without any gate having checked that it hashes to the engine's
ID, which is the invariant the whole design rests on. The backend therefore treats
`EngineDetails.name` as a record rather than a source: it logs a warning when the two
disagree and resolves to the entry point, and logs a warning and falls back to the registry
or the hex ID when the schema names an engine the entry point did not.

A plugin that wants its name honored must export the entry point. Populating
`EngineDetails.name` alone is a plugin defect. Both are optional and neither absence is an
error; see [Plugin-Supplied Engine Names](#plugin-supplied-engine-names).

### Frontend Implementation

The frontend accepts both names and IDs: `Graph::set_preferred_engine_id_ext` is overloaded on
`std::optional<int64_t>` and `const std::string&`, and the name overload resolves through
`engineNameOrIdToId` so a hexadecimal ID spelling reaches the engine it displays under. See
`Graph.hpp` for the signatures and the resolution timing they guarantee.

### Backend Duplicate Detection

The backend checks for duplicate engine IDs while loading plugins, and resolves the conflict
rather than failing the load:

- A plugin that repeats an ID **within itself** is malformed and is rejected whole.
- When two plugins claim the same ID, the plugin that loaded first keeps the engine. The
  backend logs an error naming both plugins and the engine, and drops the later plugin's
  engine while the rest of that plugin stays loaded.

A dropped engine is absent from enumeration, from applicability, and from dispatch, so it
cannot be reached by ID or by name.

## Plugin-Supplied Engine Names

A plugin names its own engines through the `hipdnnEnginePluginGetEngineName` entry point
(engine plugin API 1.4.0), so a drop-in plugin can name engines absent from the built-in
registry without a change to hipDNN source. A reported name takes precedence over the
registry, which remains the source for in-tree engines and the fallback for plugins that
report no name.

The hash relation is a **requirement**: an engine ID must equal the hash of its name, which
`HIPDNN_REGISTER_ENGINE` satisfies by construction. The backend verifies
`engineNameToId(name) == engineId` at load and drops any engine that fails, under the same
first-wins, log-an-error rule as a duplicate ID. An engine reporting no name is exempt, so
plugins predating the entry point keep loading unchanged.

That requirement is what makes names keys: they inherit the uniqueness of IDs, a name always
resolves to the engine reporting it, and a plugin-chosen name cannot shadow a registry entry
without colliding on the ID first.

## Engine Name Registration

### Process for Adding New Engine Names

1. **Choose a Unique Name**: Select a descriptive, unique engine name
   - Good: `"VENDOR_FAST_CONV_V2"`
   - Bad: `"ENGINE_1"`, `"FAST"`

2. **Test Locally**: Use the name in your plugin without modifying the header. An unregistered name
   loads like any other, as long as the engine's ID is the hash of it.

3. **Submit PR**: Register your engine name in
   `data_sdk/include/hipdnn_data_sdk/utilities/EngineNames.hpp`
   ```cpp
   HIPDNN_REGISTER_ENGINE(MY_VENDOR_FAST_CONV)
   ```
   Register through the macro, not through a bare `constexpr const char*`. The macro is what
   generates the `_ID` constant and instantiates the `EngineRegistrar`; a plain constant
   registers nothing, so the name stays invisible to collision detection and to reverse
   lookup even though it compiles cleanly.

4. **Document Capabilities**: Include documentation per the standards below

### Guidelines

- Engine names should be UPPER_CASE with underscores
- Include a vendor/organization prefix. This identifies the author, and it is the only thing
  keeping two vendors from independently shipping a `FAST_CONV`: names are unique keys now,
  so the second one to load is dropped rather than renamed
- Be descriptive about the engine's purpose
- Once merged, names should not be changed. A name is a user-referenceable key and it fixes
  the engine's ID, so changing it breaks both

### Forward Compatibility

If a plugin name is not known (not in the shared header), the hash function can still generate a unique ID.  An unregistered name is not an error in itself; what the backend enforces is that the engine's ID is the hash of the name it declares.  This allows newer plugins to be used without needing to update the shared header.

## Error Handling

### Duplicate IDs

Duplicate detection runs at two levels, because the two cases have different consequences.

**In-tree engines** are caught at static initialization. `EngineRegistrar` throws if the ID
it is registering is already claimed, distinguishing a duplicate registration (the same name
twice) from a hash collision (two different names hashing to one ID). Because every in-tree
engine goes through `HIPDNN_REGISTER_ENGINE`, this needs no hand-maintained list of engine
names and no test update when an engine is added, which is what the registration option was
selected for.

**Plugin-supplied engines** are checked at load, where throwing would take down an otherwise
usable plugin. The backend applies the first-wins rule described in
[Backend Duplicate Detection](#backend-duplicate-detection): the later engine is dropped and
an error is logged naming both plugins, the engine, and the colliding ID. The same rule
covers an engine whose ID does not hash from its declared name, and an engine whose name
collides with a registry entry.

## Examples

### Example 1: Custom Plugin Development

```cpp
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

// MyCustomPlugin.cpp
class MyCustomPlugin {
    void initialize() {
        // Constants generated by HIPDNN_REGISTER_ENGINE in the shared header
        const std::string engineName = hipdnn_data_sdk::utilities::MY_CUSTOM_ENGINE_NAME;
        const int64_t engineId = hipdnn_data_sdk::utilities::MY_CUSTOM_ENGINE_ID;

        // An engine absent from the shared header hashes its own name instead, and reports
        // that name through hipdnnEnginePluginGetEngineName:
        //   const int64_t engineId =
        //       hipdnn_data_sdk::utilities::engineNameToId("MY_CUSTOM_ENGINE");

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
    hipdnn_frontend::graph::Graph graph;

    // Option 1: Use string name directly
    graph.set_preferred_engine_id_ext("MIOPEN_ENGINE");

    // Option 2: Use the constant generated by HIPDNN_REGISTER_ENGINE
    graph.set_preferred_engine_id_ext(hipdnn_data_sdk::utilities::MIOPEN_ENGINE_NAME);

    // Option 3: Use a plugin-supplied name absent from the shared header. This resolves if a
    // loaded plugin declares it; otherwise the preference falls back to the heuristics.
    graph.set_preferred_engine_id_ext("MY_CUSTOM_ENGINE_V2");

    // Option 4: Still support int64_t for compatibility
    graph.set_preferred_engine_id_ext(0x123456789ABCDEF0LL);
}
```

## Future Improvements

### Benchmarking Support

In the future we would like to be able to compare plugins against themselves over time in order to track performance and regressions.  To facilitate this we can add a compile time override that appends a prefix to the engine name when generating the ID.  This will allow multiple unique IDs for the same engine name when benchmarking without needing to modify the engine name registration header.  The engine under test just needs to be compiled with the `HIPDNN_BENCHMARK_MODE` macro defined.

```cpp
// Update EngineNames.hpp

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
    inline constexpr const char* name##_NAME = HIPDNN_CONCAT_PREFIX(HIPDNN_ENGINE_PREFIX, name); \
    inline const int64_t name##_ID = \
        hipdnn_data_sdk::utilities::engineNameToId(name##_NAME); \
    inline const hipdnn_data_sdk::utilities::EngineRegistrar name##_registrar{name##_NAME};
```

Note that this shifts an engine's ID, so a benchmark build and a normal build cannot share a
plugin binary or a cached heuristics config.
