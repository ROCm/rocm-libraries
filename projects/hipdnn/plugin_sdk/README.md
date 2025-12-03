# hipDNN Plugin SDK

A lightweight, header-only library providing the essential API and utilities for developing hipDNN plugins.

## Overview

The `hipdnn_plugin_sdk` is a focused subset of the main `hipdnn_sdk` that contains only the core components needed for plugin development. It provides:

- **Plugin API definitions** - Core plugin interface (`PluginApi.h`, `EnginePluginApi.h`)
- **Data types** - Plugin status codes and type definitions (`PluginApiDataTypes.h`)
- **Exception handling** - Plugin-specific exception classes and macros (`PluginException.hpp`)
- **Helper utilities** - Error management and common patterns (`PluginHelpers.hpp`, `PluginLastErrorManager.hpp`)
- **Flatbuffer wrappers** - Convenient C++ wrappers for flatbuffer data structures
- **Type helpers** - String conversion and formatting for plugin types (`PluginDataTypeHelpers.hpp`)

## Structure

```
plugin_sdk/
├── include/hipdnn_plugin_sdk/
│   ├── PluginApi.h                      # Core plugin API
│   ├── EnginePluginApi.h                # Engine plugin API
│   ├── PluginApiDataTypes.h             # Plugin data types
│   ├── PluginException.hpp              # Exception handling
│   ├── PluginHelpers.hpp                # Helper utilities
│   ├── PluginLastErrorManager.hpp       # Error management
│   ├── PluginDataTypeHelpers.hpp        # Type conversion utilities
│   └── flatbuffer_utilities/
│       ├── EngineConfigWrapper.hpp      # Engine config wrapper
│       ├── EngineDetailsWrapper.hpp     # Engine details wrapper
│       ├── GraphWrapper.hpp             # Graph wrapper
│       └── NodeWrapper.hpp              # Node wrapper
└── tests/                               # Unit tests
```

## Dependencies

The plugin SDK depends on:
- `hipdnn_sdk` - For logging infrastructure and data object schemas
- `spdlog` - For logging (transitively through hipdnn_sdk)
- `flatbuffers` - For serialization (transitively through hipdnn_sdk)

## Usage

### CMake Integration

```cmake
find_package(hipdnn_plugin_sdk REQUIRED)

add_library(my_plugin SHARED my_plugin.cpp)
target_link_libraries(my_plugin PRIVATE hipdnn_plugin_sdk)
```

### Example Plugin Implementation

```cpp
#include <hipdnn_plugin_sdk/PluginApi.h>
#include <hipdnn_plugin_sdk/PluginHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginLastErrorManager.hpp>

using namespace hipdnn_plugin;

// Define the thread_local error buffer
// NOLINTNEXTLINE(modernize-avoid-c-arrays)
thread_local char PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH] = "";

extern "C" {

HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)
{
    return tryCatch([&]() {
        throwIfNull(name);
        *name = "MyPlugin";
    });
}

HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)
{
    return tryCatch([&]() {
        throwIfNull(version);
        *version = "1.0.0";
    });
}

HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)
{
    return tryCatch([&]() {
        throwIfNull(type);
        *type = HIPDNN_PLUGIN_TYPE_ENGINE;
    });
}

HIPDNN_PLUGIN_EXPORT void hipdnnPluginGetLastErrorString(const char** error_str)
{
    if(error_str != nullptr)
    {
        *error_str = PluginLastErrorManager::getLastError();
    }
}

} // extern "C"
```

## Testing

The plugin SDK includes comprehensive unit tests covering:
- Exception handling and macros
- Helper utilities (`tryCatch`, `throwIfNull`)
- Error management
- Type conversion and formatting
- Flatbuffer wrapper functionality

Run tests with:
```bash
cd build
ninja hipdnn_plugin_sdk_tests
./bin/hipdnn_plugin_sdk_tests
```

## Design Principles

1. **Header-only** - No compilation required, just include paths
2. **Minimal dependencies** - Only essential dependencies for plugin development
3. **Self-contained** - Can be used independently of the full SDK
4. **Backward compatible** - Maintains compatibility with existing SDK through symlinks
