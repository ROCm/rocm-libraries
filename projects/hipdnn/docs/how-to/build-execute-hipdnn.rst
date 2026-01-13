.. meta::
  :description: Build and execute operation graphs in hipDNN
  :keywords: hipDNN, ROCm, API, how-to 

.. _build-execute:

********************************************
Build and execute operation graphs in hipDNN
********************************************

This section covers how to use the various components of hipDNN in your applications.

### Using the Frontend

The hipDNN frontend provides a C++ header-only API for building and executing operation graphs. For detailed architecture and design information, see the [Frontend section in the Design Guide](./Design.md#frontend).

#### File Structure
- Library includes: [`frontend/include/`](../frontend/include/)
- Unit tests: [`frontend/tests/`](../frontend/tests/)
- Samples: [`samples/`](../samples/)

### Using the Backend

The hipDNN backend is a shared library that provides the core C API for graph execution and plugin management. For comprehensive details about the backend architecture, descriptor types, and workflow, see the [Backend section in the Design Guide](./Design.md#backend).

#### File Structure
- Public includes: [`backend/include/`](../backend/include/)
- Public API tests: [`tests/backend/`](../tests/backend/)

### Using the SDKs

hipDNN provides three header-only C++ SDK libraries for plugin development and testing. For complete SDK functionality and roadmap, see the [SDKs section in the Design Guide](./Design.md#sdks).

#### Key Components
- **Data SDK**: Schema files and data structures: [`data_sdk/schemas/`](../data_sdk/schemas/)
- **Plugin SDK**: Plugin interface definitions: [`plugin_sdk/include/hipdnn_plugin_sdk/EnginePluginApi.h`](../plugin_sdk/include/hipdnn_plugin_sdk/EnginePluginApi.h)
- **Test SDK**: Test utilities and CPU reference implementations: [`test_sdk/include/hipdnn_test_sdk/`](../test_sdk/include/hipdnn_test_sdk/)
- Logging: [`data_sdk/include/hipdnn_data_sdk/logging/Logger.hpp`](../data_sdk/include/hipdnn_data_sdk/logging/Logger.hpp)

### CMake Integration

hipDNN components can be easily integrated into your CMake projects using the installed package files.

> [!NOTE]
> Enable PIC/PIE to ensure compatibility with the plugin loader system (dlopen). This prevents potential Thread Local Storage (TLS) allocation issues (such as static TLS exhaustion) between the executable and dynamically loaded backend plugins.
> ```cmake
> set(CMAKE_POSITION_INDEPENDENT_CODE ON)
> ```

#### Frontend Integration
```cmake
find_package(hipdnn_frontend REQUIRED)
target_link_libraries(your_target PRIVATE hipdnn_frontend)
```

#### Backend Integration
```cmake
find_package(hipdnn_backend REQUIRED)
target_link_libraries(your_target PRIVATE hipdnn_backend)
```

#### Data SDK Integration
```cmake
find_package(hipdnn_data_sdk REQUIRED)
target_link_libraries(your_plugin PRIVATE hipdnn_data_sdk)
```

#### Plugin SDK Integration
```cmake
find_package(hipdnn_plugin_sdk REQUIRED)
target_link_libraries(your_plugin PRIVATE hipdnn_plugin_sdk)
```

#### Test SDK Integration
```cmake
find_package(hipdnn_test_sdk REQUIRED)
target_link_libraries(your_test PRIVATE hipdnn_test_sdk)
```

#### Using AMD Half or BFloat16 Types
If you use AMD half or bfloat16 types (via the Data SDK's `UtilsFp16.hpp` or `UtilsBfp16.hpp`), you need:
```cmake
find_package(hip REQUIRED)
enable_language(HIP)
target_link_libraries(your_target hip::host hip::device)
```

> [!NOTE]
> 📝 If CMake cannot find the packages after installation, ensure your `CMAKE_PREFIX_PATH` includes the install location. By default on Linux systems, hipDNN CMake files are installed to `/opt/rocm/lib/cmake`.

### Logging Setup

hipDNN uses the spdlog header-only library for logging. See the [Environment docs](./Environment.md#logging-configuration) for further details.

> [!CAUTION]
> There is a known issue on Windows where logging must be explicitly shut down before the application exits to ensure all log messages are flushed and resources are released. See [spdlog Windows Issues](https://github.com/gabime/spdlog/wiki/Asynchronous-logging#windows-issues) for more information.
> ```cpp
> spdlog::shutdown();
> ```

### Working with Schemas

hipDNN uses FlatBuffers for schema-based data objects to describe graphs and operations.

#### Key Concepts
- Graphs and operations are defined using `.fbs` schema files
- Attributes marked as `long` types in graphs are foreign keys to the `uid` in `tensor_attributes`
- Schema files are located in [`data_sdk/schemas/`](../data_sdk/schemas/)
---