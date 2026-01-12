.. meta::
  :description: Component how-to
  :keywords: Component, ROCm, API, how-to 

***************************************************
Add new plugins and operations to hipDNN (Advanced)
***************************************************

## Extending hipDNN

This section covers how to extend hipDNN with new functionality.

### Adding a New Plugin

Plugins extend hipDNN to support new or additional implementations of kernel engines, benchmarking, and heuristics. For comprehensive guidance on plugin development, including architecture details, implementation steps, and examples, see the [Plugin Development Guide](./PluginDevelopment.md).

### Adding a New Operation

Adding a new operation requires coordinated changes across multiple components. Here's the complete workflow:

#### Prerequisites

When adding a completely new operation type (not currently supported in hipDNN), you'll need to:

1. Define the operation in the Data SDK schemas
2. Create frontend classes
3. Implement the operation in target plugins

#### Data SDK Schema Changes

If the operation is new to hipDNN, start by defining its data structures:

1. **Create Attribute Schema**
   - Add a new `.fbs` file in [`data_sdk/schemas/`](../data_sdk/schemas/)
   - Define the operation's attributes (parameters, configurations)
   - Example: [`data_sdk/schemas/batchnorm_attributes.fbs`](../data_sdk/schemas/batchnorm_attributes.fbs)

2. **Update Graph Schema**
   - Modify [`data_sdk/schemas/graph.fbs`](../data_sdk/schemas/graph.fbs)
   - Add your new attributes to the `NodeAttributes` union
   - Include your schema file

Example:
```flatbuffers
include "your_operation_attributes.fbs";

union NodeAttributes {
    BatchnormInferenceAttributes,
    PointwiseAttributes,
    ...
    YourOperationAttributes  // Add your new operation
}
```

After updating FlatBuffer schemas, regenerate the C++ headers:

```bash
ninja generate_hipdnn_data_sdk_headers
```

#### Frontend Implementation

Create C++ classes to expose the operation to users:

1. **Create Node Class**
   - Add header file in [`frontend/include/hipdnn_frontend/node/`](../frontend/include/hipdnn_frontend/node/)
   - Inherit from the base `Node` class
   - Example: [`frontend/include/hipdnn_frontend/node/BatchnormNode.hpp`](../frontend/include/hipdnn_frontend/node/BatchnormNode.hpp)

2. **Create Attribute Classes**
   - Add corresponding attribute classes in [`frontend/include/hipdnn_frontend/attributes/`](../frontend/include/hipdnn_frontend/attributes/)
   - These wrap the FlatBuffer-generated structures

3. **Update Frontend Tests**
   - Add tests for your new node and attributes
   - See examples in [`frontend/tests/`](../frontend/tests/)

#### Plugin Integration

Refer to the [Plugin Development Guide](./PluginDevelopment.md) to implement the operation execution in target plugins.

---

## Development Workflow

### Typical Development Flow

1. **For New Operations**:
   ```
   Data SDK Schema → Frontend Classes → Plugin Implementation → Tests
   ```

2. **For Existing Operations in New Plugins**:
   ```
   Plugin Implementation → Integration Tests
   ```

### Building and Testing

1. **Rebuild hipDNN**: After changing hipDNN, you will need to rebuild. See the [quick start steps in the build guide](./Building.md#quick-start-guide), or rebuild the specific targets.

3. **Test Your Implementation**:
   - Unit tests for individual components
   - Integration tests for new and untested end-to-end functionality

### Important Considerations

- **Backward Compatibility**: Ensure schema changes don't break existing operations
- **Plugin Discovery**: For example, engine plugins are loaded from `hipdnn_plugins/engines/` relative to the backend library
- **Error Handling**: Implement proper error reporting through the plugin API
- **Performance**: Optimization is critical for facilitating plugin adoption

### Debugging Tips

- Enable logging with environment variables (see [Environment Configuration](./Environment.md))
- Use integration tests to verify operation behavior
- Check plugin loading with `HIPDNN_LOG_LEVEL=info`
- For plugin issues, check the default plugin path or use custom paths with `hipdnnSetEnginePluginPaths_ext`

## ⚠️ Troubleshooting

### Segmentation Faults during Graph Execution Plan Build

If you are seeing segfaults when building execution plans for graphs, this might be caused by Thread Local Storage (TLS) allocation issues (such as static TLS exhaustion) between the executable and dynamically loaded backend plugins.

To resolve this, enable PIC/PIE to ensure compatibility with the plugin loader system (dlopen). This setting instructs CMake to emit position-independent code (e.g., via `-fPIC`  or `-fPIE`), which is necessary for creating shared libraries or executables that load plugins dynamically.

```cmake
set(CMAKE_POSITION_INDEPENDENT_CODE ON)