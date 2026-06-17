# hipDNN How-To Guide

This guide provides practical information for both using hipDNN components and extending the framework with new functionality.

## Table of Contents

- [Consuming hipDNN](#consuming-hipdnn)
  - [Using the Frontend](#using-the-frontend)
  - [Using the Backend](#using-the-backend)
  - [Using the SDKs](#using-the-sdks)
  - [CMake Integration](#cmake-integration)
  - [Logging Setup](#logging-setup)
  - [Working with Schemas](#working-with-schemas)
  - [Configuring Engine Knobs](#configuring-engine-knobs)
- [Extending hipDNN](#extending-hipdnn)
  - [Adding a New Plugin](#adding-a-new-plugin)
  - [Adding a New Operation](#adding-a-new-operation)
  - [Descriptor Code Generator](#descriptor-code-generator)
  - [Development Workflow](#development-workflow)

---

## Consuming hipDNN

> [!TIP]
> For a minimal end-to-end example of using hipDNN in a CMake project, see the [Consumer Quick Start](./ConsumerQuickStart.md).

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
- **FlatBuffers SDK**: Schema files and generated data objects: [`flatbuffers_sdk/schemas/`](../flatbuffers_sdk/schemas/)
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

### Working with Schemas

hipDNN uses FlatBuffers for schema-based data objects to describe graphs and operations.

#### Key Concepts
- Graphs and operations are defined using `.fbs` schema files
- Attributes marked as `long` types in graphs are foreign keys to the `uid` in `tensor_attributes`
- Schema files are located in [`flatbuffers_sdk/schemas/`](../flatbuffers_sdk/schemas/)

### Configuring Engine Knobs

hipDNN engines support runtime configuration through **knobs** - configurable parameters that control engine behavior, performance tuning, and feature selection.

> [!TIP]
> For comprehensive knobs documentation including all available knobs, constraints, validation, and advanced usage, see the [Knobs Documentation](./Knobs.md).

---

## Extending hipDNN

This section covers how to extend hipDNN with new functionality.

### Adding a New Plugin

Plugins extend hipDNN to support new or additional implementations of kernel engines, benchmarking, and heuristics. The Plugin SDK provides interfaces and utilities to simplify plugin development:

- **Engine interfaces**: `IEngine`, `IPlanBuilder`, `IPlan` templates for building plugin components
- **Engine management**: `EngineManager` template for managing multiple engines
- **Knob utilities**: `KnobFactory`, `KnobSettingFactory`, and `GlobalKnobDefines` for implementing runtime-configurable knobs

For comprehensive guidance on plugin development, including architecture details, implementation steps, and examples, see the [Plugin Development Guide](./PluginDevelopment.md).

### Adding a New Operation

Adding a new operation touches every layer of the stack — FlatBuffers schema, backend descriptor and enums, frontend attributes/packer/unpacker/node/Graph API, JSON utilities, optional Python bindings, and tests at four levels. The [`hipdnn-codegen`](../tools/DescriptorGenerator/.claude/skills/hipdnn-codegen/SKILL.md) agent skill (backed by [`tools/DescriptorGenerator/`](../tools/DescriptorGenerator/)) automates the mechanical parts.

> [!TIP]
> See the [Adding a New Operation Guide](./AddingNewOperations.md) for the full contributor walkthrough — cuDNN naming parity rules, the recommended workflow, layer-by-layer reference, testing matrix, and PR checklist.

#### Plugin Integration

Once the operation lands across the layers above, refer to the [Plugin Development Guide](./PluginDevelopment.md) to implement execution in target plugins.

### Descriptor Code Generator

The Descriptor Code Generator ([`tools/DescriptorGenerator/`](../tools/DescriptorGenerator/)) is a Python tool that generates the C++ boilerplate required to land a new operation type in hipDNN. Each operation in hipDNN requires a consistent set of files — descriptors, packers, unpackers, attributes, nodes, enum entries, factory wiring, and tests — and the generator produces all of them from a single YAML configuration.

#### What It Generates

From one YAML config, the tool produces:

- **Backend descriptor** (`.hpp`/`.cpp`) with `setAttribute`/`getAttribute`, `finalize`, `buildNode`, `fromNode`, and `toString`
- **Frontend packer** and **unpacker** for lowering and lifting between frontend graph nodes and backend descriptors
- **Frontend attributes** class and **node** class with graph method
- **Unit tests** for the descriptor, graph building, and `fromNode` round-trips
- **Integration tests** for lowering and lifting round-trips
- **Fragment files** with enum entries, factory cases, CMake entries, and string utility switch cases for insertion into existing shared files

#### When to Use It

| Scenario | Generator Mode |
|----------|---------------|
| Brand new operation (nothing exists yet) | `--mode full` |
| Adding backend only (frontend exists or will be added later) | `--mode backend` |
| Adding frontend only (backend descriptor already exists) | `--mode frontend` |

#### Existing Configurations

The tool ships with validated configs for all current operations:

| Config | Operation |
|--------|-----------|
| `convolution_fwd.yaml` | Convolution forward (reference config) |
| `convolution_bwd.yaml` | Convolution backward data |
| `convolution_wrw.yaml` | Convolution backward weights |
| `matmul.yaml` | Matrix multiplication |
| `pointwise.yaml` | Pointwise operations |
| `batchnorm.yaml` | Batch normalization (training forward) |
| `batchnorm_backward.yaml` | Batch normalization backward |
| `batchnorm_inference.yaml` | Batch normalization inference |
| `batchnorm_inference_variance_ext.yaml` | Batch normalization inference (variance extension) |
| `sdpa.yaml` | Scaled dot-product attention |

These serve as references when creating a config for a new operation. Use `convolution_fwd.yaml` as the primary reference — it exercises all config features.

#### Quick Start

```bash
cd projects/hipdnn/tools/DescriptorGenerator

# One-time setup
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Preview what would be generated (dry run)
.venv/bin/python generate.py \
    --config configs/your_operation.yaml \
    --output-dir /tmp/preview \
    --mode full \
    --dry-run

# Generate into the project tree
.venv/bin/python generate.py \
    --config configs/your_operation.yaml \
    --output-dir ../../ \
    --mode full
```

For full documentation on YAML config format, field types, mode enum definitions, and post-generation integration steps, see the [Descriptor Code Generator README](../tools/DescriptorGenerator/README.md).

---

## Development Workflow

### Typical Development Flow

1. **For New Operations** (using the code generator):
   ```
   FBS Schema → YAML Config → Code Generator → Place Files & Fragments → Plugin Implementation → Tests
   ```

2. **For New Operations** (manual):
   ```
   Data SDK Schema → Backend Descriptor → Frontend Classes → Plugin Implementation → Tests
   ```

3. **For Existing Operations in New Plugins**:
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

### CI Maintenance

- **TheRock CI**: Uses a pinned Git commit hash from the TheRock repository. `therock_ci.yml` and several other workflows need their hash updated at a frequent cadence for CI to build hipDNN with recent deps.
- **ROCm Version (hipdnn-clang-tidy.yml)**: Uses a fixed ROCm release version from TheRock artifacts (e.g., `7.11.0a20260112`). Update on-demand by changing the `--release` arg in the `install_rocm_from_artifacts.py` call. It will **need** to be bumped when new dependency APIs are required that are absent from past ROCm releases.


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
```
