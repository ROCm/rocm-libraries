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

Adding a new operation requires coordinated changes across multiple components: Data SDK schemas, backend descriptors, frontend classes, tests, and plugin integration. hipDNN provides a **Descriptor Code Generator** tool that automates the most labor-intensive parts of this process.

#### Overview

The high-level workflow for adding a new operation is:

```
FBS Schema → YAML Config → Code Generator → Integration → Plugin Implementation → Tests
```

The code generator (`tools/DescriptorGenerator/`) produces backend descriptors, packers, unpackers, frontend attributes, nodes, graph methods, and comprehensive test suites from a single YAML configuration file. This replaces hours of manual copy-paste-adapt work and ensures consistency across all generated artifacts.

> [!TIP]
> See the [Descriptor Code Generator README](../tools/DescriptorGenerator/README.md) for full tool documentation, YAML config format, and field type reference.

#### Step 1: Define the FlatBuffer Schema

> [!NOTE]
> The FBS schema is used by the **backend** and **Data SDK** for serialization and internal data representation. The frontend does not depend on FlatBuffers directly — it uses the backend C API descriptors to build and inspect operation graphs.

Start by defining the operation's data structures:

1. **Create Attribute Schema**
   - Add a new `.fbs` file in [`flatbuffers_sdk/schemas/`](../flatbuffers_sdk/schemas/)
   - Define the operation's attributes (parameters, configurations)
   - Example: [`flatbuffers_sdk/schemas/batchnorm_attributes.fbs`](../flatbuffers_sdk/schemas/batchnorm_attributes.fbs)

2. **Update Graph Schema**
   - Modify [`flatbuffers_sdk/schemas/graph.fbs`](../flatbuffers_sdk/schemas/graph.fbs)
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

#### Step 2: Create the YAML Configuration

Create a YAML config in `tools/DescriptorGenerator/configs/` that maps your FBS schema fields to hipDNN backend API concepts. The config describes tensor fields, data fields, enum modes, frontend class names, and test data.

Use `convolution_fwd.yaml` as the reference config — it is the most complete and validated example. The mapping rules are:

| FBS Field Pattern | YAML Section | YAML `type` |
|---|---|---|
| `*_tensor_uid: long` | `tensor_fields` | (implicit) |
| `field: [long]` | `data_fields` | `vector_int64` |
| `field: SomeEnum` | `data_fields` | `mode` |
| `field: float` | `data_fields` | `scalar_float` |
| `field: long` (non-UID) | `data_fields` | `scalar_int64` |
| `field: bool` | `data_fields` | `bool` |

See the [Descriptor Code Generator README](../tools/DescriptorGenerator/README.md#yaml-config-format) for the full YAML format and all available field properties.

#### Step 3: Run the Code Generator

```bash
cd projects/hipdnn/tools/DescriptorGenerator

# One-time setup
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Generate all backend + frontend artifacts
.venv/bin/python generate.py \
    --config configs/your_operation.yaml \
    --output-dir /tmp/codegen-output \
    --mode full

# Or generate directly into the project tree
.venv/bin/python generate.py \
    --config configs/your_operation.yaml \
    --output-dir ../../ \
    --mode full
```

The generator supports four modes:

| Mode | What It Generates |
|------|-------------------|
| `backend` | Descriptor, packer, backend tests, CMake/enum fragments |
| `frontend` | Attributes class, node class, graph method, frontend tests |
| `full` | All backend + frontend artifacts |
| `lift-only` | Unpacker, fromNode tests, lifting integration tests, wiring fragments (for operations that already have descriptors) |

#### Step 4: Integrate Generated Code

After generation, the output directory contains ready-to-use C++ source files and fragment files. Place the generated source files into the project tree and insert the fragment snippets into existing shared files (enum headers, string utilities, factory switches, CMake lists).

The generated files include:

| Generated File | Target Location |
|----------------|-----------------|
| `<Op>OperationDescriptor.hpp/.cpp` | `backend/src/descriptors/` |
| `<Op>Packer.hpp` | `frontend/include/hipdnn_frontend/detail/` |
| `<Op>Unpacker.hpp` | `frontend/include/hipdnn_frontend/detail/` |
| `<Op>Attributes.hpp` | `frontend/include/hipdnn_frontend/attributes/` |
| `<Op>Node.hpp` | `frontend/include/hipdnn_frontend/node/` |
| `Test<Op>OperationDescriptor.cpp` | `backend/tests/descriptors/` |
| `TestGraphDescriptor<Op>.cpp` | `backend/tests/descriptors/` |
| `Integration<Op>DescriptorLowering.cpp` | `tests/frontend/` |
| `fragments/*.txt` | Snippets for manual insertion into existing files |

The fragment files tell you exactly what to insert and where — enum entries, factory cases, CMake entries, and string utility switch cases.

> [!IMPORTANT]
> The generator produces a starting point, not a final product. Review generated code against the current state of hipDNN. Files, enums, or plumbing may already exist partially. When a target file already exists, compare and merge rather than overwrite.

#### Step 5: Frontend Implementation

If you used `--mode full`, the generator produces frontend attributes, node, and graph method files. Otherwise, create them manually:

1. **Create Node Class**
   - Add header file in [`frontend/include/hipdnn_frontend/node/`](../frontend/include/hipdnn_frontend/node/)
   - Inherit from the base `Node` class
   - Example: [`frontend/include/hipdnn_frontend/node/BatchnormNode.hpp`](../frontend/include/hipdnn_frontend/node/BatchnormNode.hpp)

2. **Create Attribute Classes**
   - Add corresponding attribute classes in [`frontend/include/hipdnn_frontend/attributes/`](../frontend/include/hipdnn_frontend/attributes/)
   - These define operation-specific parameters for the frontend

3. **Update Frontend Tests**
   - Add tests for your new node and attributes
   - See examples in [`frontend/tests/`](../frontend/tests/)

#### Step 6: Plugin Integration

Refer to the [Plugin Development Guide](./PluginDevelopment.md) to implement the operation execution in target plugins.

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
| Adding lifting support to an existing operation | `--mode lift-only` |

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
