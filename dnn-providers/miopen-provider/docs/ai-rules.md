---
alwaysApply: true
---

# AI Rules for MIOpen Provider Plugin

## Project Overview

MIOpen Provider is a hipDNN plugin that wraps AMD's MIOpen library to provide GPU-accelerated deep learning operations. This plugin implements the hipDNN Plugin SDK interface to expose MIOpen kernels as execution engines.

**hipDNN Context**: hipDNN is a graph-based deep learning library for AMD GPUs with a plugin-based architecture. Plugins like this one implement the Plugin SDK to provide engine implementations that execute operation graphs.

### Plugin Architecture

This plugin implements the following hipDNN Plugin SDK interfaces:

| Interface | File | Purpose |
|-----------|------|---------|
| **Plugin API** | `hipdnn_plugin_sdk/PluginApi.h` | Plugin metadata (name, version, type), error reporting, logging callbacks |
| **Engine Plugin API** | `hipdnn_plugin_sdk/EnginePluginApi.h` | Engine lifecycle - create handle, set stream, get engines, execute graphs |
| **IEngine** | `hipdnn_plugin_sdk/interfaces/IEngine.hpp` | Engine interface - checks applicability, provides details, creates plans |
| **IPlanBuilder** | `hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp` | Builds execution plans for specific operation patterns |
| **IPlan** | `hipdnn_plugin_sdk/interfaces/IPlan.hpp` | Executable plan that runs operations on GPU |

### Execution Flow

```
FlatBuffer Graph → Engine Selection → Plan Creation → GPU Execution
       ↓                  ↓                 ↓              ↓
   IGraph API    isApplicable()     buildPlan()     execute()
```

1. **Graph Analysis**: Backend deserializes FlatBuffer graph, queries plugins for applicable engines
2. **Engine Selection**: Plugin's `isApplicable()` checks if any plan builder can handle the graph
3. **Plan Building**: Selected plan builder creates concrete `IPlan` instance with MIOpen setup
4. **Execution**: Plan's `execute()` launches MIOpen kernels on GPU

### Component Linkage (Do Not Modify)

- **Plugin uses**: Plugin SDK (header-only), Flatbuffers SDK (header-only), Data SDK (header-only), MIOpen library
- **Plugin SDK uses**: Flatbuffers SDK, Data SDK
- **Flatbuffers SDK uses**: Data SDK, FlatBuffers
- **No direct dependencies** on hipDNN Backend or Frontend

### Key Classes

- **HipdnnMiopenHandle**: Plugin instance with MIOpen handle, stream, engine manager
- **MiopenContainer**: Owns the `EngineManager` and registers engines and plan builders
- **EngineManager** (from Plugin SDK): Manages available engines and routes requests to applicable engines
- **MiopenEngine**: Concrete engine with multiple plan builders for different operations
- **HipdnnMiopenContext**: Stores the execution plan for a graph
- **HipdnnMiopenSettings**: Per-context execution settings (benchmarking, workspace size limit)
- **Plan Builders**: `MiopenConvPlanBuilder`, `MiopenBatchnormPlanBuilder`, etc.
- **Plans**: `ConvFwdPlan`, `BatchnormFwdInferencePlan`, etc.

---

## Building & Testing

### Build Commands

**Via Superbuild (recommended)**:
```bash
cd <workspace>
cmake --preset miopen-provider
cmake --build build
```

This uses the `miopen-provider` preset which builds both hipDNN and miopen-provider together. Other available presets: `hipdnn` (hipDNN only), `hipblaslt-provider` (hipDNN + hipblaslt-provider). See the root `CMakePresets.json` for the full list.

In the superbuild, targets are prefixed and YAML category-driven (e.g., `miopen-provider-check`, `miopen-provider-quick-check`). See `projects/hipdnn/docs/Superbuild.md` for full details.

**Standalone build** (requires hipDNN installed):
```bash
cd <workspace>/dnn-providers/miopen-provider
mkdir -p build && cd build
cmake -GNinja -DCMAKE_PREFIX_PATH=<hipdnn-install-path> -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ ..
ninja
```

### Test Binaries

| Binary | Tests | Typical Use |
|--------|-------|-------------|
| `miopen_plugin_tests` | Plugin unit tests | Engine manager, plan builders, MIOpen integration |
| `miopen_plugin_integration_tests` | GPU integration tests | End-to-end operation execution |

### Running Specific Tests
Use `--gtest_filter` for fast iteration:
```bash
./bin/miopen_plugin_tests --gtest_filter="TestMiopenEngine.*"
```

### When Modifying Code

**Only build or run tests if explicitly requested in the user's prompt.** Do not proactively run `ninja`, test binaries, or build commands unless asked.

When requested to build/test:
1. Rebuild with `ninja` (from build directory)
2. Run relevant tests using `--gtest_filter` for fast iteration

---

## C++ Code Style

### Naming Conventions
- CamelCase for class/struct names (e.g., `MiopenEngine`, `ConvolutionPlanBuilder`)
- camelCase for functions, variables, and private members (e.g., `buildPlan()`, `engineId`)
- Underscore prefix for private members (e.g., `_miopenHandle`, `_planBuilders`)

### File Headers
- Copyright header on all source files: `// Copyright © Advanced Micro Devices, Inc., or its affiliates.` / `// SPDX-License-Identifier:  MIT`
- `#pragma once` immediately after copyright in header files (.h, .hpp)

### Code Practices
- Use `auto` with casts to avoid type duplication: `auto tensor = static_cast<float*>(data)`
- Use `auto` when initializing variables, unless the type is not obvious
- **Avoid implicit casts** — use explicit `static_cast<>`. The codebase compiles with `-Wconversion` and `-Wsign-conversion`
- Always use braces for if/for/while bodies, even single-line
- Use CMake for managing C/C++ dependencies
- Use Flatbuffers for serialization (via Flatbuffers SDK wrapper classes: IGraph, GraphWrapper, etc.)

### Testing
- Use Google Test (gtest) framework for all C/C++ tests
- Never generate a `main()` function in test files — `tests/main.cpp` and `integration_tests/main.cpp` provide it
- Use TEST(), TEST_F(), or TEST_P() macros as appropriate

#### Test Suite Naming

Rules apply to the TestSuite name (first param of `TEST` / `TEST_F` / `TEST_P`). TestCase (second param) should be PascalCase.

**Composition (left → right):**

1. Required `Test` (unit tests) or `Integration` (integration tests) prefix, always first
2. Optional `Gpu` immediately after `Test`/`Integration` if the test needs GPU support
3. Core Feature / Subject under test (PascalCase, no underscores); may include a layout token: `Nhwc`, `Nchw`, `Ndhwc`, `Ncdhw`
4. Optional Datatype token: `Bfp16`, `Fp16`, `Fp32`

Omit any optional position that does not apply.

**Unit tests**: Mirror the class under test — `TestMyClass` or `TestGpuMyClass` if GPU is required.

**Valid examples:**
```
IntegrationGpuConvolutionFwdFp32   TestGpuMiopenEngineFp32
TestMiopenEngineManager            TestConvolutionPlanBuilder
```
