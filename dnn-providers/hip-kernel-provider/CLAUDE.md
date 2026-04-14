# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HIP Kernel Provider is a hipDNN plugin that implements GPU operations (batch normalization, RMS normalization, SDPA) using custom HIP kernels compiled at runtime via HIPRTC. It is a standalone project that depends on the hipDNN SDK packages (`hipdnn_data_sdk`, `hipdnn_plugin_sdk`).

## Build Commands

```bash
# Configure (from project root)
mkdir -p build && cd build
cmake -GNinja -DCMAKE_CXX_COMPILER=<path-to-amdclang>/clang++ ..

# If hipDNN SDK is not in the default ROCm path:
cmake -GNinja -DCMAKE_PREFIX_PATH=<path-to-hipdnn-install> -DCMAKE_CXX_COMPILER=<path-to-amdclang>/clang++ ..

# Build
ninja

# Build specific targets
ninja hip_kernel_provider              # Plugin shared library only
ninja hip_kernel_provider_tests        # Unit test executable
ninja hip_kernel_provider_integration_tests  # Integration test executable
```

### Key CMake Options

- `ENABLE_ASM_SDPA_ENGINE` (default: ON) - Build SDPA engine with ASM kernels
- `HIPKERNELPROVIDER_ENABLE_TESTS` (default: ON) - Build tests
- `HIPKERNELPROVIDER_ENABLE_COVERAGE` (default: OFF) - Code coverage instrumentation
- `ENABLE_CLANG_TIDY` (default: ON on Linux) - Static analysis
- `BUILD_ADDRESS_SANITIZER` (default: OFF) - AddressSanitizer

## Testing

```bash
# All tests
ninja check

# Unit tests only (CPU + basic GPU tests)
ninja unit-check
# Or directly: ./bin/hip_kernel_provider_tests

# Integration tests only (requires GPU)
ninja integration-check
# Or directly: ./bin/hip_kernel_provider_integration_tests

# Run a single test or test suite via GTest filter
./bin/hip_kernel_provider_tests --gtest_filter="TestSuiteName.TestName"
./bin/hip_kernel_provider_tests --gtest_filter="TestSuiteName.*"

# List available tests
./bin/hip_kernel_provider_tests --gtest_list_tests
```

## Linting and Static Analysis

```bash
# Clang-tidy (runs on all source files)
ninja tidy
ninja tidy-cxx          # C++ files only

# Clang-format check
ninja format-check
```

Builds use `-Werror` with extensive warning flags. All clang-tidy warnings are treated as errors.

## Architecture

### Plugin Structure

The plugin follows the hipDNN plugin architecture with dependency injection:

```
HipKernelHandle (opaque handle holding stream + container)
└── HipKernelContainer (DI container)
    ├── IDevicePropertyProvider → CurrentDevicePropertyProvider
    ├── IKernelCompiler → HipKernelCompiler
    └── EngineManager
        ├── HipKernelEngine (HIPRTC-based operations)
        │   └── IPlanBuilder[] → BatchnormPlanBuilder, RMSnormPlanBuilder
        └── AsmSdpaEngine (optimized ASM kernel engine, optional)
            └── IPlanBuilder[] → SdpaFwdPlanBuilder
```

### Plan Pattern

Each operation follows a three-class pattern:
1. **PlanBuilder** (`IPlanBuilder`) - Checks applicability and constructs plans
2. **ApplicabilityChecks** - Reusable validation logic for an operation
3. **Plan** (`IPlan`) - Compiles kernel via HIPRTC and executes on GPU

### Kernel Embedding System

Device kernel sources under `kernels/` are embedded as C++ string literals at CMake configure time (via `embed_kernel_sources()` in `kernels/CMakeLists.txt`). At runtime, plans look up kernel source by filename using `getKernelSrc()` / `getKernelInc()` and compile via HIPRTC. This means **CMake must be reconfigured** when kernel source files are modified.

### Library Targets

- `hip_kernel_provider` - Shared library (the actual plugin `.so`)
- `hip_kernel_provider_private` - Static library linking the same objects, used by unit tests to access internal symbols
- `hip_kernel_provider_impl` - Object library with all implementation

### Key Directories

- `src/engines/plans/` - Operation-specific plan builders and plans (Batchnorm, RMSnorm)
- `src/engines/asm_sdpa_engine/` - SDPA engine using pre-compiled ASM kernels (gfx942/MI300)
- `src/hip/` - HIPRTC wrapper infrastructure (HipProgram, HipKernel, HipKernelCompiler)
- `kernels/` - Device-side kernel source files embedded at build time
- `src/tests/` - Unit tests (GTest/GMock)
- `src/integration_tests/` - GPU integration tests (GTest, requires hardware)

## Naming Conventions (enforced by clang-tidy)

- Namespaces: `lower_case`
- Classes/Structs: `CamelCase`
- Functions/Methods: `camelBack`
- Variables/Parameters: `camelBack`
- Private/Protected members: `_` prefix (e.g., `_memberVar`)
- Static variables: `s_` prefix
- Enums: `CamelCase`, enum constants: `UPPER_CASE`
- Global constants/constexpr: `UPPER_CASE`

## Adding a New Operation

1. Add kernel sources to `kernels/<operation>/` and register in `kernels/CMakeLists.txt`
2. Create `*ApplicabilityChecks` class
3. Create `*Plan` class inheriting from `IPlan<HipKernelHandle>`
4. Create `*PlanBuilder` class inheriting from `IPlanBuilder<HipKernelHandle, HipKernelSettings, HipKernelContext>`
5. Register plan builder in the engine
6. Add unit tests in `src/tests/engines/plans/<Operation>/`
7. Add integration tests in `src/integration_tests/<Operation>/`
8. Add new source files to `src/CMakeLists.txt` and test CMakeLists

## Code Style

- C++17, formatted per `.clang-format` (WebKit-based, 100 col limit, 4-space indent)
- `#pragma once` for header guards
- Templates do NOT use export macros (weak linkage); regular functions do
- CXX visibility is hidden by default; only the public plugin API is exported
