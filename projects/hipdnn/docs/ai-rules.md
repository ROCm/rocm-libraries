---
alwaysApply: true
---

# AI Rules for hipDNN Project

## Project Overview & Architecture

hipDNN is a graph-based deep learning library for AMD GPUs with a plugin-based architecture.

### Component Architecture (Do Not Modify Boundaries (linkage))
| Component | Type | Links To | Purpose |
|-----------|------|----------|---------|
| **Backend** (`backend/`) | Shared library (C API) | Data SDK | Core engine, plugin loading, graph execution |
| **Frontend** (`frontend/`) | Header-only C++ | Backend, Data SDK | User-friendly wrapper around backend C API |
| **Data SDK** (`data_sdk/`) | Header-only | Third-party deps | Shared data objects, Flatbuffer schemas, logging |
| **Plugin SDK** (`plugin_sdk/`) | Header-only | Data SDK | Interfaces for plugin development |
| **Plugins** (`plugins/`) | Shared libraries | Plugin SDK, Data SDK | Engine implementations (e.g., MIOpen Plugin) |
| **Test SDK** (`test_sdk/`) | Header-only | Data SDK | Shared test utilities |

---

## Building & Testing

### Build Commands
```bash
cd <workspace>/projects/hipdnn
mkdir -p build && cd build
cmake -GNinja ..

ninja              # Build everything
ninja check        # Build and run ALL tests
ninja unit-check   # Unit tests only (faster)
ninja integration-check  # Integration tests only
```

### Test Binaries in `build/bin/`
| Binary | Tests | Typical Use |
|--------|-------|-------------|
| `hipdnn_backend_tests` | Backend unit tests | Core backend logic |
| `hipdnn_frontend_tests` | Frontend unit tests | Frontend wrapper logic |
| `hipdnn_data_sdk_tests` | Data SDK unit tests | Flatbuffer objects, utilities |
| `hipdnn_plugin_sdk_tests` | Plugin SDK unit tests | Plugin interface utilities |
| `hipdnn_test_sdk_tests` | Test SDK unit tests | Test utility functions |
| `public_hipdnn_backend_tests` | Backend API tests | Public C API black-box tests |
| `public_hipdnn_frontend_tests` | Frontend integration tests | E2E frontend tests |
| `miopen_plugin_tests` | MIOpen plugin unit tests | Plugin-specific tests |
| `miopen_plugin_integration_tests` | MIOpen integration tests | GPU-required E2E tests |

### Running Specific Tests
Use `--gtest_filter` for fast iteration:
```bash
./build/bin/hipdnn_backend_tests --gtest_filter="TestBackendDescriptor.*"
```

### When Modifying Code
**Only build or run tests if explicitly requested in the user's prompt.** Do not proactively run `ninja`, `ninja check`, or test binaries unless asked.

When requested to build/test:
1. Rebuild with `ninja` (from build directory)
2. Run relevant tests using `--gtest_filter` for fast iteration

---

## C++ Code Style

### Naming Conventions
- CamelCase for class/struct names (e.g., `BatchNormTestCase`, `SimpleTensorBundle`)
- camelCase for functions, variables, and private members (e.g., `setupEnvironment()`, `tensorData`)
- Underscore prefix for private members (e.g., `_handle`, `_testData`)

### File Headers
- Copyright header on all source files: `// Copyright © Advanced Micro Devices, Inc., or its affiliates.` / `// SPDX-License-Identifier:  MIT`
- `#pragma once` immediately after copyright in header files (.h, .hpp)

### Code Practices
- Use `auto` with casts to avoid type duplication: `auto tensor = static_cast<float*>(data)`
- Use `auto` when initializing variables, unless the type is not obvious
- **Avoid implicit casts** — use explicit `static_cast<>`. The codebase compiles with `-Wconversion` and `-Wsign-conversion`
- Always use braces for if/for/while bodies, even single-line
- Use CMake for managing C/C++ dependencies
- Use Flatbuffers for serialization needs

### Testing
- Use Google Test (gtest) framework for all C/C++ tests
- Never generate a `main()` function in test files — gtest provides its own
- Use TEST(), TEST_F(), TEST_P(), or TYPED_TEST() macros as appropriate
- **Prefer TYPED_TEST for multi-datatype tests** — when testing across float, half, bfloat16, use TYPED_TEST instead of duplicating test code

#### Test Suite Naming

Rules apply to the TestSuite name (first param of `TEST` / `TEST_F` / `TEST_P`). TestCase (second param) should be PascalCase.

**Composition (left → right):**

1. Optional `Integration` prefix (only for integration tests, always first)
2. Optional `Gpu` (immediately after `Integration` if both apply, otherwise first)
3. Core Feature / Subject under test (PascalCase, no underscores)
4. Optional Datatype token: `Bfp16`, `Fp16`, `Fp32`

Omit any position that does not apply.

**Unit tests**: Mirror the class under test — `TestMyClass` or `GpuTestMyClass` if GPU is required.

**Valid examples:**
```
IntegrationGpuConvolutionPlannerNchwFp32   GpuTestActivationKernelNchwFp32
GpuTestExecutionPlanBuilderFp32            IntegrationGraphFusion
TestConvolutionHeuristicsFp32              TestConvolutionHeuristics
```

#### Choosing Between TYPED_TEST and TEST_P

| Scenario | Approach |
|----------|----------|
| Single type, no params | `TEST()` / `TEST_F()` |
| Single type, with params | `TEST_P()` |
| Multi-type, no params | `TYPED_TEST` |
| Multi-type, with params | `TEST_P` + multi-declarations |

**Key Principle:** Prefer `TEST_P` over `TYPED_TEST` when both type variation and parameterized cases are needed. `TYPED_TEST` and `TEST_P` don't mix well together.

**Multi-declarations**: Use explicit type aliases (`using ConvTestFp32 = ConvTest<float>`) with separate `TEST_P` and `INSTANTIATE_TEST_SUITE_P` per type. Prefer this over macros for debuggability.

**TypePair pattern**: For testing type combinations (e.g., input + compute type), define a `TypePair<T1, T2>` struct with `InputType`/`ComputeType` aliases and use with `TYPED_TEST_SUITE`.

---

## Conditional Guidelines

When modifying public API headers (under `backend/include/`, `frontend/include/hipdnn_frontend/`, `plugin_sdk/include/`), add Doxygen comments (`/** @brief ... */` for classes/functions/files, `///<` for enum values) to any new or changed public API. Exclude `detail/` subdirectories and generated files. For full style details, see `docs/doxygen-guidelines.md`.
