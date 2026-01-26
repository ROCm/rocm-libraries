# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# From repository root
mkdir -p build && cd build
cmake -GNinja ..
ninja                    # Build everything
ninja check              # Run all tests
ninja unit-check         # Unit tests only (faster)
ninja integration-check  # Integration tests only
ninja format             # Auto-format code
ninja check_format       # Verify formatting
ninja clang-tidy         # Static analysis
```

For samples, first build and install hipDNN or load its libs, and then:
```bash
mkdir -p samples/build
cd samples/build
CXX=/opt/rocm/llvm/bin/clang++ cmake ..
ninja
ctest # Run all samples with verification
```


## Running Individual Tests

Test binaries are in `build/bin/`. Use `--gtest_filter` for fast iteration:

```bash
./build/bin/hipdnn_backend_tests --gtest_filter="TestBackendDescriptor.*"
./build/bin/hipdnn_frontend_tests --gtest_filter="TestGraph.SerializationRoundTrip"
./build/bin/public_hipdnn_frontend_tests --gtest_filter="*Convolution*"
```

Key test binaries:
- `hipdnn_backend_tests`, `hipdnn_frontend_tests` - Unit tests
- `public_hipdnn_backend_tests`, `public_hipdnn_frontend_tests` - Integration/API tests
- `miopen_plugin_integration_test` - GPU integration tests

## Architecture

hipDNN is a graph-based deep learning library for AMD GPUs with a plugin architecture.

```
User Application
    ↓
Frontend (header-only C++) → Backend (C API shared lib) → Plugins
    ↓                            ↓
Data SDK (header-only)      Plugin SDK (header-only)
```

| Component | Type | Purpose |
|-----------|------|---------|
| **Backend** | Shared library (C API) | Plugin loading, graph execution |
| **Frontend** | Header-only C++ | User-friendly wrapper |
| **Data SDK** | Header-only | FlatBuffer schemas, logging, utilities |
| **Plugin SDK** | Header-only | Plugin interface definitions |
| **Test SDK** | Header-only | CPU reference implementations, test utilities |
| **Plugins** | Shared libraries | Engine implementations (e.g., MIOpen Legacy) |

**Critical**: Component linkage boundaries must not be modified.

## Code Style

- **Classes/structs**: PascalCase (`BatchNormTestCase`)
- **Functions/variables**: camelCase (`setupEnvironment()`)
- **Private members**: underscore prefix (`_handle`)
- **Headers**: Copyright + SPDX + `#pragma once`
- **Casts**: Always explicit `static_cast<>` (compiles with `-Wconversion -Wsign-conversion`)
- **Control flow**: Always use braces, even for single lines

```cpp
// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once
```

## Test Naming

Format: `[Integration][Gpu]FeatureName[Datatype]`

- `Integration` - First position, only for integration tests
- `Gpu` - After Integration (or first), for GPU-required tests
- Datatype - `Fp32`, `Fp16`, `Bfp16` at end

Examples: `TestBatchnorm`, `GpuTestActivationKernelFp32`, `IntegrationGpuConvolutionNchwFp32`

## Key Documentation

- `docs/Design.md` - Architecture details
- `docs/Building.md` - Build prerequisites and options
- `docs/PluginDevelopment.md` - Plugin API guide
- `docs/OperationSupport.md` - Supported operations

## Important Notes

- Uses Google Test framework (never implement `main()`)
- Uses FlatBuffers for serialization (schemas in `data_sdk/schemas/`)
- GPU tests must include `SKIP_IF_NO_DEVICE()` macro
- All tests must pass with AddressSanitizer (`-DBUILD_ADDRESS_SANITIZER=ON`)
- Plugins are loaded from `hipdnn_plugins/engines/` or via `HIPDNN_PLUGIN_DIR` env var
