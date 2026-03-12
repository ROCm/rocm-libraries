# HIP Kernel Provider Plugin

A hipDNN plugin that provides GPU kernel implementations using HIP and HIPRTC for runtime kernel compilation.

:construction: **This project is under active development** :construction:

## Overview

The HIP Kernel Provider is a hipDNN plugin that implements operations using custom HIP kernels compiled at runtime via HIPRTC. It provides an alternative backend to MIOpen for certain operations, with fine-grained control over kernel implementation.

### Current Features

- **Batch Normalization Forward Inference**: Supports NCHW, NHWC, NCDHW, and NDHWC layouts with FP32, FP16, and BFP16 data types
- **Runtime Kernel Compilation**: Uses HIPRTC to compile kernels at runtime with device-specific optimizations
- **ICompilablePlan Interface**: Separates kernel compilation from execution for efficient re-execution
- **Comprehensive Testing**: 59 unit tests + 20 integration tests covering all supported configurations

## Architecture

The plugin follows the standard hipDNN plugin architecture:

```
HipKernelEngine
├── BatchnormPlanBuilder
│   ├── BatchnormApplicabilityChecks
│   └── BatchnormFwdInferencePlan (ICompilablePlan)
│       ├── compile() - HIPRTC compilation with device-specific options
│       └── execute() - Kernel launch on GPU
└── [Future plan builders...]
```

### Key Components

- **Engines** (`src/engines/`): High-level operation orchestration
- **Plans** (`src/engines/plans/`): Kernel-specific execution logic
- **HIP Infrastructure** (`src/hip/`): HIPRTC wrapper classes for compilation and execution
- **Kernels** (`kernels/`): Device-side kernel source code embedded at build time
- **Plugin SDK Integration**: Implements `ICompilablePlan`, `IPlanBuilder`, `IEngine` interfaces

## Building

This plugin should be built as a standalone plugin. To build the plugin, first install hipDNN on the system and then follow these steps:

1. Navigate to the `dnn-providers/hip-kernel-provider` directory.
2. Make a build directory using `mkdir build && cd build`.
3. Configure the build using `cmake -GNinja -DCMAKE_CXX_COMPILER=<path to amdclang>/clang++ ..`.
4. Finally, run `ninja` to build the plugin.

### Build Requirements

- hipDNN installed (via `ninja install` from hipDNN build)
- ROCm with HIP and HIPRTC
- CMake 3.16+
- Ninja build system
- C++17 compatible compiler (amdclang++ recommended)

### Testing

After building, run the test suites:

```bash
# Unit tests (CPU + basic GPU tests)
./bin/hip_kernel_plugin_tests

# Integration tests (full GPU pipeline tests)
./bin/hip_kernel_plugin_integration_tests
```

## Directory Structure

```
hip-kernel-provider/
├── src/
│   ├── engines/              # Engine and plan implementations
│   │   ├── HipKernelEngine.hpp/cpp
│   │   └── plans/
│   │       ├── BatchnormFwdInferencePlan.hpp/cpp
│   │       ├── BatchnormPlanBuilder.hpp/cpp
│   │       └── BatchnormApplicabilityChecks.hpp/cpp
│   ├── hip/                  # HIP/HIPRTC wrapper infrastructure
│   │   ├── HipProgram.hpp/cpp    # HIPRTC compilation wrapper
│   │   ├── HipKernel.hpp/cpp     # Kernel launch wrapper
│   │   └── HipUtils.hpp          # HIP error checking macros
│   ├── tests/                # Unit tests
│   └── integration_tests/    # Integration tests
├── kernels/                  # Device-side kernel sources
│   ├── batchnorm/            # Batch normalization kernels
│   ├── common/               # Shared kernel utilities
│   ├── types/                # Data type definitions
│   └── templates/            # CMake templates for kernel embedding
└── cmake/                    # CMake helper modules
```

## Kernel Embedding System

Kernel source files (`.cpp`, `.hpp`, `.h`) under `kernels/` are embedded as C++ string literals at CMake configure time. This allows runtime compilation via HIPRTC while keeping kernel sources as regular C++ files (with syntax highlighting, IDE support, etc.).

The embedding is handled by the `embed_kernel_sources()` CMake function in `kernels/CMakeLists.txt`.

## Supported Operations

### Batch Normalization Forward Inference

- **Layouts**: NCHW (4D), NHWC (4D), NCDHW (5D), NDHWC (5D)
- **Data Types**: FP32, FP16 mixed precision, BFP16 mixed precision
- **Features**: Spatial mode, per-activation mode
- **Fused Operations**: Activation functions (ReLU, etc.) - planned

## Development Status

### Completed (Phase 1-3)

- [x] HIP/HIPRTC infrastructure (HipProgram, HipKernel)
- [x] Kernel embedding build system
- [x] Engine and plan builder framework
- [x] Batch normalization forward inference (all layouts, data types)
- [x] ICompilablePlan interface implementation
- [x] Centralized device configuration
- [x] Comprehensive unit and integration tests

### In Progress

- [ ] Bug fixes and stabilization
- [ ] Performance optimizations
- [ ] Additional kernel tuning for specific architectures

### Planned

- [ ] Additional operations (convolution, pooling, etc.)
- [ ] Fused operations support
- [ ] Auto-tuning infrastructure
- [ ] Benchmark suite

## Contributing

When adding new operations:

1. Add kernel sources to `kernels/<operation>/`
2. Implement plan class inheriting from `ICompilablePlan<HipKernelHandle>`
3. Implement plan builder inheriting from `IPlanBuilder<...>`
4. Add applicability checks
5. Register plan builder with engine
6. Add unit tests for plan builder and plan
7. Add integration tests for end-to-end verification

Follow the existing patterns in `BatchnormPlanBuilder` and `BatchnormFwdInferencePlan`.

## License

Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
