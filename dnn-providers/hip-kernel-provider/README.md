# HIP Kernel Provider Plugin

A hipDNN plugin that provides GPU kernel implementations using HIP and HIPRTC for runtime kernel compilation.

:construction: **This project is under active development** :construction:

## Overview

The HIP Kernel Provider is a hipDNN plugin that implements operations using custom HIP kernels compiled at runtime via HIPRTC.

## Architecture

The plugin follows the standard hipDNN plugin architecture:

```
HipKernelEngine
├── PlanBuilder (IPlanBuilder)
│   ├── ApplicabilityChecks
│   └── Plan (IPlan)
│       └── execute() - Compile and launch kernel on GPU
└── [Additional plan builders...]
```

### Key Components

- **Engines** (`src/engines/`): High-level operation orchestration
- **Plans** (`src/engines/plans/`): Kernel-specific execution logic
- **HIP Infrastructure** (`src/hip/`): HIPRTC wrapper classes for compilation and execution
- **Kernels** (`kernels/`): Device-side kernel source code embedded at build time
- **Plugin SDK Integration**: Implements `IPlan`, `IPlanBuilder`, `IEngine` interfaces

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

## Kernel Embedding System

Kernel source files (`.cpp`, `.hpp`, `.h`) under `kernels/` are embedded as C++ string literals at CMake configure time. This allows runtime compilation via HIPRTC while keeping kernel sources as regular C++ files (with syntax highlighting, IDE support, etc.).

The embedding is handled by the `embed_kernel_sources()` CMake function in `kernels/CMakeLists.txt`.

## Contributing

When adding new operations:

1. Add kernel sources to `kernels/<operation>/`
2. Implement plan class inheriting from `IPlan<HipKernelHandle>`
3. Implement plan builder inheriting from `IPlanBuilder<...>`
4. Add applicability checks
5. Register plan builder with engine
6. Add unit tests for plan builder and plan
7. Add integration tests for end-to-end verification

Follow the existing patterns from the codebase.

## License

Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
