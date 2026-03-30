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

This plugin is built as a standalone project outside of the main hipDNN build. It depends on the hipDNN SDK packages (`hipdnn_data_sdk` and `hipdnn_plugin_sdk`), which must be available on the system before building.

The SDK packages can come from either:

- **Building hipDNN from source** (in the `projects/hipdnn` directory of this repository): build hipDNN and run `ninja install` to install the SDK packages. Note that the `CMAKE_INSTALL_PREFIX` may need to be set when configuring the hipDNN CMake project if ROCm was not installed to the default `/opt/rocm` on your system.
- **A ROCm or TheRock installation** that includes hipDNN: the SDK packages are installed as part of the install.

Either approach works as long as the installed SDK version is compatible with the plugin version being built.

> **Avoiding header conflicts:** If you have hipDNN installed system-wide (e.g., from ROCm or TheRock) and also build hipDNN from source in the repository, the two sets of headers may conflict. To avoid this, use `CMAKE_PREFIX_PATH` to point at exactly the installation you intend to use:
>
> ```bash
> cmake -GNinja -DCMAKE_PREFIX_PATH=<path-to-hipdnn-install> -DCMAKE_CXX_COMPILER=<path-to-amdclang>/clang++ ..
> ```

### Steps

1. Navigate to the `dnn-providers/hip-kernel-provider` directory.
2. Make a build directory using `mkdir build && cd build`.
3. Configure the build using `cmake -GNinja -DCMAKE_CXX_COMPILER=<path to amdclang>/clang++ ..`.
4. Finally, run `ninja` to build the plugin.

### Build Requirements

- hipDNN SDK packages installed (`hipdnn_data_sdk`, `hipdnn_plugin_sdk`)
- ROCm with HIP and HIPRTC
- CMake 3.25+
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
