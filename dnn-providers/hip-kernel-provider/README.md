# HIP Kernel Provider Plugin

A hipDNN plugin that provides GPU kernel implementations using HIP and HIPRTC for runtime kernel compilation.

:construction: **This project is under active development** :construction:

## Overview

The HIP Kernel Provider is a hipDNN plugin that implements operations using custom HIP kernels compiled at runtime via HIPRTC.

## Architecture

The plugin follows the standard hipDNN plugin architecture:

```
HipKernelEngine
+-- PlanBuilder (IPlanBuilder)
|   +-- ApplicabilityChecks
|   +-- Plan (IPlan)
|       +-- execute() - Compile and launch kernel on GPU
+-- [Additional plan builders...]
```

### Key Components

- **Engines** (`src/engines/`): High-level operation orchestration
    - **HIP MLops engine** (`src/engines/hip_mlops_engine/`): Kernel-specific execution logic. Enabled via the compile flag `ENABLE_HIP_MLOPS_ENGINE` (on by default)
    - **ASM SDPA engine** (`src/engines/asm_sdpa_engine/`): Assembly kernels to do scaled dot-product attention (SDPA). Enabled via `ENABLE_ASM_SDPA_ENGINE` (on by default)
    - **HIP Flash2 engine** (`src/engines/hip_flash2_engine/`): Flash-Attention 2 V7 SDPA kernel using precompiled `.co` files for gfx942/gfx950 (FP16). Enabled via `ENABLE_HIP_FLASH2_ENGINE` (off by default)
- **HIP Infrastructure** (`src/hip/`): HIPRTC wrapper classes for compilation and execution
- **Kernels** (`kernels/`): Device-side kernel source code embedded at build time
- **Plugin SDK Integration**: Implements `IPlan`, `IPlanBuilder`, `IEngine` interfaces

## Building

This plugin is built as a standalone project outside of the main hipDNN build. It depends on the hipDNN SDK packages (`hipdnn_data_sdk` and `hipdnn_plugin_sdk`), which must be available on the system before building.

The SDK packages can come from either:

- **Building hipDNN from source** (in the `projects/hipdnn` directory of this repository): build hipDNN and run `ninja install` to install the SDK packages. Note that the `CMAKE_INSTALL_PREFIX` may need to be set when configuring the hipDNN CMake project if ROCm was not installed to the default `/opt/rocm` on your system.
- **A ROCm or TheRock installation** that includes hipDNN: the SDK packages are installed as part of the install.

Either approach works as long as the installed SDK version is compatible with the plugin version being built.

The [hipDNN developer image](../../projects/hipdnn/dockerfiles/README.md) supplies the third-party libraries. The steps below need no dependency download, install, or fetch flag in that image; ROCm and the compatible hipDNN SDKs must still be available.

> **Avoiding header conflicts:** If you have hipDNN installed system-wide (e.g., from ROCm or TheRock) and also build hipDNN from source in the repository, the two sets of headers may conflict. To avoid this, use `CMAKE_PREFIX_PATH` to point at exactly the installation you intend to use:
>
> ```bash
> cmake -GNinja -DCMAKE_PREFIX_PATH=<path-to-hipdnn-install> -DCMAKE_CXX_COMPILER=<path-to-amdclang>/clang++ ..
> ```

### Steps

1. Navigate to the `dnn-providers/hip-kernel-provider` directory.
2. Make a build directory using `mkdir build && cd build`.
3. Configure the build using `cmake -GNinja -DCMAKE_CXX_COMPILER=<path to amdclang>/clang++ ..`.
    - If you would like to enable/disable a specific engine, add the argument `-DENABLE_<engine>=0` (example: `-DENABLE_ASM_SDPA_ENGINE=0`)
4. Finally, run `ninja` to build the plugin.

Outside the image, this provider resolves third-party libraries like every other hipDNN component; see [Third-Party Libraries](../../projects/hipdnn/docs/Building.md#third-party-libraries) for the installed-package and fetch options. Configure from this provider's build directory:

```bash
cmake -GNinja -DCMAKE_CXX_COMPILER=/path/to/amdclang/clang++ \
    -DCMAKE_PREFIX_PATH="/path/to/hipdnn-install;/path/to/rocm;/path/to/dependencies" ..
```

Beyond the shared third-party set, the prefixes must provide HIP/HIPRTC, `hipdnn_data_sdk`, `hipdnn_flatbuffers_sdk`, `hipdnn_plugin_sdk`, and their transitive dependencies. The fetch opt-in supplies none of those.

### Build Requirements

- hipDNN SDK packages installed (`hipdnn_data_sdk`, `hipdnn_flatbuffers_sdk`, `hipdnn_plugin_sdk`)
- ROCm with HIP and HIPRTC
- CMake 3.25+
- Ninja build system
- C++17 compatible compiler (amdclang++ recommended)
- GoogleTest including GoogleMock for tests (see [Third-Party Libraries](../../projects/hipdnn/docs/Building.md#third-party-libraries))

Descriptor packaging (`HIPDNN_ENABLE_KERNEL_INGESTOR=ON`) additionally requires a supplied `Python3_EXECUTABLE` with pip, `msgpack`, and `zstandard`, a kpack source tree, and local rocKE wheels. The `hkp_rocke_wheel_python_interp` target installs those wheels only into build-owned storage with no index access or dependency resolution, then uses the supplied interpreter with a subprocess-scoped private import path. It does not install rocKE into the parent Python environment. See [Kernel packing](../../projects/hipdnn/docs/Building.md#kernel-packing-rocm_kpack) for wheel supply modes and prerequisites.

Packaging probes pack descriptors with the real packer for one explicit architecture, independent of the build's `GPU_TARGETS`, and assert the output. Every architecture named by an explicit `arch` on a production KDP or UKD gets an automatic probe in each build that does not target it; wildcard descriptors (no `arch`) are excluded, since every build's own pack covers them. An automatic probe packs, per compile group, one UKD, or for rocKE enough UKDs that each value of each varying `spec` field is packed once (value combinations are not guaranteed). An integration that needs specific UKDs packed adds a named probe, `hkp_add_packaging_probe(ARCH <gfx> NAME <integration> UKDS <ukd-name>...)`, in `descriptor-packaging/probes/probes.cmake`. Probes are superbuild-only: the `HIPKERNELPROVIDER_ENABLE_PACKAGING_PROBES` option is `OFF` by default. They do not cover device behavior, install staging, or standalone-UKD references (probe derive fails configure on those). See the [descriptor-packaging README](descriptor-packaging/README.md#packaging-probes).

### Testing

After building, run the test suites:

```bash
# Unit tests (CPU + basic GPU tests)
./bin/hip_kernel_plugin_tests

# Integration tests (full GPU pipeline tests)
./bin/hip_kernel_plugin_integration_tests
```

## Kernel Embedding System

Kernel source files (`.cpp`, `.hpp`, `.h`) under `engines/` are embedded as C++ string literals at CMake configure time. *Use the CMake function `add_kernels_for_embedding` in any engine you define that implements HIP kernels.* This allows runtime compilation via HIPRTC while keeping kernel sources as regular C++ files (with syntax highlighting, IDE support, etc.).

The embedding is handled by the `embed_kernel_sources()` CMake function in `src/cmake/KernelEmbedding.cmake`.

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

## Project policies

This plugin is part of the hipDNN project. Shared project documentation and
policies are maintained in hipDNN:

- [hipDNN Overview](../../projects/hipdnn/README.md)
- [Contributing Guidelines](../../projects/hipdnn/CONTRIBUTING.md)
- [Security Policy](../../projects/hipdnn/SECURITY.md)

## License

Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
