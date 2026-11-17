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

Descriptor packaging (`HIPDNN_ENABLE_KERNEL_INGESTOR=ON`) additionally requires a supplied `Python3_EXECUTABLE` with `msgpack` and `zstandard`, and a kpack source tree. With `HIPKERNELPROVIDER_ENABLE_ROCKE=ON` it also requires pip in that interpreter and local rocKE wheels. The `hkp_rocke_wheel_python_interp` target installs those wheels only into build-owned storage with no index access or dependency resolution, then uses the supplied interpreter with a subprocess-scoped private import path. It does not install rocKE into the parent Python environment. With `HIPKERNELPROVIDER_ENABLE_ROCKE=OFF` no wheel is needed: the hip producer packs alone, the `rocKE/` family folder is excluded from every root and any other `rocke` UKD is pruned like an arch-pruned one. See [Kernel packing](../../projects/hipdnn/docs/Building.md#kernel-packing-rocm_kpack) for wheel supply modes and prerequisites.

Packaging probes pack descriptors with the real packer for one explicit architecture, independent of the build's `GPU_TARGETS`, and assert the output. Every architecture named by an explicit `arch` on a production KDP or UKD gets an automatic probe in each build that does not target it; wildcard descriptors (no `arch`) are excluded, since every build's own pack covers them. An automatic probe packs, per compile group, one UKD, or for rocKE enough UKDs that each value of each varying `spec` field is packed once (value combinations are not guaranteed). An integration that needs specific UKDs packed adds a named probe, `hkp_add_packaging_probe(ARCH <gfx> NAME <integration> UKDS <ukd-name>...)`, in `descriptor-packaging/probes/probes.cmake`. Probes are superbuild-only: the `HIPKERNELPROVIDER_ENABLE_PACKAGING_PROBES` option is `OFF` by default. They do not cover device behavior, install staging, or standalone-UKD references (probe derive fails configure on those). See the [descriptor-packaging README](descriptor-packaging/README.md#packaging-probes).

### rocKE descriptors (DVC)

With `HIPKERNELPROVIDER_ENABLE_ROCKE=ON`, the shipped rocKE descriptor bundles keep their large files (the KDPs, kernel objects, and heuristic model files) in DVC, not git. Run `dvc pull -r ingestor <pointer>.dvc`, or let TheRock's `fetch_sources` do it, to get them. Without the pull, configure warns and the shipped bundle is not packed; builds and tests still run. Release builds should set `-DHIPKERNELPROVIDER_REQUIRE_DVC_DESCRIPTORS=ON` to make a missing file an error. To skip the shipped bundles, set `-DHIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT` to another tree, such as `descriptor-packaging/examples/descriptors`. See [Large rocKE files live in DVC](src/engines/kernel_ingestor_engine/descriptors/README.md#large-rocke-files-live-in-dvc).

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

## Saving and Loading Execution Plans

You can save a built execution plan to bytes and load it later, in the same process or in another one. A loaded plan runs without the plan build: the provider does not search descriptors, read a kpack archive, or compile a kernel.

### Supported Engines

Only kernel ingestor engines support saving and loading plans, and only when every kernel of the engine is a kpack code object. Every dispatch handler of the engine must also support saving. The kernel ingestor needs a build with `HIPDNN_ENABLE_KERNEL_INGESTOR=ON` (off by default). A supported engine reports the behavior note `SUPPORTS_EXECUTION_PLAN_SERIALIZATION`. To check the engine of a built plan:

```cpp
#include <algorithm>
#include <hipdnn_frontend.hpp>

int64_t engineId = 0;
std::vector<hipdnn_frontend::BehaviorNote> notes;
bool canSave = false;
if(graph.get_execution_plan_engine_id(engineId).is_good()
   && graph.get_behavior_notes_for_engine(engineId, notes).is_good())
{
    canSave = std::find(notes.begin(),
                        notes.end(),
                        hipdnn_frontend::BehaviorNote::SUPPORTS_EXECUTION_PLAN_SERIALIZATION)
              != notes.end();
}
if(canSave)
{
    // Save the plan with to_binary() or to_compiled_plan_binary().
}
```

The other engines of this provider do not report the note.

### Saving the Graph and the Plan Together

`to_binary()` writes the graph. When the engine of the built plan reports the note, it also writes the plan. `from_binary(handle, bytes)` loads both, and the loaded graph executes without a new build.

```cpp
// graph is built (for example with graph.build(handle)) and has executed once.
auto [bytes, error] = graph.to_binary();
if(error.is_bad())
{
    std::cerr << error.get_message() << '\n';
}

hipdnn_frontend::graph::Graph loaded;
auto loadError = loaded.from_binary(handle, bytes);
// loaded.execute(handle, variantPack, workspace);
```

When the engine does not report the note, `to_binary()` writes the graph only and logs a warning. A graph loaded from those bytes has no plan, so build its plans again before you execute it.

### Saving the Plan Only

`to_compiled_plan_binary()` writes the plan only. `from_compiled_plan_binary(handle, bytes)` loads it into a new graph. The loaded graph has no operation graph; execute it with a variant pack keyed by tensor UID.

```cpp
auto [planBytes, error] = graph.to_compiled_plan_binary();
if(error.is_bad())
{
    std::cerr << error.get_message() << '\n';
}

hipdnn_frontend::graph::Graph restored;
auto loadError = restored.from_compiled_plan_binary(handle, planBytes);

std::unordered_map<int64_t, void*> variantPack = {{xUid, xDevicePtr}, {yUid, yDevicePtr}};
auto executeError = restored.execute(handle, variantPack, workspace);
```

A plan-only save of a plan from any other engine of this provider fails. With `HIPDNN_ENABLE_KERNEL_INGESTOR=ON` the backend status is `HIPDNN_STATUS_PLUGIN_ERROR`; without the ingestor the provider has no save support and the status is `HIPDNN_STATUS_NOT_SUPPORTED`. In both cases the frontend returns `ErrorCode::HIPDNN_BACKEND_ERROR`, with the cause in the message.

### Rules

- **Benchmarking plans save only after they have chosen a kernel.** A plan built with `global.benchmarking=1` or with `HIPDNN_FORCE_BENCHMARKING=1` usually chooses its kernel at its first `execute()`. (When a stored benchmark result already covers every candidate kernel, the build makes a plain plan that saves at once.) Saving a plan that has not chosen a kernel fails with "the plan has not chosen a kernel yet; execute it once, or build it with benchmarking off, then save". Execute the plan once and then save it, or build it with benchmarking off. For an engine that reports the note, this also makes `to_binary()` fail; it does not fall back to the graph only.
- **A loaded plan cannot be saved again.** Both `to_compiled_plan_binary()` of a loaded plan and `to_binary()` of a graph loaded with `from_binary(handle, bytes)` fail with "this plan was loaded from a saved payload; re-saving is not supported; keep the original bytes". Keep the bytes you loaded, and reuse them.
- **A loaded plan runs the code it was saved with.** Upgrading the provider does not change the kernel inside a saved plan, even when the upgrade fixes that kernel. To use the new kernel, build the plan again from its graph and save it.
- **The kpack archive must not change between the plan build and the save.** A save gets the kernel's code object from the kpack archive that the plan build used, and checks it against the kernel's SHA-256. The save can read the archive through the handle that the plan build opened. If the archive changes after the plan build, the save may fail. If the save cannot read the archive, or the bytes do not match, the save fails.
- **The loading machine must have the same engine.** hipDNN loads a plan only through a plugin that serves the saved engine ID, so the descriptor set that defines the engine must be installed and loaded. The GPU must be able to run the stored code object, and this provider must still recognize the stored dispatch name, kernel arguments and launch values. The format version must be one this provider reads. The provider version does not need to match.
- **Failures name their cause.** When the provider refuses a save or a load, the backend status is `HIPDNN_STATUS_PLUGIN_ERROR`. When no loaded plugin serves the saved engine ID at all, hipDNN refuses the load before the provider runs, with `HIPDNN_STATUS_INTERNAL_ERROR` and "Invalid engine ID". In every case the frontend returns `ErrorCode::HIPDNN_BACKEND_ERROR`. The error text holds hipDNN's own text first and then the provider's message. The provider's message starts with "ingestor plan data is damaged: " or "cannot save this plan: its kernel source does not match its descriptor: " when the data is damaged or the kernel no longer matches its descriptor, and with "ingestor plan is valid but cannot be saved or loaded here: " or "cannot save this plan here: " when the plan is valid but cannot be used here. Search the error text for these prefixes; do not expect it to start with them. A load checks everything, including that the GPU driver accepts the code object, so a bad plan fails at load and not at its first execute.

### Security

A saved plan contains executable GPU code, and executing a loaded plan runs that code. Load only plans from a source you trust. The SHA-256 checks detect accidental damage, not a crafted file. A load does not repeat the path checks the provider applies to its descriptor tree, and the code object goes to the GPU driver as it is.

The full format, the load checks and their order are in [RFC 0009](../../projects/hipdnn/docs/rfcs/0009_CompiledPlanSerialization.md#hip-kernel-provider-ingestor-implementation).

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
