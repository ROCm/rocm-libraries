# Building hipDNN

## Table of Contents
- [Prerequisites](#prerequisites)
  - [System Requirements](#system-requirements)
  - [Dependencies](#dependencies)
- [Superbuild vs. Standalone Build](#superbuild-vs-standalone-build)
- [Quick Start Guide](#quick-start-guide)
- [Building the Samples](#building-the-samples)
- [Obtaining ROCm](#obtaining-rocm)
- [Build Configurations](#build-configurations)
  - [Address Sanitizer Build](#address-sanitizer-build)
  - [Disabling JSON Support](#disabling-json-support)
  - [ROCM_PATH, ROCM_CMAKE_PATH, and CMAKE_INSTALL_PREFIX](#rocm_path-rocm_cmake_path-and-cmake_install_prefix)
  - [Clang Tools](#clang-tools)
- [Build Targets](#build-targets)
- [Superbuild](#superbuild)
  - [How It Works](#how-it-works)
  - [Components](#components)
  - [Superbuild Targets](#superbuild-targets)
- [Platform-Specific Instructions](#platform-specific-instructions)
  - [Linux](#linux)
  - [Windows](#windows)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- **GPU**: AMD GPU with ROCm support
- **Operating System**:
  - Linux: Any distribution supported by [TheRock](https://github.com/ROCm/TheRock), such as Ubuntu 24
  - Windows: Windows 11 (limited support, see [Windows section](#windows))

### Dependencies
> [!TIP]
> 💡 Prebuilt binaries and Docker files are available to provide a consistent development environment with all dependencies pre-installed. This is the recommended approach for most users. For more details about these Docker images, see the [Docker README](../dockerfiles/README.md). Dockerfile development environments are not available for Windows. Refer to the [Windows](#windows) section for details on building under Windows.

#### Required Dependencies
| Dependency | Version | Description |
|------------|---------|-------------|
| ROCm | Matching TheRock (latest ROCm version available) | AMD GPU programming stack (see [Obtaining ROCm](#obtaining-rocm)) |
| CMake | 3.25.2+ | Build system generator |
| Ninja | 1.12.1+ | Faster build system (recommended) |
| C++ Compiler | C++17 compatible | AMD Clang, or MSVC on Windows (plugins using device code may require C++20 and AMD Clang)|
| HIP | Matching TheRock | GPU programming interface (included with ROCm/TheRock) |
| clang-format | 18.x | Code formatting tool |
| clang-tidy | 20.x | Static analysis tool |
| LLVM Tools | 20.x | LLVM tools for code_coverage, and ASAN enabled builds |

#### Optional Dependencies
| Dependency | Version | Description |
|------------|---------|-------------|
| Docker | Latest | For containerized build environment |
| Python3 | Latest | For test name validation |

#### Third-Party Libraries
The following libraries are automatically managed by CMake (see [Dependencies.cmake](../cmake/Dependencies.cmake)):
- [FlatBuffers](https://github.com/google/flatbuffers) - Serialization library (used by backend and data_sdk)
- [Google Test](https://github.com/google/googletest) - Unit testing framework
- [spdlog](https://github.com/gabime/spdlog) - Logging library
- [nlohmann_json](https://github.com/nlohmann/json) - JSON serialization (optional, see [Disabling JSON Support](#disabling-json-support))

## Superbuild vs. Standalone Build

hipDNN can be built two ways:

- **Superbuild (recommended).** Builds hipDNN and the providers (`dnn-providers/`) together, so you can build and test everything centrally. Configured from the repository root. See [Superbuild](#superbuild).
- **Standalone.** Builds only `projects/hipdnn`, without the providers. Configured from inside `projects/hipdnn`.

| | Superbuild (recommended) | Standalone |
|---|---|---|
| Builds providers (`dnn-providers/`) | Yes | No |
| Run CMake from | repository root | `projects/hipdnn` |
| Sparse-checkout paths | `projects/hipdnn dnn-providers cmake shared` | `projects/hipdnn` |
| Toolchain | `cmake/toolchains/rocm-clang.cmake` (baked into the presets) | `projects/hipdnn/cmake/ClangToolChain.cmake` (applied automatically) |

## Quick Start Guide

Before you begin:

- Install the required [dependencies](#dependencies).
- If you don't already have ROCm installed, follow [Obtaining ROCm](#obtaining-rocm).
- These steps build **standalone**. For the **superbuild**, see [Superbuild](#superbuild).
- Building on Windows? See [Platform-Specific Instructions](#platform-specific-instructions).
- To manage the multiple Clang tool versions hipDNN needs, see [LLVM_TOOLS_SEARCH_PREFIX](#llvm_tools_search_prefix).
- To build in a prebuilt container instead (recommended), see the [Docker README](../dockerfiles/README.md).

#### 1. Clone the rocm-libraries Repository

A full clone works for either build:
```bash
git clone https://github.com/ROCm/rocm-libraries.git
```

To check out less, use a sparse-checkout. For the **superbuild**:

```bash
git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries
git sparse-checkout init --cone
git sparse-checkout set projects/hipdnn dnn-providers cmake shared
git checkout develop # or the branch you are starting from
```

For the **standalone** build:

```bash
git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries
git sparse-checkout init --cone
git sparse-checkout set projects/hipdnn
git checkout develop # or the branch you are starting from
```

#### 2. Build hipDNN

For the **superbuild**, follow [Superbuild](#superbuild) instead. These steps build **standalone**, using the presets in `projects/hipdnn/CMakePresets.json`.

Two build types are available: `release` (optimized) and `debug` (slower, but suited for a debugger). The examples below use `release`; for a debug build, substitute `debug` for `release` throughout, including the `build/release` -> `build/debug` binary directory. By default the presets find your ROCm installation on your PATH; if ROCm is not on your PATH, add `-DROCM_CMAKE_PATH=<rocm-root>` (see [ROCM_PATH, ROCM_CMAKE_PATH, and CMAKE_INSTALL_PREFIX](#rocm_path-rocm_cmake_path-and-cmake_install_prefix)).

Configure, build, and run the tests:

```bash
cd rocm-libraries/projects/hipdnn

# Configure
cmake --preset release

# Build (some tests may take several minutes to build)
cmake --build build/release

# Run tests (--output-on-failure is optional; it prints logs for failing tests)
ctest --test-dir build/release --output-on-failure
```

See [Build Targets](#build-targets) for other build/test targets.

#### 3. Install hipDNN (optional)

Installing is only needed to consume hipDNN from another project. The standalone presets install into `build/<release|debug>/installdir` by default:

```bash
cmake --install build/release
```

To install elsewhere, set `CMAKE_INSTALL_PREFIX` (see [Build Configurations](#build-configurations)).

Next: to build and run the sample programs, see [Building the Samples](#building-the-samples) below.

## Building the Samples

The samples build against a hipDNN you have already built, and at runtime they load an engine plugin (a provider) to execute graphs. There are two ways to satisfy both.

**Superbuild (recommended).** The `hipdnn-samples` preset builds hipDNN, the providers, and the samples together in one tree, so the samples find hipDNN and the provider plugins are present to load at runtime. Nothing extra to install or configure. See [Superbuild](#superbuild).

**Standalone.** Build hipDNN standalone first ([Build hipDNN](#2-build-hipdnn)); that produces a CMake package config under `projects/hipdnn/build/<release|debug>/lib/cmake`. Then build the samples from the `samples` directory using the matching preset, which already points `CMAKE_PREFIX_PATH` at that build tree:

```bash
cd rocm-libraries/projects/hipdnn/samples
cmake --preset release   # matches the hipDNN build type; use debug for a debug build
cmake --build build/release
```

A standalone hipDNN build does **not** build any provider plugins, so the samples will have no engine to load at runtime. To run them, build a provider and either install its plugin alongside hipDNN or point `HIPDNN_PLUGIN_DIR` at the plugin's location. Because the superbuild handles this for you, it is the simpler choice for building and running samples against an in-tree hipDNN build.

For running and profiling the built samples, see the [samples README](../samples/README.md).

## Obtaining ROCm

> [!IMPORTANT]
> [TheRock RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) is the authoritative, up-to-date source for installing ROCm. The commands below are provided for convenience; consult RELEASES.md if anything here is unclear or out of date.

Linux users are encouraged to use the [Docker container](../dockerfiles/README.md), which provides ROCm and the build dependencies. This section is primarily for Windows, and for Linux without Docker.

The methods below are available for both Linux and Windows. First identify your GPU architecture (e.g. `gfx942`, `gfx1103`) using one of:

- `amdgpu-arch` (from a Clang install)
- `python -m rocm_sdk targets` (after installing the SDK)
- the device-extras table in [RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) (which also shows which architectures have builds available)

### Pointing the build at ROCm

However you obtain ROCm below, the build has to be able to find it. There are multiple ways to do this:

- **Add the ROCm `bin` folder to your `PATH`** so it is auto-detected (the presets and the default toolchains discover ROCm from `PATH`). If ROCm is on `PATH`, nothing else is needed. On Windows, adding it to `PATH` is generally required regardless, so the built executables can load the ROCm DLLs at runtime.
- **Pass the ROCm folder to CMake explicitly** with `-DROCM_PATH=<folder>` at configure time, which uses the compiler and libraries from that folder directly. Use this when ROCm is not on your `PATH`.

See [ROCM_PATH, ROCM_CMAKE_PATH, and CMAKE_INSTALL_PREFIX](#rocm_path-rocm_cmake_path-and-cmake_install_prefix) for details. Each method below notes where to find the ROCm folder it produces.

### Python wheels (recommended)

The ROCm SDK is published as Python wheels on a nightly index. Installing into a Python virtual environment is recommended to keep it isolated, though an existing environment works too.

1. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv rocm-venv
   # Linux:   source rocm-venv/bin/activate
   # Windows: rocm-venv\Scripts\activate
   ```

2. Install the SDK, selecting your GPU architecture with the `device-<arch>` extra (replace `gfx942`):
   ```bash
   pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ "rocm[libraries,devel,device-gfx942]"
   ```
   The `libraries` and `devel` extras provide the ROCm libraries, headers, CMake configuration, and compiler needed to build hipDNN; `device-<arch>` provides the device code for your GPU. This installs the latest nightly; see [RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) to pin a specific version or for other extras.

3. Initialize the SDK. This expands the development tree; re-run it if you later add or change a `device-*` wheel:
   ```bash
   python -m rocm_sdk init
   ```

4. Print the install location, which is the ROCm folder the build needs to find (see [Pointing the build at ROCm](#pointing-the-build-at-rocm) for how):
   ```bash
   python -m rocm_sdk path --root         # the ROCm folder (use as ROCM_PATH, or...)
   python -m rocm_sdk path --bin          # ...add its bin folder to PATH instead
   ```

Run the `python -m rocm_sdk` commands with the same Python you installed the wheels into. `python -m rocm_sdk version` and `python -m rocm_sdk test` can verify the install.

### Tarballs

Tarballs are published as nightly builds (dated versions, like the wheels) at `https://rocm.nightlies.amd.com/tarball-multi-arch/`. Filenames follow `therock-dist-<platform>-<group>-<version>.tar.gz`, where:

- `<platform>` is `linux` or `windows`.
- `<group>` is either `multiarch` (all supported architectures) or a specific GPU family (for example `gfx110X-all`). If in doubt or just getting started, `multiarch` is the recommended safe choice.
- `<version>` is the dated build (for example `7.15.0a20260727`); pick the latest or a specific dated version.

Use the plain distribution tarball; the parallel `...-tests.tar.gz` files hold test executables and are not needed to build hipDNN. Download and extract it:
```bash
tar -xf therock-dist-<platform>-<group>-<version>.tar.gz -C ./rocm
```

The folder you extracted to is the ROCm folder. Point the build at it (see [Pointing the build at ROCm](#pointing-the-build-at-rocm)) either by setting `ROCM_PATH` to its absolute path, or by adding its `bin` subfolder to `PATH`. For example, if you extracted to `./rocm`, that folder's absolute path is the `ROCM_PATH` value and its bin folder is `./rocm/bin` (which should contain `rocminfo`).

### Install script

TheRock's [`install_rocm_from_artifacts.py`](https://github.com/ROCm/TheRock/blob/main/build_tools/install_rocm_from_artifacts.py) downloads and extracts a full ROCm tree. It is the heaviest option (it requires cloning TheRock for the script). Clone TheRock, then:
```bash
# Latest nightly:
python TheRock/build_tools/install_rocm_from_artifacts.py --latest-release --amdgpu-family <family> --output-dir ./rocm

# A specific release:
python TheRock/build_tools/install_rocm_from_artifacts.py --release <version> --amdgpu-family <family> --output-dir ./rocm
```
`--latest-release` is convenient for development; use `--release <version>` to pull a specific build. Run the script with `--help`, or see [RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md), for the full option list.

The `--output-dir` you chose is the ROCm folder. Point the build at it (see [Pointing the build at ROCm](#pointing-the-build-at-rocm)) either by setting `ROCM_PATH` to its absolute path, or by adding its `bin` subfolder to `PATH`. For example, with `--output-dir ./rocm`, that folder's absolute path is the `ROCM_PATH` value and its bin folder is `./rocm/bin` (which should contain `rocminfo`).

## Build Configurations

These configurations are for the **standalone** build. They start from a standalone configure preset (`release` or `debug`) and add options with `-D`. Run from `projects/hipdnn`.

### Release or Debug Build

Use the `release` or `debug` preset directly:
```bash
cmake --preset release   # or: cmake --preset debug
```

### Code Coverage Build
```bash
cmake --preset release -DHIPDNN_ENABLE_COVERAGE=ON
cmake --build build/release --target coverage
# Unit tests are run and coverage reports are generated in build/release/coverage-report/
```

### Address Sanitizer Build

Build with `-DBUILD_ADDRESS_SANITIZER=ON` to compile hipDNN and its tests with AddressSanitizer instrumentation.

> [!IMPORTANT]
> ASAN is a manual process; there is no ASAN coverage in CI yet (planned). The ROCm build requirement differs by platform:
> - **Linux** requires an ASAN-enabled ROCm / TheRock build, so ASAN coverage extends into the shipped ROCm code, not just hipDNN and providers. Building TheRock with ASAN is possible but a large effort, so the Linux ASAN tests are only expected when an ASAN-enabled ROCm build is already available; building ROCm solely for ASAN testing is not expected.
> - **Windows** does not require (or use) an ASAN-enabled ROCm build; ASAN covers only the code compiled during this build, not the installed ROCm libraries.

Configure with ASAN enabled, build, then run the tests. `standard` is the recommended tier to run as the ASAN check:

```bash
cmake --preset release -DBUILD_ADDRESS_SANITIZER=ON
cmake --build build/release
ctest --test-dir build/release -L standard
```

> [!NOTE]
> Any of the `quick`, `standard`, `comprehensive`, and `full` tiers is expected to run cleanly (no ASAN errors) under an ASAN build; `standard` is the default check. See [Testing Strategy: Test Categories](./testing/TESTING_STRATEGY.md#test-categories) for current category behavior.

**Not every GPU architecture supports ASAN** on both Linux and Windows. Tests that cannot run under ASAN on the target are excluded one of two ways: individual tests guard themselves with the `SKIP_IF_ASAN()` GTest macro (so they skip at runtime under an ASAN build), or their ctest registration is disabled when configuring with `-DBUILD_ADDRESS_SANITIZER=ON`. Either way, an ASAN run reports the excluded tests as skipped rather than failing.

**Current status:**

- **Linux** - the ASAN test suite runs cleanly; all tests that are problematic under ASAN have been skipped, so a green run is expected.
- **Windows** - ASAN is supported and builds/runs, but a few issues remain that are expected to be resolved soon, so a fully clean ASAN run is not yet available on Windows.

#### Windows notes

The ASAN configure/build/test commands above apply on Windows as well (run from your Windows build environment; see the [Windows](#windows) setup section). Two Windows specifics:

- ctest sets up ASAN runtime discovery for you: the AddressSanitizer runtime DLL, the build's `bin` directory, and the ROCm `bin` directory are prepended to `PATH` for each test via CTest, so you do not need to add the Clang resource `lib/windows` directory to `PATH` by hand.
- To reduce build time you may add `-DENABLE_CLANG_TIDY=OFF -DENABLE_CLANG_FORMAT=OFF` to the configure step, and set `-DGPU_TARGETS=<target>` for your GPU (auto-detection is not supported on Windows).

### Disabling JSON Support
By default, hipDNN includes JSON serialization support via [nlohmann_json](https://github.com/nlohmann/json). To build without the nlohmann_json dependency:
```bash
cmake --preset release -DHIPDNN_FRONTEND_SKIP_JSON_LIB=ON
```
This disables JSON-based graph serialization and deserialization. Binary serialization remains available.

### ROCM_PATH, ROCM_CMAKE_PATH, and CMAKE_INSTALL_PREFIX

If the ROCm bin folder is included in your system path then the AMD toolchain should be detected automatically. If not, the following CMake variables can be used to assist CMake in the tool discovery.

- **`ROCM_PATH`**: Specifies the root ROCM folder location and the toolchain folders are hard-coded using that path, skipping auto-detection of the toolchain (does not have a default value). **DO NOT SET ROCM_PATH IN YOUR ENVIRONMENT.** Setting ROCM_PATH in the environment will cause the compiler check to fail. Instead, use the -D option to cmake. E.g.: `-DROCM_PATH=/path/to/rocm`.
- **`ROCM_CMAKE_PATH`**: Similar to `ROCM_PATH` but relies on CMake's built-in detection to locate the toolchain. (Default: `/opt/rocm` (Linux) / `C:/dist/therock` (Windows)). Unlike `ROCM_PATH`, it is safe to set in your system environment, so it does not have to be passed on every configure. Will be set automatically if the ROCm bin folder is in your system path.
  - This variable is understood by the standalone `projects/hipdnn/cmake/ClangToolChain.cmake` toolchain. The superbuild's default toolchain (`cmake/toolchains/rocm-clang.cmake`) does not understand it; it uses `ROCM_PATH` only. You can still use `ROCM_CMAKE_PATH` with the superbuild by overriding its toolchain with `-DCMAKE_TOOLCHAIN_FILE=projects/hipdnn/cmake/ClangToolChain.cmake` (see the tip under [Superbuild](#superbuild)), after which setting `ROCM_CMAKE_PATH` in your environment applies to the superbuild as well.

If both `ROCM_CMAKE_PATH` and `ROCM_PATH` are set, `ROCM_CMAKE_PATH` takes precedence: the standalone toolchain uses CMake-based auto-detection from `ROCM_CMAKE_PATH` and does not hard-code the compiler from `ROCM_PATH`. To force the compiler and libraries from a specific folder, set `ROCM_PATH` and leave `ROCM_CMAKE_PATH` unset.

The HIP compiler is required to build some integration tests but is not required for the hipDNN library itself.

Use the following CMake variable to control where the hipDNN library files will be installed when the `install` target is run:
- **`CMAKE_INSTALL_PREFIX`**: Specifies where hipDNN will be installed (defaults to `ROCM_PATH` if `ROCM_PATH` is set, then `ROCM_CMAKE_PATH` if set, otherwise uses the CMake system default).

These variables can all be set independently by adding them to the preset:

```bash
# Default: locate ROCm from your PATH, use the preset's install path.
cmake --preset release

# Install hipDNN to a custom location, find ROCm dependencies in the default location
cmake --preset release -DCMAKE_INSTALL_PREFIX=/custom/install/path

# Both custom
cmake --preset release -DROCM_CMAKE_PATH=/custom/rocm -DCMAKE_INSTALL_PREFIX=/another/path
```

### Clang Tools

Different versions of Clang tools are required. For example, clang-format version 18 and clang-tidy version 20. The hipDNN project tool discovery provides two mechanism to assist with finding the needed version of each tool.

#### Version Suffix

Before searching for the tool using it's standard name, a search will be made for a tool that has the version appended as a suffix. E.g. before looking for `clang-format` a search for a file named `clang-format-18` will be run first, and if that fails then a search will be made for `clang-format`. Similarly, `clang-tidy-20` will be searched-for first, and then `clang-tidy`. This approach can be used if it is possible to modify the Clang toolchain folder(s) on your system to give the tools the corresponding names.

#### LLVM_TOOLS_SEARCH_PREFIX

As an alternative to the above, `LLVM_TOOLS_SEARCH_PREFIX` can be set as a prefix for the folder path where the Clang tools are installed, such that `${LLVM_TOOLS_SEARCH_PREFIX}18/bin` is where the Clang version 18 tools are located, and `${LLVM_TOOLS_SEARCH_PREFIX}20/bin` is where the Clang version 20 tools are located. The CMake configuration step will automatically select the required version for each tool from these folders. For example with `-DLLVM_TOOLS_SEARCH_PREFIX=c:\tools\clang` the the following folders will be searched for Clang tools (depending on the version of each tool that is needed):
* `c:\tools\clang18\bin`
* `c:\tools\clang20\bin`
* `c:\tools\clang\bin`


## Build Targets

The `hipdnn-`prefixed target names below work in both the standalone and superbuild builds. In a standalone build, unprefixed aliases (e.g. `check`, `format`) also exist. In a superbuild, each provider and the integration-tests component add their own prefixed targets; see [Superbuild](#superbuild).

| Target | Description |
|--------|-------------|
| \<no target\> | Build all components |
| `hipdnn-check` / `hipdnn-check-verbose` | Build and run all tests (see [Testing](./testing/TESTING.md)) |
| `hipdnn-<category>-check` / `hipdnn-<category>-check-verbose` | Build and run tests for a category from `test_categories.yaml`: `quick`, `standard`, `comprehensive`, and `full` |
| `hipdnn-format` | Auto-format all C++ source files |
| `hipdnn-check-format` | Check code formatting compliance |
| `hipdnn-tidy` | Run clang-tidy on hipDNN sources |
| `hipdnn-validate_test_names` | Validate that test names conform to naming rules |
| `install` | Install libraries and headers |
| `clean` | Clean build artifacts |
| `generate_hipdnn_flatbuffers_sdk_headers` | Generate C++ headers from schema (`.fbs`) files |

The following coverage targets are standalone-oriented and require `-DHIPDNN_ENABLE_COVERAGE=ON`:

| Target | Description |
|--------|-------------|
| `coverage` | Run the tests and generate test coverage reports |
| `unit-coverage` / `integration-coverage` | Run the unit or integration tests and generate coverage reports |
| `current-coverage` | Generate coverage reports from coverage data already on disk (does not re-run the tests) |

To build a target, for example `hipdnn-check` to build and run all tests:
```bash
cmake --build build/release --target hipdnn-check
```


## Superbuild

The superbuild builds hipDNN together with the providers, samples, and other rocm-libraries components in a single build, making cross-project changes easier. It uses the checked-in **configure presets** in the root `CMakePresets.json`; the hipDNN-relevant presets are listed below. Run from the repository root; binaryDir is `build`. This requires more than a `projects/hipdnn`-only checkout: use a full clone or the wider sparse set (`projects/hipdnn dnn-providers cmake shared`) from [Quick Start step 1](#1-clone-the-rocm-libraries-repository).

Configure, build, and test:

```bash
# From the repository root

# Configure (see the preset table below for other presets)
cmake --preset hipdnn-dev-all -DROCM_LIBS_ENABLE_ROOT_CTEST=ON

# Build
cmake --build build

# Run tests (--output-on-failure is optional; it prints logs for failing tests)
ctest --test-dir build --output-on-failure
```

Root-level `ctest` (i.e. `ctest --test-dir build` from the repository root) only sees the aggregated tests when `ROCM_LIBS_ENABLE_ROOT_CTEST` is `ON`. Set it with `-D` at configure time (as above) or via the environment before a first or fresh configure. Per-component category targets do not require it. For test category targets and other details, see [Testing](./testing/TESTING.md#development-workflow).

> [!NOTE]
> `hipdnn-dev-all` builds every provider, the integration tests, and the samples, so a bare `ctest` runs a large and potentially redundant suite. Scope the run to a category or a subset of tests instead; see [Development Workflow](./testing/TESTING.md#development-workflow) for the available test paths.

To build only hipDNN core from the superbuild (the same components as the [standalone build](#2-build-hipdnn), without the providers), use the `hipdnn` preset:

```bash
# From the repository root
cmake --preset hipdnn -DROCM_LIBS_ENABLE_ROOT_CTEST=ON
```

hipDNN-relevant configure presets:

| Preset | Components enabled |
|--------|--------------------|
| `hipdnn` | hipDNN core only |
| `miopen-provider` | hipDNN core + integration-tests + MIOpen provider |
| `hipblaslt-provider` | hipDNN core + integration-tests + hipBLASLt provider |
| `hip-kernel-provider` | hipDNN core + integration-tests + HIP-kernel provider |
| `hipdnn-providers` | hipDNN core + integration-tests + MIOpen + hipBLASLt providers |
| `hipdnn-providers-all` | `hipdnn-providers` + HIP-kernel provider |
| `hipdnn-samples` | `hipdnn-providers` (hipDNN core + integration-tests + MIOpen + hipBLASLt providers) + samples |
| `hipdnn-dev-all` | everything (all providers + integration-tests + samples + Python bindings) |
| `hipdnn-python` | hipDNN core + Python frontend bindings |

> [!TIP]
> The superbuild presets bake in the superbuild toolchain (`cmake/toolchains/rocm-clang.cmake`), which defaults to `/opt/rocm` on Linux and requires `-DROCM_PATH=<rocm-root>` on Windows. hipDNN developers may prefer to override it with the hipDNN toolchain by adding `-DCMAKE_TOOLCHAIN_FILE=projects/hipdnn/cmake/ClangToolChain.cmake` to the configure step. Benefits:
> - **Auto-detection** of ROCm via `hipconfig` and system PATH, so a non-default ROCm install works without hand-set paths (hint it with `-DROCM_CMAKE_PATH=<rocm-root>` if ROCm is not discoverable).
> - **Consistency** with the standalone `projects/hipdnn` build and between Windows and Linux: the same toolchain and the same override knob (`ROCM_CMAKE_PATH`) apply everywhere, instead of `ROCM_CMAKE_PATH` for standalone and `ROCM_PATH` for the superbuild default.
>
> See [ROCM_PATH, ROCM_CMAKE_PATH, and CMAKE_INSTALL_PREFIX](#rocm_path-rocm_cmake_path-and-cmake_install_prefix) for the full toolchain-discovery details. On Windows, the resource-compiler discovery (and the `-DCMAKE_RC_COMPILER=` override) described under [Setup Environment Variables](#8-setup-environment-variables) applies to the superbuild toolchain as well.

### How It Works

The superbuild uses `add_subdirectory()` to include each component in order. Components added earlier make their CMake targets visible to components added later, removing the need for `find_package()`.

Dependency flow:

```
hipdnn (added first)
├── hipdnn_data_sdk        (INTERFACE target)
├── hipdnn_flatbuffers_sdk (INTERFACE target)
├── hipdnn_plugin_sdk      (INTERFACE target)
├── hipdnn_test_sdk        (INTERFACE target)
├── hipdnn_frontend        (INTERFACE target)
└── hipdnn_backend         (shared library)
         │
         ▼
miopen-provider / hipblaslt-provider / hip-kernel-provider (added after)
└── Use the SDK targets directly (skips find_package)
         │
         ▼
hipdnn-samples (added last)
├── Link hipdnn_frontend
└── Depend on the built provider plugins (needed at test runtime)
```

Each provider's `CMakeLists.txt` uses conditional target checks, so the same `CMakeLists.txt` works for both superbuild and standalone builds:

```cmake
if(NOT TARGET hipdnn_data_sdk)
    find_package(hipdnn_data_sdk CONFIG REQUIRED)
endif()
```

### Components

| Component | Directory | Key Target | External Dependencies |
|-----------|-----------|------------|-----------------------|
| hipdnn | `projects/hipdnn/` | `hipdnn_backend` | ROCm/HIP |
| miopen-provider | `dnn-providers/miopen-provider/` | `miopen_plugin` | MIOpen |
| hipblaslt-provider | `dnn-providers/hipblaslt-provider/` | `hipblaslt_plugin` | hipBLASLt |
| hip-kernel-provider | `dnn-providers/hip-kernel-provider/` | `hip_kernel_provider` | HIP, HIP-RTC |
| hipdnn-samples | `projects/hipdnn/samples/` | `hipdnn_samples` | ROCm/HIP; provider plugins at runtime |
| hipdnn-python | `projects/hipdnn/python/frontend_bindings/` | `hipdnn_frontend_bindings` | Python 3.12+, nanobind |

The configure presets above select the components for you. To choose components directly instead, set `ROCM_LIBS_ENABLE_COMPONENTS`:

```bash
# From the repository root

# hipDNN only (no providers)
cmake --preset hipdnn

# hipDNN + a specific set of providers (manual override)
cmake --preset hipdnn -DROCM_LIBS_ENABLE_COMPONENTS="hipdnn;miopen-provider;hipblaslt-provider"
```

Providers depend on hipDNN, so if you include a provider you must also include `hipdnn`.

### Superbuild Targets

In the superbuild, targets are prefixed with the component's project name to avoid collisions between components. The unprefixed aliases (e.g. `check`, `format`) are created only in standalone builds. The hipDNN targets themselves are the same ones listed in [Build Targets](#build-targets); the tables here add the per-provider and integration-tests targets.

Code-quality targets (prefixed with the project name):

| Target | Description |
|--------|-------------|
| `miopen-provider-check-format` / `miopen-provider-format` | Check / auto-format miopen-provider sources |
| `miopen-provider-tidy` | Run clang-tidy on miopen-provider |
| `hipblaslt-provider-check-format` / `hipblaslt-provider-format` | Check / auto-format hipblaslt-provider sources |
| `hip-kernel-provider-check-format` / `hip-kernel-provider-format` | Check / auto-format hip-kernel-provider sources |
| `hip-kernel-provider-tidy` | Run clang-tidy on hip-kernel-provider |

hipDNN additionally provides a `hipdnn-tidy-cxx` variant (clang-tidy over C++ files only, without HIP arguments); the providers do not have a `-tidy-cxx` variant in the superbuild. hipblaslt-provider does not provide standalone tidy targets (it runs clang-tidy at compile time instead), so it has no `hipblaslt-provider-tidy`.

Gating: format targets require `ENABLE_CLANG_FORMAT=ON` (the default) and a `clang-format` binary; tidy targets require a `run-clang-tidy` binary and are not created on Windows.

Test targets (each `<component>-check` has per-category variants, and each target has a `-verbose` variant):

| Target | Description |
|--------|-------------|
| `miopen-provider-check` | Run all miopen-provider tests |
| `miopen-provider-<category>-check` | Run miopen-provider tests for a category: `quick`, `standard`, `comprehensive`, `full` |
| `miopen-provider-external-integration-check` | Run the provider-agnostic integration suite independently against miopen-provider |
| `hipblaslt-provider-check` | Run all hipblaslt-provider tests |
| `hipblaslt-provider-<category>-check` | Run hipblaslt-provider tests for a category: `quick`, `standard`, `comprehensive`, `full` |
| `hipblaslt-provider-external-integration-check` | Run the provider-agnostic integration suite independently against hipblaslt-provider |
| `hipdnn-integration-tests-check` | Run the provider-agnostic integration test executable |
| `hipdnn-integration-tests-<category>-check` | Run integration tests for a configured category |

hip-kernel-provider does not register `-check` targets; its tests are staged through its install bucket rather than the shared test-target machinery.

Each `<component>-<category>-check` target runs `ctest -L <category>` for that component and works without any extra flag. To run tests through `ctest` directly from the repository root instead, enable `ROCM_LIBS_ENABLE_ROOT_CTEST` (see above).

If you see CMake errors about duplicate target names, ensure you are configuring from the repository root (not from a component subdirectory). The superbuild sets `ROCM_LIBS_SUPERBUILD=ON`, which enables target-name prefixing.

## Platform-Specific Instructions

### Linux
The standard build instructions above work for all supported Linux distributions. Ensure ROCm is properly installed and configured for your distribution.

### Windows

Windows 10 and Windows 11 are supported. Windows 11 is recommended.

> [!WARNING]
> Some GPU functionality and HIP-related tests are not currently supported on Windows.

To do a standalone build of hipDNN, you will need to set up a number of pre-requisites.

> [!NOTE]
> The standalone build of hipDNN requires a subset of the full environment required for building TheRock. Refer to [TheRock Windows Support](https://github.com/ROCm/TheRock/blob/main/docs/development/windows_support.md) for a full Windows 11 build environment setup for TheRock (_but do not perform a build of TheRock_ as this is generally not necessary for building hipDNN standalone).

#### Automated Setup Script (Optional)
An automated PowerShell script is available to perform the steps outlined below. This script is provided as a convenience and may not work in all environments. Review the script before running it to ensure it meets your needs: [windows_build_setup.ps1](../scripts/windows/windows_build_setup.ps1).

#### 1. Install a Package Manager

Though dependencies can be installed _and configured_ manually, using [winget](https://learn.microsoft.com/en-us/windows/package-manager/winget/) (the Windows Package Manager, included with Windows 11 and available via the "App Installer" from the Microsoft Store) will streamline the environment setup. The `winget` client is used in the instructions below.

#### 2. Install Utilities

The following third-party tools are needed for building hipDNN:
   - Git (installed with both git and unix tools available on the windows PATH)
   - Visual Studio 2022 with C++ workload (easy way to get Windows libraries)
   - CMake 3.25.2+
   - Ninja
   - Python 3

Using winget, install any of the missing required dependencies from an **⚠️Administrative Command Prompt (or PowerShell)**:

```cmd
winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.VC.CMake.Project --add Microsoft.VisualStudio.Component.VC.ATL --add Microsoft.VisualStudio.Component.Windows11SDK.22621 --quiet --wait"
```
```cmd
winget install --id Git.Git --custom "/o:PathOption=CmdTools"
```
```cmd
winget install --id Kitware.CMake -v 3.31.0
```
```cmd
winget install --id ninja-build.ninja
```
```cmd
winget install --id Python.Python.3.12
```

#### 3. Enable Windows 10 Long Paths

A detailed description and instructions for enabling long paths on Windows 10+ are available at https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry#enable-long-paths-in-windows-10-version-1607-and-later.

Abbreviated quotation:
>In the Windows API (with some exceptions discussed in the following paragraphs), the maximum length for a path is MAX_PATH, which is defined as 260 characters. A local path is structured in the following order: drive letter, colon, backslash, name components separated by backslashes, and a terminating null character. For example, the maximum path on drive D is `"D:\some 256-character path string<NUL>"` where `"<NUL>"` represents the invisible terminating null character for the current system codepage. (The characters `<` `>` are used here for visual clarity and cannot be part of a valid path string.)
>
>For example, you may hit this limitation if you are cloning a git repo that has long file names into a folder that itself has a long name.
>
>Starting in Windows 10, version 1607, MAX_PATH limitations have been removed from many common Win32 file and directory functions.
>
>The registry value `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem LongPathsEnabled` (Type: `REG_DWORD`) must exist and be set to `1`. The registry value will not be reloaded during the lifetime of the process. In order for all apps on the system to recognize the value, a reboot might be required because some processes may have started before the key was set.
>
> The following Administrative PowerShell command can be used to set this registry value:
>```PowerShell
>New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
>```

#### 4. Enable Windows 10 Symlinks

The instructions below are summarized from web content [here](https://portal.perforce.com/s/article/3472), [here](https://stackoverflow.com/questions/5917249/git-symbolic-links-in-windows/59761201#59761201), and [here](https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-10/security/threat-protection/security-policy-settings/create-symbolic-links).


Verify ability to create symlinks. From a command window, run `mklink`:
```cmd
> echo "test" > mklinktest.txt
> mklink linkedfile.txt mklinktest.txt
symbolic link created for link.txt <<===>> ExistingFile.txt
```
If you do not have the ability to create symlinks you will see:
```cmd
> echo "test" > mklinktest.txt
> mklink linkedfile.txt mklinktest.txt
You do not have sufficient privilege to perform this operation.
```
If you do not have the ability to enable symlinks, the simplest way to enable this is to enable "[Developer Mode](https://www.wikihow.com/Enable-Developer-Mode-in-Windows-10)" in Windows 10/11.

**Windows 10**: Settings --> Update & Security --> For Developers --> Developer Mode --> toggle `On` --> confirm `Yes`.

**Windows 11**: Settings --> System --> For Developers --> Developer Mode --> toggle `On`.

Refer to the links at the beginning of this section for alternative methods to enable symlinks on your system.

You may need to restart your computer for the settings to take effect.

#### 5. Configure Git

With _long-paths and symlinks enabled_ as described in the above sections, enable symlink and long-path support in git:

```cmd
git config --global core.symlinks true
git config --global core.longpaths true
```

> [!IMPORTANT]
> The `core.symlinks` setting is required for AI coding tool configuration files (`.clinerules`, `CLAUDE.md`, `.github/copilot-instructions.md`, `.cursor/rules/*.mdc`) which are symlinks to a central `docs/ai-rules.md`. Without this setting, these files will contain the symlink target path as plain text instead of the actual rules content.

Tip: you can use `git config --show-scope --show-origin core.symlinks` and `git config --show-scope --show-origin core.longpaths` to determine what the current active git configuration is and where that setting is configured.

#### 6. Install Clang Toolchain

Though TheRock toolchain is used to build hipDNN, utilities such as clang-format are currently provided by Clang.

Download and unzip a recent 20.x.x version of the Clang Toolchain: https://github.com/llvm/llvm-project/releases?q=20.

Unzip it to a path with no spaces. E.g. after being unzipped to `C:\dist\clang` the bin folder will be located at `C:\dist\clang\bin`.

#### 7. Install ROCm SDK

See [Obtaining ROCm](#obtaining-rocm) for the full set of install methods (wheels, tarballs, and the install script) and for how the build locates ROCm. On Windows the Python wheels are the recommended method, and its golden path is given below.

First identify your GPU architecture. Run `amdgpu-arch.exe` from the Clang release installed in the previous step:
```cmd
> c:\dist\clang\bin\amdgpu-arch
gfx1103
```

Install the ROCm SDK from the nightly wheel index, selecting your architecture with the `device-<arch>` extra (replace `gfx1103` with the architecture reported above), then expand the development tree:
```cmd
pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ "rocm[libraries,devel,device-gfx1103]"
python -m rocm_sdk init
```
The `libraries` and `devel` extras provide the ROCm libraries, headers, CMake configuration, and compiler needed to build hipDNN; `device-<arch>` provides the device code for your GPU. Re-run `python -m rocm_sdk init` if you later add or change a `device-*` wheel. To pin a specific dated build instead of the latest nightly, see [Python wheels](#python-wheels-recommended) under Obtaining ROCm.

> [!NOTE]
> A Python virtual environment is optional but recommended for isolation. Run the `python -m rocm_sdk` commands below with the same Python you installed the wheels into.

Get the SDK's install location so you can point the build at it:
```cmd
python -m rocm_sdk path --root    # the ROCm folder
python -m rocm_sdk path --bin     # its bin folder
```
On Windows the `bin` folder is typically added to `PATH` so the built executables can load the ROCm DLLs at runtime (covered in the next step). See [Pointing the build at ROCm](#pointing-the-build-at-rocm).

> [!NOTE]
> The [wheel_build_setup.ps1](../scripts/windows/wheel_build_setup.ps1) script automates the wheel install and prints the CMake variables to use. Run `.\scripts\windows\wheel_build_setup.ps1` for the latest nightlies, or with `-SHA <commit-sha>` for a specific staging build.

#### 8. Setup Environment Variables

* Add the ROCm SDK bin folder to the system PATH so that applications can find and load the ROCm DLLs. Use the path reported by `python -m rocm_sdk path --bin`, e.g.:
   ```cmd
   set PATH=<output of python -m rocm_sdk path --bin>;%PATH%
   ```
   It isn't necessary to add the Clang toolchain to your system PATH to perform the build as these can be specified using the [LLVM_TOOLS_SEARCH_PREFIX](#llvm_tools_search_prefix) option to cmake (refer to that section for more details).

   The AMD toolchain should be discovered automatically. If not, refer to the [ROCM_PATH, ROCM_CMAKE_PATH, and CMAKE_INSTALL_PREFIX](#rocm_path-rocm_cmake_path-and-cmake_install_prefix) section for additional ways to locate the toolchain.

* Set the HIP_PLATFORM environment variable:
   ```cmd
   set HIP_PLATFORM=amd
   ```

* **Resource compiler for version metadata.** hipDNN embeds a Windows VERSIONINFO resource in the backend, so the built binaries carry version metadata. Both the standalone and superbuild toolchains locate the resource compiler the same way: they prefer `llvm-rc` from the ROCm LLVM toolchain (found automatically next to `clang++`), and fall back to the Windows SDK's `rc.exe` if it is on your `PATH`. To make `rc.exe` available as the fallback, add the Windows SDK `bin` to `PATH`, e.g.:
   ```cmd
   set PATH=C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64;%PATH%
   ```
   (adjust the SDK version number to match your installed Windows Kit)

   If the toolchain does not locate a resource compiler correctly, point CMake at one explicitly with `-DCMAKE_RC_COMPILER=<path-to-llvm-rc-or-rc>`. This is honored by both toolchains and skips their auto-detection.

   A resource compiler is not strictly required: if none is found, configuration still succeeds and the binaries are built without embedded version metadata.

* If desired, set Ninja as the default generator so that `-g Ninja` doesn't need to be explicitly added to the `cmake` command line:
   ```cmd
   set CMAKE_GENERATOR=Ninja
   ```

Check that the ROCm SDK is installed and locatable:
```cmd
python -m rocm_sdk version
python -m rocm_sdk path --root
```

Use `hipconfig` to check that the ROCm SDK is installed and the PATH is set up correctly. The output from the command, as shown below, will show the detected ROCm path and ROCm clang path (the paths below will be replaced with your ROCm SDK folder, the `python -m rocm_sdk path --root` value). This command requires that the ROCm SDK bin directory is in your system PATH.
```cmd
hipconfig -rocmpath -n --hipclangpath
<ROCm SDK folder>
<ROCm SDK folder>\lib\llvm\bin
```

Example CMake configure command. Because the ROCm SDK bin folder is on your `PATH` (from the previous step), the build discovers ROCm automatically and no ROCm path variable is required. To point CMake at ROCm explicitly instead, add `-DROCM_PATH=<output of python -m rocm_sdk path --root>`.
```cmd
projects\hipdnn\build>set CMAKE_GENERATOR=Ninja
projects\hipdnn\build>cmake -DGPU_TARGETS=gfx1103 -DCMAKE_PROGRAM_PATH=c:/dist/clang/bin ..
```

See the note on setting `GPU_TARGETS` in the following section.


#### 9. Clone Repository and Build hipDNN

From here, follow the instructions in the [Quick Start Guide](#quick-start-guide) section to clone the repository and build hipDNN, **keeping in mind the following notes**:
* Do **NOT** open the "x64 Native Tools Command Prompt for VS 2022" as this will interfere with the ROCm SDK and Clang toolchain.
* Do **NOT** set `ROCM_PATH` in your environment as this will interfere with toolchain detection. If used, specify it using the `-DROCM_PATH=` option to cmake.
* When generating the project, be sure to set GPU_TARGETS to your GPU as auto-detection is not currently supported on Windows, e.g. `cmake -DGPU_TARGETS=gfx1103 ..` (replacing gfx1103 with your GPU)
* When generating the project, CMake will warn about a clang-format or clang-tidy mismatch. That's okay for now but it can be resolved by installing the missing version of the toolchain to a parallel directory and setting the [LLVM_TOOLS_SEARCH_PREFIX](#llvm_tools_search_prefix) variable accordingly.
* Generating the project files may take longer than on Linux, but should complete within a few minutes.
* You may want to limit the number of threads used by Ninja when building hipDNN so that your computer is not bogged-down by the build. You can use the `ninja -j` option to set the number of threads to something smaller than the number of threads available on your CPU.
* To reduce build time, the `-DENABLE_CLANG_TIDY=OFF` option can be used to disable clang-tidy check during development. Similarly the `-DENABLE_CLANG_FORMAT=OFF` option can be used to disable clang-format.

## Troubleshooting

### Common Build Issues

* **Out of memory during build**
   ```bash
   # Reduce parallel jobs
   ninja -j4  # or even -j2 for systems with limited RAM
   ```

* **Docker GPU access issues**
   - Ensure ROCm is installed on the host system
   - Verify GPU is visible: `amd-smi` or `rocminfo`
   - Check user is in `video` and `render` groups:
     ```bash
     sudo usermod -a -G video,render $USER
     # Log out and back in for changes to take effect
     ```
