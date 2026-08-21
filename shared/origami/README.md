# Origami: Analytical Solution Selection for GEMM Kernels

**Origami** provides a fast, analytical, deterministic methodology to select optimal GEMM configuration (such as tile size) for out-of-the-box GEMM performance. Origami estimates performance by sweeping over candidate configs (tile sizes and mapping) to select the optimal configuration based on **compute** and **memory latencies**.

## Documentation

- [Quick Start Guide](#quick-start-guide)
  - [Prerequisites](#prerequisites)
  - [Install](#install)
- [API Example](#api-example)
  - [Python API](#python-api)
  - [C++ API](#c-api)
- [Supported GPUs](#supported-gpus)
- [Build and Install](#build-and-install)
  - [Python](#build-and-install-origami-python)
  - [C++](#build-and-install-origami-c)
  - [CMake Options](#cmake-options)
  - [Origami Tests](#origami-tests)
- [Debug Logging](#debug-logging)
  - [Text Log](#text-log)
  - [CSV Log](#csv-log)
  - [Environment Variables](#environment-variables)
- [Contribute](#contribute)
- [How to Cite](#how-to-cite)

## Quick Start Guide

### Prerequisites

**ROCm/HIP**: This package requires a ROCm install. The native Origami library (`liborigami` and its CMake package) is first-class in ROCm, so a ROCm install provides it. See the [ROCm Quick Start Guide](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) for installation instructions.

By default the Python extension links the installed shared `liborigami` (`find_package(origami)`) rather than building it from source. The build therefore needs a prefix that CMake can discover (via `CMAKE_PREFIX_PATH`) containing:

```text
lib/cmake/origami/origami-config.cmake
lib/liborigami.so
include/origami/...
```

A generic "ROCm is installed" is not sufficient if that prefix is not on `CMAKE_PREFIX_PATH`. To build the native library from this source tree instead, pass `-DORIGAMI_BUILD_FROM_SOURCE=ON` (via `CMAKE_ARGS`). That is a development path, not a self-contained one: the wheel never bundles `liborigami`, so the extension still resolves `liborigami.so.1` through the loader at import time and you must install the library you built and put it on the loader path yourself.

### Install

Run the Python packaging workflows from `shared/origami`, where the project's
`pyproject.toml` lives. Use `python -m build` to create a distributable wheel,
or `pip install -e .` to create a local development installation.

#### Build a wheel

The wheel workflow selects the `python` CMake preset. It builds the extension
against an installed shared `liborigami` and does not bundle that library in the
wheel:

```bash
cd shared/origami
python -m pip install build

CMAKE_PREFIX_PATH=/path/to/rocm \
  python -m build --wheel --outdir dist
```

Install the built artifact:

```bash
python -m pip install dist/*.whl
```

The wheel links the installed shared `liborigami`; that runtime must be visible to the dynamic loader at import time (via RPATH, `LD_LIBRARY_PATH`, or `ldconfig`).

#### Create an editable development install

The editable workflow selects the `dev` CMake preset, which builds `liborigami`
from this checkout and links the extension to that build-tree library:

```bash
cd shared/origami
pip install -e .
```

Python source edits are visible through the editable installation immediately.
The native library and extension are built once during installation, so rerun
`pip install -e .` after changing C++ or binding sources. See
[Iterative development](#iterative-development) for the CMake-only edit-build-test
loop.

Alternatively, install directly from the repository in one step (builds a wheel under the hood):

```bash
pip install git+https://github.com/ROCm/rocm-libraries.git#subdirectory=shared/origami
```

To build the native library from source in the same step instead:

```bash
CMAKE_ARGS="-DORIGAMI_BUILD_FROM_SOURCE=ON" \
  pip install git+https://github.com/ROCm/rocm-libraries.git#subdirectory=shared/origami
```

## API Example

### Python API

The compiled extension is a private submodule, `origami._pyorigami`; every public
name is re-exported from the `origami` package, so import from `origami` directly.
`origami.origami` still resolves, but it is deprecated, emits a `DeprecationWarning`,
and will be removed in the next minor release. One consequence of the rename is
visible in reprs, error messages, and pickle paths: types now report
`origami._pyorigami` as their module instead of `origami.origami`.

```python
import origami

# Get hardware information for device 0
hardware = origami.get_hardware_for_device(0)

# Create a problem description
problem = origami.problem_t()
problem.size = origami.dim3_t(2048, 2048, 2048)  # M, N, K dimensions
problem.batch = 1
problem.a_transpose = origami.transpose_t.T
problem.b_transpose = origami.transpose_t.N
problem.a_dtype = origami.data_type_t.Half
problem.b_dtype = origami.data_type_t.Half
problem.c_dtype = origami.data_type_t.Half
problem.d_dtype = origami.data_type_t.Half
problem.mi_dtype = origami.data_type_t.Half
problem.a_mx_block_size = 0
problem.b_mx_block_size = 0

# Create candidate configurations
configs = []
config = origami.config_t()
config.mt = origami.dim3_t(256, 256, 64)  # Macro tile dimensions
config.mi = origami.dim3_t(16, 16, 32)    # Matrix instruction dimensions
config.occupancy = 4
configs.append(config)

# Select best configuration
best_result = origami.select_config(problem, hardware, configs)
print(f"Best latency: {best_result.latency}")
print(f"Best config: MT=({best_result.config.mt.m}, {best_result.config.mt.n}, {best_result.config.mt.k})")
```

### C++ API

```cpp
#include "origami/origami.hpp"
#include "origami/types.hpp"
#include <vector>
#include <iostream>

int main() {
    // Get hardware information for device 0
    auto hardware = origami::hardware_t::get_hardware_for_device(0);
    
    // Create a problem description
    origami::problem_t problem;
    problem.size.m = 2048;  // M dimension
    problem.size.n = 2048;  // N dimension
    problem.size.k = 2048;  // K dimension
    problem.batch = 1;
    problem.a_transpose = origami::transpose_t::T;
    problem.b_transpose = origami::transpose_t::N;
    problem.a_dtype = origami::data_type_t::Half;
    problem.b_dtype = origami::data_type_t::Half;
    problem.c_dtype = origami::data_type_t::Half;
    problem.d_dtype = origami::data_type_t::Half;
    problem.mi_dtype = origami::data_type_t::Half;
    problem.a_mx_block_size = 0;
    problem.b_mx_block_size = 0;
    
    // Create candidate configurations
    std::vector<origami::config_t> configs;
    origami::config_t config;
    config.mt.m = 256;  // Macro tile M
    config.mt.n = 256;  // Macro tile N
    config.mt.k = 64;   // Macro tile K
    config.mi.m = 16;   // Matrix instruction M
    config.mi.n = 16;   // Matrix instruction N
    config.mi.k = 32;   // Matrix instruction K
    config.occupancy = 4;
    configs.push_back(config);
    
    // Select best configuration
    auto best_result = origami::select_config(problem, hardware, configs);
    std::cout << "Best latency: " << best_result.latency << std::endl;
    std::cout << "Best config: MT=(" 
              << best_result.config.mt.m << ", " 
              << best_result.config.mt.n << ", " 
              << best_result.config.mt.k << ")" << std::endl;
    
    // Alternative: Simple selection using just M, N, K
    auto best_result_simple = origami::select_config_mnk(2048, 2048, 2048, hardware, configs);
    
    // Rank all configurations by performance
    auto ranked_configs = origami::rank_configs(problem, hardware, configs);
    std::cout << "Top 5 configs:" << std::endl;
    for (size_t i = 0; i < std::min(ranked_configs.size(), size_t(5)); ++i) {
        const auto& result = ranked_configs[i];
        std::cout << "  Rank " << (i+1) << ": latency=" << result.latency 
                  << ", MT=(" << result.config.mt.m << ", " 
                  << result.config.mt.n << ", " << result.config.mt.k << ")" << std::endl;
    }
    
    // Compute performance in GFLOPS
    double gflops = origami::compute_perf_gflops(hardware, problem, best_result.latency);
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    
    return 0;
}
```

## Supported GPUs

| LLVM Target | GPUs | Functional | Optimized |
|-------------|------|------------|-----------|
| gfx942 | MI325X, MI300X, MI300A | ✔️ | ✔️ |
| gfx950 | MI355X, MI350X | ✔️ | ✔️ |
| gfx1100 | Radeon RX 7900 XTX/XT/GRE, Radeon PRO W7900 (Dual Slot), Radeon PRO W7800 (48GB) | ✔️ | |
| gfx1150 | Radeon 890M/880M iGPU | ✔️ | |
| gfx1151 | Radeon 8060S/8050S/8040S iGPU | ✔️ | |
| gfx1152 | Radeon 860M/840M iGPU | ✔️ | |
| gfx1153 | TBA | ✔️ | |
| gfx1200 | Radeon RX 9060 (XT) | ✔️ | |
| gfx1201 | Radeon RX 9070 (XT/GRE), Radeon AI PRO R9700 (D/S) | ✔️ | |
| gfx1250 | TBA | ✔️ | |

For more information on GPU hardware specifications, check out [ROCm documentation](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html).

## Build and Install

### Build and Install Origami (Python)

Origami provides Python bindings that allow you to use Origami's functionality directly from Python.

#### Installation

The build system uses `pyproject.toml` with scikit-build-core, which integrates with CMake for building the Python bindings. Use `python -m build --wheel` for a distributable artifact or `pip install -e .` for local development, as described in [Install](#install). The repository installation workflows below are alternatives to building from an existing checkout.

Install directly from the rocm-libraries repository (this could take some time due to the size of the rocm-libraries repo):

```bash
pip install git+https://github.com/ROCm/rocm-libraries.git#subdirectory=shared/origami
```

To efficiently install directly from the rocm-libraries repository use do the following:

```bash
TEMP_DIR=$(mktemp -d)
git clone --no-checkout --filter=blob:none --sparse https://github.com/ROCm/rocm-libraries.git $TEMP_DIR
git -C $TEMP_DIR sparse-checkout set shared/origami
git -C $TEMP_DIR checkout develop
pip install $TEMP_DIR/shared/origami -v
rm -rf $TEMP_DIR
```

#### Iterative development

Both development workflows below build `liborigami` from this source tree, driven from `shared/origami`. Neither needs an installed Origami or installs the native library system-wide. The editable workflow does install the Python package. Pick a workflow based on whether you want `import origami` to work without setting `PYTHONPATH`.

CMake, no pip. The build tree holds a ready-to-import package:

```bash
cd shared/origami
cmake --preset dev
cmake --build --preset dev
PYTHONPATH=build/dev/python python -m pytest python/tests -v
```

Editing a C++ source and re-running `cmake --build --preset dev` relinks `liborigami` and the extension. The extension picks up the build-tree `liborigami` through its build RPATH, so there is nothing to install between the edit and the test run.

The `dev` preset builds `RelWithDebInfo` with `-Wall` on the extension, so a stack trace names lines and warnings are visible. `dev:debug` is the same loop at `-O0 -g`.

Editable install:

```bash
cd shared/origami
pip install -e .
python -m pytest python/tests -v
```

The install builds `liborigami` and the extension once. Python source edits are visible immediately, but a later C++ or binding edit does not reach the installed package on its own. Rerun `pip install -e .` to rebuild it, or use the preset loop above, which needs no install at all.

`scikit-build-core` can rebuild on every `import origami` (`editable.rebuild`), and this project deliberately leaves it off. Rebuild-on-import runs outside the ephemeral build environment, so it only works when every install also passes `--no-build-isolation` and the build requirements are present in the environment itself. An install that omits the flag records pip's temporary overlay in `CMakeCache.txt`; pip deletes that overlay afterwards, and the next import fails in `cmake --build` on a path that no longer exists. The preset loop gives the same edit-test cycle without that coupling.

Both suites in one tree. The `dev:tests` preset is `dev` with `ORIGAMI_BUILD_TESTING=ON`, which adds the Catch2 C++ binary and registers it alongside the Python tests so a single `ctest` covers both:

```bash
cd shared/origami
cmake --preset dev:tests
cmake --build --preset dev:tests
ctest --preset dev:tests
```

The C++ tests need Catch2 3. If CMake cannot find one, the build fetches it from GitHub, so the first `dev:tests` configure needs network access; `dev` does not.

The `ORIGAMI_BUILD_FROM_SOURCE=ON` define is what makes either loop build the native library rather than look for an installed one. Wheel builds leave it off and link the Origami that ROCm provides.

#### Plain CMake build

`shared/origami/CMakeLists.txt` is the only entry point: it either adds `origami-cpp/` or calls `find_package(origami)`, then adds `python/`. The presets above drive it; the invocation below is the same build spelled out. You'll need to manually install the Python dependencies listed in `shared/origami/python/requirements.txt`:

```bash
pip install -r shared/origami/python/requirements.txt
```

Build Python bindings using CMake from the `shared/origami` directory:

```bash
cd shared/origami

# configure with python bindings and tests enabled 
cmake -S . -B build/ \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_INSTALL_PREFIX=/opt/rocm \
  -DORIGAMI_ENABLE_PYTHON=ON \
  -DORIGAMI_BUILD_TESTING=ON

# build 
cmake --build build/ --parallel

# run tests
cd build/
ctest --output-on-failure
```

### Build and Install Origami (C++)

Build the C++ library from the `shared/origami` directory:

```bash
cd shared/origami

# configure
cmake -S . -B build/ \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_INSTALL_PREFIX=/opt/rocm

# build
cmake --build build/ --parallel
```

After configuring and building, run the following command to install:

```bash
# install
cmake --install build/
```

### CMake Options

| Option | Description | Default |
|--------|-------------|---------|
| `ORIGAMI_BUILD_SHARED_LIBS` | Build `liborigami` shared instead of static | `ON` |
| `ORIGAMI_BUILD_FROM_SOURCE` | Add `origami-cpp/` and build liborigami from this tree instead of calling `find_package(origami)` | `ON` (the wheel sets it `OFF`) |
| `ORIGAMI_ENABLE_PYTHON` | Enable Python bindings | `OFF` |
| `ORIGAMI_BUILD_TESTING` | Build the Catch2 C++ suite, and the Python suite when `ORIGAMI_ENABLE_PYTHON=ON` | `OFF` |
| `ORIGAMI_ENABLE_FETCH` | Auto-fetch dependencies with FetchContent | `ON` |


## Origami Tests

### Build and Run All Tests

Build with both C++ and Python tests enabled:

```bash
cd shared/origami

cmake -S . -B build/ \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DORIGAMI_BUILD_TESTING=ON \
  -DORIGAMI_ENABLE_PYTHON=ON

cmake --build build/ --parallel

cd build/
ctest --output-on-failure
```

> [!NOTE]
> Python tests are automatically added when `ORIGAMI_BUILD_TESTING=ON` and `ORIGAMI_ENABLE_PYTHON=ON`.

### Running Specific Tests

Run only C++ tests:

```bash
./build/origami-cpp/tests/origami-tests
```

Run a specific C++ test by name:

```bash
./build/origami-cpp/tests/origami-tests "Origami: select_config_mnk unit test"
```

Run only Python tests (from `shared/origami`), against a build tree produced by the `dev` preset:

```bash
PYTHONPATH=build/dev/python python -m pytest python/tests -v
```

Run Python tests excluding slow tests:

```bash
python -m pytest python/tests -m "not slow"
```

Run selector tests (requires torch):

```bash
python -m pytest python/tests/test_selector.py -v
```

## Debug Logging

Origami includes built-in debug logging that exposes internal latency model values (cache hit rates, memory/compute latencies, tile parameters, etc.). Debug output requires the `ANALYTICAL_GEMM_DEBUG=1` environment variable to activate the debug code paths.

### Text Log

Write human-readable debug output to a file by setting `ORIGAMI_LOG_FILE`:

```bash
export ANALYTICAL_GEMM_DEBUG=1
export ORIGAMI_LOG_FILE=/tmp/origami.log
```

Each GEMM evaluation produces a block of key-value lines in the log file, e.g.:

```
[DEBUG] gemm.cpp:99 - ======== Origami Debug Info ========
[DEBUG] gemm.cpp:100 - M: 2048
[DEBUG] gemm.cpp:101 - N: 2048
...
[DEBUG] gemm.cpp:116 - total_latency: 45678.9
[DEBUG] gemm.cpp:117 - =================================
```

### CSV Log

Write structured CSV output (one row per GEMM evaluation) by using a `.csv` file extension:

```bash
export ANALYTICAL_GEMM_DEBUG=1
export ORIGAMI_LOG_FILE=/tmp/origami_debug.csv
```

The log format is inferred from the file extension: `.csv` selects CSV mode, anything else selects human-readable text mode.

The CSV file contains columns for every value logged with `OLOG_DEBUG`, such as `M`, `N`, `K`, `L_mem`, `L_compute`, `H_mem_l2_A`, `total_latency`, etc. This is useful for bulk analysis of the latency model across many GEMM problems.

Accumulated rows are flushed to disk in two situations:

1. **At process exit** — the logger destructor writes any remaining rows.
2. **On explicit flush or reconfiguration** — calling `Logger::flush()` writes buffered rows. Calling `Logger::update_from_env()` also flushes before applying the new configuration. Subsequent rows are appended incrementally.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `ANALYTICAL_GEMM_DEBUG` | Set to `1` to enable debug code paths in the latency model |
| `ORIGAMI_LOG_FILE` | Path for log output; `.csv` extension selects CSV format, any other extension selects text |

## Contribute

If you want to submit an issue, you can do so on
[GitHub](https://github.com/ROCm/rocm-libraries/issues). To contribute to our repository, you can create a GitHub pull request.

## How to Cite

If you use Origami or reference it in your research, please cite our work:

```bibtex
@misc{Swann:2025:TTB,
  title={{tritonBLAS}: Triton-based Analytical Approach for GEMM Kernel Parameter Selection}, 
  author={Ryan Swann and Muhammad Osama and Xiaohu Guo and Bryant Nelson and Lixun Zhang and Alex Brown and Yen Ong and Ali Yazdani and Sean Siddens and Ganesh Dasika and Alex Underwood},
  year={2025},
  eprint={2512.04226},
  archivePrefix={arXiv},
  primaryClass={cs.DC},
  url={https://arxiv.org/abs/2512.04226},
}
```
