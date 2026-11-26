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
- [Code Layout](#code-layout)
  - [Selecting Behaviour at Compile Time](#selecting-behaviour-at-compile-time)
  - [Adding Things](#adding-things)
- [Contribute](#contribute)
- [How to Cite](#how-to-cite)

## Quick Start Guide

### Prerequisites

**ROCm/HIP**: This package requires ROCm/HIP to be installed on your system. ROCm cannot be installed via pip and must be installed separately. See the [ROCm Quick Start Guide](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) for installation instructions. Ensure `CMAKE_PREFIX_PATH` includes your ROCm install (default: `/opt/rocm`).

### Install

```bash
pip install git+https://github.com/ROCm/rocm-libraries.git#subdirectory=shared/origami/python
```

## API Example

### Python API

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
best_result = origami.select_config(problem, hardware, configs, origami.model_t.gemm)
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

Install directly from the rocm-libraries repository (this could take some time due to the size of the rocm-libraries repo):

```bash
pip install git+https://github.com/ROCm/rocm-libraries.git#subdirectory=shared/origami/python
```

To efficiently install directly from the rocm-libraries repository use do the following:

```bash
TEMP_DIR=$(mktemp -d)
git clone --no-checkout --filter=blob:none --sparse https://github.com/ROCm/rocm-libraries.git $TEMP_DIR
git -C $TEMP_DIR sparse-checkout set shared/origami
git -C $TEMP_DIR checkout develop
pip install $TEMP_DIR/shared/origami/python -v
rm -rf $TEMP_DIR
```

If you have already cloned the repository:

```bash
cd shared/origami/python
pip install -e .
```

The build system uses `pyproject.toml` with scikit-build-core, which integrates with CMake for building the Python bindings.

#### CMake Build (Alternative)

When building with CMake, you'll need to manually install the Python dependencies listed in `shared/origami/python/requirements.txt`:

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
| `ORIGAMI_BUILD_SHARED_LIBS` | Build shared libraries | `ON` |
| `ORIGAMI_ENABLE_PYTHON` | Enable Python bindings | `OFF` |
| `ORIGAMI_BUILD_TESTING` | Enable Python binding tests | `OFF` |
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
./build/tests/origami-tests
```

Run a specific C++ test by name:

```bash
./build/tests/origami-tests "Origami: select_config_mnk unit test"
```

Run only Python tests (from `shared/origami/python`):

```bash
pip install -e .
python -m pytest tests/ -v
```

Run Python tests excluding slow tests:

```bash
python -m pytest tests/ -m "not slow"
```

Run selector tests (requires torch):

```bash
python -m pytest tests/test_selector.py -v
```

## Debug Logging

Origami includes built-in debug logging that exposes internal latency model values (cache hit rates, memory/compute latencies, tile parameters, etc.). Debug output requires the `ORIGAMI_DEBUG=1` environment variable to activate the debug code paths.

### Text Log

Write human-readable debug output to a file by setting `ORIGAMI_LOG_FILE`:

```bash
export ORIGAMI_DEBUG=1
export ORIGAMI_LOG_FILE=/tmp/origami.log
```

Each GEMM evaluation produces a block of key-value lines in the log file, e.g.:

```
[DEBUG] tile.cpp:319 - ======== Origami Debug Info ========
[DEBUG] tile.cpp:322 - ProblemSize (MxNxBxK): 2048x2048x1x2048
[DEBUG] tile.cpp:323 - transpose: TN
...
[DEBUG] latency.cpp:386 - total_latency: 155498
[DEBUG] latency.cpp:387 - =================================
```

A config that ranking does not score is reported with the rule that rejected it:

```
[DEBUG] ranking.cpp:203 -   Config MT=(64,64,64) MI=(1,1,64) REJECTED: Dot2 is kept only for M <= 2
```

### CSV Log

Write structured CSV output (one row per GEMM evaluation) by using a `.csv` file extension:

```bash
export ORIGAMI_DEBUG=1
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
| `ORIGAMI_DEBUG` | Set to `1` to enable debug code paths in the latency model |
| `ORIGAMI_LOG_FILE` | Path for log output; `.csv` extension selects CSV format, any other extension selects text |
| `ORIGAMI_HEURISTICS` | Set to `0` to disable heuristics; they are on otherwise |
| `ORIGAMI_HEURISTICS_VARIANCE` | Relative latency band within which ranking breaks ties; `0` (the default) turns it off |
| `ORIGAMI_GEMM_PICK` | Macro tile `MxNxK` (e.g. `128x128x64`) that GEMM selection is forced to |

Every variable uses the `ORIGAMI_` prefix. The former `ANALYTICAL_GEMM_DEBUG`, `ANALYTICAL_GEMM_HEURISTICS`, `ANALYTICAL_GEMM_HEURISTICS_VARIANCE` and `ANALYTICAL_GEMM_PICK` names are still read when the new name is not set.

## Code Layout

The headers directly under `include/origami/` are the public API (`origami.hpp`, `gemm.hpp`, `attention.hpp`, `hardware.hpp`, `types.hpp`, ...). Each one only includes the pieces below. `src/origami/` has the same directories for the parts that have sources (`backends/` and `method/` are header-only, and the simulator sources are in `src/simulator/`).

| Directory | Contents |
|---|---|
| `core/` | Types every part uses: data types, dimensions, enums, `problem_t`, `config_t`, runtime options, math, logging. |
| `hardware/` | `hardware_t` and the tables behind it: architectures and families, chip variants, per-architecture traits, measured constants, matrix-instruction latencies. |
| `backends/` | Backend tags and backend-specific kernel parameters (`tensile_params_t`). The tags are a placeholder: no model is specialized per backend yet. |
| `method/` | Method tags (analytical, simulation) and the `method::evaluator` customization point. |
| `cost_models/` | One directory per cost model. `gemm/` has one file per stage of the model: `cache` (workgroup mapping and hit rates), `memory`, `compute`, `epilogue`, `tile` (context and the tile model), `streamk` (grid and launch), `heuristics`, and `latency` (admission, cost models, reduction kernel, total), plus `constants.hpp`. A family's own code is in the matching `*_cdna4.cpp` / `*_cdna5.cpp` file. `attention/` holds the attention model. |
| `selection/` | Ranking, tie-breaking and top-k, generic over cost models; the workgroup-mapping and StaggerU selectors, which are GEMM-specific. |
| `simulator/` | The Formocast simulator, which is the simulation method of GEMM. |

### Selecting Behaviour at Compile Time

Four things vary independently, and each has a tag type and one `dispatch` function that turns the run-time value into the tag. Shared code is written once as a template; whatever differs is a specialization.

| What varies | Run-time value | Tags | Where it is used |
|---|---|---|---|
| Architecture family | `hardware_t::family()` | `arch::cdna2` ... `arch::rdna4` | `cost_models/gemm/model.hpp`: `model<Arch>` |
| Cost model | `model_t` | `cost_model::gemm`, `cost_model::attention` | `cost_model_traits<Model>`, `rank<Model>()` |
| Method | `config_t::prediction_mode` | `method::analytical`, `method::simulation` | `method::evaluator<Model, Method>` |
| Backend | `config_t::target` | `backend::tensilelite`, `backend::triton`, ... | `backend_traits<B>` (not used by the library yet) |

For GEMM, `base_model` lists every decision, stage and heuristic a family may do differently, and a family that differs redeclares only those members:

| Family model | What it changes | Files |
|---|---|---|
| `cdna2_model` (gfx90a) | No conversion charge for emulated data types. | `model.hpp` |
| `cdna4_model` (gfx950) | Hand-optimized kernel table, TF32 emulation rule, StreamK hybrid-mode rule. | `heuristics_cdna4.cpp`, `streamk_cdna4.cpp` |
| `cdna5_model` (gfx1250) | Memory, compute, epilogue and tile stages, launch, workgroup mapping, fixed costs. | `*_cdna5.cpp` |

The tile model, the launch, the heuristics lookup and the total latency are written once against `model<Arch>` and do not branch on the family or the architecture.

Constants follow the same split. `heuristic_defaults_t` holds what every family shares. A family's own constants are a struct with one value set per chip variant, read through the model's `constants(hardware)`:

| Struct | Value sets | Lookup |
|---|---|---|
| `cdna4_constants_t` | `CDNA4_CONSTANTS_ID75A0`, `CDNA4_CONSTANTS_ID75A8` | `cdna4_constants(variant)` |
| `cdna5_constants_t` (measured) | `CDNA5_CONSTANTS_8XCD` (256 CUs), `CDNA5_CONSTANTS_6XCD` (96 CUs, 192 unpartitioned) | `cdna5_constants(variant)` |
| `cdna5_fit_t` (fitted) | `CDNA5_FIT_8XCD`, `CDNA5_FIT_6XCD` | `cdna5_fit(variant)` |

The second set of each struct starts as a copy of the first; assign the values that differ inside its initializer in `cost_models/gemm/constants.hpp`.

The W4A16 kernels of gfx1150 and gfx1151 (int4 A dequantized to a bf16/fp16 MI type) are not a family model. `is_w4a16()` selects their rules inside the shared code: the dequantize cost in `compute`, the decode model and the Stream-K partial tiles in `latency`, and the launch grid in `streamk`. Their constants are `w4a16_params_t`.

A chip variant (`variant_t`, for example `gfx950_id75a8` or `gfx1250_6xcd`) is a part of one architecture with its own measured constants. `resolve_variant()` is the only place that derives it from the PCI chip id or the CU count; `hardware_t::variant()` returns it.

### Adding Things

- **An architecture:** one enumerator (last, before `Count`) and one row each in `hardware/architecture.hpp`, `hardware/traits.hpp` and `hardware/constants.hpp`, and its entry in `python/src/origami/bindings.cpp`.
- **A chip variant:** one enumerator, its name and its rule in `resolve_variant()` in `hardware/variant.hpp`, one row in `hardware/constants.hpp`, and its entry in the Python bindings.
- **Values for a chip variant:** assign them in the variant's value set in `cost_models/gemm/constants.hpp`; nothing else changes.
- **Family-specific model behaviour:** derive a model struct from `base_model`, redeclare the members that differ, and specialize `model<Arch>` for the family's tag.
- **A cost model:** a `model_t` enumerator, a tag and its case in `cost_model::dispatch` in `cost_models/cost_model.hpp`, a `cost_model_traits` specialization (admission rules and latency) and a `method::evaluator` specialization for its analytical model.
- **A method:** a `prediction_modes_t` enumerator, a tag and its case in `method::dispatch` in `method/method.hpp`, and a `method::evaluator` specialization for each cost model that supports it.

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
