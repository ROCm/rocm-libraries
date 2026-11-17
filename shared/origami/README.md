# Origami : Analytical Solution Selection for GEMM Kernels

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
- [Neural Network (NN) Backend](#neural-network-nn-backend)
  - [Supported Backends](#supported-backends)
  - [Building with NN On or Off](#building-with-nn-on-or-off)
  - [Model Weights](#model-weights)
  - [Runtime Configuration](#runtime-configuration)
  - [hipBLASLt Integration](#hipblaslt-integration)
  - [NN Tests](#nn-tests)
- [Debug Logging](#debug-logging)
  - [Text Log](#text-log)
  - [CSV Log](#csv-log)
  - [Environment Variables](#environment-variables)
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
#Get hardware information for device 0
hardware = origami.get_hardware_for_device(0)

#Create a problem description
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

#Create candidate configurations
configs = []
config = origami.config_t()
config.mt = origami.dim3_t(256, 256, 64)  # Macro tile dimensions
config.mi = origami.dim3_t(16, 16, 32)    # Matrix instruction dimensions
config.occupancy = 4
configs.append(config)

#Select best configuration
best_result = origami.select_config(problem, hardware, configs)
print(f"Best latency: {best_result.latency}")
print(f"Best config: MT=({best_result.config.mt.m}, {best_result.config.mt.n}, {best_result.config.mt.k})")
```

### C++ API

```cpp
#include <iostream>
#include <vector>
#include "origami/origami.hpp"
#include "origami/types.hpp"

int main() {
  // Get hardware information for device 0
  auto hardware = origami::hardware_t::get_hardware_for_device(0);

  // Create a problem description
  origami::problem_t problem;
  problem.size.m          = 2048;  // M dimension
  problem.size.n          = 2048;  // N dimension
  problem.size.k          = 2048;  // K dimension
  problem.batch           = 1;
  problem.a_transpose     = origami::transpose_t::T;
  problem.b_transpose     = origami::transpose_t::N;
  problem.a_dtype         = origami::data_type_t::Half;
  problem.b_dtype         = origami::data_type_t::Half;
  problem.c_dtype         = origami::data_type_t::Half;
  problem.d_dtype         = origami::data_type_t::Half;
  problem.mi_dtype        = origami::data_type_t::Half;
  problem.a_mx_block_size = 0;
  problem.b_mx_block_size = 0;

  // Create candidate configurations
  std::vector<origami::config_t> configs;
  origami::config_t config;
  config.mt.m      = 256;  // Macro tile M
  config.mt.n      = 256;  // Macro tile N
  config.mt.k      = 64;   // Macro tile K
  config.mi.m      = 16;   // Matrix instruction M
  config.mi.n      = 16;   // Matrix instruction N
  config.mi.k      = 32;   // Matrix instruction K
  config.occupancy = 4;
  configs.push_back(config);

  // Select best configuration
  auto best_result = origami::select_config(problem, hardware, configs);
  std::cout << "Best latency: " << best_result.latency << std::endl;
  std::cout << "Best config: MT=(" << best_result.config.mt.m << ", " << best_result.config.mt.n
            << ", " << best_result.config.mt.k << ")" << std::endl;

  // Alternative: Simple selection using just M, N, K
  auto best_result_simple = origami::select_config_mnk(2048, 2048, 2048, hardware, configs);

  // Rank all configurations by performance
  auto ranked_configs = origami::rank_configs(problem, hardware, configs);
  std::cout << "Top 5 configs:" << std::endl;
  for (size_t i = 0; i < std::min(ranked_configs.size(), size_t(5)); ++i) {
    const auto& result = ranked_configs[i];
    std::cout << "  Rank " << (i + 1) << ": latency=" << result.latency << ", MT=("
              << result.config.mt.m << ", " << result.config.mt.n << ", " << result.config.mt.k
              << ")" << std::endl;
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

#configure with python bindings and tests enabled 
cmake -S . -B build/ \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_INSTALL_PREFIX=/opt/rocm \
  -DORIGAMI_ENABLE_PYTHON=ON \
  -DORIGAMI_BUILD_TESTING=ON

#build 
cmake --build build/ --parallel

#run tests
cd build/
ctest --output-on-failure
```

### Build and Install Origami (C++)

Build the C++ library from the `shared/origami` directory:

```bash
cd shared/origami

#configure
cmake -S . -B build/ \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_INSTALL_PREFIX=/opt/rocm

#build
cmake --build build/ --parallel
```

After configuring and building, run the following command to install:

```bash
#install
cmake --install build/
```

### CMake Options

| Option | Description | Default |
|--------|-------------|---------|
| `ORIGAMI_BUILD_SHARED_LIBS` | Build shared libraries | `ON` |
| `ORIGAMI_ENABLE_PYTHON` | Enable Python bindings | `OFF` |
| `ORIGAMI_BUILD_TESTING` | Build the C++ tests (and the Python binding tests when `ORIGAMI_ENABLE_PYTHON=ON`) | `OFF` |
| `ORIGAMI_ENABLE_FETCH` | Auto-fetch dependencies with FetchContent | `ON` |
| `ORIGAMI_ENABLE_NN` | Build NN inference stack (`origami::nn`, TWREC loader, tilewright ranking in `rank_configs`). | `OFF` |

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

## Neural Network (NN) Backend

Origami can rank GEMM kernel candidates with an optional **neural-network backend** in addition to the default analytical roofline model. NN inference is wired through `origami::rank_configs` via `rank_options_t`: callers choose **analytical**, **NN-only**, or **NN with analytical fallback**.

When `ORIGAMI_ENABLE_NN=OFF`, all NN code is compiled out and `rank_configs` always uses the analytical path (`inference_mode_t::nn` throws at runtime).

### Supported Backends

| Backend | `nn_backend_t` | Status | Weights format |
|---------|----------------|--------|----------------|
| **tilewright** | `tilewright` | Implemented (TWREC loader + in-tree inference) | TWREC YAML manifest (`.tilewright.yaml`) + int4 sidecar (`.tilewright.wts.yaml`) |
| **embedding_similarity** | `embedding_similarity` | API stub only (not yet loadable) | — |

**tilewright** is a split-tree, per-cell two-tower MLP recommender. Models are keyed by Tensile logic-library stem (dtype + transpose) and discovered from `origami_nn_index` colocated with the kernel library.

**gfx950 weights** (downloaded into `data/nn/tilewright/gfx950/`; they are not stored in git, see [Model Weights](#model-weights)):

| Dtype family | Tensile type | Transpose layouts |
|------------|--------------|-------------------|
| bf16 BBS (`bf16_r`) | `BB_BB` | TN, NN, TT, NT (all four) |
| fp8 F8BS | `F8F8_BF8` | NN only |
| fp32 with xf32 compute (`f32_r`, `--compute_type xf32_r`) | `SS_SS` (`S_MX_B`) | TN, NN, TT, NT (all four) |

If no model exists for a given logic stem, `load_models_for_logic` returns `invalid_handle` (`-1`) and `rank_configs` falls back to analytical when `ORIGAMI_INFERENCE_MODE=nn_fallback`. The miss is cached for the life of the process, so place the weights before the first lookup.

A model is also declined at rank time when its architecture does not match the hardware being ranked for (for example a gfx950 model on gfx942). `nn_fallback` then uses the analytical path, and `nn` throws.

### Building with NN On or Off

**Default (analytical-only)** — `ORIGAMI_ENABLE_NN` is `OFF` unless explicitly enabled:

```bash
cd shared/origami

cmake -S . -B build/ \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DORIGAMI_BUILD_TESTING=ON

cmake --build build/ --parallel
```

**NN enabled** — opt in to the TWREC loader, tilewright inference, and filter stack:

```bash
cmake -S . -B build/ \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DORIGAMI_ENABLE_NN=ON \
  -DORIGAMI_BUILD_TESTING=ON

cmake --build build/ --parallel
```

Download tilewright weights into `shared/origami/data/nn/tilewright/gfx950/`. The download locations are currently only accessible from an AMD-internal S3 bucket. A hipBLASLt build with `ORIGAMI_ENABLE_NN=ON` copies that directory next to each Tensile library, and `PredictionLibrary` loads the copy from the logic-file directory. Callers that do not pass a directory, such as an application calling `load_models_for_logic` directly, can set `ORIGAMI_NN_WEIGHTS_DIR` to the download directory instead (see [Weights directory](#weights-directory)).

**hipBLASLt** uses the same `ORIGAMI_ENABLE_NN` option (default **OFF**). Enable NN when configuring hipBLASLt:

```bash
cmake -S projects/hipblaslt -B build/ -DORIGAMI_ENABLE_NN=ON
```

After a hipBLASLt build, YAML weights are copied next to each arch's Tensile library under `Tensile/library/<arch>/`.

Check whether NN was built:

```bash
grep ORIGAMI_ENABLE_NN build/origami-config.cmake
# or inspect compile definitions on the origami target
```

### Model Weights

Weights live under:

```
shared/origami/data/nn/tilewright/<arch>/
  origami_nn_index          # maps logic_stem → backend → manifest
  *.tilewright.yaml         # TWREC manifest (routing + cell metadata)
  *.tilewright.wts.yaml     # int4_b64 weight sidecar
```

Index format (tab-separated; lines starting with `#` are comments, and the backend is `tilewright` or `embedding_similarity`):

```
<logic_stem>  tilewright  <manifest.yaml>
```

Load programmatically:

```cpp
#include "origami/nn/nn.hpp"

auto models = origami::nn::load_models_for_logic(logic_stem, library_dir);
// models.tilewright >= 0 on success; -1 if stem not in index
```

To load a single manifest directly (for example in a test), pass its path to `origami::nn::load_model`.

A manifest is rejected, with a message on stderr, when its `feature_names_hash` does not match the hash compiled into this build. Regenerate or re-download the weights if you see `TWREC manifest rejected: feature_names_hash ...` after updating Origami. A rejected manifest is remembered, so a manifest shared by several stems is parsed only once.

`origami::nn::unload_model(handle)` is safe to call while another thread is still ranking with that model: the model stays alive until the ranking call returns. Unloading drops the cached lookup for that handle, so the next lookup loads the model again. Cached misses are not cleared.

`origami::nn::nn_inference_requested(mode)` returns whether `rank_configs` would use ML for a call that requests `mode`, after the `ORIGAMI_INFERENCE_MODE` override is applied. Callers such as hipBLASLt use it to defer loading weights until a call can use them.


### Runtime Configuration

NN routing is controlled by `rank_options_t` fields and environment variables. Explicit `rank_options_t` values take precedence over environment defaults.

#### Inference mode

| `ORIGAMI_INFERENCE_MODE` | Behavior |
|--------------------------|----------|
| *(unset)* or `analytical` | Analytical roofline model (default) |
| `nn` | ML ranking only; throws if no model resolves or inference fails |
| `nn_fallback` | ML when a model is available; otherwise analytical |

#### Backend selection

| `ORIGAMI_NN_BACKEND` | Behavior |
|----------------------|----------|
| *(unset)* or `auto` | Prefer tilewright handle from `library_models`, then session default |
| `tilewright` | Force tilewright backend |
| `embedding_similarity` | Force embedding-similarity backend (not yet implemented) |

#### Weights directory

| Source | Behavior |
|--------|----------|
| `hint_dir` passed to `load_models_for_logic` / `load_model_by_index` | Used first. hipBLASLt passes the directory of the logic file |
| `ORIGAMI_NN_WEIGHTS_DIR` | Used when no hint directory is given |

Only one directory is searched. When a hint directory is given, `ORIGAMI_NN_WEIGHTS_DIR` is not consulted, even if the hint directory has no entry for the stem; hipBLASLt always passes one, so it ignores the variable. The directory must contain `origami_nn_index` and the manifests it lists. Both sources name a directory, so the old `ORIGAMI_NN_WEIGHTS` override, which replaced the path of every manifest, is no longer read.

#### Diagnostics

| Variable | Description |
|----------|-------------|
| `ORIGAMI_NN_DIAG` | Print model-load diagnostics when a TWREC manifest is loaded (`arch`, feature dims, cell/split counts). Also prints when a model is declined because its architecture does not match the hardware, and, in hipBLASLt, one `PredictionLibrary stem=... dir=... tilewright_handle=...` line per logic library when its model is loaded |
| `ORIGAMI_TW_LOG` | Log top-1 tilewright pick per `rank_configs` call (`[TW_PICK]` line on stderr) |
| `ORIGAMI_TW_CELL` | Debug: force TWREC cell index (overrides routing); also via `rank_options.nn.force_cell` |

### hipBLASLt Integration

The default inference mode is analytical, so hipBLASLt reaches the tilewright path only when `ORIGAMI_INFERENCE_MODE` is `nn` or `nn_fallback`. `PredictionLibrary` does not parse any weights while a logic file loads. It records the logic stem and directory, and loads the model on the first ranking call whose resolved inference mode is `nn` or `nn_fallback`. A missing manifest is cached as a failed lookup for the life of the process, so place the weights before the first lookup.

Example: run hipblaslt-bench with tilewright NN fallback on gfx950 bf16 TN layout:

```bash
export TENSILE_PREDICTION_LIB=1
export ORIGAMI_INFERENCE_MODE=nn_fallback
export ORIGAMI_NN_BACKEND=tilewright
export HIPBLASLT_TENSILE_LIBPATH=/path/to/build/Tensile/library/gfx950

hipblaslt-bench -m 3405 -n 4000 -k 5000 -r bf16_r \
  --transA T --transB N --bias_vector
```

Use `ORIGAMI_INFERENCE_MODE=nn` to require ML ranking (no silent analytical fallback). Set `ORIGAMI_NN_DIAG=1` to verify model load for the active logic stem.

| Variable | Description |
|----------|-------------|
| `TENSILE_PREDICTION_LIB=1` | Use `PredictionMatching` rows in `ExactLogicLibrary`, so tuned Equality and Range rows do not pre-empt ranking |

The NN path is reached through the `PredictionMatching` rows, so a shape that matches an Equality or Range row is not ranked by NN unless this variable is set. Shapes that match no such row reach NN without it. `TENSILE_PREDICTION_LIB=1` makes hipBLASLt skip every `EqualityMatching` and `RangeMatching` logic row. Hand-written kernels that only appear in those rows, such as the gfx950 skinny `n == 1` TN kernel, are then never candidates. Set it only for the process under test. Exporting it globally makes tests that expect those kernels, such as `HandwrittenAlgoReuse_smoke`, fail.

### NN Tests

Build with `ORIGAMI_ENABLE_NN=ON` and testing enabled, then run NN-specific Catch2 tags:

```bash
cd shared/origami

cmake -S . -B build/ \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DORIGAMI_ENABLE_NN=ON \
  -DORIGAMI_BUILD_TESTING=ON

cmake --build build/ --parallel
./build/tests/origami-tests "[nn]"
```

When `ORIGAMI_ENABLE_NN=OFF` (the default), `[nn]` filter tests are compiled out.

Tests tagged `[nn][tilewright]` need the downloaded weights. The tests read them from `shared/origami/data/nn/tilewright/gfx950/` (set at configure time) and call `SKIP()` when they are missing, so a checkout without weights reports those tests as skipped rather than failed. `SKIP()` needs Catch2 3.3 or newer. Tests that do not load weights, such as the `rank_options_t` defaults and the `nn_inference_requested` check, always run.

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
@misc{
Swann:2025:TTB,
  title={{tritonBLAS}: Triton-based Analytical Approach for GEMM Kernel Parameter Selection}, 
  author={Ryan Swann and Muhammad Osama and Xiaohu Guo and Bryant Nelson and Lixun Zhang and Alex Brown and Yen Ong and Ali Yazdani and Sean Siddens and Ganesh Dasika and Alex Underwood},
  year={2025},
  eprint={2512.04226},
  archivePrefix={arXiv},
  primaryClass={cs.DC},
  url={https://arxiv.org/abs/2512.04226},
}
```
