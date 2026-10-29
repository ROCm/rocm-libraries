#Origami : Analytical Solution Selection for GEMM Kernels

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
| `ORIGAMI_BUILD_TESTING` | Enable Python binding tests | `OFF` |
| `ORIGAMI_ENABLE_FETCH` | Auto-fetch dependencies with FetchContent | `ON` |
| `ORIGAMI_ENABLE_NN` | Build NN inference stack (`origami::nn`, TWREC loader, tilewright ranking in `rank_configs`). The boost backend is built either way. | `OFF` |

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

Origami can rank GEMM kernel candidates with an optional **neural-network backend** in addition to the default analytical roofline model. NN inference is wired through `origami::rank_configs` via `rank_options_t`: callers choose **analytical**, **NN-only**, **NN with analytical fallback**, or **analytical with a re-rank** of its top candidates.

When `ORIGAMI_ENABLE_NN=OFF`, the tilewright code is compiled out and `rank_configs` never ranks with it (`inference_mode_t::nn` throws at runtime). The boost re-rank backend has no NN or YAML dependency and is built either way.

### Supported Backends

| Backend | `nn_backend_t` | Status | Weights format |
|---------|----------------|--------|----------------|
| **tilewright** | `tilewright` | Implemented (TWREC loader + in-tree inference) | TWREC YAML manifest (`.tilewright.yaml`) + int4 sidecar (`.tilewright.wts.yaml`) |
| **embedding_similarity** | `embedding_similarity` | API stub only (not yet loadable) | — |
| **boost** | `boost` | Implemented (ORIBOOST loader + in-tree evaluation); re-rank only (`analytical_rerank`) | ORIBOOST binary (`.orr`) |

**tilewright** is a split-tree, per-cell two-tower MLP recommender. Models are keyed by Tensile logic-library stem (dtype + transpose) and discovered from `origami_nn_index` colocated with the kernel library.

**gfx950 weights shipped in-tree** (`data/nn/tilewright/gfx950/`):

| Dtype family | Tensile type | Transpose layouts |
|------------|--------------|-------------------|
| bf16 BBS (`bf16_r`) | `BB_BB` | TN, NN, TT, NT (all four) |
| fp8 F8BS | `F8F8_BF8` | NN only |
| mxfp8 S_MX_B | `SS_SS` | TN, NN, TT, NT (all four) |

If no model exists for a given logic stem, `load_models_for_logic` returns `invalid_handle` (`-1`) and `rank_configs` falls back to analytical when `ORIGAMI_INFERENCE_MODE=nn_fallback`.

**boost** (Origami Boost) is a gradient-boosted tree ensemble that predicts each candidate's `log(measured / origami_latency)`. With `inference_mode_t::analytical_rerank`, `rank_configs` ranks analytically and hands the results to the caller's `rerank_source_t::prepare` (its own adjustment of the order, if any); the backend then takes the first `top_n` (10) results the caller accepts, orders them by `origami_latency * exp(yhat)` (the prepared order on ties) and moves them to the front; every other result keeps its prepared order. Its 161 features (`features/gemm_boost`, schema c6-4) are a name mapping of Origami's own intermediates (`compute_gemm_intermediates` in `origami/gemm_intermediates.hpp`, also in the Python bindings) plus kernel fields the caller supplies per candidate (`nn/boost/types.hpp`: `rerank_source_t`, `kernel_fields_t`, including the kernel's Stream-K launch). One model serves a dtype class on one arch, shared by the class's logic stems: bf16 (A/B BFloat16), tf32 (A/B/C/D f32 with xf32 math) or fp8 (F8BS: A/B Float8 into Float8 math, C/D BFloat16, no MX blocks).

A boost model is checked when it loads (`nn/boost/load_validate.hpp`) and refused unless it matches: arch, dtype class, feature-schema hash and Origami source hash (sha256 of the analytical sources at configure time; the `nn/` tree is excluded), and the kernel library by manifest diff (`nn/boost/library_stamp.hpp`): kernels may be removed, and added when their gated structural values (DTL, MFMA, LDS mode, one LDS buffer, MT_K) were seen in training. Every call it declines (problem of another dtype class or outside the training M/N/K/batch ranges, fewer than two candidates, a feature it cannot compute) leaves the prepared order unchanged and reports why to the caller (`rerank_source_t::outcome`). A CU budget (`problem_t::num_cus`) is not a range: the features see it. A model's RNGE section holds exactly M, N, K and batch, each once; a file with any other range (for example `num_cus`) is refused at load.

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

With `ORIGAMI_ENABLE_NN=ON`, Origami can load TWREC YAML manifests/sidecars from `data/nn/tilewright/,<arch>` (typically fetched externally). Boost models (`data/nn/boost/<arch>`) load in both configurations.

**hipBLASLt** uses the same `ORIGAMI_ENABLE_NN` option (default **OFF**). Enable NN when configuring hipBLASLt:

```bash
cmake -S projects/hipblaslt -B build/ -DORIGAMI_ENABLE_NN=ON
```

After a hipBLASLt build, YAML weights (with `ORIGAMI_ENABLE_NN=ON`) and boost models are copied next to each arch's Tensile library under `Tensile/library/<arch>/`, with one `origami_nn_index` holding both backends' entries, and installed with the library (`lib/hipblaslt/library/<arch>/` on Linux).

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

Index format (tab-separated):

```
<logic_stem>  tilewright  <manifest.yaml>
```

Load programmatically:

```cpp
#include "origami/nn/nn.hpp"

auto models = origami::nn::load_models_for_logic(logic_stem, library_dir);
// models.tilewright >= 0 on success; -1 if stem not in index
```

For tests, override the manifest path with `ORIGAMI_NN_WEIGHTS` when tilewright integration tests are added.

Boost models live under `shared/origami/data/nn/boost/<arch>/` (`origami_nn_index` and the `origami_boost_<arch>_<dtype>.orr` models, all tracked) with index lines

```
<logic_stem>  boost  origami_boost_<arch>_<dtype>.orr
```

A boost model is loaded for the library it serves, since its checks need that library's manifest:

```cpp
#include "origami/nn/nn.hpp"

const std::string path =
    origami::nn::index_weights_path(logic_stem, origami::nn::backend_id_t::boost_v1, library_dir);
origami::nn::boost::deployment_t target{"gfx950", "bf16", &library_manifest};
origami::nn::boost::status_t status;
std::string detail;
auto handle = origami::nn::load_boost_model(path, target, &status, &detail);
// handle >= 0 on success; -1 with status / detail saying why the model was refused
```


### Runtime Configuration

NN routing is controlled by `rank_options_t` fields and environment variables. Explicit `rank_options_t` values take precedence over environment defaults.

#### Inference mode

| `ORIGAMI_INFERENCE_MODE` | Behavior |
|--------------------------|----------|
| *(unset)* or `analytical` | Analytical roofline model (default) |
| `nn` | ML ranking only; throws if no model resolves or inference fails |
| `nn_fallback` | ML when a model is available; otherwise analytical |

`inference_mode_t::analytical_rerank` (analytical, then the boost re-rank) is set through `rank_options_t` only, with the caller's `rank_options.nn.rerank_source`.

#### Backend selection

| `ORIGAMI_NN_BACKEND` | Behavior |
|----------------------|----------|
| *(unset)* or `auto` | Prefer tilewright handle from `library_models`, then session default |
| `tilewright` | Force tilewright backend |
| `embedding_similarity` | Force embedding-similarity backend (not yet implemented) |

With `analytical_rerank`, `auto_select` and `nn_backend_t::boost` resolve the boost handle (`nn_model`, then `library_models->boost`, then the session default).

#### Diagnostics

| Variable | Description |
|----------|-------------|
| `ORIGAMI_NN_DIAG` | Print model-load diagnostics when a TWREC manifest is loaded (`arch`, feature dims, cell/split counts) or a boost model (`arch`, dtype, schema, columns, trees) |
| `ORIGAMI_NN_WEIGHTS` | Tests: override path to a `.tilewright.yaml` manifest |
| `ORIGAMI_TW_LOG` | Log top-1 tilewright pick per `rank_configs` call (`[TW_PICK]` line on stderr) |
| `ORIGAMI_TW_CELL` | Debug: force TWREC cell index (overrides routing); also via `rank_options.nn.force_cell` |

#### hipBLASLt row filter

| Variable | Description |
|----------|-------------|
| `TENSILE_PREDICTION_LIB=1` | Use `PredictionMatching` rows in `ExactLogicLibrary` (required for NN path in hipBLASLt bench) |
| `HIPBLASLT_ORIGAMI_BOOST=1` | Load boost models with the prediction rows that use them and re-rank their candidates (default off) |

### hipBLASLt Integration

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

Example: the same layout with the boost re-rank:

```bash
export TENSILE_PREDICTION_LIB=1
export HIPBLASLT_ORIGAMI_BOOST=1
export HIPBLASLT_TENSILE_LIBPATH=/path/to/build/Tensile/library/gfx950

hipblaslt-bench -m 3405 -n 4000 -k 5000 -r bf16_r \
  --transA T --transB N --bias_vector
```

hipBLASLt loads a prediction row's boost model when it deserializes the row (`OrigamiBoost::loadForLogic`: once per model file, against the build and the library directory's prediction-row manifest) and passes `analytical_rerank` with a `rerank_source_t`: its `prepare` applies the row's Stream-K cluster charge (as for the analytical order), and its candidates are the configs that pass the row's predicates, one per kernel name; their kernel fields are the solution's `NumLoadsCoalesced` and its Stream-K launch on the call (`ContractionSolution::computeStreamKDecisions`, including the `--sm_count_target` CU budget). FP8 re-ranks plain F8BS calls (Float8 compute input, A/B scales unset or scalar); vector-scaled (A/B or alpha), MX and swizzled calls keep Origami's order. Tuned `Equality`/`Range` rows still take precedence: only the `PredictionMatching` row's candidates are re-ranked. A boost request in `rank_options_t` takes precedence over `ORIGAMI_INFERENCE_MODE` / `ORIGAMI_NN_BACKEND`, as explicit options do. With `HIPBLASLT_LOG_MASK` including info, each call logs `Origami Boost ok ...` or `Origami Boost fallback (<reason>) ...`.

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

When `ORIGAMI_ENABLE_NN=OFF` (the default), `[nn]` filter tests are compiled out; the boost tests (`[nn][boost]`) run in both configurations. Their models and feature-parity samples (bf16, tf32, fp8) are the small fixtures in `tests/data/boost/`.

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
