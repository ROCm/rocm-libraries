# GEKO - GEMM Kernel Optimizer

A comprehensive Python framework for optimizing General Matrix Multiply (GEMM) kernels in hipBLASLt. GEKO automates the complete workflow from workload analysis to the final optimized library, providing significant GEMM performance improvements for AMD GPUs.

## Table of Contents

- [Overview](#overview)
- [GEKO Package Architecture](#geko-package-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [CLI Reference](#cli-reference)
- [Usage Guide](#usage-guide)
  - [Ductile-based Optimization](#1-ductile-based-optimization)
  - [Dense Search (Offline Tuning)](#2-dense-search-offline-tuning)
  - [Benchmark](#3-benchmark)
  - [Specialize](#4-specialize-bespoke-logic-per-traced-call)
- [Module Reference](#module-reference)
- [Common Usage Patterns](#common-usage-patterns)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

GEKO provides an end-to-end solution for optimizing GEMM kernels in hipBLASLt through two alternative tuning workflows and a standalone benchmark mode:

### Optimization Workflow
Generates the most optimized kernels for individual sizes and datatypes. There are three main steps:

1. **Configuration Phase** (`scripts/configure.py`): Analyze hipblaslt logs (actual workload) → Generate tensilelite configuration yamls for tuning. There are different options to create one config file per size for the best performance, or cluster similar sizes in the same config file for more efficient tuning.

2. **Optimization Phase** (`scripts/optimize.py`): Run GEMM kernel optimization → Benchmark → Filter → Create final library. There are different options to use (Genetic Algorithm for the best performance, or exhaustive kernel search across all possible kernel based on the yaml configuration file parameters for experiments).

3. **Integration Phase**: Merge the optimized libraries into hipBLASLt → Install

Note that the CLI merges steps 1 and 2 in one command. While we recommend the two-step approach (to review configurations before running optimization), you can use the CLI for a faster method. The instruction is in the following sections. 

Optimization Workflow:
```
hipBLASLt Logs → Configure → Optimize   →    Benchmark → Filter → Merge
      ↓             ↓           ↓               ↓          ↓        ↓
  YAML logs      Tensile     Tuning         Performance  Final   hipBLASLt
                 Configs  (Ductile or full)   Analysis   Library Integration
```

### Search Workflow (Dense Benchmarking - Offline Tuning)
Performs exhaustive search across existing solutions:

1. **Search Phase (AKA offline tuning)** (`scripts/search.py` or `./bin/geko --search`): Parse logs → Dense benchmark across all solutions → Extract winners → Filter → Create final library

Here is the sample command for gfx950:

`./bin/geko --search --hipblaslt /path/to/rocm-libraries/projects/hipblaslt --workload-log hipblaslt-log-mask64.yaml --devices=0,1,2,3,4,5,6,7`

To compare against a custom reference library during search/analyze, add:

`--custom-lib-src /path/to/custom_lib`

This is the workflow:
```
hipBLASLt Logs → Search → Extract → Benchmark → Filter → Merge
      ↓            ↓         ↓          ↓          ↓       ↓
  YAML logs      Dense    Winning   Performance  Final   hipBLASLt
               Benchmark  Kernels    Analysis   Library Integration
```

2. **Integration Phase**: Merge the optimized libraries into hipBLASLt → Install

### Benchmark Mode
Runs hipblaslt-bench on a workload log without any tuning. Useful for measuring baseline performance or verifying results after integration.

`./bin/geko --bench --hipblaslt /path/to/rocm-libraries/projects/hipblaslt --workload-log hipblaslt-log-mask64.yaml --devices=0`

Optional custom library inputs for `--bench`:
- `--custom-lib-src PATH`: source logic directory to build into `workdir/ref_build` and benchmark as reference.
- `--custom-lib-dir PATH`: pre-built custom library directory containing `library/**/TensileLibrary_lazy_gfx*.dat` (or `.dat.zlib`).
- If both are provided, `--custom-lib-dir` takes precedence.

### Workload Input Options

All three modes above accept exactly one of the following workload sources:

| Option | Description |
|--------|-------------|
| `--workload-log PATH` | hipBLASLt GEMM log YAML captured with `HIPBLASLT_LOG_MASK=64`. Contains real workload sizes from an application run. |
| `--list PATH` | Generator tuning YAML (see `geko/config_generator/config.yaml` for an example). Supports two size modes: an explicit list (`SIZE_OPTION=0`) or an internally generated M×N grid filtered by CU boundary constraints (`SIZE_OPTION=1`). |
| `--inline M N batch K DataType DestDataType ComputeDataType transA transB` | Single GEMM specified directly on the command line (e.g. `--inline 1024 1024 1 1024 B B S N T`). |

## GEKO Package Architecture

The `geko` package is organized into specialized modules for different optimization workflow stages:

```
geko/
├── bench/              # Benchmarking and performance analysis
│   ├── bench.py        # Core benchmark execution
│   ├── log.py          # hipBLASLt log file parsing
│   └── utils.py        # Benchmark parsing utilities
├── optim/              # Ductile/Tensile tensilelite configuration and execution
│   ├── optim.py        # Optimization and result analysis
│   └── utils.py        # Progress tracking and device management
├── search.py           # Dense search workflow (offline tuning)
├── specialize/         # Per-call kernel specialization (--specialize)
│   ├── calls.py        # Mask-128 trace ingestion into call identities and counts
│   ├── recipe.py       # Baseline build layout, MatchTable recipe resolution, name confirmation
│   ├── facts.py        # Per-call generation/narrowing facts and refusals
│   └── writer.py       # Bespoke logic, calls.csv, manifest, refusals, --build-library
├── library/            # Kernel library management
│   ├── library.py      # Library and LibraryCollection classes
│   ├── operations.py   # Library loading, merging, creation, etc.
│   └── _bank.py        # Solution bank utilities
├── config_generator/   # Tensile configuration generation utilities
│   ├── config.yaml     # Example configuration template
│   ├── config_generator.py
│   ├── config_merger.py
│   ├── config_sections_generator.py
│   ├── load_input_config.py
│   ├── mi_designer.py
│   ├── sizes.py
│   ├── cluster_sizes.py
│   ├── fork_param_generator.py
│   ├── output_writer.py
│   ├── shared_utils.py
│   ├── utils.py
│   ├── constants.py
│   └── fork_params/    # Per-architecture tuning parameter profiles
│       ├── optimization_param.py
│       ├── post_processor.py
│       ├── param_meta.py
│       └── hw_profiles/
│           ├── gfx942/ # gfx942 optimization params and post-processing
│           └── gfx950/ # gfx950 optimization params and post-processing
├── concurrency/        # Multi-GPU concurrency management
│   ├── runner.py       # Concurrent job execution
│   └── utils.py        # Concurrency utilities
├── cli.py              # Command-line interface entry point
├── pipeline.py         # Workflow orchestration pipeline
├── schemas.py          # Data structures (GemmType, GemmConfig, etc.)
├── constants.py        # Data type mappings and field definitions
└── utils.py            # Common utilities (e.g. device management)
```

### Workflow Output Structure

**Optimization Workflow Output:**
```
workdir/
├── run_state.json                 # Workflow state tracking
├── hipblaslt-log-mask64.yaml      # Updated benchmark file 
├── hipblaslt-log-mask64.out       # hipblaslt-bench output 
├── summary.csv                    # GEMM contribution analysis
├── gemms.csv                      # Extracted unique GEMMs
├── optimizations/                 # Phase 1: Configuration output
│   ├── BBS_TN_0.yaml              # Tensile config for GEMM type
│   ├── BBS_TN_0.sh                # Execution script
│   ├── F8BS_TN_1.yaml             # Another GEMM configuration
│   ├── ...
│   └── build_*/                   # Phase 2: Optimization results
│       ├── ...
│       ├── 3_LibraryLogic/        # Optimized kernel libraries
│       │   └── *.yaml             # Individual library files
│       ├── *-tensilelite.log      # Detailed tuning logs
│       └── *-optimization.log     # Optimization logs
├── libs/                          # Merged optimized libraries
│   └── *.yaml                     # Individual library per GEMM type
├── tensile/library/*.dat          # Compiled Tensile libraries
├── benchmarks/                    # Benchmark configurations
│   ├── *_bench.yaml               # Generated benchmark inputs
│   └── *_verify.yaml              # Verification inputs
├── results/                       # Performance analysis
│   ├── raw_results.csv            # All benchmark data
│   └── final_results.csv          # Filtered results
└── final_libs/                    # Filtered optimized libraries
    └── *.yaml                     # Individual libraries for integration
```

**Search Workflow Output:**
```
workdir/
├── run_state.json                 # Workflow state tracking
├── hipblaslt-log-mask64.yaml      # Updated benchmark file
├── hipblaslt-log-mask64.out       # hipblaslt-bench output
├── summary.csv                    # GEMM contribution analysis
├── gemms.csv                      # Extracted unique GEMMs
├── winners.csv                    # Best kernel per GEMM
├── search/                        # Dense search results
│   ├── BBS_TN_M1024_N32...yaml    # Individual GEMM configs
│   ├── BBS_TN_M1024_N32...out     # Benchmark outputs
│   └── failed_jobs.log            # Failed benchmark log
├── libs/                          # Extracted winning kernels
│   └── *.yaml                     # Individual library per GEMM type
├── tensile/library/*.dat          # Compiled Tensile libraries
├── benchmarks/                    # Benchmark configurations
│   └── *_bench.yaml               # Generated benchmark inputs
├── results/                       # Performance analysis
│   ├── raw_results.csv            # All benchmark data
│   └── final_results.csv          # Filtered results
└── final_libs/                    # Filtered optimized libraries
    └── *.yaml                     # Individual libraries for integration
```

---

## Requirements

### System Dependencies

#### ROCm Installation

ROCm 6.0+ is required. Follow the [Quick start installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html).

**Verify ROCm Installation:**
```bash
/opt/rocm/bin/hipconfig --full
```

Expected output:
```
HIP version: 6.5.50421-9e0f69f3c
HIP_PATH           :/opt/rocm-7.0.0
ROCM_PATH          :/opt/rocm-7.0.0
HIP_PLATFORM       :amd
...
```

Paths and version strings vary with your installation; the listing above is illustrative.

> **Troubleshooting ROCm**  
> If you encounter ROCm issues, try a [clean removal](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.1.0/install/install-methods/package-manager/package-manager-ubuntu.html#uninstalling) and reinstall. Refer to the [Installation Troubleshooting Guide](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.1.0/reference/install-faq.html) for additional help.
> For Runtime errors, refer to the [Runtime Error Triage Checklist](https://github.com/ROCm/rocm-libraries/blob/users/davidd-amd/hipblaslt-triage-checklist.md/docs/hipblaslt-runtime-triage-checklist.md).

#### System Packages
> [!NOTE]  
> One may want to fix any broken dependencies first by executing: ```sudo apt update && sudo apt --fix-broken install```


Run the following commands to install required packages:

```
sudo apt update
sudo apt install python3 python3-yaml
sudo apt install clang lldb lld
sudo apt install libomp-dev 
sudo apt install libboost-all-dev libboost-program-options-dev libboost-filesystem-dev 
sudo apt install libtinfo-dev libzstd-dev libmsgpack-dev libgtest-dev libgmock-dev 

```
---

## Installation

### 1. Clone hipBLASLt (Sparse Checkout)
```bash
git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries
git sparse-checkout init --cone
git sparse-checkout set projects/hipblaslt shared/origami shared/stinkytofu shared/mxdatagenerator cmake
git checkout develop
```

If you already have hipblaslt, make sure you have the latest commit/version.

### 2. Build hipBLASLt
```bash
cd rocm-libraries/projects/hipblaslt
pip install -r requirements.txt

invoke build --install-deps --clients --install-pkg --architecture gfx950 --skip-rocroller  # Replace gfx950 with your GPU architecture - If you need to work with MX datatypes, remove the rocroller flag - Remove the --install-pkg to build locally and to not sytem wide.
```

### 3. Install tensilelite dependencies
```bash
cd rocm-libraries/projects/hipblaslt/tensilelite
pip install -r requirements-dev.txt
```

### 4. Install GEKO Framework

GEKO uses a `pyproject.toml` build, `tox.ini` for test environments, and a
`tasks.py` `invoke` runner. The `invoke` tasks require `invoke` (install it via
`pip install invoke` or `pip install --group dev` if not already present).

```bash
cd rocm-libraries/projects/hipblaslt/utilities/geko

pip install .  # installs the geko console script + runtime deps (from requirements.txt); use pip install -e . for editable mode
```

To set up a development environment (test runner, linters, `invoke`):

```bash
pip install -e .          # runtime deps + geko console script
pip install --group dev   # dev tools: pytest, invoke, flake8, black, isort
```

Common developer tasks are exposed via `invoke` (run `invoke --list` to see all):

```bash
invoke install            # editable install (pip install -e .)
invoke test               # run the test suite
invoke test --skip-slow   # quick run
invoke lint               # flake8 over geko/
invoke format             # black + isort
invoke build              # build sdist + wheel into dist/
```

From the GEKO repo root you can also run the driver without installing (same code path as the installed CLI; adds the repo root to ``PYTHONPATH``):

```bash
./bin/geko --help
```

For the full flag list with descriptions and defaults, see [CLI Reference](#cli-reference).

Required packages (see `requirements.txt` for pinned versions):
- `pyyaml` - YAML parsing
- `pandas` - Data manipulation and tabular data operations
- `numpy` - Numerical array operations
- `tqdm` - Progress bars
- `joblib` - Parallel execution
- `pytest` - Test runner (used by the test suite under `tests/`)

### 5. Testing GEKO Framework

From the GEKO root (with dev dependencies installed):

```bash
python3 -m pytest tests/      # direct pytest
invoke test                   # via the invoke task runner
tox                           # in an isolated tox environment
```

`tox` and `invoke test` forward extra arguments straight to pytest, e.g.
`tox -- --skip-slow --skip-geko-bin` or `invoke test --skip-slow`.

Integration tests need a hipBLASLt repo and (in some cases) a tuning config and/or a workload log. Pass these via custom pytest options:

```bash
python3 -m pytest tests/ \
    --hipblaslt-path ~/rocm-libraries/projects/hipblaslt \
    --config tests/config_generator/fixtures/minimal_config.yaml \
    --workload tests/test_data/workload.yaml
```

Optional `--hw gfx942` (etc.) overrides the architecture used by configure / optimize integration tests; the default is `gfx950`.

Test markers and skip flags (registered in `tests/conftest.py`):
- `@pytest.mark.slow` — long-running subprocess / GPU integration tests. Skip with `--skip-slow`.
- `@pytest.mark.geko_bin` — subprocess smoke tests for `bin/geko`. Skip with `--skip-geko-bin`.

```bash
# Skip slow and bin/geko subprocess tests for a quick run
python3 -m pytest tests/ --skip-slow --skip-geko-bin
```

Coverage gate (local and CI):

```bash
# Full GEKO coverage with threshold enforcement (>=80%)
tox -e coverage

# Fast PR-like coverage (exclude slow + bin/geko subprocess tests)
tox -e coverage -- --skip-slow --skip-geko-bin
```

The coverage environment writes:
- `htmlcov/index.html`
- `coverage.xml`
- `coverage.json`
- `geko_tests.xml` (JUnit)

CI integration notes:
- GEKO exposes category metadata in `test_categories.yaml` for external pytest category runners.
- In CI lanes, run coverage from `projects/hipblaslt/utilities/geko` via `tox -e coverage`.
- For Codecov status checks, upload GEKO coverage with the `GEKO` flag.

See [`tests/README.md`](tests/README.md) for further details on the test layout.


---

## CLI Reference

`./bin/geko` is the single entry point for tuning and benchmarking. The flags below match `./bin/geko --help`. Per-script flags for `scripts/configure.py`, `scripts/optimize.py`, and `scripts/search.py` are documented under [Usage Guide](#usage-guide).

### Mode (exactly one required)
| Flag | Description |
|------|-------------|
| `--tune`   | Tuning workflow: configure then optimize. |
| `--search` | Dense-search tuning workflow. |
| `--bench`  | Benchmark GEMMs from the workload only (no tuning). |
| `--specialize` | One bespoke logic file per traced call identity, CPU only. See [Specialize](#4-specialize-bespoke-logic-per-traced-call). |

### Workload source (exactly one required)
| Flag | Description |
|------|-------------|
| `--workload-log PATH` | hipBLASLt GEMM log YAML (typically captured with `HIPBLASLT_LOG_MASK=64`). |
| `--list PATH` | Generator tuning YAML. See [Specifying GEMMs via a Tuning List (`--list`)](#specifying-gemms-via-a-tuning-list---list) and [`geko/config_generator/config.yaml`](geko/config_generator/config.yaml). |
| `--inline M N batch K DataType DestDataType ComputeDataType transA transB` | Single GEMM on the command line, e.g. `--inline 1024 1024 1 1024 B B S N T`. `transA`/`transB` must each be `N` or `T`. Optional 10th arg `MX` enables Microscaling mode (F4/F8 only) and requires `--arch`. |

### Common options (apply to all modes)
| Flag | Default | Description |
|------|---------|-------------|
| `--hipblaslt PATH` | _required_ | Path to a built hipBLASLt repository. |
| `-d, --devices LIST` | _required_ (not used by `--specialize`) | Comma-separated GPU device IDs (e.g. `0,1,2,3`). |
| `--workdir PATH` | autogenerated `geko_<timestamp>/` | Output directory for all artifacts of this run. |
| `--keep_thr FLOAT` | `0.0` (`0.1` for `--search`) | Drop GEMMs that contribute less than this fraction of total runtime. |
| `-v, --verbose {0,1,2}` | `1` | Logging verbosity: 0=WARNING, 1=INFO, 2=DEBUG. |
| `--bench-freq` | off | Set `HIPBLASLT_BENCH_FREQ` during `hipblaslt-bench` runs to capture clock-frequency telemetry. Off by default to avoid the small collection overhead. |

### `--tune` options
| Flag | Default | Description |
|------|---------|-------------|
| `--arch ARCH` | _none_ | Target gfx architecture. Can also come from `ARCH:` inside a `--list` YAML. Choices: `gfx950`, `gfx950_128cu`, `gfx942`, `gfx942_80cu`, `gfx942_38cu`, `gfx942_20cu`, `gfx942_228cu`, `gfx1250`, `gfx1250_96cu`, `gfx1250_192cu`, `gfx1250-strict`, `gfx1250-strict_96cu`, `gfx1250-strict_192cu` (A0). The retired `gfx1250v0*` names are still accepted, with a deprecation warning. A0 runs set `HSA_DISABLE_GFX12_STRICT=0`. |
| `--backend {ductile,tensile}` | `ductile` | Tuning backend used in the configure step. Only used with `--tune`. |
| `-n, --n_slots INT` | `4` | Max concurrent optimization jobs per device during the optimize step. |
| `--up_thr FLOAT` | `1.03` | Performance uplift threshold (e.g. `1.03` keeps kernels with ≥3% uplift). |
| `--no_retry` | off | Disable retry of failed optimization jobs. |

### `--search` options
| Flag | Default | Description |
|------|---------|-------------|
| `--duration SEC` | `0.04` | Target dense-search benchmark duration per GEMM. |
| `--custom-lib-src PATH` | _none_ | Optional source logic directory used to build a reference custom library (`workdir/ref_build`) for summarize/search/analyze comparisons. |

### `--bench` options
| Flag | Default | Description |
|------|---------|-------------|
| `--benchmark-duration SEC` | `0.5` | Target seconds per cold and per timed phase of `hipblaslt-bench`. |
| `--custom-lib-src PATH` | _none_ | Optional source logic directory used to build a reference custom library (`workdir/ref_build`) and benchmark against it. |
| `--custom-lib-dir PATH` | _none_ | Optional pre-built custom library directory containing `library/**/TensileLibrary_lazy_gfx*.dat` (or `.dat.zlib`). If set together with `--custom-lib-src`, this flag wins. |

### Validation rules (enforced by the parser)
- `--hipblaslt` must point to an existing directory.
- `--workload-log` and `--list` paths must exist.
- `--inline` requires integer M, N, batch, K and `transA`/`transB` ∈ {`N`, `T`}.
- `--arch` is required with `--tune`.
- `--arch` is also required whenever `--inline ... MX` is used (needed to resolve the correct MX scale value).
- `MX: True` (or an MX-only data type such as F4) combined with an ARCH that doesn't support Microscaling (e.g. the gfx942 family) raises an error.

---

## Usage Guide

| Feature | Best for | Speed |
|---------|----------|-------|
| [Ductile Optimization](#1-ductile-based-optimization) | Maximum performance, custom kernel parameters | Slow (hours) |
| [Dense Search](#2-dense-search-offline-tuning) | Fast results, finding best existing kernels | Faster (minutes) |
| [Benchmark](#3-benchmark) | Baseline measurement, result verification | Fast |

### Capturing a hipBLASLt Workload Log

All tuning and benchmark workflows require a workload input. The most common source is a hipBLASLt log captured from a real application run:

```bash
HIPBLASLT_LOG_MASK=64 HIPBLASLT_LOG_FILE=hipblaslt-log-mask64.yaml python my_application.py
```

Each entry in the log looks like:
```yaml
- {function: matmul, M: 16032, N: 109, K: 16384, lda: 16384, ldb: 16384, ldc: 16032, ldd: 16032, stride_a: 0, stride_b: 0, stride_c: 0, stride_d: 0, alpha: 1.0, beta: 0.0, transA: T, transB: N, batch_count: 1, scaleA: 0, scaleB: 0, scaleAlpha_vector: false, gradient: false, use_e: false, bias_vector: false, bias_source: d, a_type: bf16_r, b_type: bf16_r, c_type: bf16_r, d_type: bf16_r, scale_type: f32_r, bias_type: f32_r, compute_type: c_f32_r, activation_type: none, flush: false, rotating: 512, cold_iters: 0, iters: 0, call_count: 1}
```

Pass it to any mode with `--workload-log`, e.g. `--workload-log hipblaslt-log-mask64.yaml`.

### Specifying GEMMs via a Tuning List (`--list`)

When you do not have (or do not need) a captured log, you can describe the workload directly with a small generator-style YAML and pass it via `--list`. This is convenient for tuning a curated set of sizes (e.g. a fixed benchmark suite) without running an application.

The minimum content is the GEMM type, the architecture, and a list of `[M, N, batch, K]` sizes. A short example:

```yaml
TRANSA: 'N'
TRANSB: 'T'

DataType: "B"          # bf16 inputs
DestDataType: "B"      # bf16 outputs
ComputeDataType: "S"   # fp32 accumulate

ARCH: "gfx950"

Sizes:
  - [1024, 1024, 1, 1024]
  - [2048, 2048, 1, 2048]
  - [4096, 4096, 1, 4096]
```

Save it as e.g. `my_gemm_list.yaml` and pass it to any mode:

```bash
./bin/geko --bench --hipblaslt /path/to/rocm-libraries/projects/hipblaslt \
  --list my_gemm_list.yaml --devices=0
```

The full template (StreamK, search space, MI filtering, `SIZE_OPTION=1` grid mode, etc.) lives at `geko/config_generator/config.yaml`. Multiple sizes per GEMM type and multiple GEMM problems per file are supported. See [Workload Input Options](#workload-input-options) for the complete set of input flags.

---

### 1. Ductile-based Optimization

Tunes new kernel parameters using a Genetic Algorithm (via Ductile) or exhaustive Tensile grid search. Produces the highest performance gains but takes hours.

#### Option A: Single CLI command

```bash
./bin/geko --tune --hipblaslt /path/to/rocm-libraries/projects/hipblaslt \
  --workload-log hipblaslt-log-mask64.yaml --arch gfx950 --devices=0,1,2,3
```

#### Option B: Two-step script control (recommended — lets you review configs before tuning)

**Step 1 — Configure** (`scripts/configure.py`): parse the log, group GEMMs by type, generate tensilelite configs.

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `-d, --device` | GPU device for baseline benchmarking | 0 |
| `--keep_thr` | Filter threshold (% of total time) | 0 |
| `-a, --architecture` | Target GPU architecture | gfx950 |
| `-b, --backend` | Tuning backend: `ductile` or `tensile` | ductile |
| `-w, --workdir` | Working directory | workdir |
| `-v, --verbose` | Logging verbosity (0=WARNING, 1=INFO, 2=DEBUG) | 1 |
| `--bench-freq` | Enable `HIPBLASLT_BENCH_FREQ` during benchmark runs (only when `--keep_thr > 0`) | False |

```bash
python scripts/configure.py /path/to/rocm-libraries/projects/hipblaslt/ hipblaslt-log-mask64.yaml \
  -a gfx950 --workdir my_optimization --keep_thr 0.05 --device 0
```

Output:
```
my_optimization/
├── run_state.json
├── hipblaslt-log-mask64.yaml / hipblaslt-log-mask64.out  # Baseline benchmark
├── summary.csv / gemms.csv           # GEMM contribution analysis
└── optimizations/
    ├── BBS_TN_0.yaml                 # Tensile config per GEMM type
    ├── BBS_TN_0.sh
    └── ...
```

Log:
```
GEKO:INFO [configure:main] Starting configuration phase...
GEKO:INFO [configure:main] Found 2 unique GEMMs after filtering
GEKO:INFO [optim:configure] GemmType(transA='T', transB='N', a_type='bf16_r', ...) with 2 sizes
GEKO:INFO [configure:main] Generated 1 configuration files in 'my_optimization/optimizations'
GEKO:INFO [configure:main] Configuration phase completed successfully!
```

**Step 2 — Optimize** (`scripts/optimize.py`): run tuning across GPUs, merge, benchmark, and filter results.

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `-w, --workdir` | Working directory with configs | workdir |
| `-d, --devices` | Comma-separated GPU device IDs | 0,1,2,3,4,5,6,7 |
| `-n, --n_slots` | Max concurrent optimization jobs per device | 4 |
| `--up_thr` | Performance uplift threshold | 1.03 (3%) |
| `--err_thr` | Error threshold for filtering | 0.03 |
| `--client_build_dir` | Directory path for tensilelite client build | build_tmp |
| `--no_retry` | Do not retry failed optimizations | False |
| `-v, --verbose` | Logging verbosity (0=WARNING, 1=INFO, 2=DEBUG) | 1 |
| `--bench-freq` | Enable `HIPBLASLT_BENCH_FREQ` during benchmark runs | False |

```bash
python scripts/optimize.py /path/to/rocm-libraries/projects/hipblaslt/ \
  --workdir my_optimization --devices=0,1,2,3 --up_thr 1.03
```

Log:
```
Optimization in progress: 100%|████████| 2/2 [04:33<00:00, 136.5s/it, n_completed: 2, n_failed: 0]
GEKO:INFO [optim:analyze] Average GEMM uplift of 3.1002% for a total of 1 GEMMs
GEKO:INFO [optimize:main] Final optimized library available in: 'my_optimization/final_libs'
GEKO:INFO [optimize:main] Optimization workflow completed successfully!
```

Output:
```
my_optimization/
├── optimizations/build_*/            # Per-GEMM tuning results and logs
├── libs/                             # Merged libraries
├── tensile/library/*.dat             # Compiled Tensile libraries
├── benchmarks/                       # Benchmark inputs
├── results/raw_results.csv / final_results.csv
└── final_libs/                       # Ready for integration
    └── gfx950_*.yaml
```

#### Step 3 — Integrate

```bash
HIPBLASLT_PATH="/path/to/rocm-libraries/projects/hipblaslt"
LIBRARY_DIR="${HIPBLASLT_PATH}/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx950/Equality/"

${HIPBLASLT_PATH}/tensilelite/tensilelite/bin/TensileMergeLibrary \
  --no_eff --force_merge True \
  "${LIBRARY_DIR}" my_optimization/final_libs "${LIBRARY_DIR}"

cd "${HIPBLASLT_PATH}"
invoke build --install-deps --clients --architecture gfx950 --skip-rocroller
```

`TensileMergeLibrary` arguments: `--no_eff` skips efficiency calculations; `--force_merge True` forces merge on conflicts; the three positional args are original dir, new libs dir, output dir (same as original to update in place).

**Verify:**
```bash
"${HIPBLASLT_PATH}/build/release/clients/staging/hipblaslt-bench" \
  --yaml my_optimization/hipblaslt-log-mask64.yaml --device 0
```

**Install system-wide (optional):**
```bash
cd "${HIPBLASLT_PATH}/build/release"
sudo make -j 64 install && sudo make package
sudo dpkg -i hipblaslt[-\_]*.deb
```

---

### 2. Dense Search (Offline Tuning)

Benchmarks all existing hipBLASLt solutions exhaustively and picks the winner per GEMM — no new kernel parameters are tuned. Much faster than Ductile optimization.

Both entry points below run the same `run_search` pipeline and produce the same `my_search/` layout.

#### Option A: Single CLI command

```bash
./bin/geko --search --hipblaslt /path/to/rocm-libraries/projects/hipblaslt \
  --workload-log hipblaslt-log-mask64.yaml \
    --devices=0,1,2,3,4,5,6,7 --workdir my_search --keep_thr 0.1 --up_thr 1.03 \
    --custom-lib-src /path/to/custom_lib
```

#### Option B: Script entry point (`scripts/search.py`)

Equivalent to Option A but takes the workload log as a positional argument and uses the script-style defaults shared with `configure.py` / `optimize.py`. Only a hipBLASLt workload log (YAML or CSV with the same fields) is accepted as input; for `--list` (generator YAML) or `--inline` GEMM specs, use Option A.

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `-d, --devices` | Comma-separated GPU device IDs | 0,1,2,3,4,5,6,7 |
| `--keep_thr` | Filter threshold (% of total time) | 0.1 |
| `--up_thr` | Performance uplift threshold | 1.03 (3%) |
| `-w, --workdir` | Working directory | workdir |
| `--duration` | Target benchmark duration per GEMM (seconds) | 0.04 |
| `--custom-lib-src` | Source logic directory for building reference custom library | None |
| `-v, --verbose` | Logging verbosity (0=WARNING, 1=INFO, 2=DEBUG) | 1 |
| `--bench-freq` | Enable `HIPBLASLT_BENCH_FREQ` during benchmark runs | False |

```bash
python scripts/search.py /path/to/rocm-libraries/projects/hipblaslt/ hipblaslt-log-mask64.yaml \
    --devices=0,1,2,3,4,5,6,7 --workdir my_search --keep_thr 0.1 --up_thr 1.03 \
    --custom-lib-src /path/to/custom_lib
```

Log:
```
Search in progress: 100%|████████| 2/2 [10:25<00:00, 20.85s/it, n_completed: 2, n_failed: 0]
GEKO:INFO [optim:analyze] Average GEMM uplift of 10.4697% for a total of 2 GEMMs
GEKO:INFO [search:main] Final remapped library available in: 'my_search/final_libs'
GEKO:INFO [search:main] Search workflow completed successfully!
```

Output:
```
my_search/
├── hipblaslt-log-mask64.yaml / hipblaslt-log-mask64.out  # Baseline benchmark
├── summary.csv / gemms.csv / winners.csv
├── search/                           # Per-GEMM benchmark files and outputs
├── libs/                             # Extracted winning kernels
├── tensile/library/*.dat
├── benchmarks/
├── results/raw_results.csv / final_results.csv
└── final_libs/                       # Ready for integration
    └── gfx950_*.yaml
```

#### Integrate

Same steps as Ductile Optimization — run `TensileMergeLibrary` then rebuild hipBLASLt with the contents of `my_search/final_libs`.

---

### 3. Benchmark

Runs hipblaslt-bench on a workload without any tuning. Useful for measuring baseline performance or verifying results after integration.

```bash
# From a workload log
./bin/geko --bench --hipblaslt /path/to/rocm-libraries/projects/hipblaslt \
  --workload-log hipblaslt-log-mask64.yaml --devices=0

# From a source custom library (builds ref library under workdir/ref_build)
./bin/geko --bench --hipblaslt /path/to/rocm-libraries/projects/hipblaslt \
    --workload-log hipblaslt-log-mask64.yaml --devices=0 \
    --custom-lib-src /path/to/custom_lib

# From a pre-built custom library directory
./bin/geko --bench --hipblaslt /path/to/rocm-libraries/projects/hipblaslt \
    --workload-log hipblaslt-log-mask64.yaml --devices=0 \
    --custom-lib-dir /path/to/custom_build

# From a single inline GEMM
./bin/geko --bench --hipblaslt /path/to/rocm-libraries/projects/hipblaslt \
  --inline 1024 1024 1 1024 B B S N T --devices=0

# From a generator tuning YAML
./bin/geko --bench --hipblaslt /path/to/rocm-libraries/projects/hipblaslt \
  --list my_gemm_config.yaml --devices=0
```

### 4. Specialize (bespoke logic per traced call)

`--specialize` turns hipBLASLt mask-128 traces into one bespoke TensileLite solution per distinct call: the stock solution hipBLASLt actually ran for the call (its fixed recipe), with the proven facts of the call written as `ProblemType` and assertion keys and `BespokeNarrowing: true`. It runs on the CPU (TensileLite from the hipBLASLt source checkout and its rocisa package importable, `amdclang++` on `PATH`; see Specialize setup below); `--devices` is not used.

```bash
./bin/geko --specialize --workload-log traces/ --route-facts route_facts/ \
  --hipblaslt /path/to/baseline-build --arch gfx950 --workdir specialize_out
```

**Specialize setup (from scratch).** Steps 1–3 of Installation give the source checkout, a cmake build of hipBLASLt (the baseline) and the TensileLite dependencies including rocisa. Then:

```bash
export HIPBLASLT_SRC=/abs/path/to/rocm-libraries/projects/hipblaslt   # the source checkout (absolute)
export BASELINE=$HIPBLASLT_SRC/build/release                         # the cmake build directory of step 2
export W=/abs/path/to/specialize-work                                # all traces and outputs live here
export PYTHONPATH=$HIPBLASLT_SRC/tensilelite:$HIPBLASLT_SRC/utilities/geko:$PYTHONPATH
export PATH=/opt/rocm/bin:$PATH                                      # amdclang++ for TensileLite
mkdir -p $W/traces/my-config $W/api_traces/my-config $W/route_facts
# Content hash of the baseline library the workload loads (binds the traces to the baseline):
LIBFP=$(python3 -c "from geko.specialize.recipe import library_fp; print(library_fp('$BASELINE/Tensile/library/gfx950'))")
# Run the workload twice on that library, one log per process (%i becomes the process id):
HIPBLASLT_TENSILE_LIBPATH=$BASELINE/Tensile/library HIPBLASLT_LOG_MASK=128 \
  HIPBLASLT_LOG_FILE=$W/traces/my-config/hipblaslt_%i.log <your workload>
HIPBLASLT_TENSILE_LIBPATH=$BASELINE/Tensile/library HIPBLASLT_LOG_MASK=16 \
  HIPBLASLT_LOG_FILE=$W/api_traces/my-config/hipblaslt_%i.log <the same workload>
# Then write the run.json beside each trace (fields below) and run, from any directory:
$HIPBLASLT_SRC/utilities/geko/bin/geko route-facts $W/traces/my-config \
  --api-log $W/api_traces/my-config --out $W/route_facts/my-config.json
$HIPBLASLT_SRC/utilities/geko/bin/geko --specialize --workload-log $W/traces --route-facts $W/route_facts \
  --hipblaslt $BASELINE --arch gfx950 --workdir $W/specialize_out
```

`model_run` writes `run.json` itself. Otherwise write it after each run with what was true for that run, as in this example (`$W/traces/my-config/run.json`). **Mandatory** (missing means unknown, and unknown is never proof): `status`, `mode`, `run_configuration`, `env_values` (a mapping), `process_inventory` (every entry with a `pid` and a known `role`) and, for `--specialize`, `library_fp`. Binding keys (`model`, `library_fp`, `fingerprints`, `cache`) must be recorded by the API run whenever the trace run records them. **Checked when recorded** (a recorded bad value is a stop; the log content also shows the mask): `hipblaslt_log_mask`, `engine_exit`, `late_signals`, `reasons`, `env_after_import`, `aiter_switches`. Record them as in the example.

```json
{"status": "complete", "mode": "analysis", "hipblaslt_log_mask": 128, "engine_exit": 0,
 "run_configuration": "my-config", "model": {"slug": "my-model"}, "library_fp": "<LIBFP>",
 "env_values": {},
 "process_inventory": [{"pid": 4242, "role": "engine"}]}
```

`$W/api_traces/my-config/run.json` is the same with `"mode": "analysis-api"`, `"hipblaslt_log_mask": 16` and its own `process_inventory`; `run_configuration`, `model` and `library_fp` must equal the trace's (they bind the API run to the traced run).
- `library_fp` is the `LIBFP` printed above, the content hash of the library directory the run loaded. A `run.json` without `library_fp` refuses that configuration (fingerprint missing). Only a configuration with no `run.json` at all is `unverified`, and then `geko route-facts` cannot prove the runtime route either, so its calls are refused too.
- `env_values` records the environment variables the run set: `{}` means none of the settings that bypass the hipBLASLt heuristic was set. A run with `PYTORCH_TUNABLEOP_ENABLED=1`, `HIPBLASLT_TUNING_OVERRIDE_FILE`, `HIPBLASLT_TUNING_MODE`, `TENSILE_SOLUTION_INDEX` or `AITER_CONFIG_GEMM_BF16` set cannot be specialized. Without `env_values` (or with a non-mapping value) the runtime route is unproven.
- `process_inventory` lists every traced process with its role: `engine` or `worker` for a process that runs the model, `supervisor` or `helper` for one that runs no GEMM. A process without a role, or with any other role (including `unattributed`), cannot be classified, so the runtime route is unproven. A log of a process not in the inventory, an `engine`/`worker` process without its log, or an `engine`/`worker` log with no captured call while other processes captured calls (a lost log) is a stop; without `process_inventory` the runtime route is unproven.
- The run configuration of a trace directory is its `run.json` `run_configuration`, else the directory name. `geko route-facts` and `--specialize` use this same rule, so a model-run `.../<run>/out` directory is matched by the name its `run.json` records, not by `out`.

**Inputs**
- `--workload-log`: one mask-128 log, or a directory of run-configuration directories, each holding that configuration's `hipblaslt_<pid>.log` files (other files and subdirectories are ignored) and, optionally, the `run.json` of the run. Collect the traces with `HIPBLASLT_LOG_MASK=128` (one record per distinct call with its `call_count`, written at process exit). Mask-64/mask-34 logs, unparseable or truncated logs, grouped-GEMM records and an unreadable `run.json` refuse the whole input: nothing is generated and `refusals.json` holds one entry naming the file. A configuration with a `run.json` (a model-run output, also for a single log with `run.json` beside it) is used only when that `run.json` reports a finished analysis run, `status: complete` and `mode: analysis`; any other status or mode (running, failed, startup, greedy, performance, the mask-16 `analysis-api` run) or a missing key refuses that configuration (code `run-incomplete`, reason naming the `run.json` and the key) and none of its calls are read. A NaN activation argument (`nan`, `-nan`) is one call identity, recorded as `nan`; `inf` and `-inf` stay distinct.
- `--route-facts` (optional): one route-facts JSON per run configuration (`<cfg>.json` in a directory, or one file), normally written by `geko route-facts` (below): per call identity the API route (`call_route`), the bias axis, whether amax output is produced, and the evidence for each. A fact without evidence counts as unknown. Without `--route-facts`, or without an entry for a call, the route is unknown and the call is refused (`missing generation fact: route`); nothing is assumed from the mask-128 trace alone. Route and domain (generation eligibility) are decided per configuration: a missing or non-ordinary route in one configuration refuses that identity's calls in that configuration only. When one call identity is eligible in several configurations, its narrowing facts (the bias axis) are decided once over those configurations: a fact holds only if every one of them proves it, so the identity gets one bespoke solution and one expected kernel.
- `--hipblaslt`: the **baseline root** whose library served the traces. It must hold that library together with the same build's own MatchTable (the recipe map; geko never substitutes another build's map), in the first of these layouts that exists: a cmake build tree (`device-library/MatchTable.yaml` and `Tensile/library/<arch>/`), the output of `geko --specialize --build-library` (`library/MatchTable.yaml` and `library/<arch>/`), or an install that kept the map (`lib/hipblaslt/library/MatchTable.yaml` beside the library files in `lib/hipblaslt/library/`; installs do not ship the map, and the fingerprint covers the whole directory, so the map must be copied there before the traced run's `library_fp` is computed, otherwise the fingerprints differ and the calls are refused). The fingerprint is the content hash of that library directory. The **stock logic root** (the directory the library was built from) is `--logic-root DIR` when given, else `logic=` on the first line of `params.txt` in the root (written by some build scripts), else the hipBLASLt source checkout directory `projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full` that holds every source logic file the MatchTable names (the normal case for a cmake build tree), else, for `--build-library` output, its `logic/` directory. A given root (`--logic-root` or `params.txt`) must hold every source logic file the MatchTable names (a symlinked path is fine); otherwise it is a usage error, so a merged build never copies a different stock tree. The MatchTable's value for an index is the solution's `SolutionIndex`, as TensileCreateLibrary writes it, and it is authoritative: exactly one solution of the file must carry it (its real position is recorded), and TensileLite's names of that solution must equal the logged names; otherwise the call is refused (`recipe-name-mismatch`), never resolved to another solution. No MatchTable, no stock logic root, a malformed MatchTable entry or a source logic file the MatchTable names but that does not exist is a usage error (exit 2) naming what is missing, before anything is written.
- `--arch gfx950|gfx942` selects the library directory `<arch>` and TensileCreateLibrary's `--architecture`. gfx950 is the validated setting; gfx942 output is labelled `untested-gfx942` in every file.
- `--keep-beta` (manifest stage `keep-beta`): beta-0 calls keep `UseBeta: true`. `--exclude-controls YAML`: `{exclusions: [{recipe: {source_file, sha256, in_file_index}, identity: {...as logged}, facts: [...]}]}` withholds request facts (`UseBeta`, `ActivationType`, `UseBias`, `UseScaleAlphaVec`, `AssertFree0ElementMultiple`, `AssertFree1ElementMultiple`, `AssertSummationElementMultiple`) for that (recipe, call) pair only. Identity values are strings exactly as logged, so quote them in the YAML (`beta: '0.000000'`); a non-string value is a usage error. `AssertSizeEqual` is mandatory and cannot be excluded. Excluding `UseBias` for a call without bias (`bias_vector: false`) on a bias-capable recipe is refused (code `unreachable`, a bias reason): a bespoke solution that keeps `UseBias` only serves calls with a bias vector, so it could never serve that call. An entry that matches no call is reported under `unused_exclusions` in the manifest.
- `--build-library OUT` builds the complete merged library in one command: `OUT/logic/stock/` is a byte-identical copy of the baseline's whole stock logic root, `OUT/logic/bespoke/` holds this run's bespoke files, and TensileCreateLibrary runs on `OUT/logic` with the cmake build's options (`--architecture=<arch>` from `--arch`, `--cxx-compiler=/opt/rocm/bin/amdclang++`, `--no-compress`, `--disable-asm-comments`, `--gfx1250v0`; msgpack, lazy loading, no kept build temporaries) into `OUT/library/<arch>`, with that build's own index map at `OUT/library/MatchTable.yaml` (next to, not inside, the library directory; its indices are new, the baseline's do not apply) and `tcl.log` in `OUT`; gfx942 also gets `OUT/UNTESTED-gfx942`. `--jobs N` sets the job count (1..64; default the CPU affinity, at most 64). With `--stock-only` no bespoke logic is generated and the same command builds the stock-only library (`OUT/logic/stock/` only); its manifest expects the stock kernels. The manifest records the build command and `timings_seconds` (`parse_specialize`, `logic_merge`, `tensile_create_library`, `total`). A full gfx950 build compiles every stock kernel (the cmake stock build took about 22 minutes with 60 jobs).

**Route facts (`geko route-facts`).** The mask-128 trace cannot show how a call reached hipBLASLt. Run the same workload once more with `HIPBLASLT_LOG_MASK=16` (API trace) and derive the route facts of each run configuration from both traces:

```bash
./bin/geko route-facts traces/my-config --api-log api_traces/my-config --out route_facts/my-config.json
```

The first argument is the mask-128 configuration directory or one log; the run configuration is the `run_configuration` of the `run.json` beside it, else the directory name (the rule `--specialize` uses too). `--api-log` is the mask-16 log or its directory. The producer proves, per configuration, that every API call is an ordinary matmul API call (`rocblaslt_create`, matrix layout, matmul desc/preference create, set attribute, destroy, `rocblaslt_matmul_algo_get_heuristic`, `rocblaslt_matmul`) with one heuristic query per matmul, no pinned algorithm index, tuning override or TunableOp, no amax output attribute (`AMAX_D` set at all is not proof of absence, whatever pointer value is printed) and no bias-gradient epilogue, that every mask-128 record is `function: matmul` with D-source, non-gradient bias (so the bias axis is M), and that the weighted mask-128 `call_count` total equals the API `rocblaslt_matmul` count. From the `run.json` (and `engine.log`, if present) beside each trace it proves the **runtime route**, affirmatively: both runs `status: complete` and clean (`engine_exit` 0, no `late_signals` or `reasons` when recorded), modes `analysis` / `analysis-api`, log masks 128 / 16 when recorded; both record `run_configuration`, `env_values` as a mapping and a `process_inventory` whose every process has a known role, that holds every supplied log, and in which every `engine`/`worker` process has a log with captured calls whenever the run has calls; the API run records every binding key the trace run records (`run_configuration`, `model`, `library_fp`, `fingerprints`, model `cache`) with the same value; and no TunableOp, tuning override or pinned solution index appears in the recorded environment (`env_values`, `env_after_import`, `aiter_switches`) or as a TunableOp message in `engine.log` (settings are reported by name, never by value). Then it writes one entry per call identity with `call_route: "ordinary matmul API"`, `bias_axis: "M"`, `output_amax: false`, `runtime_route: "hipBLASLt heuristic"` and the evidence (file, line or line range, sha256) and exits 0. Recorded evidence against the proof is a stop (e.g. `api-count-mismatch` with both counts, `run-binding` with the differing keys, `inventory-stray-log`, `inventory-missing-log`, `inventory-empty-log`): the file is still written with `stops` and no call entries, the stops are printed, the exit status is 1, and `--specialize` refuses that configuration's calls naming the stop codes (`route-facts-stopped`). Missing evidence (no `run.json`, no `env_values`, no `process_inventory`, a process without a known role, a binding key the API run does not record) leaves the runtime route unproven: the entries are written with `runtime_route: null`, `runtime_route.reason` says what is missing, the exit status is 1, and `--specialize` refuses those calls (`runtime-route-unproven`), also for a configuration without calls. Exit 2 is a usage error (no trace, not a mask-128 trace, unreadable `run.json`).

The schema is [`docs/route_facts.schema.json`](docs/route_facts.schema.json). What `--specialize` reads: `run_configuration` (else the file name), `calls` and `stops`; per call `identity` (the logged fields as strings, compared exactly), `call_route`, `bias_axis`, `output_amax`, `runtime_route` and `evidence` (a fact counts only with a non-empty evidence list); for a configuration without calls the document's `runtime_route` (`proven: true` with evidence), `zero_verified`, `thin`, `matmul_count` and `per_process`. Accepted values: `call_route` `"ordinary matmul API"` (any other string is a non-ordinary route and refused, the refusal names the accepted value); `runtime_route` `"hipBLASLt heuristic"` (missing, null or anything else is refused); `bias_axis` `"M"` (`"N"` or anything else is refused); `output_amax` `false` (`true`, unproven or any non-boolean value is refused, the refusal lists the accepted values). An unknown or missing value is never read as a default.

**Recipe identity.** The logged `solution_index` is looked up in the baseline MatchTable; the solution found must give the logged solution and kernel names under TensileLite's own naming. The recipe is recorded as (source logic file relative to the stock logic root, its sha256, position in the file). When a configuration directory holds `run.json`, its `library_fp` must equal the content hash of the baseline library directory (a `run.json` without `library_fp` refuses that configuration); without `run.json` the fingerprint is recorded as `unverified` and the name check is the identity proof.

**Outputs** (in `--workdir`)
- `logic/<arch>/*.yaml`: one single-solution Equality logic file per distinct bespoke solution, named by its content; calls whose bespoke solutions are identical share one file, even across recipes. Stock logic is never modified.
- `calls.csv`: one row per call identity, run configuration and observed recipe, with every logged identity field, `call_count`, the stock recipe, the expected bespoke kernel, the status and the call-corpus manifest columns (`problem_id`, `tier`, `provenance`, …). Logged values whose manifest column has another spelling are kept in `<field>_raw` (`pointer_mode_raw` 0/1/4 next to `pointer_mode` host/device/unknown). A field the record did not log is written `unknown`.
- `specialize_manifest.json`: `stage`, either `keep-beta` (run with `--keep-beta`: beta-0 calls keep `UseBeta: true`) or `B` (run without it: the logic is beta-off, beta-0 calls on host pointer mode are narrowed to `UseBeta: false`); per generated call its logged identity, `kernel_name` (the expected bespoke kernel, as TensileLite names it), recipe, facts applied, facts not applied with reasons and the `keep_beta` flag; custom-kernel calls with `status: custom-kernel-recipe-unsupported` and `kernel_name: null` (a fallback-counting consumer never counts them as served); the build fingerprint; `zero_call_configurations`; `configuration_errors` (configurations refused as a whole, e.g. `run-incomplete`); `unused_exclusions`; `refusals` (the same list as `refusals.json`).
- `refusals.json`: per refused call identity its reason, call count and run configurations; configurations refused as a whole have `identity: null`.

**Status values** (`status` in `calls.csv` and the manifest): `generated` (bespoke logic written for the call), `stock-only` (`--stock-only`: the call is served by its stock recipe), `refused` (not generated; the reason is in `refusals.json`) and `custom-kernel-recipe-unsupported` (the call's recipe is a hand-written custom kernel, which cannot be specialized; listed with `kernel_name: null` and refused with reason `custom-kernel recipe unsupported`). The manifest lists the `generated`, `stock-only` and `custom-kernel-recipe-unsupported` calls.

After the usage checks pass, each run first removes what an earlier run wrote in `--workdir`: `calls.csv`, `specialize_manifest.json`, `refusals.json` and the generated `logic/<arch>/<arch>_bespoke__*.yaml` files. Any other file, also under `logic/`, is kept and never built. `--build-library OUT` records every file and directory it creates in `OUT/.geko_specialize_build.json`; the next build into the same `OUT` first removes exactly those (directories only if that leaves them empty), so a rebuild — for example merged, then `--stock-only` — leaves the same library as a fresh build and no obsolete bespoke artifacts. `OUT/logic` and `OUT/library` belong to geko alone: if either holds any path that record does not list (a file you added, a stock-path collision, extra logic), or is a symlink or not a directory, the build is a usage error and nothing changes; use a clean `OUT`. `OUT` must be a directory (or not exist yet); `OUT` may itself be a symlink to a directory, and these checks then apply to its target. Files elsewhere in `OUT` are kept. `OUT` and `--workdir` must be separate, non-nested directories (after resolving symlinks); otherwise it is a usage error. A usage error (exit 2) changes nothing.

**Refusals.** A call is not generated, and is listed with its reason, when a generation fact is missing (types, sizes, transposes, leading dimensions, strides, route), when the route is not the ordinary matmul API (grouped GEMM, device args, hipblaslt-ext, RocRoller) or the runtime route is not proven, for gradient, aux, amax or scaled input/output, an unsupported activation, sigmoid with bias, a bias source or axis other than D/M, a route-fact value outside the accepted ones, a recipe type mismatch (code `type-mismatch`; its reason also names the recipe's DataType / MAC input type when the recipe's A/B data type or MAC input type differs from the call's), a non-host pointer mode (1 or 4) on a recipe without `UseScaleAlphaVec` (`recipe cannot serve device pointer mode (no SAV)`: device-pointer alpha runs through the scale-alpha-vector path), an index not in the MatchTable, a recipe name mismatch, two recipes for one call identity, a trace/build fingerprint mismatch or missing fingerprint, a bespoke derivation the target TensileLite tree rejects or that does not keep `BespokeNarrowing`/`AssertSizeEqual`, a bespoke solution whose selection facts reject its own call (code `unreachable`: a kept `UseBias` for a call whose `bias_vector` is false or not logged, `UseBias: 0` for a call with a bias vector, a bias type list without the call's bias type, `UseBeta: false` for a nonzero beta, `UseScaleAlphaVec: 0` for a call with a scale-alpha vector or device pointer mode, a single activation other than the call's, or exact sizes and assertions the call does not meet), a TensileLite error while deriving the call's stock or bespoke solution (`derivation-error`, naming the stock logic file and position, with the error text; the other calls continue), and a hand-written custom-kernel recipe (`custom-kernel recipe unsupported`). All other calls are still generated; the exit status is 1 whenever anything was refused. A configuration with no calls (empty or record-free logs) is accepted only when its route facts prove zero calls (`zero_verified`) and, if it has a `run.json`, its `library_fp` matches the build; otherwise it is reported as refused. A configuration whose `run.json` is not a complete analysis run is refused as a whole (`run-incomplete`); its calls are not read, so its `call_count` is `null` (not counted), never 0.

**Unknown facts keep the stock capability.** A pointer mode other than host (0), or not logged, and any other unknown narrowing fact (beta, activation or its clamp arguments, bias type or axis, scale-alpha vector) keep the stock capability that fact would have narrowed; the call is generated and the withheld fact is listed under `facts_not_applied`. For the pointer mode this holds on a recipe with `UseScaleAlphaVec`; a known non-host mode (1 or 4) on a recipe without it is refused (`recipe cannot serve device pointer mode (no SAV)`, see Refusals). Bias presence is the exception: a bespoke solution that keeps `UseBias` serves only calls with a bias vector, so a call whose `bias_vector` is not logged on a bias-capable recipe is refused (`unreachable`), not generated. A single sigmoid activation is not applied while bias stays general (TensileLite does not support sigmoid with bias).

**`BespokeNarrowing` contract.** Every bespoke solution carries `BespokeNarrowing: true` and the exact sizes (`AssertSizeEqual` M/N/batch/K); the generator removes code only on facts asserted that way, and the host launch guard rejects a call whose facts differ.

**Interim deployment and known limits.**
- The generated bespoke logic is deployed today by a full local rebuild of the library (stock logic plus bespoke files) loaded through `HIPBLASLT_TENSILE_LIBPATH`. That is the interim deployment, not the end state (a small basic library plus bespoke kernels, later JIT).
- `--keep-beta` output is labelled `keep-beta`: beta stays general because the model route cannot yet reach a beta-off kernel. The C heuristic queries with beta = 1, so a `UseBeta: false` bespoke solution is not selected on the model route until the heuristic selects with the real beta; output without `--keep-beta` is for a host whose heuristic does.
- Pinned-index APIs (a solution chosen by index) do not fall back: a call whose facts differ from the bespoke solution's narrowed facts (for example a non-zero beta on a beta-off solution) returns `HIPBLAS_STATUS_INVALID_VALUE` from the hipBLASLt launch guard.
- A call with alpha = 0 at a bespoke solution's exact sizes returns `HIPBLAS_STATUS_INVALID_VALUE` on the model route too: the heuristic queries with alpha = 1 and picks the bespoke solution, while the launch treats alpha = 0 as K = 0, which the exact-size launch guard rejects. The stock library runs such a call. Reselecting a stock kernel when the guard rejects a heuristic pick is not implemented yet.
- hipBLASLt logs alpha/beta with six decimals, so a non-zero |beta| < 5e-7 is logged as `0.000000` and, without `--keep-beta`, is narrowed to `UseBeta: false`. The launch guard checks the real beta, so such a call is rejected (pinned index) or, once the heuristic selects with the real beta, not given the beta-off kernel; it never runs it with a wrong result.

---

## Module Reference

### `geko.bench` - Benchmarking Module

Handles benchmark execution and performance analysis.

**Key Functions:**
- `bench.run()` - Execute hipblaslt-bench with configuration
- `bench.compare()` - Compare reference vs tuned library performance
- `bench.log.parse()` - Parse hipBLASLt log files
- `bench.log.update()` - Update benchmark configuration (e.g. duration, flush, etc.)
- `bench.log.summarize()` - Analyze and filter GEMMs by contribution
- `bench.utils.parse_benchmark_output()` - Parse benchmark results

**Example Usage:**
```python
from geko import bench

# Parse hipBLASLt logs
df = bench.log.parse("hipblaslt-log-mask64.yaml", as_df=True)

# Summarize GEMM contributions
summary_df, unique_df = bench.log.summarize(
    hipblaslt_path="/path/to/rocm-libraries/projects/hipblaslt",
    log_file="hipblaslt-log-mask64.yaml",
    output_dir="workdir",
    keep_thr=0.1,
    device=0
)

# Compare libraries
results = bench.compare(
    hipblaslt_path="/path/to/rocm-libraries/projects/hipblaslt",
    lib_dir="optimized_libs",
    custom_lib_dir="build",
    benchmark_dir="benchmarks",
    device=0
)
```

---

### `geko.optim` - Optimization Module

Orchestrates multi-GPU kernel optimization.

**Key Functions:**
- `optim.configure()` - Generate Tensile configurations
- `optim.run()` - Execute multi-GPU optimization
- `optim.analyze()` - Benchmark and filter optimized kernels

**Example Usage:**
```python
from geko import optim
from geko.schemas import GemmType, GemmConfig

# Create GEMM configuration
gemm_type = GemmType(
    transA="T", transB="N",
    a_type="bf16_r", b_type="bf16_r",
    c_type="bf16_r", compute_type="f32_r"
)
gemm_config = GemmConfig(
    gemm_type=gemm_type,
    sizes=[[16032, 109, 1, 16384], [16032, 113, 1, 16384]]
)

# Generate configuration
config = optim.configure(
    hipblaslt_path="/path/to/rocm-libraries/projects/hipblaslt",
    gemm_configs=gemm_config,
    output_dir="workdir/optimizations",
    arch="gfx950"
)

# Run optimization
optim.run(
    hipblaslt_path="/path/to/rocm-libraries/projects/hipblaslt",
    tuning_dir="workdir/optimizations",
    devices=[0, 1, 2, 3],
    retry=True
)

# Analyze results
results = optim.analyze(
    hipblaslt_path="/path/to/rocm-libraries/projects/hipblaslt",
    lib_dir="workdir/optimizations",
    output_dir="workdir/results",
    up_thr=1.03,
    error_thr=0.03
)
```

---

### `geko.search` - Dense Search Module

Implements exhaustive kernel search workflow (offline tuning) with multi-GPU support.

**Key Functions:**
- `search.configure()` - Generate benchmark configurations from GEMM DataFrame
- `search.run()` - Execute dense search across multiple GPUs

**Example Usage:**
```python
from geko import search
import pandas as pd

# Create benchmark configurations from GEMMs
df = pd.DataFrame([
    {'M': 1024, 'N': 1024, 'K': 1024, 'batch_count': 1,
     'transA': 'N', 'transB': 'N', 
     'a_type': 'f16_r', 'b_type': 'f16_r',
     'c_type': 'f16_r', 'd_type': 'f16_r',
     'compute_type': 'f32_r', 'us': 125.5}
])

configs = search.configure(
    df=df,
    duration=0.04,  # Target 40ms benchmark duration
    iters=100,
    cold_iters=20,
    rotating=512
)

# Run dense search across GPUs
winners = search.run(
    hipblaslt_path="/path/to/rocm-libraries/projects/hipblaslt",
    configs=configs,
    output_dir="search_results",
    devices=[0, 1, 2, 3, 4, 5, 6, 7],
)
```

**Features:**
- **Multi-GPU Load Balancing**: Distributes benchmarks across devices
- **Progress Tracking**: Real-time tqdm progress bar with completion/failure counts
- **Cache Support**: Skips re-running completed benchmarks
- **Detailed Logging**: Creates `failed_jobs.log` for troubleshooting

**Workflow Integration:**
```python
from pathlib import Path
from geko import bench, search, optim, library

# 1. Parse and filter GEMMs
summary_df, uniq_df = bench.log.summarize(
    hipblaslt_path="/path/to/hipblaslt",
    log_file="hipblaslt-log-mask64.yaml",
    output_dir="workdir",
    keep_thr=0.1,
    device=0
)

# 2. Configure and run search
configs = search.configure(summary_df, duration=0.04)
winners = search.run(
    hipblaslt_path="/path/to/hipblaslt",
    configs=configs,
    output_dir="workdir/search",
    devices=[0, 1, 2, 3, 4, 5, 6, 7],
)

# 3. Extract winning solutions
match_table = Path("/path/to/hipblaslt/build/release/device-library/MatchTable.yaml")
libs = library.operations.extract_solutions(winners, match_table)
libs.dump("workdir/libs")

# 4. Analyze and filter
results = optim.analyze(
    hipblaslt_path="/path/to/hipblaslt",
    lib_dir="workdir/libs",
    output_dir="workdir/results",
    device=0,
    up_thr=1.03
)

# 5. Create final library
library.from_dataframe(results, "workdir/libs").dump("workdir/final_libs")
```

---

### `geko.library` - Library Management Module

Manages Tensile library loading, merging, and manipulation.

**Key Classes:**
- `Library` - Represents a single kernel library
- `LibraryCollection` - Collection of multiple libraries

**Key Functions:**
- `library.operations.load_library()` - Load YAML library file
- `library.operations.load_collection()` - Load directory of libraries
- `library.operations.merge_solutions()` - Merge optimized solutions
- `library.operations.extract_solutions()` - Extract solutions by solution index from a DataFrame
- `library.operations.create()` - Call TensileCreateLibrary
- `library.operations.from_dataframe()` - Create libraries from filtered results
- `library.operations.prune_library()` - Trim a library to its minimum required solutions, without losing performance


**Example Usage:**
```python
from geko import library

# Load single library
lib = library.operations.load_library("gfx950_Cijk_Alik_Bljk_BBS_TN.yaml")

# Load collection
collection = library.operations.load_collection("optimized_libs/")

# Remove duplicate sizes/solutions
collection.trim()

# Add epilogue support
collection.add_epilogues()

# Create benchmark inputs
collection.create_bench_input(
    output_dir="benchmarks",
    verify=True,
    duration=1.0,
    beta=True
)

# Merge solutions
merged = library.operations.merge_solutions(
    input_dir="workdir/optimizations",
    epilogues=True
)

# Filter and save
filtered = library.operations.from_dataframe(
    df=results_df,
    lib_dir="optimized_libs"
)
filtered.dump("final/")
```

---

### `geko.schemas` - Data Structures

Defines typed data structures with validation.

**Classes:**
- `GemmType` - GEMM operation specification
- `GemmConfig` - GEMM type with problem sizes
- `RunState` - Workflow state tracking

**Example Usage:**
```python
from geko.schemas import GemmType, GemmConfig

# Prefer factory constructors (they keep Tensile + hipBLASLt fields consistent).
gemm_type = GemmType.from_hipblaslt(
    "T", "N", "f16_r", "f16_r", "f16_r", "f32_r"
)
# Or from tensilelite YAML codes only:
# gemm_type = GemmType.from_tensile("N", "T", "H", "H", "S")

gemm_config = GemmConfig(
    gemm_type=gemm_type,
    sizes=[[1024, 1024, 1, 1024], [2048, 2048, 1, 2048]],
)
rows = gemm_config.workload_log_rows()  # hipBLASLt log-shaped dicts (GEMM_LOG_FIELDS)
```

---

### `geko.utils` - Utility Functions

Common utilities for the framework.

**Key Functions:** 
- `run_silent_command()` - Execute shell commands silently
- `build_tensilelite_client()` - Build the tensilelite client (cached by hipBLASLt git hash)
- `parse_devices()` - Parse a comma-separated device string into a list of device IDs

**Example Usage:**
```python
from geko.utils import (
    build_tensilelite_client,
    run_silent_command,
)

# Run command silently
run_silent_command(["ls", "-la", "/tmp"])

# Build tensilelite client
hipblaslt_path = "/path/to/rocm-libraries/projects/hipblaslt"
build_dir = "build_tmp"
build_tensilelite_client(hipblaslt_path, build_dir)
```

---

## Common Usage Patterns

### Pattern 1: Full Ductile-based Optimization Workflow

```python
from pathlib import Path
from geko import bench, optim, library
from geko.constants import GEMM_TYPE_FIELDS
from geko.schemas import GemmType, GemmConfig

# Paths
hipblaslt_path = "/path/to/rocm-libraries/projects/hipblaslt"
workdir = Path("workdir")
log_file = "hipblaslt-log-mask64.yaml"

# Phase 1: Configure
summary_df, unique_df = bench.log.summarize(
    hipblaslt_path,
    log_file,
    output_dir=workdir,
    keep_thr=0.1,
    device=0
)

# Generate configurations for each GEMM type
for gemm_type, gby in unique_df.groupby(list(GEMM_TYPE_FIELDS)):
  sizes = gby[["m", "n", "batch_count", "k"]].values.tolist()
  gemm_type = GemmType(**dict(zip(GEMM_TYPE_FIELDS, gemm_type)))
  optim.configure(
      hipblaslt_path,
      GemmConfig(gemm_type, sizes),
      workdir / "optimizations",
      arch="gfx950",
  )

# Phase 2: Optimize
optim.run(
    hipblaslt_path,
    tuning_dir=workdir / "optimizations",
    devices=[0, 1, 2, 3],
    retry=True
)

# Phase 3: Merge and benchmark
merged = library.operations.merge_solutions(
    workdir / "optimizations",
    epilogues=True
)
merged.dump(workdir / "merged_libs")

results_df = optim.analyze(
    hipblaslt_path,
    lib_dir=workdir / "merged_libs",
    output_dir=workdir / "results",
    up_thr=1.03,
    error_thr=0.03
)

# Phase 4: Create final library
final = library.operations.from_dataframe(
    results_df,
    lib_dir=workdir / "merged_libs"
)
final.dump("final/")
```

### Pattern 2: Full Dense Search Workflow

```python
from pathlib import Path
from geko import bench, search, optim, library

# Paths
hipblaslt_path = "/path/to/rocm-libraries/projects/hipblaslt"
workdir = Path("workdir")
log_file = "hipblaslt-log-mask64.yaml"

# Phase 1: Parse and filter GEMMs
summary_df, uniq_df = bench.log.summarize(
    hipblaslt_path,
    log_file,
    output_dir=workdir,
    keep_thr=0.1,
    device=0
)

# Phase 2: Configure and run search
configs = search.configure(summary_df, duration=0.04)
winners = search.run(
    hipblaslt_path,
    configs=configs,
    output_dir=workdir / "search",
    devices=[0, 1, 2, 3, 4, 5, 6, 7],
)

# Phase 3: Extract winning solutions
match_table = Path(hipblaslt_path) / "build/release/device-library/MatchTable.yaml"
libs = library.operations.extract_solutions(winners, match_table)
libs.dump(workdir / "libs")

# Phase 4: Analyze and filter
results = optim.analyze(
    hipblaslt_path,
    lib_dir=workdir / "libs",
    output_dir=workdir / "results",
    device=0,
    up_thr=1.03
)

# Phase 5: Create final library
library.from_dataframe(results, workdir / "libs").dump(workdir / "final_libs")
```

### Pattern 3: Custom Library Manipulation

```python
from geko import library

# Load and modify library
lib = library.operations.load_library("my_library.yaml")

# Add epilogue support
lib.add_epilogues()

# Create benchmark configurations
bench_file, verify_file = lib.create_bench_input(
    output_dir="benchmarks",
    verify=True,
    duration=2.0,
    beta=True,
    initialization="trig_float"
)

# Filter sizes based on custom logic
filtered_sizes = [size for size in lib.sizes if size[0][3] < 2048]  # Keep small K

lib.sizes = filtered_sizes
lib.trim() # Remove duplicate sizes, unused solutions

# Save modified library
lib.dump("modified_libs")
```

### Pattern 4: Custom Benchmarking 

```python
from geko import bench

# Parse existing benchmark results
results = bench.utils.parse_benchmark_output_dir(
    "benchmarks",
    suffix="_bench"
)

# Run custom benchmark
hipblaslt_path = "/path/to/rocm-libraries/projects/hipblaslt"
# Build custom library (third arg is the output directory; pass version= for code-object version)
library.operations.create(hipblaslt_path, "modified_libs", "build")

benchmark_df = bench.run(
    hipblaslt_path="/path/to/rocm-libraries/projects/hipblaslt",
    bench_file="custom_bench.yaml",
    output_file="results/custom_bench.out",
    custom_lib_dir="build", 
    device=0,
    cache=True
)

# Compare libraries
comparison = bench.compare(
    hipblaslt_path="/path/to/rocm-libraries/projects/hipblaslt",
    lib_dir="optimized_libs",
    custom_lib_dir="build",
    benchmark_dir="benchmarks",
    duration=1.0,
    device=0
)
```

---

## Troubleshooting

### Common Issues

#### 1. Configuration Failures
Common causes:
- Failure to benchmark a hipBLASLt log file.
- Invalid GEMM types.
- No solutions found for a given GEMM type and size.
- No tensilelite configurations are generated.

**Solutions:**

**Failure to Benchmark a hipBLASLt Log File:**
Ensure the log file has the correct format and required fields:
```bash
# Verify log file format
head -5 hipblaslt-log-mask64.yaml
```

The log must be a YAML file with GEMM entries containing required fields. Check `geko/constants.py` for `LOG_FIELDS` to see all allowed fields. Required fields include:
- `function`: Must be "matmul"
- `M`, `N`, `K`, `batch_count`: Matrix dimensions and batch size (positive integers)
- `transA`, `transB`: Transpose flags ("T" or "N")
- `a_type`, `b_type`, `c_type`, `d_type`: Data types
- `compute_type`: Computation type

**Example valid log entry:**
```yaml
- {function: matmul, M: 1024, N: 1024, K: 1024, transA: N, transB: N, 
   a_type: f16_r, b_type: f16_r, c_type: f16_r, d_type: f16_r, 
   compute_type: c_f32_r, batch_count: 1, alpha: 1.0, beta: 0.0}
```

If the benchmark fails, check for:
- Missing required fields
- Incorrect field names (case-sensitive)
- Invalid data types
- Malformed YAML syntax

**Invalid GEMM Types:**
Check the supported data types in hipBLASLt:
- Visit the [tensilelite DataType definitions](https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipblaslt/tensilelite/tensilelite/Common/DataType.py)
- Verify your log file contains valid `a_type`, `b_type`, `c_type`, and `compute_type` combinations
- Example valid types: `f16_r`, `bf16_r`, `f32_r`, `f8_r`, `bf8_r`

**Invalid GEMM Sizes:**
Ensure all GEMM dimensions are positive integers:
```bash
# Check your log file for invalid sizes
grep -E "M: [^0-9]|N: [^0-9]|K: [^0-9]" hipblaslt-log-mask64.yaml

# Valid example: M: 1024, N: 1024, K: 1024
# Invalid example: M: 0, N: -1, K: 1024.5
```

**No Solutions Found for GEMM Type/Size:**
If you encounter errors like "No solutions found":
- **Important GEMMs**: Contact the GEMM Optimization team for support
- **Non-critical GEMMs**: Remove the problematic GEMM from your hipBLASLt log file

**No tensilelite Configurations Generated:**
If `scripts/configure.py` produces no configuration files:
```bash
# Lower the keep threshold to include more GEMMs
python scripts/configure.py /path/to/hipblaslt hipblaslt-log-mask64.yaml \
  --keep_thr 0.01  # Include GEMMs contributing >= 0.01% of total time

# Or set to 0 to include all GEMMs
python scripts/configure.py /path/to/hipblaslt hipblaslt-log-mask64.yaml --keep_thr 0
```

#### 2. Optimization Failures 
Check `workdir/optimizations/failed_jobs.log` for details.

Common causes:
- Unstable rocm-libraries version, leading to assembly compile errors.
- No available AMD GPU devices.
- No valid kernels could be generated. Re-visit the configuration step.

**Solutions:**
```bash
# To not retry failed optimizations
python scripts/optimize.py /path/to/hipblaslt --workdir workdir --no_retry

# Use the appropriate devices or check if your GPUs are available (amd-smi command)
python scripts/optimize.py /path/to/hipblaslt --devices=0,1

# Check individual logs for detailed errors
cat workdir/optimizations/build_*/BBS_TN_0-tensilelite.log
```

#### 3. Import Errors
```bash
# Ensure GEKO is in Python path
export PYTHONPATH=/path/to/geko:$PYTHONPATH

# Or install the package

pip install .

# Or install in development mode
pip install -e .
```


### Debug Mode

Enable verbose logging:
```bash
# Maximum verbosity
python scripts/configure.py /path/to/hipblaslt log.yaml -v 2
python scripts/optimize.py /path/to/hipblaslt -v 2
python scripts/search.py /path/to/hipblaslt log.yaml -v 2

# Python script
import logging
logging.getLogger("GEKO").setLevel(logging.DEBUG)
```

### Performance Tips

1. **Use appropriate keep_thr**: Higher threshold = fewer GEMMs = faster optimization
   ```bash
   python scripts/configure.py ... --keep_thr 1.0  # Only with 1% contribution or higher
   ```

2. **Optimize GPU allocation**: Match device count to available GPUs
   ```bash
   python scripts/optimize.py ... --devices=0,1  # Use only 2 GPUs
   ```

3. **Enable caching**: Reuse benchmark results
   ```python
   bench.compare(..., cache=True)
   ```

4. **Parallel library dumps**: Automatically parallelized in `LibraryCollection.dump()`

---

## Contributing

To contribute to our repository, you can create a GitHub pull request. Please follow these guidelines:

1. Follow Google-style docstrings
2. Add type hints to all functions
3. Use descriptive variable names
4. Include examples in module docstrings
5. Test on multiple GPU architectures


If you want to submit an issue, you can do so on
[GitHub](https://github.com/ROCm/rocm-libraries/issues).

---

## License

MIT License. Copyright (C) Advanced Micro Devices, Inc.
