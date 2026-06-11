# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Benchmarking and validation tool for hipDNN graphs. Loads JSON-serialized hipDNN graphs, executes them via hipDNN engine plugins, captures performance metrics, and supports explicit multi-engine comparison.

A `--backend pytorch` executor runs the same graphs through PyTorch (ROCm or CUDA). It shares the suite execution path and `SuiteResult` JSON schema with the hipDNN backend, so a CUDA host (no hipDNN/ROCm) can produce comparable JSON for offline ROCm-vs-CUDA comparison. The tool stays importable without `hipdnn_frontend`: the HIP timer imports it lazily, and every other hipDNN touchpoint is already lazy.

## Build and Development Commands

```bash
# Full setup (venv, requirements, hipDNN bindings) — skips hipDNN/provider build if already installed
./setup.sh

# Full setup AND build hipDNN + MIOpen provider from source (overwrites existing artifacts)
./setup.sh --force-build

# Manual setup for ROCm/AMD GPU development (gfx90X or gfx94X)
pip install --pre torch --index-url https://rocm.nightlies.amd.com/v2-staging/gfx94X-dcgpu/
pip install -e .                              # package + PyPI deps (numpy, pytest)

# hipDNN bindings must be installed separately from your hipDNN build
cd /path/to/hipdnn/python && pip install -e . --no-deps

# CUDA host (NVIDIA): PyTorch backend only, skips all ROCm/hipDNN setup
./setup.sh --torch-mode cuda --workspace .workspace
```

`--force-build` installs hipDNN and the MIOpen plugin to `/opt/rocm` (prompts for confirmation).
Pass `/opt/rocm/lib/hipdnn_plugins/engines/` to `--plugin-path` when running benchmarks.

`--torch-mode cuda` installs a CUDA PyTorch wheel (from PyPI, or `--torch-index-url`) and skips hipDNN/provider builds, hipDNN bindings, amdsmi, and ROCm env wiring. `--force-build` is rejected in this mode. Only `--backend pytorch` works on CUDA hosts.

### ROCm PyTorch Setup

`setup.sh` auto-detects the GPU architecture and installs PyTorch from the matching ROCm nightly index. Supported architectures:

| GPU | `--gpu-arch` | Torch index bucket |
|-----|--------------|--------------------|
| MI200/MI210/MI250 | `gfx90a` | `gfx90X-dcgpu` |
| MI300X/MI300A | `gfx942` | `gfx94X-dcgpu` |
| MI350 | `gfx950` | `gfx950-dcgpu` |

## Running Tests

```bash
# All non-GPU tests (no hipDNN required)
pytest -m "not gpu"

# All tests including GPU tests (requires hipDNN and ROCm libraries)
LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH pytest

# Single test file
LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH pytest tests/unit/execution/test_timing.py

# With coverage
LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH pytest --cov=dnn_benchmarking tests/
```

**Note:** GPU tests require ROCm libraries to be findable. Set `LD_LIBRARY_PATH=/opt/rocm/lib` before running tests that depend on `hipdnn_frontend`.

### Test tiers and platform selection

Tests fall into three tiers, distinguished by markers:

- **Unit** (`tests/unit/`, unmarked): fake torch, run on any host with no GPU.
- **GPU-generic** (`gpu`): run on any live GPU (ROCm or CUDA). Platform-
  specific assertions use `expected_timing_backend()` so the same test
  validates HIP timing on ROCm and torch.cuda timing on CUDA.
- **Platform-specific** (`gpu` + `rocm`, or `gpu` + `cuda`): assert
  behavior unique to one backend (e.g. the selected timing backend).

Every GPU test also self-skips on the wrong platform, so a bare `pytest`
run is safe anywhere — the foreign-platform tests skip with a clear
reason. Use `-m` for explicit, additive selection (no custom flag):

```bash
# On a ROCm host: unit + generic + rocm (drops cuda-only tests)
LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH pytest -m "not cuda"

# On a CUDA host: unit + generic + cuda (drops rocm-only tests)
pytest -m "not rocm"

# Only one platform's dedicated tests
pytest -m rocm
pytest -m cuda
```

Test markers: `gpu` (requires any GPU), `rocm` (requires AMD ROCm GPU),
`cuda` (requires NVIDIA CUDA GPU), `slow` (slow integration tests).

Strict profiling tests that require real profiler artifacts are skipped by
default. Run them explicitly on a known-good profiling host:

```bash
LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH pytest --profiling-strict -m profiling_strict
```

## Running the Tool

Single-graph and multi-graph runs share one execution path. Default output is a
summary table; `-v` switches to a rich per-engine block per graph (matches the
legacy single-graph format).

```bash
# Single graph (default summary table, all discovered engines)
python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json --warmup 10 --iters 100

# Single graph, verbose per-engine block
python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json -v

# Filter to one or more engine IDs (comma-separated)
python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json --engine 1
python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json --engine 1,2

# Multiple graphs via glob — same path, more rows
python -m dnn_benchmarking --graph 'graphs/*.json' --warmup 10 --iters 100

# Multi-graph with JSON output (full SuiteResult, independent of -v)
python -m dnn_benchmarking --graph 'graphs/*.json' --output results.json

# Reference validation against a PyTorch reference implementation
python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json --validate pytorch

# Reference validation with custom tolerances
python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json \
  --validate pytorch --rtol 1e-3 --atol 1e-6

# Point at a directory of plugin .so files for engine discovery
python -m dnn_benchmarking --graph 'graphs/*.json' \
  --plugin-path /path/to/hipdnn/plugins --output results.json

# Repeatable recipe from TOML config
python -m dnn_benchmarking --config sample_configs/basic.toml.example --graph ./graphs/sample_conv_fwd.json

# Engine comparison
python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json --engine 1,2

# PyTorch backend (ROCm or CUDA); shares the suite path, supports single + multi graph
python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json --backend pytorch
python -m dnn_benchmarking --graph 'graphs/*.json' --backend pytorch -o pytorch_results.json
```

`--backend pytorch` rejects `--engine`, `--plugin-path`, `--validate pytorch`, and the profiling flags (`--pmc`, `--emit-trace`, `--perf`, `--roofline`).

## Architecture

```
src/dnn_benchmarking/
├── cli/              # Entry point, parser, TOML config loading
├── common/           # Shared utilities (exceptions.py)
├── config/           # BenchmarkConfig, MetricsConfig, SuiteConfig dataclasses
├── execution/        # executor.py, buffer_manager.py, suite_runner.py,
│                     # timing.py, pytorch_executor.py,
│                     # pytorch_buffer_manager.py, pytorch_ops.py
├── graph/            # loader.py (JSON loading), validator.py, tensor_info.py
├── reporting/        # reporter.py (console output), statistics.py, suite_results.py
└── validation/       # validator.py, comparison.py, reference_provider.py
    └── providers/    # pytorch_provider.py
```

**Data flow (hipDNN backend):** CLI → SuiteConfig → GraphLoader (per graph) → suite_runner.run_graph_all_providers → Executor (per provider/engine) → BufferManager → Timing + Correctness → SuiteResult → JSON/Reporter

**Data flow (PyTorch backend):** CLI → SuiteConfig (backend="pytorch") → GraphLoader (per graph) → suite_runner.run_graph_pytorch_backend → PyTorchCudaExecutor → PyTorchCudaBufferManager → Timing → SuiteResult → JSON/Reporter. One `provider="pytorch"` row per graph; no engine discovery or plugins.

**Key external dependency:** `hipdnn_frontend` - AMD's hipDNN Python bindings (requires AMD GPU + ROCm). Required for the hipDNN backend only; imported lazily so the tool runs the PyTorch backend on CUDA hosts without it.

## Exit Codes

- 0: Success (all pass)
- 1: Error (graph load, execution, configuration)
- 2: Correctness failure (suite tolerance_match failure)
