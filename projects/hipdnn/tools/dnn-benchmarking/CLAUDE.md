# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Benchmarking and validation tool for hipDNN graphs. Loads JSON-serialized hipDNN graphs, executes them via the MIOpen plugin, captures performance metrics, and supports A/B testing between different plugin/engine configurations.

## Build and Development Commands

```bash
# Full setup (venv, requirements, hipDNN bindings) — skips hipDNN/provider build if already installed
./setup.sh

# Full setup AND build hipDNN + MIOpen provider from source (overwrites existing artifacts)
./setup.sh --force-build

# Manual setup for ROCm/AMD GPU development
pip install -r requirements-rocm.txt   # torch from ROCm nightly index
pip install -e .                        # package + PyPI deps (numpy, pytest)

# hipDNN bindings must be installed separately from your hipDNN build
cd /path/to/hipdnn/python && pip install -e . --no-deps
```

`--force-build` installs hipDNN and the MIOpen plugin to `/opt/rocm` (prompts for confirmation).
Pass `/opt/rocm/lib/hipdnn_plugins/engines/` to `--plugin-path` when running benchmarks.

### ROCm PyTorch Setup

The `requirements-rocm.txt` installs PyTorch from ROCm nightly indexes. The correct index depends on GPU architecture:

| GPU | Architecture | Index |
|-----|-------------|-------|
| MI200/MI210/MI250 | gfx90X | `v2-staging/gfx90X-dcgpu` |
| MI300X/MI300A | gfx94X | `v2/gfx94X-dcgpu` |

To switch architectures, change the `--index-url` line in `requirements-rocm.txt`.

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

Test markers: `gpu` (requires GPU), `slow` (slow integration tests).

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

# A/B testing (separate path, kept for now)
python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json --AId 1 --BId 2

# PyTorch backend (separate executor; single graph only)
python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json --backend pytorch
```

## Architecture

```
src/dnn_benchmarking/
├── cli/              # Entry point (main.py, parser.py)
├── common/           # Shared utilities (exceptions.py)
├── config/           # BenchmarkConfig, ABTestConfig, SuiteConfig dataclasses
├── execution/        # executor.py, buffer_manager.py, ab_runner.py,
│                     # suite_runner.py, timing.py,
│                     # pytorch_executor.py, pytorch_buffer_manager.py, pytorch_ops.py
├── graph/            # loader.py (JSON loading), validator.py, tensor_info.py
├── reporting/        # reporter.py (console output), statistics.py, suite_results.py
└── validation/       # validator.py, comparison.py, reference_provider.py
    └── providers/    # cpu_plugin_provider.py, pytorch_provider.py
```

**Data flow (single graph):** CLI → Config → GraphLoader → Executor → BufferManager → Timing → BenchmarkStats → Reporter

**Data flow (suite mode):** CLI → SuiteConfig → GraphLoader (per graph) → suite_runner.run_graph_all_providers → Executor (per provider/engine) → BufferManager → Timing + Correctness → SuiteResult → JSON/Reporter

**Key external dependency:** `hipdnn_frontend` - AMD's hipDNN Python bindings (requires AMD GPU + ROCm).

## Exit Codes

- 0: Success (all pass)
- 1: Error (graph load, execution, configuration)
- 2: Correctness failure (A/B comparison mismatch or suite tolerance_match failure)
