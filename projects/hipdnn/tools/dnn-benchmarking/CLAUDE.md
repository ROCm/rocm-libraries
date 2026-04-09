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
cd /path/to/hipdnn/python && pip install -e .
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

```bash
# Basic benchmark (single graph, single engine)
python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json --warmup 10 --iters 100

# A/B testing (compare two engine configurations)
python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json --AId 1 --BId 2

# Suite mode (multiple graphs, all providers/engines)
# Triggered automatically when --graph resolves to multiple files
python -m dnn_benchmarking --graph 'graphs/*.json' --warmup 10 --iters 100

# Suite mode with JSON output
python -m dnn_benchmarking --graph 'graphs/*.json' --output results.json

# Suite mode with provider/engine filters
python -m dnn_benchmarking --graph 'graphs/*.json' --provider miopen --engine 1

# Suite mode also activates with --provider/--engine on a single graph
python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json --provider miopen
```

## Architecture

```
src/dnn_benchmarking/
├── cli/              # Entry point (main.py, parser.py)
├── config/           # BenchmarkConfig, ABTestConfig, SuiteConfig dataclasses
├── execution/        # executor.py, buffer_manager.py, ab_runner.py, suite_runner.py, timing.py
├── graph/            # loader.py (JSON loading), validator.py, tensor_info.py
├── reporting/        # reporter.py (console output), statistics.py, suite_results.py
└── validation/       # validator.py, comparison.py, reference_provider.py
```

**Data flow (single graph):** CLI → Config → GraphLoader → Executor → BufferManager → Timing → BenchmarkStats → Reporter

**Data flow (suite mode):** CLI → SuiteConfig → GraphLoader (per graph) → suite_runner.run_graph_all_providers → Executor (per provider/engine) → BufferManager → Timing + Correctness → SuiteResult → JSON/Reporter

**Key external dependency:** `hipdnn_frontend` - AMD's hipDNN Python bindings (requires AMD GPU + ROCm).

## Exit Codes

- 0: Success (all pass)
- 1: Error (graph load, execution, configuration)
- 2: Correctness failure (A/B comparison mismatch or suite tolerance_match failure)

<!-- GSD:project-start source:PROJECT.md -->
## Project

**hipDNN Automated Benchmarking System**

An automated performance benchmarking system for hipDNN that runs curated graph suites against all available providers/engines, captures timing and correctness results, and integrates into CI via GitHub Actions. Built on top of the existing `dnn-benchmarking` tool, extending it from a single-graph CLI into a suite-based automation system.

**Core Value:** Reliable, automated detection of performance regressions and correctness failures across hipDNN providers — run weekly, results available without manual effort.

### Constraints

- **Sequential execution:** Graphs must run one at a time on a single GPU — no parallel batch execution
- **Platform:** AMD GPU + ROCm required for hipDNN execution; PyTorch ROCm nightly builds
- **Dependencies:** `hipdnn_frontend` must be installed separately from hipDNN build
- **CI runner:** Self-hosted with GPU access; GH Actions artifacts for result storage (ephemeral)
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- Python 3.9+ - Core benchmarking tool (src/dnn_benchmarking/)
## Runtime
- CPython 3.9, 3.10, 3.11, 3.12, 3.13
- pip (setuptools for package build)
- Lockfile: Not present (uses requirements.txt files instead)
## Frameworks & Core Dependencies
- numpy >=1.19.0 - Numerical computation for tensor manipulation
- torch (PyTorch) - GPU execution, validation, and timing
- ROCm (AMD GPU): `requirements-rocm.txt` installs PyTorch from `rocm.nightlies.amd.com/v2-staging` (nightlies, pre-release builds)
- CUDA (NVIDIA GPU): `requirements-cuda.txt` installs PyTorch from official PyTorch wheels (cu124, 2.0+)
- pytest >=7.0.0 - Test runner
- pytest-cov >=4.0.0 - Coverage reporting
- black >=23.0.0 - Code formatter
- ruff >=0.0.260 - Fast linter
- mypy >=1.0.0 - Type checker
## External Dependencies
- `hipdnn_frontend` - AMD's hipDNN Python bindings
## Configuration
- `pyproject.toml` - PEP 517/518 build config with setuptools
- Entry point: `dnn_benchmarking.cli.main:main`
- `requirements.txt` - Base dependencies (numpy only, hipdnn_frontend noted as manual install)
- `requirements-rocm.txt` - ROCm PyTorch nightly index configuration
- `requirements-cuda.txt` - CUDA PyTorch from official index
- `requirements-dev.txt` - Development tools
## Platform Requirements
- Python 3.9+ with pip
- For ROCm testing: AMD GPU + ROCm SDK + nightly PyTorch wheels available
- For CUDA testing: NVIDIA GPU + CUDA 12.4 compatible PyTorch
- For hipDNN execution: Compiled hipDNN with Python bindings
- Python runtime environment
- AMD GPU for hipDNN mode (requires hipDNN bindings)
- NVIDIA GPU optional (PyTorch CUDA backend for reference validation)
- ROCm or CUDA runtime libraries
## Key Data Types & Serialization
- JSON (hipDNN serialized graphs) - loaded via `json.load()` in `src/dnn_benchmarking/graph/loader.py`
- No schema validation beyond operation type checking
- Float32, Float64, Float16, BFloat16, Int8, Int32, UInt8, Int64, Boolean
- Mapping defined in `src/dnn_benchmarking/execution/executor.py` (_DATA_TYPE_STR_MAP)
- BFloat16 special handling via torch.bfloat16 in `src/dnn_benchmarking/execution/buffer_manager.py`
## GPU Timing Implementation
- Uses `torch.cuda.Event` for GPU kernel timing on default stream
- Explicitly records on `torch.cuda.default_stream()` to capture hipDNN kernels
- Supports both CUDA and ROCm via unified PyTorch API
- Fallback: CPU timing with `time.perf_counter()` if GPU timing unavailable
- Module: `src/dnn_benchmarking/execution/timing.py`
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- Sequential data flow from CLI → GraphLoader → Executor → Timing → Reporter
- Provider-based validation (runtime registration via decorators)
- Support for multiple execution backends (hipDNN native, PyTorch CUDA)
- Dual-timing mode: E2E wall-clock + optional GPU kernel timing
- A/B testing mode for configuration comparison
## Layers
- Purpose: Parse arguments, route to appropriate workflow (benchmark, A/B test, PyTorch)
- Location: `src/dnn_benchmarking/cli/`
- Contains: ArgumentParser, main entry point, workflow orchestration
- Depends on: Config, GraphLoader, Executor, Reporter, validation
- Used by: Python entry point `dnn-benchmark` or `python -m dnn_benchmarking`
- Purpose: Dataclass validation and configuration storage
- Location: `src/dnn_benchmarking/config/`
- Contains: BenchmarkConfig, ABTestConfig, ValidationConfig (all in benchmark_config.py)
- Depends on: Standard library only (pathlib, dataclasses)
- Used by: CLI, Executor, Reporter
- Purpose: Load, parse, and validate hipDNN graph JSON files
- Location: `src/dnn_benchmarking/graph/`
- Contains: GraphLoader (file I/O and JSON parsing), GraphValidator, TensorInfo
- Depends on: json, pathlib, custom exceptions
- Used by: CLI, main execution workflows, A/B runner
- Purpose: Build and execute hipDNN graphs, manage buffers, coordinate timing
- Location: `src/dnn_benchmarking/execution/`
- Contains:
- Depends on: hipdnn_frontend, torch, numpy
- Used by: CLI, validation
- Purpose: Compute reference outputs and compare with GPU results
- Location: `src/dnn_benchmarking/validation/`
- Contains:
- Depends on: torch (for PyTorch provider), numpy
- Used by: CLI main.py (run_reference_validation), A/B runner
- Purpose: Format and display benchmark results
- Location: `src/dnn_benchmarking/reporting/`
- Contains: Reporter (console formatting), BenchmarkStats (statistics calculation)
- Depends on: Standard library (sys, pathlib, dataclasses, json)
- Used by: CLI entry points for all workflows
- Purpose: Shared exception types
- Location: `src/dnn_benchmarking/common/`
- Contains: GraphLoadError, ExecutionError, ValidationError
- Used by: All layers for error signaling
## Data Flow
## State Management
- Executor instances are not reused between benchmark runs
- Graph JSON is immutable after loading
- Buffer allocation/deallocation per benchmark run (context manager pattern)
- No global state except provider registry
- Executor._graph, ._workspace (built in prepare())
- BufferManager._buffers dict (device pointers)
- BufferManager._host_data dict (NumPy arrays for CPU access)
- Timing accumulators in BenchmarkResult (list of elapsed_ms values)
## Key Abstractions
- Purpose: Abstract GPU timing backend
- Examples: TorchGpuTimer (PyTorch CUDA/ROCm)
- Pattern: Context manager + explicit start()/stop(), elapsed_ms()
- Implementations record on torch.cuda.default_stream() for hipDNN kernel capture
- Purpose: Abstract reference computation backend
- Examples: PyTorchReferenceProvider
- Pattern: Decorator-based registry (register decorator), interface via compute_reference()
- Enables pluggable validation backends without modifying core code
- Purpose: Manages device memory lifecycle
- Pattern: Context manager (__enter__/__exit__) for cleanup
- Responsibility: Allocation, filling, packing into variant_pack, retrieval
- Purpose: Builds and runs hipDNN graphs
- Two variants:
- Pattern: prepare() then benchmark() with reusable handle and variant_pack
## Entry Points
- Location: `src/dnn_benchmarking/cli/main.py:main()`
- Triggers: ArgumentParser, routes to run_benchmark/run_ab_test/run_pytorch_benchmark
- Responsibilities: Argument parsing, error handling, exit code management
- `run_benchmark(config, seed, validation_config, output_path, gpu_backend)` - Main benchmark
- `run_ab_test(config, ab_config, seed, gpu_backend, validation_config)` - A/B comparison
- `run_pytorch_benchmark(config, seed, output_path, device)` - PyTorch CUDA backend
## Error Handling
- GraphLoadError - File I/O, JSON parsing (graph/loader.py)
- ExecutionError - Graph building, kernel launch failures (execution/executor.py)
- ValidationError - Validation logic failures (validation/)
- ValueError - Configuration validation (config/)
- GraphLoadError → print error, return 1
- ExecutionError → print error, return 1
- ValueError (config) → print to stderr, return 1
- Generic Exception → "Unexpected error", return 1
- Validation failure → return 2 (distinct from errors)
## Cross-Cutting Concerns
- Graph validation: GraphValidator.validate_conv_fwd_only() (graph/validator.py)
- Configuration validation: BenchmarkConfig.__post_init__(), ValidationConfig.__post_init__()
- Reference validation: Optional, via ReferenceProvider (validation/)
- Mapping: _DATA_TYPE_STR_MAP in executor.py (string → hipdnn.DataType enum)
- BFloat16 special case: torch.bfloat16 ↔ bytes via untyped_storage (buffer_manager.py)
- NumPy dtype map in buffer_manager.py (string → np.dtype)
- Optional seed parameter in all benchmark functions
- Used by BufferManager.fill_inputs_random(seed) to create reproducible inputs
- np.random.RandomState(seed) for determinism
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, or `.github/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
