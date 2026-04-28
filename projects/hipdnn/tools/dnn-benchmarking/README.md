# dnn-benchmarking

Benchmarking and validation tool for hipDNN graphs.

## Overview

> **Caution**: This tool is in early development and subject to change.
> Do not use it in build workflows or CI pipelines.

This tool loads serialized hipDNN graphs, executes them via the MIOpen plugin, and captures performance metrics using PyTorch CUDA/ROCm events for kernel timing.

## Requirements

- Python 3.9+
- numpy
- PyTorch (any variant; ROCm or CUDA build required for GPU kernel timing)
- hipdnn_frontend (installed hipDNN Python bindings)
- AMD GPU with ROCm + MIOpen plugin

## Installation

### For ROCm/AMD GPUs (hipDNN benchmarking)

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install ROCm torch (from ROCm nightly index), then the package + PyPI deps.
pip install -r requirements-rocm.txt
pip install -e . --no-deps

# Install hipDNN Python bindings (from your hipDNN build)
cd /path/to/hipdnn/python && pip install -e . --no-deps && cd -
```

### For CUDA/NVIDIA GPUs (PyTorch CUDA benchmarking)

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install CUDA torch + PyPI deps
pip install -r requirements-cuda.txt
pip install -e . --no-deps
```

### Development Installation

```bash
# For ROCm development
pip install -r requirements-rocm.txt
pip install -e .

# For CUDA development
pip install -r requirements-cuda.txt
pip install -e .
```

**Note**: hipDNN Python bindings (`hipdnn_frontend`) must be installed separately for hipDNN benchmarking.
**Note**: ROCm PyTorch is required for GPU kernel timing on AMD; validation remains CPU-only.

## Usage

### Basic Benchmarking

A single graph and a glob of graphs share the same execution path. By default
results are printed as a summary table. Use `-v` for the rich per-engine block
(useful for debugging a single graph or comparing engines).

```bash
# Single graph (default summary output)
python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json --warmup 10 --iters 100

# Single graph, verbose: rich per-engine block
python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json -v

# Filter to specific engine(s) — comma-separated
python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json --engine 1
python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json --engine 1,2

# Multiple graphs (glob): same path, default summary table
python -m dnn_benchmarking --graph 'graphs/*.json' --warmup 10 --iters 100

# With reproducible random seed
python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json --seed 42
```

### A/B Testing

Compare two different plugin/engine configurations and validate accuracy:

```bash
# Compare two different engines on the default plugin
python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json --AId 1 --BId 2

# Compare two different plugins with specific engine IDs
python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json \
  --APath /path/to/pluginA --AId 1 \
  --BPath /path/to/pluginB --BId 2

# With custom tolerance for accuracy comparison
python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json \
  --AId 1 --BId 2 --rtol 1e-3 --atol 1e-6
```

### CLI Options

#### Basic Options

| Option | Description | Default |
|--------|-------------|---------|
| `--graph`, `-g` | Path to JSON-serialized hipDNN graph file, or glob pattern | Required |
| `--warmup`, `-w` | Number of warmup iterations | 10 |
| `--iters`, `-i` | Number of benchmark iterations | 100 |
| `--engine`, `-e` | Engine ID or comma-separated list (e.g. `1` or `1,2,3`); default = all discovered engines | None |
| `--seed`, `-s` | Random seed for reproducible input data | None |
| `--backend`, `-b` | Execution backend: `hipdnn` (AMD GPU via hipDNN) or `pytorch` (GPU via PyTorch) | `hipdnn` |

#### Output Options

| Option | Description | Default |
|--------|-------------|---------|
| `--output`, `-o` | Export benchmark results to JSON file (full SuiteResult; independent of `-v`) | None |
| `--verbose`, `-v` | Show detailed per-engine block per graph (default: summary table) | False |
| `--no-kernel-timing` | Disable GPU kernel timing (E2E wall-clock only) | False |

#### Reference Validation Options

| Option | Description | Default |
|--------|-------------|---------|
| `--validate` | Reference provider for correctness validation: `pytorch`, `cpu_plugin`, or `none` | `none` |

#### Suite Options

| Option | Description | Default |
|--------|-------------|---------|
| `--plugin-path` | Path to directory containing hipDNN engine plugin `.so` files | None (system default) |

#### A/B Testing Options

| Option | Description | Default |
|--------|-------------|---------|
| `--APath` | Plugin path for configuration A | None (default) |
| `--AId` | Engine ID for configuration A | Required for A/B |
| `--BPath` | Plugin path for configuration B | None (default) |
| `--BId` | Engine ID for configuration B | Required for A/B |

**Note**: A/B testing mode is enabled when both `--AId` and `--BId` are specified.

#### Comparison Options

Used by A/B testing, reference validation, and suite-mode tolerance checks.

| Option | Description | Default |
|--------|-------------|---------|
| `--rtol` | Relative tolerance for output comparison | 1e-5 |
| `--atol` | Absolute tolerance for output comparison | 1e-8 |

## Output

### Default Output (summary table)

The default console output is a compact, suite-style summary. One line per
graph reports the per-engine pass/fail counts, followed by a final summary
block. JSON output (`--output`) always contains the full per-engine
`SuiteResult` regardless of console verbosity.

```
================================================================================
hipDNN Benchmark Suite: 3 graph(s)
================================================================================

[1/3] sample_conv_fwd...
  -> 2 passed, 0 failed, 0 skipped, 0 errored
[2/3] sample_matmul...
  -> 2 passed, 0 failed, 0 skipped, 0 errored
[3/3] sample_relu...
  -> 1 passed, 1 failed, 0 skipped, 0 errored

--------------------------------------------------------------------------------
Suite Summary:
  Graphs:       3
  Combinations: 6
  Passed:       5
  Failed:       1
  Skipped:      0
  Errors:       0
================================================================================
```

### Verbose Output (`-v`)

`-v` switches to a rich per-engine block per graph (matches the legacy
single-graph format). Useful when debugging a single graph or comparing engines
side-by-side.

```
================================================================================
hipDNN Benchmark: sample_conv_fwd_16x16x16x16_k16_3x3
================================================================================
Graph:      ./graphs/sample_conv_fwd.json
Engine ID:  1 (MIOpen)
Warmup:     10 iterations
Benchmark:  100 iterations
--------------------------------------------------------------------------------

Initialization:
  Graph build time:     45.23 ms

E2E Execution Statistics:
  Mean:                 1.234 ms
  Std Dev:              0.045 ms
  Min:                  1.156 ms
  Max:                  1.456 ms
  P95:                  1.312 ms
  P99:                  1.398 ms

Kernel Execution Statistics:
  Mean:                 0.872 ms
  Std Dev:              0.012 ms
  Min:                  0.851 ms
  Max:                  0.921 ms
  P95:                  0.897 ms
  P99:                  0.910 ms

Reference Validation: SKIPPED (no reference comparison performed)
  Provider: none
================================================================================
```

## Running Tests

### Quick Start

```bash
# Activate venv
source .venv/bin/activate

# All non-GPU tests (no hipDNN required)
pytest -m "not gpu"

# All tests including GPU (requires hipDNN bindings and ROCm libraries)
LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH pytest

# Only GPU tests
LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH pytest -m gpu
```

### GPU Tests

GPU tests require hipDNN Python bindings and ROCm libraries:

```bash
source .venv/bin/activate
export CMAKE_PREFIX_PATH=/path/to/hipdnn/build/lib/cmake
cd /path/to/hipdnn/python && pip install -e .
cd -

# Run tests with ROCm libraries available
LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH pytest
```

**Note:** Set `LD_LIBRARY_PATH=/opt/rocm/lib` when running GPU tests to ensure hipdnn_frontend can load ROCm libraries.

## Limitations

- CPU reference validation is not yet implemented (CPU reference plugin not yet available in Python bindings)
- A/B testing uses `np.allclose()` for accuracy comparison between configurations
