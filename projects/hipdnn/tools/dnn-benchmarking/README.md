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
pip install -e .

# Install hipDNN Python bindings (from your hipDNN build)
cd /path/to/hipdnn/python && pip install -e . && cd -
```

### For CUDA/NVIDIA GPUs (PyTorch CUDA benchmarking)

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install CUDA torch + PyPI deps
pip install -r requirements-cuda.txt
pip install -e .
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

```bash
# Run benchmark on a serialized graph
python -m dnn_benchmarking --graph ./graphs/conv1_fwd.json --warmup 10 --iters 100

# With custom engine ID
python -m dnn_benchmarking --graph ./graphs/conv1_fwd.json --engine-id 1

# With reproducible random seed
python -m dnn_benchmarking --graph ./graphs/conv1_fwd.json --seed 42
```

### A/B Testing

Compare two different plugin/engine configurations and validate accuracy:

```bash
# Compare two different engines on the default plugin
python -m dnn_benchmarking --graph ./graphs/conv1_fwd.json --AId 1 --BId 2

# Compare two different plugins with specific engine IDs
python -m dnn_benchmarking --graph ./graphs/conv1_fwd.json \
  --APath /path/to/pluginA --AId 1 \
  --BPath /path/to/pluginB --BId 2

# With custom tolerance for accuracy comparison
python -m dnn_benchmarking --graph ./graphs/conv1_fwd.json \
  --AId 1 --BId 2 --rtol 1e-3 --atol 1e-6
```

### CLI Options

#### Basic Options

| Option | Description | Default |
|--------|-------------|---------|
| `--graph`, `-g` | Path to JSON-serialized hipDNN graph file | Required |
| `--warmup`, `-w` | Number of warmup iterations | 10 |
| `--iters`, `-i` | Number of benchmark iterations | 100 |
| `--engine-id`, `-e` | Engine ID (1 = MIOpen) | 1 |
| `--seed` | Random seed for reproducibility | None |
| `--gpu-backend` | GPU kernel timer backend (`torch`, `auto`, `none`) | auto |

#### A/B Testing Options

| Option | Description | Default |
|--------|-------------|---------|
| `--APath` | Plugin path for configuration A | None (default) |
| `--AId` | Engine ID for configuration A | Required for A/B |
| `--BPath` | Plugin path for configuration B | None (default) |
| `--BId` | Engine ID for configuration B | Required for A/B |
| `--rtol` | Relative tolerance for accuracy comparison | 1e-5 |
| `--atol` | Absolute tolerance for accuracy comparison | 1e-8 |

**Note**: A/B testing mode is enabled when both `--AId` and `--BId` are specified.

## Output

### Basic Benchmark Output

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

Execution Statistics:
  Mean:                 1.234 ms
  Std Dev:              0.045 ms
  Min:                  1.156 ms
  Max:                  1.456 ms
  P95:                  1.312 ms
  P99:                  1.398 ms

Validation: SKIPPED (CPU reference not available)
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
