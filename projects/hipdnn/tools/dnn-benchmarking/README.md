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

### Quick Setup (ROCm/AMD GPUs)

Run the provided setup script from the `dnn-benchmarking` directory:

```bash
bash setup.sh
source /workspace/.venv/bin/activate  # or $DNN_BENCH_WORKSPACE/.venv/bin/activate
```

This script handles everything automatically:
1. Creates a virtual environment under `$DNN_BENCH_WORKSPACE` (defaults to `/workspace`)
2. Detects the GPU architecture and installs ROCm-compatible PyTorch
3. Builds hipDNN and the MIOpen provider (if not already installed, or with `--force-build`)
4. Installs the hipDNN Python bindings from the hipDNN source tree

### CUDA Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

**Note**: hipDNN Python bindings (`hipdnn_frontend`) must be installed separately for hipDNN benchmarking.
**Note**: ROCm PyTorch is required for GPU kernel timing on AMD; validation remains CPU-only.

## Usage

### Basic Benchmarking

A single graph, a glob of graphs, and a tarball of graphs all share the same
execution path. By default results are printed as a summary table. Use `-v` for
the rich per-engine block (useful for debugging a single graph or comparing
engines).

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

### Running from a Tarball

Pass a tarball directly to `--graph` and all `.json` files inside are extracted
to a temporary directory and run as a suite. The archive is cleaned up
automatically when the run finishes.

Supported formats: `.tar`, `.tar.gz`, `.tgz`, `.tar.bz2`, `.tar.xz`

```bash
# Run every graph in a tarball (summary table)
python -m dnn_benchmarking --graph ./Workloads/conv_workloads.tar.gz

# Tarball + JSON output
python -m dnn_benchmarking --graph ./Workloads/conv_workloads.tar.gz --output results.json

# Tarball + verbose per-engine blocks
python -m dnn_benchmarking --graph ./Workloads/conv_workloads.tar.gz -v

# Glob that mixes tarballs and plain JSON files
python -m dnn_benchmarking --graph 'Workloads/*.tar.gz'
```

The extraction progress is reported on stderr:

```
Extracted 42 graph(s) from ./Workloads/conv_workloads.tar.gz
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
| `--graph`, `-g` | Path to a JSON graph file, glob pattern (e.g. `'graphs/*.json'`), or tarball (`.tar`, `.tar.gz`, `.tgz`, `.tar.bz2`, `.tar.xz`) containing JSON graph files | Required |
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

## Utility Tools

The package ships a helper CLI tool installed alongside the main `dnn_benchmarking` entry point.

### `dnn-convert-shapes` — Convert MIOpen driver shape files to hipDNN JSON graphs

Reads MIOpen driver shape `.txt` files (one driver invocation per line) and writes a hipDNN JSON graph file for each shape. Supports `convbfp16`/`conv` and `bnormbfp16`/`bnorm` operations, 2-D and 3-D convolutions, forward/backward/wgrad directions, and NCHW/NHWC layouts.

```bash
# Convert one or more shape files (output goes next to each input file)
dnn-convert-shapes graphs/shapes.txt graphs/shapes_3D.txt

# Write output to a specific directory
dnn-convert-shapes shapes.txt --outdir graphs/generic_convolutions/

# Convert a single inline MIOpen driver invocation
dnn-convert-shapes --args 'convbfp16 -n 16 -c 96 -H 48 -W 32 -k 96 -y 3 -x 1 -p 1 -q 0 -F 1'

# Inline args with explicit output path
dnn-convert-shapes --args 'convbfp16 -n 16 -c 96 -H 48 -W 32 -k 96 -y 3 -x 1' \
  --output graphs/my_conv.json
```

Each converted graph is written as `<stem>_conv_<direction>_n<N>c<C>H<H>W<W>_....json`. Duplicate shapes within a file get a numeric suffix. Lines beginning with `#` and blank lines are skipped. A leading repeat-count column (e.g. `5  ./bin/MIOpenDriver ...`) is stripped automatically.

Exit code is `0` if all shapes convert without warnings, `1` if any warnings were emitted.

## Workload Files (DVC)

The `Workloads/` directory contains performance benchmark workload tar files (graph collections used for benchmarking). These are tracked with [DVC](https://dvc.org/) (backed by S3). The actual archives are **not stored in git** — only the `.dvc` pointer files are. You must pull them separately.

### Pulling workload files

After cloning, switching branches, or pulling commits that change `.dvc` files:

```bash
dvc pull
```

This downloads the tar files tracked by any `.dvc` pointer files in `Workloads/`. If the files are already cached locally, DVC will restore them from cache without re-downloading.

### Adding new workload tar files

Write access to the DVC remote (`s3://therock-dvc/rocm-libraries`) is restricted. Before adding a new tar file:

1. **Request write permissions** from Joseph Macaranas.
2. Once you have access, track and push the new file:

```bash
dvc add Workloads/<new_file>.tar.gz
dvc push
git add Workloads/<new_file>.tar.gz.dvc Workloads/.gitignore
git commit -m "track <new_file>.tar.gz with DVC"
```

Commit only the `.dvc` pointer file and the updated `.gitignore` — never the tar archive itself.

## Running Tests

### Quick Start

```bash
# Activate venv
source /workspace/.venv/bin/activate  # or $DNN_BENCH_WORKSPACE/.venv/bin/activate

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
source /workspace/.venv/bin/activate  # or $DNN_BENCH_WORKSPACE/.venv/bin/activate

# Run tests with ROCm libraries available
LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH pytest
```

**Note:** Set `LD_LIBRARY_PATH=/opt/rocm/lib` when running GPU tests to ensure hipdnn_frontend can load ROCm libraries.

## Limitations

- CPU reference validation is not yet implemented (CPU reference plugin not yet available in Python bindings)
- A/B testing uses `np.allclose()` for accuracy comparison between configurations
