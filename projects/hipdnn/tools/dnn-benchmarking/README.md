# dnn-benchmarking

Benchmarking and validation tool for hipDNN graphs.

## Overview

> **Caution**: This tool is in early development and subject to change.
> Do not use it in build workflows or CI pipelines.

This tool loads serialized hipDNN graphs, executes them via installed hipDNN engine plugins, and captures performance metrics. PyTorch is optional but strongly recommended: when installed (ROCm or CUDA build) it provides GPU kernel-event timing and `torch.cuda.synchronize()` for accurate E2E timing. Without it, host-side E2E timings are still reported but may not capture full GPU execution.

## Requirements

- Python 3.9+
- numpy
- hipdnn_frontend (installed hipDNN Python bindings)
- AMD GPU with ROCm + hipDNN provider plugins for the graphs under test
- PyTorch *(optional)* — ROCm or CUDA build enables GPU kernel-event timing, the `--backend pytorch` executor, and the `--validate pytorch` reference provider. Not listed in `pyproject.toml` because it must come from the ROCm/CUDA nightly index.

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
3. Builds hipDNN and the MIOpen, hipBLASLt, and hip-kernel providers when their installed artifacts are missing (or with `--force-build`)
4. Installs the hipDNN Python bindings from the hipDNN source tree

### CUDA Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

**Note**: hipDNN Python bindings (`hipdnn_frontend`) must be installed separately for hipDNN benchmarking.
**Note**: PyTorch is optional. Without it the tool still runs and reports host-side E2E timings (may not capture full GPU execution); with a ROCm/CUDA build installed, GPU kernel-event timings, accurate E2E via `torch.cuda.synchronize()`, `--backend pytorch`, and `--validate pytorch` become available.

## Usage

### Basic Benchmarking

A single graph, a glob of graphs, and a tarball of graphs all share the same
execution path. By default results are printed as a summary table. Use `-v` for
the rich per-engine block (useful for debugging a single graph or comparing
engines).

```bash
# Single graph (default summary output)
dnn-benchmark --graph ./graphs/sample_conv_fwd.json --warmup 10 --iters 100

# Single graph, verbose: rich per-engine block
dnn-benchmark --graph ./graphs/sample_conv_fwd.json -v

# Filter to specific engine(s) — comma-separated
dnn-benchmark --graph ./graphs/sample_conv_fwd.json --engine 1
dnn-benchmark --graph ./graphs/sample_conv_fwd.json --engine 1,2

# Multiple graphs (glob): same path, default summary table
dnn-benchmark --graph 'graphs/*.json' --warmup 10 --iters 100

# With reproducible random seed
dnn-benchmark --graph ./graphs/sample_conv_fwd.json --seed 42
```

### Running from a Tarball

Pass a tarball directly to `--graph` and all `.json` files inside are extracted
to a temporary directory and run as a suite. The archive is cleaned up
automatically when the run finishes.

Supported formats: `.tar`, `.tar.gz`, `.tgz`, `.tar.bz2`, `.tar.xz`

```bash
# Run every graph in a tarball (summary table)
dnn-benchmark --graph ./Workloads/conv_workloads.tar.gz

# Tarball + JSON output
dnn-benchmark --graph ./Workloads/conv_workloads.tar.gz --output results.json

# Tarball + verbose per-engine blocks
dnn-benchmark --graph ./Workloads/conv_workloads.tar.gz -v

# Glob that mixes tarballs and plain JSON files
dnn-benchmark --graph 'Workloads/*.tar.gz'
```

The extraction progress is reported on stderr:

```
Extracted 42 graph(s) from ./Workloads/conv_workloads.tar.gz
```

### Engine Comparison

Run multiple engines by passing comma-separated engine IDs. Plugin paths may be
a single shared directory or a comma-separated list matching `--engine` order.

```bash
# Compare two engines on the default plugin path
python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json \
  --engine 1,2

# Compare two plugin directories with specific engine IDs
python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json \
  --engine 1,2 \
  --plugin-path /path/to/pluginA,/path/to/pluginB
```

### Config Files

Use `--config` for repeatable benchmark recipes. CLI flags override config
values, so a recipe can be reused with per-run workload or iteration changes.
Relative paths in a config file are resolved from that config file's directory.

```bash
python -m dnn_benchmarking --config sample_configs/basic.toml.example --graph ./graphs/sample_conv_fwd.json
python -m dnn_benchmarking --config sample_configs/config.toml.example --iters 500
```

### CLI Options

#### Basic Options

| Option | Description | Default |
|--------|-------------|---------|
| `--graph`, `-g` | Path to a JSON graph file, glob pattern (e.g. `'graphs/*.json'`), or tarball (`.tar`, `.tar.gz`, `.tgz`, `.tar.bz2`, `.tar.xz`) containing JSON graph files | Required unless provided by `--config` |
| `--config` | TOML benchmark recipe; CLI flags override config values | None |
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
| `--validate` | Reference provider for correctness validation: `pytorch`, `cpu_plugin`, or `none`. `--validate pytorch` also reports a timed PyTorch reference row when PyTorch GPU execution is available. | `none` |

#### Suite Options

| Option | Description | Default |
|--------|-------------|---------|
| `--plugin-path` | Plugin directory, or comma-separated plugin directories matching `--engine` order | None (system default) |

#### Comparison Options

Used by reference validation and suite-mode tolerance checks.

| Option | Description | Default |
|--------|-------------|---------|
| `--rtol` | Relative tolerance for output comparison. Overrides dtype-aware defaults when set; if set without `--atol`, also applies as absolute tolerance. | dtype-aware |
| `--atol` | Absolute tolerance for output comparison. Overrides dtype-aware defaults when set; if set without `--rtol`, also applies as relative tolerance. | dtype-aware |

Automatic validation tolerances are dtype-aware. BF16 outputs use `rtol=1e-2`, `atol=1e-3`; this allows BF16 output quantization and accumulation-order differences while keeping the absolute floor low enough to catch small-magnitude failures.

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

## Related Tools

For the MIOpen shape conversion tool, see the standalone [`dnn-convert-shapes`](../dnn-convert-shapes/) package.

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

# GPU tests with explicit hipDNN engine plugin directories
LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH pytest -m gpu \
  --dnn-plugin-paths /path/to/hipdnn_plugins/engines
```

### GPU Tests

GPU tests require hipDNN Python bindings and ROCm libraries:

```bash
source /workspace/.venv/bin/activate  # or $DNN_BENCH_WORKSPACE/.venv/bin/activate

# Run tests with ROCm libraries available
LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH pytest
```

GPU tests auto-discover provider build-tree and `/opt/rocm` plugin installs. Use
`--dnn-plugin-paths` with a comma-separated directory list when testing custom
engine plugin builds.

**Note:** Set `LD_LIBRARY_PATH=/opt/rocm/lib` when running GPU tests to ensure hipdnn_frontend can load ROCm libraries.

Strict profiling tests that require real profiler artifacts are skipped by
default. Run them explicitly on a known-good profiling host:

```bash
LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH pytest --profiling-strict -m profiling_strict
```

## Limitations

- CPU reference validation is not yet implemented (CPU reference plugin not yet available in Python bindings)
- Engine comparison and timed validation-provider rows are reported side by side. Reference rows are timing baselines and are not counted as hipDNN engine pass/fail combinations.
