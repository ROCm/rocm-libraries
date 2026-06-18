# dnn-benchmarking

Benchmarking and validation tool for hipDNN graphs.

## Overview

> **Caution**: This tool is in early development and subject to change.
> Do not use it in build workflows or CI pipelines.

This tool loads serialized hipDNN graphs, executes them via installed hipDNN
engine plugins, and captures performance metrics. On AMD GPUs, hipDNN kernel
timing and synchronized E2E timing use direct HIP runtime events exposed by
`hipdnn_frontend`; PyTorch is only needed for the optional PyTorch executor and
reference validation.

The `--backend pytorch` executor also runs on NVIDIA GPUs with a CUDA PyTorch
build, where it times kernels with `torch.cuda` events. Because the hipDNN and
PyTorch backends share one suite execution path and emit the same
`SuiteResult` JSON schema, the same `--graph ... --backend pytorch -o out.json`
command can be run on a ROCm host and a CUDA host and the two JSON artifacts
compared offline. See [Cross-Machine Comparison](#cross-machine-comparison-rocm-vs-cuda).

## Requirements

- Python 3.12+
- numpy
- hipdnn_frontend (installed hipDNN Python bindings) — required for the hipDNN
  backend; **not** required for the `--backend pytorch` executor
- A GPU for execution:
  - AMD GPU with ROCm + hipDNN provider plugins — for the hipDNN backend and the
    ROCm PyTorch executor
  - NVIDIA GPU with a CUDA PyTorch build — for the `--backend pytorch` executor
- PyTorch *(optional for the hipDNN backend)* — any usable PyTorch build enables
  the `--validate pytorch` reference provider. A ROCm or CUDA PyTorch build
  enables the `--backend pytorch` GPU executor. Not listed in `pyproject.toml`
  because torch package selection depends on the target environment.

## Installation

### Quick Setup (ROCm/AMD GPUs)

Run the provided setup script from the `dnn-benchmarking` directory:

Requires Python 3.12 or newer.

```bash
bash setup.sh --workspace .workspace
source .workspace/.venv/bin/activate
```

By default, setup uses `/workspace` when it already exists and is writable;
otherwise it uses `.workspace` under the `dnn-benchmarking` directory. Use
`--workspace <path>` to place the virtual environment, Python bytecode cache,
and runtime benchmark caches somewhere else:

```bash
bash setup.sh --workspace /tmp/dnn-bench
source /tmp/dnn-bench/.venv/bin/activate
```

The default `--torch-mode rocm` flow assumes no system ROCm installation:
1. Creates a virtual environment under the selected workspace (`--workspace`,
   `$DNN_BENCH_WORKSPACE`, writable `/workspace`, or local `.workspace`)
2. Detects the GPU architecture and installs the matching ROCm PyTorch nightly wheel
3. Discovers ROCm libraries from the torch wheel's bundled ROCm SDK libraries
4. Builds local hipDNN when CMake configs are absent from the selected prefix
5. Builds the local MIOpen, hipBLASLt, and hip-kernel providers when their
   installed artifacts are missing, using the bundled ROCm SDK devel wheel for
   compiler/toolchain discovery when needed
6. Installs the hipDNN Python bindings against the selected ROCm SDK libraries

The selected prefix is printed as `Using hipDNN/ROCm prefix: ...`; activation
sets `ROCM_PATH` to that prefix and prepends its `lib` directory to
`LD_LIBRARY_PATH`. dnn-benchmarking infers plugins from
`$ROCM_PATH/lib/hipdnn_plugins/engines`.
If GPU architecture detection is unavailable on the setup host, pass
`--gpu-arch gfx90a`, `--gpu-arch gfx942`, or `--gpu-arch gfx950`.

### Testing/CI Setup with CPU-Only PyTorch

When ROCm/hipDNN artifacts are installed by CI, install CPU-only PyTorch on top
so Python reference validation can use torch without pulling conflicting ROCm
torch wheels:

```bash
bash setup.sh --torch-mode cpu --rocm-prefix /opt/rocm
source /workspace/.venv/bin/activate
```

CPU-only torch never enables the PyTorch execution backend; it is only for
Python reference validation. The `--backend pytorch` executor needs a GPU torch
build: ROCm torch (HIP-event timing) or CUDA torch (torch.cuda-event timing,
see below).

Use `--torch-mode existing` to reuse torch already installed in the target
virtual environment. Existing ROCm torch uses its bundled ROCm SDK libraries;
existing CUDA torch takes the CUDA skip path (no hipDNN bindings); existing
CPU-only torch builds the hipDNN bindings against `--rocm-prefix`, `$ROCM_PATH`,
or `/opt/rocm` for the hipDNN backend.

### CUDA PyTorch (NVIDIA GPUs, `--backend pytorch` only)

To benchmark graph shapes through PyTorch on an NVIDIA GPU — for offline
comparison against ROCm results — install a CUDA PyTorch build with
`--torch-mode cuda`:

```bash
bash setup.sh --torch-mode cuda --workspace .workspace
source .workspace/.venv/bin/activate
```

`--torch-mode cuda` installs torch from PyPI (override the index with
`--torch-index-url <url>`, e.g. a specific CUDA wheel channel) and **skips all
ROCm setup**: no hipDNN build, engine plugins, hipDNN Python bindings, amdsmi,
or `ROCM_PATH`/`LD_LIBRARY_PATH` wiring. Only the `--backend pytorch` executor
is available in this mode — the hipDNN backend requires `hipdnn_frontend` and
ROCm. `--force-build` is rejected with `--torch-mode cuda` because building
hipDNN requires a ROCm toolchain.

```bash
# Benchmark a graph suite through PyTorch CUDA and emit comparable JSON
python -m dnn_benchmarking --graph 'graphs/*.json' --backend pytorch -o cuda_results.json
```

On CUDA, kernel timing uses `torch.cuda` events and the ROCm-specific metadata
fields (`rocm_version`, amdsmi GPU snapshot) are `None` in the JSON, while
`gpu_arch` is the sentinel `"unknown"` (no ROCm gfx target is detectable); the
timing statistics and graph structure are identical to a ROCm run.

### Other Non-ROCm PyTorch (CPU)

hipDNN timing requires ROCm HIP runtime libraries and the hipDNN frontend
bindings. CPU-only PyTorch wheels can be used only for reference validation;
they do not enable GPU benchmarking in this tool.

**Note**: hipDNN Python bindings (`hipdnn_frontend`) must be installed separately
for hipDNN benchmarking.
**Note**: PyTorch is optional for the hipDNN backend. CPU-only PyTorch enables
`--validate pytorch` reference computation; ROCm or CUDA PyTorch also enables
`--backend pytorch`. hipDNN-backend GPU timing and E2E synchronization always
use direct HIP APIs from `hipdnn_frontend`; the PyTorch backend uses HIP events
on ROCm and `torch.cuda` events on CUDA.

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

Run multiple engines by passing comma-separated engine IDs. By default,
dnn-benchmarking infers the plugin directory from
`$ROCM_PATH/lib/hipdnn_plugins/engines` when `ROCM_PATH` is set by `setup.sh`
activation. Plugin paths may also be a single shared directory or a
comma-separated list matching `--engine` order.

```bash
# Compare two engines using ROCM_PATH from the activated setup environment
python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json \
  --engine 1,2

# Compare two plugin directories with specific engine IDs
python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json \
  --engine 1,2 \
  --plugin-path /path/to/pluginA,/path/to/pluginB
```

### PyTorch Backend

`--backend pytorch` runs each graph through the PyTorch executor instead of
hipDNN engine plugins, producing one `provider="pytorch"` row per graph. It
shares the suite execution path with the hipDNN backend, so single-graph,
glob, and tarball inputs and `--output` JSON all work identically.

```bash
# Single graph through PyTorch
dnn-benchmark --graph ./graphs/sample_conv_fwd.json --backend pytorch

# A whole suite through PyTorch, with JSON output
dnn-benchmark --graph 'graphs/*.json' --backend pytorch -o pytorch_results.json
```

The PyTorch backend ignores hipDNN-specific selection and profiling options;
the following are rejected with `--backend pytorch`:

- `--engine` / `--plugin-path` (no hipDNN engine plugins are loaded)
- `--validate pytorch` (the backend would validate against itself)
- `--pmc` / `--emit-trace` / `--perf` / `--roofline` (rocprofv3-based passes)

### Cross-Machine Comparison (ROCm vs CUDA)

Because both backends emit the same `SuiteResult` JSON schema, you can benchmark
the same graphs on an AMD machine and an NVIDIA machine and compare offline:

```bash
# On the AMD/ROCm host (hipDNN engines):
dnn-benchmark --graph 'graphs/*.json' -o rocm_results.json

# On the AMD/ROCm host (PyTorch executor, for an apples-to-apples PyTorch row):
dnn-benchmark --graph 'graphs/*.json' --backend pytorch -o rocm_pytorch_results.json

# On the NVIDIA/CUDA host (PyTorch executor):
dnn-benchmark --graph 'graphs/*.json' --backend pytorch -o cuda_pytorch_results.json
```

Each JSON file is a full `SuiteResult`: `graphs` is a list of graph entries,
each carrying its `graph_name` plus result rows with E2E and kernel timing
statistics and whatever machine metadata the host could provide
(`rocm_version` and the amdsmi snapshot are `None` on CUDA, and `gpu_arch` is
`"unknown"`). Graphs match across files by `graph_name`, so the artifacts can
be diffed offline. (An offline comparison helper is planned but not yet
included.)

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
| `--backend`, `-b` | Execution backend: `hipdnn` (AMD GPU via hipDNN engine plugins) or `pytorch` (GPU via PyTorch; ROCm or CUDA). `pytorch` is incompatible with `--engine`, `--plugin-path`, `--validate pytorch`, and the profiling flags. | `hipdnn` |

#### Output Options

| Option | Description | Default |
|--------|-------------|---------|
| `--output`, `-o` | Export benchmark results to JSON file (full SuiteResult; independent of `-v`) | None |
| `--verbose`, `-v` | Show detailed per-engine block per graph (default: summary table) | False |

#### Reference Validation Options

| Option | Description | Default |
|--------|-------------|---------|
| `--validate` | Reference provider for correctness validation: `pytorch` or `none`. `--validate pytorch` also reports a timed PyTorch reference row when PyTorch GPU execution is available. | `none` |

#### Suite Options

| Option | Description | Default |
|--------|-------------|---------|
| `--plugin-path` | Plugin directory, or comma-separated plugin directories matching `--engine` order. Overrides the plugin directory inferred from `ROCM_PATH`. | `$ROCM_PATH/lib/hipdnn_plugins/engines` if `ROCM_PATH` is set, otherwise system default |

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

# All tests including GPU (activation sets LD_LIBRARY_PATH for setup workspaces)
pytest

# Only GPU tests
pytest -m gpu

# GPU tests with explicit hipDNN engine plugin directories
pytest -m gpu --dnn-plugin-paths /path/to/hipdnn_plugins/engines
```

### GPU Tests

GPU tests require hipDNN Python bindings and ROCm libraries. When using
`setup.sh`, activate the venv first; activation sets `ROCM_PATH` and prepends
the selected prefix's `lib` directory to `LD_LIBRARY_PATH`:

```bash
source /workspace/.venv/bin/activate  # or $DNN_BENCH_WORKSPACE/.venv/bin/activate
pytest -m gpu
```

GPU tests auto-discover provider build-tree, active-venv ROCm SDK, and
`/opt/rocm` plugin installs. Use `--dnn-plugin-paths` with a comma-separated
directory list when testing custom engine plugin builds.

GPU tests are tiered by marker: `gpu` (any GPU), `rocm` (AMD ROCm only),
`cuda` (NVIDIA CUDA only). GPU-generic tests run on either platform and
adapt their timing-backend assertion automatically (HIP on ROCm,
torch.cuda on CUDA); platform-specific tests assert one backend's unique
behavior. Every GPU test self-skips on the wrong platform, so bare
`pytest` is safe on any host. Use `-m` for explicit, additive selection:

```bash
# On a ROCm host: unit + generic + rocm (drops cuda-only tests)
pytest -m "not cuda"

# On a CUDA host: unit + generic + cuda (drops rocm-only tests)
pytest -m "not rocm"

# Only one platform's dedicated tests
pytest -m rocm
pytest -m cuda
```

Strict profiling tests that require real profiler artifacts are skipped by
default. Run them explicitly on a known-good profiling host:

```bash
LD_LIBRARY_PATH=$HIPDNN_PREFIX/lib:$LD_LIBRARY_PATH pytest --profiling-strict -m profiling_strict
```

## Limitations

- Engine comparison and timed validation-provider rows are reported side by side. Reference rows are timing baselines and are not counted as hipDNN engine pass/fail combinations; use `--validate` for reference-output correctness checks.
