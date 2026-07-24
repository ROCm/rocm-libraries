# How to Generate Roofline Graphs for rocSOLVER SYEVD

This document describes how to build rocSOLVER, run its SYEVD benchmark, and generate
roofline graphs using `rocprof-compute`. It is self-contained and written for a fresh
Claude instance or engineer with no prior context.

## Environment

- **Machine:** AMD Instinct MI300A (gfx942, 228 CUs)
- **ROCm version:** 7.2.3 (installed at `/opt/rocm-7.2.3`, symlinked from `/opt/rocm`)
- **rocprofiler-compute version:** 3.4.0 (at `/usr/bin/rocprof-compute`, backed by
  `/opt/rocm-7.2.3/libexec/rocprofiler-compute/`)

## Repository Layout

rocSOLVER lives inside the `rocm-libraries` monorepo. The working directory for all
commands in this document is the rocsolver project root:

```
<monorepo-root>/projects/rocsolver/
```

All relative paths below are from that root.

## Step 1 — Build rocSOLVER with Clients

The first-time configure-and-build (release mode, with test/bench clients, targeting
gfx942 on MI300A):

```bash
./install.sh -cna gfx942 --cmake-arg="-DROCSOLVER_FIND_PACKAGE_LAPACK_CONFIG=OFF"
```

Flag meanings:
| Flag | Effect |
|------|--------|
| `-c` | Include clients (tests + benchmarks) |
| `-n` | Skip specialized small-matrix kernels (faster build) |
| `-a gfx942` | Target GPU architecture |

Output lands in `build/release/`. For incremental rebuilds after the first configure:

```bash
cd build/release && make
```

The benchmark binary is at:

```
build/release/clients/staging/rocsolver-bench
```

## Step 2 — Verify the Build

Test/bench binaries link against the system `/opt/rocm/lib/librocsolver.so.0` by
default. Always prepend `LD_LIBRARY_PATH` to use the locally built library:

```bash
cd build/release
LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH \
  ./clients/staging/rocsolver-bench -f "syevd --uplo L" -n 512 --perf 1 --iters 5 -r s
```

Expected output: a line of performance numbers (time in ms, GFLOPS, etc.).

## Step 3 — Set Up the Python Environment for rocprof-compute

`rocprof-compute` is a Python application. Its dependencies are listed in:

```
/opt/rocm-7.2.3/libexec/rocprofiler-compute/requirements.txt
```

The system Python does not have these packages. Create a dedicated conda environment:

```bash
conda create -n rocprof-compute python=3.11
conda activate rocprof-compute
pip install -r /opt/rocm-7.2.3/libexec/rocprofiler-compute/requirements.txt
```

The environment that produced the results in this document had the following key package
versions (for reproducibility):

| Package | Version |
|---------|---------|
| Python | 3.11.15 |
| pandas | 3.0.3 |
| numpy | 2.4.6 |
| matplotlib | 3.11.0 |
| PyYAML | 6.0.3 |
| SQLAlchemy | 2.0.51 |
| dash | 4.3.0 |
| kaleido | 0.2.1 |
| plotext | 5.3.2 |
| tabulate | 0.10.0 |
| tqdm | 4.68.3 |
| textual | 8.2.7 |

Verify the tool works:

```bash
conda run -n rocprof-compute rocprof-compute --version
```

Expected output:
```
----------------------------------------
rocprofiler-compute version: 3.4.0 (release)
----------------------------------------
```

## Step 4 — Run the Roofline Profile

Run from `build/release/`. The command profiles SYEVD at a single matrix size and
generates the roofline PDF:

```bash
cd build/release

conda run -n rocprof-compute rocprof-compute profile \
  --name syevd_roofline_n512 \
  --roof-only \
  --roofline-data-type FP32 FP64 \
  --kernel-names \
  --format-rocprof-output rocpd \
  -p ./workloads \
  -- \
  env LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH \
  ./clients/staging/rocsolver-bench -f "syevd --uplo L" -n 512 --perf 1 --iters 5 -r s
```

### Key flags

| Flag | Purpose |
|------|---------|
| `--roof-only` | Collect only roofline counters (bandwidth + FLOPs), skip full profiling |
| `--roofline-data-type FP32 FP64` | Generate plots for both FP32 and FP64 ceilings |
| `--kernel-names` | Label each kernel dot in the plot |
| `--format-rocprof-output rocpd` | Use SQLite database output (see bug note below) |
| `-p ./workloads` | Save output under `./workloads/` |
| `--` | Separator between rocprof-compute args and the profiled command |

### What it does internally

The tool makes **3 profiling passes** (one per counter group), each re-running the
binary with hardware counters injected via rocprofiler-sdk. After the 3 passes it also
runs an **empirical roofline benchmark** on the GPU — microkernels that measure peak
bandwidth (HBM, MALL/L3, L2, L1, LDS) and peak FLOP rates (FP8, FP16, BF16, FP32,
FP64, MFMA variants, INT8/32/64). This hardware benchmark takes most of the total
runtime (~5–10 minutes for a first run).

On the MI300A (GPU 0) the measured empirical peaks were:

| Resource | Measured peak |
|----------|---------------|
| HBM bandwidth | 3,407 GB/s |
| MALL (L3) bandwidth | 5,102 GB/s |
| L2 bandwidth | 19,027 GB/s |
| L1 bandwidth | 25,003 GB/s |
| FP32 scalar | 93,427 GFLOPS |
| FP64 scalar | 48,876 GFLOPS |
| MFMA FP32 | 103,374 GFLOPS |
| MFMA FP64 | 106,883 GFLOPS |

### Reusing the empirical roofline data

Once `roofline.csv` exists in the workloads directory, subsequent `--roof-only` runs
**reuse it** and skip the ~5-minute hardware benchmark. To profile a different matrix
size without re-measuring the hardware:

```bash
conda run -n rocprof-compute rocprof-compute profile \
  --name syevd_roofline_n2048 \
  --roof-only \
  --roofline-data-type FP32 FP64 \
  --kernel-names \
  --format-rocprof-output rocpd \
  -p ./workloads \
  -- \
  env LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH \
  ./clients/staging/rocsolver-bench -f "syevd --uplo L" -n 2048 --perf 1 --iters 5 -r s
```

### bench_syevd.sh reference

`build/release/bench_syevd.sh` drives the standard benchmark sweep. Its active size
ranges (as of the time of writing) are:

- **Medium:** n = 320 to 2048, step 64, 101 iterations each
- **Large:** n = 2176 to 4096, step 128, 51 iterations each

For roofline profiling, use a small number of iterations (5–10) and pick representative
sizes (e.g. 512, 1024, 2048, 4096) rather than the full sweep, since each size requires
3 re-runs of the binary under the profiler.

The benchmark takes optional arguments:

```bash
./bench_syevd.sh <bench_binary> [func] [precision] [device_id]
# defaults: func="syevd --uplo L", precision=s (float), device=0
```

## Step 5 — Locate the Output

After a successful run the workloads directory contains:

```
build/release/workloads/
├── empirRoof_gpu-0_FP32_FP64.pdf   ← the roofline plot
├── log.txt
├── perfmon/                         ← counter config files used by the profiler
├── pmc_perf.csv                     ← merged counter data
├── profiling_config.yaml
├── roofline.csv                     ← empirical roofline peaks (reused on next run)
└── sysinfo.csv
```

The PDF name encodes the GPU index and the data types requested.

## Known Bug: CSV Conversion Failure with rocprofiler-sdk Output

### Symptom

When `--format-rocprof-output` is left at its default value (`csv`), the tool prints:

```
WARNING Error converting <file>_counter_collection.csv from v3 to v2 csv:
  You are trying to merge on str and int64 columns for key 'Agent_Id'.
WARNING Cannot write results for pmc_perf_N.csv due to no counter csv files generated.
WARNING Incomplete or missing profiling data. Skipping roofline.
```

No PDF is produced.

### Root cause

The rocprofiler-sdk (the underlying profiler backend) changed the format of `Agent_Id`
in its CSV output from a bare integer (e.g. `6`) to a string with a prefix
(e.g. `"Agent 6"`). The conversion code in
`/opt/rocm-7.2.3/libexec/rocprofiler-compute/utils/utils.py`
(function `v3_counter_csv_to_v2_csv`, around line 567) already contains a fix for
this — it uses a regex to extract the numeric part and converts back to `int64`.
However, in the live code path the fix runs inside a `try/except` block that silently
swallows any exception, leaving `Agent_Id` as `object` dtype. The subsequent
`.merge()` with `agent_info.csv` (where `Node_Id` is `int64`) then fails with the
type mismatch error shown above. The outer exception handler at line 1067 catches this,
prints the warning, and returns an empty list, which causes the roofline step to be
skipped entirely.

Notably, reproducing the conversion manually in Python on the actual CSV files works
correctly — the regex and type conversion succeed. The failure is likely a
pandas-version-specific interaction with the `pivot_table` index dtype that only
manifests in the tool's call path.

### Workaround

Pass `--format-rocprof-output rocpd` to the `profile` subcommand. This switches the
profiler output to SQLite database (`.db`) files instead of CSV, which takes a
completely different code path in the tool and bypasses the broken converter entirely.
This is the flag used in all commands in this document.

Note: `rocpd` mode is expected to become the default in a future release of
rocprofiler-compute (the tool already prints a deprecation warning when using `csv`
mode).

## Troubleshooting

**`rocprof-compute` exits immediately with Python import errors**

The `rocprof-compute` conda environment is not active. Use `conda run -n rocprof-compute`
as shown, or activate it first with `conda activate rocprof-compute`.

**Benchmark binary uses the system librocsolver instead of the local build**

Always prefix with `LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH` when running
from `build/release/`. Without it, the binary silently picks up
`/opt/rocm/lib/librocsolver.so.0`.

**The `env` keyword inside `--`**

The `env VAR=val cmd` form is needed because the `--` separator passes the rest of the
command line as a subprocess invocation. Shell variable expansion (e.g. `$(pwd)`)
happens in the outer shell before the `--` boundary, so `LD_LIBRARY_PATH` is correctly
expanded. The `env` prefix ensures the variable is actually passed into the subprocess
environment.

**GPU Device 1: Skipped** in the empirical benchmark output

Normal on MI300A — the system appears as two GPU devices to the tool but only GPU 0 is
profiled.
