# GDN Decode Driver, Benchmark, and Tuning

This directory contains the host-side tools for the gfx950 GDN decode kernel.
The kernel emitter itself lives at
[`library/kernels/gfx950/gdn_decode.py`](../../../kernels/gfx950/gdn_decode.py).

Start with [`ALGORITHM.md`](ALGORITHM.md) if you want to understand the equations
and GPU thread mapping. Use this page when you want to run, check, benchmark, or
retune the kernel.

## Contents

- [Files](#files)
- [Environment](#environment)
- [Check correctness](#check-correctness)
- [Benchmark dispatched kernels](#benchmark-dispatched-kernels)
- [Retune the tile table](#retune-the-tile-table)
- [Run tests](#run-tests)
- [Understand the output](#understand-the-output)
- [Exit codes](#exit-codes)
- [Common failures](#common-failures)

## Files

| File | Purpose |
|---|---|
| [`gdn_decode.py`](gdn_decode.py) | Compile a spec, build inputs, launch the kernel, compare with the fp32 reference, and optionally time it |
| [`tune.py`](tune.py) | Search all legal tiles and report the fastest correct tile for each batch |
| [`ALGORITHM.md`](ALGORITHM.md) | Explain the gated delta rule and GPU mapping |
| [`library/benchmarks/gfx950/gdn/benchmark_gdn_decode.py`](../../../benchmarks/gfx950/gdn/benchmark_gdn_decode.py) | Benchmark the configuration selected by dispatch |
| [`library/dispatch/gdn/gfx950.py`](../../../dispatch/gdn/gfx950.py) | Store the gfx950 capability and tuned batch-to-tile table |
| [`library/tests/test_gdn_decode_spec.py`](../../../tests/test_gdn_decode_spec.py) | CPU validator and IR-emission coverage |
| [`library/tests/test_gdn_decode_gfx950_numeric.py`](../../../tests/test_gdn_decode_gfx950_numeric.py) | On-device output and state correctness |
| [`library/tests/test_gdn_decode_golden.py`](../../../tests/test_gdn_decode_golden.py) | Detect unexpected LLVM-IR changes |

## Environment

Run from `dnn-providers/hip-kernel-provider/rocke` with both the library and the
platform Python package on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/library:$PWD/platform/python${PYTHONPATH:+:$PYTHONPATH}"
```

The driver, benchmark, tuning sweep, and numeric tests require:

- ROCm torch with a visible gfx950 GPU;
- a working ROCm comgr library for compiling the emitted LLVM IR;
- the rocKE Python package from `platform/python`.

The spec, dispatch, and golden tests do not need a GPU. The golden test lowers
to LLVM IR but does not invoke comgr.

## Check correctness

Run the default warp-tiled path across several batch sizes:

```bash
python3 library/builders/gfx950/gdn/gdn_decode.py \
  --batches 1,16,64,256
```

Example output shape:

```text
kernel: <compiled kernel name>  block=<threads per workgroup>
B=1     grid=<workgroups> out_err=<error> state_err=<error> OK
B=16    grid=<workgroups> out_err=<error> state_err=<error> OK
worst=<largest error> tol=1.0e-02
```

The driver checks **two results**:

- `out_err`: maximum absolute error in this token's output;
- `state_err`: maximum absolute error in the updated recurrent state.

Both must stay below `TOL`. Checking only `out` is insufficient because a bad
state write may not affect the visible output until the next decode step.

Check the simple one-thread-per-row reference body:

```bash
python3 library/builders/gfx950/gdn/gdn_decode.py \
  --batches 1,16 --variant simple
```

Also report wall time from the driver:

```bash
python3 library/builders/gfx950/gdn/gdn_decode.py \
  --batches 1,16 --bench
```

`--no-check` skips the fp32 reference and should be used only for focused
measurement after correctness has already been established.

## Benchmark dispatched kernels

The benchmark asks the dispatcher which tile production would use for each
batch. It does **not** benchmark one hardcoded spec across the whole range.

```bash
python3 library/benchmarks/gfx950/gdn/benchmark_gdn_decode.py \
  --batches 1,16,64,256
```

It prints:

| Column | Meaning |
|---|---|
| `batch` | Active sequences in this decode step |
| `tile` | `(num_warps, warp_threads_k, blocks_per_v_dim)` selected by dispatch |
| `spec_id` | Stable name of the selected tuned band |
| `grid` | Number of launched workgroups |
| `eager_us` | One launch plus one synchronization, including host launch cost |
| `device_us` | Per-launch GPU time from a replayed HIP graph |
| `correctness` | Largest of output and state error |

Small-batch decode often spends more time submitting the launch than executing
the GPU kernel. That is why both eager and device time are shown. They answer
different questions and should not be treated as interchangeable.

If graph capture is unavailable in the environment, skip device timing:

```bash
python3 library/benchmarks/gfx950/gdn/benchmark_gdn_decode.py --no-device
```

## Retune the tile table

The dispatcher table is empirical. Re-run the sweep after changing the kernel,
compiler, target, or supported shape:

```bash
python3 library/builders/gfx950/gdn/tune.py
```

Tune only selected batches or print fewer candidates:

```bash
python3 library/builders/gfx950/gdn/tune.py \
  --batches 1,16 --top 5
```

The sweep:

1. enumerates the configured tile search space;
2. lets `is_valid_spec` reject illegal combinations;
3. computes the fp32 reference once per batch;
4. runs every valid tile and rejects numerically wrong results;
5. graph-times the remaining tiles and prints them fastest first.

Device time is the tuning metric because host launch overhead is nearly the same
for every tile and can hide kernel differences at small batch.

If the winners disagree with the table in
[`library/dispatch/gdn/gfx950.py`](../../../dispatch/gdn/gfx950.py), update
`_TUNED_TILES`, keep bands coarse, and rerun the dispatch wiring and numeric
tests. Only measured batch points are evidence; boundaries between measured
points are interpolation.

## Run tests

CPU-only coverage:

```bash
python3 -m pytest \
  library/tests/test_gdn_decode_spec.py \
  library/tests/test_gdn_decode_golden.py \
  library/tests/dispatch/gdn/test_gfx950_wiring.py \
  -m "not gpu"
```

On-device numeric coverage:

```bash
python3 -m pytest \
  library/tests/test_gdn_decode_gfx950_numeric.py \
  -m gpu
```

Re-record the golden LLVM-IR hashes **only when an emitted-code change is
intentional and reviewed**:

```bash
python3 library/tests/test_gdn_decode_golden.py --write
```

Then rerun the golden test. A changed hash means the emitted LLVM IR changed; it
does not by itself say whether the new code is correct.

The project-level check entry point is:

```bash
python3 tools/run_checks.py
```

## Understand the output

`grid` is the number of workgroups:

```text
grid = batch * num_v_heads * blocks_per_v_dim
```

At small batch, `blocks_per_v_dim` may be greater than one to create more
workgroups and fill the GPU. At large batch, dispatch normally reduces the split
because the batch already provides enough workgroups.

`out_err` and `state_err` are maximum absolute errors against the fp32 reference.
In the current coverage, state error is larger than output error. Both remain
separate because state becomes an input to the next decode step; a correct
current output cannot prove that the next step will read correct state.

## Exit codes

The driver and benchmark use meaningful process status:

| Code | Meaning |
|---|---|
| `0` | All requested correctness checks passed |
| `1` | At least one checked shape exceeded tolerance or no valid timed tile remained |
| `2` | No GPU was visible or the requested spec was invalid |

Scripts and CI should check the exit code instead of relying on printed text.

## Common failures

### `no HIP device visible`

The script cannot see a ROCm GPU. Check the job's GPU allocation and
`ROCR_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES`.

### `invalid gdn_decode spec`

The requested dimensions or tiling violate a validator rule. The error message
names the rejected rule. Do not bypass the validator; change the shape or tile.

### Graph capture is unavailable

Use `--no-device` for the benchmark. Eager timing still works. Graph support is
environment-sensitive and does not mean the kernel itself is invalid.

### Golden test reports IR drift

First decide whether the generated code was meant to change. If not, find the
emitter change that caused the drift. If yes, review the new IR and numeric
results, then regenerate with `--write` in the same change.

### Correct output but wrong state

Treat this as a failure. The next decode step reads that state, so checking the
visible output alone is not enough.
