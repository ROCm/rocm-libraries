# GDN/KDA Host Tools: Driver, Benchmark, and Tuning

This directory contains the host-side tools for the shared gfx950 GDN/KDA
single-token decode emitter. The device-code emitter lives at
[`library/kernels/gfx950/gdn_decode.py`](../../../kernels/gfx950/gdn_decode.py).

**GDN prefill** is driven from
[`library/builders/gfx950/kda/gdn_prefill.py`](../../kda/gdn_prefill.py) and runs
the shared KDA chunkwise kernels in `gate_kind="gdn"` mode, so its tools live in
the `kda/` directory rather than here. Its commands are in [Prefill](#prefill)
below.

Start with [`ALGORITHM.md`](ALGORITHM.md) for the equations, gate-kind
difference, and GPU thread mapping. Use this page to run, check, benchmark, or
retune either decode mode or the GDN prefill path.

## Contents

- [Files](#files)
- [Environment](#environment)
- [Check correctness](#check-correctness)
- [Benchmark dispatched kernels](#benchmark-dispatched-kernels)
- [Retune the tile table](#retune-the-tile-table)
- [Run tests](#run-tests)
- [Prefill](#prefill)
- [Understand the output](#understand-the-output)
- [Exit codes](#exit-codes)
- [Common failures](#common-failures)

## Files

| File | Purpose |
|---|---|
| [`gdn_decode.py`](gdn_decode.py) | Compile a spec, build inputs, launch the shared decode emitter, and compare with the independent fp32 reference |
| [`tune.py`](tune.py) | Search every legal tile for GDN or KDA; GDN reports by batch, KDA compares equal-work head geometries |
| [`ALGORITHM.md`](ALGORITHM.md) | Explain the gated delta rule, scalar/vector gate delta, and GPU mapping |
| [`library/benchmarks/gfx950/gdn/benchmark_gdn_decode.py`](../../../benchmarks/gfx950/gdn/benchmark_gdn_decode.py) | Benchmark GDN's dispatcher-selected kernel |
| [`library/benchmarks/gfx950/gdn/benchmark_kda_decode.py`](../../../benchmarks/gfx950/gdn/benchmark_kda_decode.py) | Benchmark KDA fused/precomputed/simple variants from the production dispatcher; optionally sweep all legal tiles |
| [`library/dispatch/gdn/gfx950.py`](../../../dispatch/gdn/gfx950.py) | Store gfx950 capability plus separate GDN batch-keyed and KDA work-keyed tables |
| [`library/tests/test_gdn_decode_spec.py`](../../../tests/test_gdn_decode_spec.py) | CPU validator and IR-emission coverage |
| [`library/tests/test_gdn_decode_prepare.py`](../../../tests/test_gdn_decode_prepare.py) | Host-side input validation: shapes, dtypes, contiguity, pool-index range |
| [`library/tests/test_gdn_decode_gfx950_numeric.py`](../../../tests/test_gdn_decode_gfx950_numeric.py) | On-device GDN output and state correctness |
| [`library/tests/test_kda_decode_gfx950_numeric.py`](../../../tests/test_kda_decode_gfx950_numeric.py) | On-device KDA output/state correctness, tuned tiles, and dispatch-to-launch coverage |
| [`library/tests/test_gdn_decode_golden.py`](../../../tests/test_gdn_decode_golden.py) | Detect unexpected LLVM-IR changes in both gate kinds |
| [`library/builders/gfx950/kda/gdn_prefill.py`](../../kda/gdn_prefill.py) | Drive chunkwise prefill (the KDA chunkwise kernels in `gate_kind="gdn"` mode) and hold its fp64 oracle |
| [`library/benchmarks/gfx950/gdn/sweep_prefill_value_splits.py`](../../../benchmarks/gfx950/gdn/sweep_prefill_value_splits.py) | Sweep `value_splits` for prefill at a given `batch_heads` |
| [`library/tests/dispatch/gdn/test_gfx950_prefill_wiring.py`](../../../tests/dispatch/gdn/test_gfx950_prefill_wiring.py) | Prefill dispatch: candidate selection, the two-launch guard, launch geometry |
| [`library/tests/test_gdn_prefill_decay_guard.py`](../../../tests/test_gdn_prefill_decay_guard.py) | Pin the supported envelope of the unbounded GDN decay gate |
| [`library/tests/test_kda_gdn_gfx950_numeric.py`](../../../tests/test_kda_gdn_gfx950_numeric.py) | On-device prefill correctness in `gate_kind="gdn"` mode |

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

Both benchmarks ask dispatch which tile production would use for each batch;
neither times one hardcoded tile across the whole range.

GDN:

```bash
python3 library/benchmarks/gfx950/gdn/benchmark_gdn_decode.py \
  --batches 1,16,64,256
```

KDA, production fused and recurrence-only precomputed-log-decay variants:

```bash
python3 -m benchmarks.gfx950.gdn.benchmark_kda_decode \
  --batches 1,8,32,128
```

The KDA benchmark also offers the diagnostic simple emitter and an exhaustive
legal-tile proof:

```bash
python3 -m benchmarks.gfx950.gdn.benchmark_kda_decode \
  --batches 8 --variants fused,precomputed,simple \
  --sweep-tiles --top 5
```

`precomputed` excludes log-decay production cost; it measures recurrence-only
cost, not an unfused production pipeline. `--sweep-tiles` prints the
dispatcher-selected tile, fastest legal tile, their ratio, and top candidates.

Both benchmarks print eager and device timing. Eager includes host launch and
synchronisation; device timing uses replayed HIP graphs. Small-batch decode can
be launch-bound, so the two clocks answer different questions.

If graph capture is unavailable, pass `--no-device`.

## Retune the tile tables

The dispatcher tables are empirical. Re-run the sweep after changing the
kernel, compiler, target, or supported shape.

GDN uses its original batch-keyed table:

```bash
python3 library/builders/gfx950/gdn/tune.py \
  --gate-kind gdn --batches 1,16,64,256
```

KDA uses `work = batch × num_v_heads`, so compare equal-work cells across head
geometries:

```bash
python3 library/builders/gfx950/gdn/tune.py \
  --gate-kind kda \
  --geometries 16/32,8/16,4/8 \
  --batches 1,2,4,8,16,32,64,128 \
  --top 5
```

The sweep:

1. enumerates the configured tile search space;
2. lets `is_valid_spec` reject illegal combinations;
3. computes the fp32 reference once per shape;
4. runs every valid tile and rejects numerically wrong results;
5. graph-times the remaining tiles and prints them fastest first;
6. reports whether equal-work KDA cells agree on the best tile.

Device time is the tuning metric because host launch overhead is nearly the
same for every tile and can hide kernel differences at small batch. Update
`_TUNED_TILES_GDN` or `_TUNED_TILES_KDA` as appropriate, keep bands coarse,
and rerun dispatch wiring plus numeric tests. Only measured points are
evidence; boundaries between measured points are interpolation.

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
  library/tests/test_kda_decode_gfx950_numeric.py \
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

## Prefill

Prefill is the KDA chunkwise pair in GDN gate mode; the driver and its fp64
oracle are in `kda/`, not this directory.

Check correctness against the oracle:

```bash
python3 library/builders/gfx950/kda/gdn_prefill.py
```

Sweep `value_splits` for a given `batch_heads` band:

```bash
python3 library/benchmarks/gfx950/gdn/sweep_prefill_value_splits.py
```

Tests:

```bash
python3 -m pytest \
  library/tests/dispatch/gdn/test_gfx950_prefill_wiring.py \
  library/tests/test_gdn_prefill_decay_guard.py \
  library/tests/test_kda_chunkwise_spec.py
```

On-device numeric coverage (needs a gfx950 GPU):

```bash
python3 -m pytest library/tests/test_kda_gdn_gfx950_numeric.py
```

## Understand the output

For the warp-tiled path:

```text
grid = batch * num_v_heads * blocks_per_v_dim
```

For the simple reference path, there is no V split:

```text
grid = batch * num_v_heads
```

At small batch, `blocks_per_v_dim` may be greater than one to create more
workgroups and fill the GPU. At large batch, dispatch normally reduces the split
because the batch already provides enough workgroups.

`out_err` and `state_err` are maximum absolute errors against the fp32 reference.
In the current coverage, state error is larger than output error. Both remain
separate because state becomes an input to the next decode step; a correct
current output cannot prove that the next step will read correct state.

## Exit codes

Exit codes differ slightly by command:

| Command | `0` | `1` | `2` |
|---|---|---|---|
| `gdn_decode.py` | all requested checks passed, or checks were disabled | at least one checked batch exceeded tolerance | no GPU visible or the fixed spec was rejected |
| `benchmark_gdn_decode.py` | every batch passed correctness and was timed | at least one batch exceeded tolerance | no GPU visible |
| `benchmark_kda_decode.py` | every requested variant/sweep passed correctness and timing | any requested variant/sweep failed or was incomplete | visible HIP device is not gfx950 |
| `tune.py` | every requested cell produced at least one correct, timed tile | some requested cell produced no correct, timeable tile | no GPU visible |

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
