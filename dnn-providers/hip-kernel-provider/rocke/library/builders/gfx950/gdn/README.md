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
- [Benchmark registered candidates](#benchmark-registered-candidates)
- [Measure candidate tiles](#measure-candidate-tiles)
- [Run tests](#run-tests)
- [Prefill](#prefill)
- [Understand the output](#understand-the-output)
- [Exit codes](#exit-codes)
- [Common failures](#common-failures)

## Files

| File | Purpose |
|---|---|
| [`gdn_decode.py`](gdn_decode.py) | Compile a spec, build inputs, launch the shared decode emitter, and compare with the independent fp32 reference |
| [`tune.py`](tune.py) | Measure every legal GDN registry tile or KDA tile (base configuration) against the `auto` tile |
| [`sweep.py`](sweep.py) | Tiles x knobs sweep: every registered configuration of a case (sharded across devices, cold memory), merged into a ratio ranking and re-timed interleaved |
| [`ALGORITHM.md`](ALGORITHM.md) | Explain the gated delta rule, gate kinds, and GPU mapping |
| [`library/benchmarks/gfx950/gdn/benchmark_gdn_decode.py`](../../../benchmarks/gfx950/gdn/benchmark_gdn_decode.py) | Benchmark every legal GDN registry tile (base configuration) and the dispatcher `auto` configuration (its case's tile and knob stack) |
| [`library/benchmarks/gfx950/gdn/benchmark_kda_decode.py`](../../../benchmarks/gfx950/gdn/benchmark_kda_decode.py) | Benchmark KDA fused/precomputed/simple variants from the dispatcher; optionally sweep all legal tiles |
| [`library/dispatch/gdn/gfx950.py`](../../../dispatch/gdn/gfx950.py) | Register every (gate kind, tile) and select the `auto` configuration of each case; the per-case defaults are data in [`axes.py`](../../../dispatch/gdn/axes.py) (`AUTO_DEFAULTS`) |
| [`library/dispatch/gdn/axes.py`](../../../dispatch/gdn/axes.py), [`rules.py`](../../../dispatch/gdn/rules.py) | Each tile's knob space: the knob axes as data, and their scope and legality |
| [`library/tests/dispatch/gdn/test_gfx950_registry.py`](../../../tests/dispatch/gdn/test_gfx950_registry.py) | CPU GDN registry count, identity, legality, and selection coverage |
| [`library/tests/dispatch/gdn/test_tuning_contract.py`](../../../tests/dispatch/gdn/test_tuning_contract.py) | CPU tuning contract: knob space per mode, pins (`tuning_id` / `tuning_knobs`), refusals |
| [`library/tests/test_gdn_decode_spec.py`](../../../tests/test_gdn_decode_spec.py) | CPU validator and IR-emission coverage |
| [`library/tests/test_gdn_decode_prepare.py`](../../../tests/test_gdn_decode_prepare.py) | Host-side input validation: shapes, dtypes, contiguity, devices, pointer alignment, pool-index range, fused-mode inputs |
| [`library/tests/test_gdn_decode_fused.py`](../../../tests/test_gdn_decode_fused.py) | CPU contract of the fused conv1d / gated-RMSNorm modes: spec rules, ABI, IR alignment claims, tap-shift barrier, fp32 reference |
| [`library/tests/test_gdn_decode_fused_gfx950_numeric.py`](../../../tests/test_gdn_decode_fused_gfx950_numeric.py) | On-device fused-mode output, state and conv-state correctness |
| [`library/tests/test_gdn_decode_gfx950_numeric.py`](../../../tests/test_gdn_decode_gfx950_numeric.py) | On-device GDN output and state correctness |
| [`library/tests/test_kda_decode_gfx950_numeric.py`](../../../tests/test_kda_decode_gfx950_numeric.py) | On-device KDA output/state correctness and dispatch-to-launch coverage |
| [`library/tests/test_gdn_decode_knobs.py`](../../../tests/test_gdn_decode_knobs.py) | CPU contract of the performance knobs: validation, `*_applies` predicates, name injectivity, emitted effect |
| [`library/tests/test_gdn_decode_knobs_gfx950_numeric.py`](../../../tests/test_gdn_decode_knobs_gfx950_numeric.py) | On-device correctness of every knob on every registered tile (and every `BPV=1` fused tile) it applies to at the default heads, at head dims 64 and 256, and `out_lds` at `DV` 256 / 512 |
| [`library/tests/test_gdn_decode_golden.py`](../../../tests/test_gdn_decode_golden.py) | Detect unexpected LLVM-IR changes in both gate kinds (Python lowering) |
| [`library/tests/test_gdn_decode_tokens.py`](../../../tests/test_gdn_decode_tokens.py), [`test_gdn_decode_tokens_gfx950_numeric.py`](../../../tests/test_gdn_decode_tokens_gfx950_numeric.py) | Multi-token steps (`num_tokens`): CPU contract, and on-device output / per-token snapshot correctness |
| [`library/tests/test_gdn_decode_norm_gate.py`](../../../tests/test_gdn_decode_norm_gate.py), [`test_gdn_decode_norm_gate_gfx950_numeric.py`](../../../tests/test_gdn_decode_norm_gate_gfx950_numeric.py) | `norm_gate_act` (sigmoid-gated GDN norm): CPU contract and on-device correctness |
| [`library/tests/test_gdn_decode_sweep.py`](../../../tests/test_gdn_decode_sweep.py) | CPU tests of `sweep.py`: sharding, space, ranking, ratio-only output |
| [`library/tests/test_gdn_decode_ir_cpp_parity.py`](../../../tests/test_gdn_decode_ir_cpp_parity.py) | The C++ engine lowers every golden case to byte-identical LLVM IR |
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

- `out_err`: maximum absolute error in this token's output (for a
  `fuse_out_norm` spec, relative to `max(1, |ref|)`; see below);
- `state_err`: maximum absolute error in the updated recurrent state (and, for
  a `fuse_conv` spec, the conv state).

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

### f32 state and fused modes

The recurrent state may be `f32` (`GdnDecodeSpec(state_dtype="f32")`); only the
state widens, and the inputs, output and fp32 reference are unchanged. Two
optional flags fuse a hybrid layer's neighbours into the kernel: `fuse_conv`
(causal conv1d + SiLU on the packed q/k/v row) and `fuse_out_norm` (gated
RMSNorm on the output; SiLU gate for GDN, sigmoid for KDA). What they compute,
the tensors they add and the spec rules they need are in
[`ALGORITHM.md` §4.9](ALGORITHM.md#49-optional-fusions-conv1d-and-gated-rmsnorm).

`make_inputs()` draws the extra tensors for a fused spec (`mixed_qkv` replaces
`query`/`key`/`value`), `ref_fp32()` applies the conv first and the norm last,
and `check()` also compares the written conv state and the untouched conv-state
slots. For a `fuse_out_norm` spec `out_err` is **relative to `max(1, |ref|)`**,
not absolute: the normalised output reaches a few units, where one bf16
rounding step of an exact answer already exceeds `TOL` in absolute terms. It
equals the absolute error wherever `|ref| <= 1`:

```python
from builders.gfx950.gdn.gdn_decode import TOL, check
from kernels.gfx950.gdn_decode import GdnDecodeSpec

spec = GdnDecodeSpec(num_k_heads=16, num_v_heads=16, gate_kind="kda",
                     state_dtype="f32", num_warps=4, warp_threads_k=16,
                     blocks_per_v_dim=1, fuse_conv=True, fuse_out_norm=True)
out_err, state_err = check(spec, batch=8)
assert max(out_err, state_err) < TOL
```

The same modes are reachable through dispatch. `GdnDecodeRequest` takes
`state_dtype="f32"` and the `fuse_conv` / `fuse_out_norm` flags; `auto` runs
one measured tile and knob stack per case, gate kind x fused x state width
([`ALGORITHM.md` §4.6](ALGORITHM.md#46-registry-and-tile-selection)).
Launch the result's kernel spec with the builder's inputs, as for any
dispatched spec:

```python
from builders.gfx950.gdn.gdn_decode import launcher_for, make_inputs, run_values
from dispatch.gdn import GdnDecodeRequest, dispatch_gdn_decode

result = dispatch_gdn_decode(GdnDecodeRequest(
    batch=8, arch="gfx950", gate_kind="kda", num_k_heads=16, num_v_heads=16,
    state_dtype="f32", fuse_conv=True, fuse_out_norm=True))
spec = result.spec.kernel_spec
inp = make_inputs(spec, 8)
values = run_values(spec, inp, launcher_for(spec), 8)
```

### Performance knobs

Ten spec fields change how the warp-tiled kernel computes, not what it
computes: `state_load_hint`, `state_store_hint`, `dpp_reduce`, `xcd_remap`,
`stream_rows`, `interleave_cols`, `conv_once`, `norm_gate_once`, `out_lds` and
`waves_per_eu`. Every one defaults to the code the kernel emitted before it
existed, so a default spec keeps its name and its IR. `auto` runs its case's
knob stack (§4.6); a bare `spec_id` pin runs the tile with every knob at its
default (flags off, hints `default`, `waves_per_eu` `0`). Each registered
tile's knob space holds them, so any (tile x knob) configuration is
selectable by pinning its `spec_id` plus `tuning_knobs` (and the `tuning_id` a
sweep recorded with them). `dispatch_gdn_decode_all` lists every configuration
of the listed axes: all nine knob axes, and `waves_per_eu` `{0, 4}` at the
default `production` sweep level or every legal value `0`-`8` at `full`; any
legal `waves_per_eu` is pinnable. `dispatch_gdn_decode_bases` lists only each
tile's base configuration. `tuning_knobs` alone carry a configuration to
another shape (a knob that shape does not read is dropped); a stored
`tuning_id` replays only where all of its knobs are still read, and is
refused elsewhere. Their rules and the tiles where each changes code are in
[`ALGORITHM.md` §4.10](ALGORITHM.md#410-performance-knobs); the registry and
pin rules in [§4.6](ALGORITHM.md#46-registry-and-tile-selection).

```python
result = dispatch_gdn_decode(GdnDecodeRequest(
    batch=8, arch="gfx950", spec_id="nw2_wtk16_bpv8",
    tuning_knobs={"dpp_reduce": True, "state_load_hint": "streaming"}))
result.spec.tuning_id  # "nw2_wtk16_bpv8_wpe0@<config_key>", stored with result.spec.knobs
```

## Benchmark registered candidates

The GDN benchmark asks the registry for every legal tile and times its base
configuration (`dispatch_gdn_decode_bases`: every knob at its default). The
default D128 request admits 54 of 180
configured triples; the benchmark also reports the dispatcher `auto`
configuration:

```bash
python3 library/benchmarks/gfx950/gdn/benchmark_gdn_decode.py \
  --batches 1,16,64,256
```

Each row includes a stable `spec_id`
`nw<num_warps>_wtk<warp_threads_k>_bpv<blocks_per_v_dim>`. Dispatcher `auto`
runs its case's static default whenever its tile is legal; measurements never
change that choice.

KDA has its own benchmark of the dispatched kernel and its variants:

```bash
python3 -m benchmarks.gfx950.gdn.benchmark_kda_decode \
  --batches 1,8,32,128
```

Both benchmarks print eager and device timing. Eager includes host launch and
synchronisation; device timing uses replayed HIP graphs. Small-batch decode can
be launch-bound, so the two clocks answer different questions. If graph capture
is unavailable, pass `--no-device`.

## Measure candidate tiles

`tune.py` consumes the dispatch candidate set. It correctness-checks every tile
against the FP32 reference, graph-times the correct candidates, then reports
them fastest first:

```bash
python3 library/builders/gfx950/gdn/tune.py \
  --gate-kind gdn --batches 1,16,64,256 --top 5
```

For every GDN cell, it also reports the `auto` tile at its base configuration,
the fastest legal candidate, that tile's rank and time ratio, and a manual
review line. Both gate kinds time tiles at their base configuration only (every
knob off), not the `auto` knob stack; measurements never update the shipped
default.

The KDA tile study across head geometries:

```bash
python3 library/builders/gfx950/gdn/tune.py \
  --gate-kind kda --geometries 16/32,8/16,4/8 \
  --batches 1,2,4,8,16,32,64,128 --top 5
```

Both are report-only. Changing an `auto` default needs a cold-memory sweep of
the case's tiles x knobs, which `sweep.py` runs: shard the space over devices,
merge the shards into a ranking (geometric mean over the batches of each
configuration's time relative to the fastest), then re-time the leaders
interleaved:

```bash
for i in 0 1 2; do HIP_VISIBLE_DEVICES=$i python3 -m builders.gfx950.gdn.sweep run \
  --geometries 16/48 --state-dtype f32 --batches 1,2,3,4 --cold 64 --level full \
  --shard $i/3 --jsonl sweep_$i.jsonl & done; wait
python3 -m builders.gfx950.gdn.sweep merge sweep_*.jsonl --top 20
python3 -m builders.gfx950.gdn.sweep retime sweep_*.jsonl --top 12 --rounds 5 --jsonl retime.jsonl
```

`--num-tokens T` sweeps the multi-token step, `--fuse-out-norm` the norm case
(the two compose). The sweep gate bounds each output absolutely (`TOL`) and per
(sequence, token, head) row relative to the reference (`OUT_RTOL`,
`out_rel_error`); a multi-token row of `AUTO_DEFAULTS_MULTI_TOKEN` changes with
its literal table in `test_gfx950_wiring.py` and ALGORITHM.md §4.6.
Absolute times go only to the `--jsonl` files; `merge` and `retime` print
ratios. Change the row in
`AUTO_DEFAULTS`, the literal table in `test_gfx950_wiring.py` and the table in
[`ALGORITHM.md` §4.6](ALGORITHM.md#46-registry-and-tile-selection) together,
then rerun the dispatch wiring, golden and numeric tests.

## Run tests

CPU-only coverage:

```bash
python3 -m pytest \
  library/tests/test_gdn_decode_spec.py \
  library/tests/test_gdn_decode_fused.py \
  library/tests/test_gdn_decode_knobs.py \
  library/tests/test_gdn_decode_golden.py \
  library/tests/test_gdn_decode_prepare.py \
  library/tests/test_gdn_decode_ir_cpp_parity.py \
  library/tests/dispatch/gdn/test_gfx950_registry.py \
  library/tests/dispatch/gdn/test_gfx950_wiring.py \
  library/tests/dispatch/gdn/test_tuning_contract.py \
  library/tests/test_gdn_decode_tune.py \
  -m "not gpu"
```

`test_gdn_decode_ir_cpp_parity.py` needs the `rocke_engine` extension (built
from `platform/cpp/bindings`) on `PYTHONPATH`. Without it the test skips,
unless `ROCKE_BACKEND` is `cpp` or `both`: there the engine is expected, so
its absence fails the test.

On-device numeric coverage:

```bash
python3 -m pytest \
  library/tests/test_gdn_decode_gfx950_numeric.py \
  library/tests/test_kda_decode_gfx950_numeric.py \
  library/tests/test_gdn_decode_fused_gfx950_numeric.py \
  library/tests/test_gdn_decode_knobs_gfx950_numeric.py \
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

`auto` runs one static tile and knob stack per case (gate kind x fused x state
width) whenever the tile is legal; batch and `batch * num_v_heads` change the
grid but not the configuration.

`out_err` and `state_err` are maximum absolute errors against the fp32 reference
(except `out_err` of a `fuse_out_norm` spec, relative to `max(1, |ref|)`; see
[f32 state and fused modes](#f32-state-and-fused-modes)).
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
