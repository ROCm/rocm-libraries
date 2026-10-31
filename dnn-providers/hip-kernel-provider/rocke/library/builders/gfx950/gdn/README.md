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
- [Measure and re-tune defaults](#measure-and-re-tune-defaults)
- [Run tests](#run-tests)
- [Prefill](#prefill)
- [Understand the output](#understand-the-output)
- [Exit codes](#exit-codes)
- [Common failures](#common-failures)

## Files

| File | Purpose |
|---|---|
| [`gdn_decode.py`](gdn_decode.py) | Compile a spec, build inputs, launch the shared decode emitter, and compare with the independent fp32 reference |
| [`library/benchmarks/gfx950/gdn/decode_table_sweep.py`](../../../benchmarks/gfx950/gdn/decode_table_sweep.py) | Time every registered (tile x knob) configuration of each decode shape at the chosen `--sweep-level` (default `production`: `waves_per_eu` 0 and 4 only; `full`: the dispatcher lists every legal configuration, tens of thousands per shape, so the tool requires `--tuning-sample N` and times a random sample of `N` per tile) in cold memory, fp32-gated, in two stages, against `auto`; JSON rows, sharding, summary. It regenerates the `AUTO_DEFAULTS` evidence from rocKE's own registry only |
| [`library/benchmarks/gfx950/gdn/decode_defaults_report.py`](../../../benchmarks/gfx950/gdn/decode_defaults_report.py) | From those rows, `auto` against the best per shape and per case, and an advisory proposal per case by the `AUTO_DEFAULTS` rule |
| [`library/benchmarks/gfx950/gdn/decode_shapes.json`](../../../benchmarks/gfx950/gdn/decode_shapes.json) | The shape grid [`ALGORITHM.md` §4.6](ALGORITHM.md#46-registry-and-tile-selection) documents for each `AUTO_DEFAULTS` case (3 head geometries x 7 batches, 8 cases) |
| [`ALGORITHM.md`](ALGORITHM.md) | Explain the gated delta rule, gate kinds, and GPU mapping |
| [`CASE_STUDY.md`](CASE_STUDY.md) | Per-tile register, wave and scratch changes of the decode load-first order, per compiler, the spill snapshot of the forced `row_loads` order (§8), and the commands that regenerate them |
| [`library/benchmarks/gfx950/gdn/benchmark_gdn_decode.py`](../../../benchmarks/gfx950/gdn/benchmark_gdn_decode.py) | Benchmark every legal GDN registry tile (base configuration) and the dispatcher `auto` configuration (its case's tile and knob stack) |
| [`library/benchmarks/gfx950/gdn/benchmark_kda_decode.py`](../../../benchmarks/gfx950/gdn/benchmark_kda_decode.py) | Benchmark KDA fused/precomputed/simple variants from the dispatcher; optionally time every legal tile's base configuration and `auto` (`decode_table_sweep.py` stage 1) |
| [`library/dispatch/gdn/gate_rules.py`](../../../dispatch/gdn/gate_rules.py), [`cli_args.py`](../../../dispatch/gdn/cli_args.py) | The torch-free rules the driver and every tool share: the fp32 pass test (`Gate`, `TOL`), the cold-memory copy-count arithmetic, `--batches` / `--rotate-mb` parsing, the NaN-safe maximum |
| [`library/dispatch/gdn/gfx950.py`](../../../dispatch/gdn/gfx950.py) | Register every (gate kind, tile) and select the `auto` configuration of each case; the per-case defaults are data in [`axes.py`](../../../dispatch/gdn/axes.py) (`AUTO_DEFAULTS`) |
| [`library/dispatch/gdn/axes.py`](../../../dispatch/gdn/axes.py), [`rules.py`](../../../dispatch/gdn/rules.py) | Each tile's knob space: the knob axes as data, and their scope and legality |
| [`library/tests/dispatch/gdn/test_gfx950_registry.py`](../../../tests/dispatch/gdn/test_gfx950_registry.py) | CPU GDN registry count, identity, legality, and selection coverage |
| [`library/tests/dispatch/gdn/test_tuning_contract.py`](../../../tests/dispatch/gdn/test_tuning_contract.py) | CPU tuning contract: knob space per mode, pins (`tuning_id` / `tuning_knobs`), refusals |
| [`library/tests/dispatch/gdn/test_gfx950_wiring.py`](../../../tests/dispatch/gdn/test_gfx950_wiring.py) | CPU dispatch wiring: the literal `AUTO_DEFAULTS` table and `auto` selection per case |
| [`library/tests/test_gdn_decode_spec.py`](../../../tests/test_gdn_decode_spec.py) | CPU validator and IR-emission coverage |
| [`library/tests/test_gdn_decode_prepare.py`](../../../tests/test_gdn_decode_prepare.py) | Host-side input validation: shapes, dtypes, contiguity, devices, pointer alignment, pool-index range, fused-mode inputs |
| [`library/tests/test_gdn_decode_fused.py`](../../../tests/test_gdn_decode_fused.py) | CPU contract of the fused conv1d / gated-RMSNorm modes: spec rules, ABI, IR alignment claims, tap-shift barrier, fp32 reference |
| [`library/tests/test_gdn_decode_fused_gfx950_numeric.py`](../../../tests/test_gdn_decode_fused_gfx950_numeric.py) | On-device fused-mode output, state and conv-state correctness |
| [`library/tests/test_gdn_decode_gfx950_numeric.py`](../../../tests/test_gdn_decode_gfx950_numeric.py) | On-device GDN output and state correctness |
| [`library/tests/test_kda_decode_gfx950_numeric.py`](../../../tests/test_kda_decode_gfx950_numeric.py) | On-device KDA output/state correctness and dispatch-to-launch coverage |
| [`library/tests/test_gdn_decode_knobs.py`](../../../tests/test_gdn_decode_knobs.py) | CPU contract of the performance knobs: validation, `*_applies` predicates, name injectivity, emitted effect |
| [`library/tests/test_gdn_decode_knobs_gfx950_numeric.py`](../../../tests/test_gdn_decode_knobs_gfx950_numeric.py) | On-device correctness of every knob on every registered tile (and every `BPV=1` fused tile) it applies to at the default heads, at head dims 64 and 256, `out_lds` at `DV` 256 / 512, and with f16 I/O and an f16 state |
| [`library/tests/test_gdn_decode_golden.py`](../../../tests/test_gdn_decode_golden.py) | Detect unexpected LLVM-IR changes in both gate kinds (Python lowering) |
| [`library/tests/test_gdn_decode_ir_cpp_parity.py`](../../../tests/test_gdn_decode_ir_cpp_parity.py) | The C++ engine lowers every golden case to byte-identical LLVM IR |
| [`library/tests/dispatch/gdn/test_gate_rules.py`](../../../tests/dispatch/gdn/test_gate_rules.py) | CPU contract of `gate_rules.py`, with no torch: what passes the gate (finite, within `TOL`, no spill; a NaN never passes), the rotation sizing, and that the driver keeps no copy of either |
| [`library/tests/test_gdn_decode_timing.py`](../../../tests/test_gdn_decode_timing.py) | The driver's tensor code on CPU tensors (needs torch): cold copy sets, timing labels, batch and index validation, and that the gate rejects a perturbed output, state or untouched slot |
| [`library/tests/test_gdn_decode_table_sweep.py`](../../../tests/test_gdn_decode_table_sweep.py) | CPU contract of `decode_table_sweep.py`: walks, shards, stage 1 / stage 2 files, summaries, exit codes and refused flags, and that a configuration whose verdict fails is never timed or ranked |
| [`library/tests/test_gdn_decode_defaults_report.py`](../../../tests/test_gdn_decode_defaults_report.py) | CPU contract of `decode_defaults_report.py`: the selection rule, the tie band, withheld proposals, exit codes |
| [`library/tests/test_kda_decode_benchmark.py`](../../../tests/test_kda_decode_benchmark.py) | CPU contract of `benchmark_kda_decode.py`: variant parsing, `--batches` validation (also of `benchmark_gdn_decode.py`), failure collection, exit code, and that both benchmarks never time a launch whose verdict fails (no torch needed) |
| [`library/builders/gfx950/kda/gdn_prefill.py`](../../kda/gdn_prefill.py) | Drive chunkwise prefill (the KDA chunkwise kernels in `gate_kind="gdn"` mode) and hold its fp64 oracle |
| [`library/benchmarks/gfx950/gdn/sweep_prefill_value_splits.py`](../../../benchmarks/gfx950/gdn/sweep_prefill_value_splits.py) | Sweep `value_splits` for prefill at a given `batch_heads` |
| [`library/tests/dispatch/gdn/test_gfx950_prefill_wiring.py`](../../../tests/dispatch/gdn/test_gfx950_prefill_wiring.py) | Prefill dispatch: candidate selection, the two-launch guard, launch geometry |
| [`library/tests/test_kda_chunkwise_spec.py`](../../../tests/test_kda_chunkwise_spec.py) | CPU chunkwise KDA spec coverage: the prep, fused and scan specs' admission and rejection rules, kernel naming, and each builder's code object against the resource budget (no GPU) |
| [`library/tests/test_gdn_prefill_decay_guard.py`](../../../tests/test_gdn_prefill_decay_guard.py) | Pin the supported envelope of the unbounded GDN decay gate |
| [`library/tests/test_kda_gdn_gfx950_numeric.py`](../../../tests/test_kda_gdn_gfx950_numeric.py) | On-device prefill correctness in `gate_kind="gdn"` mode |

## Environment

Run from `dnn-providers/hip-kernel-provider/rocke` with both the library and the
platform Python package on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/library:$PWD/platform/python${PYTHONPATH:+:$PYTHONPATH}"
```

The driver, benchmarks, sweep, and numeric tests require:

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
B=1     grid=<workgroups> out_err=<error> state_err=<error> conv_err=<error> spill=False OK
B=16    grid=<workgroups> out_err=<error> state_err=<error> conv_err=<error> spill=False OK
0 batches failed the fp32 gate (tol=1.0e-02)
```

`check()` runs one launch and scores it with `gate()`, the one fp32 pass test
every tool here that gates uses (all but `gdn_decode.py --no-check`, below;
`Gate.passed`, defined with no torch in `dispatch/gdn/gate_rules.py`):

- `out_err`: maximum absolute error in this token's output (for a
  `fuse_out_norm` spec, relative to `max(1, |ref|)`; see below);
- `state_err`: maximum absolute error in the written recurrent state slots;
- `conv_err`: the same for the written conv-state slots (`fuse_conv` only);
- `spill`: whether any slot the kernel was not told to write changed at all.

It passes when every error is finite and within `TOL` and nothing spilled; a
NaN anywhere fails. Checking only `out` is insufficient because a bad state
write may not affect the visible output until the next decode step.

Check the simple one-thread-per-row reference body:

```bash
python3 library/builders/gfx950/gdn/gdn_decode.py \
  --batches 1,16 --variant simple
```

Also report the cold-memory device time of each batch (see
[Benchmark registered candidates](#benchmark-registered-candidates)):

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
from builders.gfx950.gdn.gdn_decode import check
from kernels.gfx950.gdn_decode import GdnDecodeSpec

spec = GdnDecodeSpec(num_k_heads=16, num_v_heads=16, gate_kind="kda",
                     state_dtype="f32", num_warps=4, warp_threads_k=16,
                     blocks_per_v_dim=1, fuse_conv=True, fuse_out_norm=True)
verdict = check(spec, batch=8)
assert verdict.passed, verdict
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

Eleven spec fields change how the warp-tiled kernel computes, not what it
computes (the two state hints and `waves_per_eu` also apply on the reference
path): `state_load_hint`, `state_store_hint`, `dpp_reduce`, `xcd_remap`,
`stream_rows`, `interleave_cols`, `conv_once`, `norm_gate_once`, `out_lds`,
`waves_per_eu` and `row_loads`. Every one defaults to the code the kernel
emitted before it existed, so a default spec keeps its name and its IR.
`row_loads` (`"auto"`, `"first"`, `"late"`) forces where a single-wave
`fuse_conv` kernel loads its rows' late inputs (§4.3, "One exception"); `auto`
dispatch never sets it. `auto` runs its case's knob stack (§4.6); a bare
`spec_id` pin runs the tile with every knob at its default. Each registered
tile's knob space holds them: every (tile x knob) configuration is listed by
`dispatch_gdn_decode_all` (and each tile's base configuration by
`dispatch_gdn_decode_bases`) and pinnable by its `spec_id` plus `tuning_knobs`
and/or `tuning_id`. On single-wave `fuse_conv` tiles each knob set at each
`waves_per_eu` value is also listed in the `row_loads` order auto does not
choose there. Adding `row_loads` changed every `config_key` and every result's
`spec_hash`, `compile_key` and `selection_key` (`auto`'s included; its kernel
name and IR are unchanged, so it costs one recompile): re-sweep a
`tuning_id` recorded before it. A knob that changes no code on another shape
(one that shape does not read, or a `row_loads` that restates auto's order
there) is dropped from `tuning_knobs`; a stored `tuning_id` replays only where
none of its knobs is dropped. The sweep levels and pin rules are in
[`ALGORITHM.md` §4.6](ALGORITHM.md#46-registry-and-tile-selection); each knob's
rules and the tiles where it changes code are in
[§4.10](ALGORITHM.md#410-performance-knobs).

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
synchronisation; device timing is the median per-launch time of replayed HIP
graphs. Small-batch decode can be launch-bound, so the two clocks answer
different questions. If graph capture is unavailable, pass `--no-device`.

`--batches` of both benchmarks, of `decode_table_sweep.py` and of `gdn_decode.py`
takes comma-separated positive integers, parsed by one function
(`parse_batches` in `dispatch/gdn/cli_args.py`, which also holds the one
`--rotate-mb` rule, `>= 0`, and `gate_rules.py` beside it the copy-count
arithmetic behind `--rotate-mb`); a non-integer, an empty item, `0`
or a negative value is refused with exit 2 before any device is touched.

All timing (both benchmarks, `decode_table_sweep.py` and
`gdn_decode.py --bench`) is cold memory, through the driver's shared helpers
(`gdn_decode.py`: `device_us`, `eager_us`, `capture_cold`). The tools rotate
through independent copies of the inputs and state, sized by `--rotate-mb`
(what it guarantees is in `gdn_decode.cold_copies`), and report medians. A
buffer set is reused only after at least `--rotate-mb` MiB, less one set, of
other traffic. The `--rotate-mb` default is 1024 MiB: it is meant to exceed
the caches, and this is not measured here. A smaller positive value may leave
the state cache-resident, so those times are not comparable with cold ones and
are for debugging; the output then labels the timing as partly warm, and
`decode_defaults_report.py` withholds its proposal from such rows.
`--rotate-mb 0` times warm, for debugging only, and the output says so in a
`WARNING` line.

## Measure and re-tune defaults

Two tools measure the registry and propose `auto` defaults. Neither changes
what `auto` selects. After `pip install -e library` they are also the
`rocke-gdn-decode-table-sweep` and `rocke-gdn-decode-defaults-report` commands.

**`decode_table_sweep.py`** walks `dispatch_gdn_decode_all` for each decode
*shape* (gate kind, `--mode plain|fused`, state dtype, `Hk/Hv`, head dim,
batch). `--mode fused` means both `fuse_conv` and `fuse_out_norm` (and
`fuse_conv` needs `Hk == Hv`); a shape's single head dim sets both `DK` and
`DV`. The walk is every registered tile x knob configuration at `--sweep-level`
(`production`: every knob set at `waves_per_eu` 0 and 4; `full` with
`--tuning-sample N`: N random configurations per tile). It checks each
configuration once against the fp32 reference on disjoint read and write slots
(output, written state and conv-state slots, untouched slots bit-exact), then
times the ones that pass, on in-place slots as a serving engine updates them:

1. Stage 1: each configuration on its own, median of `--rounds` (5) replays.
2. Stage 2: the shape's `--top-k` (8) fastest plus `auto`, re-timed together:
   replays interleaved, the order reversed every round and changed between
   runs, `--s2-rounds` (30) rounds x `--runs` (3); the reported time is the
   median of the run medians.

`auto` is always timed. A JSON row carries, in this order of origin:

- the run settings (`run_meta`): `arch`, `device`, `sha` (the source sha;
  `-dirty` when tracked files differ), the toolchain (`rocm_version`,
  `llvm_version`, `torch_version`, `backend`, `llvm_flavor`; `unknown` where one
  could not be read), the walk (`sweep_level`, `tuning_sample`, `seed`,
  `bases_only`), `rotate_mb`, `cold_target_mb`, `rounds`, `s2_rounds`, `top_k`;
- the shape (`gate_kind`, `mode`, `state_dtype`, `num_k_heads`, `num_v_heads`,
  `head_dim`, `batch`), its `case` and the `stage`;
- the configuration: `index`, `spec_id`, `tile`, `tuning_id`, `knobs`, `auto`;
- the result: `status` (`ok`, `fail` = fp32 gate, `error`, `no_graph`),
  `reason`, `us` (the median time), `runs` (the per-run medians), `out_err`,
  `state_err`, `conv_err`, `spill` and `copies` (the buffer sets).

A shape the dispatcher refuses has one `error` row with `index` -1 and no
`spec_id`, `tile` or `tuning_id`; the stage-2 `auto` row that stage 1 did not
list also has `index` -1. `ROW_KEYS` in `decode_table_sweep.py` lists
the keys every reader needs. Any row's `spec_id` + `tuning_id` + knobs pin that
configuration (see [Performance knobs](#performance-knobs)). The run, and
`--summarize`, exit 1 if any row of any stage is not `ok`; `--summarize` of no
rows exits 1.
A run that times nothing exits 1 when `--offset` / `--limit` leave no
configuration of any shape. A `--shard` that owns nothing while another shard
does says so and exits 0: so does `--stage1-json F --shard I/N` with `I` past
the last recorded shape (a small shape set may be split over more shards than
it has shapes). No recorded shape at all exits 2.
A `--tuning-ids` entry the walk does not list is an error, never dropped; an
empty list or empty entry is refused. `--tuning-sample N` needs
`--sweep-level full` and `full` needs `N > 0`; `--bases-only` is refused with
`full` (a sampled walk keeps only the bases the sample happens to contain, not
each tile's). Flags a mode would ignore are
refused with exit 2 and a message naming the flag; a flag is detected only when
its value differs from its default, so an explicit `--rounds 5` is accepted.
With `--stage1-json`, none of the flags it would ignore: `--rounds`,
`--stage1-only`, `--list-only`, `--offset`, `--limit`, `--shapes`,
`--bases-only`, `--tuning-ids`, `--sweep-level`, `--tuning-sample` or `--seed`.
With `--list-only` none of `--output-json`, `--rotate-mb`, `--rounds`,
`--s2-rounds`, `--runs`, `--top-k`, `--stage1-only` or `--source-sha`. With
`--stage1-only` none of `--top-k`, `--s2-rounds` or `--runs`. With
`--summarize` no other flag. A file that cannot be read, is not valid JSON or is
not a sweep file (`--shapes`, `--stage1-json`, `--summarize`) exits 2 with the
path and cause, as does the report's input; so does a saved row that lacks a key
the tools read (`ROW_KEYS`) or holds a value they cannot use (a shape the tool
refuses, a `stage` not 1 or 2, a `status` the tool never writes, a `us` that is
not a finite number in a sane range (> 0) on an `ok` row or not null on any other, an `index`
below -1, or -1 on a stage-1 row that is not an `error` row, a `tuning_id` that
is not a string (or empty, except on that `error` row), a `spec_id` or `tile`
that disagrees with it (empty exactly when it is), `knobs` that is not an
object, an `auto` that is not a bool, a `reason` that is neither a string nor
null, an empty or non-string `sha` or `device`, an unknown `sweep_level`, a
`tuning_sample` (below 0) or `seed` that is not an int, a `bases_only` that is not a
bool, a `rotate_mb` that is not a finite number >= 0 or a `cold_target_mb` that
is not a finite number > 0), or a
`--shapes` entry that lacks a required key or is malformed (`heads` or
`batches` not a list, a head that is not an `Hk/Hv` string, a value the shape
refuses), naming the row or entry.
Stage 2 from saved rows takes the sweep level, sample, seed and `--bases-only`
from stage 1, and
refuses stage-1 rows that disagree. The header says: the sweep gates disjoint
slots, one launch, and times in-place slots (the timed in-place launch is not
itself gated).

```bash
# One shape, both stages.
python3 -m benchmarks.gfx950.gdn.decode_table_sweep \
  --gate-kind kda --mode fused --state-dtype f32 --heads 32/32 --batches 16 \
  --output-json kda_fused_f32.json

# Each tile's base configuration (and auto) only, stage 1: the quick tile view.
python3 -m benchmarks.gfx950.gdn.decode_table_sweep \
  --gate-kind gdn --heads 16/32 --batches 1,16,64,256 --bases-only --stage1-only

# What a run would time, on a CPU host.
python3 -m benchmarks.gfx950.gdn.decode_table_sweep \
  --shapes library/benchmarks/gfx950/gdn/decode_shapes.json --list-only

# Best configuration per shape against auto, from saved rows.
python3 -m benchmarks.gfx950.gdn.decode_table_sweep --summarize 'out/*.json'
```

`--shapes FILE` reads a JSON list of entries (`gate_kind`, `mode`,
`state_dtype`, `heads` list, `batches` list, optional `head_dim`); the shape
flags then filter it. A long run splits across GPUs: `--shard I/N` takes every
N-th configuration of each shape in stage 1 (`--offset` / `--limit` take a
window, to resume), and stage 2 then runs from the saved stage-1 rows, with
`--shard` splitting the shapes. For example, one case on an `N`-task array job:

```bash
# --shard I/N needs 0 <= I < N, so count the array index from zero
# whatever its first index is.
I=$((SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN)) N=$SLURM_ARRAY_TASK_COUNT
python3 -m benchmarks.gfx950.gdn.decode_table_sweep \
  --shapes library/benchmarks/gfx950/gdn/decode_shapes.json \
  --gate-kind kda --mode plain --state-dtype bf16 \
  --stage1-only --shard $I/$N --output-json s1_$I.json
# after every stage-1 task finished:
python3 -m benchmarks.gfx950.gdn.decode_table_sweep \
  --stage1-json 's1_*.json' --shard $I/$N --output-json s2_$I.json
```

**`decode_defaults_report.py`** reads those rows and prints, per case
(`AUTO_DEFAULTS` row: gate kind x fused x state width), `auto` and the
proposal against the fastest configuration of every shape, `auto`'s score
(the geomean of those ratios) and worst shape, and the proposed default by
the rule `AUTO_DEFAULTS` was chosen with
([`ALGORITHM.md` §4.6](ALGORITHM.md#46-registry-and-tile-selection)): the step
of the rule that decided, its score, worst shape, non-default knob count, its
score over `auto`'s, a ready-to-paste `AUTO_DEFAULTS` entry and a shortlist.
Beside the `rows from sha ... on device` line it prints the rows' toolchain
(ROCm, LLVM, torch, lowering backend and LLVM flavor), one line per distinct
set, so rows timed with different toolchains show it; a mismatch withholds
nothing.
A shape with no timed row, or where `auto` failed, is listed as `UNSCORED`
and the report exits 1. It also exits 1 when rows are warm or below the cold
target, or mix `--rotate-mb` values: it then withholds everything (a
`WITHHELD` line and no case). Rows from mixed walks (different `--sweep-level`,
`--tuning-sample`, `--seed` or `--bases-only`) withhold only the proposal of
the case they mix in; the other cases still print. Each proposal prints the
sweep level and walk beside it. It is advisory. To revisit a case:

1. Stage 1 of every shape of the case (above, `decode_shapes.json` holds the
   shape grid of every case).
2. `python3 -m benchmarks.gfx950.gdn.decode_defaults_report 's1_*.json'` scores
   every configuration timed at every shape and prints a `re-time:` line: the
   stage-1 walk flags and a `--tuning-ids` shortlist (the top candidates, the
   proposal and each shape's fastest).
3. Re-time the shortlist together at every shape (both stages; split the jobs
   by `--heads` / `--batches`):
   `decode_table_sweep.py --shapes ... --gate-kind ... <the re-time: flags>
   --output-json retime_<part>.json`, i.e. `--sweep-level`, `--tuning-sample`
   and `--seed` (and `--bases-only`, when stage 1 had it) as in stage 1,
   `--tuning-ids <list> --top-k <its length>`.
4. `python3 -m benchmarks.gfx950.gdn.decode_defaults_report --stage 2
   'retime_*.json'` gives the proposal at stage-2 precision.
5. A person reviews it and changes the row in `AUTO_DEFAULTS`, the literal
   table in `test_gfx950_wiring.py` and the table in
   [`ALGORITHM.md` §4.6](ALGORITHM.md#46-registry-and-tile-selection) together,
   then reruns the dispatch wiring, golden and numeric tests.

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
  library/tests/dispatch/gdn/test_gate_rules.py \
  library/tests/test_gdn_decode_timing.py \
  library/tests/test_gdn_decode_table_sweep.py \
  library/tests/test_gdn_decode_defaults_report.py \
  library/tests/test_kda_decode_benchmark.py \
  -m "not gpu"
```

`test_gdn_decode_ir_cpp_parity.py` needs the `rocke_engine` extension (built
from `platform/cpp/bindings`) on `PYTHONPATH`. Without it the test skips,
unless `ROCKE_BACKEND` is `cpp` or `both`: there the engine is expected, so
its absence fails the test.

Without torch (as in CI) `test_gdn_decode_timing.py` and
the two tensor-driven cases of the sweep and KDA benchmark tests skip; the
gate's pass test, the rotation sizing, and what the sweep and both benchmarks
do with a failing verdict run anyway (`test_gate_rules.py` and the verdict
tests in the sweep and benchmark files). The tensor reductions inside `gate()`
(the error values and the untouched-slot comparison) run only where torch is
installed: `test_gdn_decode_timing.py` on CPU tensors, and the on-device
numeric files below. Install CPU torch to run all of them.

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
| `gdn_decode.py` | all requested checks passed, or checks were disabled | at least one checked batch exceeded tolerance | no GPU visible, the fixed spec was rejected, or a refused flag (a negative `--rotate-mb`, a `--variant` that is not `tiled` or `simple`, a malformed or non-positive `--batches`) |
| `benchmark_gdn_decode.py` | every batch passed correctness (timing may print `n/a` when graph capture is unavailable, and always with `--no-device`) | at least one batch exceeded tolerance, or the registry yielded other than 54 candidates | no GPU visible, or a refused flag (a negative `--rotate-mb`, a malformed or non-positive `--batches`) |
| `benchmark_kda_decode.py` | every requested variant/sweep passed correctness and timing | any requested variant/sweep failed or was incomplete | no ROCm torch or no visible HIP device, the visible device is not gfx950, or a refused flag (a negative `--rotate-mb`, a malformed or non-positive `--batches`, an unknown `--variants` name) |
| `decode_table_sweep.py` | every row passed the gate and was timed, and every shape timed `auto` (also `--list-only`; `--stage1-json F --shard I/N` with `I` past the last recorded shape) | some configuration failed the gate, raised or could not be timed, a shape has no timed `auto` (not checked in a `--shard` / `--offset` / `--limit` stage-1 run, which times `auto` in one window), the run timed nothing (`--offset` / `--limit` leave no configuration of any shape), `--summarize` found no rows, or (`--list-only`) a shape was refused | a refused flag or flag combination (`--bases-only` with `--sweep-level full` included), an unreadable, non-sweep or malformed input file (a `--shapes` entry or a saved row with a missing key or a bad value), stage-1 rows from different walks, no gfx950 device visible, or no torch |
| `decode_defaults_report.py` | a proposal and an `auto` score for every case | some case has no configuration timed at every shape, or `auto` untimed at a shape; rows from mixed walks or mixed `--rotate-mb`; or warm rows (below the cold target), where the proposal is withheld | a refused flag value, a missing file argument, or an unreadable, non-sweep or malformed (a row lacks a key or holds a bad value) input file (argparse) |

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
