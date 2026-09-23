<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# wgrad LDS layout and occupancy — gfx950 case study

Per [`platform/AGENTS.md`](../../../../../AGENTS.md) §Compliance this document
records **methodology and levers only**. No measured throughput, latency, or
achieved-FLOP figure appears here or in any commit message; those live on the
protected AMD Confluence page. Ratios quoted below are described qualitatively.

## Table of contents

- [What was investigated](#what-was-investigated)
- [The instrument](#the-instrument)
- [Finding 1: transpose-on-store serialises the global loads](#finding-1-transpose-on-store-serialises-the-global-loads)
- [Finding 2: a correctness hole in the K-outer fragment mapping](#finding-2-a-correctness-hole-in-the-k-outer-fragment-mapping)
- [Finding 3: the surviving limit is occupancy, not the epilogue](#finding-3-the-surviving-limit-is-occupancy-not-the-epilogue)
- [Rejected: widening the split-K atomic epilogue](#rejected-widening-the-split-k-atomic-epilogue)
- [Not evaluated: the portable M-outer fix](#not-evaluated-the-portable-m-outer-fix)
- [Replay](#replay)
- [Config table](#config-table)
- [Keep / revert decisions](#keep--revert-decisions)

## What was investigated

Backward-weight convolution (`--direction wgrad`) on gfx950 was materially
behind the reference implementation for a 3x3 bf16 shape. The question was
whether the gap was structural or a tuning artifact.

Wgrad's GEMM is skinny and very deep: `M = K`, `N = Y*X*C`, and the reduction
axis is `N*Ho*Wo`. Both operands are contiguous along the GEMM's *free* axis
and strided along its reduction axis, so feeding the MFMA requires a transpose
somewhere. Where that transpose is placed is the whole subject of this study.

## The instrument

[WaveScope](../../../../../dsl_docs/architecture/wavescope_integration.md) ATT
traces with source correlation, captured with
[`capture_wavescope_trace.py`](../../../../../dsl_docs/optimization/utilities/tools/wavescope/capture_wavescope_trace.py).
The per-instruction stall totals joined to the Python authoring stack via
`inline_frames.json` are what made each finding falsifiable — every claim below
is anchored to a line the sidecar attributed, not to a reading of the source.

Two notes for anyone replaying this:

- The ATT decoder is not part of a stock ROCm install. Point
  `ROCPROF_TRACE_DECODER_LIB` at a directory containing
  `librocprof-trace-decoder.so`.
- `code.json`'s stall and latency columns are hit-weighted **totals**. Divide by
  `Hit`. Multiplying instead yields figures larger than the kernel's wall clock,
  which is the tell.
- Never combine a numeric verify and a timing loop in one process: rocke's HIP
  runtime and torch's fight over the process HIP context, and the timings are
  meaningless. Time in a separate process.

## Finding 1: transpose-on-store serialises the global loads

The default (M-outer) LDS tile transposes **on store**, in
[`helpers/loads.py`](../../../../helpers/loads.py) `_store_tile`, the
`vector_axis == "row"` branch: one 1-element `smem_store_vN` per free-axis
element. With `load_vec = 8` each 16-byte global load becomes eight 2-byte LDS
writes plus their address arithmetic.

The trace showed the main K-loop spending only a small minority of wave time in
`EXEC`, with `s_waitcnt` the dominant stall class and the LDS-write class
second. Crucially, the top stalling instructions were eight separate
`s_waitcnt vmcnt(0)` — one per `buffer_load_dwordx4` — all attributed to
`loads.py`'s store line.

That is the real damage, and it is second-order: each load's result feeds eight
`vec_extract` + eight scattered stores with eight computed addresses, so
keeping eight loads in flight exceeds the register budget and the backend
serialises them. Every load's full latency is exposed. The K-outer tile issues
all eight loads first and drains them with `vmcnt(7)`, `vmcnt(6)`, … — a proper
software pipeline — because each load has exactly one wide consumer.

The instruction-count ratio in the K-loop (instructions issued per MFMA) tracked
the measured throughput ratio almost exactly, which is the signature of an
issue-bound loop rather than a bandwidth-bound one.

**Lever: prefer the K-outer LDS tile (`lds_k_outer=True`) wherever the gfx950
transpose read is available.** This is not an exotic layout — it is what the
reference implementation does; the transpose read is only the gfx950
accelerator on top of it.

A control experiment worth recording: all five pipelines (`mem`, `compv3`,
`compv4`, `wavelet`, `basic`) land within a few percent of each other on the
M-outer path, including the two that already batch loads ahead of stores. The
scatter is inherited by every pipeline, so this is **not** fixable by pipeline
scheduling.

## Finding 2: a correctness hole in the K-outer fragment mapping

The K-outer path was gated behind an opt-in flag, and sweeps had been run
without `--verify`. Verifying the same config matrix with and without the flag
isolated a failure that reproduces only with `lds_k_outer=True` and only for the
`16x16x16` atom.

The transpose-read lane mapping steps the k index by the MFMA operand length:
lane `l` owns `k = (l // MN)*n .. +n-1`. `n` is 8 for `32x32x16` and `16x16x32`
but **4** for `16x16x16`. The stride was hardcoded to 8, so the 4-element atom
read k rows 8..27 of a 16-row tile — past the end — and produced NaN.

Why it survived: every k-outer test pinned `_WARP_TILE_MN = 32`, and both
parity configs used `32x32x16`. Nothing in the tree instantiated the only 16-bit
atom with a 4-element fragment.

Fixed in both engines by carrying `n` instead of the literal. The emitted IR is
unchanged for the two 8-element atoms, so no golden moved. Coverage added at
both levels: parity config 13 (engine divergence) and
`test_lds_k_outer_atom_16x16x16` (numeric). The numeric test was confirmed to
fail with the fix reverted and pass with it restored — a regression test that
was never observed to fail is not evidence.

## Finding 3: the surviving limit is occupancy, not the epilogue

Re-tracing after the layout fix, the K-loop's largest single cost became the
global-load wait, attributed to the wide LDS store line. The kernel had become
memory-latency bound at roughly two waves per SIMD — too few to cover the
latency.

The productive lever was **waves per workgroup**, not prefetch depth. Sweeping
`warp_m x warp_n` found a clean interior optimum at `2x2` (four waves): both
`1x2` and the eight-wave configurations (`4x2`, `2x4`) are meaningfully worse.
Pipeline choice at the optimum is within measurement noise across all five,
which confirms the limit is occupancy rather than software pipelining. Note
that only `compv4`, `async_dma` and `unroll_k` double-buffer at all
([`conv_implicit_gemm_wgrad.py`](../../../instances/common/conv_implicit_gemm_wgrad.py)),
and double buffering did not separate from the rest here.

The prior tuning data was collected on the M-outer path and does not transfer:
the M-outer optimum and the K-outer optimum differ in atom, pipeline **and**
warp geometry. Re-sweep after changing the layout.

## Rejected: widening the split-K atomic epilogue

After the layout fix the epilogue became the second-largest cost class, so it
was attacked directly: hoist the loop-invariant bounds predicate and offset
base, widen the per-chunk LDS read from `sv` narrow reads to one wide read, and
reassociate the address so the element index folds into the instruction's
immediate offset.

This reduced kernel instruction count and LDS reads measurably and was
numerically correct — and produced **no throughput change** at any split-K
degree, including a deliberately epilogue-heavy one, with deltas below the
run-to-run spread.

The reason is arithmetic and was predictable: the packed atomic count is
`split_k * M * N / 2`, invariant to store vector width, tile shape and block
size, because `sv` cancels. `global_atomic_pk_add_bf16` is capped at two
elements. So the epilogue's cost is atomic **latency and contention**, not the
instructions around it, and reducing the surrounding instructions cannot help.

**Reverted.** It would have cost a C++ hand-port and a golden re-bless for no
measured benefit. Recorded here so it is not re-attempted.

The genuine lever for this cost class is an fp32 workspace plus a separate
reduction pass, which would also remove a real numerics liability: today each
CTA rounds its partial to bf16 *before* the atomic, so a degree-32 split
performs 32 round-to-bf16 events and 32 bf16 adds in nondeterministic order.
Nothing in the tree supports a partial-tensor reduction today. Not attempted.

## Not evaluated: the portable M-outer fix

For targets without the gfx950 transpose read, the equivalent of Finding 1
would be a cross-lane register transpose (`ds_bpermute_b32` / `ds_swizzle_b32`,
both available on gfx9\* and gfx95\*) before a single wide LDS write.

This was **not** evaluated, for two reasons. First, its target hardware was not
present — this box exposes gfx950 only, and per
[`platform/AGENTS.md`](../../../../../AGENTS.md) a lane must not be faked.
Second, a first-principles count is discouraging: an 8-way transpose of 2-byte
elements needs on the order of 8 shuffles, against the 8 narrow stores it would
replace. Packing pairs into dwords and transposing 4x4 improves that but not
decisively. Treat it as an open question requiring gfx942 hardware, not as a
queued task.

## Replay

From `platform/`, with `PYTHONPATH=$(pwd)/python`.

Correctness — the regression test for Finding 2, and the full suite:

```bash
python3 -m pytest tests/instances/test_conv_wgrad_correctness.py -q
python3 -m pytest tests/instances/test_conv_wgrad_correctness.py -q \
    -k "atom_16x16x16 or LdsKOuter"
```

Byte identity, both flavors (the merge gate for the two-engine mirror):

```bash
export ROCKE=$(pwd) PYTHONPATH=$ROCKE/python
python3 tools/check_byte_identity.py --only conv_implicit_gemm_wgrad
ROCKE_LLVM_FLAVOR=llvm22 python3 tools/check_byte_identity.py
```

Step 0 lever sweep. Shapes come from `--miopen-cmd` (or `--miopen-file` for a
batch); there are no per-dimension flags:

```bash
python3 python/rocke/benchmark/benchmark_implicit_gemm_conv.py \
    --direction wgrad --arch gfx950 --dtype bf16 \
    --miopen-cmd "./MIOpenDriver convbfp16 -n 8 -c 128 -H 32 -W 32 -k 128 \
        -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F 4 -in_layout=NHWC" \
    --split-k 0 --split-k-prune 15 --jobs 32 --sample 0.03 --seed 0 \
    --top 5 --warmup 3 --iters 10
```

**There is no `--lds-k-outer` flag to A/B against.** It was removed along with
`--lds-k-pad`, `--dtype-d` and the `ROCKE_WGRAD_LDS_K_OUTER` env override once
the layout became a pure function of `(arch, dtype, atom, wave)` —
`WgradConvSpec.default_lds_k_outer`, which both the sweep driver and library
dispatch call. The sweep therefore reports the deduced layout, not a choice.

To A/B the layout itself, build both specs in-process and compare, which is
what the numeric tests do — they assert the two are bitwise identical, so this
doubles as the correctness gate:

```bash
python3 -m pytest tests/instances/test_conv_wgrad_correctness.py -q -rs -k k_outer
```

Read the `-rs` skip list. A sweep whose subTests all skip still reports
`passed`; `_assert_ran()` makes that a hard failure for the K-outer sweeps, but
only for those.

Source-correlated trace:

```bash
export ROCPROF_TRACE_DECODER_LIB=<dir containing librocprof-trace-decoder.so>
python3 dsl_docs/optimization/utilities/tools/wavescope/capture_wavescope_trace.py \
    --output-dir ./att_out --kernel-regex "rocke_bench_igemm_wgrad.*kouter_spkrt" \
    -- python3 <single-config driver>
```

Open the `ui_output_*_dispatch_*` directory (not its parent) with
**WaveScope: Open Trace Folder…**, and switch the Source tab to `+ inlined`.

## Config table

| Axis | M-outer optimum | K-outer optimum |
| --- | --- | --- |
| tile (m, n, k) | 64, 64, 64 | 64, 64, 64 |
| warp (m, n) | 1, 2 | **2, 2** |
| atom edge | 32 | 16 |
| pipeline | `mem` | insensitive |
| epilogue | `cshuffle` | `cshuffle` |
| split-K degree | 32 | 32 |
| `lds_k_outer` | off | **on** |

The split-K degree has a narrow optimum in both cases: halving or doubling it
costs materially, so a mis-tuned auto-selection is expensive. `select_split_k_wgrad`
assumes a fixed waves-per-CU occupancy that the layout change invalidates and
should be revisited.

## Keep / revert decisions

| Change | Decision | Basis |
| --- | --- | --- |
| Fragment k-stride carries the operand length | **Keep** | Fixes NaN on the `16x16x16` atom; IR unchanged for other atoms; regression test proven to fail without it |
| Parity config 13 + numeric regression test | **Keep** | The gap that let Finding 2 ship |
| `WgradConvSpec.default_lds_k_outer` selection policy | **Keep** | Never a regression across five shapes; spec default left off so goldens stay layout-stable |
| Epilogue instruction reduction | **Revert** | No throughput change at any split-K degree; cost is atomic latency, not issue |
| Flipping the `lds_k_outer` *spec* default | **Defer** | Repo-wide: re-blesses every wgrad golden and removes M-outer coverage from the gate. Needs a deliberate decision, not a perf argument |
| Cross-lane transpose for M-outer | **Defer** | Target hardware unavailable; count argument is unfavourable |
