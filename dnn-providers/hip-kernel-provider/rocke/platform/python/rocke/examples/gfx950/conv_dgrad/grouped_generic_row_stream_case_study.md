<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Grouped dgrad, generic kernel: row-stream knobs (gfx950)

Measured numbers are deliberately absent — see `platform/AGENTS.md` §Compliance.

**Kernel.** `build_direct_conv` (`DirectConvSpec`) in
[`library/kernels/common/conv_direct_grouped.py`](../../../../../../library/kernels/common/conv_direct_grouped.py),
the generic row stream the cached variant `direct_grouped_dgrad_fused` runs
over the transposed problem for every admitted grouped dgrad shape that is not
the 4c row (see
[`grouped_direct_dgrad_dispatch_case_study.md`](grouped_direct_dgrad_dispatch_case_study.md)).
C++ mirror: `rocke_build_direct_conv`
(`platform/cpp/instances/common/conv_direct_grouped_build_generic.cpp`).

**Knobs.** `prefetch_rows`, `lds_only_sync`, `waves_m`, `lds_pad`,
`stage_out`, `xcd_tiles`, and the AOT-only `row_guard` / `row_exit`. Their
semantics, validator rules and name tags are in
[Generic Kernel dgrad: fused weights](../../../../../dsl_docs/instances/convolution.md#generic-kernel-dgrad-fused-weights-directconvspec-aot-variant-direct_grouped_dgrad_fused);
this study records why they exist and how dispatch picks them.

## What limited the kernel

The kernel streams one input row at a time through a double-buffered LDS row,
computes every output row that uses it and flushes each finished one.
Hardware counters against a hand-written reference showed it was neither
MFMA-bound nor simply latency-bound (a deeper register prefetch alone did
nothing). Four memory-path effects remained:

1. **Output stores in the MFMA C layout**: 8 bytes per lane, one store
   instruction covering 16 pixels a channel row apart, so twice the cache
   write requests of 16-byte lanes over the workgroup's channel span.
2. **LDS bank conflicts on fragment reads**: a staged column that is an even
   number of 16-byte units puts the 16 q-lanes of a read on few banks (the
   same effect made a wider `block_groups` slower).
3. **Row barrier waits for memory**: the per-row `s_barrier` came with a
   `vmcnt(0)`, so each row waited for its prefetched loads and the previous
   row's stores.
4. **Each pixel line read by two XCDs**: a workgroup covers half a 128-byte
   pixel line and neighbouring group tiles launch round-robin across XCDs.

With two 16-wide output tiles per group, one wave also held both tiles'
weights and accumulators, so the kernel ran at a low wave count.

## Levers (each verified before it was timed)

| Knob | Mechanism | Decision |
| --- | --- | --- |
| `stage_out` | Finished row through a double-buffered LDS tile, 16-byte stores; its barrier is the row barrier | kept where dX channels per group are a multiple of 16 |
| `lds_pad` | Makes the staged column an odd number of 16-byte units | kept (`DGRAD_FUSED_LDS_PAD`), only where the column is an even number of units |
| `lds_only_sync` + `prefetch_rows` | Barrier drains LDS only; rows loaded two ahead stay in flight across it | kept as a pair, two rows (a third gave no robust gain and costs registers) |
| `xcd_tiles` | All workgroups of an image row band on one XCD, so both halves of a pixel line share an L2 | kept; needs `XCD_TILES_MIN_ROW_BANDS` row bands |
| `waves_m` | One wave per 16-wide output tile half, sharing the staged row | knob kept, not dispatched |
| 16-byte row loader | 8-channel row copies | dropped: net loss once pad and staged stores were in |
| wider `block_groups` | Reference-like channel span | not adopted: ties the XCD remap with the pad, bank cliff without |

Only the combination pays: the XCD remap alone did nothing while stores and
banks were the limit, and the barrier change alone did little. Two accounting
fixes came with it: `direct_conv_lds_bytes` follows the LDS pool packer (the
staged weight slice shares a slot with the first row buffer; pad and output
tiles counted; `test_lds_bytes_match_the_lowered_pool`), and
`preload_weight_vgprs` counts per wave under `waves_m`
(`test_waves_m_splits_the_weight_register_budget`).

## What the AOT port changed

* **`xcd_tiles` from kernargs.** The grid and the chunk (the workgroups of one
  row band) are computed from `p_Wo`, `p_groups`, `p_N` and `p_Hi`, and the
  remap is the new helper `chiplet_transform_chunked_runtime` (Python and C++,
  with a pure-Python reference that is a bijection for every grid and chunk).
  Only the XCD count is baked, so the knob is built for gfx950 only; the
  row-band minimum is a `validate()` rule that `plan_for` re-checks on the
  real problem.
* **`stage_out`** takes its global offsets from the output stride kernargs.
* **`row_exit` is new.** A whole-image pick runs on a taller cached tile
  (`dgrad_fused_block_h`); `row_guard` skips the dead rows' MFMAs, and
  `row_exit` leaves the workgroup once the next row to flush is past `p_Hi`,
  skipping their loads, staging, barriers and flushes. The cache sets it on
  `DGRAD_FUSED_EXIT_BLOCK_H`, the tiles only whole-image picks reach.
* **The stack is part of the cached policy image** (`DGRAD_FUSED_STREAM`,
  filtered by `_dgrad_fused_policy_reason`): on gfx950, for square groups on
  16-column strips where `waves_m = 2` is not valid, `prefetch_rows =
  DGRAD_FUSED_PREFETCH_ROWS`, `lds_only_sync`, `xcd_tiles` and each
  (`lds_pad`, `stage_out`) pair the policy chooses among, cached beside the
  spec without the stack. gfx942 gets no stream entries.
* **No runtime switches.** The PR's module flags for disabling the stack or
  opting into the `waves_m = 2` stack are gone: dispatch picks only what is
  cached, and the split-M stack is not. `waves_m` stays available to explicit
  specs.

## Dispatch rule

`_fused_stream_knobs` in
[`library/dispatch/grouped_conv_dgrad_direct.py`](../../../../../../library/dispatch/grouped_conv_dgrad_direct.py)
runs last in `select_fused`. A fused pick takes the stack only with square
groups (`cpg == kpg`) on 16-column strips; inside the image box (the policy's
whole-image picks up to `_STREAM_UNTILED_MAX_WO` output columns; H-tiled picks
for filters above 1x1, and 1x1 from `_STREAM_TILED_1X1_MIN_CPG` channels per
group); where `waves_m = 2` is not valid; and where the stack, `xcd_tiles`
included, is valid on the problem. It then tries (pad, staged output) and
(pad, unstaged) and keeps the first that adds no launch rounds over the stack
without them (`_lds_rounds`), else the bare stack. The guard exists because
pad and staging add LDS: on grids sized to about one launch round, a few
workgroups spill into a nearly empty second round. Every other pick keeps
the spec without the stack.

Fitting: per-arm kernel screens layer by layer with LDS and register counts
recorded for every build; the rule timed at dispatch level against the
knob-off pick with interleaved, order-alternated arms; then a disjoint random
hold-out. Review rounds narrowed the region with structural conditions rather
than thresholds: `waves_m = 2` picks fell behind on small batches with many
groups (few row bands leave the XCD remap off), so they keep the previous
spec; non-square groups and picks without the XCD order lost where every
square-group pick with it was ahead; 32-column strips lost on several channel
counts; and untiled wide images of non-power-of-two groups formed a
reproducible pocket that neither the prefetch pair nor the XCD order alone
recovered, hence the image box. Inside the admitted region no shape that
changed pick was reproducibly slower than before, apart from one near-parity
case that flips with cache state.

## Byte identity and correctness

The C++ engine mirrors every path, knob, validator rule and reason string
(`conv_direct_grouped_spec_and_validate.cpp`); `lower_conv_direct_grouped(spec,
kind="generic")` forwards every field. Parity: `conv_direct_dgrad_ext`
configs 30..44; every `direct_grouped_dgrad_fused` entry of the AOT
enumeration lowers to the same `.ll` in both engines. Golden entries
`conv_direct_dgrad/gfx950/mfma_dgrad_fwl_c16_stream_bf16` and
`conv_direct_dgrad/gfx950/mfma_dgrad_fwl_c32_k32_wm2_bf16` pin the emission.

Tests (dX pre-filled with NaN), under `library/tests/`:
`test_direct_mfma_dgrad_correctness.py` (validator and names, LDS footprint,
register split, cached binaries on adversarial shapes),
`test_conv_dgrad_fused_weights.py` (`TestRowStreamKnobNumerics`: each knob
alone and stacked, one binary on several shapes),
`test_direct_kernel_cache.py` (cached scope of the stack and `row_exit`;
`test_dgrad_fused_policy_image_is_enumerated`),
`test_conv_direct_grouped_backend_parity.py`, and
`dispatch/test_grouped_conv_wgrad_dispatch.py`
(`test_routes_with_cached_knobs`).

## Replay

From `library/`, with `PYTHONPATH=$(pwd)/../platform/python:$(pwd)`; build the
caches as in the dispatch study. The first command times every cached fused
form of the shape, with and without the stack, verified:

```bash
python3 -m benchmarks.common.benchmark_direct_conv --direction dgrad \
    --run-from-cache <direct cache> --variants direct_grouped_dgrad_fused \
    --N 4 --Hi 12 --Wi 12 --C 96 --K 96 --Y 3 --X 3 --pH 1 --pW 1 \
    --groups 12 --dtype bf16 --verify
python3 -m benchmarks.common.grouped_conv.run_direct_dgrad_dispatch \
    --N 4 --H 12 --W 12 --C 96 --K 96 --G 12 --Y 3 --dtype bf16 \
    --cache-dir <igemm cache> --direct-cache-dir <direct cache> --verify
cd ../platform && python3 tools/check_byte_identity.py --only conv_direct
```

## Not done here

* Async (`buffer_load ... lds`) row loads into a swizzled LDS row instead of
  register staging and a pad.
* `stage_out` on LDS-bound wide-channel picks without an extra launch round
  (weights staged in two halves, or the dead weight slot reused).
* A separately validated region, and cache entries, for the `waves_m = 2`
  stack.
