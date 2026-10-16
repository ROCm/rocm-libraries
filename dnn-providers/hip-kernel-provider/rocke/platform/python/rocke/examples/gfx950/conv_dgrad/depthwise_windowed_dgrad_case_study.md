<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Windowed depthwise dgrad — gfx950 case study

Measured numbers are deliberately absent — see `platform/AGENTS.md` §Compliance.
What is recorded here is mechanism, instruction-count evidence, the levers that
were tried (kept and reverted), what the AOT port changed, and the replay path.
The knob reference is in
[`dsl_docs/instances/convolution.md`](../../../../../dsl_docs/instances/convolution.md),
section "Windowed Depthwise Dgrad Kernel".

## Kernel and where it lives

`DirectDepthwiseDgradWindowedSpec` / `build_direct_depthwise_dgrad_windowed`
(VALU and `dot2` forms) and `_build_dw_dgrad_win_mfma` (Toeplitz MFMA form) in
[`conv_direct_grouped.py`](../../../../../../library/kernels/common/conv_direct_grouped.py);
C++ mirror [`conv_direct_grouped_build_depthwise_dgrad_win.cpp`](../../../../../cpp/instances/common/conv_direct_grouped_build_depthwise_dgrad_win.cpp).
AOT cache variant `direct_depthwise_dgrad_win` (`DW_WIN_*`,
`_dw_win_policy_reason` in
[`direct_kernel_sweep.py`](../../../../../../library/benchmarks/common/direct_kernel_sweep.py));
dispatch candidates `direct_depthwise_dgrad_win` / `direct_depthwise_dgrad_stream`
with the policy in
[`grouped_conv_dgrad_direct.py`](../../../../../../library/dispatch/grouped_conv_dgrad_direct.py).
Scope: `cpg = kpg = 1`, stride 1, fp16 / bf16 (implicit GEMM declines
`cpg = 1`). The row-streaming kernel it replaces showed, at 7×7, hundreds of
`buffer_load_ushort` (one guarded load per tap) and over a thousand
`v_cndmask_b32` (address *and* value selected per tap) per lane.

## Levers (VALU form)

**1 — one window load per dY row (KEEP).** Block columns `[wi0, wi0 + block_w)`
need dY columns `wi0 + PAD - (KW - 1) + t`, `t < block_w + KW - 1`; the window
is loaded once and tap `(s, j)` reads `window[j + KW - 1 - s]`. Out-of-range
columns read zero through the sentinel, so the FMA chain has no select. Loads
per row drop from `KW * block_w` to `block_w + KW - 1`, and compile time falls
sharply with the per-tap address arithmetic gone.

**2 — lane/row split addressing, two sentinels (KEEP, second form).** Every
offset is `lane + row`: the row-invariant `(channel, column)` part or
`DW_DGRAD_WIN_OOB_LANE`, plus the block-uniform row part or
`DW_DGRAD_WIN_OOB_UNIFORM`. The IR `add` lowers to `add nsw`, so one sentinel
plus an offset would be signed overflow; the two sum to the largest positive
i32, keeping every add in range and every invalid access past the buffer —
hence the validator cap `DW_DGRAD_WIN_MAX_TENSOR_BYTES` on dY / dX.
*First form REVERTED:* column validity in the uniform part made every offset
scalar; LLVM hoisted the `s_cselect`s and spilled SGPRs to VGPR lanes
(`v_writelane` / `v_readlane` storm). One `v_cndmask` per window column in the
lane VGPR removed the spills and nearly all selects.

**3 — next-row prefetch + `sched_barrier(0)` (KEEP, hardwired).** The
scheduler issued each window load just before its first use. Issuing the next
live row's window before the current row's FMAs, fenced by `sched_barrier(0)`,
moves the waits to row boundaries and lowers VGPR use. Two-row lookahead and
the unfenced variant were dropped.

**4 — `dot2` tap pairing (KEEP as knob).** At 7×7 the kernel is VALU-issue
bound. `v_dot2c_f32_{bf16,f16}` pairs taps `(s, s+1)`, halving the
multiply-add issue count and removing the 16→32-bit widening (windows stay
packed as `(x[t], x[t-1])`, weights packed once with a zero partner for an odd
KW). This added `arith.fdot2` to both engines (`IRBuilder.fdot2`,
`_op_arith_fdot2`; `rocke_b_fdot2`, `ROCKE_OP_ARITH_FDOT2` appended so no
opcode value moves). The bf16 intrinsic is not selectable on gfx942, so `dot2`
is gfx950-only (`DW_DGRAD_WIN_DOT2_ARCHES`). Neutral on memory-bound 3×3.

**Odd-KW tail pair (correctness fix).** The first form fed the last pair
`(w[KW-1], 0)` the operand `(x[j], x[j-1])`, so the zero weight met a column
outside the receptive field and an Inf there gave `0 * Inf = NaN` in a finite
dX column. The tail now reads `(x[j], 0)`. Of three encodings,
`vec_pack(x[j], 0.0)` (an extra `v_perm_b32` per column, more VGPRs) and a
zero-extend of the raw 16 bits placed after the full pairs (raw columns live
across them, VGPRs past 256) were reverted; the zero-extend with the tail pair
issued *first* keeps VGPRs under 256 with fewer instructions than the first
form, and was kept.

**Other levers.** `block_w` dividing W (KEEP; non-dividing widths always
worse). `ch_per_lane = 2` with `v_pk_fma_f32` (KEEP for 3×3 with enough
channels; loses on 7×7 to VGPR pressure); `ch_per_lane = 4` never won. Row
tiles help small grids and re-read a `KH - 1` row halo. Open: with the tail
fixed, `dot2` on 3×3 fp16 was slightly ahead of channel pairs on one cohort;
the cache and dispatch still pair taps only for `KW >= 5` until the 3×3 cohort
is re-swept.

## Lever 5 — Toeplitz MFMA form (KEEP)

Knobs `mfma`, `w_fold`, `prefetch_rows`; gfx950 (`DW_DGRAD_MFMA_ARCHES`),
`groups % DW_DGRAD_MFMA_CH == 0`. `v_dot2c` and `v_pk_fma_f32` issue slower
than `v_fmac_f32`, so they save instructions but not multiply-add throughput:
the arithmetic had to move to the matrix core. Each wave owns 8 channels and a
16×16 fp32 tile per dX row in flight (row `m = 8q + k`: channel `k`, column
parity `q`; column `n`: a column pair of one of `w_fold` images). A is a
one-hot diagonal weight fragment, B 8 channels × 4 window columns, so one
`v_mfma_f32_16x16x32_{f16,bf16}` covers 4 taps of both parities and a filter
row takes `ceil((KW + 1) / 4)` passes (`_dw_mfma_passes`). Seven of eight
products are discarded; the kernel becomes a memory-access problem.

Memory path, each step verified and compared with the previous one: per-lane
global B loads with 8-byte D stores (REVERT: neighbouring lanes and passes
re-load the same columns), `ds_bpermute`-shared B (superseded), half-exchange
16-byte writes (neutral), an XCD-sharing workgroup remap (helps only small
workgroups) and deeper prefetch / H split on the per-lane form (more re-reads)
were all reverted. KEPT and hardwired: dX rows staged in an LDS double buffer
and drained as 16-byte channel runs (a store-free diagnostic build showed the
dX writes were the largest cost), and dY window rows loaded block-coalesced
into a second LDS double buffer in the same barrier phase, B read with
`ds_read_b128`. `prefetch_rows = 2` was ahead of 1 and 3. None of the reverted
steps is a spec field; `block_w` / `ch_per_lane` do not apply to the MFMA form
and `w_fold` / `prefetch_rows` are rejected on the VALU form.

Validator guards (same text in both engines) bound `unrolled_mfmas()`,
`block_waves` (two waves per SIMD keep 256 VGPRs each), `mfma_frag_vgprs()`
(calibrated on compiled VGPR counts, no spills), `mfma_lds_bytes()` (equal to
the compiled group segment) and `prefetch_rows` by their `DW_DGRAD_MFMA_*`
constants. ISA (7×7, 8 waves): one MFMA per tap row and pass, no multiply-add
VALU work, one barrier per dY row, no spills, about half the `dot2` VGPRs.

### Non-finite semantics (accepted)

Finite inputs give the VALU results up to fp32 summation order. A non-finite
dY value does not stay in its channel: the zero off-diagonal weights give
`0 * Inf = NaN` in the other 7 channels of its group, and the zero-weight taps
of the passes widen the receptive field to `4 * ceil((KW + 1) / 4)` columns —
one extra column for KW = 3, 7, 11, two for KW = 6, 10, three for KW = 5, 9,
four for KW = 4, 8, 12 (the side depends on the output column parity). Rows
stay exact and nothing crosses images. This is inherent to the one-hot
formulation; gradient overflow checks treat NaN and Inf alike.
`TestNonFiniteGradients` (`tests/test_conv_dgrad_depthwise_windowed.py`) keeps
the strict per-channel assertion for the VALU / `dot2` forms (it fails on the
first tail form) and, for the MFMA form, requires every out-of-tolerance
element to lie in the 8-channel group of a non-finite input inside the widened
field, with no non-finite reference value coming out finite; single-Inf tests
bound each spread to its rows, its group and `4 * ceil((KW + 1) / 4)`
contiguous columns.

## What the AOT port changed

- **ABI:** the direct dgrad ABI (`emit_direct_params(direction="dgrad")`); N,
  H, W, Ho, Wo, groups and the dY / dX element strides are `p_*` kernargs and
  byte offsets come from the strides.
- **Capabilities:** `KH`, `KW`, `PAD`; cache rows are the square
  `_DEPTHWISE_FILTERS` with every padding. **Knobs:** `block_w`, `block_h`,
  `block_waves`, `ch_per_lane`, `dot2`, `mfma`, `w_fold`, `prefetch_rows`. The
  kernel name spells filter, padding, dtype and every knob, no extent.
- **`block_h` is a mandatory row-tile height.** Block z is decoded against the
  runtime tile count `ceil(p_Hi / block_h)` (`_dw_win_tile`) and the tile's row
  loop is unrolled. dY rows outside `[0, p_Ho)` read zero through the uniform
  sentinel; dX rows past `p_Hi` drop their stores. The `KH - 1` halo rows that
  fall outside the image run their taps (VALU) or staging and MFMAs (MFMA form)
  under a block-uniform `scf.if` plus a one-trip loop carrying the
  accumulators; every MFMA phase still syncs, so the LDS double buffers keep
  alternating. A tile covering the whole image multiplies the zero padding
  rows: the pre-port build-time pruning of padding rows went with the baked height.
- **Plan-time rules:** `groups % DW_DGRAD_MFMA_CH`, `groups % ch_per_lane` and
  the sentinel range; the IR does not depend on them.
- **Cached image** (`_knob_grid` filtered by `_dw_win_policy_reason`):
  `DW_WIN_TILES` (whole-map tiles `block_w == block_h` and column tiles over row
  chunks) with `DW_WIN_WAVES`, channel pairs (`DW_WIN_PAIR_WAVES`) for 3×3
  only, `dot2` exactly where it pairs taps, at most `DW_WIN_UNROLL_BUDGET`
  unrolled FMAs; the MFMA form for "same"-padded `DW_WIN_MFMA_FILTERS` over
  `DW_WIN_MFMA_FOLDS` × `DW_WIN_MFMA_WAVES` × `DW_WIN_MFMA_BLOCK_H` with
  `DW_WIN_MFMA_PREFETCH`. The PR's unroll shrink for very large filters has no
  counterpart: those filters have no cache row.

## How dispatch picks it

`direct_depthwise_dgrad_win` (priority 5) ranks ahead of
`direct_depthwise_dgrad_stream` (priority 6), which serves stride 2 and every
stride-1 request the windowed kernel declines (a tensor past the sentinel
range, no valid cached tile). `select_depthwise_win` checks `depthwise_errors`
and stride 1, then takes `_dw_mfma_knobs` inside the admission box, else
`_dw_valu_knobs`. `launch_errors` drops picks past the i32 byte sizes or the
y/z grid limit.

**MFMA admission box** (`_dw_mfma_admits`): square `_DW_MFMA_FILTER` filters
with "same" padding; C a multiple of `_DW_MFMA_CH_MULTIPLE` up to
`_DW_MFMA_MAX_C`; N up to `_DW_MFMA_MAX_N`; H in `_DW_MFMA_H_RANGE`; W in
`_DW_MFMA_W_BANDS`; dY up to `_DW_MFMA_MAX_DY_BYTES`. It is the region where the
MFMA form was ahead of the `dot2` kernel with warm *and* cold caches. An earlier
rule (every filter from 5×5 up, plus exceptions added after each review found
cold losses in a new corner) was inverted after a dense 5×5 / 7×7 probe: 5×5
is near memory-bound in the `dot2` kernel and lost cold on large tensors; taller
images re-read a `KH - 1` row halo per chunk; images shorter than the filter
and widths that fill a tile only partly lost or broke even. A fresh hold-out
inside the box stayed ahead; just outside each edge the previous pick stays.

**MFMA knobs** (`_dw_mfma_knobs`): `w_fold` from `_dw_mfma_fold` (by width up
to 32 columns, beyond that the fold with the least modelled padded-tile plus
window work); `_DW_MFMA_WAVES` waves, halved where `_dw_mfma_blocks_per_cu`
fits one block per CU and the grid ends in a partial round; images up to
`_DW_MFMA_WHOLE_H` rows whole, taller ones in `_DW_MFMA_ROW_STEPS` chunks; below
`_DW_MFMA_TARGET_BLOCKS` workgroups first 4 waves, then smaller chunks, the
`_DW_MFMA_SHORT_ROWS` step only below `_DW_MFMA_SHORT_ROWS_BLOCKS` (a first
form that stepped down on mid-sized grids lost there); every row count on the
smallest covering `DW_WIN_MFMA_BLOCK_H`.

**VALU knobs** (`_dw_valu_knobs`): `dot2` for `KW >= 5` on a dot2 arch;
channel pairs for 3×3 with enough channels; 4-wave blocks where C fills them;
the cached tile minimising padded columns × dY rows per row band, scaled up
when the grid holds fewer than `_DW_TARGET_WAVES` waves. Known losses against
the per-shape optimum sit on mid-sized grids where 8 waves with smaller chunks
and 4 waves with larger ones are close.

## Correctness and gates

- `library/tests/test_conv_dgrad_depthwise_windowed.py`: validation and
  lowering on CPU; GPU numerics of both forms on adversarial shapes (odd sizes,
  N = 1, partial channel blocks, tiles taller than the image, odd N under
  `w_fold`), one binary on several shapes, `TestNonFiniteGradients`. Bad when
  `|out - ref| > tol + tol * |ref|`, `tol = 1e-2`, dX NaN-filled first.
- `library/tests/dispatch/test_grouped_conv_dgrad_depthwise_dispatch.py` (box
  edges, knob rules, windowed vs streaming routing);
  `library/tests/dispatch/test_grouped_conv_dgrad_dispatch_aot.py` (every
  selection a cached, valid identity); `library/tests/test_direct_kernel_cache.py`
  (`test_windowed_depthwise_dgrad_*`); `platform/tests/core/test_fdot2_lowering.py`.
- Byte identity: `conv_direct_dgrad_ext` indices 0..19 (header of
  `library/tests/parity/conv_direct_dgrad_ext_emit.py`), `target_intrinsics`
  for `fdot2`. Golden: `conv_direct_dgrad/gfx950/dw_dgrad_win_dot2_k7`,
  `dw_dgrad_win_mfma_k7`.

## Replay

From `library/`, `PYTHONPATH=$(pwd)/../platform/python:$(pwd)`:

```bash
python3 -m benchmarks.common.benchmark_direct_conv --compile-all \
    --direction dgrad --dtype bf16 --cache-dir <direct cache> --jobs 0
# exactly what dispatch ships: a dot2 pick, then an MFMA pick (box)
python3 -m benchmarks.common.grouped_conv.run_direct_dgrad_dispatch \
    --N 2 --C 96 --K 96 --H 12 --W 12 --G 96 --Y 7 --dtype bf16 \
    --direct-cache-dir <direct cache> --verify
python3 -m benchmarks.common.grouped_conv.run_direct_dgrad_dispatch \
    --N 2 --C 192 --K 192 --H 10 --W 20 --G 192 --Y 7 --dtype bf16 \
    --direct-cache-dir <direct cache> --verify   # --time in a separate run
# every cached depthwise dgrad binary that fits the shape, each verified
python3 -m benchmarks.common.benchmark_direct_conv --direction dgrad \
    --N 2 --Hi 10 --Wi 20 --C 192 --K 192 --groups 192 --Y 7 --X 7 \
    --pH 3 --pW 3 --dtype bf16 --verify --run-from-cache <direct cache> \
    --variants direct_depthwise_dgrad_win,direct_depthwise_dgrad
python3 -m pytest -q tests/test_conv_dgrad_depthwise_windowed.py \
    tests/dispatch/test_grouped_conv_dgrad_depthwise_dispatch.py
cd ../platform && export ROCKE=$(pwd) PYTHONPATH=$(pwd)/python:$(pwd)/../library
python3 tools/check_byte_identity.py --only conv_direct_dgrad_ext,target_intrinsics
python3 tests/instances/rocke_ir_parity_harness.py \
    --check tests/golden/rocke_representative_ir_sha256.json --flavor llvm22
```

ISA: `llvm-objdump -d --mcpu=gfx950`; count `v_cndmask_b32`, `buffer_load_*`,
`v_fmac_f32` / `v_dot2c_*` / `v_mfma_*`; `llvm-readelf --notes` for spills.
