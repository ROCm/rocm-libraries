<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Stride-1 implicit-GEMM dgrad on gfx950 — case study

Measured numbers are deliberately absent — see `platform/AGENTS.md` §Compliance.

Companion to [`dgrad_lds_layout_case_study.md`](dgrad_lds_layout_case_study.md),
which covers the K-outer B tile. This study covers the dense stride-1 path of
the implicit-GEMM dgrad kernel: what dominated its main loop, the levers tried,
which were kept or reverted and why, and how the result was carried into the
ahead-of-time (AOT) kernel cache. The binaries are not dispatched yet (the
grouped-convolution dispatcher is being reworked separately); they are
reached through the cache (`--run-from-cache`). Recorded here:
mechanism, instruction-level evidence, keep/revert decisions, correctness
evidence and replay commands.

The knob reference (fields, validator rules, name tags) lives in
[`README_conv_implicit_gemm_dgrad.md`](../../../../../../library/kernels/common/README_conv_implicit_gemm_dgrad.md):
[Stride-1 record from kernargs](../../../../../../library/kernels/common/README_conv_implicit_gemm_dgrad.md#stride-1-record-from-kernargs),
[Tap-outer K loop](../../../../../../library/kernels/common/README_conv_implicit_gemm_dgrad.md#tap-outer-k-loop-tap_outer_k),
[dY halo reuse](../../../../../../library/kernels/common/README_conv_implicit_gemm_dgrad.md#dy-halo-reuse-dy_halo)
and
[XCD-contiguous tile order](../../../../../../library/kernels/common/README_conv_implicit_gemm_dgrad.md#xcd-contiguous-tile-order-xcd_tile_order).
This document does not repeat it.

## Where it lives

| Piece | File |
|---|---|
| Builder, spec, validator | `library/kernels/common/conv_implicit_gemm_dgrad.py` (`DgradConvSpec`, `_build_tilde_dgrad`, `is_valid_dgrad_spec`) |
| C++ mirror (byte-identical IR) | `platform/cpp/instances/common/conv_implicit_gemm_dgrad.cpp`, header `platform/cpp/include/rocke/instance_conv_implicit_gemm_dgrad.h` |
| AOT grid | `library/kernels/conv/aot_jobs.py` (`_dgrad_caps`, `_dgrad_tap_outer_variants`, `_dgrad_halo_jobs_for`, `CACHE_DGRAD_*`) |
| Cache identity / predicates | `library/kernels/conv/kernel_identity.py` (`KernelIdentity`, `identity_supports_problem`; `KernelCache.supports_problem` in `library/benchmarks/common/kernel_cache.py` wraps it) |

The kernel is shape-generic. Every problem extent (N, H, W, C, K, groups,
pads, strides, filter size) is a kernarg of `conv_args_signature(dtype,
"dgrad")`, which also carries `sub_gemm_buf` / `num_sub_gemms` (the tilde
record buffer and its magic-division table). What a cached binary bakes is
its capability (tilde vs stride 1 / dilation 1, and for the dY halo binaries
the filter and `halo_max_wo`) and its config knobs.

## Starting point

Every dgrad problem, stride 1 included, ran through the tilde-decomposition
kernel. Each CTA binary-searched the record buffer for its sub-GEMM and loaded
the record fields with scalar global loads. The address closures then decoded
every loaded vector inside the K loop, `k_dg -> (y, x, k_out)` and
`m -> (n, hi, wi)`, against record and kernarg values the compiler cannot see,
so the decode was a chain of runtime divides and runtime-stride products on the
path to every `buffer_load`.

The K loop is one `scf.for` running load, `sync`, MFMA, `sync`. On a dense 3x3
cohort the main-loop ISA had roughly an order of magnitude more VALU
instructions than MFMAs, and the address math sat between the global loads and
the MFMAs.

`uniform_k_decode` (an earlier change, default-on for gfx950 wave64 via
`DgradConvSpec.default_uniform_k_decode`) already gives the tilde binary a
per-tile uniform decode when `kpg % tile_k == 0`, with the plain loop kept as a
run-time fallback. The levers below work on top of, or in place of, that.

## Lever 1: the stride-1 record from kernargs (kept)

At stride 1 the tilde decomposition is the identity in both spatial axes
(`y_tilde = x_tilde = 1`, whatever the dilation, pad, filter or grouping), so
there is exactly one sub-GEMM and every field of its record is a kernarg or a
constant. `DgradConvSpec.folds_sub_gemm_record` (stride 1) makes the builder
bind those values instead of emitting the binary search and the dependent record
loads. The unit strides and zero offsets fold into the address arithmetic, and
the `block_start` subtraction disappears.

What changed from the PR's version of this lever:

- **The PR folded the record into immediates** (`arith.constant` per field,
  so divides by record fields became divides by constants). The AOT kernel has
  no problem constants to fold: the record is rebuilt from kernargs. Divides
  that needed the record still go through the magic-division pairs, which the
  kernel reads from the one table entry after the one record at constant
  offsets (eight loads that depend on nothing). What the fold removes is the
  search, the record loads and the dependency chain behind them.
- **No `static_sub_gemm` knob.** The fold is the stride-1 code path. It keys on
  the stride alone, which an AOT binary carries as its stride capability (the
  probe problem and every problem the binary serves agree on it). There is
  nothing to switch off: the folded and the loaded record are equal by
  construction (`test_dgrad_stride1_record_is_a_function_of_the_kernargs` in
  `tests/test_conv_abi.py`).
- **The PR's ungrouped-pointwise exclusion is gone.** In the PR, folding made
  the pointwise K-loop trip count a constant and LLVM unrolled the loop with no
  overlap to show for it. With runtime extents there is no constant trip count,
  so folding the pointwise record is a straight substitution. The AOT grid also
  builds no separate pointwise kernel: 1x1 problems run the general kernels.

C++: `_dgrad_folds_sub_gemm_record` / `fold_record` in `_build_tilde_dgrad`,
field for field and op for op.

### Flat folded loop follow-ups: not ported

The PR carried five review rounds on the folded record with the flat K loop
(a K tile straddling taps): a derived `waves_per_eu` floor on two tiles of
the PR's tile table, an accumulator-count cap on the fold, explicit batching of the flat
loop's global reads, and their interplay with narrow loads. None of that is in
the port, because the AOT grid builds no flat-loop stride-1 binary (see
"What the AOT port changed"); those problems run the tilde binary of the same
configuration with `uniform_k_decode`, whose uniform loop already issues all of
a tile's reads before its LDS writes. Two findings from those rounds remain
useful as mechanism:

- **AGPR round trip.** On gfx950 the backend picks the AGPR form of the MFMA
  whenever the per-wave VGPR budget may exceed 256, which is the case at the
  default occupancy floor of one wave. In a flat loop with a constant record the
  accumulators were then copied AGPR -> VGPR -> AGPR every iteration
  (`v_accvgpr_read` + `v_accvgpr_write` per accumulator in the loop body). A
  `waves_per_eu` floor of 2 caps the budget at 256, the backend selects the
  VGPR-form MFMA and the copy disappears. `"amdgpu-agpr-alloc"="0"` alone did
  not change the binary.
- **Serialized staging.** A loop whose loads no longer interleave with address
  math lets the scheduler serialize every staged `buffer_load` with its
  `ds_write` through one register quad (`buffer_load` -> `s_waitcnt vmcnt(0)` ->
  `ds_write`, per vector). Too high an occupancy ceiling in `waves_per_eu`
  produced the same pattern. Narrow loads mean more vectors per tile, so the
  effect is worst there. The fix in every loop that kept it is the same:
  `load_global` for A and B, then `store_lds` for both.

## Lever 2: tap-outer K loop (`tap_outer_k`, kept)

The flat loop index is `k_dg = (y*X + x)*kpg + k_out`. When `kpg % tile_k == 0`
no K tile straddles two filter taps, so the loop splits in two: tap
`t = y*X + x` outer, output-channel chunk `kb` inner. The reduction order is
unchanged (tap-major, then chunk), so dX is bit-identical to the flat loop's.

Each loop-invariance class becomes its own operand subtree:

| Term | Depends on | Lands in |
|---|---|---|
| `(n, hi, wi)` decode (magic pairs of the folded record), the per-pixel dY / W bases | thread only | kernel preamble |
| `ydot = tap / p_X` (one scalar divide), `dh = p_pH - ydot`, `dw = p_pW - xdot`, the dY shift `(dh*p_Wo + dw)*p_K`, the tap's dY bounds, the W term `tap*Cs` | tap only | outer loop body |
| `+ kb` on dY, `kb*Y*X*Cs` on W | chunk | inner loop, one uniform add per operand |

(`Cs` is `p_cpg` grouped, `p_C` otherwise; the chunk loop runs to `p_kpg`, or
`p_K` ungrouped.) The dY descriptor returns the raw offset and its predicate
without a select of its own: the loader already maps an invalid lane to the
out-of-bounds sentinel, which the buffer load returns as zero, so the old
double select collapses to one. The W descriptor returns no predicate, since
`k_abs < kpg` and `t < Y*X` hold by construction.

ISA evidence (32x32x16 atom, bf16): inner-loop VALU went from tens of
instructions per K tile to a handful, one `v_add` plus one `v_cndmask` per A
vector and one `v_add` per B vector.

**The first cut was slower.** With the address math gone, the per-vector
`load -> store` pairs of `CoalescedTileLoader.load` left the scheduler nothing
to overlap, and the ISA showed the serialized staging pattern above for every
vector:

```text
buffer_load_dwordx4 v[50:53], ...
s_waitcnt vmcnt(0)
ds_write_b128 ..., v[50:53]
```

The fix, part of the lever: the load phase issues `load_global` for A and B,
then `store_lds` for both. The ISA then shows the tile's `buffer_load`s back to
back, followed by `vmcnt(N-1) .. vmcnt(0)` before the `ds_write`s.

**Relation to `uniform_k_decode`.** Both need `kpg % tile_k == 0` to win.
`uniform_k_decode` keeps a per-tile compare-and-wrap and per-tile bounds limits
and carries a fallback loop; tap-outer drops the wrap and the per-tile limits,
keeps the decode and the bounds out of the inner loop, and has no fallback. The
validator therefore rejects `tap_outer_k` with stride or dilation other than 1,
`kpg % tile_k != 0`, `split_k > 1`, wave32 / the wavelet pipeline, and with
`uniform_k_decode=True` (whose loop it replaces). The two loops are exclusive
in one binary; both stay in the grid (`uniform_k_decode` on the tilde binary,
`tap_outer_k` on the stride-1 binary).

C++: `tap_dy_ctx_t` / `tap_w_ctx_t`, `_tap_dy_descriptor`,
`_tap_w_descriptor{,_kouter}`, the header field (appended last), the validator
(reasons verbatim) and the `_tapk` name tag.

## Lever 3: one-tile register prefetch across (tap, chunk) (reverted)

The next tile's global reads were issued right after the current tile was
written to LDS, so they overlapped the MFMAs. At a chunk boundary the next tile
is chunk 0 of the next tap; both taps' bases and predicates are outer-invariant,
so the choice reduces to a scalar-condition select, and the staged vectors ride
both loops' `iter_args`. Correct, but neutral to slower everywhere: VGPR use
rose and cost resident waves per SIMD, while the 4- and 8-wave workgroups
already co-reside several per CU and hide global latency across workgroups.

## Where the time goes after levers 1-2

A speed-of-light split with temporary builds (full kernel; no global loads;
no MFMA) showed the global-to-LDS phase as the larger half. That is the
structural im2col cost: dY is re-gathered once per tap and W is re-read by
every M tile. A direct convolution loads a halo'd input tile once and reuses it
across all taps; lever 6 brings that reuse to the implicit GEMM by swapping the
loop order.

`rocprofv3 --pmc SQ_INSTS_LDS SQ_LDS_BANK_CONFLICT` on the same split builds:
all LDS bank conflicts are in the MFMA phase, all of them from the K-outer B
transpose reads (`ds_read_b64_tr_b16`); the load phase is conflict-free.

## Lever 4: K-outer pad (reverted as a default; kept as a halo knob)

The K-outer B row pad (`_KOUTER_PAD`, 8 elements) was swept. For the 32x32x16
atom a pad of 32 makes the transpose reads conflict-free: four K rows of 64
bytes per 32-lane phase must tile the 256-byte address class, so the row stride
has to be congruent to 64 or 192 modulo 256. For the 16x16x32 atom no uniform
pad helps: eight K rows of 32 bytes collide modulo 256 for any stride, and a fix
would need a K-row permutation mirrored on the write side. On the plain
tap-outer loop the change was inside session noise, so `_KOUTER_PAD` stays 8.
Once lever 6 removes most dY traffic the conflicts matter; the wider pad comes
back there as `dy_halo_kouter_pad`.

## Lever 5: XCD-contiguous tile order (dense: reverted; grouped: kept)

**Dense.** A contiguous-chunk-per-XCD remap and the existing
`chiplet_aware_super_tile` helper were both tried on ungrouped problems. Results
were mixed and close to the noise floor; neither was kept, and `chiplet_swizzle`
stays rejected by the dgrad validator.

**Grouped.** A review found the default tile behind the previous tree on
many-group problems with a small per-group M, a partial N tile and a long
reduction, with warm and with cold caches. Variant arms isolated the trigger
without the cause: any change that made the K loop issue its loads faster lost
on that class. Hardware counters (`TCC_HIT`, `TCC_MISS`, `TCC_EA0_RDREQ`)
explained it: the faster-issuing builds made the same number of L2 requests but
hit L2 less often and sent more reads to the fabric. The time was set by L2
reuse between workgroups, not by the loop body.

Mechanism: the hardware hands workgroups to the XCDs round-robin in launch order
and each XCD has its own L2. With the N tile fastest, the two workgroups that
read the same dY rows always land on different XCDs, and a group's M tiles (all
reading the group's weight slice) are spread over every XCD, so every XCD
fetches every group's operands. Grouped problems share nothing between groups,
so nothing is gained by spreading a group out; how much refetch L2 absorbs
depended on how the workgroups that share data happened to line up in time,
which is why the loss followed the load pattern without being caused by it.

Fix: remap the launch-order linear id with one contiguous chunk per XCD and
split it back into tile and group, so each XCD runs a contiguous range of whole
groups. One chunk per XCD beat smaller fixed chunks: a co-resident set of
workgroups on one XCD then spans one contiguous range rather than several far
apart.

**Single N tile, 1x1.** With one N tile and a 1x1 filter every dY element is
read by one workgroup and the only shared operand is the small weight slice; the
order removed almost no refetch and only moved concurrently streamed dY rows
apart (more `TCC_EA0_RDREQ_DRAM_CREDIT_STALL` and L2 tag stalls). The order is
withheld there. Both conditions are needed: a 3x3 filter on one N tile shares dY
halo rows between neighbouring M tiles, and two N tiles share dY rows, which is
the reuse the order exists for.

**What the AOT port changed.** The PR decided all of this at build time from
baked extents. In the port (`xcd_tile_order_xcds`, `emit_xcd_tile_order`):

- Build time: target in `_XCD_TILE_ORDER_ARCHES` (gfx950 only), the knob on,
  a grouped kernel, a folded record (stride 1, so one sub-GEMM of
  `num_pid_m * num_pid_n` tiles) and `split_k == 1` (split-K slices ride
  `blockIdx.z`).
- Run time, one uniform select on kernargs: `groups > 1`, `Y*X > 1 or
  num_pid_n > 1`, and `num_wgs >= 8`. The grid and the chunk
  (`num_wgs / 8`) are runtime too (`chiplet_transform_chunked_runtime` in
  `rocke/helpers/grid.py`). A grouped binary also serves `groups == 1`; the
  run-time test keeps that launch in launch order.
- The remap is a bijection of the launch grid, so correctness never depends on
  the decision.
- `xcd_tile_order` is default on; `False` (name tag `_noxcd`) exists only for
  an A/B build and is never swept, so ABI, identity, label and sweep are
  unchanged.
- C++: mirrored op for op (`_dgrad_xcd_tile_order_xcds`,
  `ROCKE_DGRAD_XCD_TILE_ORDER_ARCH`). The PR's C++ builder refused grouped
  dgrad; the port's mirrors it, so the order now has parity configs.
- Scope consequence: grouped stride-1 problems whose `kpg` is not a multiple of
  the binary's `tile_k` run the tilde binary, which has no folded record and so
  keeps launch order. The PR's losing class included such problems; it was not
  re-measured on the AOT binaries.

## Lever 6: dY halo reuse (`dy_halo`, kept; cached on gfx950)

**Observation.** At stride 1 with an output the size of the input, tap `(y, x)`
reads dY pixel `m + (pH - y)*Wo + (pW - x)` for dX pixel `m` in linear
`(n, h, w)` order. All taps of a tile of `tile_m` consecutive dX pixels read one
contiguous dY range: the tile plus `(Y-1)*Wo + (X-1)` halo pixels. Counters on
the tap-outer kernel showed several times the L2 requests and misses of a direct
convolution on the same problems.

**Change.** `dy_halo` swaps the loop to output-channel chunk outer and filter
tap inner (unrolled). Per chunk the extended A tile is loaded into LDS once and
each tap reads it at a row shift of `(Y-1-y)*Wo + (X-1-x)`. A lane whose
shifted pixel leaves the image (row wrap, image edge, batch edge) reads a
trailing all-zero LDS row: one select on the LDS row index per fragment row, no
per-element masking. The epilogue is untouched; the M mapping stays linear.

Steps, one lever each, A/B in one locked session per comparison with arms
interleaved and the order alternated per round (identical binaries measured
apart at different list positions, so unbalanced orders misattribute small
levers):

- **Staged halo alone (`dy_halo=1`).** L2 requests and misses fell to about the
  direct convolution's level and MFMA utilization rose, but every tap still
  exposed one B global load, two barriers and a full wait.
- **Double-buffered B with a register prefetch (`dy_halo=2`).** The first build
  was no faster: LLVM sank the next tap's `buffer_load`s to just before their
  `ds_write`, after the MFMA block, so nothing overlapped. A `sched_barrier(0)`
  after the prefetch issue and another before the LDS store keep the loads at
  the top of the tap. The pin is implied by `dy_halo=2`, not a knob, and was the
  largest single step after the halo itself.
- **Larger M tiles, 4x1 waves.** With A reused, a larger M tile halves the B
  traffic per output pixel. LDS fragment reads then became a co-limiter: with
  2x2 waves a workgroup reads more LDS bytes per tap than the CU feeds at the
  MFMA rate; 4x1 waves (each wave covers every B column) cut that.
- **K-outer B pad 32 (`dy_halo_kouter_pad`).** Lever 4's conflicts matter once
  dY traffic is gone. The pad removes them on the 32x32x16 atom and wins where
  it keeps the workgroups per CU; where the extra LDS crosses an occupancy step
  it loses, and on 2x2-wave tiles it was slower at equal occupancy. A pad of 3
  gave wrong dX (the transpose read needs aligned rows): the validator accepts
  only multiples of 8.
- **`s_setprio` around each tap's MFMAs (`dy_halo_setprio`).** A small gain on
  the 4x1-wave tiles with several N tiles, slightly negative on the 2x2 tile.
- **Reverted / not kept:** an A row pad of 16 (helps the 16x16x32 atom, hurts
  the 32x32x16 atom the halo tiles use); prefetching the next chunk's halo
  during the last tap (more VGPRs, lost occupancy, and it issued a load past
  the last chunk); an XCD-contiguous order for dense problems (within noise).
- **`dy_halo_2d` (zero-bordered 2-D halo) is not ported.** It needs an exact
  baked `Wo`; in the PR it was neutral on average and never adopted.

**Occupancy decides the tile.** The best halo tiles run at one or two
workgroups per CU with large LDS footprints; one LDS step costs more than any
small lever gains. `vgpr_count` in the gfx950 code-object notes already includes
the AGPRs (it equals the accumulation offset plus the AGPR count); adding
`agpr_count` on top double counts and wrongly predicts one wave per SIMD for the
256x64 tile. `MeanOccupancyPerActiveCU` confirmed the LDS steps.

**What the AOT port changed.**

- The filter is baked (the taps are unrolled; the identity records
  `filter_h/w`) and so is the pad, which `Ho == Hi` at stride 1 fixes to
  `((Y-1)/2, (X-1)/2)`, so the sign of every tap's `dh` / `dw` (which bounds a
  tap needs) is known at build time.
- New capability `halo_max_wo`: the staged tile is sized for it
  (`DgradConvSpec.dy_halo_lds_rows`: `tile_m + (Y-1)*halo_max_wo + (X-1)` rows
  in whole loader passes, plus the zero row), so the LDS shape is a property of
  the spec. `Wo`, the tap shifts, the row bounds and the loader's real-row guard
  are runtime; a binary serves every `Wo <= halo_max_wo`. Rows past a narrower
  image's real extent load nothing.
- The LDS charge (`_dgrad_lds_charge`) counts the halo rows and the second B
  buffer of `dy_halo=2`; the charge equals the allocation.
- Validator (`_dy_halo_conflict`, both engines, reasons verbatim): every knob
  that would be ignored or cannot compute is rejected, so a kernel name never
  claims a knob that did not take effect.
- `identity_supports_problem` (`kernels/conv/kernel_identity.py`) adds
  `Ho == Hi`, `Wo == Wi` and `Wo <= halo_max_wo` to the filter,
  stride/dilation and `kpg % tile_k` checks.
- The AOT grid builds a capped sub-grid on gfx950, not the geometry product:
  `dy_halo=2` only (`CACHE_DGRAD_HALO_MODES`), the three halo tiles
  (`CACHE_DGRAD_HALO_TILES`), square filters (`CACHE_DGRAD_HALO_FILTERS`),
  `halo_max_wo` buckets (`CACHE_DGRAD_HALO_MAX_WO`), the setprio / pad pairs of
  `_dgrad_halo_knobs` (with and without each on the 4x1-wave tiles, neither on
  the 2x2 tile), K-outer only, A 8 wide (`CACHE_DGRAD_HALO_VEC_A`), B = D in
  `CACHE_DGRAD_HALO_VEC_BC`. `dy_halo=1` remains a JIT knob.
- C++: the spec fields, `_dgrad_dy_halo_ok`, `_dgrad_dy_halo_lds_rows`,
  `_dgrad_lds_charge`, `_halo_dy_descriptor`, the halo K loop and the
  `halo` / `hprio` / `hkp` / `hwo` name tags in Python flag order.

## What the AOT port changed, in one place

| PR (JIT, baked problem) | Port (AOT) |
|---|---|
| `static_sub_gemm` knob, record as immediates | no knob; record rebuilt from kernargs, keyed on the stride capability |
| Flat folded loop + `waves_per_eu` hint + flat-loop read batching | no flat-loop stride-1 binary; those problems take the tilde binary with `uniform_k_decode` |
| `tap_outer_k` default on | field default off; the stride-1 capability is built only with `tap_outer_k=True` |
| XCD order decided at build time from extents | decided per launch from kernargs; build-time gate on arch, groups, fold, split-K |
| `dy_halo` sized for the exact image | sized for `halo_max_wo`, filter baked, served `Wo <= halo_max_wo` |
| C++ refused grouped dgrad | C++ mirrors grouped, the XCD order and the halo loop |

The AOT grid (`_dgrad_jobs`) per wave64 MFMA configuration: the tilde binary
(`uniform_k_decode` from `default_uniform_k_decode`), and next to it a stride-1
/ dilation-1 binary (`CACHE_DGRAD_STRIDE1`, `_dgrad_caps`) whose only K loop is
the tap-outer one (`_dgrad_tap_outer_variants`). A flat-loop stride-1 binary
would serve exactly the problems the tilde binary of its configuration already
serves with the same K loop, so it is not built. The tilde binary keeps serving
stride 1 too. On top, the halo sub-grid above. `KernelIdentity` gains
`tap_outer_k`, `dy_halo`, `dy_halo_setprio`, `dy_halo_kouter_pad` and
`halo_max_wo`, all dropped from `KernelIdentity.stable_hash` at their
defaults, so existing identities keep their hashes.

## Validator changes made along the way

- `async_dma`, `unroll_k` and `chiplet_swizzle` are rejected (Python and C++):
  the builder implements none of them, and the first two doubled the charged
  LDS.
- `compv4` stays legal as a schedule policy and is charged one A and one B tile.
- An accumulator tile above `_MAX_ACC_REGS_PER_LANE` fp32 registers per lane is
  rejected (`ROCKE_DGRAD_MAX_ACC_REGS_PER_LANE`; the AOT grid stops at the same
  bound, `CACHE_MAX_ACC_REGS`). The PR's trigger was a 256x256 tile with 1x2
  warps of the 16x16x32 atom: it holds the whole wave64 register file in
  accumulators, spills heavily, and its runtime-record build returned wrong dX
  while the folded build with the same spill footprint passed. That points at a
  compiler or scratch-setup problem, not at the kernel; the bound is a guard,
  not a root-cause fix, and not a no-spill guarantee (some tiles at the bound
  spill too).
- The K-outer B fragment rule moved from a builder raise into both validators,
  so a sweep filters such a configuration instead of losing it to a build error.
- A non-default `lds_k_pad` reaches the kernel name (`_kp<n>`).

## Choosing among the cached binaries

The cache offers a stride-1 problem the tilde binaries, the tap-outer
stride-1 binaries (where `kpg % tile_k == 0`) and, on gfx950, the halo
binaries (square 3x3 / 5x5 / 7x7 filters, `Ho == Hi`, `Wo == Wi`, a
`halo_max_wo` bucket covering `Wo`); `identity_supports_problem` decides
what a binary serves. Which of them runs is the dispatcher's choice, left to
its rework. Findings from fitting that choice, recorded as mechanism:

- **Tile size.** With levers 1-2 in place, a 128x128x64 tile (2x2 waves,
  16x16x32) halves operand traffic per MFMA against 64x64x64 (2x2 waves,
  32x32x16); the 64x64 tile stayed ahead on grids that land at about one
  workgroup per CU, the larger one from a little above that. Strided
  problems were not tuned (small-M, few-channel and strided problems want the
  larger grid). On small channel groups (`cpg <= 32`) a 64-wide N tile is
  mostly empty, so narrow N tiles with a taller M tile (and `tile_k` 32 where
  `kpg <= 32`) keep the MFMAs full where the grid stays large enough.
- **Pointwise.** A 128x64 entry for small pointwise problems halved an
  already small grid, with no grid threshold separating its wins from its
  losses; 1x1 problems run the general kernels.
- **`tile_k` 32 for `kpg % 64 == 32`** kept the tap-outer loop at twice the
  trip count and won in back-to-back launch timing, but lost across classes
  when re-timed with a large fill before every launch to evict L2 and the
  last-level cache. A dgrad kernel in training normally runs after other
  kernels have replaced its operands in cache, so the cold result is the one
  to follow: on those problems a `tile_k` 64 tile on the tilde binary is
  the better choice.
- **dY halo tiles.** Among 64x64 (2x2 waves), 128x64 and 256x64 (4x1 waves),
  occupancy is the LDS-limited workgroup count per CU of the *bucketed*
  binary. With one workgroup per CU on a multi-round grid nothing hides the
  per-chunk halo load; on a grid that fits one round the halo still wins.
  Where the plain 128x128 tile covers a group's input channels in one or two
  exact N tiles it reads each dY row once, while the 128x64 halo tile reads it
  twice and at low occupancy exposes the halo load. Reuse is
  `taps * tile_m / (tile_m + (Y-1)*Wo + (X-1))`: a small filter on a wide
  image stages nearly as many rows as the tap-outer loop gathers anyway. On
  256x64 with many taps the per-tap row masks lift the register count to one
  wave per SIMD. One-dimensional filters showed losses no single threshold
  separated; the cache's square filters exclude them.
- **Halo knobs.** `dy_halo_setprio` and `dy_halo_kouter_pad` pay on the
  4x1-wave tiles where the padded binary keeps the LDS-limited workgroups per
  CU; on 256x64 with a large dY tensor one padded workgroup per CU ran ahead
  of two unpadded ones, and the bucketed `halo_max_wo` puts many images on
  that occupancy step.

The PR also capped the halo at 81 taps for code size and compile time; with
the cached filters stopping at 7x7 the cap is moot and the constant is gone.

## Correctness

- **Bit identity.** The tap-outer loop reduces in the flat loop's order; the
  XCD remap is a bijection of the grid. `TestConvDgradStride1Specializations`
  in `tests/test_conv_dgrad_correctness.py` builds the stride-1 binaries once
  against the AOT probe (dilation 1 and 2), the tap-outer binaries (K-outer and
  M-outer), a launch-order `_noxcd` build and the tilde binary (stride-2 probe),
  launches them through `ConvArgs` launch values on adversarial stride-1 shapes
  (odd H/W, a single image, 5x5 and 1x3 filters, pad 0/1/2 and asymmetric,
  partial M/N tiles, `kpg` not a multiple of `tile_k`, grouped, dilation 2)
  with dX poisoned, and requires `torch.equal` across the binaries the cache
  would offer plus the reference tolerance, with split-K against the reference.
- **Record equivalence.** `tests/test_conv_abi.py`: the packed record of the one
  sub-GEMM equals the record rebuilt from launch values over stride-1 problems,
  dilations, pads, filters, groups, tiles and split-K; the stride-1 IR reads
  only the magic-table entries and runs no search; one IR across C/K, Y/X and
  pad (`test_dgrad_stride1_is_shape_invariant`), across group counts
  (`test_dgrad_stride1_xcd_order_ignores_the_group_count`), and across
  `Wo <= halo_max_wo` (`test_dgrad_halo_ir_is_shape_invariant`).
- **dY halo.** `TestConvDgradDyHalo`: one binary per (tiling, filter,
  `halo_max_wo`) built against a grouped probe, launched on narrower images,
  odd widths, a one-pixel column or image, partial N tiles and groups, with dX
  pre-filled with NaN and NaN/Inf counted as bad;
  `test_halo_max_wo_does_not_change_the_result`. In the PR, a mutation that drops
  the zero-row select failed this test.
- **Policy / validator / identity.** `tests/test_conv_dgrad_spec_policy.py`:
  `TestValidatorRules`, `TestKernelName`, `TestTapOuterK`,
  `TestTapOuterKIdentity` (including
  `test_cache_rejects_tap_outer_on_unaligned_kpg` and the grid carrying it on
  stride-1 identities only), `TestXcdTileOrder` (build-time rule, withheld for
  ungrouped / strided / split-K, emitted ops equal the pure-Python model
  `python_xcd_tile_order` and form a bijection, never swept), `TestDyHalo`
  (LDS sized for `halo_max_wo`, unrolled taps, identity hash / label, the AOT
  halo slice). `tests/test_conv_dgrad_uniform_k_decode.py` covers the tilde
  binary's uniform loop.
- Cache predicates: `test_aot_dgrad_builds_tilde_and_stride1_binaries` and
  `test_cache_serves_halo_binaries_within_their_capabilities` in
  `tests/test_conv_abi.py`.

## Replay

All commands run from `rocke/library` with
`PYTHONPATH=$(pwd)/../platform/python:$(pwd)`. Verify and time in separate
processes: torch's HIP runtime and rocKE's share the process HIP context, and
timings taken in a process that also verified are unreliable.

```bash
# 1. Build the implicit-GEMM AOT cache for dgrad (shape-independent, no GPU).
#    --limit N gives a smoke-sized cache.
python3 -m benchmarks.common.benchmark_implicit_gemm_conv --compile-all \
    --direction dgrad --arch gfx950 --dtype bf16 --cache-dir <igemm cache> --jobs 0

# 2. Every cached binary that serves a shape (tilde, tap-outer, halo buckets),
#    verified against the reference; drop --verify in a separate run to time.
#    This grouped stride-1 problem is served by all three kinds.
python3 -m benchmarks.common.benchmark_implicit_gemm_conv --run-from-cache <igemm cache> \
    --direction dgrad --dtype bf16 --N 2 --Hi 12 --Wi 12 --C 192 --K 192 \
    --groups 3 --Y 3 --X 3 --pH 1 --pW 1 --verify

# 3. One kernel of the same problem, built and timed on its own (no verify),
#    for a trace. --kouter / --tap-outer / --uniform-k default to what the AOT
#    cache builds for the shape.
python3 ../platform/python/rocke/examples/gfx950/conv_dgrad/run_one_dgrad.py \
    --miopen-cmd "./MIOpenDriver convbfp16 -n 2 -c 192 -k 192 -H 12 -W 12 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 3 -F 2 -in_layout=NHWC"
rocprofv3 --kernel-trace -d prof -o run -- \
    python3 ../platform/python/rocke/examples/gfx950/conv_dgrad/run_one_dgrad.py \
    --miopen-cmd "./MIOpenDriver convbfp16 -n 2 -c 192 -k 192 -H 12 -W 12 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 3 -F 2 -in_layout=NHWC" \
    --iters 20

# 4. One JIT config: A/B of the K loops on the same tile (the kernel name differs
#    per variant, so the compile cache cannot alias them). auto = what the AOT cache builds.
python3 ../platform/python/rocke/examples/gfx950/conv_dgrad/run_one_dgrad.py \
    --miopen-cmd "./MIOpenDriver convbfp16 -n 2 -c 96 -k 128 -H 12 -W 12 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F 2 -in_layout=NHWC" \
    --tile-m 128 --tile-n 64 --warp-m 4 --warp-n 2 --warp-tile-mn 32 \
    --tap-outer off --uniform-k off   # plain flat loop (folded record)
#   ... --tap-outer off --uniform-k on  # per-tile uniform decode
#   ... --tap-outer on                  # tap-outer loop (what the stride-1 binary runs)
#   ... --kouter off                    # M-outer B tile for the layout A/B
#   ... --stride 2                      # the tilde path, record searched and loaded
#   ... --print-name-only               # the kernel name for a --kernel-regex capture
```

Single-knob halo, XCD and pad A/Bs have no CLI flag: build `DgradConvSpec` with
`dy_halo` / `dy_halo_setprio` / `dy_halo_kouter_pad` / `halo_max_wo` /
`xcd_tile_order=False` set one at a time (each is tagged in the kernel name), or
compare the cached halo variants of one shape in step 2. Evidence counters:
`rocprofv3 --pmc TCC_HIT_sum TCC_MISS_sum TCC_EA0_RDREQ_sum` for the XCD order
(add `TCC_EA0_RDREQ_DRAM_CREDIT_STALL_sum TCC_TAG_STALL_sum` for the single-N-tile
1x1 class), `SQ_INSTS_LDS SQ_LDS_BANK_CONFLICT` for the pad,
`MeanOccupancyPerActiveCU` for the halo tiles. Main-loop opcode mix: dump the
HSACO with `llvm-objdump -d --mcpu=gfx950` and count opcodes between the backward
branch and its target, as described in
[`isa-inspection-rocke.md`](../../../../../dsl_docs/optimization/utilities/skills/isa-inspection-rocke.md).
The speed-of-light split is a temporary edit around `emit_load_phase` /
`emit_mfma_phase`; do not commit it.

```bash
# Tests.
python3 -m pytest tests/test_conv_dgrad_correctness.py -k "Stride1Specializations or DyHalo"
python3 -m pytest tests/test_conv_dgrad_spec_policy.py tests/test_conv_dgrad_uniform_k_decode.py \
    tests/test_conv_abi.py -k dgrad

# Byte identity, Python vs C++ engine (from rocke/platform).
cd ../platform && python3 tools/check_byte_identity.py --only conv_implicit_gemm_dgrad
```

Parity configs of `library/tests/parity/conv_implicit_gemm_dgrad_emit.py`
for this work: 40-43 (the stride-1 record: dilation 2 with the tilde epilogue, a
grouped 5x5 pad-0 entry with K-outer, `uniform_k_decode` and explicit widths,
split-K, a gfx942 grouped 1x3); 44-48 (tap-outer: 44-46 build, 47 `kpg` not a
multiple of `tile_k` and 48 stride 2 are rejected by both validators); 49-50
(XCD order on and `_noxcd`); 51-56 (dY halo: both modes, setprio, the pad knob,
a non-square filter, groups; 55 without the tap-outer loop and 56 with
`Wo > halo_max_wo` are rejected). The C++ tile loader stages through f16-typed
LDS stores, so bf16 configs cannot be byte-compared and these configs are fp16.
Representative goldens in `platform/tests/golden/rocke_representative_ir_sha256.json`
moved for the stride-1 `conv_dgrad` cases (the record fold), the grouped gfx950
stride-1 cases (the XCD order), and gained a `_tapk` and a halo case.
