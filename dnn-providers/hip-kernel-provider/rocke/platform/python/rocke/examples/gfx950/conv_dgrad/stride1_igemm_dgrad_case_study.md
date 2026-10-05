<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Stride-1 implicit-GEMM dgrad on gfx950 — case study

Companion to [`dgrad_lds_layout_case_study.md`](dgrad_lds_layout_case_study.md),
which covers the K-outer B tile. This study covers the dense stride-1 3x3 path
of `library/kernels/common/conv_implicit_gemm_dgrad.py`: what dominated its main
loop, the levers tried, which were kept, and how to replay each A/B.

Measured numbers are deliberately absent — see `platform/AGENTS.md`
§Compliance. Recorded here: mechanism, instruction-level evidence, levers,
replay commands and keep/revert decisions.

## Starting point

Every dgrad problem — stride 1 included — ran through the tilde-decomposition
builder `_build_tilde_dgrad`. Each CTA binary-searched a packed record buffer
for its sub-GEMM and loaded 21 record fields with scalar global loads. The
address closures (`dy_descriptor`, `w_descriptor`) then decoded every loaded
vector inside the K loop: `k_sub -> (y, x, k_out)` and `m -> (n, hi, wi)`, with
the divisors read from the record. Their values were unknown to the compiler,
so it emitted a full runtime unsigned divide.

The K loop is a single `scf.for` running load, `sync`, MFMA, `sync`. On the
dense cohort the main-loop ISA had about 9 to 27 VALU instructions per MFMA,
and the address math sat between the global loads and the MFMAs.

## Lever 1: fold the stride-1 record into immediates (`static_sub_gemm`, kept)

At stride 1 and dilation 1 the tilde decomposition has exactly one sub-GEMM.
Its record is a pure function of `(problem, tile_m, tile_n, tile_k, split_k)`,
all of which are spec constants. `static_sub_gemm=True` (the default) emits
each field as an `arith.constant`. This removes three things:

- the binary search;
- the record loads;
- the subtraction `block_start` (it is 0).

The kernel signature is unchanged: `sub_gemm_buf` and `num_sub_gemms` stay
in it and are not read. Every runtime divide by a record field becomes a
divide by a constant, and LLVM strength-reduces it.

Evidence:
- On the `t128x64x64 w4x2 a32x32x16` main loop, the per-iteration VALU count
  drops by about a third.
- The remaining VALU is the per-vector `(y, x, k_out)` / `(n, hi, wi)` decode,
  now in mul-hi/shift form. The compiler narrows much of it to 16-bit ops
  (`v_mad_legacy_u16`, `v_mul_lo_u16`).

Mirrors:
- C++: `_dgrad_folds_sub_gemm_record` / `fold_record` in `_build_tilde_dgrad`, in `platform/cpp/instances/common/conv_implicit_gemm_dgrad.cpp`.
- Parity: config 3 of the parity emitter exercises the runtime-record build on a stride-1 problem; configs 4 and 17 the unfolded ungrouped pointwise problem; config 21 the unfolded large-accumulator flat loop.

### Exclusion: ungrouped pointwise keeps the runtime record (review fix)

A review pass measured the ungrouped stride-1 pointwise (1x1, pad 0) problem
with `rocprofv3 --kernel-trace` over a pointwise cohort spanning small to
large M and C/K, and found the folded build slower than the runtime-record
build at the same tile on every shape in it.

Mechanism, from the ISA of a small 1x1 problem:
- That path already had its own divide-free descriptors (Y = X = 1, so the
  KYXC filter is a plain `[K, cpg]` matrix and the pixel is the GEMM row), and
  it is excluded from the tap-outer loop. Folding removed nothing from its K
  loop.
- What folding did change is the K-loop trip count: it became a constant,
  and LLVM unrolled the loop 2x. The binary went from 4 to 8 MFMAs and from
  4 to 8 `buffer_load`s per loop body, with no extra overlap to show for it.

Decision: `DgradConvSpec.folds_sub_gemm_record` (and the C++
`_dgrad_folds_sub_gemm_record`) excludes `is_pointwise and groups <= 1`.
`static_sub_gemm` stays on by default; for this problem it is simply a no-op,
and the emitted IR equals the `static_sub_gemm=False` build apart from the
kernel name. The dispatch-built HSACO for every small ungrouped pointwise
problem in the cohort is byte-identical to the unmodified tree's. Grouped
pointwise has no such fast path and keeps the fold and the tap-outer loop
(one tap), where it measured faster.

### Flat folded loop: accumulators pinned to VGPRs (second review fix)

A second review measured the folded record against the unmodified tree on
grouped stride-1 problems with `kpg < tile_k`, a 5x5 or 7x7 filter and a
small grid, which take the folded record with the flat K loop (no tap-outer
loop because a K tile straddles taps). The folded build was slower there; the
`static_sub_gemm=False` build of the same tree matched the unmodified tree.

Mechanism, from the ISA of the folded vs runtime-record flat loop:
- Same MFMA and `buffer_load` counts per iteration, no unroll, no scratch.
- The folded loop keeps the MFMA accumulators in AGPRs (`v_mfma ... a[0:15]`)
  and copies all of them AGPR -> VGPR -> AGPR every iteration (16
  `v_accvgpr_read` + 16 `v_accvgpr_write` in the loop body). The
  runtime-record loop has no such copy. With many K iterations and a grid
  smaller than the CU count the copy is on the critical path and outweighs
  the removed record loads and divides.
- The copy is a register-allocation artefact, not something the emitter
  asks for: it also shows up in the unmodified tree's runtime-record loop on
  some ungrouped flat-loop problems.

Root-cause lever, not a narrowing of the fold: on gfx950 the backend chooses
the AGPR form of the MFMA whenever the per-wave VGPR budget may exceed 256,
which is the case at the default occupancy floor of one wave. The attribute
`"amdgpu-agpr-alloc"="0"` alone does not change that on this LLVM (the same
binary came out); `-amdgpu-mfma-vgpr-form` is not an option this LLVM knows.
A `waves_per_eu` floor of 2 (`"amdgpu-waves-per-eu"="2,8"`) caps the budget at
256 VGPRs, the backend then selects the VGPR-form MFMA (`v_mfma ... v[0:15]`),
the loop has zero `v_accvgpr` instructions, and the loop body loses about a
fifth of its VALU work along with the copy.

Experiments, one lever each, same session, `rocprofv3 --kernel-trace`, process
pairs alternating:
- `waves_per_eu = (2, 8)` on every flat folded loop: fixes the slow cohort and
  speeds up most flat-loop problems, but makes problems with 8-byte loads
  (K or C a multiple of 4 but not 8) much slower. ISA: with the VGPR form the
  occupancy-driven scheduler serializes every `buffer_load` with its
  `ds_write` through one register pair (`s_waitcnt vmcnt(0)` after each load)
  instead of issuing all loads first. Narrow loads mean twice as many loads
  per tile, so the serialization dominates.
- `waves_per_eu = (2, 4)`: fixes the narrow-load problems (the scheduler stops
  chasing occupancy) but loses on occupancy-bound wide-load problems.
  Reverted.
- `waves_per_eu = (2, 6)`: same narrow-load loss as `(2, 8)` on that
  8-byte-load problem, so it was reverted at this step. (History: the fourth
  review fix below brought `(2, 6)` back once the narrow-load exclusion
  existed, and it is what ships.)
- Kept at this step: `(2, 8)` only with 16-byte dY and W loads. Narrow-load
  problems keep the plain folded build, which was already faster than the
  unmodified tree on every narrow-load problem measured. Superseded by the
  fourth and fifth review fixes: the shipped hint is `(2, 6)`, on the two
  dispatch warp tiles only.

Rule (`flat_fold_acc_waves_per_eu` in Python, `_dgrad_flat_fold_acc_waves_per_eu`
in C++), as revised by the fourth and fifth review fixes below: emit
`waves_per_eu = (2, 6)` when all of these hold:
- the spec has no explicit `waves_per_eu`;
- gfx950;
- folded record with the flat loop (`folds_sub_gemm_record and not uses_tap_outer_k`);
- `pipeline == "mem"`;
- 16-bit dY and W;
- `load_vec_a == load_vec_b == 8`;
- the warp tile is one of the two dispatch tiles, 64x64x64 `w2x2` `32x32x16`
  or 128x128x64 `w2x2` `16x16x32` (`_ACC_HINT_TILES`, C++
  `ROCKE_DGRAD_ACC_HINT_TILES`; see the fifth review fix).

This is not a new knob. It is a derived default for the existing
`waves_per_eu` field, and an explicit value always wins. The kernel name does
not change. gfx90a/gfx942 also have AGPRs but were not measured, so the rule
stays off there. The tap-outer loop is untouched; some tap-outer kernels show
the same copy, which is a follow-up.

Pinned by `TestFlatFoldAccumulatorHint` in `tests/test_conv_dgrad_spec_policy.py`
(applies, withheld outside the flat folded loop, withheld with narrow loads,
withheld above 128 accumulators per lane with the fold, applied only on the
dispatch warp tiles, explicit value wins, withheld on gfx942). Parity configs
14 (64x64 tile) and 25 (128x128 tile) carry the hint, 18 is the narrow-load
exclusion, 19 the explicit override, 21 the large-accumulator exclusion, and
10, 22 and 23 are folded flat loops on non-dispatch tiles (no hint).

### Large accumulator tiles: no fold and no hint on the flat loop (third review fix)

A review pass swept the 256-accumulator tiles that `is_valid_dgrad_spec`
accepts and the benchmark sweep's tile list reaches (256x256 `w2x2` `16x16`,
256x128 `w1x2` `32x32`, 256x256 `w2x2` `32x32`), which dispatch never picks.
On problems where they take the flat loop:

- The `waves_per_eu = (2, 8)` floor caps arch VGPRs + AGPRs at 256 per lane.
  A tile that alone needs 256 accumulator registers then leaves nothing for
  operands and addresses: heavy VGPR spilling to scratch, far slower than the
  unmodified tree. The rule had no footprint check.
- Without the hint, the folded flat loop itself still spilled on most of these
  configs where the runtime-record build (and the unmodified tree) did not,
  and was slower than it on all but one of them.

Measured (hipEvent, same session, base / folded / runtime record / folded
without the 128 cap, 256- and 128-accumulator configs on dense, grouped,
odd-shaped and small problems; spill counts from the HSACO notes):

- With the hint gated on the footprint, no hinted kernel spills; the
  128-accumulator 256x128 `w2x2` tile keeps the hint, 0 spills and its win.
- The `static_sub_gemm=False` build of every flat 256-accumulator config has
  the unmodified tree's register footprint and time.
- The tap-outer loop at 256 accumulators is the opposite case: folding there
  removes the unmodified tree's spills on most problems and wins by a wide
  margin. It keeps the fold. It is not a win everywhere, though: see the
  fifth review fix for small grouped problems where some of these configs
  spill and are slower than the unmodified tree.

Decision:
- `flat_fold_acc_waves_per_eu` (C++ `_dgrad_flat_fold_acc_waves_per_eu`)
  requires at most `_FLAT_FOLD_MAX_ACC_REGS` = 128 fp32 accumulators per lane
  (`ROCKE_DGRAD_FLAT_FOLD_MAX_ACC_REGS`, half of the validator's cap).
- `folds_sub_gemm_record` (C++ `_dgrad_folds_sub_gemm_record`) also requires
  it unless the tap-outer loop applies (`_tap_outer_loop_eligible`, the
  tap-outer gate without its fold condition). Above the cap the flat loop
  keeps the runtime record, which is the unmodified tree's kernel apart from
  the name. With the fold gone the hint gate is not reachable from the fold
  path any more; it stays so the hint never depends on that.
- Cost: one flat 256-accumulator config on one grouped 5x5 problem was faster
  folded than with the runtime record; it goes back to the unmodified tree's
  time. No dispatch pick is affected (dispatch tiles hold at most 64
  accumulators per lane), and the dispatch-built HSACOs are byte-identical to
  the previous revision's.

Pinned by `test_withheld_above_128_accumulators_per_lane` (flat loop: no fold,
no hint, also with `tap_outer_k=False`; tap-aligned problem: fold and
tap-outer loop kept; 128 accumulators: fold kept) and parity config 21.

### Hint footprint and ceiling (fourth review fix)

The third fix bounded the hint by accumulator count alone (at most 128 per
lane). A fourth review showed that this does not bound register pressure.
At 128 accumulators, the `16x16x32` atom with `tile_k` 64 on one or two warps
(64x128 and 128x64 `w1x1`, 64x256 `w1x2`, 256x64 `w2x1`, 128x128 `w1x2`)
needs more operand fragments per K tile and more staged loads per thread than
the 256-register cap leaves room for. Those configs spilled under the hint
and several ran much slower than the unmodified tree; built without the hint,
every one of them was at or below it. Dispatch never picks these tiles.

Measured (hipEvent, same session, both orderings; base / hint as shipped /
hint stripped, 21 explicit configs x 6 flat-loop problems, spill counts from
the HSACO notes):
- At 128 accumulators the hint helped the `32x32` atom, the `16x16` atom
  with `tile_k` 32, and the four-warp tiles; it hurt or spilled the `16x16`
  `tile_k` 64 tiles on one or two warps. No single register estimate (the
  accumulators plus fragment and staged-load VGPRs) separated the two groups:
  the 64x128 `w1x1` tile has the same estimate with either atom and goes
  opposite ways.
- At 64 accumulators and below, including both dispatch tiles and the
  `tile_k` 128 tiles, the hint did not spill and won or tied on the configs
  measured here. The fifth review swept the whole tile space and found that
  this does not hold in general (see below).

The same sweep showed a second problem on the 64x64 dispatch tile, which holds
only 16 accumulators: on ungrouped problems whose `kpg` is not a multiple of
`tile_k`, `(2, 8)` was slower than no hint (still faster than the unmodified
tree). ISA: the copy it removes is small at 16 accumulators, but with a
ceiling of 8 waves the scheduler targets 64 VGPRs and serializes each staged
`buffer_load` with its `ds_write` through one register quad, the same
pathology the narrow-load exclusion avoids. Without the hint the same loop
issues all loads first and keeps the copy. Sweeping the ceiling on the
dispatch picks of 22 flat-loop problems: `(2, 6)` matched or beat both
`(2, 8)` and no hint on nearly all of them. It keeps the loads batched and
the accumulators in VGPRs. `(2, 4)` lost the grouped wins. The second-review
experiment that rejected `(2, 6)` was on an 8-byte-load problem, which the
narrow-load exclusion now covers.

Decision (the footprint bound was replaced by the fifth review fix):
- `flat_fold_acc_waves_per_eu` (C++ `_dgrad_flat_fold_acc_waves_per_eu`)
  required at most 64 fp32 accumulators per lane, a quarter of the
  validator's cap. The fold keeps its own 128 bound
  (`_FLAT_FOLD_MAX_ACC_REGS`).
- The hint is `(2, _ACC_HINT_MAX_WAVES_PER_EU)` = `(2, 6)`
  (`ROCKE_DGRAD_ACC_HINT_MAX_WAVES_PER_EU`).
- Cost: explicit 128-accumulator configs that won with the hint (`32x32`
  atom, `16x16` with `tile_k` 32, four-warp tiles) lose that win and keep the
  fold alone, which is still faster than the unmodified tree on them. A
  finer rule that keeps those wins (for example 128 accumulators only with
  the `32x32` atom, `tile_k` 32 or four warps) fits the data but was not
  validated under the new ceiling, so it is left as a follow-up. Against
  `(2, 8)`, `(2, 6)` is slightly slower on a few grouped `kpg 16` problems and
  much faster on ungrouped odd-`kpg` and dilated single-sub-GEMM problems. No
  dispatch pick is slower than the unmodified tree.

Not fixed, noted: the fold alone, without the hint, is still slower than the
runtime record on a few explicit flat-loop configs (the 64x128 `w1x1`
`16x16x32` tile on a `kpg` 40 problem, the 64x64x128 `w1x1` tile on a grouped
`kpg` 16 problem, the 256x128x32 tile on a grouped 7x7 problem). Every
other problem measured with those configs is faster folded. No dispatch pick
uses those tiles and no register-count rule separates the cases, so they are
left to the sweep, which can set `static_sub_gemm=False` for them.

Pinned at the time by a 64-accumulator test and parity config 22; that test
is now `test_hint_only_on_the_dispatch_warp_tiles` (fifth review fix).

### Hint only on the dispatch warp tiles (fifth review fix)

A fifth review compiled the whole sweep tile space (tile 32-256, `tile_k` 32
and 64, 1-4 x 1-4 warps, both atoms, both pipelines, both epilogues) on two
flat-loop problems, a dense 3x3 one and a grouped 5x5 one, and timed every
config that took the hint three ways: unmodified tree, hint, hint stripped.
At 64 accumulators or fewer the hint still made a minority of those configs
slower than the unmodified tree (confirmed with kernel traces), and two
hinted configs spilled. For each loss confirmed with kernel traces, the
hint-stripped build of the same config was faster than the unmodified tree,
so the hint caused it. By accumulator count, atom,
warp count and epilogue, nothing separated the configs the hint helps from
the ones it hurts. On the dispatch tiles the hint does help.

Decision:
- `flat_fold_acc_waves_per_eu` (C++ `_dgrad_flat_fold_acc_waves_per_eu`)
  applies only to the two dispatch warp tiles (tile, `tile_k` 64, warps and
  atom; either epilogue), listed in `_ACC_HINT_TILES` / C++
  `ROCKE_DGRAD_ACC_HINT_TILES`. The accumulator-count bound is gone; it is
  implied (16 and 64 accumulators per lane).
- Verification: rebuilt the previously hinted configs with this rule. Every
  non-dispatch config now builds a HSACO byte-identical to the
  hint-stripped build, and the dispatch-tile configs are byte-identical to
  the previously hinted build. All dispatch picks are byte-identical to the
  previous revision. Measured again (hipEvent, both orderings): a small set of
  explicit flat-loop configs remains slower than the unmodified tree, with or
  without the hint. That is the fold-alone case described in the fourth
  review fix, not the hint. Every other config is faster or at parity.
- Cost: explicit tiles where the hint helped lose that gain. A sweep can still
  set `waves_per_eu` explicitly per tile.
- On the 64x64 dispatch tile of a large dense problem, the hint is slightly
  slower than no hint for that config (still faster than the unmodified tree).
  Dispatch picks the 128x128 tile there, where the hint helps.

Pinned by `test_hint_only_on_the_dispatch_warp_tiles` (both dispatch tiles
with both epilogues get `(2, 6)`; the probed examples, such as 256x64 `w2x2`
`32x32`, 64x32 `w1x2` `16x16`, 256x32 `w1x2` `16x16`, 128x128 `w4x1`
`32x32`, a single-warp 64x64 `16x16` tile and the 128x128 tile with `tile_k`
32, keep the fold and get no hint). Parity config 23 (256x64 `w2x2` `32x32`
flat loop, no hint) and config 25 (128x128 dispatch tile on a flat loop,
hint) cover both sides in the C++ mirror.

Tap-outer loop at 256 accumulators on small grouped problems (not fixed,
noted): on a grouped 5x5 problem with 32 channels per group, and on a grouped
3x3 problem with 32 channels per group, some 256-accumulator explicit configs
(mostly `tile_k` 32, almost all with the default epilogue) spill with the
fold and tap-outer loop and run slower than the unmodified tree. The review
suggested dropping the fold when the N tile is wider than the per-group
channels. That rule was measured on 472 such configs across five problems
(two grouped, three ungrouped with 64 or 128 channels): it removes every
loss, but the same rule also removes large wins on most of those configs,
including every config on the ungrouped 64-channel problems, so it was not
adopted. Neither the N-tile overhang nor the warp count, atom or pipeline
separates the losses. The epilogue nearly does: the `cshuffle` epilogue
loses on almost none of them. These configs are not dispatch picks, and a
sweep can use `cshuffle` or `static_sub_gemm=False` for them.

### Flat folded loop: batch the global reads (found by adversarial measurement)

An adversarial measurement round sampled a class no earlier cohort had
covered: grouped problems with `cpg % 8 != 0` (channels per group too wide
for the direct candidate) together with `kpg % 64 != 0`, so the folded record
runs the flat K loop, plus dense problems with `cpg % 4 != 0`. Through
production dispatch the folded flat loop was reproducibly slower than the
unmodified tree there. An arm that only forced the runtime record
(`static_sub_gemm=False`) on the same tree was at parity with the unmodified
tree on every one of those shapes, so the fold alone caused the loss.

Mechanism: the same failure the tap-outer loop hit (see "The first cut was
slower" under lever 2). With the record folded, the per-vector
`load -> store` pairs of `CoalescedTileLoader.load` no longer interleave with
any address math, and the scheduler serializes every staged load behind its
LDS store through one register quad (`buffer_load` -> `s_waitcnt vmcnt(0)`
-> `ds_write`, per vector). The runtime-record build keeps all of the tile's
loads in flight before the first wait. Narrow (8- or 4-byte) loads mean more
vectors per tile, hence the loss on exactly these channel classes. The dense
`cpg = 36/44` problems showed the same serialized loop yet still beat the
unmodified tree, so the vector width alone did not predict which shapes
lost.

Two fixes were considered: keep the runtime record on the flat loop for the
losing class (a dispatch rule), or batch the flat loop's reads the way the
tap-outer loop does. Batching was measured first and removes the cause, so
no class rule was needed.

Decision:
- On gfx950 the flat K loop of a folded record issues `load_global` for A
  and B, then `store_lds` for both (`batch_loads` in
  `_build_tilde_dgrad`; C++ `batch_loads`, `ROCKE_DGRAD_FLAT_FOLD_BATCH_ARCH`).
  The ISA shows all of the tile's `buffer_load`s back to back followed by
  `vmcnt(N-1) .. vmcnt(0)` before the `ds_write`s.
- Not under the `waves_per_eu = (2, 6)` hint (16-byte loads on the dispatch
  warp tiles): its six-wave ceiling already lets the scheduler batch the
  loads, and explicit batching measured slightly slower than the per-vector
  pairs on several of those problems and no faster on the rest. Those
  configs keep the previous IR exactly.
- Not on other targets: gfx90a/gfx942 and the wave32 targets were not
  measured, so their flat folded loop keeps the per-vector pairs (their
  representative goldens are unchanged).
- Measured same-session against the unmodified tree on the regressing class,
  on the earlier wins of the flat folded loop (dense `cpg % 8 == 0` and
  `cpg = 36/44`, 5x5, 1x1 grouped, the 128x128 tile), and on a fresh
  hold-out cohort over random `cpg`, `kpg % 64 != 0`, groups 1-8, 1x1 to 7x7,
  1x3 / 3x1 filters and pad 0: no shape is slower than the unmodified tree,
  the previously losing shapes are now faster than it, and the dense
  `cpg = 36/44` wins grew. All picks pass the reference check (fp64
  reference, NaN-poisoned outputs, NaN counted as a failure).
- Pinned by `TestFlatFoldLoadBatching` (the grouped `cpg = 36` and dense
  `C = 50` flat loops batch; the tap-outer loop batches; the runtime record,
  gfx942 and the hinted config keep the per-vector pairs). Parity configs 7,
  10, 18, 19, 22 and 23 take the batched flat loop; configs 14 and 25 (hint)
  and the wave32 configs 8, 9 and 13 keep the pairs.

## Lever 2: tap-outer K loop (`tap_outer_k`, kept)

The flat loop index is `k_dg = (y*X + x)*kpg + k_out`. When `kpg % tile_k == 0`,
no K tile straddles two filter taps. The loop can then be split into two:
tap `t = y*X + x` in the outer loop, output-channel chunk `kb` in the inner loop.
The reduction order is unchanged (tap-major, then chunk), so the result is
bit-identical to the flat loop. The test suite asserts this with `torch.equal`.

The descriptors are rewritten so that each loop-invariance class is its own
operand subtree:

| Term | Depends on | Lands in |
|---|---|---|
| `(n, hi, wi)` decode, `((n*Ho + hi)*Wo + wi)*K + col`, `col*Y*X*C + c` | thread only | kernel preamble (LICM) |
| `dh = pH - y`, `dw = pW - x`, dY bounds predicate, `pix + (dh*Wo + dw)*K` | tap only | outer loop body |
| `+ kb` (dY), `kb*Y*X*C + t*C` (W, one scalar) | chunk | inner loop (one add per vector) |

The dY descriptor returns the raw offset and its predicate without a select of
its own. The loader already maps an invalid lane to the out-of-bounds
sentinel, which the buffer load returns as zero, so the old double select
(descriptor `select(valid, off, 0)`, then loader `select(valid, bytes, OOB)`)
collapses to one select. The W descriptor returns no predicate, because
`k_abs < kpg` and `t < Y*X` hold by construction.

Evidence (`t128x64x64 w4x2 a32x32x16`, bf16):
- Inner-loop VALU went from tens of instructions per K tile to a handful.
- LLVM unrolls the inner loop by two.
- The remaining VALU is one `v_add` plus one `v_cndmask` per A vector, and one `v_add` per B vector.

### The first cut was slower: batch the global reads (part of lever 2)

With the address math gone, the per-vector `load -> store` pairs that
`CoalescedTileLoader.load` emits left the scheduler nothing to overlap. The ISA
showed the following sequence for every vector:

```text
buffer_load_dwordx4 v[50:53], ...
s_waitcnt vmcnt(0)
ds_write_b128 ..., v[50:53]
```

A single VGPR quad was reused, so the global latencies were fully serialised.
Wide tiles still improved; the narrow `t128x64` tile regressed below the
lever-1 build.

The fix: on the tap-outer path the load phase issues
`load_global` for A and B, then `store_lds` for both. The ISA then shows all
of the tile's `buffer_load`s back to back, followed by `vmcnt(N-1) .. vmcnt(0)`
before the `ds_write`s. The narrow tile became the fastest config, and every
dense shape improved over lever 1.

Mirrors:
- C++: `_tap_dy_descriptor` / `_tap_w_descriptor` and the `use_tap_outer` branch.
- Parity: configs 0-2, 11, 12, 15 and 16 take the tap-outer loop; config 10 takes the flat loop; config 14 has `kpg % tile_k != 0`.

## Lever 3: one-tile register prefetch across (tap, chunk) (reverted)

The global reads for the next tile were issued right after the current tile
was written to LDS, so they overlapped the MFMAs. At a chunk boundary the
next tile is chunk 0 of the next tap. Both taps' bases and predicates are
outer-invariant, so in the loop the choice reduces to a scalar-condition
select. The staged vectors ride both loops' `iter_args`.

The result was correct, but neutral to slower on every shape:
- VGPR use rose by about a third and dropped resident waves per SIMD.
- The 4- and 8-wave workgroups already co-reside two to three per CU and
  hide global latency across workgroups.

The out-of-tree copy is kept with the stream logs, not in this tree.

## Where the time goes after levers 1-2

A speed-of-light split on a dense bf16 3x3 problem with many channels and a small
spatial extent used temporary env-guarded
builds, since reverted:
- full kernel;
- no global loads (LDS + MFMA only);
- no MFMA (global load + LDS write + barriers only).

The global-to-LDS phase is the larger half. This is the structural im2col cost:
dY is re-gathered once per tap, and W is re-read by every M tile.

A direct-convolution kernel loads a halo'd input tile once and reuses it across
all nine taps. That is where the remaining gap lives, and it is out of reach for a
loop-level change to the igemm.

`rocprofv3 --pmc SQ_INSTS_LDS SQ_LDS_BANK_CONFLICT` with the same split builds
shows the following:
- All LDS bank conflicts are in the MFMA phase.
- All of those come from the K-outer B transpose reads (`ds_read_b64_tr_b16`).
  An M-outer B tile shows zero conflicts there.
- The load phase is conflict-free.

## Lever 4: K-outer pad (`_KOUTER_PAD`) (reverted)

The pad was swept over 8-64 elements.

For the `32x32x16` atom on a 64- or 128-wide tile:
- Pad 32 makes the transpose reads conflict-free.
- The reason: four K rows of 64 bytes per 32-lane phase must tile the
  256-byte address class, so the row stride has to be congruent to 64 or 192
  modulo 256.

For the `16x16x32` atom no uniform pad helps:
- Eight K rows of 32 bytes (`k = (l / 16) * 8 + (l % 16) / 4`) collide modulo 256
  for any stride.
- Fixing it would take a K-row permutation applied identically to the write side.

The timing change was inside session noise, so the pad stays 8.

## Lever 5: XCD-contiguous tile order (dense: reverted; grouped: kept)

First attempt, on the dense path this study started from. Two variants were
tried on the folded path:
- a contiguous-chunk-per-XCD remap of the flat tile id;
- the existing `chiplet_aware_super_tile` helper.

Results were mixed and close to the noise floor, so neither was kept for
ungrouped problems, and the `chiplet_swizzle` knob stays rejected for dgrad.

Second attempt, on grouped problems, prompted by a review that found the
default tile below the unmodified tree, warm and cold, on many-group
problems with a small per-group M, a partial N tile and a long reduction
(see "`tile_k` 32 was a warm-cache win" above). The variant arms
isolated the trigger without finding the cause:
- The runtime record (`static_sub_gemm=False`) on the same tile was at
  parity with the unmodified tree.
- Without the `waves_per_eu` hint the flat loop batches its loads, and it lost
  as much as with the hint.
- Only the per-vector load -> store pairs (no hint and no batching) were at
  parity or better. That build lost badly elsewhere, though, for example on
  many-group `cpg` 32 problems.

So any change that made the K loop issue its loads faster lost on this class.
Hardware counters (`TCC_HIT`, `TCC_MISS`, `TCC_EA0_RDREQ`) on a losing
problem showed why. The hinted and batched builds issued the same number of
L2 requests as the serialized build but hit L2 far less often, and sent
correspondingly more reads to the fabric. On a `cpg` 32 problem where the
serialized build lost, the pattern was reversed. The time was set by L2
reuse between workgroups, not by the loop body.

Mechanism. The grid is `(tiles, groups)` with the N tile fastest, and the
hardware hands workgroups to the XCDs round-robin in launch order. Each XCD
has its own L2. With two N tiles per M tile, the two workgroups that read
the same dY rows always land on different XCDs. A group's M tiles, which
all read that group's weight slice, are spread over all eight XCDs, so
every XCD fetches every group's weights. Grouped problems have no operand
shared between groups, so nothing is gained by spreading a group out. How
much of the refetch L2 absorbs then depends on how workgroups that share data
happen to line up in time, and faster load issue changes that alignment.
That is why the loss followed the load pattern without being caused by it.

Fix. `xcd_contiguous_tile_order` (gfx950) remaps the launch-order linear id
`blockIdx.x + blockIdx.y * tiles` with
`rocke.helpers.grid.chiplet_transform_chunked`, one chunk per XCD (chunk =
workgroups / XCDs, remainder in launch order), and splits the result back
into the tile and the group. Each XCD then runs one contiguous range of whole
groups, with both N tiles of every M tile, so each group's operands are
fetched into one L2. Scope:
- Grouped problems only. Ungrouped problems share their operands across all
  tiles, and the first attempt above found no consistent gain there.
- Folded record only. Tried on the runtime record of strided grouped
  problems, the remap won on most problems but lost badly on some, because
  sub-GEMMs of different sizes make the contiguous ranges uneven across the
  XCDs.
- `split_k == 1` (the slices ride `blockIdx.z`), and at least as many
  workgroups as XCDs.
- Not for 1x1 problems with a single N tile (`cpg <= tile_n`); see
  "Single N tile, 1x1" below.
- No knob and no kernel-name tag: like the `waves_per_eu` hint and load
  batching it is a derived gfx950 rule of the builder.
  `_XCD_TILE_ORDER_ARCHES` holds the targets and their XCD counts.
- Python only: the C++ builder refuses grouped dgrad (see "C++ mirror
  scope"), so no parity config or golden changes. The representative grouped
  gfx950 golden has fewer workgroups than XCDs and keeps launch order.

The chunk size was measured too. One chunk per XCD beat 32-, 64- and
128-workgroup chunks (`chiplet_transform_chunked`'s default chunk is 64),
because a co-resident set of workgroups on one XCD then spans a single
contiguous range rather than several ranges far apart.

Validation, with kernel traces of the unmodified tree, the new build and the
previous build (`noxcd`) in one locked session, timed both back to back and
with a 512 MiB fill before every launch:
- The review's losing problems are now well above the unmodified tree, and
  the hint and batching rules no longer lose there.
- An earlier version of this section said that across the review cohorts no
  differently-routed problem was below the unmodified tree in either timing
  mode. That was wrong: the next review found grouped 1x1 problems with a
  single N tile, many groups and a large `N*H*W` below the unmodified tree
  with cold caches (they won with warm caches), and the build without the
  order (`noxcd`) above it on every one. The rule below fixed them.
- With that rule, the review's losing problems, a fresh cohort around the
  boundary (`cpg` 16..128, so one or two N tiles, 1x1 and 3x3, 4..64
  groups, large and small `N*H*W`), the earlier 1x1 single-N-tile probes,
  a sample of the problems where the order won, and the target problems
  were all above the unmodified tree in both timing modes.
- The order still gives up some wins of the previous build, in both timing
  modes, mostly on small grids, where all workgroups are resident at once
  and launch order matters little.

Single N tile, 1x1. With one N tile (`cpg <= tile_n`) and a 1x1 filter,
every dY element is read by exactly one workgroup, and the only operand
tiles share is the group's weight slice, which is small. The order then
has almost no refetch to remove. Hardware counters on losing problems
showed the same fabric read count with and without the order, but more
DRAM credit stalls (`TCC_EA0_RDREQ_DRAM_CREDIT_STALL`) and L2 tag stalls:
the order only moved the dY rows that are streamed at the same time apart.
Across that class the order was slower on average with cold caches, so
`xcd_contiguous_tile_order` withholds it there. Both conditions are needed:
- With one N tile and a 3x3 or larger filter, neighbouring M tiles share dY
  halo rows. The order kept clear wins there, so withholding it on a single
  N tile alone would have given them up.
- With two or more N tiles, the N tiles of an M tile read the same dY rows,
  which is the reuse the order was introduced for.

The class is not uniform. Some single-N-tile 1x1 problems still ran faster
with the order, mostly ones with many groups and few M tiles per group,
where every XCD refetches each group's weights. A weight-refetch term
recovered part of that on the measured problems, but it needs a fitted
cutoff and its gain was small, so it was not added. No problem in the class
is below the unmodified tree with launch order.
- All picks pass the reference check (fp64 reference, NaN-poisoned output,
  NaN counted as a failure), apart from the known unwritten pixels of 1x1
  stride-2 problems, a pre-existing issue in the unmodified tree.
- Pinned by `TestXcdContiguousTileOrder` (applies to grouped folded flat
  and tap-outer loops, never to ungrouped, runtime-record, split-K, strided,
  tiny-grid or gfx942 builds, or to 1x1 problems with a single N tile while
  a second N tile or a 3x3 filter keeps it, and the remap is a permutation
  with one contiguous range per XCD). Two grouped cases with a launch-order remainder
  were added to `_STRIDE1_CASES`, compared bit for bit with the flat and
  runtime-record builds and against the reference.

## Knob defaults

The flat-loop `waves_per_eu` floor above adds no knob; it is the derived
default of the existing `waves_per_eu` field. The flat-loop load batching and
the XCD-contiguous tile order of grouped problems add none either: both are
derived gfx950 rules of the builder.

`static_sub_gemm` and `tap_outer_k` are new knobs that ship default ON, with
the affected goldens re-blessed. That is a deliberate exception to the
default-OFF rule for new knobs: both are productionised optimisations that
dispatch relies on, each has an opt-out that tags the kernel name, and parity
configs cover both the on and off builds.

## C++ mirror scope

The C++ dgrad instance builder lowers the ungrouped layout only (its
descriptors carry no group offset). It now refuses `groups > 1` with an
explicit error instead of emitting IR that would silently differ from the
Python engine; grouped dgrad is built by the Python engine, and no parity
config is grouped. Porting the grouped descriptors and adding a grouped
parity config is a follow-up. The grouped-only XCD-contiguous tile order
(lever 5) is part of that follow-up.

The C++ `CoalescedTileLoader` mirror has no `elem_dtype` (it always loads
`half`), so a bf16 dgrad config cannot be byte-compared. Parity config 20,
the 128x128x64 `w2x2` `16x16x32` tile that dispatch picks for large problems,
is therefore fp16.

## Validator changes made along the way

- `async_dma`, `unroll_k` and `chiplet_swizzle` are rejected by
  `is_valid_dgrad_spec` (Python and C++). The dgrad builder implements none of
  them, and the first two used to double the charged LDS for nothing.
- `compv4` stays legal, because it is a schedule policy here. It is no longer
  charged a second LDS buffer.
- An accumulator tile above 256 fp32 registers per lane is rejected
  (`_MAX_ACC_REGS_PER_LANE` / `ROCKE_DGRAD_MAX_ACC_REGS_PER_LANE`). Root cause
  of the reported wrong-result config, a 256x256 tile with 1x2 warps of the
  `16x16x32` atom on a large dense bf16 3x3 problem:
  - It holds 512 accumulators per lane, the entire wave64 register file.
  - The compiler spills on the order of a thousand VGPRs to scratch.
  - The runtime-record build of that spilled kernel returns wrong dX; the
    folded build happens to pass.
  - Configs at or below 256 accumulators compute correct dX, but this is a
    correctness guard, not a no-spill guarantee: several 256-accumulator
    configs spill on the unmodified tree too (for example 256x256 `w2x2`
    with the `16x16` atom, and 256x128 `w1x2` with the `32x32` atom).
  - This is a guard, not a root cause: a spilling kernel should still be
    correct, and the folded build with the same spill footprint passes, so
    the wrong dX of the runtime-record build points at a compiler or
    scratch-setup problem. It is tracked as an open issue outside this tree;
    forward and wgrad could hit the same failure mode.

## Dispatch: shape-keyed tile table (kept)

With levers 1-2 in place, the dispatcher's single 64x64x64 `w2x2` `32x32x16` tile
was re-derived over a cohort of dense 3x3/5x5 stride-1, 1x1, stride-2 and
grouped problems (`_gfx950_dgrad_tile` in `library/dispatch/grouped_convolution.py`):

- **Ungrouped stride-1 pointwise with `kpg <= 128` (a K loop of one or two
  `tile_k` steps):** the old 64x64x64 default, whatever the grid
  (`_GFX950_DGRAD_POINTWISE_LARGE_MIN_K_STEPS`). See below.
- **Stride 1, `cpg >= 128`, and a 128x128 grid with at least 1.25 workgroups per CU:**
  128x128x64 `w2x2` `16x16x32`. The 128x128 tile halves operand traffic per MFMA.
  This includes large pointwise problems with three or more K steps (wide-C
  pointwise problems with a short K loop need 8 workgroups per CU; see below). The floor was one workgroup per CU
  until a review found the 64x64 tile faster on problems that land exactly on
  it. Re-measured both tiles on the 23 problems the old floor sent to
  128x128. Up to about 1.1 workgroups per CU the 64x64 tile won on all of
  them. From 1.25 the 128x128 tile won on most, and at 1.5 to 1.8 by a wide
  margin, so an alternative floor of two to four waves would have
  regressed those. The floor is now 1.25 (`_GFX950_DGRAD_LARGE_MIN_CTAS`,
  pinned by `test_one_workgroup_per_cu_keeps_the_64x64_tile`).
- **Ungrouped stride-1 pointwise, otherwise:** the old 64x64x64 default,
  unchanged (and, with the exclusion above, the same binary as before).
- **`kpg % 64 != 0` and `kpg % 32 == 0`:** the 64x64x64 default. `tile_k`
  64 straddles taps there and loses lever 2; a `tile_k` 32 entry that kept
  it was tried and removed (see below).
- **Everything else, including all strided problems:** the old 64x64x64 default.
  Small-M, few-channel and strided problems want the larger grid.

Earlier sweeps without levers 1-2 had picked 128x64 `w4x2` for small M. With
the address math gone, the 64x64 tile's larger grid wins those shapes instead.

The first version of the table also sent every smaller pointwise problem to a
128x64x64 `w2x2` tile, with no lower bound on the grid. A pointwise probe
cohort showed that entry halving an already small grid. A re-measurement of
64x64, 128x64 and 128x128 on the unfolded pointwise kernel (kernel-trace,
small to large M and C/K) found no grid-size threshold that separates the
128x64 wins from its losses: it lost on most small and mid problems and also
on the largest one. That largest one is covered by the 128x128 rule. The
entry was dropped (reverted), and `test_small_pointwise_keeps_the_default_tile`
pins the result.

The 128x128 rule did not win on every pointwise problem it selected, though.
A fifth review found ungrouped 1x1 problems with `kpg` 32 or 64 (a single K
step) on large grids running slower on 128x128 than on the old 64x64 tile
(kernel traces). Re-measured with kernel traces across `kpg` 32 to 256:
- `kpg` 64 or less: 64x64 was faster or tied on nearly every problem.
- `kpg` 80 to 112: mixed.
- `kpg` 128: mixed. 128x128 won on some problems and lost on others with
  similar grids, with no grid rule that separates them.
- `kpg` 160 and up: 128x128 won on every problem measured.

So an ungrouped pointwise problem now takes 128x128 only when its K loop has
at least three `tile_k` steps. Every pointwise problem below that builds the
same HSACO as the unmodified tree. The cost is the 128x128 wins at `kpg` 128.
Pinned by `test_short_k_pointwise_keeps_the_64x64_tile_on_large_grids`, and
by `test_large_pointwise_takes_the_128x128_tile`, which also covers a
three-step problem.

A later adversarial pass found a second pointwise corner: wide-C problems
such as the ResNet-50 stage-3 1x1 (`C` 1024, `K` 256, 14x14) at `N` 64 and
128 ran slower on 128x128 than on the unmodified tree, and a variant forced
back to 64x64 on the same tree restored parity, so the tile was the cause.
The pointwise cohort behind the three-step rule had sampled `C` 1024 only at
small batch. The rule was refit on a cohort over `C` 128..3072, `K`
192..2048, `N` 8..256 and 7x7..56x56 images, with kernel traces of the
unmodified tree, the 128x128 tile and the 64x64 tile in one locked session,
then checked on a fresh hold-out cohort:
- `cpg` 768 or less: 128x128 was at parity or faster on every grid above the
  1.25-per-CU floor.
- `cpg` above 768 with a K loop of at most eight `tile_k` steps (`kpg` <=
  512): losses on several grids under 8 workgroups per CU, next to wins on
  similar grids. Neither the grid size nor the fill of the last round of
  workgroups separated them.
- Longer K loops, or 8 workgroups per CU and more: at parity or faster.

So an ungrouped pointwise problem with `cpg > 768` and `kpg <= 512` now needs
a 128x128 grid of at least 2048 workgroups (8 per CU on gfx950;
`_GFX950_DGRAD_POINTWISE_WIDE_*`). Below that it keeps the 64x64 tile, whose
HSACO is byte-identical to the unmodified tree's for every problem in the
cohort. The recorded cost is the 128x128 wins this gives up on wide-C
problems with small grids (for example `C` 1280-1536 at small batch), which
the measured data could not separate from the losses. The new floor is
pinned by `test_wide_c_pointwise_needs_eight_workgroups_per_cu`. As with the
rest of the table, the refit was measured with the LLVM 20 backend only.

### `tile_k` 32 was a warm-cache win (tried, then removed)

With `kpg % 64 == 32`, `tile_k` 32 keeps the tap-outer loop that `tile_k` 64
loses to tap straddling, at twice the K-loop trip count. The table used to
select it there, and back-to-back launch timing (the method used for every
other entry) favoured it on most problems. Three review rounds then found
problems where it ran below the unmodified tree: first a band of many-group,
long-reduction problems on grids of a few workgroups per CU (patched with a
grid band), then many-group problems at `kpg` 96 and above just outside
that band, and few-group 7x7 problems at `kpg` 96. Each patch was another
threshold.

The root cause showed up on a scan of one 7x7, `kpg` 96, `cpg` 64 problem
family. The per-step cost of both tiles had two distinct levels, and which
one a problem landed on depended on how much operand data each XCD's
workgroups touch, not on `N` or the group count alone: `tile_k` 32 won by
a clear margin on the low level and lost on the high one. That pattern
pointed at cache residency between back-to-back launches of the same
kernel. Re-timing with a 512 MiB fill before every launch (which evicts L2
and the Infinity Cache; the fill kernels are filtered out of the trace)
confirmed it. With cold caches `tile_k` 32 lost to the default on the
geomean in every class of a 352-problem cohort (`kpg` 32..288, 1x1..7x7,
1..300 groups, short and long reductions, all grid sizes), and ran below
the unmodified tree on many of them. The default tile never did.

In a training step a dgrad kernel normally runs after other kernels have
replaced its operands in cache, not right after an identical launch. So the
cold result is the safer one to follow. The `tile_k` 32 entry and its band were removed.
These problems now take the 64x64x64 default with the folded record and
the flat K loop. The cost is the warm-cache `tile_k` 32 wins, which the cold
timing shows are not real wins. Pinned by
`test_kpg_odd_multiple_of_32_keeps_the_default_tile`.

An earlier version of this section said the default tile was never below the
unmodified tree with warm caches either, and the investigation notes called
the remaining losses a warm-cache artifact. Both were wrong. The next review
found a class where the default tile ran below the unmodified tree with warm
and with cold caches: many groups with a small per-group M, a partial N tile
(`cpg` 80 or 96) and a long reduction (`kpg` 160..288, 3x3 or 5x5). The
`tile_k` 32 entry had been below the unmodified tree on several of the same
problems, so removing it did not create the class, but it did send every
`kpg % 64 == 32` problem to the kernel that loses there. The cause was the
launch order of the workgroups across the XCDs, not the tile: see lever 5,
which fixed it.

The other gfx950 dgrad dispatch choices (the 128x128 and pointwise entries,
the direct pre-pass pipeline and its decline model) were re-checked under
the same cold-cache timing on the review cohorts. None fell below the
unmodified tree by more than run-to-run noise. Those cohorts were measured with the LLVM 20 backend only.

## Replay

All commands run from `rocke/library`, with `PYTHONPATH` covering
`platform/python` and `library`. For the C++ lowering path, also set
`ROCKE_CPP_STRICT=1` and add the pybind build dir.

```bash
# A/B of the two levers on one config. A different kernel name per variant
# means the compile cache cannot alias them.
python ../platform/python/rocke/examples/gfx950/conv_dgrad/run_one_dgrad.py \
    --miopen-cmd "./MIOpenDriver convbfp16 -n 4 -c 512 -k 512 -H 16 -W 16 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F 2" \
    --tile-m 128 --tile-n 64 --warp-m 4 --warp-n 2 --warp-tile-mn 32 \
    --static-record off --tap-outer off      # baseline body
# ... --static-record on --tap-outer off    # lever 1
# ... --static-record on --tap-outer on     # lever 1 + 2 (shipped)

# Correctness: bit-identity of the three builds + reference tolerance on
# adversarial shapes, plus the policy / validator / dispatch tests.
python -m pytest tests/test_conv_dgrad_correctness.py -k Stride1Specializations
python -m pytest tests/test_conv_dgrad_spec_policy.py tests/dispatch/test_grouped_conv_wgrad_dispatch.py

# Flat folded loop: AGPR copy vs VGPR-form MFMA. Build one flat-loop spec on a
# dispatch warp tile (it gets the shipped waves_per_eu=(2, 6)) and again with
# kernel.attrs["waves_per_eu"] removed, dump both HSACOs and count
# v_accvgpr_read/v_accvgpr_write between the backward branch and its target.
# (Earlier experiments used (2, 8) the same way.)

# XCD-contiguous tile order (lever 5): build a grouped folded spec twice, the
# second time with conv_implicit_gemm_dgrad._XCD_TILE_ORDER_ARCHES = {} (launch
# order), and time both with kernel traces, back to back and with a large
# fill before every launch. The kernel names are the same, so keep the two
# HSACOs apart. For the L2 evidence, run rocprofv3 --pmc TCC_HIT_sum
# TCC_MISS_sum TCC_EA0_RDREQ_sum on each; for the single-N-tile 1x1 class add
# TCC_EA0_RDREQ_DRAM_CREDIT_STALL_sum TCC_TAG_STALL_sum.

# Byte identity, Python vs C++ engine.
cd ../platform && python tools/check_byte_identity.py --only conv_implicit_gemm_dgrad
```

Main-loop opcode mix: dump the HSACO with `llvm-objdump -d --mcpu=gfx950` and
count opcodes between the backward branch and its target, as described in
`dsl_docs/optimization/utilities/skills/isa-inspection-rocke.md`. The
speed-of-light split is a two-line temporary edit around `emit_load_phase` /
`emit_mfma_phase` in the tap-outer loop body. Do not commit it.

## Goldens re-blessed

- `platform/tests/golden/rocke_representative_ir_sha256.json`: the eight
  stride-1 `conv_dgrad` cases per flavor (gfx90a, gfx942, gfx950, gfx1151 and
  gfx1250, ungrouped and grouped). The stride-2 cases are unchanged.
- `platform/tests/instances/differential/golden/llvm_gfx_all.json`
  `ir_canonical.conv_implicit_gemm_dgrad`: all parity configs (re-blessed
  again after the pointwise exclusion: config 4, the ungrouped pointwise
  config, changed back to the runtime record, and config 17 was added). This entry was
  already stale against the unmodified tree before this change; it was
  re-recorded from a green gate and spliced in textually, because the file
  carries duplicate keys that a JSON round-trip would drop. Config 4's hash
  change is not an IR change on the pointwise path: its `.ll` is
  byte-identical to the unmodified tree's.
- After the flat-loop `waves_per_eu` floor: both files again. In the
  representative golden only the two gfx950 stride-1 cases that take the flat
  folded loop changed (the attribute line), in all three flavors; the gfx90a,
  gfx942 and wave32 cases are unchanged. In `ir_canonical`, configs 18-20 are
  new. Config 21 (the large-accumulator exclusion) was added after the third
  review, config 22 after the fourth. After the fifth review, config 10 (a
  folded flat loop on a non-dispatch tile) lost the hint, and configs 23-25
  were added. The representative golden did not change. After the flat-loop
  read batching: `ir_canonical` configs 7, 10, 18, 19, 22 and 23 (the
  gfx950 folded flat loops without the hint); the representative golden did
  not change (its folded flat-loop cases are gfx90a/gfx942, or hinted on
  gfx950). Re-record the
  differential golden with `--record-golden` into a scratch
  copy and splice only the `conv_implicit_gemm_dgrad` block: with `--only` the
  tool rewrites the whole `ir_canonical` section with the one family.
