<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# gfx950 MXFP6 GEMM optimization plan

Per [`platform/AGENTS.md`](../../../../../../AGENTS.md) §Compliance this plan
records methodology, design math and levers only. Measured throughput, clocks
and utilization stay outside the repository (the tuner writes them to its
ledger). How to rebuild and re-measure is in the [README](README.md).

## Contents

- [Goal and roofline definition](#goal-and-roofline-definition)
- [Starting point](#starting-point)
- [Design math for the target shape](#design-math-for-the-target-shape)
- [Method: knobs and hill-climbing with look-back](#method-knobs-and-hill-climbing-with-look-back)
- [Phase 0: calibration](#phase-0-calibration)
- [Phase 1: structural](#phase-1-structural)
- [Phase 2: instruction and barrier scheduling](#phase-2-instruction-and-barrier-scheduling)
- [Phase 3: direct-compiler co-optimization](#phase-3-direct-compiler-co-optimization)
- [Phase 4: integration](#phase-4-integration)
- [Exit criteria](#exit-criteria)
- [Status](#status)

## Goal and roofline definition

Target: `C[4096, 4096] = A[4096, 65536] @ B[4096, 65536]^T`, FP6 (E2M3 or
E3M2) inputs with E8M0 K32 scales, BF16 output, on gfx950.

The shape is compute-bound: its arithmetic intensity is far above the HBM
ridge point, so the binding roofline is the MFMA one. Two facts shape the
target:

- The nominal FP6 rate assumes the maximum shader clock. With toggling
  operand data the FP6 MFMA pipe is power-limited well below that clock, even
  in a loop with no memory traffic, so the nominal rate is not reachable with
  realistic data.
- The FP6 operands are the most expensive to feed per MFMA cycle of the three
  f8f6f4 formats (see the design math).

The goals are therefore:

1. **MFMA utilization of at least 90% at the clock the kernel actually runs
   at** (`SQ_VALU_MFMA_BUSY_CYCLES` over `GRBM_GUI_ACTIVE` per SIMD). This
   measures the kernel, independent of power.
2. **At least 90% of the measured MFMA-only ceiling for the same operand
   data** (Phase 0 measures it), which folds in the power limit.
3. For shapes that are bandwidth-bound instead, **at least 90% of the HBM
   roofline**. The tuner records fabric read traffic (`FETCH_SIZE`) for every
   accepted configuration so both utilizations are visible.

## Starting point

`instances/gfx950/block_scaled_gemm.py` is a correctness-first builder: one
wave per 32x32 tile, no LDS, the K loop unrolled in Python, one accumulator
chain. Profiling it at the target shape showed:

- very low MFMA utilization, at the full clock;
- the vector L1 near its request limit: each MFMA needs six per-lane,
  row-strided global loads that touch dozens of cache lines each;
- no operand reuse between waves, so the fabric moves many times the input
  footprint even with a high L2 hit rate;
- most wave time in `s_waitcnt`, straight-line code far larger than the
  instruction cache, and serialized MFMAs on one accumulator.

## Design math for the target shape

All figures follow from the gfx950 ISA (32x32x64 f8f6f4 MFMA in 8 passes for
FP6/FP4, 16 for FP8; 64 LDS banks; 160 KiB LDS; 512 registers per lane).

| Quantity, 256x256 tile, 4 waves (2x2) | Value |
|---|---|
| MFMA cycles per K128 stage per SIMD | 32 MFMAs x 32 cycles = 1024 |
| A+B bytes per K128 stage per CU | 2 x 256 rows x 96 B = 48 KiB |
| Global/L2 -> LDS demand at full MFMA rate | 48 B per clock per CU (FP4 and FP8 need 32) |
| LDS operand reads at full MFMA rate | 96 B per clock per CU (of 256 peak) |
| Accumulators per lane | 16 tiles x 16 = 256 (all AGPRs) |
| LDS per stage | 48 KiB, so at most 3 stages |
| Tiles | 16 x 16 = 256, one per CU, no tail |

Consequences:

- The tile must be 256x256: it is the largest the accumulators allow, and
  anything smaller raises the feed demand above what the L2 path can deliver.
- The feed is the structural risk. Each 96-byte row slice touches two 64-byte
  sectors, so the sector traffic is a third above the useful bytes. Phase 0
  measures the L2 -> LDS DMA rate for exactly this pattern.
- With one wave per SIMD nothing hides a stall: the schedule must keep an MFMA
  issuing every 32 cycles while it issues the stage's LDS reads, DMA and waits.

## Method: knobs and hill-climbing with look-back

Every optimization is a defaulted field of `TiledBlockScaledGemmSpec`, so a
configuration is a point in knob space and every experiment is replayable.
[`tune.py`](tune.py) searches the space:

1. Evaluate the starting configuration.
2. Each round, evaluate every single-knob move away from the current best,
   including moves that failed in earlier rounds.
3. Accept the best move if it beats the current best by the threshold
   (default 1%). A knob that failed alone is thereby re-tried on top of every
   newly accepted configuration.
4. If no single move is accepted, **look back**: evaluate pairs of the moves
   that failed this round (on different knobs) and accept the best pair if it
   clears the threshold.
5. Stop when neither step improves, or at the evaluation budget.

Every evaluation passes two correctness gates before it can score: an exact
comparison on a small multi-tile shape with dyadic inputs (every FP32 partial
sum exact), and a sampled comparison of the target output against a float64
reference. Wrong or spilling configurations are recorded, never accepted.
Accepted configurations are profiled for MFMA utilization, effective clock and
fabric read traffic. New knobs join `KNOBS` as they are implemented; a resumed
run re-reads its ledger and never re-measures a configuration.

The [case study](mxfp6_gemm_case_study.md) records each knob's hypothesis and
its keep/reject decision; the ledger records the numbers.

## Phase 0: calibration

| Step | Status |
|---|---|
| MFMA-only ceilings with toggling, constant and zero operand data; effective clock from `GRBM_GUI_ACTIVE` | done |
| Read-bandwidth ceilings: L2-resident, MALL-resident, HBM-streaming | done; the MALL probe needs rework |
| Barrier-free L2-resident streaming through VGPR loads and both LDS-DMA forms, 4 / 8 / 16 waves | done |
| Barrier-free replay of the kernel's DMA address stream, row-major and K-blocked, 2 and 4 stages in flight | done |
| Board power and clocks under a sustained loop; random versus all-zero operand data | done |
| LDS read rate for b128 + b64 fragment reads with and without the swizzle | to do |
| Vendor-library reference at the target shape | done |

The DMA replay is the floor for any kernel with this tiling: with row-major
operands it lies above the MFMA-only ceiling for random data, with K-blocked
operands below it. See the case study, Findings 7 to 9.

## Phase 1: structural

Builder: [`instances/gfx950/block_scaled_gemm_tiled.py`](../../../../instances/gfx950/block_scaled_gemm_tiled.py).

First version:

- 256x256 tile, 2x2 waves of 128x128, accumulators in AGPRs, one wave per SIMD;
- K as a runtime loop of K128 stages, the body unrolled by the ring depth;
- 16-byte buffer-to-LDS DMA into a `stages`-deep ring; K offset and tile base
  in the scalar offset, so a lane keeps six loop-invariant DMA offsets;
- K-block pairing {0,2}/{1,3} so each fragment is one aligned `ds_read_b128`
  plus one `ds_read_b64`;
- per-row chunk rotation (`lds_swizzle`) for conflict-free b128 reads;
- scales prefetched one stage ahead: one A dword per row per stage (shifted
  per MFMA), coalesced B bytes;
- partial `vmcnt` per stage, counted in whole stages of VMEM ops, and one bare
  barrier per stage.

Added since, each a knob:

- `lds_alias_scopes`: per-buffer LDS alias scopes, so the compiler waits only
  for the DMA into the buffer being read;
- `frag_pipeline=step`: step-ahead fragment reads with the next stage's
  barrier between a stage's two K64 steps;
- `a_layout` / `b_layout` / `a_scale_layout` / `b_scale_layout` = `kblock`:
  K-blocked operand and scale layouts, one contiguous block per stage of a
  tile, chosen per operand;
- `mfma_group`: split MFMA blocks with the stage's reads, DMA and scale loads
  placed between them in program order (asm blocks, or intrinsic MFMAs fenced
  by scheduling barriers);
- `a_scale_fetch=iter`: row-major A scales loaded once per row per loop
  iteration for all of its stages;
- `backend=direct`: IR shaped for the direct code-object compiler (intrinsic
  MFMAs, no inline asm), both pipelines;
- diagnostics: combinable `ablate` sets and `dma_probe`;
- instruments: sustained-load timing, an instruction-text hash per code
  object, and paired re-timing before the tuner accepts a move.

**Input contract decision.** The layouts change what the caller passes. With
row-major operands the DMA replay floor caps the kernel well short of the
goals; pre-packed weights (`b_layout=kblock`) recover part of the gap and
fully pre-packed operands and scales the most. Tuning campaigns fix the
contract and climb the remaining knobs; integration (Phase 4) ships the
contract chosen here.

**Canonical-A contract (current direction).** A and its scales stay
row-major, as the caller has them; B and its scales may be pre-packed
(`b_layout=kblock`, `b_scale_layout=kblock` or `kblock4`; with
`b_release=half` B is packed in two K halves). The non-MFMA work and the data
flow are to be hidden behind the MFMAs by fetching deeper, not by changing
A:

| Step | Status |
|---|---|
| A scales: one load per row per loop iteration (`a_scale_fetch=iter`) instead of a scattered dword per stage | done; large gain on both backends |
| MFMAs visible to the compiler: intrinsic split blocks and step pipeline for the direct backend | done (see Phase 3) |
| A data: whole cache lines per row and DMA (`a_ring=line`: three lines per row, B in two K128 slots) | done; large gain on the direct backend, the compute side now binds |
| Compute side: the stage's non-MFMA work issued in MFMA shadows | traced (case study, Finding 15) |
| Fillers split between the steps: line refill and A scale group into step 1 (`fill_split=line1`) | done; gain (Finding 16) |
| Scale byte select in the MFMA with half-rotated scale dwords (`scale_opsel`, one permute per scale dword) | done; gain (Finding 17) |
| Step pipeline for the line ring: barrier between the steps, no read burst after it (`frag_pipeline=step`) | done; gain, B's DMA lookahead now binds at the mid-stage barrier (Finding 18) |
| End-of-K clamps peeled out of the steady-state loop (`k_peel`) | done; a shorter stage but below the threshold, rejected (Finding 19); induction-variable offsets for the DMA and B scale addresses still to do |
| Barrier waits counted from the emitted VMEM list, also over the loop's back edge (no closed forms) | done, byte-identical (Finding 20); a dead-load-aware wait in the compiler still to do |
| VMEM order inside a step (`step_vmem_order`) | done; directions as the memory-pipe model predicts, no full-chip gain, rejected (Finding 21) |
| B released in two K halves with an end-of-stage barrier (`b_release=half`, split-packed B) | done; alone the stall only moves; kept with the next row (Finding 22) |
| A scale group every other iteration, split over the lane halves (`a_scale_fetch=iter2`) | done; a gain together with `b_release=half` (Finding 23) |
| B scales as one two-dword load per stage in MFMA byte order (`b_scale_layout=kblock4`) | done; no standalone gain, in the tuner's best (Finding 24) |
| A scales through LDS, eight stages per DMA (`a_scale_path=lds2`) | done; a gain, in the best (Finding 25) |
| 1x4 waves with B straight into registers (`b_path=vgpr`) | done; rejected: the register loads' issue time outweighs the B DMA and reads they replace (Finding 26) |
| Two waves per SIMD (2x4) on the line ring, with A scales in LDS | done; slower: more LDS reads, a shared memory queue, wider barriers; `b_release=half` with 8 waves is now rejected by the spec check (its B half passes did not divide over the threads) |
| Next stage's step-0 reads first in step 1 (`step_vmem_order=frag_first`) | done; the end barrier's read wait goes, the reads stall after the mid-stage barrier instead; rejected (Finding 28) |
| Two A line slots with an in-LDS remnant copy, three B slots, one barrier per stage (`a_ring=line2r`) | done; correct and fewer instructions per stage, but the one-stage refill lead and the copy's writes cost more; rejected (Finding 29) |
| Serpentine MFMA order inside a step (`mfma_order=serp`) | done; a small gain, in the best |
| Four stages of A scales per LDS read (`as_read=b128x4`) and lane-linear B halves (`b_layout=lanes`) | done; the bank conflicts go but were off the critical path; within noise, rejected |
| Next stage's reads from the middle of step 1 (`step_vmem_order=frag_mid`) | done; the reads still land in the refill's window; slower, rejected |
| Every DMA in its own MFMA gap, half of each refill a stage earlier (`dma_place`) | done; spacing alone gains nothing (each DMA stalls longer at issue); B's half first (`even_bfirst`) a gain, in the best |
| Lane half h on K32 block 2h + s with 16-bit A-scale reads (`k_order=t10`) | done; with the plain compiler the scale widening and the unused upper dwords of a 16-byte read add LDS waits; with the compiler's `lds_waw` and `u16_zext` (`cc_extra`) a small gain with both formats, in the best |
| Barrier groups split or sunk below the next step's first MFMA (`sync_place`) | done; the LDS drain shrinks, the barrier wait does not; rejected |
| Fixed cost: prologue waits only for stage 0 (`prologue_wait=stage0`); epilogue variants (`epilogue`, `store_nt`); early accumulator zeroing (`acc_init=early`) | `stage0` a small gain, in the best; the transposed and non-temporal stores coalesce worse, rejected; with the compiler's `acc_init_keep` the zeroing moves before the prologue wait, within noise, kept as a knob; kernarg preload not built (no compiler support) |
| B scales of two stages per four-dword load (`b_scale_fetch=pair`) | done; a gain, in the best |
| The direct compiler's optional passes as kernel attributes (`cc_passes`) | done; a gain, in the best (needs a compiler build with the passes) |
| Arrival skew between the waves of a CU (`wave_skew`) | not built: the DMA issue stall is nearly flat over the arrival order, the queue persists across gaps |
| One M0 per B half with immediate DMA offsets (`dma_m0=bshare`) | done; each wave fills consecutive regions of the half (same bytes per region, so no B layout change), one LDS base and voffset per half with the step in the instruction offset, M0 writes deduplicated by the compiler's `m0_cse`; within noise (the M0 writes were in MFMA shadow), kept as a knob |
| DMAs kept in their builder slots (`cc_extra=pin_vmem`) | done; the listing obeys the slot rules; within noise, kept as a knob |
| Open: fewer B-scale requests shared by the two waves of a column; induction-variable offsets | not built; below the threshold at the measured realization |

Deeper A fetches need more resident K per A row than three K128 stages of
A and B leave room for. Barrier-free replays of the candidate address
streams (Phase 0 method) settle what a deeper fetch must look like:

- One DMA instruction must read each row's chunk contiguously. Issuing the
  K128 pieces of two stages back to back gains nothing; the L1 does not
  merge the split lines.
- What counts is whole cache lines. A row's K128 slice is 96 bytes and its
  lines are 128, so only chunks of a multiple of 384 bytes (K512) are whole
  lines. K256 and K384 chunks give the same partial gain; K512 chunks reach
  the floor of fully pre-packed operands.
- Loading B straight into VGPRs instead of through LDS costs nothing, and
  more loads in flight change nothing: the stream is bandwidth-bound.

The implemented layout (`a_ring=line`, case study Finding 15) keeps 2x2
waves. A is a ring of three whole 128-byte lines per row (96 KiB): three
lines are exactly four K128 stages, so a four-stage loop body fixes every
fragment's slot and offset. Every DMA moves whole lines, and every line
lands two stages before its first read. B keeps two K128 slots (48 KiB), one
stage of lookahead, which the replay shows suffices.

Kept in reserve if B's lookahead ever binds: 1x4 waves of 256x64. Every wave
reads all of A from LDS (the same LDS read traffic as 2x2) and only its own B
columns. B then loads straight into VGPRs from a lane-contiguous pre-packed
layout without duplicated traffic, and LDS holds only A. Built as
`b_path=vgpr` and rejected (Finding 26): in the full kernel, unlike the
barrier-free stream, the register loads block in-order issue.

**Producer-consumer waves** do not fit this tile: registers are allocated
per kernel, so a producer wave costs a consumer's register budget. A
consumer needs the 256 accumulators plus its fragments, more than half a
SIMD's file, so a producer cannot share its SIMD, and a 256x128 tile with
two consumers raises the feed per FLOP by half on a feed-bound kernel.

Knobs to add, in order of expected impact:

| Knob | Idea |
|---|---|
| `barrier_every` | one barrier per two stages with deeper DMA lookahead, so the DMA queue never drains between stages |
| `dma_lead` | issue each stage's DMA right after the barrier that frees its buffer, before the stage's other work |
| `scale_lds` | for row-major scales: stage A scales through LDS, one DMA per several stages |
| `waves_per_cu=8` | two waves per SIMD at 64x128 or 128x64, so one wave's waits hide behind the other's MFMAs |
| `scale_opsel` | done for asm MFMAs and the direct backend (Finding 17); the LLVM intrinsic lowering (both engines) still lacks the selects |
| `epilogue` | LDS-staged, packed BF16 stores |
| `dma_split` | 12-byte DMA chunks (one per 16 FP6 values) as an alternative to the 16-byte rotation layout |
| `k_unroll` | unroll the stage loop beyond the ring depth |

## Phase 2: instruction and barrier scheduling

Knobs (`sched`, `setprio` exist; the rest to add):

- explicit MFMA / LDS-read / VMEM interleave per stage (`sched=interleave`),
  LLVM's canned patterns (`iglp0`, `iglp1`);
- `s_setprio` around MFMA bursts;
- barrier placement: barrier after the next stage's DMA issue versus before;
  wait depth per ring size; one barrier per two stages with a deeper ring;
- `amdgpu-sched-strategy` selection per kernel;
- data-movement energy: fewer address VALU ops, no redundant LDS reads, DMA
  cache policy (`dma_cache`), since every non-MFMA operation costs clock under
  the power limit.

Each change is checked in ATT traces (WaveScope, `ROCKE_DEBUG_LOC=1`): gaps
between MFMA issues and their stall reasons. Tune with zero operand data first
to separate scheduling from power, then with random data.

## Phase 3: direct-compiler co-optimization

The direct code-object compiler is the in-tree crate
[`platform/rust/rocke-codeobj/`](../../../../../../rust/rocke-codeobj/README.md);
the [README](README.md) builds its `target_compile` tool. It is a knob
(`backend=direct`; `bench.py` runs the compiler named by
`ROCKE_DIRECT_TARGET_COMPILE`). It schedules each block with a window list
scheduler that keeps program order within the window, so the builder's
order is the schedule's starting point: split blocks (`mfma_group`) fenced by
`sched_barrier(0)`, which the compiler treats as a scheduling boundary,
place the reads, DMA and scale loads between individual MFMAs.

Compiler changes, ranked by impact on this kernel (done ones are in the
in-tree crate, with unit tests):

1. done: format-aware MFMA timing. Scaled FP6/FP4 32x32x64 runs in 8 passes,
   not 16, for the pipe model, the hazard waits and the schedule cost;
2. done: independent A and B scale VGPRs. The scales were one register pair,
   so every MFMA first copied its two scales into a fresh pair; done: byte
   select, a `scale_sel` attribute on the scaled `tile.mma` encoded as the
   `v_mfma_ld_scale_b32` half's `op_sel` / `op_sel_hi`;
3. one `m0` write per DMA is followed by a hazard `s_nop`: reuse `m0` across
   DMAs of one buffer with immediate offsets, or schedule other work into the
   gap; done: the kernel attribute `m0_early` (builder `dma_m0="early"`) moves
   each `m0` write above the MFMA before its DMA after allocation, so the
   hazard pass needs no `s_nop` for it; no measurable gain (Finding 27),
   so `m0` reuse is unlikely to pay either;
4. LDS address disambiguation (alias scopes per allocation, as
   `lds_alias_scopes` gives the LLVM path), so DMA into the next buffer can
   move around reads of the current one;
5. `ds_read_b96`, b128 + b64 for six-dword reads, 3-dword DMA with immediate
   offsets, a wave64 gfx950 bank model for the swizzle;
6. partial unroll, branch relaxation and immediate-offset folding.

AGPRs for the accumulators are the compiler's choice; it takes no AGPR
budget. `tile.sync` drains `vmcnt` in the direct compiler, so the ring must
use bare barriers with explicit waits (the builder already does).

## Phase 4: integration

C++ mirror of the builder with Python/C++ byte identity, registration in the
GEMM dispatcher and the provider, after the kernel meets the exit criteria.

## Exit criteria

- Both correctness gates pass for every accepted configuration, for E2M3 and
  E3M2.
- MFMA utilization of at least 90% at the achieved clock on the target shape.
- Throughput within 10% of the measured MFMA-only ceiling for the same data.
- No spills; the case study lists every knob with its keep/reject decision.

## Status

The knob-level campaign is closed. The final best is the canonical-A
configuration in the case study ("Current best configurations").

| Phase | Status |
|---|---|
| 0: calibration | done |
| 1: structural | done for the 256x256 tile with one wave per SIMD; the levers left (deeper A ring, third B slot, K256 stages, a barrier per two stages) do not fit the LDS, two waves per SIMD and B into registers are measured no-gos; pre-packed A is outside the canonical-A contract |
| 2: scheduling | done on the direct backend (split MFMA blocks, step pipeline, filler placement, VMEM order); further issue trims realize little while the memory path binds |
| 3: direct compiler | items 1 and 2 done, item 3 done as `m0_early` (no gain); items 4 to 6 open, not expected to pay at this shape |
| 4: integration | not started |

Exit criteria: the gates pass for E2M3 and E3M2 and there are no spills;
the case study records where the throughput criteria stand. With random
data the kernel is limited by the overlap of a
memory side (line-miss latency times misses in flight) and a compute side
(MFMAs at the power-limited clock) that each fit the budget alone (case
study, "Where the remaining gap is"). Meeting the target with random data
needs a structural change outside this tile's LDS and register budget.
