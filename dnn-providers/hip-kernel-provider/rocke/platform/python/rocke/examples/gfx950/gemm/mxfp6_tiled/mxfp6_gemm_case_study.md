<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# gfx950 MXFP6 GEMM: building and tuning the tiled kernel

Per [`platform/AGENTS.md`](../../../../../../AGENTS.md) §Compliance this
document records **methodology, levers and decisions only**. No measured
throughput, latency, clock or utilization figure appears here or in any
commit message; those stay outside the repository (the tuner writes them to
its ledger). Effects are described qualitatively. The measured scope is one
GEMM shape, `C[4096, 4096] = A[4096, 65536] @ B[4096, 65536]^T` with FP6
inputs, E8M0 K32 scales and BF16 output, on gfx950.

The plan and the design math are in [`PLAN.md`](PLAN.md); this file is the
running record of what was tried, why, and what was kept.

## Table of contents

- [The kernel](#the-kernel)
- [The instruments](#the-instruments)
- [Finding 1: accumulators must be pinned to AGPRs](#finding-1-accumulators-must-be-pinned-to-agprs)
- [Finding 2: LDS reads can sink past a bare barrier](#finding-2-lds-reads-can-sink-past-a-bare-barrier)
- [Finding 3: ring depth and XCD placement](#finding-3-ring-depth-and-xcd-placement)
- [Finding 4: the compiler drained every DMA before every LDS read](#finding-4-the-compiler-drained-every-dma-before-every-lds-read)
- [Finding 5: barrier wait counts must count whole stages](#finding-5-barrier-wait-counts-must-count-whole-stages)
- [Finding 6: step-ahead fragment reads fix compute, not the kernel](#finding-6-step-ahead-fragment-reads-fix-compute-not-the-kernel)
- [Finding 7: the row-major access pattern caps the DMA](#finding-7-the-row-major-access-pattern-caps-the-dma)
- [Finding 8: K-blocked operand and scale layouts](#finding-8-k-blocked-operand-and-scale-layouts)
- [Finding 9: the kernel runs at the power limit](#finding-9-the-kernel-runs-at-the-power-limit)
- [Finding 10: split MFMA blocks fix compute, the DMA binds](#finding-10-split-mfma-blocks-fix-compute-the-dma-binds)
- [Finding 11: two waves per SIMD land at the same limit](#finding-11-two-waves-per-simd-land-at-the-same-limit)
- [Finding 12: the binding limit is instruction issue, not memory waits](#finding-12-the-binding-limit-is-instruction-issue-not-memory-waits)
- [Finding 13: canonical A: the scale loads cost more than the data](#finding-13-canonical-a-the-scale-loads-cost-more-than-the-data)
- [Finding 14: the direct compiler: program order, the scale pair and FP6 timing](#finding-14-the-direct-compiler-program-order-the-scale-pair-and-fp6-timing)
- [Finding 15: whole cache lines for row-major A](#finding-15-whole-cache-lines-for-row-major-a)
- [Finding 16: the line refill belongs in step 1](#finding-16-the-line-refill-belongs-in-step-1)
- [Finding 17: scale byte select in the MFMA](#finding-17-scale-byte-select-in-the-mfma)
- [Finding 18: the step pipeline pays once the line ring feeds it](#finding-18-the-step-pipeline-pays-once-the-line-ring-feeds-it)
- [Finding 19: peeling the end-of-K clamps](#finding-19-peeling-the-end-of-k-clamps)
- [Finding 20: barrier waits from the emitted load list](#finding-20-barrier-waits-from-the-emitted-load-list)
- [Finding 21: the VMEM order inside a step follows the memory pipe](#finding-21-the-vmem-order-inside-a-step-follows-the-memory-pipe)
- [Finding 22: B in two half-stage releases](#finding-22-b-in-two-half-stage-releases)
- [Finding 23: the A scale group every other iteration](#finding-23-the-a-scale-group-every-other-iteration)
- [Finding 24: one B scale load per stage](#finding-24-one-b-scale-load-per-stage)
- [Finding 25: A scales through LDS](#finding-25-a-scales-through-lds)
- [Finding 26: B straight into registers](#finding-26-b-straight-into-registers)
- [Finding 27: early M0 writes](#finding-27-early-m0-writes)
- [Measurement: time a warm, settled GPU](#measurement-time-a-warm-settled-gpu)
- [Where the remaining gap is](#where-the-remaining-gap-is)
- [Current best configurations](#current-best-configurations)
- [Replay](#replay)
- [Knob table and decisions](#knob-table-and-decisions)

## The kernel

[`instances/gfx950/block_scaled_gemm_tiled.py`](../../../../instances/gfx950/block_scaled_gemm_tiled.py)
computes a 256x256 tile per workgroup with four waves (2x2, 128x128 each),
the scaled 32x32x64 f8f6f4 MFMA with FP6 operands, all 256 accumulator
registers per lane in AGPRs, and a ring of K128 stages filled by 16-byte
buffer-to-LDS DMA. Every optimization is a defaulted spec field, so each
configuration is a JSON dict of knob values and every experiment replays.

## The instruments

- [`bench.py`](bench.py) builds one configuration, checks it, and times it
  in its own process. The gates are an exact comparison on a small multi-tile
  shape with dyadic inputs (every FP32 partial sum is exact) and a sampled
  comparison of the target output against a float64 reference. A wrong or
  spilling configuration is recorded and never accepted.
- [`tune.py`](tune.py) is the hill-climb with look-back described in the
  plan. It appends every evaluation to `ledger.jsonl` and profiles accepted
  configurations with `rocprofv3` (`GRBM_GUI_ACTIVE` for the effective
  clock, `SQ_VALU_MFMA_BUSY_CYCLES` for MFMA utilization, `FETCH_SIZE` for
  fabric reads).
- **Ablations** (`ablate`, combinable with `+`): `no_mfma`, `no_dma`,
  `no_lds_read`, `no_scales`. Removing one part of the pipeline at a time
  splits a stage's time into its DMA side and its compute side. Results are
  wrong by design, so the gates are skipped.
- **DMA probes** (`dma_probe`): `l2` (every tile streams the first tile's
  rows, so the DMA hits L2), `contig` (each stage's rows read as one
  contiguous block of the same size) and `l2contig`. They separate cache-line
  fragmentation from L2-miss traffic.
- **Code-object reading**: `llvm-objdump` of the cached code object, split
  at the loop back-edge, with the op mix and the run-length sequence of
  waits, barriers, DMA, LDS reads and MFMAs per loop iteration.
- **Barrier-free microbenchmarks** (kept with the measurements, outside the
  repository): L2-resident streaming through VGPR loads,
  `global_load_lds_dwordx4` and `buffer_load_dwordx4 ... lds` at 4, 8 and 16
  waves per CU; and a replay of exactly this kernel's DMA address stream at
  the target shape, with no barriers, at two and four stages in flight.
- **Power sampling**: `amd-smi metric -p -c -v` while a long launch loop
  runs, plus the same configuration timed with all-zero operand data.
- **Thread trace (ATT)**: `rocprofv3` with `advanced_thread_trace` on one CU,
  decoded with `rocprof-trace-decoder` (a separate install). The K loop's
  per-instruction latency and stall totals in `code.json` give the per-stage
  cycle budget by instruction class.

## Finding 1: accumulators must be pinned to AGPRs

*Hypothesis*: the MFMA intrinsic with 256 accumulator registers per lane
leaves allocation to the compiler, which may not keep them in AGPRs.

*Result*: the compiler picked the VGPR form of the MFMA and copied the
accumulators between register files. Passing the loop-carried accumulators
through an AGPR-output asm (`acc_fence=agpr`) made it worse: it copied
around every MFMA. Emitting each K64 step's MFMAs as one asm block with tied
`=a` outputs (`mfma_form=asm_agpr`) fixed allocation. The hazard recognizer
cannot see inside the asm, so each block ends with an `s_nop`
(`asm_tail_nop`) and the epilogue waits out the last results.

*Decision*: keep `asm_agpr` and `acc_fence=none`. The intrinsic form stays a
knob; the tuner re-tries it every round, and it has lost every time.

## Finding 2: LDS reads can sink past a bare barrier

At three stages the sampled gate failed on a handful of outputs. A bare
`s_barrier` has no memory semantics, so the compiler moved a stage's
fragment reads below the barrier after which another wave's DMA overwrites
that buffer. An empty asm with a `~{memory}` clobber before each stage's
wait pins the reads on the correct side. This is a correctness fix, not a
knob.

## Finding 3: ring depth and XCD placement

- `stages=3` beats `stages=2` by a large margin: with two buffers the next
  stage's DMA has only one stage of compute to hide behind. Three is the
  maximum that fits in LDS for this tile.
- `xcd_remap=xcd_block` gives each XCD a 4x8 block of tiles, so A and B rows
  are reused inside one L2 instead of being fetched by every XCD. Fabric
  reads drop substantially. Kept.
- `lds_swizzle=rot16` (conflict-free b128 reads) beats no swizzle. Kept.

## Finding 4: the compiler drained every DMA before every LDS read

*Observation*: the loop disassembly had `s_waitcnt vmcnt(0)` immediately
after each stage's DMA issue and before its fragment reads, in both pipeline
variants. Every read therefore waited for every DMA in flight, including the
DMA just issued for a future stage, so the ring gave no lookahead.

*Cause*: rocKE allocates all LDS from one pool global. Without alias
information SIInsertWaitcnts must assume any LDS read can alias any
in-flight LDS DMA.

*Lever*: the existing kernel attribute `lds_alias_scopes` gives every
`smem_alloc` its own alias scope, noalias against allocations with disjoint
pooled byte ranges, on both the DMA calls and the LDS accesses. The builder
exposes it as the `lds_alias_scopes` knob; it is opt-in, so no other
kernel's output changes. With six buffers the compiler tracks each buffer's
DMA separately and waits only for the buffer being read.

*Result*: a large gain on the full kernel and on the DMA-only ablation. Kept
in every best configuration since.

## Finding 5: barrier wait counts must count whole stages

With the wait fixed, the disassembly showed the scheduler issuing a stage's
scale loads after its DMA, not before. The stage barrier's `vmcnt` was
computed for program order, so it then forced the DMA just issued to finish.
Each barrier now counts whole stages of VMEM ops (DMA plus scale loads,
whatever their order within the stage), and the prologue waits for stage 0
explicitly. This is a robustness fix. Its before/after timing reused a stale
cached binary (see [Measurement](#measurement-time-a-warm-settled-gpu)), so
its effect on its own was never measured; every later configuration includes
it.

## Finding 6: step-ahead fragment reads fix compute, not the kernel

*Hypothesis*: with one wave per SIMD, each stage starts with its fragment
reads and the MFMAs wait for them. `frag_pipeline=step` reads every K64
step's fragments during the previous step's MFMAs and moves the next stage's
barrier between a stage's two steps. The final stage is peeled. The MFMA
blocks are `sideeffect` asm so the reads keep their program order.

*Result*: with DMA enabled the step pipeline is slower than the stage
pipeline under every layout tried. The barrier, the DMA issue and the scale
loads now sit between the two MFMA blocks of a stage, where nothing else can
issue. An earlier compute-only ablation appeared to favour the step pipeline,
but that ablation had no fragment reads left in it (see
[Measurement](#measurement-time-a-warm-settled-gpu)). With the corrected
ablation the step pipeline does not improve the compute side.

*Decision*: rejected, kept as a knob; the tuner's look-back re-tries it.

## Finding 7: the row-major access pattern caps the DMA

A stage reads 96 bytes of each of 512 rows that are a full K row apart. A
96-byte slice straddles cache lines, so the line traffic is well above the
useful bytes, and the line-request rate per CU is high.

- `dma_probe=contig` (same bytes, contiguous per stage) speeds up the
  DMA-only ablation more than `dma_probe=l2` does, and `l2contig` is
  fastest. Fragmentation matters more than L2-miss traffic.
- The DMA hardware path is not the limit. In the barrier-free L2-resident
  microbenchmark `buffer_load_dwordx4 ... lds` streams faster than VGPR loads
  at both 4 and 8 waves.
- The barrier-free replay of this kernel's own address stream is much
  slower than the L2-resident ceiling for row-major operands, and doubling
  the stages in flight does not help. The pattern is bandwidth-bound, not
  latency-bound, and the replay is a floor for any kernel with this tiling
  and layout. The kernel's DMA-only ablation is only modestly slower than the
  replay, so barriers and waits are a secondary cost.
- The A scale layout made it worse. `A_scale[M][K/32]` gives each lane its
  own cache line for a single dword per stage, more line requests than the
  A/B data itself. The `no_scales` ablation shows a large cost.

## Finding 8: K-blocked operand and scale layouts

`a_layout`/`b_layout=kblock` store each 128-wide K block of all rows
together (`[K/128][rows][96 B]`), so a stage of a tile is one contiguous
block. `a_scale_layout`/`b_scale_layout=kblock` store one dword per row (A)
or column (B) per stage (`[K/128][rows]`), so one scale load instruction
covers one cache line.
B then needs one dword per column instead of a byte per column and step.
`bench.py` packs the inputs (`device_inputs`) and checks against the
row-major reference, so both gates apply unchanged.

*Result*: each is a large gain, and together they give the best
configuration so far. `b_layout=kblock` alone (pre-packed weights,
row-major activations) recovers part of it. K-blocked scales also cut the
register count and speeds up the compute-only ablation, because the B
scale bytes were costing VALU work and VGPRs.

*Decision*: the layouts are part of the kernel's input contract, not tuning
knobs. They are fixed per tuning campaign and are not in `KNOBS`. Which
contract to ship is open: pre-packed weights are common practice, but
pre-packed activations need a producer that writes the layout or a
repacking pass.

## Finding 9: the kernel runs at the power limit

With the K-blocked layouts the DMA-only and compute-only ablations each take
far less than the full kernel. Under a sustained loop with random data the
board sits at its socket power limit and the graphics clock drops far below
its maximum. The same configuration with all-zero operand data is much
faster, while the row-major configuration speeds up much less (it is still
DMA-bound).

Every byte moved (DMA, LDS writes and reads, fabric) costs power the MFMAs
then cannot use. The goal "at least 90% of the MFMA-only ceiling for the same
data" therefore competes with the irreducible data movement of a 256x256 FP6
tile. The goal "at least 90% MFMA utilization at the achieved clock" is
still the right per-cycle target: a lower clock gives the memory side more
cycles per byte.

## Finding 10: split MFMA blocks fix compute, the DMA binds

*Observation*: in the best stage-pipeline loop, the step-1 fragment reads
issue only after all of step 0's MFMAs, and the compiler hoists the next
stage's barrier above step 1's MFMAs. Each stage therefore exposes LDS read
latency twice and a barrier wait while the MFMA pipe is idle. In the step
pipeline, the load-store optimizer merges narrow reads of neighbouring
fragments into `ds_read2st64_b64`. Assembling the six-dword MFMA operands
then needs register copies, and those copies wait on LDS.

*Lever*: `mfma_group` splits each K64 step's asm into blocks of that many
MFMAs. The blocks keep their program order (`sideeffect`). The stage's other
work is placed between them: the next step's fragment reads, and after the
barrier the DMA issue and the next scales. So that work issues in the
shadow of the MFMA pipe. Fragment reads alternate between A and B, so no two
narrow reads of one buffer are adjacent to merge. Each split block starts
with a short `s_nop`, because it can directly follow the VALU shift of its
scales and the hazard recognizer cannot see into the asm. The default (one
block per step) emits byte-identical IR to the code before the split.

*Result*: both gates pass for every group size with both pipelines, and
every split variant is slower than one block per step. The thread trace
(Finding 12) shows why: each split block carries the hazard `s_nop`s the asm
needs. With small blocks the nops consume most of each MFMA's issue shadow,
and the wave spends more of a stage on nops than it saves by interleaving.

*Decision*: kept as a knob (`MFMA_GROUPS`), not adopted. Interleaving needs
MFMAs the compiler can see, so its hazard recognizer pads only where a
hazard exists.

## Finding 11: two waves per SIMD land at the same limit

*Hypothesis*: with one wave per SIMD nothing hides a barrier or a wait. Eight
waves in a 2x4 or 4x2 grid (`waves_m`, `waves_n`, `waves_per_eu=2`,
`agpr_alloc="128,128"`) give each wave a 128x64 or 64x128 tile. It needs half
the accumulators, so two waves fit a SIMD's register file, and one wave's
stall can hide behind the other's MFMAs.

*Result*: both gates pass. MFMAs with fragment reads (the corrected
compute-only ablation) are faster than with four waves, because the second
wave per SIMD hides part of the read issue. The full kernel still lands at
the same throughput as four waves. The thread trace shows the cost moved
rather than vanished. Eight waves skew against each other at every stage
barrier, and their scale and DMA loads stall at issue on the shared
vector-memory queue. Without an AGPR budget the compiler spills; the step
pipeline with one block per step also spills at this register budget.

*Decision*: kept as a spec configuration, not adopted.

## Finding 12: the binding limit is instruction issue, not memory waits

*Instrument*: an ATT thread trace of one CU at the target shape, decoded with
`rocprof-trace-decoder`. The per-instruction latency and stall totals of the
K loop give each instruction class's share of a stage's cycles.

*Result*, best configuration (K-blocked contract, stage pipeline):

- MFMA issue takes about the ideal MFMA time of a stage: back-to-back MFMAs
  issue at the pipe's cadence.
- Nearly all the rest is the issue time of the other instructions: DMA, LDS
  fragment reads, VALU, SALU (mostly `m0` setup for each DMA), scale loads
  and the hazard `s_nop`s after each asm block. They outnumber the MFMAs about
  three to one.
- Waiting on memory is a small share. `vmcnt` and `lgkmcnt` waits and the
  stage barrier together are a few percent. The memory system shows up as
  back-pressure instead: each DMA instruction stalls at issue.

With one wave per SIMD and in-order issue, an instruction that is not an
MFMA can only overlap an MFMA's execution if it is placed directly after
that MFMA in program order. The asm blocks put sixteen MFMAs back to back,
so the other instructions run between blocks while the MFMA pipe drains. A
stage costs roughly the MFMA time plus the issue time of everything else.

In the row-major contract the same trace is dominated by stalls at issue for
the DMA and the byte-wide scale loads, about as large as the MFMA time:
that contract is memory-bound outright.

*Consequence*: reaching high utilization needs the non-MFMA instructions
issued in the shadow of individual MFMAs, and fewer of them. That needs
compiler-visible MFMAs with AGPR accumulators (or a hand-scheduled stage
body), fewer reads per fragment (`ds_read_b96`), scale byte selection in the
MFMA instead of a VALU shift, and `m0` reuse across consecutive DMAs.

## Finding 13: canonical A: the scale loads cost more than the data

*Contract*: A and its scales stay row-major, as the caller holds them; B and
its scales are pre-packed (`b_layout=kblock`, `b_scale_layout=kblock`).

*Observation*: the ablations split a canonical-A stage. The A scale loads
cost more than the fragmentation of A's 96-byte row slices. They are one
dword per row per stage, four rows per lane, and each row sits on its own
cache line. Without MFMAs, dropping the scale loads removes a large part of
the stage. In the full kernel part of their cost hides under the MFMAs, but
the share left is still large.

*Lever*: `a_scale_fetch=iter`. Each loop iteration loads, for every A row of
the lane, the scale dwords of the next iteration's `stages` stages in one
vector load (`global_load_dwordx3` for three stages), and stage u takes
element u. The load for the iteration after the loop starts `stages` dwords
before the row's end, so the tail stages read the last dwords of the row in
bounds. A row's line is touched once per iteration instead of every stage.
The group loads stay out of the per-stage VMEM count. The barrier waits
still count whole stages, and the stage that issues the group loads waits a
little longer than it needs to.

*Result*: both gates pass, and the gain is large on both backends; the
DMA-side ablation (`no_mfma`) improves the most. The full LLVM kernel now
runs only slightly slower than its own `no_mfma` ablation. The memory side
binds the canonical-A contract: row-major A DMA, LDS reads and barriers.

*Decision*: keep `iter` for row-major A scales; it is in `KNOBS`. The next
A-side lever is deeper A data fetches (see the plan, canonical-A contract).

## Finding 14: the direct compiler: program order, the scale pair and FP6 timing

*Bring-up*: `backend=direct` builds the same kernel with intrinsic MFMAs and
no inline asm; the memory-clobber fences go too, because the direct
compiler's barriers order LDS accesses. The direct compiler (its
`target_compile` example) writes the code object. Both gates passed on the
first build. With one MFMA block per step it was slower than the LLVM asm
path.

*Observation*, in the loop listing:

1. After each barrier the compiler issues all of the stage's DMA (each with
   an `m0` write and a hazard `s_nop`), then the 32 fragment reads of both
   steps, then the scale loads, and only then the MFMAs; the second step's
   16 MFMAs run back to back. Its list scheduler issues, within a window of
   the program order, whichever ready instruction can start soonest. While
   the first MFMAs wait for their reads, every other ready instruction goes
   first. And the LDS counter holds at most 15 outstanding reads, so with 32
   reads in flight the first MFMA's wait drains 18 of them, not just its own
   four.
2. Every scaled MFMA was preceded by two `v_mov_b32`. The compiler held the
   A and B scales as one register pair (an instruction had at most four
   operands, and A, B, C and two scales are five), so each MFMA copied its
   two scales into a fresh pair.
3. The FP6 MFMA was timed as a 16-pass one (its opcode's latency class, the
   FP8 rate) in the pipe model and the hazard waits.

*Levers*:

- Builder: `mfma_group` and `frag_pipeline=step` take intrinsic MFMAs. Each
  split block and the fillers placed after it are fenced from the next block
  by `sched_barrier(0)`, a scheduling boundary in both compilers, so the
  interleave holds whatever the scheduler prefers.
- Compiler (the in-tree crate `platform/rust/rocke-codeobj`, with its unit
  tests and the `llvm-mc` encoding cross-check): the scaled MFMA reads its scales as two independent VGPR
  operands (five uses per instruction), and an f8f6f4 MFMA whose A and B
  formats are both 6- or 4-bit takes half the passes (8 for 32x32x64) in
  the pipe model, the hazard waits and the schedule cost.

*Result*: both gates pass for every variant. The compiler changes alone
remove the scale copies and give a small gain. With split blocks the direct
kernel beats the best LLVM configuration of the same contract, and one MFMA
per block is the best group size. The step pipeline regresses on this
backend as it does on LLVM (Finding 6). The clock drops as utilization
rises, so part of each gain goes to the power limit (Finding 9).

*Decision*: the canonical-A contract climbs on `backend=direct` with
`mfma_group=1` as its start. The compiler changes are part of the
in-tree crate. A hill-climb with look-back from there accepted `setprio`
and then stopped: no other single move or pair cleared the threshold.

## Finding 15: whole cache lines for row-major A

*Observation*: ablations of the best canonical-A kernel put both its DMA
side and its compute side above the target's time budget. Barrier-free
replays of the address stream (see the plan) show the DMA side is set by
cache lines. A row's K128 slice is 96 bytes; its lines are 128, so every
stage's DMA splits lines that the next stage fetches again. Only chunks of
whole lines, 384 bytes per row (K512), reach the floor of pre-packed
operands, and splitting a chunk into back-to-back per-stage DMAs does not
help: one instruction must read each row's piece contiguously.

*Lever*: `a_ring=line`. A lives in LDS as a ring of three 128-byte lines per
row (`[slot][row][128 B]`, one allocation). Each DMA moves one line for eight
rows, so every line is fetched once and used whole. Three lines hold exactly
four K128 stages, so the loop body covers four stages; there every
fragment piece's slot and offset are fixed per stage, and one per-lane index
per (stage, step, piece) addresses it. At the start of every stage but the
second of the four, the stage refills the line whose slot the previous stage
freed, two stages before that line is first read. B keeps a two-slot K128
ring. Within a stage B's DMA issues before the refill, and the next
barrier's wait leaves the refill in flight. A per-row rotation of the
16-byte chunks keeps the b128 fragment reads conflict-free.

The ring is one allocation because the two halves of a wave read a fragment
from different lines. The direct compiler tags LDS accesses per allocation
and would make every read wait for the refill landing in another slot. The
kernel attribute `lds_dma_explicit` (a compiler change) leaves those waits to
the kernel's own barrier waits, which order every refill.

*Result*: both gates pass. A pure-Python model of the DMA lane mapping,
the refill schedule and the read indices checked every fragment byte of two
loop iterations before the first GPU run. On the direct backend the line
ring is a large gain over the stage ring at every group size, and the DMA
ablation improves the most. With it the compute side (MFMAs, LDS reads,
scales and barriers, no DMA) is the larger of the two and binds the kernel.
On LLVM the line ring is correct but slow: LLVM waits for every refill
before any read of the ring, so the barriers there drain all DMA.

*Decision*: keep `a_ring=line` for the canonical-A contract on the direct
backend (in `KNOBS`). B straight into VGPRs (1x4 waves) is not needed while
the compute side binds. A hill-climb from the line ring accepted `setprio`
again. Dropping the line rotation (`lds_swizzle=none`) is a large
regression, as the bank-conflict model predicts.

*Thread trace* of the line ring: a stage's non-MFMA instructions (DMA,
fragment reads, scale shifts, scalar address work with an `m0` write and a
hazard `s_nop` per DMA, scale loads, waits and the barrier) take more issue
and stall time than the 32 MFMA shadows of a stage can hold. The DMA
instructions stall at issue while the CU's memory pipeline serves the other
waves' DMA. Reaching the target needs fewer of these instructions per stage
(scale byte select instead of shifts, induction-variable offsets without
per-stage clamps, `m0` writes scheduled away from their DMA) and an even
spread of the rest over the MFMA shadows.

## Finding 16: the line refill belongs in step 1

*Hypothesis*: the thread trace of the line ring with one MFMA per block puts
almost all of a stage's DMA, the line refill last and two per gap, into the
gaps of step 0, where the DMAs stall at issue while the memory pipeline
serves the other waves; the A scale group loads congest it further in the
iteration's first stage. Step 1's gaps carry nothing.

*Lever*: `fill_split="line1"` (grouped stage pipeline only). Step 0 keeps the
step-1 fragment reads, B's DMA and the next stage's scale loads; A's DMA (with
the line ring its refill) and the per-iteration A scale loads become the
fillers of step 1, the loads in the iteration's second stage. VMEM order stays
B, then the scales, then the refill, so the next barrier's wait still leaves
the refill (and after the second stage the scale group) in flight. A tail
stage after the second phase has no group loads, so its successor waits for
everything: an unconditional allowance would let B's DMA through the barrier,
which the CPU replay of every barrier wait catches.

*Result*: both gates pass. The listing keeps the instruction count; the
refill DMAs move behind the seventeenth MFMA of their stage and the longest
run of back-to-back MFMAs shrinks to a few. In the trace the DMA stall almost
disappears, step 0's excess over the MFMA shadows falls to a small fraction
and the barrier shortens; the stage head (the fragment-read burst after the
barrier) is unchanged. A clear gain with random and with zero data; under the
board power limit the clock falls, so the time gain is smaller than the
per-cycle gain. `setprio` no longer matters.

*Decision*: kept (in `KNOBS`); the tuner accepted it. The canonical-A best
uses it.

## Finding 17: scale byte select in the MFMA

*Hypothesis*: sixteen VALU shifts per stage select each operand's scale byte,
and half of them sit between the barrier and the first MFMA. The scaled MFMA
can select the byte itself (`op_sel` / `op_sel_hi` of its `v_mfma_ld_scale_b32`
half), but lane half h needs byte 2h in step 0 and 3-2h in step 1, and the
select is per instruction.

*Lever*: `scale_opsel`. Lane half 1 keeps its scale dwords rotated by 16 bits,
one `v_perm_b32` per dword with a per-lane selector (the identity for half 0),
so both halves read byte 0 in step 0 and byte 3 in step 1. The next stage's
scale dwords are rotated after the stage's last MFMA (they land during the
stage); the per-iteration A scales are carried as rotated dwords, the next
group rotated during the iteration's last stage, each dword after the MFMAs
that read the one it replaces. Asm MFMAs write the selects into `op_sel` /
`op_sel_hi`; intrinsic MFMAs carry a `scale_sel` attribute that the direct
compiler encodes (a compiler change; without the attribute its output is
unchanged). The LLVM intrinsic lowering does not take it, so that
combination is rejected. `llvm-mc` fixed the bit convention (`op_sel` holds
bit 0 of the byte index, `op_sel_hi` bit 1, A in lane 0 and B in lane 1, as
LLVM lowers the intrinsic's selects) before any timing; the exact gate, for
E2M3 and E3M2, confirmed the byte semantics.

*Result*: both gates pass. The loop loses every scale shift and keeps one
permute per scale dword and stage, and most of the back-edge copies of the
A scale group disappear. In the trace the stage head shrinks by the shifts;
step 1 grows a little where a rotation waits for its scale load and where the
group is rotated. A gain over the filler split with both data sets.

*Decision*: kept (in `KNOBS`); the tuner accepted it on top of the filler
split.

## Finding 18: the step pipeline pays once the line ring feeds it

*Hypothesis*: with the filler split and byte select, the largest remaining
cost of a line-ring stage is its head: after the barrier every wave issues
its step-0 fragment reads at once and the LDS serves them in a queue before
the first MFMA. Finding 6 rejected the step pipeline, but with the stage
ring, per-stage A scales and no filler placement.

*Lever*: `frag_pipeline="step"` with `a_ring="line"` (now valid, also with
`a_scale_fetch=iter` and `scale_opsel`). Stage s runs step 0 on fragments read
during the previous stage's step 1; its blocks carry the stage's step-1 reads,
the next stage's scale loads, the line refill (the slot the previous stage
freed is free once its barrier has passed) and in the second stage the next
A scale group. Then `lgkmcnt(0)`, a `vmcnt` that leaves step 0's VMEM in flight,
and the barrier: the next stage's B has landed and every wave is done with
the B slot the following DMA overwrites. Step 1's blocks carry the next
stage's step-0 reads and B's DMA two stages ahead. The last stage reads one
stage past K (clamped addresses) and discards it, so nothing is peeled.

Before the first GPU run a pure-Python model that interprets the builder's
IR for every thread of a workgroup checked every MFMA operand (the 24 bytes of
each lane's A and B block, in order, and the selected scale byte) and the
happens-before rules of every LDS read against the DMA writes, the explicit
waits and the barriers, for this pipeline and the existing ones. It catches a
wait one too large, a missing barrier, a wrong read index, a wrong byte
select and a broken rotation.

*Result*: both gates pass, for E2M3 and E3M2. In the listing each barrier sits
between a stage's sixteenth and seventeenth MFMA and no read stands between a
barrier and the next MFMA; the loop-carried fragments cost a few register
copies (their zero padding). In the trace the head and the read queue are
gone (the fragment reads barely stall) and the compute-only ablation gains
the most. With DMA the gain is smaller: in two of the four stages of an
iteration B's DMA for the next stage has not landed when the mid-stage
barrier waits, which is the lookahead risk of moving the barrier; the step-0
VMEM also stalls more. A gain with both data sets; the clock falls further at
the power limit. Byte select adds little time on top of it, although it
lowers the cycle count.

*Decision*: kept; the tuner accepted it from the filler split with byte
select. Next on this path: give B's DMA more lead inside step 1, and reduce
the VMEM ahead of it.

## Finding 19: peeling the end-of-K clamps

*Hypothesis*: every stage clamps its next stage, its scale stage, its line
refill and the A scale group start against the end of K, with a compare and
a select each, although only the last loop iteration can reach it. In the
step pipeline some of these sit between the barrier and the next MFMA.

*Lever*: `k_peel`. The loop runs one iteration short with the clamps off and
the last iteration is emitted after it with them (the tail keeps them too);
the clamp-free iterations stay inside K. Induction-variable offsets for the
DMA and scale addresses are not part of it.

*Result*: the loop loses every clamp select. The first version failed the
sampled gate at the target shape in the step pipeline while the exact
small-shape gate passed. The peeled iteration was the last of K there and
loaded the next iteration's A scale group, which nothing reads; the direct
compiler drops such dead loads, but the barrier's explicit `vmcnt` counted
them, so part of B's DMA could pass the barrier. The exact shape has a tail
after the peeled iteration, which reads that group. The same holds for a
single-trip loop with nothing after it, whose carried values are dead
after the loop. The builder now loads no next group in a last iteration
without a tail (and the step pipeline's final stage no next-stage scales),
and the functional model drops dead loads as the compiler does, so it
reports the old version. Lesson: a hand-counted `vmcnt` is only right if
every counted load survives dead-code elimination.

With the fix both gates pass. The thread trace shows a consistently shorter
stage (fewer scalar instructions in the MFMA shadows and the barrier gap) on
both pipelines and without DMA, but under the power limit the time gain with
random data stays below the tuner's threshold; with zero data it is larger.
The step pipeline's barrier interval is set by B's DMA wait, not by the
clamps.

*Decision*: rejected, kept as a knob (the tuner re-tries it). Induction-
variable offsets remain the larger half of the idea.

## Finding 20: barrier waits from the emitted load list

*Hypothesis*: every later change to the order or count of VMEM ops in a
stage (new knobs, peeling, unrolling) breaks closed-form `vmcnt` formulas,
as Finding 19 showed.

*Lever*: no knob; a builder change with byte-identical output. The builder
logs every VMEM op it emits with a tag (B's DMA, a line refill, a scale
load, an A scale group load) and marks the step boundaries. A barrier's
wait is the number of ops logged after the last op it needs, before a given
mark. Inside the K loop the count is also taken over the back edge (the
loop body followed by the path to the wait) and the smaller value is used.

*Result*: every existing configuration keeps its IR; the knobs below get
their waits without hand counting. Dead loads (Finding 19) still need the
builder rule and the functional model's dead-load emulation: the log sees
the emitted ops, not the ones the compiler keeps.

## Finding 21: the VMEM order inside a step follows the memory pipe

*Hypothesis*: the order of a step's VMEM ops decides which one waits at
the next barrier. A queue model of the texture-address unit predicts that
moving the A scale group behind B's DMA is clearly worse and that moving
the refill or B's DMA earlier is a small gain.

*Lever*: `step_vmem_order` (`reads_first`, `group_late`, `b_early`,
`refill_early`), line ring step pipeline.

*Result*: the direction of all three moves matches the model, and the
size of the regression matches it closely in a one-CU trace at the full K.
The early refill shortens the one-CU stage more than predicted (the barrier
wait nearly disappears) but not the full-chip time: the stall moves into
LDS-read waits and MFMA issue, and under the power limit the clock follows.

*Decision*: rejected, kept as a knob. One-CU trace gains of a few percent
are not a reliable predictor of full-chip time.

## Finding 22: B in two half-stage releases

*Hypothesis*: B's DMA has half a stage to a stage of lead before the
mid-stage barrier, and the in-order `vmcnt` stacks it behind the refill and
the A scale group, so the barrier waits for B.

*Lever*: `b_release="half"` (line ring step pipeline, K-blocked B). B is
pre-packed in two K halves (K32 blocks 0 and 2, then 3 and 1, the blocks
each step reads), each in its own sub-slot of B's LDS slot. B(s+2)'s first
half is issued first in step 0 of stage s, its second half in step 1. An
end-of-stage barrier frees the first half's sub-slot; the mid-stage
barrier waits only for the first half (and the refill of the stage before,
whose line the next stage's first step reads). Each half gets about one and
a half stages of lead. The packing order keeps each lane's two fragment
reads at fixed offsets, so the reads stay conflict-free.

*Result*: the barrier wait for B disappears in the one-CU trace, but the
stage shortens much less than predicted: the stall moves into DMA issue,
the B scale loads and the second barrier. Alone the time gain is within
noise with random data and negative with zero data. The memory pipe, not
B's lead, limits the stage.

*Decision*: alone, rejected; kept, see Finding 23.

## Finding 23: the A scale group every other iteration

*Hypothesis*: each A scale group load touches one cache line per row, both
lane halves request the same rows, and both waves of a column ask for them
too: many line requests for a few percent of the bytes, in the same memory
pipe as B's DMA.

*Lever*: `a_scale_fetch="iter2"` (with `scale_opsel`, line ring step
pipeline). Lane half h loads the row's scales of iteration j+1+h in one
load, every other iteration, and one cross-half swap per dword gives both
halves both iterations. The loop body holds two iterations; the waits come
from the load log (Finding 20). The second half's start is always clamped,
also in a peeled loop, because it reads two iterations ahead.

*Result*: alone, within noise: the halved group loads only move the stall.
With `b_release="half"` it is a gain with random and zero data in
several runs, and both gates pass for E2M3 and E3M2: the freed memory-pipe
slots are what B's earlier release needs.

*Decision*: kept together with `b_release="half"`. The tuner accepted it
on top of `b_release="half"` and `b_scale_layout="kblock4"` (Finding 24).

## Finding 24: one B scale load per stage

*Hypothesis*: four B scale dwords per lane and stage, rotated for lane half
1, could be one load if the layout already holds each lane's bytes in step
order.

*Lever*: `b_scale_layout="kblock4"`: per stage, column group and lane, two
dwords (one per step) whose byte j belongs to the MFMA of column j, which
selects byte j. One two-dword load per stage, no rotations for B.

*Result*: the loads and rotations go as planned, the barrier waits shrink by
the three loads; the time effect is within the run-to-run noise. The B scale
lines each CU requests do not change: the loads were already whole lines,
only the instruction count drops.

*Decision*: kept in the canonical-A best. Alone it is no gain, but the
tuner's look-back accepted it as a pair with `b_release="half"` (B's halves
and its scales then come from the same per-stage blocks), and the best
climbed from there.

## Finding 25: A scales through LDS

*Hypothesis*: with `a_scale_fetch="iter2"` every wave still loads the A
scales of its own rows from global memory, so both waves of a row pair
request the same lines. Staging the rows' scales through LDS once per
workgroup halves those line requests, and the rest of the memory pipe gets
the slots.

*Lever*: `a_scale_path="lds2"` (line ring step pipeline, `direct`). One DMA
per 32 rows brings each row's 32 bytes of scales for the next eight stages
into a two-slot LDS ring (lane halves fetch the row's two 16-byte halves, so
each row segment is requested once). The barrier before the first read of a
slot waits for its DMA, which was issued a group earlier. Each wave then
reads one scale dword per row-block and stage from LDS.

*Result*: the L1 accesses per stage drop by the predicted amount, the L2
requests barely change, and the stage gets shorter by about what the stage
model predicted, more than the earlier knobs' realized share of their
modelled gains suggested. The
memory-only (no MFMA) floor drops as well. Full-chip time improves with
random and zero data, both gates pass for E2M3 and E3M2, and the tuner
accepted it. With `b_release="full"` it is shorter in one-CU traces too, but
slower on the full chip; the half release stays.

*Decision*: keep `a_scale_path="lds2"` with `b_release="half"`.

## Finding 26: B straight into registers

*Hypothesis*: with B out of LDS the waves need no B DMA, no B LDS reads and
no barrier for B, and the A ring can grow. With 1 x 4 waves each wave
streams its own 64 columns of B straight into registers a step and a half
ahead, and the line requests stay the same if B is packed so that each
load covers whole lines.

*Lever*: `b_path="vgpr"` (line ring step pipeline, `direct`): 1 x 4 waves,
B pre-packed per stage, 32-column block and step as the 64 lanes' 16-byte
pieces followed by their 8-byte pieces, so one 16-byte and one 8-byte load
per fragment each cover whole lines; B scales one dword per lane and stage.
Step 0 of stage t loads the step-1 fragments of stage t+1, step 1 the step-0
fragments of stage t+2; the loads past K are skipped where the stage is
known. The barrier waits only for A's refills and the A-scale DMA.

*Result*: the register budget fits (no spills, with A scales in LDS or in
registers); the barrier waits leave the B loads in flight as planned and the
B `vmcnt` waits stay small. The memory side behaves as predicted: L2
requests unchanged, fewer address stalls in the TA, lower L1 latency, and a
higher clock under the same power limit. But the stage gets longer in
cycles. Issuing the register loads costs far more issue time than the B DMA
and B reads it replaces: the in-order issue blocks while each wave's loads
wait for the TA, and four waves now issue them instead of two. The four
waves also read all of A from LDS at the same time, so the LDS read time
grows. The memory-only floor is much worse than the 2 x 2 kernel's. With
zero data (lower power) it is the fastest configuration; with random data it
loses against the 2 x 2 kernel with LDS A scales, and the tuner rejects it.
With A scales in registers it is worse still, as the model predicted.

*Decision*: reject; kept as a knob. The rest of the planned package (a 4-line
A ring, an even refill and 16-stage A-scale groups, all needing a 16-stage
loop body) was not built, because together it was modelled at less than
the measured loss.

## Finding 27: early M0 writes

*Hypothesis*: every LDS DMA needs its `m0` written by a SALU op with one
wait state in between, and the direct compiler placed each write right
before its DMA, so the hazard pass inserted an `s_nop` per DMA. Moving the
write one MFMA earlier hides the wait state, and the issue slots go to
other work.

*Lever*: `dma_m0="early"` sets the kernel attribute `m0_early`; the direct
compiler then runs a post-allocation pass before the hazard pass that moves
each such `m0` write upward, past instructions that neither touch `m0` nor
define its sources, until a wait state covers it. Without the attribute
the output is unchanged.

*Result*: almost all of the loop's `s_nop`s disappear, and the trace shows
their issue time gone. But the stage barely shortens: the DMAs now wait
longer for the address unit and the MFMA chain stalls a little more, so only
a small part of the removed time reaches the stage. The stage is bound by
the L2 misses in flight, not by issue slots. The full-chip change is within
noise and the tuner does not accept it. The other on-CU trims (fewer `m0`
writes, SALU hoisting, the remaining hazard waits) promise less than this
one and were not built.

*Decision*: reject; kept as a knob.

## Finding 28: the next stage's reads first in step 1

*Hypothesis*: with the half B release, the end-of-stage barrier waits in
`lgkmcnt(0)` for the next stage's step-0 fragment reads, which the round
robin puts into the last blocks of step 1. Issued first, one pair per block,
they land well before that barrier.

*Lever*: `step_vmem_order="frag_first"`: step 1's fragment reads one per MFMA
block from the first (the stage's A-scale dwords with the first ones), B's
step-1 DMA one per block after them. Same instructions and barrier waits.

*Result*: the end barrier's `lgkmcnt(0)` wait nearly disappears, as
predicted. But the reads now issue right after the mid-stage barrier, while
the line refill bunched before that barrier is landing in LDS and all four
waves restart at once: the LDS reads stall at issue for several times the
cycles saved, and the end barrier waits longer for the slowest wave. Slower
with random and Gaussian data, E2M3 and E3M2.

*Decision*: reject; kept as a knob. LDS-read placement has to be costed
against the DMA landing pattern, not only against the next wait.

## Finding 29: two line slots, a remnant buffer and three B slots

*Hypothesis*: a 2-line A ring frees enough LDS for a third B slot if the
32 bytes of each four-stage window's first line that its second stage reads
are copied, in LDS, into a remnant buffer (the line's slot then takes the
window's third line). With three B slots released whole, the barrier
between a stage's steps is the only one: no end-of-stage barrier, no half
release, and B two stages ahead.

*Lever*: `a_ring="line2r"` (step pipeline, `direct`; elsewhere it is the
three-line ring): line l in slot l % 2, the remnant copy (two 16-byte LDS
reads and, eight MFMA blocks later, two writes per thread) in each window's
first stage, B(s+2)'s DMA in step 0 into the slot B(s-1) left, the line
refill in step 1 right after the barrier; it replaces `stages` (3) and
`b_release` (full). The slots and the remnant buffer use a layout whose
32-row block step is the same 256 bytes, every fragment read's address is a
per-lane select, and B and A share one allocation, so no read needs an extra
address add past the 64 KiB DS offset range. LDS is exactly full with A
scales in LDS. B's slot period (3) and the slots' parity (8 stages) make a
24-stage loop body.

*Result*: correct (both gates, single-trip and multi-trip loop bodies and
tails, both formats); the functional model of the ring, the copy and its
provenance and the bank model pass, and every LDS access is conflict-free
except the b64 reads (2-way, as before). Fewer instructions per stage: one
barrier instead of two, less SALU, VALU and hazard `s_nop`. But the two line
slots leave every refill only one stage between the barrier that frees its
slot and the one that needs it, and on the full chip that is not enough: the
barrier now waits for the refill (most of all for the next window's first
line), and the memory-only floor rises too. The copy's LDS writes, issued
while B's DMA lands, stall far longer than estimated, and step 0, which now
also carries B's DMA, reads more slowly. Slower with random data on one
measured board and level on another (it loses less on the slower board).

*Decision*: reject; kept as a knob. A full-stage fragment prefetch alone
does not lengthen the refill window (both ends move together); it does with
a release barrier after the prefetch, which brings back a second barrier per
stage.

## Measurement: time a warm, settled GPU

Three effects made early timings inconsistent:

- A short timing loop right after a process starts catches the clock still
  ramping up from idle, which makes the kernel look slow.
- Back-to-back short evaluations (the tuner) run on a warm GPU and look
  faster.
- Configurations that differ only in knobs with no code effect compiled to
  the same instructions, and one was accepted on noise.

Two more problems invalidated earlier conclusions, and both are fixed:

- **An ablation that deleted work.** With `ablate=no_dma` nothing wrote LDS,
  so the compiler folded every fragment read of the never-written pool away.
  Every compute-only number before the fix measured MFMAs and scale loads
  with no reads at all. `no_dma` now keeps the prologue DMA, so the pool is
  written and the reads stay. Check the loop's op counts in the disassembly
  before trusting an ablation.
- **A cache keyed by kernel name.** `bench.py` cached code objects by name,
  and a builder change keeps the name. A rerun after a code change could time
  the old binary. The cache key now includes a hash of the generated IR.

A fourth problem was a gap in the gates. The exact small-shape check could
not hold the 4x8-per-XCD grid that `xcd_remap=xcd_block` needs, so it
reported "skipped" for every configuration using the remap, and only the
sampled check ran. The remap only permutes workgroups over tiles, so the
exact check now runs the small shape with `xcd_remap="none"` and records the
override; the sampled check still covers the remap at the target shape. Every
best configuration in this study was re-verified with both gates, for E2M3
and E3M2.

Fixes, all in the instruments: `bench.py --sustain-ms` runs back-to-back
launches until the clock has settled before timing (the tuner's default).
`bench.py` hashes the disassembled instruction text (`isa_sha`), and the
tuner never accepts a move whose code equals the incumbent's. A winning
move must also beat the incumbent in alternating re-timings (`--confirm`)
before it is accepted. Compare numbers only within one campaign. `rocprofv3`
counters come from short cold runs, so their clocks are conservative.

## Where the remaining gap is

This is the state at the end of the campaign, for the canonical-A best on
the direct backend at the target shape. Per K128 stage a SIMD needs 1024
MFMA cycles. Same-run ablations of the full chip split the stage time into
a memory side (`no_mfma`: DMA, landing, barriers) and a compute side
(`no_dma`: MFMAs at the power-limited clock, LDS reads, barriers, scales):

1. **Each side alone is close to the target.** The memory side alone and
   the compute side alone each take about the time the target allows per
   stage; the full kernel is clearly slower than either. The gap is the
   **overlap loss**: the two sides do not hide each other.
2. **The memory side is miss latency, not bandwidth.** A K scan shows the
   memory-only stage gets much shorter while A and B stay resident in the
   last-level cache and longer once each first reader misses to HBM; the
   fetch rate is far below the HBM peak. A fixed number of line misses in
   flight per CU times the miss latency sets it. More misses in flight would
   need more LDS, and the 2x2 kernel with A scales in LDS uses all of it.
3. **The overlap loss is mostly cycles, partly clock.** With one wave per
   SIMD the MFMA wave itself issues the DMAs, waits at the barriers for
   landed lines and reads them, so the memory path sits on its critical
   path. Data movement also costs power under the board limit, so the
   full kernel runs at a lower clock than the compute side alone. With
   all-zero data (low MFMA power) the same kernel meets the target time.
4. **On-CU trims barely reach the stage.** Removing issue cycles that are
   off the memory path (hazard `s_nop`s, SALU) only lets the wave reach its
   next wait earlier (Finding 27); moves that shorten the memory path itself
   (fewer line requests, Finding 25) realize their full modelled gain.
5. **The structural levers do not fit.** A deeper A ring, a third B slot,
   K256 stages or one barrier per two stages each need LDS the tile does not
   have. Two waves per SIMD add LDS reads, share the memory queue and widen
   the barriers (Finding 11; also measured on the line ring). B straight
   into registers costs more issue time than it saves (Finding 26). A
   producer wave per SIMD that does only the memory path would take it off
   the MFMA wave's critical path, but every wave of a dispatch gets the same
   register count, and two MFMA-sized waves do not fit the register file.
6. **Outside the contract**: pre-packed A frees LDS for a deeper ring and
   drops the per-row line addressing; it is the only modelled structural
   gain, and still short of the target.

So the remaining gap with random data is power plus imperfect overlap, and
closing it needs the memory path off the MFMA wave's critical path, which
the 256x256 tile with one wave per SIMD cannot provide. Further knob-level
rounds are not expected to pay.

Open directions, in order of expected impact (none with evidence of a gain
inside the contract):

| Direction | Idea | Status |
|---|---|---|
| pre-packed A | contract change: A in K128 slots, LDS for a deeper ring and fewer `m0`/SALU ops | outside the canonical-A contract; the builder has the K-blocked A path, the step pipeline and LDS A scales would need redoing |
| a different tile | a tile whose LDS leaves room for more misses in flight, at a higher feed per FLOP | not modelled |
| B scale requests shared by a column's two waves | fewer line requests on the memory path | small; not built |
| `m0` reuse across DMAs (immediate offsets) | fewer SALU ops | off the memory path; below threshold after Finding 27 |
| LDS-staged, packed BF16 epilogue | shorter fixed cost outside the K loop | the fixed cost is a few percent of the kernel |
| row-major contract | stage A scales through LDS, one DMA per several stages | the row-major DMA floor stays far below the other contracts |

## Current best configurations

Keys not listed keep their defaults. The ledger holds the numbers.

Row-major contract (`A[M][K]`, `B[N][K]`, row-major scales):

```json
{"stages": 3, "xcd_remap": "xcd_block", "lds_alias_scopes": true, "sched": "interleave"}
```

K-blocked contract (pre-packed operands and scales):

```json
{"stages": 3, "xcd_remap": "xcd_block", "lds_alias_scopes": true, "sched": "interleave",
 "a_layout": "kblock", "b_layout": "kblock", "a_scale_layout": "kblock", "b_scale_layout": "kblock"}
```

Canonical-A contract (row-major A and A scales, pre-packed B and B scales),
direct backend (needs the compiler changes of Findings 14, 15 and 17; B
packed in two K halves for `b_release="half"`; A scales through LDS,
Finding 25):

```json
{"stages": 2, "a_ring": "line", "xcd_remap": "xcd_block", "mfma_form": "intrinsic", "agpr_alloc": "auto",
 "backend": "direct", "b_layout": "kblock", "b_scale_layout": "kblock4", "a_scale_fetch": "iter2", "mfma_group": 1,
 "setprio": true, "fill_split": "line1", "scale_opsel": true, "frag_pipeline": "step", "b_release": "half",
 "a_scale_path": "lds2", "mfma_order": "serp", "dma_place": "even_bfirst", "cc_passes": "all",
 "b_scale_fetch": "pair", "prologue_wait": "stage0", "k_order": "t10", "cc_extra": "lds_waw+u16_zext",
 "as_dma_place": "split_light"}
```

This was the best configuration with random operand data as the metric.
(`cc_passes` and `cc_extra` turn on optional passes of the in-tree direct
compiler; `k_order="t10"` is a gain only with `cc_extra`'s `lds_waw` and
`u16_zext`. With a compiler that lacks those passes the best is the same configuration without `cc_passes`,
`cc_extra` and `k_order`. `fill_split` has no effect in the step pipeline; it is the configuration the
tuner climbed through.)

It is the final best of the random-data campaign. A last interleaved check on one GPU
against the moves that had come closest to the tuner's threshold
(`step_vmem_order="refill_early"`, `k_peel`, `k_peel` with
`b_release="full"`, and `refill_early` with `k_peel`) found none faster with
random data; three of the four were slower. Both gates pass for E2M3 and
E3M2.

With Gaussian operand data as the metric the kernel moved on, first to
`k_peel` with `peel_dma="skip"` and `shape_args` off, then to the 16x16x128
scaled MFMA (`mfma_atom`, with `sched16`, `epi16` and `loop_pad`); the knob
table gives each decision. That configuration is
[`best_config.json`](best_config.json). It needs the direct compiler's
16x16x128 support, run with its `fp6pass` and `scalesep` levers (see the
[README](README.md)). The 32x32x64 schedule fields it carries over
(`dma_place`, `as_dma_place`, `k_order`, `k_peel`, `peel_dma` and the others
the builder lists) do not apply on the 16x16x128 path. Both gates pass for
E2M3 and E3M2, the exact gate at K = 1024 and K = 2048 (one and two trips of
eight stages):

```json
{"stages": 2, "a_ring": "line", "xcd_remap": "xcd_block", "mfma_form": "intrinsic", "agpr_alloc": "auto",
 "backend": "direct", "b_layout": "kblock", "b_scale_layout": "kblock4", "a_scale_fetch": "iter2", "mfma_group": 1,
 "setprio": true, "fill_split": "line1", "scale_opsel": true, "frag_pipeline": "step", "b_release": "half",
 "a_scale_path": "lds2", "mfma_order": "serp", "dma_place": "even_bfirst", "cc_passes": "all",
 "b_scale_fetch": "pair", "prologue_wait": "stage0", "k_order": "t10", "cc_extra": "lds_waw+u16_zext",
 "as_dma_place": "split_light", "k_peel": true, "shape_args": false, "peel_dma": "skip",
 "mfma_atom": "16x16x128", "sched16": "fence+pairlight+midlgkm", "epi16": "swap", "loop_pad": 3}
```

The same contract on the LLVM backend:

```json
{"stages": 3, "xcd_remap": "xcd_block", "lds_alias_scopes": true, "sched": "interleave",
 "b_layout": "kblock", "b_scale_layout": "kblock", "a_scale_fetch": "iter"}
```

## Replay

The [README](README.md) has the full reproduction steps, including the
direct compiler build. From `dnn-providers/hip-kernel-provider/rocke/platform`,
on a gfx950 host, with `MXFP6_CACHE` and `MXFP6_LEDGER` set to directories
outside the repository:

```bash
export PYTHONPATH=python ROCKE_BACKEND=python
# one configuration: build, both gates, timing
python -m rocke.examples.gfx950.gemm.mxfp6_tiled.bench --cache "$MXFP6_CACHE" \
  --config '{"stages": 3, "xcd_remap": "xcd_block", "lds_alias_scopes": true, "sched": "interleave"}'
# the same with all-zero operand data (power), or an ablation / probe
python -m rocke.examples.gfx950.gemm.mxfp6_tiled.bench --cache "$MXFP6_CACHE" --data zero \
  --config '{"stages": 3, "xcd_remap": "xcd_block", "lds_alias_scopes": true}'
python -m rocke.examples.gfx950.gemm.mxfp6_tiled.bench --cache "$MXFP6_CACHE" \
  --config '{"stages": 3, "xcd_remap": "xcd_block", "lds_alias_scopes": true, "ablate": "no_mfma+no_scales", "dma_probe": "contig"}'
# a tuning campaign for one contract; keep the ledger outside the repository
python -m rocke.examples.gfx950.gemm.mxfp6_tiled.tune --ledger "$MXFP6_LEDGER" \
  --cache "$MXFP6_CACHE" --pmc best \
  --start '{"stages": 3, "xcd_remap": "xcd_block", "lds_alias_scopes": true, "sched": "interleave", "a_layout": "kblock", "b_layout": "kblock", "a_scale_layout": "kblock", "b_scale_layout": "kblock"}'
# the direct code-object compiler instead of COMGR (ROCKE_DIRECT_TARGET_COMPILE names
# the target_compile tool built from platform/rust/rocke-codeobj, see the README)
python -m rocke.examples.gfx950.gemm.mxfp6_tiled.bench \
  --cache "$MXFP6_CACHE" --config '{"stages": 3, "xcd_remap": "xcd_block", "mfma_form": "intrinsic",
  "agpr_alloc": "auto", "backend": "direct", "b_layout": "kblock", "b_scale_layout": "kblock", "a_scale_fetch": "iter"}'
# CPU checks: every knob value builds and lowers
python -m pytest tests/instances/test_gfx950_block_scaled_gemm_tiled.py -q
```

## Knob table and decisions

"Single move" is the result against the best configuration of its time.
Rejected knobs stay in `KNOBS` and are re-tried every round.

| Knob | Values | Hypothesis | Result | Decision |
|---|---|---|---|---|
| `mfma_form` | `asm_agpr`, `intrinsic` | asm blocks pin accumulators to AGPRs | on LLVM the intrinsic form copies accumulators; the direct compiler keeps them in AGPRs itself | `asm_agpr` on LLVM, `intrinsic` on `direct` |
| `acc_fence` | `none`, `agpr` | asm fence keeps loop-carried accumulators in AGPRs | copies around every MFMA | reject |
| `stages` | 2, 3 | deeper ring hides DMA | large gain with the stage ring; the line ring's lookahead comes from its lines | 3 with the stage ring, 2 with the line ring |
| `lds_swizzle` | `rot16`, `none` | conflict-free b128 reads | gain | keep `rot16` |
| `xcd_remap` | `none`, `xcd_block`, `xcd_block84` | per-XCD tile blocks reuse rows in L2; `xcd_block84`: 8 x 4 blocks instead of 4 x 8, the same L2 footprint, but A rows shared by four XCDs instead of two | `xcd_block` a gain, fewer fabric reads; `xcd_block84` slower with both data sets (same K loop code): the slowest XCDs are a clock effect, not the A-sharing pairs | keep `xcd_block` |
| `lds_alias_scopes` | off, on | let the compiler wait only for the buffer read | large gain | keep on |
| `sched` | `none`, `interleave`, `iglp0`, `iglp1` | MFMA / LDS / VMEM interleave | `interleave` small to moderate gain; `iglp*` regress | keep `interleave` |
| `setprio` | off, on | raise priority around MFMA bursts | regression | reject |
| `waves_per_eu` | 1, 0 | occupancy hint | within noise | default |
| `agpr_alloc` | `256,256`, `auto` | fixed AGPR budget | within noise | default |
| `dma_cache` | 0, 1, 2, 3 | DMA cache policy | GLC within noise to small gain; streaming policies regress | under tuning |
| `frag_pipeline` | `stage`, `step` | read the next step during the current MFMAs | stage ring: full kernel regresses on both backends, the earlier compute-side gain was an ablation artifact; line ring (Finding 18): a gain on `direct`, B's DMA lookahead binds at the mid-stage barrier | `step` with the line ring on `direct`; reject with the stage ring |
| `mfma_group` | 16, 8, 4, 2, 1 | place reads, DMA and scale loads between MFMA blocks | asm blocks: the per-block hazard nops cost more than the interleaving saves; intrinsic blocks on the direct backend: a gain, best at 1 | reject for asm; keep 1 on `direct` |
| `a_layout`, `b_layout` | `row`, `kblock` (B also `lanes`) | contiguous stage blocks; `lanes`: with the half B release, B's halves packed lane-linear so B's LDS reads are conflict-free | large gain; `lanes` removes its bank conflicts but within noise | contract decision; `lanes` rejected, kept as a knob |
| `a_scale_layout`, `b_scale_layout` | `row`, `kblock` (B also `kblock4`) | one cache line per scale load; `kblock4`: one two-dword B load per stage in MFMA byte order | large gain, fewer registers; `kblock4` within noise alone, accepted by the tuner with `b_release="half"` (Finding 24) | contract decision; `kblock4` in the canonical-A best |
| `a_scale_fetch` | `stage`, `iter`, `iter2` | row-major A scales: one load per row per loop iteration, a row's line touched once per iteration; `iter2`: one load per two iterations, split over the lane halves | large gain on both backends (Finding 13); `iter2` a gain only with `b_release="half"` (Finding 23) | `iter2` with `b_release="half"` |
| `step_vmem_order` | `reads_first`, `group_late`, `b_early`, `refill_early`, `frag_first`, `frag_mid` | the VMEM order inside a step decides which op waits at the barrier | direction as the queue model predicts; no full-chip gain (Finding 21); `frag_first` removes the end barrier's read wait but the reads stall after the mid-stage barrier (Finding 28); `frag_mid` (the first post-barrier blocks left free) still puts the reads into the window where the refill lands, slower | reject, kept as a knob |
| `dma_place` | `default`, `even`, `even_bfirst`, `dense_bfirst`, `pair_bfirst` | step pipeline with the half B release: every DMA in its own MFMA gap, two blocks apart, half of each refill moved into the previous stage's step 1, so no two DMAs issue back to back; `dense_bfirst` / `pair_bfirst` keep the split refill and B's half first but put a step's DMAs one / two per gap in consecutive gaps | the back-to-back DMA stalls go, but each single DMA then stalls longer at issue (the address path stays busy), so spacing alone gains nothing; with B's half ahead of the refill (`even_bfirst`) a gain with random and Gaussian data, `even` slower; without the spacing (`dense_bfirst`, `pair_bfirst`) clearly slower: DMAs one gap apart issue faster than the address path drains, so the issue stall and the barrier wait grow more than the MFMA stall shrinks | `even_bfirst` |
| `mfma_order` | `rows`, `serp` | serpentine MFMA order inside a step (odd A rows walk the B columns backward): consecutive MFMAs change one operand | a small gain | `serp` |
| `sync_place` | `none`, `split`, `sink`, `split_sink`, `split_lgkm` | under `dma_place`, split each barrier group into its LDS wait, its DMA wait and the barrier: the waits a block early (`split`, `split_lgkm`), or the group below the next step's first MFMA (`sink`) | the end-of-stage LDS drain shrinks, but the barrier wait does not: the waves still arrive skewed behind the DMA issue stall; within noise or slower | reject, kept as a knob |
| `prologue_wait` | `all`, `stage0` | the prologue waits only for what stage 0 reads; A's line 1 stays in flight until the mid-stage barrier of stage 0 | a small gain (once per tile) | `stage0` |
| `prologue_order` | `default`, `dma_first` | the first fill's DMAs issued one by one among the prologue's per-lane address math instead of after all of it | the first DMA issues much earlier; within noise (once per tile). On the peeled incumbent a gain on Gaussian data, mostly because the longer prologue moves the K loop off its slow placement (as `loop_pad`); nothing left once the loop is placed | reject, kept as a knob |
| `as_dma_place` | `default`, `spaced`, `light0`, `light4`, `split_light`, `split_late` | with `lds2` and `dma_place`: an A scale DMA touches a line per row (row-major scales), several times an operand DMA's address-path work, and the two of an epoch went out back to back before the trip's heaviest half-step; move them four gaps apart in the same step (`spaced`), both into step 1 of the trip's first or fifth stage (three B DMAs, no refill; `light0` / `light4`), or one into each (`split_light`); `split_late`: one in step 0 of the second and of the sixth stage, after the B scale pair load there, so the pair's in-order wait no longer covers it | `split_light` a gain with both formats and Gaussian data, confirmed on a second board; the others within noise or slower: two A scale DMAs in one light half-step still back up the queue the next heavy half-step inherits; `split_late` slower, most on Gaussian data: it puts an A scale DMA back into the half-step whose backlog `split_light` had removed | `split_light` |
| `aux_place` | `default`, `light`, `pre_z` | with `b_scale_fetch="pair"`: the second B scale pair load of an iteration a stage and a half earlier, in step 1 of its first stage instead of step 0 of its fourth (`light`); `pre_z`: the first pair load of an iteration in step 1 of its first stage, ahead of `split_light`'s A scale DMA, so the pair's in-order wait no longer covers that DMA | `light` within noise or slower, with uniform and Gaussian data; the load was cheap where it was. `pre_z`: the wait shrinks as intended, but the light half-step then issues five VMEM ops and the backlog moves to the next heavy one; slower on uniform data; on Gaussian data a gain on one board that a second board did not repeat | reject, kept as a knob |
| `clamp_hoist` | off, on | the clamped next-stage indices of a loop body computed once at its start instead of at every stage's start | within noise | reject, kept as a knob |
| `z_addr` | `inline`, `hoist`, `split` | with `lds2` scales: the A scale DMA's scalar address math one MFMA gap ahead of the DMA (`hoist`), or spread over the two gaps before it (`split`), out of the gap that issues it | within noise on uniform data; `split` slower on Gaussian data. Like `clamp_hoist`, a scalar trim below the climb's resolution | reject, kept as a knob |
| `trip_salu` | `default`, `hoist` | the loop body's last end-of-stage barrier group moved to the head of the next trip, below the trip's index math and its first stage's clamped indices, so those issue under the last MFMAs | fewer scalar instructions after the barrier; within noise on uniform data, slower on Gaussian data | reject, kept as a knob |
| `pre_e_frag` | `off`, `g3`, `g5`, `g9` | with `dma_place`: the next stage's first step-1 A fragment read in the current stage's step 1, n MFMA gaps ahead of the end barrier's LDS wait, into registers of its own, instead of right after that barrier into registers the last MFMAs still read | more registers; `g5` a gain on Gaussian data and slower on uniform data, `g3` slower, `g9` the reverse. Its extra prologue code also moves the K loop; on the peeled incumbent it is a clear gain together with `loop_pad` | `g5` with `loop_pad` and `k_peel` (Gaussian metric); not confirmed on a second board |
| `loop_pad` | 0 to 31 | code-placement probe: n four-byte `s_setprio 0` (the priority there is already 0) right before the line ring's K loop, so the loop starts 4n bytes later and nothing else changes | the same loop code runs measurably slower with its head on a 64-byte boundary, which is what `epi_store="soff"` lost; other offsets level. On the peeled incumbent, one pad (with `pre_e_frag="g5"`) is a gain and undoing it is a loss; larger pads there gain on one board only. The 16x16x128 loop has its own fast offsets: one pad is a gain on one board and slightly ahead on another, its neighbours are level | 3 on the 16x16x128 path (Gaussian metric); otherwise 0 |
| `epilogue` | `default`, `sa`, `pk2`, `ct`, `ct4`, `lds`, `ct_lds` | fewer epilogue instructions: scalar address terms (`sa`), two slots rounded together (`pk2`), or the MFMA computing C^T so a lane holds consecutive columns, stored 8 bytes (`ct`) or, after a lane-half swap, 16 bytes (`ct4`) per lane; `lds` / `ct_lds`: after one barrier each wave stages its C rows in the A line ring (two passes, XOR-swizzled chunks) and stores whole rows 16 bytes per lane | fewer instructions, but the transposed stores walk rows and coalesce worse; `sa` / `pk2` add scalar work; the staged variants restore whole-row stores, but the default's stores already write whole lines, so the barrier and LDS round trip are a net cost; within noise or slower | reject, kept as a knob |
| `store_nt` | off, on | non-temporal C stores | slower | reject, kept as a knob |
| `epi_store` | `global`, `soff` | with the default epilogue: C stores through a buffer resource, the lane's offset in one register per column block and each row's in a scalar register, two elements per packed convert: about half the epilogue's instructions | the epilogue's time does not shrink (its stores bound it, not its instructions), and the shorter prologue moves the K loop's code, which costs more than the epilogue saves (see `loop_pad`) | reject, kept as a knob |
| `b_scale_fetch` | `stage`, `pair` | `kblock4` B scales of two stages in one four-dword load per lane, issued by the odd stage before them: half the B-scale instructions at the address path | a gain alone and on top of `cc_passes` | `pair` |
| `cc_passes` | `none`, `dead_lanes`, `m0_fill`, `peep_waits`, `peep_shifts`, `ds_rebase`, `all` | the direct compiler's optional passes as kernel attributes: no copies of the FP6 operand's unread padding dwords, M0 writes moved off the DMA's hazard slot, adjacent waits merged, shift chains folded, one 64 KiB LDS rebase per address | each alone a small gain; all together a gain with both formats | `all` (needs a compiler build with these passes) |
| `as_read` | `b32`, `b128x4` | `lds2` A scales read one 16-byte LDS read per four stages instead of one dword per stage | fewer LDS instructions and bank conflicts, but the conflicts were not on the critical path; within noise | reject, kept as a knob |
| `k_order` | `rot`, `t10` | lane half h takes K32 block 2h + s in step s: a lane's A bytes contiguous (two 16-byte reads in step 0), A scales one 16-bit LDS read with no rotation | correct; slower with the plain compiler: the 16-bit scale's widening and the 16-byte read's unused upper dwords (which the register allocator reuses while the load is in flight) add LDS waits in every step. With `cc_extra`'s `lds_waw` and `u16_zext` those waits and the widening go: a small gain with both formats; with `b_layout="lanes"` and `as_read="b128x4"` slower | `t10` with `cc_extra="lds_waw+u16_zext"` |
| `cc_extra` | `none`, names from `lds_waw`, `u16_zext`, `acc_init_keep`, `pin_vmem`, `m0_cse` joined with `+`, `all` | more optional direct-compiler passes as kernel attributes: no LDS wait before an LDS load that overwrites an in-flight one's registers, no masking after a 16-bit LDS read, the loop's initial accumulator copies kept at the zeros, every VMEM op kept in its MFMA slot, repeated M0 writes dropped | `lds_waw+u16_zext` is what makes `t10` pay; `pin_vmem` within noise (the listing then obeys the builder's slot map) | `lds_waw+u16_zext` with `t10` |
| `kernarg_preload` | off, on | direct backend: the kernel arguments preloaded into scalar registers (kernel attribute; needs a compiler build that knows it) instead of loaded at the start | removes one early argument wait, but the preload entry block moves the K loop's code; slower | reject, kept as a knob |
| `acc_init` | `default`, `early` | line ring: zero the accumulators before the prologue's wait (with `acc_init_keep`), so the AGPR writes overlap the prologue's DMA | the writes move as intended; within noise, `prologue_wait="stage0"` had already hidden them | `default`, kept as a knob |
| `b_release` | `full`, `half` | B in two half-stage releases with an end-of-stage barrier: more lead per half | the B wait goes, the stall moves; a gain only with `a_scale_fetch="iter2"` (Findings 22, 23). Valid only when the thread count divides a B half's 16-byte chunks (`block_n * 3`): with 8 waves the per-thread pass count was truncated and the result was wrong, so the spec check rejects that case | `half` with `iter2` |
| `backend` | `llvm`, `direct` | the direct compiler schedules and allocates the visible MFMAs | with split intrinsic blocks, faster than the LLVM asm path (Finding 14) | `direct` for the canonical-A contract |
| `a_ring` | `stage`, `line`, `line2r` | row-major A in whole 128-byte lines, so no DMA splits a line; `line2r`: two line slots and an in-LDS remnant copy free room for a third B slot and a single barrier per stage | large gain on `direct`; correct but slow on LLVM (Finding 15); `line2r` correct and leaner per stage, but its one-stage refill lead and the copy's LDS writes make it slower or level (Finding 29) | `line` for the canonical-A contract on `direct` |
| `fill_split` | `none`, `line1` | split MFMA blocks: A's DMA (the line refill) and the A scale group into step 1, B's DMA and the reads stay in step 0 | DMA stalls and step-0 overflow mostly gone; gain with random and zero data (Finding 16) | `line1` |
| `scale_opsel` | off, on | MFMA scale byte select instead of a VALU shift per operand, half-1 scale dwords rotated by 16 bits | no shifts left, one permute per scale dword; gain on top of `fill_split` (Finding 17) | on (`direct`, asm); rejected for LLVM intrinsic MFMAs |
| `k_peel` | off, on | clamp-free steady-state loop, last iteration peeled | clamps leave the loop and a stage gets slightly shorter, but the time gain stays below the threshold under the power limit; the first version broke the barrier waits through dead loads (Finding 19). On the line ring's step pipeline the loop's stage is measurably shorter, but the peeled stages are several KB of straight-line code that run cold at the kernel's end: slower on uniform data. With `peel_dma="skip"` a gain on Gaussian data on two boards, and the loop gets shorter | on with `peel_dma="skip"` and `shape_args` off when Gaussian data is the metric; reject for uniform data |
| `peel_dma` | `clamp`, `skip` | with static stage indices (the peeled iteration, the tail): no DMA for a stage past K instead of a clamped re-load of the last one | with `k_peel` a gain on Gaussian data on two boards; the pair stays slower than the best configuration on uniform data | `skip` with `k_peel` (Gaussian metric) |
| `shape_args` | on, off | drop the unread `M`, `N`, `K` kernel parameters: each spec compiles its own kernel with the shape as constants | identical machine code, a smaller argument segment; the launch packs the kernel's own parameter list. On the 16x16x128 path the code differs and keeping the parameters is clearly slower on Gaussian data | off with `k_peel`; off on the 16x16x128 path |
| `mfma_atom` | `32x32x64`, `16x16x128` | the 16x16x128 scaled MFMA with its own K loop (trips of eight stages, K a power-of-two multiple of eight stages), `sched16` and `epi16` | with `sched16="fence+pairlight+midlgkm"`, `epi16="swap"` and its own `loop_pad`, faster than the best 32x32x64 configuration on Gaussian E2M3 on two boards, E3M2 level; dropping any schedule token or the swap epilogue is a loss | `16x16x128` (Gaussian metric); the other port knobs at their defaults |
| `peel16` | `none`, `skip` | 16x16x128 path: the K loop's last trip runs outside the loop without the next trip's DMAs, pair load and reads (the loop's last trip copies stages past K from wrapped sources into slots nothing reads) | the saved work is one trip's look-ahead and the end of K is not DMA-bound; the loop's scalar schedule changes, and what remains is the size of code placement, with signs that differ between boards | `none`, kept as a knob |
| `a_scale_path` | `direct`, `lds2` | A scales staged through LDS once per workgroup, eight stages per DMA: half the A-scale line requests | fewer L1 accesses and a shorter stage as modelled; gain with random and zero data, accepted by the tuner (Finding 25) | `lds2` |
| `b_path` | `lds`, `vgpr` | 1 x 4 waves, B straight into registers: no B DMA, B reads or B barrier | memory side as predicted, but the register loads' issue time and the shared A reads make the stage longer; faster only with zero data (Finding 26) | reject, kept as a knob |
| `dma_m0` | `default`, `early`, `bshare`, `early_bshare` | direct compiler moves each DMA's `m0` write an MFMA earlier, so no hazard `s_nop`; `bshare` (half B release): each wave fills consecutive 1 KiB regions of a B half, so its DMAs of the half share one LDS base and voffset with the step in the LDS-DMA instruction offset, and `m0_cse` drops the repeated M0 writes | `early`: the `s_nop`s go, the stage barely changes; within noise (Finding 27). `bshare`: a third of the M0 writes go, within noise (they were in MFMA shadow). With Gaussian data, `early` is a gain alone on the pipelined line ring and on its peeled version, but adds nothing on top of `pre_e_frag` with `loop_pad` | `default`, kept as a knob |
| `waves_m`, `waves_n` | 2 x 2, 1 x 4, 2 x 4, 4 x 2 | more waves per SIMD hide one wave's waits behind another's MFMAs; 1 x 4 lets B bypass LDS | 8 waves: more LDS reads, a shared memory queue and wider barriers; slower on both contracts (Finding 11) and on the line ring; 1 x 4 only with `b_path="vgpr"` (Finding 26) | 2 x 2 |
| `asm_tail_nop` | wait states after an asm MFMA block | covers the hazard LLVM cannot see into the block | needed for correctness with `asm_agpr` | default; not tuned |
| `ablate`, `dma_probe` | diagnostic | split the stage time; separate fragmentation from L2 misses | see Findings 6 to 9 | diagnostic only |
