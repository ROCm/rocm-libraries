# RMSEpilogue overhead investigation: bf16 MT512x128 on gfx950 (MI355X)

Pinned config throughout: bf16, MI16x16x32, MIWaveTile[16,4], MIWaveGroup[2,2] (MT512x128),
StreamK=3, StreamKForceDPOnly=1, GlobalSplitU=1, ScheduleIterAlg=3, PrefetchGlobalRead=2,
DirectToLds=1, UseSubtileImpl=True, DepthU=64 — the winning tile pinned in
`epilogues/YAMLs/redesign_baseline_bf16.yaml`, benchmarked at two problem sizes with equal
FLOPs but different output surface: 2048×16384×1×2048 and 4096×4096×1×4096.

Goal: understand why `SubtileMegaFusedEmit.py`'s RMSEpilogue path is slower than
`GlobalWriteBatch.py`'s plain beta*C path for the identical tile/schedule, find concrete
divergences, fix what's fixable, and honestly size what isn't.

## Executive summary

- The RMSEpilogue kernel is **HBM-write-bandwidth-bound**, not compute-bound and not
  barrier-bound. It writes 2.39× more bytes than the beta*C baseline (D + ResidualOut +
  partialBuf vs. D-only), at an operating point (1 WG/CU, 4 waves/CU) that is the
  *deliberately chosen, optimal* occupancy for this tile — so there is no spare-wave slack
  to hide the extra traffic.
- One real, validated fix landed: hoisting the barrier-free intra-wave RMS-reduction
  butterfly ahead of the store drain (commit `8cef26d13bb`), **+5.0% / +2.3%** on the two
  pinned shapes.
- Two further schedule-level attempts (a full-reduction hoist, and deferring the cross-wave
  reduction past the D-store) were tried and correctly abandoned: the first broke
  correctness, the second was proven correct but measured perf-neutral within noise.
- A hardware-counter profiling pass (rocprofv3) confirmed the remaining gap is dominated by
  raw write-byte volume, not by VALU, coalescing, or barrier stalls, and a subsequent
  candidate-ranking pass concluded that **no further in-kernel code change clears a
  reasonable risk/reward bar** — the epilogue is at the hardware limit for its required
  functional output volume. Further gains would require an API/pipeline-level change
  (e.g. fusing the downstream RMS-finalization kernel so `partialBuf` never round-trips
  HBM), which is outside this emitter's scope.

## Methodology

Three kernel variants were built from the same pinned solution, isolating exactly one
functional layer at a time:

1. **No epilogue at all** (`UseBeta:False`, `UseRMSEpilogue:False`) — pure GEMM, `D =
   alpha*acc`, no C load, no epilogue math whatsoever.
2. **Beta*C only** ("GWB baseline") — `UseBeta:True`, `UseRMSEpilogue:False`,
   `RMSEpilogue:[False]`. Exercises `GlobalWriteBatch.py`'s plain beta-combine path with
   every other solution knob (StreamK, tile, prefetch) held identical.
3. **RMSEpilogue** — the pinned YAML as shipped (`UseRMSEpilogue:True`,
   `RMSEpilogue:[True]`), routing through `SubtileMegaFusedEmit.py`.

All three share the exact StreamK/tile/prefetch configuration, so any divergence is
attributable to the epilogue layer, not to schedule differences. Builds were produced via
`./Tensile/bin/Tensile <yaml> <outdir>` (never copied into the shipped `Logic/` directory —
these are research-only scratch builds). Correctness was gated throughout on
`bash epilogues/scripts/correctness_harness.sh`, always checked by grepping the log for
`PASSED`, never by trusting the exit code (this project's established convention).

Two investigation modes were used, in sequence:

- **Assembly-level comparison**: diffing the generated `.s` for the RMSEpilogue kernel
  against the beta*C-only kernel, instruction by instruction, looking for scheduling
  divergences, extra waits, branch/instruction-selection differences.
- **Hardware-counter profiling** (`rocprofv3`, MI355X/gfx950): VALU utilization, memory
  bandwidth/cache behavior, stall categories, occupancy — to check what the assembly-level
  view could not see (actual cycles, actual bandwidth, actual stalls).

## Part 1 — Assembly-level findings

### Confirmed divergences in the "should be equivalent" load+add+store portion

1. **Add timing: monolithic vs. interleaved.** GWB defers the C-load and beta-combine into
   the store batch itself, interleaved with per-element store-address setup
   (`GlobalWriteBatch.py`, `_addSumAlphaWithCBeta` ~4364-4374). The RMS path (pre-fix)
   computed the *entire* residual-add + RMS-reduction as one monolithic block, fully
   completing ~25,000 assembly lines before ever reaching the shared store-dispatch label —
   zero interleaving between "add" and "store-address setup".
2. **Load strategy: dwordx2 (GWB) vs. dwordx4 + `ds_bpermute` (RMS).** GWB issues
   `buffer_load_dwordx2` per lane, no cross-lane movement. RMS pairs lanes to issue wider
   `buffer_load_dwordx4` then redistributes via `ds_bpermute_b32`
   (`_issueResidualDwordx4`/`_redistributeResidualLoad`). **Verified via a separate
   experiment that dwordx4 is the higher-throughput approach** — GWB's narrower load is the
   one that's behind, not the reverse.
3. **Combine instruction: scalar (GWB) vs. packed (RMS).** GWB does scalar
   `v_cvt_f32_bf16` + `v_fmac_f32` per element. RMS uses 2-wide `v_pk_add_f32`. Not fully
   apples-to-apples (beta needs a runtime scalar multiply; plain residual-add doesn't), but
   RMS's packed add is already the more efficient instruction.
4. **Branch trampoline unique to GWB.** Beta is a runtime scalar in GWB, so it compiles both
   beta==0/!=0 bodies and reaches the hot path via an `s_getpc_b64`/compare/`s_setpc_b64`
   long-branch trampoline — overhead inherent to GWB's generic-beta contract, not portable
   to (and not needed by) a compile-time-flagged epilogue.

### Confirmed NOT divergent

- **Waitcnt discipline** is equally fine-grained on both sides — progressive `vmcnt(N)`
  draining interleaved with work, no conservative wait-all instructions found in either.
- **The final D-store routine is structurally identical** — a normalized diff (register
  numbers and label suffixes stripped) of the shared store-dispatch region showed zero
  instruction-sequence differences between the two kernels.

### Expected-different, out of scope

Gamma LDS broadcast staging and the row-wise RMS sum-of-squares reduction/cross-lane tree
reduction have no GWB analog by design — these are RMSEpilogue-specific functionality, not
overhead to eliminate.

## Part 2 — Landed fix: intra-wave reduction hoist (commit `8cef26d13bb`)

The RMS reduction has two phases: an **intra-wave** `ds_bpermute` butterfly (no LDS memory,
no barrier — safe to run concurrently with in-flight stores) and a **cross-wave** phase
(real LDS store/load + `s_barrier`, only present when `MIWaveGroup`'s M-extent — `wg_m` —
exceeds 1). The original code ran both phases *after* the teardown's `vscnt=0` store drain.

The fix splits `_reduceFree0` into `_reduceRowGroupFree0` (intra-wave) and
`_reduceCrossWaveFree0` (cross-wave), and reorders `emit()` so the intra-wave butterfly runs
**before** the drain (overlapping in-flight ResidualOut/MXScale stores), while the
cross-wave phase stays fenced **after** the drain, unchanged.

- Correctness: harness PASSED (bf16 + mxfp8), 0/0 VGPR/SGPR spills before and after.
- Perf (pinned config, fresh baseline same session):
  - 2048×16384×1×2048: 173.66 → 165.45 µs (**+5.0%**)
  - 4096×4096×1×4096: 141.55 → 138.38 µs (**+2.3%**)

This is the only code change landed as a result of the whole investigation.

## Part 3 — Two attempts that were correctly abandoned

### Full-reduction hoist (broke correctness)

Hoisting the *entire* reduction (including the cross-wave phase) above the store drain
seemed like a natural extension of the fix above, but **failed the correctness harness** —
wrong values in `d` and `residualOut` on 521×521 boundary tiles, specifically on every
`wg_m ≥ 2` config (the pinned target's `MIWaveGroup[2,2]` is `wg_m=2`).

Root cause, verified against source: `_crossWaveReduceFree0` opens with a bare `s_barrier`
for WAR ordering on shared LDS scratch. On gfx950 with `PrefetchGlobalRead ≥ 2`, `s_barrier`
carries no implicit vmem drain. Hoisting the barrier above the drain moves it into a window
where ResidualOut vmem stores (which themselves touch LDS via `ds_bpermute` in wide-store
paths) are still in flight — corrupting the boundary tile. **The drain sitting before the
cross-wave barrier is load-bearing for `wg_m ≥ 2`, which is the pinned target's
configuration.** Reverted; nothing landed from this attempt.

### Deferring the reduction past the D-store dispatch (correct, but perf-neutral)

A sharper reframing: since the D-store value (`H*gamma`) doesn't depend on the RMS
reduction at all (the reduction only feeds a separate `partialBuf` for a downstream
finalization kernel), could the reduction be deferred to run **after** the D-store dispatch
instead of hoisted before the store drain — sidestepping the WAR hazard above by construction
(the D-store's own lane-pairing uses `ds_bpermute` + `v_permlane32_swap_b32`, which touches
zero LDS memory and cannot alias the reduction's LDS scratch)?

A full CFG/barrier-convergence audit was required first, since `s_barrier` is a workgroup
*control* barrier (deadlocks if any wave skips it), unlike a memory fence. The audit
**cleared**: RMSEpilogue's own validation constraints (`_validateRMSEpilogue` rejects
`AdaptiveGemmGSUA`, `MultipleBufferSingleKernel`, `MultipleBuffer` GSU, `GroupedGemm`, and
mandates `StreamKForceDPOnly=1`) collapse `globalWriteElements`'s `globalWriteModes` to a
single non-divergent `GSU1` path for any RMSEpilogue kernel — so every wave that runs the
epilogue body provably reaches the proposed deferred-reduction site. One additional hazard
was found and fixed during the audit: the join label (`GW_End`/`endLabel`) is recreated per
`gsuLimitIdx` iteration, so the deferred call had to be gated to the same iteration as the
body, or a workgroup that never ran the body could execute the barrier and corrupt
`partialBuf`.

Implementation: split `SubtileMegaFusedEmitter.emit()` into `emitBody()` (called at the
existing site, before the D-store dispatch) and `emitReduce()` (called after the D-store
dispatch, before any `KernelEnd` branch). Result:

- Correctness harness PASSED under an explicit timeout (a barrier bug would hang, not fail
  cleanly — this was treated as the binding safety rule for this experiment).
- 0 spills, VGPR 480/512 (gfx950 combined arch+acc ceiling).
- Perf, 5 runs per config: **flat within run-to-run noise** (±3-5 µs) on both shapes — no
  measurable gain on top of the already-landed intra-wave hoist.

Reverted (working tree only, never committed) since it adds real control-flow complexity
(two emission phases, a new iteration guard, a stashed-emitter handoff) for no measured
benefit.

A third option — merging the D-store's own address computation with the per-tile combine,
GWB-style — was assessed and **not attempted**: the store-order (paired `dwordx4`, two
accumulator groups per instruction) doesn't match the epilogue's per-tile production order,
the store address is already latency-hidden under the existing `ds_bpermute` wait window, and
the seam is shared code with every non-fused kernel in the library — high blast radius for a
small expected win (eliminating one `ValuC` register round-trip).

## Part 4 — Hardware-counter profiling (rocprofv3)

### Methodology and caveats

MI355X/gfx950, both kernels rebuilt fresh on the post-`8cef26d13bb` HEAD.
`rocprofv3`'s named derived metrics (VALUUtilization, GPUBusy, …) failed to register on this
build, so raw basic counters were collected and the derived formulas recomputed directly
from the ROCm SDK's own counter-definition XML. Multi-pass collection was split across 10
hardware-limited counter groups. Block-level ratios are reported as directly-comparable
per-active-cycle ratios (RMS vs. GWB), not calibrated absolute percentages, since
rocprofv3's per-instance summing didn't match the vendor formula's assumptions on this
build. **Clean wall-clock timing (profiler detached) is the authoritative magnitude for the
overall gap; the counter run's own cycle count is inflated by per-dispatch profiling
overhead** and is used only to see *where* the cycles go, not the absolute *how much*.

### Findings

1. **Extra HBM write traffic dominates (2.39×, +88.8 MB on 2048×16384).** GWB writes only D
   (64 MB). RMS writes D + ResidualOut + partialBuf (64 + 64 + ~25 MB ≈ 153 MB), matching
   measured bytes almost exactly. Reads are ~equal between kernels (both stream one extra
   M×N input — residual vs. C). L2 hit rate is identical (78%) — this is not a
   cache-behavior difference, it's genuine extra traffic.
2. **Stores are under-vectorized by instruction count, not by width** — see the coalescing
   deep-dive below; this is refined further, not a separate independent finding.
3. **Occupancy is LDS-capped at 4 waves/CU for both kernels, by design.** Both pin 160 KiB
   LDS/CU → 1 WG/CU. This is **the deliberately chosen, optimal operating point** for this
   winning tile (software producer/consumer specialization isn't viable on this hardware for
   this shape) — not a defect and not an available lever. The practical implication: there
   is no spare-wave slack to hide *any* extra memory traffic the epilogue adds, so every
   extra byte lands directly on the critical path. This is why the remaining gap must be
   attacked at the epilogue's own memory footprint, not at occupancy.
4. **Extra VALU math is real but negligible.** RMS issues 1.31× more VALU instructions
   (residual-add, RMS-square, gamma-scale, bf16-pack), yet VALU busy is only ~2.4% in *both*
   kernels, with no divergence (VALU-util 99.7%/99.8%). Compute is not the bottleneck. This
   **refines** the conclusion of the older `epilogue_overhead_analysis.md` (a different,
   fp8/MXFP8-dynquant shape, where the conversion/quant chain genuinely did dominate) — for
   this bf16 shape the quant chain is absent and memory writes dominate instead.
5. **The cross-wave barrier/LDS reduction is NOT the bottleneck.** Despite 1.21× more LDS
   instructions (gamma broadcast + reduction) and 1.28× more LDS bank conflicts, RMS's
   lgkmcnt/barrier wait is actually *lower* than GWB's (0.87×). This empirically explains why
   the reduction-deferral experiment (Part 3) measured perf-neutral: at the hardware level,
   the barrier was never the dominant remaining cost.

## Part 5 — Coalescing deep-dive

A natural follow-up question: is the "under-vectorized stores" finding (3.13× store
instructions for only 2.39× bytes) actually a **coalescing** problem (scattered per-lane
addresses forcing extra HW transactions) rather than an instruction-width problem?

**Ruled out as fragmentation.** Both kernels achieve **100% 64-byte HBM requests**, for reads
and writes alike — perfectly coalesced, zero sub-line fragmentation. RMS's write-*request*
count (2.387×) tracks its write-*byte* count (2.387×) essentially one-to-one; a fragmentation
problem would show a falling avg-bytes/request or a rising 32-byte-partial fraction, and
neither appears.

**The real mechanism: half-payload wide stores, intrinsic to ResidualOut's data granularity.**
Measured RMS store instructions each commit ~782 bytes vs. GWB's 1024 bytes/instruction.
Root cause, verified in code: for this bf16 (non-MXFP8) config, `_useDwordx4Interior()`
(`SubtileMegaFusedEmit.py` ~line 337) is `True`, so ResidualOut's interior path uses
`_storeResidualOutRowDwordx4` (~lines 926-967): lanes are paired via `ds_bpermute` so the
lower lane of each pair gathers its partner's packed dwords and issues **one**
`buffer_store_dwordx4` (B128), while **the upper lane's own store address is clamped to
`BufferOOB` and dropped**. ResidualOut's per-lane payload (4 bf16 = 8 B) is half of D's (8
bf16 = 16 B), so lane-pairing packs two lanes into one B128 — but only one lane's worth of
"real" data actually lands; the other lane's slot in the same instruction is wasted relative
to D's all-64-lanes store. `partialBuf` and the shared byte-store edge/straddle path were
checked and ruled out as contributors (identical or negligible between the two kernels).

This is a real, precisely diagnosed mechanism — but see Part 7 for why fixing it doesn't
move the needle.

## Part 6 — Three-way timing and the relationship to shape

Isolating "any epilogue at all" (the beta*C step) from "RMS specifically", against a true
pure-GEMM baseline, clean wall-clock (profiler detached, several reps; noise floor ≈ ±3 µs):

| Shape | no-epilogue | beta-only (GWB) | RMS-epilogue | noep→beta | beta→RMS | **noep→RMS (total)** |
|---|--:|--:|--:|--:|--:|--:|
| 2048×16384×1×2048 | 135.2 µs | 150.4 µs | 167.0 µs | +11.3% | +11.0% | **+23.5%** |
| 4096×4096×1×4096  | 129.7 µs | 134.5 µs | 143.3 µs |  +3.7% |  +6.5% | **+10.5%** |

**Relation to shape: epilogue overhead scales with output surface (M×N), not FLOPs.** The
two shapes are FLOP-equal (hence pinned to the same winning tile), but 2048×16384 has 2× the
output elements of 4096×4096. It pays roughly 2× the total epilogue overhead (23.5% vs.
10.5%) — direct confirmation that the cost is bound by *bytes written per output element*,
not by compute. On the write-heavy shape, roughly half the total epilogue cost is simply
"touching C/D at all" (beta*C, +11.3%); RMSEpilogue adds a comparable further amount on top
(+11.0%) for its own ResidualOut/partialBuf writes and reduction. On the smaller-surface
shape, both effects shrink roughly proportionally (+3.7% / +6.5%).

Sizing the RMS-specific increment (2048×16384, +16.6 µs) against write volume: ResidualOut
(64 MB) accounts for ≈ 64⁄89 × 16.6 ≈ **11.9 µs**; partialBuf (25 MB) accounts for
≈ 25⁄89 × 16.6 ≈ **4.7 µs**. The implied effective bandwidth (~5.4 TB/s, accounting for the
78% L2 hit rate) is near the practical HBM3e ceiling — i.e. this is genuinely
bandwidth-limited, not issue-limited.

## Part 7 — Final candidate ranking: nothing clears the bar

Given the write-bound diagnosis, four further candidates were ranked against the sizing
above before any implementation was attempted. **None cleared a reasonable risk/reward bar;
no further code was changed.**

1. **Overlap epilogue stores with the K-loop tail — infeasible, not merely hard.** Every
   epilogue write derives from `H = acc + residual`, and `acc` is a reduction over the full
   K dimension — no output element is final before the last MFMA. `PrefetchGlobalRead=2`
   only double-buffers A/B *operand loads*, it does not retire any partial accumulator
   early. `StreamKForceDPOnly=1` means each tile is computed whole (no split-K partials to
   exploit). Confirmed against `KernelWriter.py`'s `endSummation` ordering and
   `LogicalScheduler.py`'s NLL/tail-retirement sequencing — there is no data-ready-early
   opportunity to exploit.
2. **Fix ResidualOut's half-payload store — rejected, sub-noise-floor reward.** The fix only
   reduces store-*instruction* count/issue overhead, not byte volume (already optimal,
   100% coalesced) — and the kernel is bandwidth-bound, not issue-bound (VALU 2.4%). Upper
   bound: ~5-10% of ResidualOut's 11.9 µs ≈ **0.6-1.2 µs, inside the ±3 µs noise floor**. The
   structural change required (batching two M-tiles per lane, touching lane-pairing/OOB
   masking) is correctness-sensitive for a saving indistinguishable from noise.
3. **Shrink/defer/overlap `partialBuf` — rejected.** Even *total elimination* caps at 4.7 µs
   (~28% of the 16.6 µs RMS-specific increment from Part 6), and elimination is impossible: `partialBuf`'s f32
   width is load-bearing for the downstream RMS-finalization kernel's `rstd` computation
   (narrowing it corrupts the reduction), and `PARTIALRMS_STREAMK_PARTIALBUF_BUG.md`
   documents this path as already correctness-fragile under StreamK splitting — not a good
   place to add risk. The write is already emitted as late/overlapped as the dependency
   chain (full K-reduction, then cross-wave reduction) permits.
4. **Anything else — out of scope or already maximal.** D's store lives in the shared
   `GlobalWriteBatch`/`globalWriteElements` path used by every non-fused kernel in the
   library (same seam flagged in Part 3 as high-blast-radius); overlapping it with
   ResidualOut wouldn't reduce total bytes in any case (both contend for the same HBM). Store
   overlap *within* the epilogue is already maximal post-`8cef26d13bb` — there is no
   remaining per-tile store stall to remove.

## Conclusion

The RMSEpilogue kernel, at its pinned winning tile, is **bandwidth-bound at the hardware
limit** for its required functional output (D + ResidualOut + partialBuf = 152.8 MB on the
write-heavy shape), at a deliberately-chosen, optimal occupancy that leaves no slack to hide
extra traffic. One real fix was found and landed (`8cef26d13bb`, +5.0%/+2.3%); everything
else investigated — further reduction/barrier scheduling, store-instruction packetization,
partialBuf format/timing — was sized against real measurements and found to be either
infeasible, correctness-risky, or below the noise floor. Further improvement is not a
scheduling problem inside this emitter; it would require an API/pipeline-level decision
above it — most plausibly, fusing the downstream RMS-finalization kernel so `partialBuf`
never round-trips through HBM, or a calling convention that doesn't require the
`ResidualOut` output at all.

## Artifacts and references

- Landed commit: `8cef26d13bb` — `Tensile/Components/Subtile/SubtileMegaFusedEmit.py`
  (`_reduceRowGroupFree0`, `_reduceCrossWaveFree0`, `emit()` tail reorder).
- Key source locations (line numbers approximate at time of writing; re-verify before citing
  in future work — the file changes as the redesign continues):
  `SubtileMegaFusedEmit.py`: `_pass1AccResRms` (residual-add + Σx² accumulate),
  `_pass3GammaAmax` (gamma combine), `_writeAccFrom` (accumulator writeback),
  `_reduceAndWriteRms`/`_reduceRowGroupFree0`/`_reduceCrossWaveFree0` (RMS reduction),
  `_storeResidualOutRowDwordx4`/`_issueResidualOutWide` (ResidualOut store),
  `_writePartialsFree0` (partialBuf write), `_emitTeardown` (store drain),
  `_useDwordx4Interior` (store-width gating), `_emitGammaLdsSetup`/`_ldsReadGammaBlock`
  (gamma LDS staging — unrelated to this investigation).
  `GlobalWriteBatch.py`: `_addSumAlphaWithCBeta` (beta combine), `_emitSubtilePackedPermute`/
  `emitAddrWhilePermuting` (store address setup), `_emit16bitSubtilePairedStore` (shared
  D-store emitter).
  `KernelWriterAssembly.py`: `emitSubtileFusedEpilogue` (call site into the fused emitter),
  `globalWriteElements`/`generateBetaModules` (D-store dispatch).
- Related docs: `epilogues/docs/epilogue_overhead_analysis.md` (a different, fp8/MXFP8
  shape — methodology reference only, numbers don't transfer),
  `epilogues/docs/PARTIALRMS_STREAMK_PARTIALBUF_BUG.md` (partialBuf's StreamK-split
  correctness constraints).
- Reference YAMLs: `epilogues/YAMLs/redesign_baseline_bf16.yaml` (RMSEpilogue config, pinned
  tile). Scratch YAMLs used for the beta-only and no-epilogue variants were built by flipping
  `UseBeta`/`UseRMSEpilogue`/`RMSEpilogue` on a copy of the above; recreate as needed, do not
  copy their output into the shipped `Logic/` directory.
