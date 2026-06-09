/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

/// Creates a pass that inserts cluster-barrier instructions at five
/// well-defined rules covering the main and tail loops. Rules are
/// numbered in **kernel-execution order** (the rule with the lowest
/// number is the first to fire when the kernel runs):
///
/// Rule 1 -- signal-only (no leading cluster wait) immediately AFTER
/// each `label_GSU_1:` label (Tensile's post-GSU==1-guard label),
/// wrapped in an outer `LoopCounterL != 0` gate so the cluster-barrier
/// signal only fires on non-zero iterations. A workgroup-scope
/// `s_barrier_signal -1` / `s_barrier_wait -1` pair sits INSIDE the
/// outer LCL skip region (and BEFORE the inner WaveIdx gate) so every
/// wave in the workgroup has reached the post-GSU==1 join before any
/// wave issues the cluster signal:
///     s_cmp_eq_u32 s[sgprLoopCounterL], 0
///     s_cbranch_scc1 label_skipCBPreSignal_LCL_<HASH_OUTER>
///     s_barrier_signal -1                                          // workgroup signal
///     s_barrier_wait -1                                            // workgroup wait
///     s_cmp_eq_u32 s[sgprWaveIdx], 0
///     s_cbranch_scc0 label_skipCBPreSignal_<HASH_INNER>
///     s_barrier_signal -3
///   label_skipCBPreSignal_<HASH_INNER>:
///   label_skipCBPreSignal_LCL_<HASH_OUTER>:
///
/// Rule 2 (kernel scope only) -- a single `s_barrier_wait -3` immediately
/// before the first `tensor_load_to_lds` of the whole kernel.
///
/// Rule 3 -- LoopCounterL-gated signal-only handshake at
/// the LDS publication point that precedes `label_openLoopL:`. Same
/// shape as Rule 1 but with the outer gate set to
/// `s_cmp_le_u32 s[sgprLoopCounterL], pgrValue` (the cbranch skips the
/// signal when there are too few iterations left, so the producer is not
/// needed). The gate mirrors Tensile's own `s_cmp_le_u32 LCL, pgrValue /
/// s_cbranch_scc1 LoopEndL` loop-entry guard, so the cluster signal is
/// suppressed on the exact same control-flow paths where the corresponding
/// `s_barrier_wait -3` inside the unrolled loop body is skipped -- keeping
/// `signal -3` / `wait -3` paired everywhere.
///
/// Two anchor modes (backward scan from `label_openLoopL:`):
///   (a) `s_barrier_wait -1` -- publication point already exists (typical
///       for PLR > 0 schedules). Anchor at the successor of that wait.
///       No new workgroup sync is synthesized. The scan stops as soon
///       as a `tensor_load_to_lds` is reached: that instruction marks
///       the prefetch section, before any LW-to-PLR sync could sit, so
///       an earlier workgroup wait would be unrelated. Defers to Rule
///       4 if the same wait would also be a Rule-4 trigger.
///   (b) No `s_barrier_wait -1` between the prefetch tail and
///       `label_openLoopL:` (typical for PLR == 0 schedules where the
///       prologue has no local-read preamble barrier). Only active
///       when `plrValue == 0`; anchor at the label and synthesize an
///       `s_barrier_signal -1` / `s_barrier_wait -1` pair INSIDE the
///       LCL skip region (between the outer LCL skip-branch and the
///       inner WaveIdx gate) so the workgroup sync sits on the same
///       control-flow path as the cluster signal -- both are bypassed
///       together on the `LCL <= pgrValue` skip path (matching
///       Tensile's loop-entry guard: when the unrolled loop body is
///       skipped, the LDS reads inside it are skipped too, so no LDS
///       publication is needed). Emitted shape immediately BEFORE the
///       `label_openLoopL:` label:
///
///           s_cmp_le_u32 s[sgprLoopCounterL], <pgrValue>          // outer LCL gate
///           s_cbranch_scc1 label_skipCBPreSignal_LCL_<HASH_OUTER> // skip when LCL <= pgr
///           s_barrier_signal -1                                  // workgroup signal
///           s_barrier_wait -1                                    // LW to PLR, sync LDS0
///           s_cmp_eq_u32 s[sgprWaveIdx], 0                        // inner wave gate
///           s_cbranch_scc0 label_skipCBPreSignal_<HASH_INNER>
///           s_barrier_signal -3
///         label_skipCBPreSignal_<HASH_INNER>:
///         label_skipCBPreSignal_LCL_<HASH_OUTER>:
///
/// Internal control-flow labels inside the prefetch prologue (e.g.
/// `label_skipPGR2_*`) do not match `label_openLoopL` by exact-name
/// comparison and are walked through. Section-level idempotency: the
/// backward scan also flags whether a cluster-scope signal/wait
/// already sits in the section; if so (e.g. a prior pass run already
/// emitted Rule 3), Rule 3 self-disables. Unlike a TEXTBLOCK anchor,
/// the label/instruction-based scan survives
/// `ScopeAdaptor::moveIRToBlock`, so Rule 3 keeps working whenever
/// `Gfx1250Backend::buildGfx1250Pipeline` runs this pass at kernel scope
/// when `moduleOptions.ClusterBarrier == true`.
///
/// Rule 4 -- cluster wait + LoopCounterL-gated signal after each
/// workgroup-scope wait that precedes a `tensor_load_to_lds`. For
/// every load in a label-/branch-delimited segment we walk backward
/// to the nearest preceding `s_barrier_wait -1`; triggers are
/// deduplicated by identity so multiple loads sharing the same anchor
/// wait yield exactly one handshake. Two emission modes, selected by
/// `findLiveLoopCounterLCmpUpstream`:
///
///   (a) inherited-SCC mode -- when SIA (typically `ScheduleIterAlg=4`)
///       hoists the loop-exit `s_cmp_eq_{u32,i32} LCL, imm` above the
///       anchor, the outer cbranch reuses that SCC instead of emitting
///       a fresh gate cmp (which would clobber the SCC the downstream
///       cbranch still needs). The cluster WAIT is ungated and only the
///       wave-gated signal is suppressed (single-iter skip); a clone of
///       the upstream cmp is inserted between the inner and outer skip
///       labels to rebuild SCC for the downstream cbranch. Left
///       untouched by "Fix B".
///
///   (b) fresh-gate mode ("Fix B" drain-drop) -- when no such upstream
///       cmp is live, the whole cluster handshake is dropped on the
///       drain iterations where the paired `tensor_load_to_lds` is
///       disabled (`LCL <= pgrValue`). Because the ping-pong pairing is
///       offset (each wait consumes the PREVIOUS signal), the WAIT and
///       SIGNAL are gated with ASYMMETRIC thresholds: skip the WAIT at
///       `LCL <= pgrValue` and the SIGNAL one stage earlier at
///       `LCL <= pgrValue+1` (so the trailing leftover signal is dropped
///       too). This keeps cluster signal/wait balanced for every trip
///       count, and the `<= pgr` predicate matches Tensile's own
///       `s_cmp_le_i32 LCL, pgrValue` load-disable cmov.
///
///       LCL pre-decrement compensation: the two thresholds are
///       calibrated against the loop counter value at segment entry.
///       Some schedules hoist the per-iteration
///       `s_sub_{u32,i32} LCL, LCL, <imm>` ABOVE the anchor, so the
///       gate would otherwise read an already-decremented LCL and fire
///       one iteration too early. The pass therefore sums those
///       hoisted decrements (backward scan, bounded by the segment) and
///       subtracts the total from BOTH thresholds, so the gate keys off
///       the same absolute iteration regardless of where the decrement
///       landed. When no decrement precedes the anchor (the default
///       schedule) the sum is 0 and the thresholds are unchanged.
///
/// Shape (fresh-gate / Fix B mode shown; inherited-SCC mode keeps an
/// ungated leading wait, drops the signal gate cmp, and inserts a
/// `<clone of upstream LCL cmp>` line between the two skip labels):
///
///     s_cmp_le_i32 s[sgprLoopCounterL], <pgrValue>                  // WAIT gate (drain iter)
///     s_cbranch_scc1 label_skipCBWait_LCL_<HASH_WAIT>
///     s_barrier_wait -3                                            // cluster barrier wait
///   label_skipCBWait_LCL_<HASH_WAIT>:
///     s_cmp_le_i32 s[sgprLoopCounterL], <pgrValue+1>                // SIGNAL gate (one stage earlier)
///     s_cbranch_scc1 label_skipCBPreSignal_LCL_<HASH_OUTER>
///     s_cmp_eq_u32 s[sgprWaveIdx], 0                                // inner wave gate
///     s_cbranch_scc0 label_skipCBPreSignal_<HASH_INNER>
///     s_barrier_signal -3
///   label_skipCBPreSignal_<HASH_INNER>:
///   label_skipCBPreSignal_LCL_<HASH_OUTER>:
///
/// Rule 5 (kernel scope only) -- tail-loop cluster handshake (paired).
/// Anchors on the first `tensor_load_to_lds` that follows the
/// `/* Tail Loop */` TEXTBLOCK marker; the wait and signal are emitted
/// at two distinct sites because the load and the preceding workgroup
/// wait sit in different label/branch-delimited segments (the tail
/// TDM-reset block between them is not synchronization-critical, so
/// collapsing both into a single site would unnecessarily serialize
/// the cluster).
///   5a -- signal-only handshake (no LoopCounterL gate) immediately
///        AFTER the workgroup-scope wait (`s_barrier_wait -1`) that
///        precedes the tail load (searched backward from the load,
///        bounded by the `/* Tail Loop */` marker).
///        Defers to Rule 4 if that wait is already a Rule-4 trigger.
///   5b -- a single `s_barrier_wait -3` immediately BEFORE the tail
///        tensor_load itself.
/// Each half has its own idempotency check so re-runs are no-ops.
///
/// `<HASH>` is a fresh 16-character alphanumeric identifier per insertion.
/// Only the first wave (WaveIdx == 0) executes the signal; the other waves
/// fall through to the label.
///
/// Idempotency: each rule has its own skip check so re-running the pass is
/// a no-op when the handshake is already present.
///
/// \p isKernelScope must be true for the GFX1250 backend pipeline (whole-
/// kernel insertion). When false, the pass is intended for region-scoped
/// invocation via `createKernelToRegionsPassAdaptor` (not used by the
/// backend today). Rule 2 only fires when this is true because the "first
/// tensor_load of the whole kernel" anchor is meaningful only at kernel
/// scope.
///
/// \p pgrValue is Tensile's `PrefetchGlobalRead` setting. It controls the
/// outer LoopCounterL gates of Rules 3 and 4:
///   - Rule 3 skips the signal when `LoopCounterL <= pgrValue`.
///   - Rule 4 fresh-gate ("Fix B") mode skips the WAIT when
///     `LoopCounterL <= pgrValue` and the SIGNAL when
///     `LoopCounterL <= pgrValue + 1`; inherited-SCC mode keeps an
///     ungated wait and reuses an upstream `LoopCounterL == imm` compare
///     for a single-iter signal skip.
/// The default of 1 matches PGR=1 (`<= 1` for Rule 3; `<= 1` wait / `<= 2`
/// signal for Rule 4 fresh-gate mode).
///
/// \p plrValue is Tensile's `PrefetchLocalRead` setting. It enables Rule
/// 3's anchor mode (b): when `plrValue == 0` and the backward scan from
/// `label_openLoopL:` finds no `s_barrier_wait -1` before reaching the
/// prefetch boundary (`tensor_load_to_lds`), the rule synthesizes the
/// missing publication point (workgroup `s_barrier_signal -1` /
/// `s_barrier_wait -1`) followed by the same `LCL <= pgrValue` gated
/// cluster signal as anchor mode (a). Any non-zero value disables mode
/// (b) (default).
///
/// This pass mutates the CFG (new branches and a new label), so dependent
/// CFG / dominance analyses are invalidated.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createInsertClusterBarrierPass(bool isKernelScope = true,
                                                                       int pgrValue = 1,
                                                                       int plrValue = 1);

}  // namespace stinkytofu
