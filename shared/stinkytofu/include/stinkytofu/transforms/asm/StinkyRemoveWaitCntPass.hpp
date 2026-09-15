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

/// Policy knobs for StinkyRemoveWaitCntPass, applied only to removals that are
/// already legal. Legality is decided first and is not configurable: a wait is
/// only a candidate if some pass regenerates it, per
/// `waitcnt::waitReconstruction()`. Anything naming STOREcnt therefore survives
/// with no knob at all, and needs none until a store counter exists.
///
/// These fields only narrow the candidates further. This struct is the single
/// place documenting *why* each exception exists; call sites should point here
/// rather than restate it.
struct RemoveWaitCntOptions {
    /// Also strip `s_wait_tensorcnt`. On by default so the insertion pass owns
    /// every wait; turn it off to let that pass reuse the incoming ones.
    ///
    /// `s_wait_tensorcnt` carries `IF_WaitTensorCnt`, a flag disjoint from
    /// `IF_WaitCnt`, so it needs its own check -- `isWaitCnt()` misses it.
    bool removeTensor = true;

    /// Also strip `s_wait_xcnt`. This is the one opcode whose regenerator is
    /// `Gfx1250HazardPass` rather than the wait-count dataflow, so removing it is
    /// legal only on that pass's promise. Off by default because that pass
    /// rebuilds from its own XNACK replay-group rules (atomic-after-memory, FLAT
    /// and SMEM source overlap), not from where the drains originally sat:
    /// TensileLite emits `s_wait_xcnt 0` ahead of any volatile/atomic VMEM op --
    /// in StreamK the release-side flag store, the acquire-side flag load, and
    /// the work-queue atomic, the first two of which are plain volatile MUBUF and
    /// match no rule above. Enabled only where that risk is accepted for the
    /// tighter placement. TODO: confirm the rule set covers volatile MUBUF, then
    /// this knob can go away.
    bool removeXcnt = false;

    /// Also strip `s_wait_kmcnt`. Off by default because wait-count insertion
    /// is region-scoped: an `s_load` in the kernel prologue (argument preload)
    /// never enters its dataflow, so an in-region consumer would be left
    /// unguarded. TODO: drop this knob once insertion covers the whole kernel.
    bool removeKmcnt = false;
};

/// Strip wait-counter instructions from a function.
///
/// Runs over every basic block approved by
/// `PassContext::shouldProcessBasicBlock`. Precondition pass for
/// StinkyWaitCntInsertionPass, which expects to own every emitted wait; see
/// docs/user/stinky-waitcnt-insertion-pass.md, section
/// "The reconstruction contract".
STINKYTOFU_EXPORT std::unique_ptr<Pass> createStinkyRemoveWaitCntPass(
    RemoveWaitCntOptions options = {});

}  // namespace stinkytofu
