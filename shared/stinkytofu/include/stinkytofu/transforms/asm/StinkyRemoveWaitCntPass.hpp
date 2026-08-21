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

/// Which wait opcodes StinkyRemoveWaitCntPass strips. The default is "all of
/// them except the two below", so a wait survives only by opting out here.
///
/// This struct is the single place documenting *why* each exception exists;
/// call sites should point here rather than restate it.
struct RemoveWaitCntOptions {
    /// Also strip @c s_wait_tensorcnt. On by default so the insertion pass owns
    /// every wait; turn it off to let that pass reuse the incoming ones.
    ///
    /// @c s_wait_tensorcnt carries @c IF_WaitTensorCnt, a flag disjoint from
    /// @c IF_WaitCnt, so it needs its own check -- @c isWaitCnt() misses it.
    bool removeTensor = true;

    /// Also strip @c s_wait_xcnt. Off by default because only the O3 path has a
    /// hazard pass that re-places xcnt; elsewhere hand-authored drains must
    /// survive. TensileLite emits @c s_wait_xcnt @c 0 ahead of a volatile/atomic
    /// VMEM op, since XNACK-replay can reorder one past in-flight VMEM -- in
    /// StreamK that is the release-side flag store, the acquire-side flag load,
    /// and the work-queue atomic. TODO: drop this knob once a dedicated hazard
    /// pass places those drains.
    bool removeXcnt = false;

    /// Also strip @c s_wait_kmcnt. Off by default because wait-count insertion
    /// is region-scoped: an @c s_load in the kernel prologue (argument preload)
    /// never enters its dataflow, so an in-region consumer would be left
    /// unguarded. TODO: drop this knob once insertion covers the whole kernel.
    bool removeKmcnt = false;
};

/**
 * @brief Strip wait-counter instructions from a function.
 *
 * Runs over every basic block approved by
 * @c PassContext::shouldProcessBasicBlock. Precondition pass for
 * StinkyWaitCntInsertionPass, which expects to own every emitted wait; see
 * docs/user/stinky-waitcnt-insertion-pass.md, section
 * "Companion: StinkyRemoveWaitCntPass".
 */
STINKYTOFU_EXPORT std::unique_ptr<Pass> createStinkyRemoveWaitCntPass(
    RemoveWaitCntOptions options = {});

}  // namespace stinkytofu
