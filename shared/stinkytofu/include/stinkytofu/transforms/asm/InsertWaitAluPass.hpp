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
class ModulePass;

/// Pass policy, deliberately not in HWModel (see its SCOPE note: facts, not knobs).
///
/// The baseline is the per-pipe / per-FIFO follower count alone: always correct on any
/// arch, and stricter than it needs to be. Every refinement below is off by default and
/// added per arch by the backend that knows the rule holds there.
struct InsertWaitAluOptions {
    /// Stamp VALU source operands as well as dests, for the src-operand WAR hazard.
    bool enableESM2TrackValuVsrc = false;
    /// Count a CSMACC producer's followers across the whole VA order rather than its
    /// own pipe, while XDL and CSMACC are the only units in flight and so complete in
    /// issue order. Weaker wait, same guarantee.
    bool sharedOrderCountFollowers = false;
    /// Treat a producer as retired once an emitted count drained the shared order past
    /// it. A per-pipe floor rises more slowly and still reports such a producer live.
    bool sharedOrderDrainRetires = false;
};

/// Insert s_wait_alu instructions for SCHED_MODE 2 (VA_VDST + VM_VSRC).
///
/// Function pass: full scoreboard analysis when run on the entry, conservative
/// entry drain when run on a callable function. Used by stinkytofu-opt single-pass
/// mode and unit tests.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createInsertWaitAluPass(InsertWaitAluOptions opts = {});

/// Whole-kernel driver: full analysis on the entry function, then the conservative
/// call-boundary drain on every callee. Reserves a seam for future caller<->callee
/// analysis.
STINKYTOFU_EXPORT std::unique_ptr<ModulePass> createInsertWaitAluModulePass(
    InsertWaitAluOptions opts = {});

}  // namespace stinkytofu
