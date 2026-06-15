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

/// Insert s_wait_alu instructions for SCHED_MODE 2 (VA_VDST + VM_VSRC).
///
/// Operates on whatever Function it is given — a real kernel function, or a
/// region extracted by ScopeAdaptor. Owns the mode2 lifecycle within that
/// scope: enables mode2 at the entry block, disables it before calls/returns,
/// emits s_wait_alu wherever the VA_VDST / VM_VSRC scoreboard requires.
/// Tracks only VA_VDST and VM_VSRC counters — memory completion counters
/// are owned by the memory waitcnt pass.
///
/// Caller responsibilities when running on a sub-region (ScopeAdaptor):
///   1. Mode state at the region boundary is the caller's concern. The pass
///      flips mode2 on/off inside its own scope; redundant flips on splice-back
///      are not removed here.
///   2. The scope must contain every producer of every VGPR consumed inside
///      it. Producers outside the scope (e.g. pre-loop preloads) are invisible
///      to the scoreboard and will not generate waits.
///
/// RemoveWaitAluPass must run first to strip any pre-existing wait_alu state.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createInsertWaitAluPass();

}  // namespace stinkytofu
