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

#include <unordered_map>
#include <unordered_set>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu {
class IRBase;

namespace cluster_barrier {
namespace test {

struct Rule3SignalAnchorResult {
    IRBase* anchor = nullptr;
    int hops = 0;
    bool crossedLoopHead = false;
    /// Set when the climb nominated a spot outside the wait's segment with no hop counted,
    /// as it stood before the SCC correction moved it (unit tests only). Reporting such an
    /// anchor is a hard error, so this is the only way a test can see one that was corrected
    /// back into the segment before it got there.
    IRBase* outOfSegmentNomination = nullptr;
};

/// Unit-test entry point for the Rule 3 cycle-lead anchor search (linked from the pass TU).
Rule3SignalAnchorResult findRule3SignalAnchorByCycleLeadForUnitTest(
    StinkyInstruction* referenceAnchor, BasicBlock::iterator segBegin, IRBase* defaultAnchor,
    const std::unordered_map<const StinkyInstruction*, uint32_t>& cycleMap, int leadCycles,
    int maxLeadCycles, const std::unordered_set<StinkyInstruction*>& priorWaitAnchors, int maxHops,
    StinkyInstruction* loopHead);

}  // namespace test
}  // namespace cluster_barrier
}  // namespace stinkytofu
