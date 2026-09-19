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

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"

namespace stinkytofu {

struct StinkyInstruction;

namespace dag {
struct RegionDAG;
}

enum class WmmaHideBudgetBarrierPosition : uint8_t { Before, After };

struct WmmaHideBudgetBarrierInfo {
    StinkyInstruction* barrier = nullptr;
    WmmaHideBudgetBarrierPosition position = WmmaHideBudgetBarrierPosition::Before;
    int threshold = 0;
    int dsLoadCount = 0;
    int dsLoadWmmaNeeded = 0;
};

// -------------------------------------------------------------------------
// Per-WMMA instruction budget
//
// Each budget is a count of non-WMMA instructions, matching the scheduler's
// nonWmmaIssuedThisRegion_ counter. It is intentionally not a cycle estimate:
// instructions with different issueCycles each consume one budget unit.
// Barriers and pseudo instructions are included because they are DAG nodes and
// are counted by the scheduler when picked.
// -------------------------------------------------------------------------

/// Number of non-WMMA instructions assigned to one matrix-op window.
struct WmmaWindowBudget {
    StinkyInstruction* wmma = nullptr;
    /// Non-WMMA instructions the scheduling policy assigns to this window.
    int issueBudget = 0;
    /// ds_load-only count assigned to this window: barrier-attributed
    /// ds_loads (from the Before/After barrier steps) plus this window's
    /// share of any ds_load with no barrier relationship at all. Kept
    /// separate from issueBudget (which stays a mixed ds+non-ds total driving
    /// the existing hold-back-next-WMMA decision) so the scheduler's ds
    /// per-window cap has one authoritative source instead of a second,
    /// independently-computed number that can disagree with this analysis.
    int dsLoadBudget = 0;
};

/// Summed hide budget of one scheduling region.
struct RegionHideBudget {
    std::vector<WmmaWindowBudget> windows;  ///< region program order
    std::unordered_map<const StinkyInstruction*, int> windowIndex;
    std::vector<WmmaHideBudgetBarrierInfo> barriers;
    /// Selects how scheduler clients query issueBudget: false uses the WMMA
    /// instruction identity; true uses the scheduler's current WMMA index.
    bool issueBudgetByWmmaIndex = false;
    /// Raw instruction counts in RegionDAG. Every non-WMMA node, including
    /// barriers and pseudo instructions, contributes to nonWmmaInstructionCount.
    int wmmaInstructionCount = 0;
    int nonWmmaInstructionCount = 0;
    int dsLoadInstructionCount = 0;
    int nonDsLoadInstructionCount = 0;
    int wmmaHideBudgetBase = 0;
    /// ds_load count that belongs after the last WMMA in the region (this
    /// window's share of the even ds_load spread). There is no WMMA left to
    /// hold back there, so this is consulted only by the ds cap, never by the
    /// hold-back-next-WMMA logic.
    int dsLoadTailBudget = 0;

    int numWindows() const {
        return static_cast<int>(windows.size());
    }
    /// Policy-assigned non-WMMA instruction count for \p wmma.
    int issueBudgetFor(const StinkyInstruction* wmma) const {
        if (issueBudgetByWmmaIndex)
            report_fatal_error(
                "RegionHideBudget is configured for WMMA-index "
                "lookup, but issueBudgetFor was "
                "called with a StinkyInstruction");
        auto it = windowIndex.find(wmma);
        return it == windowIndex.end() ? 0 : windows[static_cast<size_t>(it->second)].issueBudget;
    }
    /// Index-based form for clients selected by issueBudgetByWmmaIndex.
    int issueBudgetFor(int wmmaIndex) const {
        if (!issueBudgetByWmmaIndex)
            report_fatal_error(
                "RegionHideBudget is configured for StinkyInstruction "
                "lookup, but issueBudgetFor "
                "was called with a WMMA index");
        return wmmaIndex < 0 || wmmaIndex >= numWindows()
                   ? 0
                   : windows[static_cast<size_t>(wmmaIndex)].issueBudget;
    }
    /// ds_load-only budget for \p wmmaIndex; returns dsLoadTailBudget once
    /// \p wmmaIndex reaches or passes the last WMMA window (mirrors how
    /// CDNA5ReadyQueue clamps wmmaIssuedCountThisRegion_ into a window index).
    int dsLoadBudgetFor(int wmmaIndex) const {
        if (wmmaIndex < 0) return 0;
        return wmmaIndex >= numWindows() ? dsLoadTailBudget
                                         : windows[static_cast<size_t>(wmmaIndex)].dsLoadBudget;
    }
};

/// True when \p pos -- cycles elapsed since a matrix op issued -- lands on a
/// cycle its blockedScaleMask reserves. The mask is END-anchored (bit 0 = the
/// window's LAST cycle) so a single declaration stays correct across every
/// per-format latency override; see HwInstDesc::blockedScaleMask. Shared with
/// the scheduler, which asks the same question of its live window.
///
/// Defined inline: the scheduler calls this once per window cycle from
/// advanceTime(), computeValuAdvanceCycles() and freeCoIssueSpace(), so it must
/// not become a call.
inline bool isBlockedWindowCycle(int pos, int latency, uint16_t blockedMask) {
    if (blockedMask == 0 || pos < 0 || pos >= latency) return false;
    const int fromEnd = latency - 1 - pos;
    constexpr int kBlockedBits = static_cast<int>(sizeof(blockedMask) * 8);
    return fromEnd < kBlockedBits && ((blockedMask >> fromEnd) & 1u) != 0u;
}

/// Analyse \p regionDag using the final barrier placement metadata computed by
/// the scheduler. A barrier present in both estimators has separate Before and
/// After records.
/// \p dsReadPerWmmaCap ceilings each window's share of ds_loads that have no
/// barrier relationship at all (dagFeatures.dsReadPerWmma /
/// StinkyTofuDsReadPerWmma). It intentionally does NOT ceiling
/// barrier-attributed ds_loads (Steps 2/3): those counts reflect a real
/// dependency requirement, and capping them below what a barrier needs would
/// reintroduce the exact cap-vs-hide-budget disagreement this analysis exists
/// to prevent.
STINKYTOFU_EXPORT RegionHideBudget analyzeWmmaHideBudget(
    const dag::RegionDAG& regionDag, const std::vector<WmmaHideBudgetBarrierInfo>& barriers,
    int wmmaHideBudgetBase, int dsReadPerWmmaCap);

}  // namespace stinkytofu
