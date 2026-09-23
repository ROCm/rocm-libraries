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
    bool overlap = false;
};

/// Inputs used to reproduce the CDNA5 DS issue-density estimate per WMMA
/// window. Non-positive values disable throttle-shaped distribution and fall
/// back to the existing even distribution.
struct DsLoadBudgetConfig {
    int dsReadPerCap = 0;
    int dsReadQueueDepth = 0;
    int dsReadThrottleLatency = 0;
    double dsReadThrottleTransitionFactor = 1.0;
    int dsReadThrottleTransitionEntries = 0;
    int wmmaLatency = 0;
};

/// Per-window DS allocation produced by the shared hard-cap/throttle model.
STINKYTOFU_EXPORT std::vector<int> computeDsLoadWmmaWindowDistribution(
    int dsLoadCount, const DsLoadBudgetConfig& config);

/// Number of windows occupied by computeDsLoadWmmaWindowDistribution().
STINKYTOFU_EXPORT int computeDsLoadWmmaWindowsNeeded(int dsLoadCount,
                                                     const DsLoadBudgetConfig& config);

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
    /// DS reads assigned to this window. Included in issueBudget.
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
    /// Policy-assigned DS-read subset of issueBudget for a WMMA issue slot.
    int dsLoadBudgetFor(int wmmaIndex) const {
        if (!issueBudgetByWmmaIndex)
            report_fatal_error(
                "RegionHideBudget is configured for StinkyInstruction "
                "lookup, but dsLoadBudgetFor was called with a WMMA index");
        return wmmaIndex < 0 || wmmaIndex >= numWindows()
                   ? 0
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
STINKYTOFU_EXPORT RegionHideBudget analyzeWmmaHideBudget(
    const dag::RegionDAG& regionDag, const std::vector<WmmaHideBudgetBarrierInfo>& barriers,
    int wmmaHideBudgetBase, const DsLoadBudgetConfig& dsLoadConfig = {});

}  // namespace stinkytofu
