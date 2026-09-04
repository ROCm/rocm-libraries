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

class PassContext;
struct StinkyInstruction;

namespace dag {
struct RegionDAG;
}

enum class WmmaHideBudgetBarrierPosition : uint8_t { Before, After };

struct WmmaHideBudgetBarrierInfo {
  StinkyInstruction *barrier = nullptr;
  WmmaHideBudgetBarrierPosition position =
      WmmaHideBudgetBarrierPosition::Before;
  int threshold = 0;
  int dsLoadCount = 0;
  int dsLoadWmmaNeeded = 0;
};

// -------------------------------------------------------------------------
// WMMA latency-window hide budget
//
// "Hiding" is what the DAG scheduler does when it parks non-WMMA work inside a
// matrix op's latency shadow so those cycles cost nothing. The shadow is
// finite, and two separate limits bound it:
//
//   * cycles -- every cycle after the op's own issue slot can take SOMETHING
//   (SALU,
//               memory, VALU), except the ones HwInstDesc::blockedScaleMask
//               reserves for the hardware itself (the LD_SCALE half of a
//               VOP3PX2/VOP3PX3 scale pair; no pipe issues there).
//   * VALU   -- a VALU pick additionally needs a set bit in coIssueWindow, so
//   its
//               budget is the co-issue bits that are not also blocked. On
//               gfx1250 this can be zero: v_wmma_scale16_* at FP4/FP4 resolves
//               to latency 4 with coIssueWindow 0x0008, and blockedScaleMask
//               0x0001 blocks that very cycle.
//
// The budget is per window, not per region, because the work is not
// interchangeable: a ds_load feeding WMMA 6 has to be issued before WMMA 6
// whether or not a shadow has room for it, while an independent SALU can wait
// forever. So the question each window answers is "may I issue more than my
// slot?".
// -------------------------------------------------------------------------

/// What one matrix op's window can absorb, and what it is obliged to absorb
/// anyway.
struct WmmaWindowBudget {
  StinkyInstruction *wmma = nullptr;
  int capacityCycles = 0; ///< issue cycles this window can hide
  int capacityValu = 0;   ///< the VALU-capable subset of them
  /// Counts observed while traversing RegionDAG::nodes, immediately before this
  /// WMMA.
  int precedingWmmaCount = 0;
  int precedingNonWmmaCount = 0;
  /// ceil(precedingNonWmmaCount / precedingWmmaCount), or 0 for the first WMMA.
  int precedingDensity = 0;
  /// Maximum density from this WMMA through the end, computed bottom-up.
  int requiredBudget = 0;
  /// Issue cycles this window must take BEYOND capacityCycles. For every later
  /// WMMA, analysis measures how many non-WMMA instructions precede it and
  /// divides that count by the number of preceding WMMA windows. A window whose
  /// physical capacity is below any density that covers it receives the
  /// shortfall here.
  int extraIssue = 0;

  bool mustIssuePastSlot() const { return extraIssue > 0; }
};

/// Summed hide budget of one scheduling region.
struct RegionHideBudget {
  std::vector<WmmaWindowBudget> windows; ///< region program order
  std::unordered_map<const StinkyInstruction *, int> windowIndex;
  std::vector<WmmaHideBudgetBarrierInfo> barriers;
  /// Selects how scheduler clients query extraIssue: false uses the WMMA
  /// instruction identity; true uses the scheduler's current WMMA index.
  bool extraIssueByWmmaIndex = false;
  /// Raw instruction counts in RegionDAG. Every non-WMMA node, including
  /// barriers and pseudo instructions, contributes to nonWmmaInstructionCount.
  int wmmaInstructionCount = 0;
  int nonWmmaInstructionCount = 0;
  int dsLoadInstructionCount = 0;
  int nonDsLoadInstructionCount = 0;
  int wmmaHideBudgetBase = 0;

  /// Work that must precede the FIRST WMMA. No window exists yet, so it is
  /// nobody's overrun -- it is simply the region's prologue.
  int prologueCycles = 0;
  /// Work some WMMA transitively depends on (prologue included): it has a
  /// deadline.
  int deadlinedCycles = 0;
  /// Work no WMMA depends on. It still competes for window space at pick time,
  /// but it can always be deferred, so it never forces a window past its slot.
  int floatingCycles = 0;

  int numWindows() const { return static_cast<int>(windows.size()); }
  /// How many cycles \p wmma may issue beyond its slot. 0 when it fits.
  int extraIssueFor(const StinkyInstruction *wmma) const {
    if (extraIssueByWmmaIndex)
      report_fatal_error("RegionHideBudget is configured for WMMA-index "
                         "lookup, but extraIssueFor was "
                         "called with a StinkyInstruction");
    auto it = windowIndex.find(wmma);
    return it == windowIndex.end()
               ? 0
               : windows[static_cast<size_t>(it->second)].extraIssue;
  }
  /// Index-based form for clients selected by extraIssueByWmmaIndex.
  int extraIssueFor(int wmmaIndex) const {
    if (!extraIssueByWmmaIndex)
      report_fatal_error("RegionHideBudget is configured for StinkyInstruction "
                         "lookup, but extraIssueFor "
                         "was called with a WMMA index");
    return wmmaIndex < 0 || wmmaIndex >= numWindows()
               ? 0
               : windows[static_cast<size_t>(wmmaIndex)].extraIssue;
  }
  int windowsPastSlot() const;
  /// Windows that HAD co-issue slots and lost every one to blockedScaleMask. A
  /// matrix op that declares no co-issue window at all is not counted --
  /// nothing was blocked there.
  int windowsWithValuBlockedOut() const;
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
  if (blockedMask == 0 || pos < 0 || pos >= latency)
    return false;
  const int fromEnd = latency - 1 - pos;
  constexpr int kBlockedBits = static_cast<int>(sizeof(blockedMask) * 8);
  return fromEnd < kBlockedBits && ((blockedMask >> fromEnd) & 1u) != 0u;
}

/// Analyse \p regionDag using the final barrier placement metadata computed by
/// the scheduler. A barrier present in both estimators has separate Before and
/// After records.
STINKYTOFU_EXPORT RegionHideBudget
analyzeWmmaHideBudget(const dag::RegionDAG &regionDag,
                      const std::vector<WmmaHideBudgetBarrierInfo> &barriers,
                      int wmmaHideBudgetBase);

/// Report \p budget through the optimization-remark channel: which windows are
/// obliged to issue past their slot, and which have no VALU slot to hide
/// anything in. A region whose work all fits says nothing. Self-gated on
/// --remarks by emitRemark.
STINKYTOFU_EXPORT void reportWmmaHideBudget(const PassContext &passCtx,
                                            const RegionHideBudget &budget);

} // namespace stinkytofu
