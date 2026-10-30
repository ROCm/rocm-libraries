/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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

// Live intervals over attached SSA values, the input every register allocator
// reads.
//
// This is not the liveness the lifter runs. That one is keyed by physical
// RegKey and only decides where a merge is worth placing; this one is keyed by
// StinkySSAValue and describes where each value is live, which is what decides
// whether two values may share a physical register.
//
// A range is a set of half-open segments over SSASlotIndexes. Segments are
// needed rather than one span per value: a value defined in the entry and used
// on only one arm of a diamond is dead on the other arm, and a single span would
// claim a register across it.
//
// Block arguments are handled per CFG edge. An incoming value is live to the end
// of the predecessor it arrives from, and is not live inside the join, so a merge
// and its inputs do not interfere and can share one register.

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/analysis/asm/ssa/SSASlotIndexes.hpp"
#include "stinkytofu/ir/asm/StinkyRegister.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"

namespace stinkytofu {

class Function;
struct DominanceInfo;
struct SSALiveIntervalsBuilder;

/// Half-open [start, end) run of program points.
struct LiveSegment {
    SlotIndex start = 0;
    SlotIndex end = 0;

    bool covers(SlotIndex point) const {
        return point >= start && point < end;
    }

    bool operator==(const LiveSegment& other) const {
        return start == other.start && end == other.end;
    }
};

/// Where one SSA value is live, as sorted disjoint non-adjacent segments.
class LiveRange {
   public:
    bool empty() const {
        return segments_.empty();
    }

    std::span<const LiveSegment> segments() const {
        return segments_;
    }

    /// First live point, or SSASlotIndexes::kInvalidSlot when empty.
    SlotIndex start() const;

    /// One past the last live point, or SSASlotIndexes::kInvalidSlot when empty.
    SlotIndex end() const;

    /// Number of program points covered. This is the interval length an
    /// allocator divides by when it computes a spill weight.
    SlotIndex length() const;

    bool covers(SlotIndex point) const;

    bool overlaps(const LiveRange& other) const;

    /// Append a segment. Call finalize() once all segments are added.
    void addSegment(SlotIndex start, SlotIndex end);

    /// Sort, then merge overlapping and adjacent segments.
    void finalize();

    /// Half-open segments as "[start,end)", each index carrying the slot it
    /// names: 'u' where operands are read, 'd' where results are written.
    /// An empty range prints "-".
    std::string toString() const;

   private:
    std::vector<LiveSegment> segments_;
};

class SSALiveIntervals {
   public:
    /// True when the function carried no attached SSA, so nothing was computed.
    bool empty() const {
        return byValue_.size() <= 1;
    }

    /// Highest valid SSA value ID; IDs are 1-based and dense.
    size_t valueCount() const {
        return byValue_.empty() ? 0 : byValue_.size() - 1;
    }

    /// Live range of \p id, or an empty range for an unknown or dead ID.
    const LiveRange& rangeOf(SSAValueID id) const;

    /// True when \p a and \p b are live at a common point, so they cannot share
    /// a physical register.
    bool overlap(SSAValueID a, SSAValueID b) const;

    /// Peak simultaneously live DWORDs in \p regClass over the whole function.
    /// Derived from the same segments, so no separate pressure analysis exists.
    uint32_t peakPressure(RegType regClass) const;

    const std::map<RegType, uint32_t>& peakPressure() const {
        return peakByClass_;
    }

    const SSASlotIndexes& slots() const {
        return slots_;
    }

    /// Deterministic dump: one line per value, then one line per class peak.
    std::string toString() const;

    /// A copy of \p base in which each listed value is live from at least the
    /// given slot, with peak pressure recomputed to match.
    ///
    /// Purely mechanical: it knows nothing about why a caller wants a range to
    /// start earlier. Register allocation uses it to model early clobber, where
    /// a destination starting at its instruction's 'u' point rather than its 'd'
    /// point makes it overlap the sources dying there.
    ///
    /// Widening only -- a value already live at or before its override is left
    /// alone -- so a segment can never be inverted and no range can shrink.
    static SSALiveIntervals withEarlierStarts(
        const SSALiveIntervals& base,
        std::span<const std::pair<SSAValueID, SlotIndex>> earliestStart);

   private:
    friend struct SSALiveIntervalsBuilder;

    /// Refill peakByClass_ from the current segments. Keeps the peak from
    /// drifting away from the ranges it summarises, which is why every mutation
    /// of byValue_ ends here.
    void recomputePeakPressure();

    SSASlotIndexes slots_;
    std::vector<LiveRange> byValue_;  // indexed by valueId; slot 0 unused
    std::vector<RegType> classByValue_;
    std::vector<uint16_t> widthByValue_;  // DWORDs per value, for the peak sweep
    std::map<RegType, uint32_t> peakByClass_;
};

/// Compute live intervals for \p function, which must carry attached SSA.
/// A function without attached SSA yields an empty result.
SSALiveIntervals computeSSALiveIntervals(const Function& function);

/// As above, taking the block visit order from dominance information the caller
/// already holds. Only convergence speed depends on the order.
///
/// Exported because SSALiveIntervalsAnalysis::run() is defined in its header, so
/// registerAllAnalyses() compiles this call into every consumer of the shared
/// library. The single-argument overload has no such caller and stays internal.
STINKYTOFU_EXPORT SSALiveIntervals computeSSALiveIntervals(const Function& function,
                                                           const DominanceInfo& dominance);

}  // namespace stinkytofu
