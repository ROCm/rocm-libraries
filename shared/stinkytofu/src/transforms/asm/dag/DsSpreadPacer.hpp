// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <climits>
#include <cmath>
#include <unordered_map>
#include <vector>

#include "stinkytofu/core/Types.hpp"

namespace stinkytofu {

// The pace used when neither the options nor an analysis model (PassFeatureConfig::dsPaceModel)
// set one: no cycle rate, only separators. Each group of ds_loads is followed by at least
// `separators` real instructions, and a group carries as many ds_loads as the region's separators
// need to hold them all, so a region with more ds_loads than separators spreads them evenly
// instead of draining the surplus at its end.
inline PassFeatureConfig::DsPace defaultDsPace(const PassFeatureConfig::DsPaceRegion& r,
                                               int requestedSeparators) {
    PassFeatureConfig::DsPace pace;
    pace.cycles = 0;
    pace.separators = requestedSeparators >= 0 ? requestedSeparators : 1;
    pace.count = 1;
    if (r.dsTotal > 0 && pace.separators > 0) {
        const int groups = r.separatorTotal / pace.separators + 1;
        pace.count = std::max(1, (r.dsTotal + groups - 1) / groups);
    }
    return pace;
}

// Issues a region's ds_loads at their own pace, independent of the WMMA count: groups of
// `count` ds_loads, one group every `cycles` cycles of the region clock, and at least
// `separators` real instructions (WMMA / VALU / SALU) between two groups. The instructions are
// counted as they are emitted, so the spacing exists in the instruction stream and not only in
// the model's clock; a ds_load following another with nothing in between would otherwise stall
// on the hardware.
//
// The next group is allowed `cycles` after the previous group's allowed time, not after its
// actual issue: the clock moves in whole WMMA windows, and counting from the issue would round
// every period up and let the lost cycles pile into a burst at the region end. A long stall
// releases one group at once, not several.
//
// A load whose first consumer WMMA needs it earlier than its turn has its own deadline,
// consumerWindow * windowCycles - leadCycles after the region start; from then on it issues
// regardless of the pace, so one tight consumer does not open the gate for every other load.
class DsSpreadPacer {
   public:
    struct Load {
        unsigned nodeId;
        int consumerWindow;  // WMMA index of the first consumer, INT_MAX if none in the region
    };

    void reset() {
        deadline_.clear();
        pace_ = {};
        nextGroupAt_ = kNever;
        runLength_ = 0;
        separatorsSinceDs_ = 0;
        active_ = false;
        issueClocks_.clear();
    }

    // loads: the region's ds_loads in program order. startClock is the region clock at the plan.
    void plan(const std::vector<Load>& loads, const PassFeatureConfig::DsPace& pace,
              int windowCycles, int leadCycles, int startClock = 0) {
        reset();
        if (loads.empty() || windowCycles <= 0) return;
        active_ = true;
        pace_.cycles = std::max(0, pace.cycles);
        pace_.count = std::max(1, pace.count);
        pace_.separators = std::max(0, pace.separators);
        for (const Load& l : loads) {
            if (l.consumerWindow != INT_MAX)
                deadline_[l.nodeId] = startClock + l.consumerWindow * windowCycles - leadCycles;
        }
    }

    // WMMA windows the even spread gives `count` ds_loads of a region: wmmaTotal / dsTotal per
    // load, rounded up.
    static int windowsFor(int count, int wmmaTotal, int dsTotal) {
        if (count <= 0 || wmmaTotal <= 0 || dsTotal <= 0) return 0;
        return static_cast<int>((static_cast<long long>(count) * wmmaTotal + dsTotal - 1) /
                                dsTotal);
    }

    bool active() const {
        return active_;
    }

    const PassFeatureConfig::DsPace& pace() const {
        return pace_;
    }

    // Whether the ds_load may issue at region clock `clock`.
    bool released(unsigned nodeId, int clock) const {
        if (!active_) return true;
        auto dl = deadline_.find(nodeId);
        if (dl != deadline_.end() && clock >= dl->second) return true;
        if (continuesGroup()) return true;
        return clock >= nextGroupAt_ && (runLength_ == 0 || separatorsSinceDs_ >= pace_.separators);
    }

    // A ds_load issued at region clock `clock`.
    void noteIssue(int clock) {
        if (continuesGroup()) {
            ++runLength_;
        } else {
            runLength_ = 1;
            // From the previous allowed time, but never more than one period behind the clock.
            nextGroupAt_ = nextGroupAt_ == kNever
                               ? clock + pace_.cycles
                               : std::max(nextGroupAt_ + pace_.cycles, static_cast<double>(clock));
        }
        separatorsSinceDs_ = 0;
        issueClocks_.push_back(clock);
    }

    // A WMMA / VALU / SALU instruction was emitted.
    void noteSeparator() {
        ++separatorsSinceDs_;
    }

    // Coefficient of variation of the gaps between consecutive ds_load issues (0 = perfectly
    // even). 0 with fewer than three issues.
    double gapCv() const {
        if (issueClocks_.size() < 3) return 0.0;
        std::vector<double> gaps;
        for (size_t i = 1; i < issueClocks_.size(); ++i)
            gaps.push_back(static_cast<double>(issueClocks_[i] - issueClocks_[i - 1]));
        double mean = 0;
        for (double g : gaps) mean += g;
        mean /= static_cast<double>(gaps.size());
        if (mean <= 0.0) return 0.0;
        double var = 0;
        for (double g : gaps) var += (g - mean) * (g - mean);
        var /= static_cast<double>(gaps.size());
        return std::sqrt(var) / mean;
    }

   private:
    static constexpr double kNever = -1e18;  // no group started yet

    // The ds_load right after another one, nothing emitted in between, joins its group while the
    // group has room.
    bool continuesGroup() const {
        return runLength_ > 0 && separatorsSinceDs_ == 0 && runLength_ < pace_.count;
    }

    std::unordered_map<unsigned, int> deadline_;
    std::vector<int> issueClocks_;
    PassFeatureConfig::DsPace pace_;
    double nextGroupAt_ = kNever;
    int runLength_ = 0;
    int separatorsSinceDs_ = 0;
    bool active_ = false;
};

}  // namespace stinkytofu
