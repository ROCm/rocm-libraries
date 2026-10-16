// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <climits>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace stinkytofu {

// Spreads a region's ds_loads evenly over the time the region has left, in cycles.
//
// The time left is what the WMMAs still to issue will take, less the cycles the last ds_load needs
// to return before the region ends (`tailReserveCycles`) and the manual `toleranceCycles`. Each
// issue sets the next allowed time one `gap` later,
//     gap = timeLeft / dsLeft   (fractional),
// counted from the previous allowed time, not from the actual issue: the clock moves in whole
// WMMA windows, so a gap shorter than a window would otherwise round up to a full window every
// time and the lost fraction would pile up into a burst at the region end. Counted from the
// allowed time, the fraction accumulates into evenly spaced double issues instead. After a stall
// at most one extra load goes back to back. The gap is re-derived at each issue, so a held load
// (no credit, a dependency, a barrier) makes the following gaps shorter and the last ds_load still
// issues its return latency before the region ends.
//
// A load whose first consumer WMMA needs it earlier than its turn has its own deadline,
// consumerWindow * windowCycles - leadCycles after the region start; from then on it issues
// regardless of the gap, so one tight consumer does not open the gate for every other load.
class DsSpreadPacer {
   public:
    struct Load {
        unsigned nodeId;
        int consumerWindow;  // WMMA index of the first consumer, INT_MAX if none in the region
    };

    void reset() {
        deadline_.clear();
        toleranceCycles_ = 0;
        tailReserveCycles_ = 0;
        nextAllowed_ = kNever;
        active_ = false;
        issueClocks_.clear();
    }

    // loads: the region's ds_loads in program order. startClock is the region clock at the plan.
    // tailReserveCycles: cycles the last ds_load needs to return before the region ends.
    // manualToleranceCycles >= 0 adds that many cycles of lead on top of the tail reserve;
    // < 0 adds none (the per-load deadlines being the only other early release).
    void plan(const std::vector<Load>& loads, int windowCycles, int manualToleranceCycles,
              int leadCycles, int tailReserveCycles, int startClock = 0) {
        reset();
        if (loads.empty() || windowCycles <= 0) return;
        active_ = true;
        toleranceCycles_ = std::max(0, manualToleranceCycles);
        tailReserveCycles_ = std::max(0, tailReserveCycles);
        for (const Load& l : loads) {
            if (l.consumerWindow != INT_MAX)
                deadline_[l.nodeId] = startClock + l.consumerWindow * windowCycles - leadCycles;
        }
    }

    // Cycles between two consecutive ds_loads when `wmmaCyclesLeft` cycles of WMMAs and `dsLeft`
    // ds_loads remain. 0 once there is nothing left to spread over (issue as soon as ready).
    double gapCycles(int wmmaCyclesLeft, int dsLeft) const {
        if (dsLeft <= 0) return 0.0;
        const int budget = wmmaCyclesLeft - tailReserveCycles_ - toleranceCycles_;
        return budget > 0 ? static_cast<double>(budget) / dsLeft : 0.0;
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

    int toleranceCycles() const {
        return toleranceCycles_;
    }

    int tailReserveCycles() const {
        return tailReserveCycles_;
    }

    // Whether the ds_load may issue at region clock `clock`.
    bool released(unsigned nodeId, int clock) const {
        if (!active_) return true;
        auto dl = deadline_.find(nodeId);
        if (dl != deadline_.end() && clock >= dl->second) return true;
        return clock >= nextAllowed_;
    }

    // A ds_load issued at `clock`, with `wmmaCyclesLeft` cycles of WMMAs still to issue in the
    // region and `dsLeft` ds_loads still to issue (this one included).
    void noteIssue(int clock, int wmmaCyclesLeft, int dsLeft) {
        const double gap = gapCycles(wmmaCyclesLeft, dsLeft);
        // From the previous allowed time, but never more than one gap behind the clock: a stall
        // lets one extra load go, not a burst. The first load only sets the pace.
        nextAllowed_ =
            nextAllowed_ == kNever ? clock + gap : std::max(nextAllowed_ + gap, double(clock));
        issueClocks_.push_back(clock);
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
    static constexpr double kNever = -1e18;  // no ds_load issued yet

    std::unordered_map<unsigned, int> deadline_;
    std::vector<int> issueClocks_;
    double nextAllowed_ = kNever;
    int toleranceCycles_ = 0;
    int tailReserveCycles_ = 0;
    bool active_ = false;
};

}  // namespace stinkytofu
