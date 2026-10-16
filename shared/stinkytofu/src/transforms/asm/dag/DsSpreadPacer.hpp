// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <climits>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace stinkytofu {

// Spreads a region's ds_loads evenly over the region's timeline, in cycles. The region's issue
// time is `regionCycles` (wmmaTotal * windowCycles, less the WMMA queue cover the wave runs
// ahead of the pipe); the k-th ds_load (program order) is planned at
// start + k * regionCycles / n, one `period` after the previous one. A ds_load may
// issue once the clock reaches its planned cycle (less `toleranceCycles`) and at least one
// period after the previous ds_load, so a backlog (a held load, a barrier wait) drains at the
// planned rate instead of as a burst.
//
// A load whose first consumer WMMA needs it earlier than its plan has its own deadline,
// consumerWindow * windowCycles - leadCycles after the region start; from then on it issues
// regardless of the plan and of the period, so one tight consumer does not open the gate for
// every other load.
class DsSpreadPacer {
   public:
    struct Load {
        unsigned nodeId;
        int consumerWindow;  // WMMA index of the first consumer, INT_MAX if none in the region
    };

    void reset() {
        planned_.clear();
        deadline_.clear();
        toleranceCycles_ = 0;
        periodCycles_ = 0;
        lastIssue_ = kNever;
        active_ = false;
        issueClocks_.clear();
    }

    // loads: the region's ds_loads in program order. regionCycles is the region's issue time and
    // startClock the region clock at the plan.
    // manualToleranceCycles >= 0 lets every load issue that many cycles before its planned
    // cycle; < 0 is strict (0), the per-load deadlines being the only early release.
    void plan(const std::vector<Load>& loads, int regionCycles, int windowCycles,
              int manualToleranceCycles, int leadCycles, int startClock = 0) {
        reset();
        const int n = static_cast<int>(loads.size());
        if (n == 0 || regionCycles <= 0 || windowCycles <= 0) return;
        active_ = true;
        periodCycles_ = std::max<long long>(1, regionCycles / n);
        toleranceCycles_ = std::max(0, manualToleranceCycles);
        for (int k = 0; k < n; ++k) {
            planned_[loads[k].nodeId] =
                startClock + static_cast<int>(static_cast<long long>(k) * regionCycles / n);
            if (loads[k].consumerWindow != INT_MAX)
                deadline_[loads[k].nodeId] =
                    startClock + loads[k].consumerWindow * windowCycles - leadCycles;
        }
    }

    // Region cycles per ds_load: the spacing between two consecutive ds_loads.
    int periodCycles() const {
        return static_cast<int>(periodCycles_);
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

    // Whether the ds_load may issue at region clock `clock`.
    bool released(unsigned nodeId, int clock) const {
        if (!active_) return true;
        auto dl = deadline_.find(nodeId);
        if (dl != deadline_.end() && clock >= dl->second) return true;
        auto it = planned_.find(nodeId);
        if (it == planned_.end()) return true;
        return clock + toleranceCycles_ >= it->second && clock - lastIssue_ >= periodCycles_;
    }

    void noteIssue(int clock) {
        lastIssue_ = clock;
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
    static constexpr long long kNever = -(1LL << 40);  // no ds_load issued yet

    std::unordered_map<unsigned, int> planned_;
    std::unordered_map<unsigned, int> deadline_;
    std::vector<int> issueClocks_;
    long long periodCycles_ = 0;
    long long lastIssue_ = kNever;
    int toleranceCycles_ = 0;
    bool active_ = false;
};

}  // namespace stinkytofu
