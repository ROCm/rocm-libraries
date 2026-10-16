// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <climits>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace stinkytofu {

// Spreads a region's ds_loads evenly over its WMMAs. The k-th ds_load (program order) is planned
// for WMMA window k * wmmaTotal / dsTotal and is released once that many WMMAs have issued, less
// `slack` windows of tolerance. Slack 0 is as even as possible; a large slack releases every
// ds_load at the start (greedy).
//
// Auto slack is the smallest one that still lets every ds_load reach its first consumer WMMA
// `leadWindows` windows ahead of it, so the spread is as even as the dependencies allow.
class DsSpreadPacer {
   public:
    struct Load {
        unsigned nodeId;
        int consumerWindow;  // WMMA index of the first consumer, INT_MAX if none in the region
    };

    void reset() {
        planned_.clear();
        slack_ = 0;
        active_ = false;
        issueClocks_.clear();
    }

    // loads: the region's ds_loads in program order.
    // manualSlack >= 0 fixes the slack in WMMA windows; < 0 derives it.
    void plan(const std::vector<Load>& loads, int wmmaTotal, int manualSlack, int leadWindows) {
        reset();
        const int n = static_cast<int>(loads.size());
        if (n == 0 || wmmaTotal <= 0) return;
        active_ = true;
        int need = 0;
        for (int k = 0; k < n; ++k) {
            const int planned = static_cast<int>(static_cast<long long>(k) * wmmaTotal / n);
            planned_[loads[k].nodeId] = planned;
            if (loads[k].consumerWindow != INT_MAX)
                need = std::max(need, planned + leadWindows - loads[k].consumerWindow);
        }
        slack_ = manualSlack >= 0 ? manualSlack : need;
    }

    bool active() const {
        return active_;
    }

    int slack() const {
        return slack_;
    }

    // Whether the ds_load may issue once `wmmaIssued` WMMAs of the region have.
    bool released(unsigned nodeId, int wmmaIssued) const {
        if (!active_) return true;
        auto it = planned_.find(nodeId);
        return it == planned_.end() || wmmaIssued + slack_ >= it->second;
    }

    void noteIssue(int clock) {
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
    std::unordered_map<unsigned, int> planned_;
    std::vector<int> issueClocks_;
    int slack_ = 0;
    bool active_ = false;
};

}  // namespace stinkytofu
