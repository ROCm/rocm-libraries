// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <climits>
#include <unordered_map>
#include <vector>

namespace stinkytofu {

// Tells the scheduler how many ds_loads to put into each batch of WMMAs. It never picks a
// ds_load: it only says how many the current batch may take.
//
// A batch is `batch` WMMAs. The region's ds_loads are spread evenly over its batches, the last
// `drainWindows` WMMAs kept free for the last load to return: X = ceil(loads / usable batches).
// A sparse region gets a small X, so its loads leave room for stalls; a dense one an X above what
// one batch holds at the LDS rate (the LDS credits then throttle it). A batch that began behind
// the even line (a dependency delayed its loads) may take `catchUp` more, so the region closes
// the gap gradually. A load past its deadline (its first consumer, or the barrier it feeds) is
// always released.
class DsSpreadPacer {
   public:
    struct Load {
        unsigned nodeId;
        int deadline;  // WMMA index the load must have issued by, INT_MAX = the region end
    };

    void reset() {
        deadline_.clear();
        total_ = 0;
        share_ = 0;
        issuedTotal_ = 0;
        active_ = false;
        batchIdx_ = -1;
        issuedInBatch_ = 0;
        allow_ = 0;
    }

    // loads: the region's ds_loads. wmmaTotal: the region's WMMAs. drainWindows: WMMAs kept free
    // in front of the region end for the last load to return.
    void plan(const std::vector<Load>& loads, int wmmaTotal, int batch, int catchUp,
              int drainWindows) {
        reset();
        if (loads.empty() || wmmaTotal <= 0) return;
        active_ = true;
        batch_ = std::max(1, batch);
        catchUp_ = std::max(0, catchUp);
        const int regionEnd = std::max(0, wmmaTotal - std::max(0, drainWindows));
        const int batches = std::max(1, (regionEnd + batch_ - 1) / batch_);
        total_ = static_cast<int>(loads.size());
        share_ = (total_ + batches - 1) / batches;
        for (const Load& l : loads) {
            const int d = std::min(std::max(0, l.deadline), regionEnd);
            deadline_[l.nodeId] = d;
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

    // ds_loads the current batch may still take.
    int allowance() const {
        return std::max(0, allow_ - issuedInBatch_);
    }

    // Whether the ds_load may issue with `wmmaIssued` WMMAs of the region issued so far.
    bool released(unsigned nodeId, int wmmaIssued) const {
        if (!active_) return true;
        auto dl = deadline_.find(nodeId);
        if (dl == deadline_.end()) return true;
        if (wmmaIssued >= dl->second) return true;
        openBatch(wmmaIssued);
        return issuedInBatch_ < allow_;
    }

    // A ds_load issued with `wmmaIssued` WMMAs of the region issued so far.
    void noteIssue(unsigned nodeId, int wmmaIssued) {
        if (!active_) return;
        openBatch(wmmaIssued);
        ++issuedInBatch_;
        ++issuedTotal_;
        deadline_.erase(nodeId);
    }

   private:
    // The allowance is set once, when the first ds_load decision of a batch is made.
    void openBatch(int wmmaIssued) const {
        const int idx = wmmaIssued / batch_;
        if (idx == batchIdx_) return;
        batchIdx_ = idx;
        issuedInBatch_ = 0;
        const bool behind = issuedTotal_ < std::min(total_, share_ * idx);
        allow_ = share_ + (behind ? catchUp_ : 0);
    }

    std::unordered_map<unsigned, int> deadline_;
    int batch_ = 1;
    int catchUp_ = 0;
    int total_ = 0;
    int share_ = 0;
    int issuedTotal_ = 0;
    // The batch state opens lazily from the (const) release query.
    mutable int batchIdx_ = -1;
    mutable int issuedInBatch_ = 0;
    mutable int allow_ = 0;
    bool active_ = false;
};

}  // namespace stinkytofu
