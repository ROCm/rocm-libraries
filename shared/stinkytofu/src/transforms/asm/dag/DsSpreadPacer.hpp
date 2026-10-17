// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <climits>
#include <set>
#include <unordered_map>
#include <vector>

namespace stinkytofu {

// Tells the scheduler how many ds_loads to put into each batch of WMMAs. It never picks a
// ds_load and holds none that a deadline needs: it only says "this batch may take N".
//
// A batch is `batch` WMMAs. What one batch can hold is `perBatch` (the LDS return rate against the
// batch's cycles). Every ds_load has a deadline, the WMMA by which it must have issued (the
// barrier it feeds minus the drain, or its first consumer). At the start of a batch the allowance
// is the even share the deadlines ask for,
//     max over deadlines d of  ceil(unissued ds_loads due by d / batches left until d),
// capped at perBatch + catchUp. A sparse region gets a small share, so its loads are spread and
// leave room for stalls. A batch that lost ds_loads to a dependency leaves more unissued, so the
// next batches get a larger share (up to the cap) until the even line is reached again. A load
// past its deadline is always released.
class DsSpreadPacer {
   public:
    struct Load {
        unsigned nodeId;
        int deadline;  // WMMA index the load must have issued by, INT_MAX = the region end
    };

    void reset() {
        deadline_.clear();
        unissued_.clear();
        active_ = false;
        batchIdx_ = -1;
        issuedInBatch_ = 0;
        allow_ = 0;
    }

    // loads: the region's ds_loads. wmmaTotal: the region's WMMAs. drainWindows: WMMAs kept free
    // in front of the region end for the last load to return.
    void plan(const std::vector<Load>& loads, int wmmaTotal, int batch, int perBatch, int catchUp,
              int drainWindows) {
        reset();
        if (loads.empty() || wmmaTotal <= 0) return;
        active_ = true;
        batch_ = std::max(1, batch);
        perBatch_ = std::max(1, perBatch);
        catchUp_ = std::max(0, catchUp);
        const int regionEnd = std::max(0, wmmaTotal - std::max(0, drainWindows));
        for (const Load& l : loads) {
            const int d = std::min(std::max(0, l.deadline), regionEnd);
            deadline_[l.nodeId] = d;
            unissued_.insert(d);
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
        auto dl = deadline_.find(nodeId);
        if (dl == deadline_.end()) return;
        auto it = unissued_.find(dl->second);
        if (it != unissued_.end()) unissued_.erase(it);
        deadline_.erase(dl);
    }

   private:
    // The allowance is set once, when the first ds_load decision of a batch is made.
    void openBatch(int wmmaIssued) const {
        const int idx = wmmaIssued / batch_;
        if (idx == batchIdx_) return;
        batchIdx_ = idx;
        issuedInBatch_ = 0;
        int need = 0;
        int due = 0;
        for (auto it = unissued_.begin(); it != unissued_.end();) {
            const int d = *it;
            while (it != unissued_.end() && *it == d) {
                ++due;
                ++it;
            }
            const int batchesLeft = std::max(1, (d - wmmaIssued + batch_ - 1) / batch_);
            need = std::max(need, (due + batchesLeft - 1) / batchesLeft);
        }
        allow_ = std::min(need, perBatch_ + catchUp_);
    }

    std::unordered_map<unsigned, int> deadline_;
    std::multiset<int> unissued_;  // deadlines of the loads not issued yet
    int batch_ = 1;
    int perBatch_ = 1;
    int catchUp_ = 0;
    // The batch state opens lazily from the (const) release query.
    mutable int batchIdx_ = -1;
    mutable int issuedInBatch_ = 0;
    mutable int allow_ = 0;
    bool active_ = false;
};

}  // namespace stinkytofu
