// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <map>
#include <unordered_map>
#include <vector>

namespace stinkytofu {

// Tells the scheduler how many ds_loads to put into each batch of WMMAs. It never picks a
// ds_load: it only says how many the current batch may take.
//
// A batch is `batch` WMMAs, counted from the stage's own start. A ds_load belongs to a stage: it
// can not start before the barrier it reads behind (`start`, a WMMA index) and must be done
// `drainWindows` WMMAs before the barrier it feeds (`end`, INT_MAX = the region end). The loads of
// a stage are spread evenly over the stage's batches: X = ceil(loads / batches). A sparse stage
// gets a small X, so its loads leave room for stalls; a dense one an X above what one batch holds
// at the LDS rate (the LDS credits then throttle it). A batch that began behind the stage's even
// line (a dependency delayed its loads) may take `catchUp` more, so the stage closes the gap
// gradually. A stage the drain leaves no room for (its end falls at or before its start) has one
// batch: all its loads are released at once, as early as the graph lets them. Stages are paced
// independently: two stages open at the same time each take their own share. Once the region's last
// WMMA has issued nothing is held.
class DsSpreadPacer {
   public:
    struct Load {
        unsigned nodeId;
        int start;  // WMMA index of the barrier the load waits behind (0 = none)
        int end;    // WMMA index of the barrier it feeds (INT_MAX = the region end)
    };

    void reset() {
        active_ = false;
        stageOf_.clear();
        stages_.clear();
        wmmaTotal_ = 0;
    }

    // loads: the region's ds_loads. wmmaTotal: the region's WMMAs. drainWindows: WMMAs kept free
    // in front of a barrier (and the region end) for the last load feeding it to return.
    void plan(const std::vector<Load>& loads, int wmmaTotal, int batch, int catchUp,
              int drainWindows) {
        reset();
        if (loads.empty() || wmmaTotal <= 0) return;
        active_ = true;
        batch_ = std::max(1, batch);
        catchUp_ = std::max(0, catchUp);
        wmmaTotal_ = wmmaTotal;
        std::map<std::pair<int, int>, int> index;
        for (const Load& l : loads) {
            const int start = std::clamp(l.start, 0, wmmaTotal);
            const int end = std::clamp(std::min(l.end, wmmaTotal) - std::max(0, drainWindows),
                                       start, wmmaTotal);
            auto [it, fresh] = index.try_emplace({start, end}, static_cast<int>(stages_.size()));
            if (fresh) {
                Stage st;
                st.start = start;
                st.batches = std::max(1, (end - start + batch_ - 1) / batch_);
                stages_.push_back(st);
            }
            ++stages_[it->second].total;
            stageOf_[l.nodeId] = it->second;
        }
        for (Stage& st : stages_) st.share = (st.total + st.batches - 1) / st.batches;
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

    // Whether the ds_load may issue with `wmmaIssued` WMMAs of the region issued so far. The first
    // load of a batch always may (a stage's share is at least 1).
    bool released(unsigned nodeId, int wmmaIssued) const {
        auto it = stageOf_.find(nodeId);
        if (!active_ || it == stageOf_.end() || wmmaIssued >= wmmaTotal_) return true;
        const Stage& st = stages_[it->second];
        return st.batchIdx != batchOf(st, wmmaIssued) || st.inBatch < st.allow;
    }

    // A ds_load issued with `wmmaIssued` WMMAs of the region issued so far.
    void noteIssue(unsigned nodeId, int wmmaIssued) {
        auto it = stageOf_.find(nodeId);
        if (!active_ || it == stageOf_.end()) return;
        Stage& st = stages_[it->second];
        const int idx = batchOf(st, wmmaIssued);
        if (idx != st.batchIdx) {  // the allowance is set when a batch gets its first load
            st.batchIdx = idx;
            st.inBatch = 0;
            const bool behind = st.issued < std::min(st.total, st.share * idx);
            st.allow = st.share + (behind ? catchUp_ : 0);
        }
        ++st.inBatch;
        ++st.issued;
    }

   private:
    struct Stage {
        int start = 0;    // WMMA index the stage opens at
        int batches = 1;  // batches between its start and its end minus the drain
        int total = 0;    // its loads
        int share = 0;    // loads per batch on the even line
        int issued = 0;
        int batchIdx = -1;
        int inBatch = 0;
        int allow = 0;
    };

    // The stage's batch holding `wmmaIssued`. Counted from the stage's start, so every batch is a
    // full `batch_` WMMAs wide whatever WMMA index the barrier it waits behind sits at (the plan
    // counts `batches` the same way).
    int batchOf(const Stage& st, int wmmaIssued) const {
        return std::max(0, wmmaIssued - st.start) / batch_;
    }

    std::unordered_map<unsigned, int> stageOf_;
    std::vector<Stage> stages_;
    int batch_ = 1;
    int catchUp_ = 0;
    int wmmaTotal_ = 0;
    bool active_ = false;
};

}  // namespace stinkytofu
