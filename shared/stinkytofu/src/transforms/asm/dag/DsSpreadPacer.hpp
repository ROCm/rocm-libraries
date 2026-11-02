// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <climits>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace stinkytofu {

// Tells the scheduler how many ds_loads to put into each batch of WMMAs. It never picks a
// ds_load: it only says how many the current batch may take.
//
// A batch is `batch` WMMAs, counted from the stage's own start. A ds_load belongs to a stage: it
// can not start before the barrier it reads behind (`start`, a WMMA index) and must be done
// `drain` WMMAs before the barrier it feeds (`end`; INT_MAX = the region end, which needs no
// drain because the loop returns to its start). The barrier is the source of truth: its planned
// window less the drain it was planned with is where its last load goes. The loads of
// a stage are spread evenly over the stage's batches: by the end of batch k, ceil(loads * k /
// batches) are placed (63 loads over 12 batches: 5 5 5 6 ...), so the last batch ends the stage. A
// sparse stage gets a small quota, so its loads leave room for stalls; a dense one a quota above
// what one batch holds at the LDS rate (the LDS credits then throttle it). A batch that began
// behind the stage's even line (a dependency delayed its loads) may take `catchUp` more, so the
// stage closes the gap gradually. A stage the drain leaves no room for (its end falls at or before
// its start) has one batch: all its loads are released at once, as early as the graph lets them.
// Stages are paced independently: two stages open at the same time each take their own quota. Once
// the region's last WMMA has issued nothing is held.
class DsSpreadPacer {
   public:
    struct Load {
        unsigned nodeId;
        int start;         // WMMA index of the barrier the load waits behind (0 = none)
        int end;           // WMMA index of the barrier it feeds (INT_MAX = the region end)
        int earliest = 0;  // WMMA count the graph lets it issue at the soonest
        int drain = 0;     // WMMAs its barrier keeps for the last load to return
    };

    // A stage the LDS rate can not finish by its end: its loads need `needed` WMMAs from its start
    // at cyclesPerLoad, but it has end - start. Its last load lands late, whatever the pace.
    struct RateLimited {
        int start;
        int end;
        int total;
        int needed;
    };
    const std::vector<RateLimited>& rateLimited() const {
        return rateLimited_;
    }
    // The debug warning for a rate-limited stage, at the rate the stage was planned with.
    static std::string rateLimitedMessage(const RateLimited& r, int cyclesPerLoad) {
        std::ostringstream os;
        os << "stage [" << r.start << ", " << r.end << ") has " << r.total << " ds_loads that need "
           << r.needed << " WMMAs at " << cyclesPerLoad << " cycles per load but has "
           << (r.end - r.start)
           << ": the pace is too sparse for the stage, its last ds_load will land late (lower "
              "DsReadCyclesPerLoad)";
        return os.str();
    }

    void reset() {
        rateLimited_.clear();
        active_ = false;
        stageOf_.clear();
        stages_.clear();
        wmmaTotal_ = 0;
    }

    // loads: the region's ds_loads. wmmaTotal: the region's WMMAs.
    // maxPerBatch: the most loads the hardware takes in one batch (0 = not bounded), which caps
    // what a batch that is behind its line may take.
    // cyclesPerLoad, windowCycles: the LDS return rate and the WMMA window (0 = no rate floor). A
    // stage's loads can not all be placed before start + total * cyclesPerLoad / windowCycles.
    void plan(const std::vector<Load>& loads, int wmmaTotal, int batch, int catchUp,
              int maxPerBatch = 0, int cyclesPerLoad = 0, int windowCycles = 0) {
        reset();
        if (loads.empty() || wmmaTotal <= 0) return;
        active_ = true;
        batch_ = std::max(1, batch);
        catchUp_ = std::max(0, catchUp);
        wmmaTotal_ = wmmaTotal;
        maxPerBatch_ = std::max(0, maxPerBatch);
        std::map<std::pair<int, int>, int> index;
        for (const Load& l : loads) {
            const int start = std::clamp(l.start, 0, wmmaTotal);
            // Only a barrier needs the drain: the region end loops back to the start, so the last
            // load of the region has nothing to be done before.
            const int drain = l.end == INT_MAX ? 0 : std::max(0, l.drain);
            const int end = std::clamp(std::min(l.end, wmmaTotal) - drain, start, wmmaTotal);
            auto [it, fresh] = index.try_emplace({start, end}, static_cast<int>(stages_.size()));
            if (fresh) {
                Stage st;
                st.start = start;
                st.end = end;
                stages_.push_back(st);
            }
            Stage& cur = stages_[it->second];
            ++cur.total;
            cur.lastDs = std::max(cur.lastDs, l.earliest);
            stageOf_[l.nodeId] = it->second;
        }
        // The stage ends at the later of its planned end and where its last load can first issue
        // (the dependencies fix that, the pacer cannot move it), so the plan is what happens.
        for (Stage& st : stages_) {
            // Loads can not all be returned before start + total * cyclesPerLoad / windowCycles, so
            // a stage planned to end later than that ends there, never before its last load's slot.
            const int atRate =
                cyclesPerLoad > 0 && windowCycles > 0
                    ? st.start + (st.total * cyclesPerLoad + windowCycles - 1) / windowCycles
                    : 0;
            if (atRate > 0) st.end = std::min(st.end, atRate);
            if (st.lastDs > st.end) st.end = std::min(st.lastDs, wmmaTotal);
            if (atRate > st.end)
                rateLimited_.push_back({st.start, st.end, st.total, atRate - st.start});
            st.batches = std::max(1, (st.end - st.start + batch_ - 1) / batch_);
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

    // Whether the ds_load may issue with `wmmaIssued` WMMAs of the region issued so far.
    bool released(unsigned nodeId, int wmmaIssued) const {
        auto it = stageOf_.find(nodeId);
        if (!active_ || it == stageOf_.end() || wmmaIssued >= wmmaTotal_) return true;
        const Stage& st = stages_[it->second];
        const int idx = batchOf(st, wmmaIssued);
        if (idx >= st.batches)
            return true;  // past the stage's end: what is left is late, not paced
        const bool open = st.batchIdx == idx;  // the batch already took its first load
        return (open ? st.inBatch : 0) < (open ? st.allow : allowFor(st, idx));
    }

    // Loads the even line has placed by the current batch minus the loads the stage has issued
    // (> 0: the load's stage is behind its line).
    int behind(unsigned nodeId, int wmmaIssued) const {
        auto it = stageOf_.find(nodeId);
        if (!active_ || it == stageOf_.end() || wmmaIssued >= wmmaTotal_) return 0;
        const Stage& st = stages_[it->second];
        const int idx = batchOf(st, wmmaIssued);
        if (idx >= st.batches) return 0;
        return std::max(0, evenLine(st, idx) - st.issued);
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
            st.allow = allowFor(st, idx);
        }
        ++st.inBatch;
        ++st.issued;
    }

   private:
    struct Stage {
        int start = 0;    // WMMA index the stage opens at
        int end = 0;      // WMMA index the stage ends at
        int lastDs = 0;   // WMMA count its last load can first issue at
        int batches = 1;  // batches between its start and its end minus the drain
        int total = 0;    // its loads
        int issued = 0;
        int batchIdx = -1;
        int inBatch = 0;
        int allow = 0;
    };

    // Loads the even line has placed by the start of batch `idx`: ceil(total * idx / batches), so a
    // stage of 63 loads over 12 batches takes 5 5 5 6 ... and ends with its last batch.
    static int evenLine(const Stage& st, int idx) {
        return std::min(st.total, (st.total * idx + st.batches - 1) / st.batches);
    }

    // The loads batch `idx` (inside the stage) may take: its slice of the even line, plus the
    // catch-up step when the stage is behind that line.
    int allowFor(const Stage& st, int idx) const {
        const int quota = evenLine(st, idx + 1) - evenLine(st, idx);
        if (st.issued >= evenLine(st, idx)) return quota;
        // Behind the line: the catch-up step, or what the stage needs to end on time if that is
        // more, up to what the hardware takes in a batch.
        int allow = quota + catchUp_;
        if (maxPerBatch_ > 0) {
            const int left = std::max(1, st.batches - idx);
            const int need = (st.total - st.issued + left - 1) / left;
            allow = std::max(allow, std::min(need, maxPerBatch_));
        }
        return allow;
    }

    // The stage's batch holding `wmmaIssued`. Counted from the stage's start, so every batch is a
    // full `batch_` WMMAs wide whatever WMMA index the barrier it waits behind sits at (the plan
    // counts `batches` the same way).
    int batchOf(const Stage& st, int wmmaIssued) const {
        return std::max(0, wmmaIssued - st.start) / batch_;
    }

    int maxPerBatch_ = 0;
    std::vector<RateLimited> rateLimited_;
    std::unordered_map<unsigned, int> stageOf_;
    std::vector<Stage> stages_;
    int batch_ = 1;
    int catchUp_ = 0;
    int wmmaTotal_ = 0;
    bool active_ = false;
};

}  // namespace stinkytofu
