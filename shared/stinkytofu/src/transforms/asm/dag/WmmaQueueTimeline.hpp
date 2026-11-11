// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <climits>
#include <cstdint>
#include <vector>

namespace stinkytofu {

// The single timeline model for the CDNA5 WMMA hardware queue. A live scheduler binds this view
// to its queue state; a read-only policy estimate binds it to a local copy and runs the same
// append/advance/occupancy transitions.
class WmmaQueueTimeline {
   public:
    static constexpr uint8_t kValuSlot = 1;
    static constexpr uint8_t kBlockedSlot = 2;

    struct State {
        int position = 0;
        int latency = 0;
        std::vector<uint8_t> slots;
        std::vector<int> ends;
    };

    struct Window {
        int latency = 0;
        int issue = 0;
        uint16_t coIssueMask = 0;
        uint16_t blockedMask = 0;
    };

    enum class AdvanceKind { Elapsed, Issue, ValuIssue };

    explicit WmmaQueueTimeline(State& state)
        : state_{state.position, state.latency, state.slots, state.ends} {}

    WmmaQueueTimeline(int& position, int& latency, std::vector<uint8_t>& slots,
                      std::vector<int>& ends)
        : state_{position, latency, slots, ends} {}

    static bool isBlockedWindowCycle(int pos, int latency, uint16_t blockedMask) {
        if (blockedMask == 0 || pos < 0 || pos >= latency) return false;
        const int fromEnd = latency - 1 - pos;
        constexpr int kBlockedBits = static_cast<int>(sizeof(blockedMask) * 8);
        return fromEnd < kBlockedBits && ((blockedMask >> fromEnd) & 1u) != 0u;
    }

    static bool isBlocked(const std::vector<uint8_t>& slots, int latency, int pos) {
        return pos >= 0 && pos < latency && (slots[pos] & kBlockedSlot) != 0;
    }

    bool isBlocked(int pos) const {
        return isBlocked(state_.slots, state_.latency, pos);
    }

    bool windowOpen() const {
        return state_.position < state_.latency;
    }

    bool valuSlotOpen() const {
        return windowOpen() && (state_.slots[state_.position] & kValuSlot) != 0;
    }

    void append(const Window& wmma) {
        constexpr int kCoIssueBits = static_cast<int>(sizeof(wmma.coIssueMask) * 8);
        const int latency = std::max(1, wmma.latency);
        for (int cycle = 0; cycle < latency; ++cycle) {
            uint8_t slot = 0;
            if (cycle < kCoIssueBits && ((wmma.coIssueMask >> cycle) & 1u) != 0u) slot |= kValuSlot;
            if (isBlockedWindowCycle(cycle, latency, wmma.blockedMask)) slot |= kBlockedSlot;
            state_.slots.push_back(slot);
        }
        state_.latency = static_cast<int>(state_.slots.size());
        state_.ends.push_back(state_.latency);
    }

    // Advance the queue and return elapsed hardware cycles. Issue skips scale-blocked cycles;
    // ValuIssue additionally advances only through VALU co-issue slots.
    int advance(int cycles, AdvanceKind kind) {
        if (cycles <= 0) return 0;
        int elapsed = cycles;
        if (kind == AdvanceKind::ValuIssue) {
            elapsed = valuAdvanceCycles(cycles);
        } else if (kind == AdvanceKind::Issue) {
            int at = state_.position;
            for (int cycle = 0; cycle < cycles; ++cycle) {
                while (isBlocked(at)) ++at;
                ++at;
            }
            elapsed = at - state_.position;
        }
        int landing = state_.position + elapsed;
        while (isBlocked(landing)) ++landing;
        elapsed = landing - state_.position;
        state_.position = landing;
        return elapsed;
    }

    int valuAdvanceCycles(int issueCycles) const {
        return valuAdvanceCycles(state_.slots, state_.position, state_.latency, issueCycles);
    }

    static int valuAdvanceCycles(const std::vector<uint8_t>& slots, int position, int latency,
                                 int issueCycles) {
        if (issueCycles <= 0 || position >= latency) return std::max(0, issueCycles);
        int elapsed = 0;
        int issued = 0;
        while (issued < issueCycles) {
            const int pos = position + elapsed;
            if (pos >= latency) {
                elapsed += issueCycles - issued;
                break;
            }
            const uint8_t slot = slots[pos];
            if ((slot & (kValuSlot | kBlockedSlot)) == kValuSlot) ++issued;
            ++elapsed;
        }
        return elapsed;
    }

    int outstanding() const {
        return outstanding(state_.ends, state_.position);
    }

    static int outstanding(const std::vector<int>& ends, int position) {
        return static_cast<int>(
            std::count_if(ends.begin(), ends.end(), [=](int end) { return end > position; }));
    }

    int coverCycles() const {
        return coverCycles(state_.ends, state_.position);
    }

    static int coverCycles(const std::vector<int>& ends, int position) {
        return ends.empty() ? 0 : std::max(0, ends.back() - position);
    }

    int soonestEnd() const {
        return soonestEnd(state_.ends, state_.position);
    }

    static int soonestEnd(const std::vector<int>& ends, int position) {
        int soonest = INT_MAX;
        for (int end : ends)
            if (end > position) soonest = std::min(soonest, end - position);
        return soonest;
    }

    int freeCoIssueSpace() const {
        return freeCoIssueSpace(state_.slots, state_.position, state_.latency);
    }

    static int freeCoIssueSpace(const std::vector<uint8_t>& slots, int position, int latency) {
        for (int pos = position; pos < latency; ++pos)
            if (isBlocked(slots, latency, pos)) return pos - position;
        return latency - position;
    }

    // Remove finished queue entries and compact a long active window. Returns the number of
    // oldest entries removed so pointer/source sidecars can erase the same prefix.
    int prune() {
        int removed = 0;
        while (state_.ends.size() > 1 && state_.ends.front() <= state_.position) {
            state_.ends.erase(state_.ends.begin());
            ++removed;
        }
        constexpr int kCompactAt = 64;
        if (state_.position >= kCompactAt && state_.position < state_.latency) {
            state_.slots.erase(state_.slots.begin(), state_.slots.begin() + state_.position);
            for (int& end : state_.ends) end -= state_.position;
            state_.latency -= state_.position;
            state_.position = 0;
        }
        return removed;
    }

    void reset() {
        state_.position = 0;
        state_.latency = 0;
        state_.slots.clear();
        state_.ends.clear();
    }

   private:
    struct View {
        int& position;
        int& latency;
        std::vector<uint8_t>& slots;
        std::vector<int>& ends;
    };
    View state_;
};

}  // namespace stinkytofu
