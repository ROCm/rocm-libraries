// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "RepairRules.hpp"

#include <algorithm>

#include "MoveChecker.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu::coissue {

size_t BlockView::target(int window, bool atStart) const {
    const size_t start = windowStart[window];
    if (atStart) return start + 1;
    size_t k = window + 1 < windows() ? windowStart[window + 1] : order->size();
    while (k - 1 > start && !timed[k - 1]->isLabel && timed[k - 1]->kind == IssueClass::MemWait)
        --k;
    return k;
}

int BlockView::windowOf(size_t k) const {
    const auto it = std::upper_bound(windowStart.begin(), windowStart.end(), k);
    return static_cast<int>(it - windowStart.begin()) - 1;
}

bool BlockView::movable(size_t k) const {
    return checker->movable(*(*order)[k]);
}

namespace {

// A window where the pipe loses time gives one of its fillers to a nearby window, nearest
// first.
class MakeRoom : public RepairRule {
   public:
    const char* name() const override {
        return "make-room";
    }
    void propose(const WindowDamage& damage, const BlockView& view,
                 std::vector<Move>& out) const override {
        const int win = damage.window;
        const size_t start = view.windowStart[win];
        const size_t next =
            win + 1 < view.windows() ? view.windowStart[win + 1] : view.order->size();
        for (size_t i = start + 1; i < next; ++i) {
            if (!view.movable(i)) continue;
            for (int d = 1; d <= view.radius; ++d)
                for (int tw : {win - d, win + d})
                    if (tw >= 0 && tw < view.windows())
                        for (bool atStart : {true, false})
                            out.push_back({name(), i, view.target(tw, atStart), win, tw,
                                           atStart ? "start" : "end"});
        }
    }
};

// A VALU that lost its co-issue slot moves within its window or to a nearby one, ahead of
// whatever made it late.
class ClearSlot : public RepairRule {
   public:
    const char* name() const override {
        return "clear-slot";
    }
    void propose(const WindowDamage& damage, const BlockView& view,
                 std::vector<Move>& out) const override {
        const int win = damage.window;
        for (size_t i : damage.slipped) {
            if (!view.movable(i)) continue;
            for (int d = 0; d <= view.radius; ++d) {
                std::vector<int> targets{win - d};
                if (d > 0) targets.push_back(win + d);
                for (int tw : targets)
                    if (tw >= 0 && tw < view.windows())
                        for (bool atStart : {true, false})
                            out.push_back({name(), i, view.target(tw, atStart), win, tw,
                                           atStart ? "start" : "end"});
            }
        }
    }
};

}  // namespace

std::vector<std::unique_ptr<RepairRule>> coreRules() {
    std::vector<std::unique_ptr<RepairRule>> rules;
    rules.push_back(std::make_unique<MakeRoom>());
    rules.push_back(std::make_unique<ClearSlot>());
    return rules;
}

}  // namespace stinkytofu::coissue
