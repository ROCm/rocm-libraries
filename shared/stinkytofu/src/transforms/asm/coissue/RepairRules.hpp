// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Rules propose moves for a damaged window; legality (MoveChecker) and profit (robust
// acceptance) are decided by the engine (WAITCNT_COISSUE.md 7.9). The core rules always
// run; pattern plug-ins (RepairPatterns.hpp) are rules registered per architecture.

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "DamageReport.hpp"
#include "TimingProfile.hpp"

namespace stinkytofu {
struct StinkyInstruction;
}

namespace stinkytofu::coissue {

class MoveChecker;

/// order[from] goes in front of position `to` (to may be the order's size).
struct Move {
    const char* rule = "";
    size_t from = 0;
    size_t to = 0;
    int fromWindow = 0;
    int toWindow = 0;
    /// Where in the target window: "start", "end", or a pattern's own place.
    const char* place = "";
};

/// What the rules see of the loop: its order, windows and the predicted final code.
struct BlockView {
    const std::vector<StinkyInstruction*>* order = nullptr;
    /// The timeline's view of each order position.
    std::vector<const TimedInst*> timed;
    /// Order position of each window's matrix op.
    std::vector<size_t> windowStart;
    /// Per order position: the predicted final code has an s_set_vgpr_msb right in front.
    std::vector<bool> switchBefore;
    const MoveChecker* checker = nullptr;
    int radius = 4;
    const DamageReport* damage = nullptr;

    int windows() const {
        return static_cast<int>(windowStart.size());
    }
    /// The position a filler moved to the start or the end of `window` goes in front of.
    /// The end is in front of the memory waits that lead into the next matrix op.
    size_t target(int window, bool atStart) const;
    /// The window position k sits in (-1 before the first matrix op).
    int windowOf(size_t k) const;
    bool movable(size_t k) const;
};

class RepairRule {
   public:
    virtual ~RepairRule() = default;
    virtual const char* name() const = 0;
    /// Architecture gate: whether the shape this rule fixes exists under `profile`.
    virtual bool appliesTo(const TimingProfile& profile) const {
        (void)profile;
        return true;
    }
    virtual void propose(const WindowDamage& damage, const BlockView& view,
                         std::vector<Move>& out) const = 0;
};

/// MakeRoom and ClearSlot.
std::vector<std::unique_ptr<RepairRule>> coreRules();

}  // namespace stinkytofu::coissue
