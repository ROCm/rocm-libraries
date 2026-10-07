// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Which instructions the co-issue repair may move, and where to (WAITCNT_COISSUE.md 7.12).
// Only SALU and VALU fillers move. A move is legal when the filler has no dependence edge
// (RAW, WAR or WAW, SCC and VCC included; buildRegisterDependencyDAG) with anything it
// crosses, crosses no label, branch, barrier or block boundary, and, if it touches a
// register a memory op writes, stays on its side of every memory wait. The passes after the
// repair re-derive everything they insert, so any order that respects this is correct.

#include <cstddef>
#include <memory>
#include <unordered_set>
#include <vector>

#include "IssueTimeline.hpp"

namespace stinkytofu {
struct StinkyInstruction;
}

namespace stinkytofu::coissue {

class MoveChecker {
   public:
    /// `order` is the loop in layout order; `blockOf[k]` is the block of position k, and
    /// `slotOf[k]` the number of non-instruction IR nodes before it in that block.
    MoveChecker(const std::vector<StinkyInstruction*>& order, const std::vector<int>& blockOf,
                const std::vector<int>& slotOf, TimedInstCache& timed);
    ~MoveChecker();

    /// Pin position k, e.g. an instruction an s_wait_alu stays in front of.
    void pin(const StinkyInstruction* inst) {
        pinned_.insert(inst);
    }
    bool movable(const StinkyInstruction& inst) const;
    /// Can order[i] go in front of position j (j may be order.size())?
    bool legal(const std::vector<StinkyInstruction*>& order, size_t i, size_t j) const;
    /// The order changed in `block`: rebuild its dependence graph.
    void update(const std::vector<StinkyInstruction*>& order, int block);

   private:
    struct BlockGraph;
    bool crossable(const StinkyInstruction& y) const;

    const std::vector<int>& blockOf_;
    const std::vector<int>& slotOf_;
    TimedInstCache& timed_;
    std::vector<std::unique_ptr<BlockGraph>> graphs_;
    std::unordered_set<const StinkyInstruction*> pinned_;
    /// Register slots memory ops of the loop write.
    std::vector<bool> memoryDefs_;
};

}  // namespace stinkytofu::coissue
