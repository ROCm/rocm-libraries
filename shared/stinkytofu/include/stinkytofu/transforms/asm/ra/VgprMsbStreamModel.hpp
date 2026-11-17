// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// s_set_vgpr_msb as a BankCostModel.
//
// InsertVgprMsbPass emits a switch whenever the next instruction needs another
// VGPR bank word (VgprMsbTracker). This prices a choice of banks by those
// switches, each weighted like the greedy engine weights a block's uses:
// 10^min(loop depth, 4).

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/support/LoopDetection.hpp"
#include "stinkytofu/transforms/asm/ra/BankCostModel.hpp"

namespace stinkytofu {

/// Built once, from the function in the order the insertion pass will see it.
///
/// An event is an instruction that carries a bank word. A block start, a label
/// or a call leaves no word known, so it begins a segment, and the first event
/// of a segment always switches. Each slot of an event is the bank of a value
/// (unit 0 of a lifted operand), a fixed bank (an operand not lifted), or bank 0
/// where the instruction has no operand, as the word itself holds it.
///
/// Whether event j switches depends only on j and the event before it in its
/// segment, so moving a value's bank changes only its uses and the event after
/// each: delta() costs O(uses).
class STINKYTOFU_EXPORT VgprMsbStreamModel final : public BankCostModel {
   public:
    VgprMsbStreamModel(const Function& function, const std::vector<Loop>& loops);

    std::span<const SSAValueID> bankSensitiveValues() const override;
    double useWeight(SSAValueID value) const override;
    double cost(std::span<const uint8_t> bankOfValue) const override;
    double delta(std::span<const uint8_t> bankOfValue, std::span<const SSAValueID> moved,
                 uint8_t bank) const override;

   private:
    static constexpr uint32_t kNoEvent = UINT32_MAX;

    struct Event {
        /// The value whose bank each slot takes, or kInvalidSSAValueID for a
        /// fixed one.
        std::array<SSAValueID, 4> value{};
        /// The bank of each slot without a value.
        std::array<uint8_t, 4> fixed{};
        double weight = 1.0;
        /// The event before and after this one in its segment.
        uint32_t previous = kNoEvent;
        uint32_t next = kNoEvent;
    };

    /// Does \p event switch, with the values in \p moved in \p bank and the rest
    /// where \p bankOfValue has them?
    bool switchesAt(uint32_t event, std::span<const uint8_t> bankOfValue,
                    std::span<const SSAValueID> moved, uint8_t bank) const;

    std::vector<Event> events_;
    std::vector<SSAValueID> values_;
    /// By value: the heaviest event weight among its uses.
    std::vector<double> useWeight_;
    /// The events using value v are useEvents_[useBegin_[v], useBegin_[v + 1]).
    std::vector<uint32_t> useBegin_;
    std::vector<uint32_t> useEvents_;
};

}  // namespace stinkytofu
