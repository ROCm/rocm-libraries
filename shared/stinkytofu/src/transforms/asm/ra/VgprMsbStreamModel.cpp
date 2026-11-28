// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/ra/VgprMsbStreamModel.hpp"

#include <algorithm>
#include <cassert>
#include <utility>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationConstraints.hpp"

namespace stinkytofu {
namespace {

// BankCostModel's unit is the greedy engine's loop weighting, so a switch and a
// use weigh alike; like the greedy engine's, it stops growing at this depth.
constexpr uint32_t kMaxLoopDepth = 4;

uint8_t bankIn(std::span<const uint8_t> bankOfValue, SSAValueID value) {
    return value < bankOfValue.size() ? bankOfValue[value] : kUnknownBank;
}

}  // namespace

VgprMsbStreamModel::VgprMsbStreamModel(const Function& function, const std::vector<Loop>& loops) {
    const RegClassSet& lifted = function.ssaArena().liftedClasses();
    std::vector<std::pair<SSAValueID, uint32_t>> uses;
    for (const BasicBlock& bb : function) {
        uint32_t nesting = 0;
        for (const Loop& loop : loops) {
            if (loop.contains(&bb)) ++nesting;
        }
        double weight = 1.0;
        for (uint32_t level = 0; level < std::min(nesting, kMaxLoopDepth); ++level)
            weight *= kBankCostLoopWeight;

        // Walked as InsertVgprMsbPass walks it, so the segments are where its
        // tracker forgets the word or takes one from a setter in the stream.
        uint32_t previous = kNoEvent;
        std::optional<std::array<uint8_t, 4>> inForce;
        for (const IRBase& ir : bb) {
            const auto* inst = dyn_cast<StinkyInstruction>(&ir);
            if (inst == nullptr) continue;
            if (inst->getUnifiedOpcode() == GFX::LABEL) {
                previous = kNoEvent;
                inForce.reset();
                continue;
            }
            if (isPseudoInst(inst)) continue;
            if (isCall(*inst)) {
                previous = kNoEvent;
                inForce.reset();
                continue;
            }
            if (inst->getUnifiedOpcode() == GFX::s_set_vgpr_msb) {
                previous = kNoEvent;
                inForce.reset();
                const int word = vgprMsbSetterWord(*inst);
                if (word < 0) continue;
                inForce.emplace();
                for (size_t slot = 0; slot < inForce->size(); ++slot)
                    (*inForce)[slot] =
                        static_cast<uint8_t>(decodeVgprMsbForSlot(word, static_cast<int>(slot)));
                continue;
            }
            if (!carriesVgprMsb(*inst)) continue;

            Event event;
            event.value.fill(kInvalidSSAValueID);
            bool hasVgpr = false;
            const OperandGroups groups =
                inst->hasAttachedSSA() ? operandGroupsOf(*inst, lifted) : OperandGroups{};
            forEachVgprMsbSlotOperand(*inst, [&](const StinkyRegister& reg, size_t operand,
                                                 bool isDest, int slot) {
                hasVgpr = true;
                const std::span<const SSAValueID> ids = groups.at(operand, isDest);
                const bool liftedOperand = !ids.empty() && ids.front() != kInvalidSSAValueID;
                event.value[slot] = liftedOperand ? ids.front() : kInvalidSSAValueID;
                event.fixed[slot] = liftedOperand ? 0 : static_cast<uint8_t>(getMsbFromVgpr(reg));
            });
            if (!hasVgpr) continue;

            event.weight = weight;
            event.previous = previous;
            if (previous == kNoEvent) event.inForce = inForce;
            const auto index = static_cast<uint32_t>(events_.size());
            if (previous != kNoEvent) events_[previous].next = index;
            for (size_t slot = 0; slot < event.value.size(); ++slot) {
                const SSAValueID value = event.value[slot];
                if (value == kInvalidSSAValueID) continue;
                // One use per event, however many of its slots the value fills.
                const auto seen = event.value.begin() + static_cast<std::ptrdiff_t>(slot);
                if (std::find(event.value.begin(), seen, value) != seen) continue;
                uses.emplace_back(value, index);
            }
            events_.push_back(event);
            previous = index;
        }
    }

    std::sort(uses.begin(), uses.end());
    const SSAValueID highest = uses.empty() ? 0 : uses.back().first;
    useBegin_.assign(static_cast<size_t>(highest) + 2, 0);
    useWeight_.assign(static_cast<size_t>(highest) + 1, 0.0);
    useEvents_.reserve(uses.size());
    for (const auto& [value, event] : uses) {
        ++useBegin_[value + 1];
        useEvents_.push_back(event);
        useWeight_[value] = std::max(useWeight_[value], events_[event].weight);
        if (values_.empty() || values_.back() != value) values_.push_back(value);
    }
    for (size_t value = 1; value < useBegin_.size(); ++value)
        useBegin_[value] += useBegin_[value - 1];
}

std::span<const SSAValueID> VgprMsbStreamModel::bankSensitiveValues() const {
    return values_;
}

double VgprMsbStreamModel::useWeight(SSAValueID value) const {
    return value < useWeight_.size() ? useWeight_[value] : 0.0;
}

bool VgprMsbStreamModel::switchesAt(uint32_t index, std::span<const uint8_t> bankOfValue,
                                    std::span<const SSAValueID> moved, uint8_t bank) const {
    const Event& event = events_[index];
    if (event.previous == kNoEvent && !event.inForce.has_value()) return true;
    const auto slotBank = [&](const Event& of, size_t slot) -> uint8_t {
        const SSAValueID value = of.value[slot];
        if (value == kInvalidSSAValueID) return of.fixed[slot];
        if (std::binary_search(moved.begin(), moved.end(), value)) return bank;
        return bankIn(bankOfValue, value);
    };
    for (size_t slot = 0; slot < event.value.size(); ++slot) {
        const uint8_t before = event.previous == kNoEvent ? (*event.inForce)[slot]
                                                          : slotBank(events_[event.previous], slot);
        if (slotBank(event, slot) != before) return true;
    }
    return false;
}

double VgprMsbStreamModel::cost(std::span<const uint8_t> bankOfValue) const {
    double total = 0.0;
    for (uint32_t index = 0; index < events_.size(); ++index) {
        if (switchesAt(index, bankOfValue, {}, kUnknownBank)) total += events_[index].weight;
    }
    return total;
}

double VgprMsbStreamModel::delta(std::span<const uint8_t> bankOfValue,
                                 std::span<const SSAValueID> moved, uint8_t bank) const {
    assert(std::is_sorted(moved.begin(), moved.end()) && "moved values must be ascending");
    std::vector<uint32_t> touched;
    for (const SSAValueID value : moved) {
        if (static_cast<size_t>(value) + 1 >= useBegin_.size()) continue;
        for (uint32_t use = useBegin_[value]; use < useBegin_[value + 1]; ++use) {
            const uint32_t event = useEvents_[use];
            touched.push_back(event);
            if (events_[event].next != kNoEvent) touched.push_back(events_[event].next);
        }
    }
    std::sort(touched.begin(), touched.end());
    touched.erase(std::unique(touched.begin(), touched.end()), touched.end());

    double change = 0.0;
    for (const uint32_t event : touched) {
        const bool before = switchesAt(event, bankOfValue, {}, kUnknownBank);
        const bool after = switchesAt(event, bankOfValue, moved, bank);
        if (before != after) change += after ? events_[event].weight : -events_[event].weight;
    }
    return change;
}

}  // namespace stinkytofu
