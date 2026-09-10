/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include "stinkytofu/transforms/asm/ra/AllocationScope.hpp"

#include <algorithm>
#include <vector>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/AsmTargetRegisters.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace stinkytofu {
namespace {

constexpr const char kClassReason[] = "in a class this run is not colouring";
constexpr const char kRegionReason[] = "outside the region this run is colouring";
constexpr const char kHeldRegisterReason[] = "in a register this run is holding";
constexpr const char kUnbankableReason[] =
    "in a register an operand with no VGPR bank selector names";

std::vector<const char*> emptyReasons(size_t valueCount) {
    return std::vector<const char*>(valueCount + 1, nullptr);
}

}  // namespace

std::optional<std::string> AllocationScope::validateClasses(const Function& function,
                                                            RegClassSet classes) {
    const RegClassSet& lifted = function.ssaArena().liftedClasses();
    if (!classes.isSubsetOf(lifted)) {
        return "@" + function.getName() + ": asked to allocate " + classes.toString() +
               " but this function was lifted for " + lifted.toString();
    }
    return std::nullopt;
}

AllocationScope::AllocationScope(RegClassSet classes, std::vector<const char*> reasonByValue,
                                 std::optional<SlotIndex> regionCut, Containment containment)
    : classes_(classes),
      reasonByValue_(std::move(reasonByValue)),
      regionCut_(regionCut),
      containment_(containment) {}

void AllocationScope::applyClassScope(const AllocationConstraints& constraints, RegClassSet classes,
                                      std::vector<const char*>& reasons) {
    for (size_t id = 1; id < reasons.size(); ++id) {
        if (reasons[id] != nullptr) continue;
        const RegType regClass = constraints.classOf(static_cast<SSAValueID>(id));
        if (regClass == RegType::UNKNOWN) continue;
        if (!classes.contains(regClass)) reasons[id] = kClassReason;
    }
}

void AllocationScope::applyRegionScope(const SSALiveIntervals& intervals, SlotIndex cut,
                                       Containment rule, std::vector<const char*>& reasons) {
    for (size_t id = 1; id < reasons.size(); ++id) {
        if (reasons[id] != nullptr) continue;
        const LiveRange& range = intervals.rangeOf(static_cast<SSAValueID>(id));
        if (range.empty()) continue;
        const bool mobile =
            rule == Containment::DefinedIn ? range.start() < cut : range.end() <= cut;
        if (!mobile) reasons[id] = kRegionReason;
    }
}

void AllocationScope::hold(const AllocationConstraints& constraints,
                           std::span<const HeldRange> ranges, const char* reason) {
    for (const HeldRange& range : ranges) {
        if (range.regClass == RegType::UNKNOWN || range.end < range.start) continue;
        pinnedRanges_.push_back(range);
    }

    // The range says who may not come in; this loop says the occupant may not
    // leave. Without both, the register ends up withheld rather than frozen.
    //
    // A reason already set is kept, so holding twice reports whichever hold
    // explains the register best rather than whichever ran last.
    for (size_t id = 1; id < reasonByValue_.size(); ++id) {
        if (reasonByValue_[id] != nullptr) continue;
        const std::optional<RegKey> hint = constraints.hintFor(static_cast<SSAValueID>(id));
        if (!hint.has_value()) continue;
        if (isPinnedRegister(hint->type, hint->idx)) reasonByValue_[id] = reason;
    }
}

void AllocationScope::pinRegisters(const AllocationConstraints& constraints,
                                   std::span<const HeldRange> ranges) {
    hold(constraints, ranges, kHeldRegisterReason);
}

void AllocationScope::holdUnbankableOperands(const AllocationConstraints& constraints,
                                             std::span<const HeldRange> ranges) {
    hold(constraints, ranges, kUnbankableReason);
}

std::vector<AllocationScope::HeldRange> AllocationScope::unbankableOperandRegisters(
    const Function& function, const AsmTargetRegisters& target) {
    std::vector<HeldRange> ranges;
    // Eight bits of index reach the whole file, so no field is short of a bank
    // and there is nothing here to constrain. Asked of the target rather than
    // the architecture, so a reduced file answers for itself.
    if (target.indexCount(RegType::V) <= kVgprBankSize) return ranges;

    for (const BasicBlock& block : function) {
        for (const IRBase& ir : block) {
            const auto* instruction = dyn_cast<StinkyInstruction>(&ir);
            if (instruction == nullptr) continue;
            // The shared field walk, not a private one: an operand attributed to
            // the wrong field would be held for a constraint it does not have,
            // or left free under one it does.
            forEachVgprOperandField(
                *instruction, [&](const StinkyRegister& reg, size_t, bool, int slot) {
                    if (slot >= 0) return;
                    if (!reg.isRegister() || reg.reg.type != RegType::V) return;
                    const uint32_t width = std::max<uint32_t>(1, reg.reg.num);
                    ranges.push_back(HeldRange{RegType::V, reg.reg.idx, reg.reg.idx + width - 1});
                });
        }
    }

    // One range per register run rather than one per operand: the same registers
    // are named by every instruction that reads them, and isPinnedRegister scans
    // the list per value.
    std::sort(ranges.begin(), ranges.end(), [](const HeldRange& a, const HeldRange& b) {
        if (a.regClass != b.regClass) return a.regClass < b.regClass;
        return a.start != b.start ? a.start < b.start : a.end < b.end;
    });
    ranges.erase(std::unique(ranges.begin(), ranges.end(),
                             [](const HeldRange& a, const HeldRange& b) {
                                 return a.regClass == b.regClass && a.start == b.start &&
                                        a.end == b.end;
                             }),
                 ranges.end());
    return ranges;
}

bool AllocationScope::isPinnedRegister(RegType regClass, uint32_t idx) const {
    for (const HeldRange& range : pinnedRanges_) {
        if (range.regClass != regClass) continue;
        if (idx >= range.start && idx <= range.end) return true;
    }
    return false;
}

const char* AllocationScope::immobileReason(SSAValueID id) const {
    if (id == kInvalidSSAValueID || id >= reasonByValue_.size()) return nullptr;
    return reasonByValue_[id];
}

AllocationScope AllocationScope::wholeFunction(const AllocationConstraints& constraints,
                                               const SSALiveIntervals& intervals,
                                               RegClassSet classes) {
    std::vector<const char*> reasons = emptyReasons(intervals.valueCount());
    applyClassScope(constraints, classes, reasons);
    return AllocationScope(classes, std::move(reasons), std::nullopt, Containment::ContainedIn);
}

AllocationScope AllocationScope::upTo(const AllocationConstraints& constraints,
                                      const SSALiveIntervals& intervals, RegClassSet classes,
                                      SlotIndex cut, Containment rule) {
    size_t valueCount = intervals.valueCount();
    std::vector<const char*> reasons = emptyReasons(valueCount);
    applyClassScope(constraints, classes, reasons);
    applyRegionScope(intervals, cut, rule, reasons);
    return AllocationScope(classes, std::move(reasons), cut, rule);
}

}  // namespace stinkytofu
