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

#include "stinkytofu/core/Function.hpp"

namespace stinkytofu {
namespace {

constexpr const char kClassReason[] = "in a class this run is not colouring";
constexpr const char kRegionReason[] = "outside the region this run is colouring";
constexpr const char kHeldRegisterReason[] = "in a register this run is holding";

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

void AllocationScope::pinRegisters(const AllocationConstraints& constraints,
                                   std::span<const HeldRange> ranges) {
    for (const HeldRange& range : ranges) {
        if (range.regClass == RegType::UNKNOWN || range.end < range.start) continue;
        pinnedRanges_.push_back(range);
    }

    // The range says who may not come in; this loop says the occupant may not
    // leave. Without both, the register ends up withheld rather than frozen.
    for (size_t id = 1; id < reasonByValue_.size(); ++id) {
        if (reasonByValue_[id] != nullptr) continue;
        const std::optional<RegKey> hint = constraints.hintFor(static_cast<SSAValueID>(id));
        if (!hint.has_value()) continue;
        if (isPinnedRegister(hint->type, hint->idx)) reasonByValue_[id] = kHeldRegisterReason;
    }
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
