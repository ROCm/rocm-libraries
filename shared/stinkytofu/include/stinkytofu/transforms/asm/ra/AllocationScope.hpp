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
#pragma once

// What this register-allocation run may relocate.
//
// Policy-independent like AllocationConstraints, but it states this run's remit
// rather than the program's legality. Function live-ins stay in
// AllocationConstraints::isPinned(); this object covers class scope and region
// scope only.

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "stinkytofu/analysis/asm/ssa/SSALiveIntervals.hpp"
#include "stinkytofu/analysis/asm/ssa/SSASlotIndexes.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/StinkyRegister.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationConstraints.hpp"

namespace stinkytofu {

class AllocationScope {
   public:
    enum class Containment { DefinedIn, ContainedIn };

    /// Inclusive [start, end] in one class, reading like the s[6:26] the IR
    /// prints. A single register is {class, n, n}; both bounds are always
    /// written out, since no default for `end` could mean "same as start".
    /// Unlike the half-open AsmTargetRegisters::ReservedRange on purpose.
    struct HeldRange {
        RegType regClass = RegType::UNKNOWN;
        uint32_t start = 0;
        uint32_t end = 0;
    };

    /// Returns an error when \p classes is not a subset of the function's lifted
    /// classes.
    static std::optional<std::string> validateClasses(const Function& function,
                                                      RegClassSet classes);

    static AllocationScope wholeFunction(const AllocationConstraints& constraints,
                                         const SSALiveIntervals& intervals, RegClassSet classes);

    static AllocationScope upTo(const AllocationConstraints& constraints,
                                const SSALiveIntervals& intervals, RegClassSet classes,
                                SlotIndex cut, Containment rule = Containment::ContainedIn);

    /// Hold \p ranges: every register in them keeps the value lifted into it,
    /// and no other value may be placed there.
    ///
    /// Not AsmTargetRegisters::reserve(), which withholds a register from
    /// everyone and so rejects the value already in it -- a live-in most of all.
    void pinRegisters(const AllocationConstraints& constraints, std::span<const HeldRange> ranges);

    /// True when only the value lifted from \p idx may occupy it.
    bool isPinnedRegister(RegType regClass, uint32_t idx) const;

    /// Why \p id must keep its lifted register for this run, or null when it may
    /// move. Does not cover function live-ins; those are legality in
    /// AllocationConstraints::isPinned().
    const char* immobileReason(SSAValueID id) const;

    RegClassSet classes() const {
        return classes_;
    }

    /// One past the last slot in the region, or unset for the whole function.
    std::optional<SlotIndex> regionCut() const {
        return regionCut_;
    }

    Containment containment() const {
        return containment_;
    }

   private:
    AllocationScope(RegClassSet classes, std::vector<const char*> reasonByValue,
                    std::optional<SlotIndex> regionCut, Containment containment);

    static void applyClassScope(const AllocationConstraints& constraints, RegClassSet classes,
                                std::vector<const char*>& reasons);

    static void applyRegionScope(const SSALiveIntervals& intervals, SlotIndex cut, Containment rule,
                                 std::vector<const char*>& reasons);

    RegClassSet classes_;
    std::vector<const char*> reasonByValue_;
    std::vector<HeldRange> pinnedRanges_;
    std::optional<SlotIndex> regionCut_;
    Containment containment_;
};

}  // namespace stinkytofu
