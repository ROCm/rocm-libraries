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

#include <optional>
#include <sstream>
#include <string>

#include "../ssa/AttachedSSATestUtils.hpp"
#include "TestHelpers.hpp"
#include "stinkytofu/analysis/asm/ssa/SSALiveIntervals.hpp"
#include "stinkytofu/hardware/AsmTargetRegisters.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"
#include "stinkytofu/support/LoopDetection.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationConstraints.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationScope.hpp"
#include "stinkytofu/transforms/asm/ra/RegisterAllocator.hpp"
#include "stinkytofu/transforms/asm/ssa/LiftAsmRegistersToSSAPass.hpp"

namespace stinkytofu {
namespace test {

inline constexpr GfxArchID kRaTestArch = GfxArchID::Gfx1250;

inline bool liftForAllocation(Function& function) {
    Expected<LiftAttachedSSAResult> lifted = liftAsmRegistersToAttachedSSA(function);
    if (!lifted.hasValue()) {
        ADD_FAILURE() << lifted.getError();
        return false;
    }
    if (!function.hasAttachedSSA()) {
        ADD_FAILURE() << "lift produced no attached SSA";
        return false;
    }
    return true;
}

inline std::string physicalIR(const Function& function) {
    std::ostringstream out;
    AsmPrinter printer(out);
    printer.print(function);
    return out.str();
}

/// Namespace scope so its default member initializers are usable in the default
/// argument below; a nested aggregate cannot be default-initialized while its
/// enclosing class is still incomplete.
struct AllocationRegionOptions {
    std::optional<SlotIndex> cut;
    AllocationScope::Containment containment = AllocationScope::Containment::ContainedIn;
};

/// Owns the analyses an AllocationContext references.
class AllocationSetup {
   public:
    using RegionOptions = AllocationRegionOptions;

    explicit AllocationSetup(Function& function,
                             RegClassSet allocate = RegClassSet::only(RegType::V),
                             RegionOptions region = {})
        : intervals_(computeSSALiveIntervals(function)),
          target_(AsmTargetRegisters::forFunction(function)),
          constraints_(AllocationConstraints::build(function, target_)),
          loops_(detectLoops(function)),
          scope_(buildScope(constraints_, intervals_, allocate, region)),
          context_{function, intervals_, target_, constraints_, loops_, scope_} {}

    const AllocationContext& context() const {
        return context_;
    }

    const AllocationScope& scope() const {
        return scope_;
    }

    AsmTargetRegisters& target() {
        return target_;
    }

    const AllocationConstraints& constraints() const {
        return constraints_;
    }

    const SSALiveIntervals& intervals() const {
        return intervals_;
    }

   private:
    static AllocationScope buildScope(const AllocationConstraints& constraints,
                                      const SSALiveIntervals& intervals, RegClassSet allocate,
                                      const RegionOptions& region) {
        if (region.cut.has_value()) {
            return AllocationScope::upTo(constraints, intervals, allocate, *region.cut,
                                         region.containment);
        }
        return AllocationScope::wholeFunction(constraints, intervals, allocate);
    }

    SSALiveIntervals intervals_;
    AsmTargetRegisters target_;
    AllocationConstraints constraints_;
    std::vector<Loop> loops_;
    AllocationScope scope_;
    AllocationContext context_;
};

}  // namespace test
}  // namespace stinkytofu
