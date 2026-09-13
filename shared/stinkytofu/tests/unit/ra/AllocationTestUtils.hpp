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

#include <array>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

#include "../ssa/AttachedSSATestUtils.hpp"
#include "TestHelpers.hpp"
#include "stinkytofu/analysis/asm/ssa/SSALiveIntervals.hpp"
#include "stinkytofu/hardware/AsmTargetRegisters.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"
#include "stinkytofu/support/LoopDetection.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationConstraints.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationRules.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationRulesRegistry.hpp"
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

/// The ISA triple of kRaTestArch, which is what the rules registry is keyed on.
inline std::array<int, 3> raTestTriple() {
    const auto* info = ArchHelper::getInstance().getArchInfo(kRaTestArch);
    return {static_cast<int>(info->major), static_cast<int>(info->minor),
            static_cast<int>(info->stepping)};
}

/// A one-row table forbidding odd bases in class V. Shared because more than one
/// suite needs *some* placement rule and none should depend on a shipped one.
inline AllocationRules evenVBasesOnly(RuleStatus status = RuleStatus::Active) {
    AllocationRule rule;
    rule.name = "EvenVBase";
    rule.description = "a V block must start even";
    rule.status = status;
    rule.forbidsBase = [](RegType regClass, uint32_t base, uint32_t) {
        return regClass == RegType::V && base % 2 != 0;
    };
    return AllocationRules({rule});
}

/// Registers \p rules for the test triple for the duration of a scope.
///
/// allocateRegisters resolves the table from the triple rather than taking it as
/// a parameter, so a test that wants a rule in force at the driver level has to
/// go through the registry. Below the driver, pass a table to AllocationSetup.
class ScopedArchRules {
   public:
    explicit ScopedArchRules(AllocationRules rules) {
        AllocationRulesRegistry::setArch(
            raTestTriple(), [table = std::move(rules)](const AsmCapsConfig&) { return table; });
    }

    ~ScopedArchRules() {
        AllocationRulesRegistry::clearArch(raTestTriple());
    }

    ScopedArchRules(const ScopedArchRules&) = delete;
    ScopedArchRules& operator=(const ScopedArchRules&) = delete;
};

/// Namespace scope so its default member initializers are usable in the default
/// argument below; a nested aggregate cannot be default-initialized while its
/// enclosing class is still incomplete.
struct AllocationRegionOptions {
    std::optional<SlotIndex> cut;
    AllocationScope::Containment containment = AllocationScope::Containment::ContainedIn;

    /// Registers to hold. Stated here rather than set afterwards because
    /// AllocationContext copies the scope, so a later change would not reach it.
    std::vector<AllocationScope::HeldRange> pinRegisters;
};

/// Owns the analyses an AllocationContext references.
class AllocationSetup {
   public:
    using RegionOptions = AllocationRegionOptions;

    /// \p rules defaults to an empty table. A test exercising a rule passes its
    /// own one-row table here rather than depending on a shipped per-arch table.
    explicit AllocationSetup(Function& function,
                             RegClassSet allocate = RegClassSet::only(RegType::V),
                             RegionOptions region = {}, AllocationRules rules = {})
        : rules_(std::move(rules)),
          intervals_(computeSSALiveIntervals(function)),
          target_(AsmTargetRegisters::forFunction(function)),
          constraints_(AllocationConstraints::build(function, target_, rules_)),
          loops_(detectLoops(function)),
          ruleIntervals_(applyEarlyClobber(function, intervals_, rules_)),
          scope_(buildScope(constraints_, ruleIntervals_, allocate, region)),
          context_{function, ruleIntervals_, target_, constraints_, loops_, rules_, scope_} {}

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

    /// The program's own ranges, as the shadow report sees them. Early-clobber
    /// widening lives in context().intervals instead, so pressure reported here
    /// stays a property of the program.
    const SSALiveIntervals& intervals() const {
        return intervals_;
    }

   private:
    static AllocationScope buildScope(const AllocationConstraints& constraints,
                                      const SSALiveIntervals& intervals, RegClassSet allocate,
                                      const RegionOptions& region) {
        AllocationScope scope =
            region.cut.has_value()
                ? AllocationScope::upTo(constraints, intervals, allocate, *region.cut,
                                        region.containment)
                : AllocationScope::wholeFunction(constraints, intervals, allocate);
        if (!region.pinRegisters.empty()) scope.pinRegisters(constraints, region.pinRegisters);
        return scope;
    }

    // Declared first: constraints_ and ruleIntervals_ are built against it.
    AllocationRules rules_;
    SSALiveIntervals intervals_;
    AsmTargetRegisters target_;
    AllocationConstraints constraints_;
    std::vector<Loop> loops_;
    // Mirrors allocateRegisters: the allocator and verifier read the widened
    // ranges, intervals_ stays the program's own.
    SSALiveIntervals ruleIntervals_;
    AllocationScope scope_;
    AllocationContext context_;
};

}  // namespace test
}  // namespace stinkytofu
