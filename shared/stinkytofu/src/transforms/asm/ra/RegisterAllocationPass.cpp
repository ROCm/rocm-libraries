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
#include "stinkytofu/transforms/asm/ra/RegisterAllocationPass.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/asm/ssa/SSALiveIntervals.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/AsmTargetRegisters.hpp"
#include "stinkytofu/ir/asm/StinkyRegister.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"
#include "stinkytofu/support/LoopDetection.hpp"
#include "stinkytofu/support/OptimizationRemark.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationConstraints.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationRules.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationRulesRegistry.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationScope.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationVerifier.hpp"
#include "stinkytofu/transforms/asm/ra/AllocatorRegistry.hpp"
#include "stinkytofu/transforms/asm/ra/LegacyColoring.hpp"
#include "stinkytofu/transforms/asm/ra/RegisterSymbolSync.hpp"
#include "stinkytofu/transforms/asm/ssa/SSADestruction.hpp"

#define DEBUG_TYPE "RegisterAllocationPass"

namespace stinkytofu {
namespace {

constexpr const char* kPassName = "RegisterAllocation";

const BasicBlock* findExcludedBlock(const Function& func, const PassContext& passCtx) {
    for (const BasicBlock& bb : func) {
        if (!passCtx.shouldProcessBasicBlock(bb)) return &bb;
    }
    return nullptr;
}

const BasicBlock* findBlockByLabel(const Function& function, std::string label) {
    if (!label.empty() && label.front() == '^') label.erase(label.begin());
    for (const BasicBlock& bb : function) {
        if (bb.getLabel() == label) return &bb;
    }
    return nullptr;
}

/// Peak simultaneously live DWORDs in \p regClass over slots [0, \p cut).
uint32_t peakPressureUpTo(const Function& function, const SSALiveIntervals& intervals,
                          const AllocationConstraints& constraints, RegType regClass,
                          SlotIndex cut) {
    if (cut == 0) return 0;
    std::vector<int32_t> delta(cut + 1, 0);
    for (size_t id = 1; id <= intervals.valueCount(); ++id) {
        if (constraints.classOf(static_cast<SSAValueID>(id)) != regClass) continue;
        const StinkySSAValue* value = function.ssaArena().get(static_cast<SSAValueID>(id));
        const uint32_t width =
            (value == nullptr || value->type().dwordWidth == 0) ? 1u : value->type().dwordWidth;
        const LiveRange& range = intervals.rangeOf(static_cast<SSAValueID>(id));
        for (const LiveSegment& segment : range.segments()) {
            if (segment.start >= cut) continue;
            delta[segment.start] += static_cast<int32_t>(width);
            const SlotIndex end = std::min(segment.end, cut);
            delta[end] -= static_cast<int32_t>(width);
        }
    }
    int32_t live = 0;
    int32_t peak = 0;
    for (SlotIndex slot = 0; slot < cut; ++slot) {
        live += delta[slot];
        peak = std::max(peak, live);
    }
    return static_cast<uint32_t>(peak);
}

/// Highest index \p result hands out in \p regClass, or 0 when it uses none.
/// One past this is the count a resource descriptor would have to declare.
uint32_t highestIndex(const AllocationResult& result, RegType regClass) {
    uint32_t highest = 0;
    for (SSAValueID id = 1; id <= result.valueCount(); ++id) {
        if (!result.isAssigned(id)) continue;
        const RegKey physical = result.assignmentOf(id);
        if (physical.type == regClass) highest = std::max(highest, physical.idx);
    }
    return highest;
}

std::string wavesOf(GfxArchID arch, uint32_t vgprs) {
    const int waves = getWavesPerSimd(arch, static_cast<int>(vgprs));
    return waves == std::numeric_limits<int>::max() ? "n/a" : std::to_string(waves);
}

/// One line per kernel comparing a colouring against the producer's: what it
/// would cost, next to the pressure floor it could not go below.
std::string shadowReport(const Function& function, const AllocationResult& coloured,
                         const SSALiveIntervals& intervals,
                         const AllocationConstraints& constraints, const AllocationScope& scope,
                         const AllocationRules& rules, const char* allocator) {
    const AllocationResult producer = createLegacyColoring(function);
    const std::array<int, 3>& isa = function.getGemmTileConfig().arch;
    const GfxArchID arch =
        getGfxArchID(static_cast<uint32_t>(isa[0]), static_cast<uint32_t>(isa[1]),
                     static_cast<uint32_t>(isa[2]));

    std::string text = "@" + function.getName() + ": " + allocator +
                       " shadow: values=" + std::to_string(coloured.valueCount());
    for (const auto& [regClass, peak] : intervals.peakPressure()) {
        const uint32_t was = highestIndex(producer, regClass);
        const uint32_t now = highestIndex(coloured, regClass);
        text += " " + regTypeToString(regClass) + "[peak=" + std::to_string(peak) +
                " highest=" + std::to_string(was) + "->" + std::to_string(now);
        if (const std::optional<SlotIndex> cut = scope.regionCut()) {
            text += " regionPeak=" + std::to_string(peakPressureUpTo(function, intervals,
                                                                     constraints, regClass, *cut));
        }
        // Occupancy is a VGPR question on this target, and it moves in granule
        // steps, so a lower index need not buy a wave.
        if (regClass == RegType::V)
            text += " waves=" + wavesOf(arch, was + 1) + "->" + wavesOf(arch, now + 1);
        text += "]";
    }
    // Which rules were in force, so a report can be read without knowing the
    // triple or how capabilities were configured. Silent when there are none,
    // which keeps every existing report byte-identical.
    for (const AllocationRule& rule : rules.all()) {
        text += " rule[" + std::string(rule.name) + "=" + ruleStatusName(rule.status) + "]";
    }
    return text;
}

/// The architecture's rules with any forced statuses applied, or a message
/// naming what was wrong with the request.
///
/// One place builds the table, so the pass and the driver cannot disagree about
/// which rules are in force.
Expected<AllocationRules> resolveRules(const Function& function,
                                       const RegisterAllocationOptions& options) {
    AllocationRules rules =
        AllocationRulesRegistry::forArch(function.getGemmTileConfig().arch, options.caps);

    // A malformed row is a mistake in the architecture's table, not a property
    // of the chip, and it silently disables the rule -- so it is refused rather
    // than dropped quietly.
    if (!rules.problems().empty()) {
        std::string message = "@" + function.getName() + ": malformed allocation rule table:";
        for (const std::string& problem : rules.problems()) message += "\n  " + problem;
        return Expected<AllocationRules>::Error(message);
    }
    if (const std::vector<std::string> unknown = rules.unknownNames(options.rules);
        !unknown.empty()) {
        std::string names;
        for (const std::string& name : unknown) names += (names.empty() ? "" : ", ") + name;
        return Expected<AllocationRules>::Error("@" + function.getName() +
                                                ": no such allocation rule: " + names);
    }
    rules.force(options.rules);
    return rules;
}

}  // namespace

Expected<AllocationResult> allocateRegisters(Function& function, RegisterAllocator& allocator,
                                             const RegisterAllocationOptions& options,
                                             std::string* report) {
    if (!function.hasAttachedSSA()) {
        return Expected<AllocationResult>::Error("@" + function.getName() +
                                                 ": no attached SSA; nothing to colour");
    }

    const AllocatorCapabilities caps = allocator.capabilities();
    if (caps.maySpill) {
        return Expected<AllocationResult>::Error("@" + function.getName() + ": allocator '" +
                                                 allocator.name() +
                                                 "' requires spilling, which is not implemented");
    }
    if (caps.mayRecolourMerges) {
        return Expected<AllocationResult>::Error(
            "@" + function.getName() + ": allocator '" + allocator.name() +
            "' may recolour merges, which needs copy insertion that is not implemented");
    }

    // Colouring a class the lift left physical would produce assignments for
    // values that do not exist, so the request is checked rather than quietly
    // reduced to whatever happens to be there.
    if (const std::optional<std::string> classError =
            AllocationScope::validateClasses(function, options.allocate)) {
        return Expected<AllocationResult>::Error(*classError);
    }

    // An unregistered triple is an empty table, so every query there answers
    // "no opinion".
    Expected<AllocationRules> resolved = resolveRules(function, options);
    if (resolved.hasError()) return Expected<AllocationResult>::Error(resolved.getError());
    const AllocationRules rules = std::move(*resolved);

    const SSALiveIntervals intervals = computeSSALiveIntervals(function);
    AsmTargetRegisters target = AsmTargetRegisters::forFunction(function);
    const AllocationConstraints constraints = AllocationConstraints::build(function, target, rules);
    const std::vector<Loop> loops = detectLoops(function);

    // Two sets of ranges on purpose: the allocator and the verifier want the
    // early-clobber widening, while the shadow report wants the program's own
    // pressure floor, which widening would inflate. Identical while no
    // clobbersEarly rule is Active.
    const SSALiveIntervals ruleIntervals = applyEarlyClobber(function, intervals, rules);

    // Scope reads the same ranges the allocator will, so "movable" cannot mean
    // something the allocator then finds it cannot place. Slot numbering is
    // untouched by widening, so the cut below means the same in either set.
    AllocationScope scope =
        AllocationScope::wholeFunction(constraints, ruleIntervals, options.allocate);
    if (!options.regionEnd.empty()) {
        const BasicBlock* endBlock = findBlockByLabel(function, options.regionEnd);
        if (endBlock == nullptr) {
            return Expected<AllocationResult>::Error("@" + function.getName() +
                                                     ": region end block '" + options.regionEnd +
                                                     "' was not found");
        }
        const SlotIndex cut = ruleIntervals.slots().blockEnd(endBlock);
        scope = AllocationScope::upTo(constraints, ruleIntervals, options.allocate, cut);
    }

    if (!options.pinRegisters.empty()) {
        // A backwards pair holds nothing, which reads as "holding made no
        // difference". Hold nothing by passing nothing instead.
        for (const AllocationScope::HeldRange& range : options.pinRegisters) {
            if (range.end < range.start) {
                return Expected<AllocationResult>::Error(
                    "@" + function.getName() + ": asked to hold " +
                    regTypeToString(range.regClass) + std::to_string(range.start) + " through " +
                    regTypeToString(range.regClass) + std::to_string(range.end) +
                    ", which is empty because the bounds are inclusive and run backwards");
            }
        }
        scope.pinRegisters(constraints, options.pinRegisters);
    }

    const AllocationContext context{function, ruleIntervals, target, constraints,
                                    loops,    rules,         scope};

    Expected<AllocationResult> allocated = allocator.allocate(context);
    if (allocated.hasError()) return allocated;

    if (options.verify) {
        const AllocationVerificationResult checked =
            verifyAllocation(function, *allocated, context);
        if (!checked.ok()) {
            return Expected<AllocationResult>::Error(checked.toString());
        }
    }

    // Before destruction, which clears the attached SSA the report reads.
    if (options.report && report != nullptr) {
        *report = shadowReport(function, *allocated, intervals, constraints, scope, rules,
                               allocator.name());
    }

    if (options.applyToOperands) {
        const SSADestructionResult destroyed = destroyAttachedSSA(function, *allocated);
        if (!destroyed.ok()) {
            return Expected<AllocationResult>::Error(destroyed.toString());
        }
        SymbolSyncOptions syncOptions;
        syncOptions.emitRegisterMap = options.emitRegisterMap;
        syncOptions.emitBreadcrumbs = options.emitSymbolBreadcrumbs;
        SymbolSyncReport syncReport;
        syncRegisterSymbols(function, destroyed.rewritten, syncOptions, &syncReport);

        // An operand whose name never described where it sat is the shape of a
        // wrong-register bug: the emitter prints the name, the allocator reasoned
        // about the index, and nothing else reconciles them. Reported whatever
        // `report` asks for, because it is a correctness warning rather than a
        // statistic -- silence here is how a stale name reaches the assembler.
        if (!syncReport.suspectOperands.empty() && report != nullptr) {
            if (!report->empty()) *report += "\n";
            *report += "symbol sync: " + std::to_string(syncReport.suspectOperands.size()) +
                       " operand(s) whose symbolic name disagrees with their register";
            for (const std::string& suspect : syncReport.suspectOperands) {
                *report += "\n  " + suspect;
            }
        }
    }

    return allocated;
}

class RegisterAllocationPassImpl : public Pass {
   public:
    static char ID;

    RegisterAllocationPassImpl(RegisterAllocationOptions options,
                               std::unique_ptr<RegisterAllocator> allocator)
        : options_(std::move(options)), allocator_(std::move(allocator)) {}

    const char* getName() const override {
        return "Register Allocation";
    }

    PassID getPassID() const override {
        return &RegisterAllocationPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager&) override {
        if (const BasicBlock* excluded = findExcludedBlock(func, passCtx)) {
            missed(passCtx, "@" + func.getName() + ": basic-block filtering excludes ^" +
                                excluded->getLabel() +
                                "; register allocation needs the whole "
                                "function");
            return preserveCFGAnalyses();
        }

        if (!func.hasAttachedSSA()) {
            missed(passCtx, "@" + func.getName() + ": no attached SSA; nothing to colour");
            return preserveCFGAnalyses();
        }

        if (allocator_ == nullptr) {
            allocator_ = AllocatorRegistry::createAllocator(options_.allocator);
            if (allocator_ == nullptr) {
                missed(passCtx, "@" + func.getName() + ": allocator '" + options_.allocator +
                                    "' is not registered");
                return preserveCFGAnalyses();
            }
        }

        // The free driver takes only options, so capabilities reach the rules
        // through them. Only the pass has a PassContext to read them from.
        RegisterAllocationOptions options = options_;
        options.caps = passCtx.getAsmCapsConfig();

        emitRuleAudit(func, passCtx, options);

        std::string report;
        Expected<AllocationResult> result = allocateRegisters(func, *allocator_, options, &report);
        if (result.hasError()) {
            PASS_DEBUG(std::cerr << "RegisterAllocation: " << result.getError() << "\n");
            missed(passCtx, result.getError());
            return preserveCFGAnalyses();
        }
        if (!report.empty()) {
            emitRemark(passCtx,
                       {OptimizationRemark::Kind::Analysis, kPassName, "ShadowReport", report});
        }

        const std::string summary = "@" + func.getName() + ": coloured " +
                                    std::to_string(result->valueCount()) + " value(s) with " +
                                    allocator_->name();
        if (options_.applyToOperands) {
            emitRemark(passCtx, {OptimizationRemark::Kind::Passed, kPassName, "AllocatedRegisters",
                                 summary});
        } else {
            emitRemark(passCtx, {OptimizationRemark::Kind::Analysis, kPassName, "ShadowColoring",
                                 summary + " (shadow, not applied)"});
        }
        return preserveCFGAnalyses();
    }

   private:
    static void missed(const PassContext& passCtx, const std::string& message) {
        emitRemark(passCtx, {OptimizationRemark::Kind::Missed, kPassName, "NotAllocated", message});
    }

    /// Report where the producer's own colouring already breaks a rule.
    ///
    /// Lives in the pass rather than in allocateRegisters because emitting a
    /// remark needs a PassContext, which the free driver does not have. Runs
    /// against the producer's registers, so a finding is about the *input*: if
    /// this fires on a corpus the rule is not ready to be made Active, since
    /// verifyAllocation would then refuse those kernels outright.
    static void emitRuleAudit(const Function& func, const PassContext& passCtx,
                              const RegisterAllocationOptions& options) {
        if (!passCtx.getRemarksEnabled()) return;

        // A bad request is reported by allocateRegisters a moment later, so this
        // stays quiet rather than duplicating the diagnostic.
        Expected<AllocationRules> resolved = resolveRules(func, options);
        if (resolved.hasError() || resolved->empty()) return;

        for (const std::string& finding : auditRules(func, createLegacyColoring(func), *resolved)) {
            emitRemark(passCtx, {OptimizationRemark::Kind::Analysis, kPassName, "RuleAudit",
                                 finding + "; producer colouring already violates this rule"});
        }
    }

    RegisterAllocationOptions options_;
    std::unique_ptr<RegisterAllocator> allocator_;
};

char RegisterAllocationPassImpl::ID = 0;

std::unique_ptr<Pass> createRegisterAllocationPass(RegisterAllocationOptions options,
                                                   std::unique_ptr<RegisterAllocator> allocator) {
    return std::make_unique<RegisterAllocationPassImpl>(std::move(options), std::move(allocator));
}

}  // namespace stinkytofu
