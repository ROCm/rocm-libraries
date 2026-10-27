// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/ra/RegisterAllocationPass.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
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
#include "stinkytofu/support/PassArguments.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationConstraints.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationRules.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationRulesRegistry.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationScope.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationVerifier.hpp"
#include "stinkytofu/transforms/asm/ra/AllocatorRegistry.hpp"
#include "stinkytofu/transforms/asm/ra/LegacyColoring.hpp"
#include "stinkytofu/transforms/asm/ra/RegisterAllocationConfig.hpp"
#include "stinkytofu/transforms/asm/ra/RegisterSymbolSync.hpp"
#include "stinkytofu/transforms/asm/ssa/SSADestruction.hpp"

#define DEBUG_TYPE "RegisterAllocationPass"

namespace stinkytofu {
namespace {

constexpr const char* kPassName = "RegisterAllocation";

/// The remark every "register allocation did not run" ends in, whoever decides it.
void missed(const PassContext& passCtx, const std::string& message) {
    emitRemark(passCtx, {OptimizationRemark::Kind::Missed, kPassName, "NotAllocated", message});
}

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

/// How many values carry an index ceiling, and the tightest one.
///
/// The Allocate counterpart of the held-range list: same constraint, but it
/// reaches the allocator as a limit per value rather than as frozen registers,
/// so there are no ranges to name and the report counts instead.
struct CappedValues {
    size_t count = 0;
    uint32_t ceiling = 0;
};

CappedValues cappedValuesOf(const Function& function, const AllocationConstraints& constraints) {
    constexpr uint32_t kNoLimit = std::numeric_limits<uint32_t>::max();
    CappedValues capped;
    const size_t valueCount = function.ssaArena().valueCount();
    for (size_t id = 1; id <= valueCount; ++id) {
        const uint32_t ceiling = constraints.maxIndexFor(static_cast<SSAValueID>(id));
        if (ceiling == kNoLimit) continue;
        if (capped.count == 0 || ceiling < capped.ceiling) capped.ceiling = ceiling;
        ++capped.count;
    }
    return capped;
}

/// Why a pairing rule did not get what it asked for.
///
/// A soft rule stays silent when it loses. A bare count therefore cannot say
/// whether the register was unavailable, or whether placement simply did not
/// take it. The two need different fixes -- shorter live ranges for the first,
/// better placement for the second -- and the final colouring tells them
/// apart.
///
/// The two counts differ in how sure they are. `blocked` is certain: a third
/// value sits on the register across this value's whole range, so no order of
/// placement could have paired them. `missed` is an upper bound: the register
/// was free for this one value, but a value moves together with the block it
/// is tied to, and this report does not see blocks. Read `missed` as how much
/// is worth investigating.
struct UnmetPreferences {
    size_t blocked = 0;  ///< A third value holds the partner's register.
    size_t missed = 0;   ///< It was free, and was not used.
};

/// Which values sit on each register. Built once, so the question above costs
/// a lookup instead of a scan over every value per preference.
///
/// One register per value, which is how the allocator binds them. The lift
/// makes one SSA value per DWORD, so a tuple is a run of values rather than
/// one wide value.
using RegOccupants = RegKeyMap<std::vector<SSAValueID>>;

RegOccupants occupantsOf(const AllocationResult& coloured) {
    RegOccupants occupants;
    for (SSAValueID id = 1; id <= coloured.valueCount(); ++id) {
        if (!coloured.isAssigned(id)) continue;
        occupants[coloured.assignmentOf(id)].push_back(id);
    }
    return occupants;
}

/// Could \p mover have taken \p reg, given where everything else ended up?
///
/// Asks what placement asks, in the same order: first whether the register is
/// off limits, then whether anything else is on it. \p partner is left out
/// because sharing with it is the whole point. Counting the partner as a
/// neighbour would report every pair as impossible.
bool couldHaveTaken(RegKey reg, SSAValueID mover, SSAValueID partner,
                    const SSALiveIntervals& intervals, const AllocationConstraints& constraints,
                    const AllocationScope& scope, const AsmTargetRegisters& target,
                    const AllocationRules& rules, const RegOccupants& occupants) {
    if (reg.idx > constraints.maxIndexFor(mover)) return false;
    if (!target.isAllocatable(reg.type, reg.idx)) return false;
    // A held register accepts only the value that was lifted from it, which is
    // the rule reachableAt applies during placement.
    if (scope.isPinnedRegister(reg.type, reg.idx)) {
        const std::optional<RegKey> hint = constraints.hintFor(mover);
        if (!hint.has_value() || *hint != reg) return false;
    }
    if (rules.forbidsBase(reg.type, reg.idx, /*width=*/1) != nullptr) return false;

    const auto found = occupants.find(reg);
    if (found == occupants.end()) return true;
    const LiveRange& range = intervals.rangeOf(mover);
    for (const SSAValueID resident : found->second) {
        if (resident == mover || resident == partner) continue;
        if (intervals.rangeOf(resident).overlaps(range)) return false;
    }
    return true;
}

/// How the colouring treated operands that cannot name a bank.
struct UnbankableChoice {
    /// The registers held, or empty when the operands were allocated.
    std::span<const AllocationScope::HeldRange> held;
    /// Auto chose the treatment by trying, rather than being told.
    bool automatic = false;
    /// Why allocating them was refused, when Auto went on to hold them.
    std::string allocateRefusal;
};

/// One line per kernel comparing a colouring against the producer's: what it
/// would cost, next to the pressure floor it could not go below.
std::string shadowReport(const Function& function, const AllocationResult& coloured,
                         const SSALiveIntervals& intervals,
                         const AllocationConstraints& constraints, const AllocationScope& scope,
                         const AllocationRules& rules, const AsmTargetRegisters& target,
                         const UnbankableChoice& unbankable, const char* allocator) {
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
    // How much of what each pairing rule asked for it actually got. A hard
    // rule announces itself by refusing. A soft rule says nothing, so without
    // this line a rule that collects nothing reads exactly like a rule that is
    // satisfied everywhere. Prints nothing when the chip declares no pairing
    // rule.
    const RegOccupants occupants =
        constraints.preferences().empty() ? RegOccupants{} : occupantsOf(coloured);
    for (size_t index = 0; index < rules.all().size(); ++index) {
        const AllocationRule& rule = rules.all()[index];
        if (rule.kind() != RuleKind::Pairing) continue;
        size_t offered = 0;
        size_t satisfied = 0;
        UnmetPreferences unmet;
        for (const Preference& preference : constraints.preferences()) {
            if (preference.rule != index) continue;
            ++offered;
            if (!coloured.isAssigned(preference.a) || !coloured.isAssigned(preference.b)) continue;
            const RegKey a = coloured.assignmentOf(preference.a);
            const RegKey b = coloured.assignmentOf(preference.b);
            if (rules.satisfiedBy(preference, a, b)) {
                ++satisfied;
                continue;
            }
            // Moving either end onto the other would have paired them. So both
            // directions must be blocked by a third value before this pair can
            // be called impossible.
            const bool reachable = couldHaveTaken(b, preference.a, preference.b, intervals,
                                                  constraints, scope, target, rules, occupants) ||
                                   couldHaveTaken(a, preference.b, preference.a, intervals,
                                                  constraints, scope, target, rules, occupants);
            if (reachable)
                ++unmet.missed;
            else
                ++unmet.blocked;
        }
        text += " pref[" + std::string(rule.name) + "=" + ruleStatusName(rule.status) + " " +
                std::to_string(satisfied) + "/" + std::to_string(offered);
        if (unmet.blocked != 0 || unmet.missed != 0) {
            text += " unmet " + std::to_string(unmet.blocked) + " blocked " +
                    std::to_string(unmet.missed) + " missed";
        }
        text += "]";
    }
    // The live-ins left unpinned. Named because moving them rests on nothing
    // having defined them, which holds only while lifting saw every definition.
    // Silent when there are none, like the rules above.
    //
    // A count and a bounded sample, not the list. A kernel lifted mid-stream
    // names hundreds of vector registers it never defines, and a line naming
    // every one is long enough that nobody reads any of it. The sample is for
    // spot-checking that the hints look like what the producer used; the count
    // is the figure worth watching.
    const std::span<const SSAValueID> undefined = constraints.undefinedLiveIns();
    if (!undefined.empty()) {
        constexpr size_t kSampleSize = 8;
        const size_t sampled = std::min(kSampleSize, undefined.size());
        text += " undefinedLiveIn[" + std::to_string(undefined.size());
        for (size_t i = 0; i < sampled; ++i) {
            text += " %" + std::to_string(undefined[i]);
            if (const std::optional<RegKey> hint = constraints.hintFor(undefined[i]))
                text += "=" + regKeyToString(*hint);
        }
        if (sampled != undefined.size())
            text += " +" + std::to_string(undefined.size() - sampled) + " more";
        text += "]";
    }
    // Narrow writes nothing reads, usually a missing RW. Sampled like the line above.
    const std::span<const SSAValueID> unread = constraints.unreadPartialWrites();
    if (!unread.empty()) {
        constexpr size_t kSampleSize = 8;
        const size_t sampled = std::min(kSampleSize, unread.size());
        text += " unreadPartialWrite[" + std::to_string(unread.size());
        for (size_t i = 0; i < sampled; ++i) {
            text += " %" + std::to_string(unread[i]);
            if (const std::optional<RegKey> hint = constraints.hintFor(unread[i]))
                text += "=" + regKeyToString(*hint);
        }
        if (sampled != unread.size())
            text += " +" + std::to_string(unread.size() - sampled) + " more";
        text += "]";
    }
    // What was done about operands that cannot name a bank. Reported in both
    // modes, because under Hold these registers are why the high-water mark
    // cannot fall below them, and under Allocate the constraint is still in
    // force even though nothing is frozen -- a silent report there would read
    // as no constraint at all. "auto" marks a treatment found by trying; the
    // refusal that sent Auto on to holding follows on a line of its own.
    const std::string chosen = unbankable.automatic ? "auto " : "";
    if (!unbankable.held.empty()) {
        text += " unbankable[" + chosen + "held";
        for (const AllocationScope::HeldRange& range : unbankable.held) {
            text += " " + regTypeToString(range.regClass) + std::to_string(range.start);
            if (range.end != range.start) text += ":" + std::to_string(range.end);
        }
        text += "]";
    } else if (const CappedValues capped = cappedValuesOf(function, constraints);
               capped.count > 0) {
        text += " unbankable[" + chosen + "allocated " + std::to_string(capped.count) +
                " value(s) max " + std::to_string(capped.ceiling) + "]";
    }
    if (!unbankable.allocateRefusal.empty())
        text +=
            "\n  held them because the allocate attempt was refused: " + unbankable.allocateRefusal;
    return text;
}

/// \p message without the "@function: " that every refusal starts with, for
/// quoting one refusal inside another.
std::string withoutFunctionName(const Function& function, const std::string& message) {
    const std::string prefix = "@" + function.getName() + ": ";
    return message.starts_with(prefix) ? message.substr(prefix.size()) : message;
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
    if (const std::vector<std::string> both = AllocationRules::contradictoryNames(options.rules);
        !both.empty()) {
        std::string names;
        for (const std::string& name : both) names += (names.empty() ? "" : ", ") + name;
        return Expected<AllocationRules>::Error(
            "@" + function.getName() +
            ": asked to activate and to disable the same allocation rule: " + names);
    }
    rules.force(options.rules);
    return rules;
}

/// Write \p allocated into the operands, the only step that changes the
/// function. A message when destruction refuses, nullopt otherwise.
std::optional<std::string> applyColouring(Function& function, const AllocationResult& allocated,
                                          const RegisterAllocationOptions& options,
                                          std::string* report) {
    const SSADestructionResult destroyed = destroyAttachedSSA(function, allocated);
    if (!destroyed.ok()) return destroyed.toString();
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
    return std::nullopt;
}

/// Why lowering cannot take \p allocator's colourings, or nullopt when it can.
std::optional<std::string> unsupportedCapability(const Function& function,
                                                 const RegisterAllocator& allocator) {
    const AllocatorCapabilities caps = allocator.capabilities();
    if (caps.maySpill) {
        return "@" + function.getName() + ": allocator '" + allocator.name() +
               "' requires spilling, which is not implemented";
    }
    if (caps.mayRecolourMerges) {
        return "@" + function.getName() + ": allocator '" + allocator.name() +
               "' may recolour merges, which needs copy insertion that is not implemented";
    }
    return std::nullopt;
}

/// The constraints, after stamping the live-ins the dispatch does not fill:
/// build() reads the stamp rather than work it out, and destruction has no
/// other way to learn it.
AllocationConstraints markAndBuild(Function& function, const AsmTargetRegisters& target,
                                   const AllocationRules& rules) {
    markUndefinedLiveIns(function, target);
    return AllocationConstraints::build(function, target, rules);
}

/// The classes and, with \p regionEnd, the slot prefix a run may relocate.
///
/// Scope reads the same ranges the allocator will, so "movable" cannot mean
/// something the allocator then finds it cannot place. Slot numbering is
/// untouched by early-clobber widening, so the cut means the same in either set.
AllocationScope regionScopeOf(const AllocationConstraints& constraints,
                              const SSALiveIntervals& intervals, RegClassSet classes,
                              const BasicBlock* regionEnd) {
    if (regionEnd == nullptr)
        return AllocationScope::wholeFunction(constraints, intervals, classes);
    return AllocationScope::upTo(constraints, intervals, classes,
                                 intervals.slots().blockEnd(regionEnd));
}

/// Everything colouring a function takes that no allocator changes. Built once,
/// so every allocator in a chain, and every attempt each one makes, shares it.
struct ColouringRun {
    ColouringRun(Function& func, const RegisterAllocationOptions& opts, AllocationRules resolved,
                 const BasicBlock* regionEnd)
        : function(func),
          options(opts),
          rules(std::move(resolved)),
          intervals(computeSSALiveIntervals(func)),
          target(AsmTargetRegisters::forFunction(func)),
          constraints(markAndBuild(func, target, rules)),
          loops(detectLoops(func)),
          ruleIntervals(applyEarlyClobber(func, intervals, rules)),
          regionScope(regionScopeOf(constraints, ruleIntervals, opts.allocate, regionEnd)),
          unbankable(opts.unbankableOperands ==
                             RegisterAllocationOptions::UnbankableOperands::Allocate
                         ? std::vector<AllocationScope::HeldRange>{}
                         : AllocationScope::unbankableOperandRegisters(func, target)) {}

    Function& function;
    const RegisterAllocationOptions& options;
    const AllocationRules rules;
    /// The program's own ranges, which the shadow report reads for its
    /// pressure floor.
    const SSALiveIntervals intervals;
    const AsmTargetRegisters target;
    const AllocationConstraints constraints;
    const std::vector<Loop> loops;
    /// The ranges with early-clobber widening, which the allocator and the
    /// verifier read. Identical to `intervals` while no clobbersEarly rule is
    /// Active.
    const SSALiveIntervals ruleIntervals;
    /// The scope every attempt starts from: classes and region, nothing held.
    const AllocationScope regionScope;
    /// Registers named by operand fields that cannot select a bank. Empty
    /// under Allocate, which never holds them.
    const std::vector<AllocationScope::HeldRange> unbankable;
};

/// The allocator-independent half of colouring \p function, or why no
/// allocator could colour it as asked.
Expected<std::unique_ptr<ColouringRun>> prepareColouring(Function& function,
                                                         const RegisterAllocationOptions& options) {
    using Result = Expected<std::unique_ptr<ColouringRun>>;
    if (!function.hasAttachedSSA())
        return Result::Error("@" + function.getName() + ": no attached SSA; nothing to colour");

    // Colouring a class the lift left physical would produce assignments for
    // values that do not exist, so the request is checked rather than quietly
    // reduced to whatever happens to be there.
    if (const std::optional<std::string> classError =
            AllocationScope::validateClasses(function, options.allocate))
        return Result::Error(*classError);

    // An unregistered triple is an empty table, so every query there answers
    // "no opinion".
    Expected<AllocationRules> rules = resolveRules(function, options);
    if (rules.hasError()) return Result::Error(rules.getError());

    const BasicBlock* regionEnd = nullptr;
    if (!options.regionEnd.empty()) {
        regionEnd = findBlockByLabel(function, options.regionEnd);
        if (regionEnd == nullptr) {
            return Result::Error("@" + function.getName() + ": region end block '" +
                                 options.regionEnd + "' was not found");
        }
    }

    // A backwards pair holds nothing, which reads as "holding made no
    // difference". Hold nothing by passing nothing instead.
    for (const AllocationScope::HeldRange& range : options.pinRegisters) {
        if (range.end < range.start) {
            return Result::Error(
                "@" + function.getName() + ": asked to hold " + regTypeToString(range.regClass) +
                std::to_string(range.start) + " through " + regTypeToString(range.regClass) +
                std::to_string(range.end) +
                ", which is empty because the bounds are inclusive and run backwards");
        }
    }
    return std::make_unique<ColouringRun>(function, options, std::move(*rules), regionEnd);
}

/// Colour with \p allocator, trying each treatment of the operands that cannot
/// name a bank that the options ask for, then report and apply the first
/// colouring that succeeds. Nothing is written unless one does.
Expected<AllocationResult> colourWith(const ColouringRun& run, RegisterAllocator& allocator,
                                      std::string* report) {
    Function& function = run.function;
    const RegisterAllocationOptions& options = run.options;
    if (std::optional<std::string> problem = unsupportedCapability(function, allocator))
        return Expected<AllocationResult>::Error(*problem);

    // Which treatments to try, in order. Auto tries Allocate, then Hold,
    // because each refuses kernels the other colours. Holding differs from
    // allocating only when such an operand exists, so without one Auto is a
    // single attempt. The ceiling is collected either way; Allocate places
    // under it and Hold leaves it nothing to move.
    using Unbankable = RegisterAllocationOptions::UnbankableOperands;
    const bool automatic = options.unbankableOperands == Unbankable::Auto;
    std::vector<Unbankable> attempts{options.unbankableOperands};
    if (automatic) {
        attempts = {Unbankable::Allocate};
        if (!run.unbankable.empty()) attempts.push_back(Unbankable::Hold);
    }

    std::vector<std::string> refusals;
    for (const Unbankable attempt : attempts) {
        const std::span<const AllocationScope::HeldRange> held =
            attempt == Unbankable::Hold
                ? std::span<const AllocationScope::HeldRange>(run.unbankable)
                : std::span<const AllocationScope::HeldRange>{};
        // Held before the requested registers, so a register in both reports
        // the constraint rather than the request.
        AllocationScope scope = run.regionScope;
        if (!held.empty()) scope.holdUnbankableOperands(run.constraints, held);
        if (!options.pinRegisters.empty())
            scope.pinRegisters(run.constraints, options.pinRegisters);

        AllocationContext context{function,  run.ruleIntervals, run.target, run.constraints,
                                  run.loops, run.rules,         scope};
        context.preferProducerQuadrant =
            options.quadrant == RegisterAllocationOptions::Quadrant::Prefer;
        if (!options.blockDumpPrefix.empty()) {
            context.blockDumpPath = options.blockDumpPrefix + allocator.name() + "." +
                                    unbankableOperandsName(attempt) + ".tsv";
        }

        Expected<AllocationResult> allocated = allocator.allocate(context);
        if (allocated.hasValue() && options.verify) {
            const AllocationVerificationResult checked =
                verifyAllocation(function, *allocated, context);
            if (!checked.ok()) allocated = Expected<AllocationResult>::Error(checked.toString());
        }
        if (allocated.hasError()) {
            refusals.push_back(allocated.getError());
            continue;
        }

        // Before destruction, which clears the attached SSA the report reads.
        if (options.report && report != nullptr) {
            UnbankableChoice choice{held, automatic, {}};
            if (!refusals.empty())
                choice.allocateRefusal = withoutFunctionName(function, refusals.front());
            *report = shadowReport(function, *allocated, run.intervals, run.constraints, scope,
                                   run.rules, run.target, choice, allocator.name());
        }
        if (options.applyToOperands) {
            if (std::optional<std::string> problem =
                    applyColouring(function, *allocated, options, report))
                return Expected<AllocationResult>::Error(*problem);
        }
        return allocated;
    }

    if (refusals.size() == 1) return Expected<AllocationResult>::Error(refusals.front());
    std::string message = "@" + function.getName() +
                          ": refused whether the operands that cannot name a bank are allocated "
                          "or held";
    for (size_t i = 0; i < refusals.size(); ++i) {
        message += "\n  " + std::string(unbankableOperandsName(attempts[i])) + ": " +
                   withoutFunctionName(function, refusals[i]);
    }
    return Expected<AllocationResult>::Error(message);
}

}  // namespace

Expected<AllocationResult> allocateRegisters(Function& function, RegisterAllocator& allocator,
                                             const RegisterAllocationOptions& options,
                                             std::string* report) {
    Expected<std::unique_ptr<ColouringRun>> run = prepareColouring(function, options);
    if (run.hasError()) return Expected<AllocationResult>::Error(run.getError());
    return colourWith(**run, allocator, report);
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

        // The allocators to try, in order: the injected one, or every name in
        // the options. A chain is tried until one colours the function, and
        // only that one's colouring is applied.
        std::vector<std::unique_ptr<RegisterAllocator>> created;
        std::vector<RegisterAllocator*> chain;
        if (allocator_ != nullptr) {
            chain.push_back(allocator_.get());
        } else {
            for (const std::string& name : splitList(options_.allocator, '+')) {
                std::unique_ptr<RegisterAllocator> allocator =
                    AllocatorRegistry::createAllocator(name);
                if (allocator == nullptr) {
                    missed(passCtx,
                           "@" + func.getName() + ": allocator '" + name + "' is not registered");
                    return preserveCFGAnalyses();
                }
                chain.push_back(allocator.get());
                created.push_back(std::move(allocator));
            }
            if (chain.empty()) {
                missed(passCtx, "@" + func.getName() + ": no allocator named");
                return preserveCFGAnalyses();
            }
        }

        // The free driver takes only options, so capabilities reach the rules
        // through them. Only the pass has a PassContext to read them from.
        RegisterAllocationOptions options = options_;
        options.caps = passCtx.getAsmCapsConfig();

        emitRuleAudit(func, passCtx, options);

        Expected<std::unique_ptr<ColouringRun>> run = prepareColouring(func, options);
        if (run.hasError()) {
            PASS_DEBUG(std::cerr << "RegisterAllocation: " << run.getError() << "\n");
            missed(passCtx, run.getError());
            return preserveCFGAnalyses();
        }

        std::string report;
        std::vector<std::string> refusals;
        std::optional<AllocationResult> coloured;
        RegisterAllocator* winner = nullptr;
        for (RegisterAllocator* allocator : chain) {
            Expected<AllocationResult> result = colourWith(**run, *allocator, &report);
            if (result.hasError()) {
                PASS_DEBUG(std::cerr << "RegisterAllocation: " << result.getError() << "\n");
                refusals.push_back(result.getError());
                continue;
            }
            coloured.emplace(std::move(*result));
            winner = allocator;
            break;
        }
        if (winner == nullptr) {
            std::string message = refusals.front();
            if (refusals.size() > 1) {
                message = "@" + func.getName() + ": every allocator in " + options_.allocator +
                          " refused";
                for (size_t i = 0; i < refusals.size(); ++i) {
                    message += "\n  " + std::string(chain[i]->name()) + ": " +
                               withoutFunctionName(func, refusals[i]);
                }
            }
            missed(passCtx, message);
            return preserveCFGAnalyses();
        }
        if (!report.empty()) {
            emitRemark(passCtx,
                       {OptimizationRemark::Kind::Analysis, kPassName, "ShadowReport", report});
        }

        std::string summary = "@" + func.getName() + ": coloured " +
                              std::to_string(coloured->valueCount()) + " value(s) with " +
                              winner->name();
        if (!refusals.empty()) {
            summary += " after";
            for (size_t i = 0; i < refusals.size(); ++i)
                summary += std::string(i == 0 ? " " : ", ") + chain[i]->name();
            summary += " refused";
        }
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

class RegisterAllocationNotRunPassImpl : public Pass {
   public:
    static char ID;

    explicit RegisterAllocationNotRunPassImpl(std::string reason) : reason_(std::move(reason)) {}

    const char* getName() const override {
        return "Register Allocation Not Run";
    }

    PassID getPassID() const override {
        return &RegisterAllocationNotRunPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager&) override {
        missed(passCtx, "@" + func.getName() + ": " + reason_);
        return PreservedAnalyses::all();
    }

   private:
    std::string reason_;
};

char RegisterAllocationNotRunPassImpl::ID = 0;

std::unique_ptr<Pass> createRegisterAllocationPass(RegisterAllocationOptions options,
                                                   std::unique_ptr<RegisterAllocator> allocator) {
    return std::make_unique<RegisterAllocationPassImpl>(std::move(options), std::move(allocator));
}

std::unique_ptr<Pass> createRegisterAllocationNotRunPass(std::string reason) {
    return std::make_unique<RegisterAllocationNotRunPassImpl>(std::move(reason));
}

}  // namespace stinkytofu
