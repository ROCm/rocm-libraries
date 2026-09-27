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
#include "stinkytofu/transforms/asm/ra/AllocationRules.hpp"

#include <algorithm>
#include <sstream>
#include <utility>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace stinkytofu {
namespace {

std::string valueName(SSAValueID id) {
    return "%" + std::to_string(id);
}

std::string mnemonicOf(const StinkyInstruction& instruction) {
    const HwInstDesc* desc = instruction.getHwInstDesc();
    if (desc == nullptr || desc->mnemonic == nullptr) return "an instruction";
    return desc->mnemonic;
}

/// Every SSA result of \p instruction, skipping holes.
void forEachResult(const StinkyInstruction& instruction, auto&& fn) {
    for (size_t result = 0; result < instruction.getNumSSAResults(); ++result) {
        const StinkySSAValue* value = instruction.getSSAResult(result);
        if (value != nullptr) fn(value->valueId());
    }
}

/// Every SSA operand of \p instruction, skipping holes.
void forEachOperand(const StinkyInstruction& instruction, auto&& fn) {
    for (size_t operand = 0; operand < instruction.getNumSSAOperands(); ++operand) {
        const StinkySSAValue* value = instruction.getSSAOperandValue(operand);
        if (value != nullptr) fn(value->valueId());
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// AllocationRule
// ---------------------------------------------------------------------------

RuleKind AllocationRule::kind() const {
    const int filled = static_cast<int>(static_cast<bool>(forbidsBase)) +
                       static_cast<int>(static_cast<bool>(clobbersEarly)) +
                       static_cast<int>(static_cast<bool>(addRelations)) +
                       static_cast<int>(static_cast<bool>(baseCost));
    if (filled != 1) return RuleKind::Empty;
    if (forbidsBase) return RuleKind::Placement;
    if (clobbersEarly) return RuleKind::Interference;
    if (addRelations) return RuleKind::Offset;
    return RuleKind::Preference;
}

const char* ruleStatusName(RuleStatus status) {
    switch (status) {
        case RuleStatus::Off:
            return "off";
        case RuleStatus::Audit:
            return "audit";
        case RuleStatus::Active:
            return "active";
    }
    return "off";
}

const char* ruleKindName(RuleKind kind) {
    switch (kind) {
        case RuleKind::Empty:
            return "empty";
        case RuleKind::Placement:
            return "placement";
        case RuleKind::Interference:
            return "interference";
        case RuleKind::Offset:
            return "offset";
        case RuleKind::Preference:
            return "preference";
    }
    return "empty";
}

// ---------------------------------------------------------------------------
// AllocationRules
// ---------------------------------------------------------------------------

AllocationRules::AllocationRules(std::vector<AllocationRule> rules) : rules_(std::move(rules)) {
    // A row that fills in no function, or more than one, is a table mistake
    // rather than a hardware fact. Dropping and naming it beats asserting: the
    // rest of the chip's rules still work, and the report says what went wrong.
    std::vector<AllocationRule> kept;
    kept.reserve(rules_.size());
    for (AllocationRule& rule : rules_) {
        if (rule.kind() == RuleKind::Empty) {
            problems_.push_back("rule " + std::string(rule.name) +
                                " fills in none or several of forbidsBase, clobbersEarly, "
                                "addRelations, baseCost; exactly one is required");
            continue;
        }
        if (rule.kind() == RuleKind::Preference && rule.status == RuleStatus::Audit) {
            // Audit asks "does the input already violate this", which presumes a
            // violation. A preference has none, so Audit here would be silently
            // inert -- how a rule ends up believed-enabled and doing nothing.
            problems_.push_back("rule " + std::string(rule.name) +
                                " is a preference, which has no Audit state");
            rule.status = RuleStatus::Off;
        }
        kept.push_back(std::move(rule));
    }
    rules_ = std::move(kept);
    refresh();
}

void AllocationRules::refresh() {
    prices_ = std::any_of(rules_.begin(), rules_.end(), [](const AllocationRule& rule) {
        return rule.status == RuleStatus::Active && static_cast<bool>(rule.baseCost);
    });
}

const AllocationRule* AllocationRules::forbidsBase(RegType regClass, uint32_t base,
                                                   uint32_t width) const {
    for (const AllocationRule& rule : rules_) {
        if (rule.status != RuleStatus::Active || !rule.forbidsBase) continue;
        if (rule.forbidsBase(regClass, base, width)) return &rule;
    }
    return nullptr;
}

const AllocationRule* AllocationRules::clobbersEarly(const StinkyInstruction& inst) const {
    for (const AllocationRule& rule : rules_) {
        if (rule.status != RuleStatus::Active || !rule.clobbersEarly) continue;
        if (rule.clobbersEarly(inst)) return &rule;
    }
    return nullptr;
}

void AllocationRules::addRelations(const Function& function, std::vector<TupleRun>& tupleRuns,
                                   std::vector<AffinitySet>& affinitySets) const {
    for (const AllocationRule& rule : rules_) {
        if (rule.status != RuleStatus::Active || !rule.addRelations) continue;
        rule.addRelations(function, tupleRuns, affinitySets);
    }
}

double AllocationRules::baseCost(RegType regClass, uint32_t base, uint32_t width) const {
    double total = 0.0;
    for (const AllocationRule& rule : rules_) {
        if (rule.status != RuleStatus::Active || !rule.baseCost) continue;
        total += rule.baseCost(regClass, base, width);
    }
    return total;
}

std::vector<std::string> AllocationRules::unknownNames(const RuleOverrides& overrides) const {
    std::vector<std::string> unknown;
    for (const std::string& wanted : overrides.activate) {
        const bool found =
            std::any_of(rules_.begin(), rules_.end(),
                        [&wanted](const AllocationRule& rule) { return rule.name == wanted; });
        if (!found) unknown.push_back(wanted);
    }
    return unknown;
}

void AllocationRules::force(const RuleOverrides& overrides) {
    if (overrides.disableAll) {
        rules_.clear();
        refresh();
        return;
    }
    for (AllocationRule& rule : rules_) {
        const bool named = std::find(overrides.activate.begin(), overrides.activate.end(),
                                     rule.name) != overrides.activate.end();
        if (overrides.activateAll || named) {
            rule.status = RuleStatus::Active;
        } else if (overrides.auditAll && rule.kind() != RuleKind::Preference) {
            rule.status = RuleStatus::Audit;
        }
    }
    refresh();
}

std::string AllocationRules::toString() const {
    std::ostringstream out;
    out << "rules=" << rules_.size() << '\n';
    for (const AllocationRule& rule : rules_) {
        out << rule.name << ' ' << ruleKindName(rule.kind()) << ' ' << ruleStatusName(rule.status)
            << ": " << rule.description << '\n';
    }
    for (const std::string& problem : problems_) out << "problem: " << problem << '\n';
    return out.str();
}

// ---------------------------------------------------------------------------
// Applying rules
// ---------------------------------------------------------------------------

SSALiveIntervals applyEarlyClobber(const Function& function, const SSALiveIntervals& base,
                                   const AllocationRules& rules) {
    if (!function.hasAttachedSSA() || rules.empty()) return base;

    const SSASlotIndexes& slots = base.slots();
    std::vector<std::pair<SSAValueID, SlotIndex>> earliest;

    for (const BasicBlock& block : function) {
        for (const IRBase& ir : block) {
            const auto* instruction = dyn_cast<StinkyInstruction>(&ir);
            if (instruction == nullptr || !instruction->hasAttachedSSA()) continue;
            if (rules.clobbersEarly(*instruction) == nullptr) continue;

            const SlotIndex use = slots.useSlot(instruction);
            if (use == SSASlotIndexes::kInvalidSlot) continue;
            forEachResult(*instruction, [&](SSAValueID id) { earliest.emplace_back(id, use); });
        }
    }

    return SSALiveIntervals::withEarlierStarts(base, earliest);
}

std::vector<std::string> auditRules(const Function& function, const AllocationResult& coloring,
                                    const AllocationRules& rules) {
    std::vector<std::string> findings;
    if (!function.hasAttachedSSA() || rules.empty()) return findings;

    const std::string prefix = "@" + function.getName() + ": rule ";
    auto report = [&](const AllocationRule& rule, const std::string& detail) {
        findings.push_back(prefix + std::string(rule.name) + ": " + detail + " (" +
                           std::string(rule.description) + ")");
    };

    // Audit ignores status on purpose: the question is whether the input already
    // breaks a rule, which is exactly what you want answered *before* turning
    // it on. Each rule is evaluated separately, so one Off rule cannot mask
    // another's finding.
    for (const AllocationRule& rule : rules.all()) {
        switch (rule.kind()) {
            case RuleKind::Placement: {
                // Per value, at its own width: the blocks a placement rule really
                // sees are built by the allocator, and a colouring alone cannot
                // show which values it would have tied together.
                for (StinkySSAValue* value : function.ssaArena().values()) {
                    if (value == nullptr) continue;
                    const SSAValueID id = value->valueId();
                    if (!coloring.isAssigned(id)) continue;
                    const RegKey physical = coloring.assignmentOf(id);
                    const uint32_t width =
                        value->type().dwordWidth == 0 ? 1 : value->type().dwordWidth;
                    if (rule.forbidsBase(physical.type, physical.idx, width)) {
                        report(rule, valueName(id) + " is " + regKeyToString(physical));
                    }
                }
                break;
            }
            case RuleKind::Offset: {
                std::vector<TupleRun> runs;
                std::vector<AffinitySet> sets;
                rule.addRelations(function, runs, sets);
                for (const TupleRun& run : runs) {
                    if (run.units.size() < 2 || !coloring.isAssigned(run.units.front())) continue;
                    const RegKey first = coloring.assignmentOf(run.units.front());
                    for (size_t unit = 1; unit < run.units.size(); ++unit) {
                        const SSAValueID id = run.units[unit];
                        if (!coloring.isAssigned(id)) continue;
                        const RegKey physical = coloring.assignmentOf(id);
                        if (physical.type == first.type && physical.idx == first.idx + unit)
                            continue;
                        report(rule, valueName(id) + " is " + regKeyToString(physical) +
                                         " but must sit " + std::to_string(unit) + " after " +
                                         regKeyToString(first));
                        break;
                    }
                }
                for (const AffinitySet& set : sets) {
                    if (set.members.size() < 2 || !coloring.isAssigned(set.members.front()))
                        continue;
                    const RegKey first = coloring.assignmentOf(set.members.front());
                    for (size_t i = 1; i < set.members.size(); ++i) {
                        const SSAValueID id = set.members[i];
                        if (!coloring.isAssigned(id)) continue;
                        if (coloring.assignmentOf(id) == first) continue;
                        report(rule, valueName(id) + " is " +
                                         regKeyToString(coloring.assignmentOf(id)) +
                                         " but must share " + regKeyToString(first));
                        break;
                    }
                }
                break;
            }
            case RuleKind::Interference: {
                for (const BasicBlock& block : function) {
                    for (const IRBase& ir : block) {
                        const auto* instruction = dyn_cast<StinkyInstruction>(&ir);
                        if (instruction == nullptr || !instruction->hasAttachedSSA()) continue;
                        if (!rule.clobbersEarly(*instruction)) continue;

                        forEachResult(*instruction, [&](SSAValueID dest) {
                            if (!coloring.isAssigned(dest)) return;
                            const RegKey destReg = coloring.assignmentOf(dest);
                            forEachOperand(*instruction, [&](SSAValueID src) {
                                if (src == dest || !coloring.isAssigned(src)) return;
                                if (coloring.assignmentOf(src) != destReg) return;
                                report(rule, valueName(dest) + " and " + valueName(src) +
                                                 " share " + regKeyToString(destReg) + " across " +
                                                 mnemonicOf(*instruction));
                            });
                        });
                    }
                }
                break;
            }
            case RuleKind::Preference:
                // Nothing to report: paying a preference produces a working
                // kernel that is merely slower, which the shadow report already
                // measures end to end.
                break;
            case RuleKind::Empty:
                break;
        }
    }

    return findings;
}

}  // namespace stinkytofu
