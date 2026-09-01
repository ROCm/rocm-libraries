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
#include "stinkytofu/transforms/asm/ra/AllocationConstraints.hpp"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/AsmTargetRegisters.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/ssa/SSAOperandUnits.hpp"
#include "stinkytofu/ir/asm/ssa/StinkyOpOperand.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationRules.hpp"

namespace stinkytofu {
namespace {

void ensureIndex(std::vector<RegType>& classes, std::vector<std::optional<RegKey>>& hints,
                 SSAValueID id) {
    if (id >= classes.size()) {
        classes.resize(id + 1, RegType::UNKNOWN);
        hints.resize(id + 1);
    }
}

void recordValue(const StinkySSAValue* value, std::vector<RegType>& classes,
                 std::vector<std::optional<RegKey>>& hints) {
    if (value == nullptr) return;
    const SSAValueID id = value->valueId();
    if (id == kInvalidSSAValueID) return;
    ensureIndex(classes, hints, id);
    classes[id] = value->type().regType;
    if (!value->hasPhysicalBinding()) return;
    const StinkySSAValue::PhysicalBinding& binding = value->physical();
    hints[id] = RegKey{binding.type, binding.idx, RegHalf::NONE};
}

void collectUnits(const std::vector<StinkySSAValue*>& values, std::vector<TupleRun>& runs) {
    if (values.size() < 2) return;
    TupleRun run;
    run.units.reserve(values.size());
    for (StinkySSAValue* value : values) {
        if (value == nullptr) return;
        run.units.push_back(value->valueId());
    }
    runs.push_back(std::move(run));
}

void collectLiftedDestinations(const StinkyInstruction& instruction, const RegClassSet& classes,
                               std::vector<TupleRun>& runs) {
    size_t cursor = 0;
    const std::vector<StinkyRegister>& destRegs = instruction.getDestRegs();
    for (size_t operand = 0; operand < destRegs.size(); ++operand) {
        const size_t units = liftedSSAUnits(destRegs[operand], classes);
        if (units == 0) continue;
        if (cursor + units > instruction.getNumSSAResults()) return;
        std::vector<StinkySSAValue*> values;
        values.reserve(units);
        for (size_t unit = 0; unit < units; ++unit)
            values.push_back(instruction.getSSAResult(cursor++));
        collectUnits(values, runs);
    }
}

void collectLiftedSources(const StinkyInstruction& instruction, const RegClassSet& classes,
                          std::vector<TupleRun>& runs) {
    size_t cursor = 0;
    const std::vector<StinkyRegister>& srcRegs = instruction.getSrcRegs();
    for (size_t operand = 0; operand < srcRegs.size(); ++operand) {
        const size_t units = liftedSSAUnits(srcRegs[operand], classes);
        if (units == 0) {
            if (cursor < instruction.getNumSSAOperands()) ++cursor;
            continue;
        }
        if (cursor + units > instruction.getNumSSAOperands()) return;
        std::vector<StinkySSAValue*> values;
        values.reserve(units);
        for (size_t unit = 0; unit < units; ++unit)
            values.push_back(instruction.getSSAOperandValue(cursor++));
        collectUnits(values, runs);
    }
}

/// SSA values behind each register operand, by operand position. A register
/// operand covers as many values as it has lifted DWORDs, so this is positional
/// rather than one value per operand.
std::vector<std::vector<SSAValueID>> valueGroups(const StinkyInstruction& instruction,
                                                 const RegClassSet& classes, bool destinations) {
    std::vector<std::vector<SSAValueID>> groups;
    const std::vector<StinkyRegister>& regs =
        destinations ? instruction.getDestRegs() : instruction.getSrcRegs();
    const size_t available =
        destinations ? instruction.getNumSSAResults() : instruction.getNumSSAOperands();
    size_t cursor = 0;
    for (const StinkyRegister& reg : regs) {
        const size_t units = liftedSSAUnits(reg, classes);
        std::vector<SSAValueID> ids;
        if (units == 0) {
            // A source the lift did not cover still consumes an operand slot.
            if (!destinations && cursor < available) ++cursor;
            groups.push_back(std::move(ids));
            continue;
        }
        for (size_t unit = 0; unit < units && cursor < available; ++unit) {
            const StinkySSAValue* value = destinations ? instruction.getSSAResult(cursor)
                                                       : instruction.getSSAOperandValue(cursor);
            ++cursor;
            ids.push_back(value == nullptr ? kInvalidSSAValueID : value->valueId());
        }
        groups.push_back(std::move(ids));
    }
    return groups;
}

/// Tie a read-write destination to the source naming the same register.
///
/// The hardware reads such a destination on the path where it does not write
/// it: `s_cmov_b32 d, s` is `if (SCC) d = s`, so on the untaken path d keeps
/// what it already held. Give d and that source different registers and the
/// untaken path yields whatever happened to be in d, not the value the IR says
/// it should. HwInstDesc marks the field, and AsmVerifierPass already requires
/// the register to appear on both sides; this is what makes the allocator keep
/// it that way.
void collectReadWriteTies(const StinkyInstruction& instruction, const RegClassSet& classes,
                          std::vector<AffinitySet>& sets) {
    const HwInstDesc* desc = instruction.getHwInstDesc();
    if (desc == nullptr || desc->operandFields.empty()) return;

    const std::vector<StinkyRegister>& destRegs = instruction.getDestRegs();
    const std::vector<StinkyRegister>& srcRegs = instruction.getSrcRegs();
    const std::vector<std::vector<SSAValueID>> destGroups =
        valueGroups(instruction, classes, /*destinations=*/true);
    const std::vector<std::vector<SSAValueID>> srcGroups =
        valueGroups(instruction, classes, /*destinations=*/false);

    // Only the destination side needs walking: the source that pairs with a
    // read-write destination is the one naming the same register, which is what
    // AsmVerifierPass requires to be there, so it is found by lookup rather
    // than by tracking a second cursor.
    size_t destIdx = 0;
    for (const HwInstDesc::OperandFieldDesc& field : desc->operandFields) {
        if (!field.isDest) continue;
        const size_t destSlot = destIdx++;
        if (!field.isReadWrite) continue;
        if (destSlot >= destRegs.size() || destSlot >= destGroups.size()) continue;

        const StinkyRegister& reg = destRegs[destSlot];
        for (size_t source = 0; source < srcRegs.size() && source < srcGroups.size(); ++source) {
            if (!(srcRegs[source] == reg)) continue;
            const std::vector<SSAValueID>& written = destGroups[destSlot];
            const std::vector<SSAValueID>& read = srcGroups[source];
            for (size_t unit = 0; unit < written.size() && unit < read.size(); ++unit) {
                if (written[unit] == kInvalidSSAValueID || read[unit] == kInvalidSSAValueID)
                    continue;
                if (written[unit] == read[unit]) continue;
                AffinitySet set;
                set.members = {written[unit], read[unit]};
                std::sort(set.members.begin(), set.members.end());
                sets.push_back(std::move(set));
            }
            break;
        }
    }
}

std::string joinIds(const std::vector<SSAValueID>& ids) {
    std::ostringstream out;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) out << ", ";
        out << '%' << ids[i];
    }
    return out.str();
}

}  // namespace

AllocationConstraints AllocationConstraints::build(const Function& function,
                                                   const AsmTargetRegisters& target,
                                                   const AllocationRules& rules) {
    AllocationConstraints constraints;
    constraints.target_ = &target;

    const size_t valueCount = function.ssaArena().valueCount();
    constraints.classByValue_.assign(valueCount + 1, RegType::UNKNOWN);
    constraints.hintByValue_.assign(valueCount + 1, std::nullopt);
    constraints.pinnedByValue_.assign(valueCount + 1, false);

    for (StinkySSAValue* value : function.ssaArena().values()) {
        recordValue(value, constraints.classByValue_, constraints.hintByValue_);
    }

    const RegClassSet& liftedClasses = function.ssaArena().liftedClasses();

    for (const BasicBlock& block : function) {
        for (const IRBase& ir : block) {
            const auto* instruction = dyn_cast<StinkyInstruction>(&ir);
            if (instruction == nullptr || !instruction->hasAttachedSSA()) continue;
            collectLiftedDestinations(*instruction, liftedClasses, constraints.tupleRuns_);
            collectLiftedSources(*instruction, liftedClasses, constraints.tupleRuns_);
            collectReadWriteTies(*instruction, liftedClasses, constraints.affinitySets_);
        }

        for (const SSABlockArgument& arg : block.ssaArguments()) {
            if (arg.value == nullptr) continue;
            // No incoming edge means nothing in the function defines this value:
            // it arrives in a register the dispatch chose, so it cannot move.
            if (arg.incoming.empty()) {
                const SSAValueID id = arg.value->valueId();
                if (id != kInvalidSSAValueID && id < constraints.pinnedByValue_.size())
                    constraints.pinnedByValue_[id] = true;
                continue;
            }
            AffinitySet set;
            set.members.push_back(arg.value->valueId());
            for (const SSABlockIncoming& incoming : arg.incoming) {
                const StinkyOpOperand* use = incoming.use.get();
                const StinkySSAValue* value = use == nullptr ? nullptr : use->value();
                if (value == nullptr) continue;
                set.members.push_back(value->valueId());
            }
            std::sort(set.members.begin(), set.members.end());
            set.members.erase(std::unique(set.members.begin(), set.members.end()),
                              set.members.end());
            if (set.members.size() < 2) continue;
            constraints.affinitySets_.push_back(std::move(set));
        }
    }

    // Last, so a rule-imposed relation is appended to the IR-derived ones and
    // both reach OffsetUnion and the verifier identically. Only Active rules
    // contribute; the table decides that, not the rule.
    rules.addRelations(function, constraints.tupleRuns_, constraints.affinitySets_);

    return constraints;
}

RegType AllocationConstraints::classOf(SSAValueID id) const {
    if (id == kInvalidSSAValueID || id >= classByValue_.size()) return RegType::UNKNOWN;
    return classByValue_[id];
}

bool AllocationConstraints::isAllocatable(SSAValueID id) const {
    if (target_ == nullptr) return false;
    return target_->isAllocatableClass(classOf(id));
}

std::optional<RegKey> AllocationConstraints::hintFor(SSAValueID id) const {
    if (id == kInvalidSSAValueID || id >= hintByValue_.size()) return std::nullopt;
    return hintByValue_[id];
}

bool AllocationConstraints::isPinned(SSAValueID id) const {
    if (id == kInvalidSSAValueID || id >= pinnedByValue_.size()) return false;
    return pinnedByValue_[id];
}

std::string AllocationConstraints::toString() const {
    std::ostringstream out;
    out << "values=" << (classByValue_.empty() ? 0 : classByValue_.size() - 1);
    out << " tuples=" << tupleRuns_.size();
    out << " affinity=" << affinitySets_.size() << '\n';
    for (size_t id = 1; id < hintByValue_.size(); ++id) {
        out << '%' << id << ':' << regTypeToString(classOf(static_cast<SSAValueID>(id)));
        if (hintByValue_[id].has_value()) out << " hint " << regKeyToString(*hintByValue_[id]);
        out << '\n';
    }
    for (const TupleRun& run : tupleRuns_) {
        out << "tuple [" << joinIds(run.units) << "]\n";
    }
    for (const AffinitySet& set : affinitySets_) {
        out << "affinity {" << joinIds(set.members) << "}\n";
    }
    return out.str();
}

}  // namespace stinkytofu
