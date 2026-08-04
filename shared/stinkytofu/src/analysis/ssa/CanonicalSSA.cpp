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
#include "stinkytofu/analysis/ssa/CanonicalSSA.hpp"

#include <algorithm>
#include <cassert>
#include <limits>
#include <map>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "stinkytofu/analysis/controlflow/Dominance.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace stinkytofu {
namespace {

const std::vector<SSAPhiID> kEmptyPhiList;

void mixWord(uint64_t& hash, uint64_t word) {
    hash ^= word + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
}

void mixOperand(uint64_t& hash, const StinkyRegister& reg) {
    if (!reg.isRegister()) {
        mixWord(hash, 0);
        return;
    }
    mixWord(hash, 1);
    mixWord(hash, static_cast<uint64_t>(reg.reg.type));
    mixWord(hash, reg.reg.idx);
    mixWord(hash, reg.reg.num);
}

}  // namespace

uint64_t computeFunctionShape(const Function& function) {
    uint64_t hash = 0x27d4eb2f165667c5ULL;
    for (const BasicBlock& bb : function) {
        mixWord(hash, bb.getPredecessors().size());
        mixWord(hash, bb.getSuccessors().size());
        for (const IRBase& ir : bb) {
            const auto* instruction = dyn_cast<StinkyInstruction>(&ir);
            if (instruction == nullptr) continue;
            mixWord(hash, instruction->getUnifiedOpcode());
            for (const StinkyRegister& reg : instruction->getSrcRegs()) mixOperand(hash, reg);
            // Separator, so moving one operand from sources to destinations
            // cannot produce the same hash.
            mixWord(hash, 0);
            for (const StinkyRegister& reg : instruction->getDestRegs()) mixOperand(hash, reg);
        }
    }
    return hash == kUnstampedShape ? 1 : hash;
}

bool CanonicalSSA::empty() const {
    return values_.empty() && phis_.empty() && instructions_.empty() && blockPhis_.empty();
}

size_t CanonicalSSA::valueCount() const {
    return values_.size();
}

size_t CanonicalSSA::phiCount() const {
    return phis_.size();
}

bool CanonicalSSA::containsValue(SSAValueID id) const {
    return id != kInvalidSSAValueID && id <= values_.size();
}

bool CanonicalSSA::containsPhi(SSAPhiID id) const {
    return id != kInvalidSSAPhiID && id <= phis_.size();
}

const SSAValue& CanonicalSSA::value(SSAValueID id) const {
    assert(containsValue(id) && "invalid SSA value ID");
    return values_.at(id - 1);
}

const SSAPhi& CanonicalSSA::phi(SSAPhiID id) const {
    assert(containsPhi(id) && "invalid SSA PHI ID");
    return phis_.at(id - 1);
}

const std::vector<SSAValue>& CanonicalSSA::values() const {
    return values_;
}

const std::vector<SSAPhi>& CanonicalSSA::phis() const {
    return phis_;
}

const SSAInstructionInfo* CanonicalSSA::findInstructionInfo(
    const StinkyInstruction& instruction) const {
    auto it = instructions_.find(&instruction);
    return it == instructions_.end() ? nullptr : &it->second;
}

const std::vector<SSAPhiID>& CanonicalSSA::phisForBlock(const BasicBlock& block) const {
    auto it = blockPhis_.find(&block);
    return it == blockPhis_.end() ? kEmptyPhiList : it->second;
}

size_t CanonicalSSA::instructionInfoCount() const {
    return instructions_.size();
}

size_t CanonicalSSA::blockPhiListCount() const {
    return blockPhis_.size();
}

uint64_t CanonicalSSA::shape() const {
    return shape_;
}

SSAValueID CanonicalSSABuilder::addValue(SSAValue value) {
    assert(ssa_.values_.size() < std::numeric_limits<SSAValueID>::max() && "SSA value ID overflow");
    value.id = static_cast<SSAValueID>(ssa_.values_.size() + 1);
    ssa_.values_.push_back(std::move(value));
    return ssa_.values_.back().id;
}

SSAPhiID CanonicalSSABuilder::addPhi(SSAPhi phi) {
    assert(ssa_.phis_.size() < std::numeric_limits<SSAPhiID>::max() && "SSA PHI ID overflow");
    phi.id = static_cast<SSAPhiID>(ssa_.phis_.size() + 1);
    ssa_.phis_.push_back(std::move(phi));
    return ssa_.phis_.back().id;
}

SSAValue& CanonicalSSABuilder::value(SSAValueID id) {
    assert(ssa_.containsValue(id) && "invalid SSA value ID");
    return ssa_.values_.at(id - 1);
}

SSAPhi& CanonicalSSABuilder::phi(SSAPhiID id) {
    assert(ssa_.containsPhi(id) && "invalid SSA PHI ID");
    return ssa_.phis_.at(id - 1);
}

void CanonicalSSABuilder::setInstructionInfo(const StinkyInstruction& instruction,
                                             SSAInstructionInfo info) {
    ssa_.instructions_.insert_or_assign(&instruction, std::move(info));
}

void CanonicalSSABuilder::addPhiToBlock(const BasicBlock& block, SSAPhiID phiID) {
    assert(ssa_.containsPhi(phiID) && "invalid SSA PHI ID");
    ssa_.blockPhis_[&block].push_back(phiID);
}

void CanonicalSSABuilder::setShape(uint64_t shape) {
    ssa_.shape_ = shape;
}

CanonicalSSA CanonicalSSABuilder::take() {
    CanonicalSSA result = std::move(ssa_);
    ssa_ = CanonicalSSA{};
    return result;
}

std::string CanonicalSSAVerificationResult::toString() const {
    std::ostringstream out;
    for (size_t i = 0; i < errors.size(); ++i) {
        if (i > 0) out << "\n";
        out << errors[i];
    }
    return out.str();
}

namespace {

/// Number of atomic SSA units one physical operand is expected to bind.
///
/// Only allocatable physical ranges participate: literals, special registers,
/// pseudo registers, and unresolved template virtual registers bind nothing.
/// True16 halves are not modelled yet, so a range binds one unit per DWORD.
size_t expectedUnits(const StinkyRegister& reg) {
    if (!reg.isRegister() || reg.isVirtualReg()) return 0;
    if (!isAllocatableReg(reg.reg.type)) return 0;
    return reg.reg.num;
}

size_t countInstructionUses(const SSAValue& value, const StinkyInstruction* instruction,
                            uint32_t operand, uint32_t unit) {
    size_t matches = 0;
    for (const SSAUse& use : value.uses) {
        if (!use.isPhiUse() && use.instruction == instruction && use.operand == operand &&
            use.unit == unit)
            ++matches;
    }
    return matches;
}

size_t countPhiUses(const SSAValue& value, SSAPhiID phi, const BasicBlock* predecessor) {
    size_t matches = 0;
    for (const SSAUse& use : value.uses) {
        if (use.isPhiUse() && use.phi == phi && use.predecessor == predecessor) ++matches;
    }
    return matches;
}

/// True when \p edge is the first slot carrying its (predecessor, value) pair.
/// Used to report duplicate predecessor edges once, in slot order.
bool isFirstEdgeOfGroup(const SSAPhi& phi, size_t edge) {
    for (size_t earlier = 0; earlier < edge; ++earlier) {
        if (phi.incoming[earlier].predecessor == phi.incoming[edge].predecessor &&
            phi.incoming[earlier].value == phi.incoming[edge].value)
            return false;
    }
    return true;
}

/// Verifies one graph. Diagnostics are emitted in function/graph order so
/// repeated runs produce identical output.
class Verifier {
   public:
    Verifier(const Function& function, const CanonicalSSA& ssa, const DominanceInfo* dominance)
        : function_(function), ssa_(ssa), dominance_(dominance) {
        uint32_t blockIndex = 0;
        uint32_t instructionIndex = 0;
        for (const BasicBlock& bb : function_) {
            blockOrder_.emplace(&bb, blockIndex++);
            for (const IRBase& ir : bb) {
                if (const auto* inst = dyn_cast<StinkyInstruction>(&ir)) {
                    instructionOrder_.emplace(inst, instructionIndex++);
                    instructionBlock_.emplace(inst, &bb);
                }
            }
        }
    }

    CanonicalSSAVerificationResult run() {
        // A graph that describes a different program cannot be walked safely:
        // its instruction pointers may already be dangling. Report and stop.
        if (!checkShape()) return std::move(result_);

        checkValues();
        checkInstructionBindings();
        checkPhis();
        checkBlockPhiIndex();
        return std::move(result_);
    }

   private:
    void error(const std::string& message) {
        result_.errors.push_back(message);
    }

    bool knownBlock(const BasicBlock* block) const {
        return block != nullptr && blockOrder_.find(block) != blockOrder_.end();
    }

    bool knownInstruction(const StinkyInstruction* instruction) const {
        return instruction != nullptr &&
               instructionOrder_.find(instruction) != instructionOrder_.end();
    }

    static std::string valueRef(SSAValueID id) {
        return "%" + std::to_string(id);
    }

    static std::string phiRef(SSAPhiID id) {
        return "phi#" + std::to_string(id);
    }

    std::string instructionRef(const StinkyInstruction* instruction) const {
        if (instruction == nullptr) return "<null-instruction>";
        auto it = instructionOrder_.find(instruction);
        if (it == instructionOrder_.end()) return "<foreign-instruction>";
        return "#" + std::to_string(it->second);
    }

    std::string blockRef(const BasicBlock* block) const {
        if (block == nullptr) return "<null-block>";
        auto it = blockOrder_.find(block);
        if (it == blockOrder_.end()) return "<foreign-block>";
        if (!block->getLabel().empty()) return "^" + block->getLabel();
        return "^bb" + std::to_string(it->second);
    }

    /// True when the graph still describes this function. A hand-built graph
    /// carries no fingerprint, so there is nothing to compare and it passes.
    bool checkShape() {
        if (ssa_.shape() == kUnstampedShape) return true;
        const uint64_t actual = computeFunctionShape(function_);
        if (actual == ssa_.shape()) return true;
        error("@" + function_.getName() +
              ": the function changed after it was lifted, so this canonical SSA describes a "
              "different program");
        return false;
    }

    void checkValues();
    void checkValueDefinition(const SSAValue& value);
    void checkValueUses(const SSAValue& value);
    void checkInstructionBindings();
    void checkBinding(const StinkyInstruction& instruction, bool isDestination, uint32_t operand,
                      const StinkyRegister& reg, const SSAOperandBinding& binding);
    void checkPhis();
    void checkBlockPhiIndex();

    /// Block holding the definition of \p value, or null when it has none.
    const BasicBlock* definitionBlock(const SSAValue& value) const {
        switch (value.kind) {
            case SSAValueKind::InstructionDef: {
                auto it = instructionBlock_.find(value.definingInstruction);
                return it == instructionBlock_.end() ? nullptr : it->second;
            }
            case SSAValueKind::Phi:
                return ssa_.containsPhi(value.definingPhi) ? ssa_.phi(value.definingPhi).block
                                                           : nullptr;
            case SSAValueKind::LiveIn:
            case SSAValueKind::Undef:
                // Entry values are available everywhere.
                return function_.empty() ? nullptr : &*function_.begin();
        }
        return nullptr;
    }

    const Function& function_;
    const CanonicalSSA& ssa_;
    const DominanceInfo* dominance_ = nullptr;

    std::unordered_map<const BasicBlock*, uint32_t> blockOrder_;
    std::unordered_map<const StinkyInstruction*, uint32_t> instructionOrder_;
    std::unordered_map<const StinkyInstruction*, const BasicBlock*> instructionBlock_;

    // Definition sites already claimed, to catch two values defining one slot.
    std::map<std::tuple<const StinkyInstruction*, uint32_t, uint32_t>, SSAValueID> definitionSites_;
    std::map<SSAPhiID, SSAValueID> phiResults_;

    CanonicalSSAVerificationResult result_;
};

void Verifier::checkValues() {
    const std::vector<SSAValue>& values = ssa_.values();
    for (size_t index = 0; index < values.size(); ++index) {
        const SSAValue& value = values[index];
        const SSAValueID expectedID = static_cast<SSAValueID>(index + 1);
        if (value.id != expectedID) {
            error("value at index " + std::to_string(index) + " has id " + valueRef(value.id) +
                  ", expected " + valueRef(expectedID));
        }
        if (!isAllocatableReg(value.origin.type)) {
            error(valueRef(expectedID) + " has non-allocatable origin " +
                  regKeyToString(value.origin));
        }
        checkValueDefinition(value);
        checkValueUses(value);
    }
}

void Verifier::checkValueDefinition(const SSAValue& value) {
    const std::string self = valueRef(value.id);
    switch (value.kind) {
        case SSAValueKind::LiveIn:
        case SSAValueKind::Undef:
            if (value.definingInstruction != nullptr)
                error(self + " is a live-in or undef value but names a defining instruction");
            if (value.definingPhi != kInvalidSSAPhiID)
                error(self + " is a live-in or undef value but names a defining phi");
            break;

        case SSAValueKind::InstructionDef: {
            if (value.definingPhi != kInvalidSSAPhiID)
                error(self + " is an instruction definition but also names a defining phi");
            if (value.definingInstruction == nullptr) {
                error(self + " is an instruction definition without a defining instruction");
                break;
            }
            if (!knownInstruction(value.definingInstruction)) {
                error(self + " is defined by an instruction outside the function");
                break;
            }
            const auto site = std::make_tuple(value.definingInstruction, value.definingOperand,
                                              value.definingUnit);
            auto [it, inserted] = definitionSites_.emplace(site, value.id);
            if (!inserted) {
                error(self + " and " + valueRef(it->second) + " both define " +
                      instructionRef(value.definingInstruction) + " dst" +
                      std::to_string(value.definingOperand) + " unit " +
                      std::to_string(value.definingUnit));
            }
            break;
        }

        case SSAValueKind::Phi: {
            if (value.definingInstruction != nullptr)
                error(self + " is a phi result but also names a defining instruction");
            if (!ssa_.containsPhi(value.definingPhi)) {
                error(self + " is a phi result with invalid defining phi " +
                      phiRef(value.definingPhi));
                break;
            }
            const SSAPhi& phi = ssa_.phi(value.definingPhi);
            if (phi.result != value.id) {
                error(self + " names " + phiRef(phi.id) +
                      " as its definition, but that phi's "
                      "result is " +
                      valueRef(phi.result));
            }
            if (!(phi.origin == value.origin)) {
                error(self + " has origin " + regKeyToString(value.origin) + " but " +
                      phiRef(phi.id) + " has origin " + regKeyToString(phi.origin));
            }
            auto [it, inserted] = phiResults_.emplace(value.definingPhi, value.id);
            if (!inserted) {
                error(self + " and " + valueRef(it->second) + " are both results of " +
                      phiRef(value.definingPhi));
            }
            break;
        }
    }
}

void Verifier::checkValueUses(const SSAValue& value) {
    const std::string self = valueRef(value.id);
    for (size_t useIndex = 0; useIndex < value.uses.size(); ++useIndex) {
        const SSAUse& use = value.uses[useIndex];
        const std::string where = self + " use " + std::to_string(useIndex);

        if (use.isPhiUse()) {
            if (use.instruction != nullptr)
                error(where + " is a phi-edge use but also names an instruction");
            if (!ssa_.containsPhi(use.phi)) {
                error(where + " references invalid " + phiRef(use.phi));
                continue;
            }
            if (!knownBlock(use.predecessor)) {
                error(where + " names a predecessor outside the function");
                continue;
            }
            const SSAPhi& phi = ssa_.phi(use.phi);
            bool found = false;
            for (const SSAPhiIncoming& incoming : phi.incoming) {
                if (incoming.predecessor == use.predecessor && incoming.value == value.id) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                error(where + " claims edge " + blockRef(use.predecessor) + " of " +
                      phiRef(phi.id) + ", but that edge does not carry " + self);
            }
            continue;
        }

        if (use.instruction == nullptr) {
            error(where + " names neither an instruction nor a phi");
            continue;
        }
        if (!knownInstruction(use.instruction)) {
            error(where + " references an instruction outside the function");
            continue;
        }

        const SSAInstructionInfo* info = ssa_.findInstructionInfo(*use.instruction);
        if (info == nullptr) {
            error(where + " references " + instructionRef(use.instruction) +
                  ", which has no operand bindings");
        } else if (use.operand >= info->sources.size()) {
            error(where + " references src" + std::to_string(use.operand) + " of " +
                  instructionRef(use.instruction) + ", which has " +
                  std::to_string(info->sources.size()) + " bound source operand(s)");
        } else {
            const std::vector<SSAValueID>& units = info->sources[use.operand].units;
            if (use.unit >= units.size()) {
                error(where + " references unit " + std::to_string(use.unit) + " of " +
                      instructionRef(use.instruction) + " src" + std::to_string(use.operand) +
                      ", which binds " + std::to_string(units.size()) + " unit(s)");
            } else if (units[use.unit] != value.id) {
                error(where + " is not mirrored by " + instructionRef(use.instruction) + " src" +
                      std::to_string(use.operand) + " unit " + std::to_string(use.unit) +
                      ", which binds " + valueRef(units[use.unit]));
            }
        }

        const BasicBlock* useBlock = instructionBlock_.at(use.instruction);
        const BasicBlock* defBlock = definitionBlock(value);

        if (value.kind == SSAValueKind::InstructionDef &&
            knownInstruction(value.definingInstruction) && defBlock == useBlock) {
            // Same block: sources are read before destinations are defined, so
            // the definition must appear strictly earlier than the use.
            if (instructionOrder_.at(value.definingInstruction) >=
                instructionOrder_.at(use.instruction)) {
                error(self + " is defined by " + instructionRef(value.definingInstruction) +
                      " but used earlier or at the same position by " +
                      instructionRef(use.instruction) + " in " + blockRef(defBlock));
            }
            continue;
        }

        if (dominance_ != nullptr && defBlock != nullptr &&
            !dominates(*dominance_, defBlock, useBlock)) {
            error(self + " is defined in " + blockRef(defBlock) + ", which does not dominate " +
                  blockRef(useBlock) + " where " + instructionRef(use.instruction) + " uses it");
        }
    }
}

void Verifier::checkInstructionBindings() {
    size_t reachableInfos = 0;

    for (const BasicBlock& bb : function_) {
        for (const IRBase& ir : bb) {
            const auto* instruction = dyn_cast<StinkyInstruction>(&ir);
            if (instruction == nullptr) continue;

            const SSAInstructionInfo* info = ssa_.findInstructionInfo(*instruction);
            if (info == nullptr) {
                bool allocatable = false;
                for (const StinkyRegister& reg : instruction->getSrcRegs())
                    allocatable = allocatable || expectedUnits(reg) > 0;
                for (const StinkyRegister& reg : instruction->getDestRegs())
                    allocatable = allocatable || expectedUnits(reg) > 0;
                if (allocatable) {
                    error(instructionRef(instruction) +
                          " has allocatable register operands but no SSA operand bindings");
                }
                continue;
            }
            ++reachableInfos;

            const std::vector<StinkyRegister>& srcRegs = instruction->getSrcRegs();
            const std::vector<StinkyRegister>& destRegs = instruction->getDestRegs();
            if (info->sources.size() != srcRegs.size()) {
                error(instructionRef(instruction) + " binds " +
                      std::to_string(info->sources.size()) + " source operand(s) but has " +
                      std::to_string(srcRegs.size()));
            }
            if (info->destinations.size() != destRegs.size()) {
                error(instructionRef(instruction) + " binds " +
                      std::to_string(info->destinations.size()) +
                      " destination operand(s) but has " + std::to_string(destRegs.size()));
            }

            const size_t sourceCount = std::min(info->sources.size(), srcRegs.size());
            for (size_t operand = 0; operand < sourceCount; ++operand) {
                checkBinding(*instruction, /*isDestination=*/false, static_cast<uint32_t>(operand),
                             srcRegs[operand], info->sources[operand]);
            }
            const size_t destCount = std::min(info->destinations.size(), destRegs.size());
            for (size_t operand = 0; operand < destCount; ++operand) {
                checkBinding(*instruction, /*isDestination=*/true, static_cast<uint32_t>(operand),
                             destRegs[operand], info->destinations[operand]);
            }
        }
    }

    if (ssa_.instructionInfoCount() > reachableInfos) {
        error(std::to_string(ssa_.instructionInfoCount() - reachableInfos) +
              " operand-binding entr(ies) reference instructions outside the function");
    }
}

void Verifier::checkBinding(const StinkyInstruction& instruction, bool isDestination,
                            uint32_t operand, const StinkyRegister& reg,
                            const SSAOperandBinding& binding) {
    const std::string role = isDestination ? "dst" : "src";
    const std::string where = instructionRef(&instruction) + " " + role + std::to_string(operand);

    const size_t expected = expectedUnits(reg);
    if (binding.units.size() != expected) {
        error(where + " expects " + std::to_string(expected) + " SSA unit(s) but binds " +
              std::to_string(binding.units.size()));
    }

    for (size_t unit = 0; unit < binding.units.size(); ++unit) {
        const SSAValueID id = binding.units[unit];
        const std::string slot = where + " unit " + std::to_string(unit);
        if (!ssa_.containsValue(id)) {
            error(slot + " references invalid value " + valueRef(id));
            continue;
        }

        const SSAValue& value = ssa_.value(id);
        if (reg.isRegister() && unit < reg.reg.num) {
            const RegKey expectedKey = toRegKey(reg, static_cast<unsigned>(unit));
            if (!(value.origin == expectedKey)) {
                error(slot + " binds " + valueRef(id) + " with origin " +
                      regKeyToString(value.origin) + ", but the operand unit is " +
                      regKeyToString(expectedKey));
            }
        }

        if (isDestination) {
            const bool defined = value.kind == SSAValueKind::InstructionDef &&
                                 value.definingInstruction == &instruction &&
                                 value.definingOperand == operand && value.definingUnit == unit;
            if (!defined) {
                error(slot + " binds " + valueRef(id) +
                      ", which does not record this slot as its "
                      "definition");
            }
            continue;
        }

        const size_t matches =
            countInstructionUses(value, &instruction, operand, static_cast<uint32_t>(unit));
        if (matches != 1) {
            error(slot + " binds " + valueRef(id) + ", whose use list records this slot " +
                  std::to_string(matches) + " time(s) (expected 1)");
        }
    }
}

void Verifier::checkPhis() {
    const std::vector<SSAPhi>& phis = ssa_.phis();
    for (size_t index = 0; index < phis.size(); ++index) {
        const SSAPhi& phi = phis[index];
        const SSAPhiID expectedID = static_cast<SSAPhiID>(index + 1);
        if (phi.id != expectedID) {
            error("phi at index " + std::to_string(index) + " has id " + phiRef(phi.id) +
                  ", expected " + phiRef(expectedID));
        }
        const std::string self = phiRef(expectedID);

        if (!isAllocatableReg(phi.origin.type))
            error(self + " has non-allocatable origin " + regKeyToString(phi.origin));

        if (!ssa_.containsValue(phi.result)) {
            error(self + " has invalid result " + valueRef(phi.result));
        } else {
            const SSAValue& result = ssa_.value(phi.result);
            if (result.kind != SSAValueKind::Phi)
                error(self + " has result " + valueRef(phi.result) + ", which is not a phi value");
            else if (result.definingPhi != phi.id)
                error(self + " has result " + valueRef(phi.result) + ", which is defined by " +
                      phiRef(result.definingPhi));
        }

        if (!knownBlock(phi.block)) {
            error(self + " belongs to a block outside the function");
            continue;
        }

        const std::vector<BasicBlock*>& predecessors = phi.block->getPredecessors();
        if (phi.incoming.size() != predecessors.size()) {
            error(self + " in " + blockRef(phi.block) + " has " +
                  std::to_string(phi.incoming.size()) + " incoming value(s) for " +
                  std::to_string(predecessors.size()) + " predecessor(s)");
        }

        for (size_t edge = 0; edge < phi.incoming.size(); ++edge) {
            const SSAPhiIncoming& incoming = phi.incoming[edge];
            const std::string slot = self + " edge " + std::to_string(edge);

            if (edge < predecessors.size() && incoming.predecessor != predecessors[edge]) {
                error(slot + " names " + blockRef(incoming.predecessor) + " but predecessor " +
                      std::to_string(edge) + " of " + blockRef(phi.block) + " is " +
                      blockRef(predecessors[edge]));
            }

            if (!ssa_.containsValue(incoming.value)) {
                error(slot + " references invalid value " + valueRef(incoming.value));
                continue;
            }
            const SSAValue& value = ssa_.value(incoming.value);
            if (!(value.origin == phi.origin)) {
                error(slot + " carries " + valueRef(incoming.value) + " with origin " +
                      regKeyToString(value.origin) + ", but " + self + " has origin " +
                      regKeyToString(phi.origin));
            }

            if (dominance_ != nullptr && knownBlock(incoming.predecessor)) {
                // A PHI input is used on the edge, so it must dominate the end
                // of the predecessor rather than the PHI's own block.
                const BasicBlock* defBlock = definitionBlock(value);
                if (defBlock != nullptr &&
                    !dominates(*dominance_, defBlock, incoming.predecessor)) {
                    error(slot + " carries " + valueRef(incoming.value) + " defined in " +
                          blockRef(defBlock) + ", which does not dominate predecessor " +
                          blockRef(incoming.predecessor));
                }
            }

            // A block may appear as a predecessor more than once, so the number
            // of expected use records is the number of matching edge slots.
            if (isFirstEdgeOfGroup(phi, edge)) {
                size_t slots = 0;
                for (const SSAPhiIncoming& other : phi.incoming) {
                    if (other.predecessor == incoming.predecessor && other.value == incoming.value)
                        ++slots;
                }
                const size_t matches = countPhiUses(value, phi.id, incoming.predecessor);
                if (matches != slots) {
                    error(slot + " carries " + valueRef(incoming.value) +
                          ", whose use list records this edge " + std::to_string(matches) +
                          " time(s) (expected " + std::to_string(slots) + ")");
                }
            }
        }
    }
}

void Verifier::checkBlockPhiIndex() {
    std::map<SSAPhiID, uint32_t> listedCount;
    size_t reachableLists = 0;

    for (const BasicBlock& bb : function_) {
        const std::vector<SSAPhiID>& ids = ssa_.phisForBlock(bb);
        if (!ids.empty()) ++reachableLists;
        for (SSAPhiID id : ids) {
            if (!ssa_.containsPhi(id)) {
                error(blockRef(&bb) + " lists invalid " + phiRef(id));
                continue;
            }
            if (ssa_.phi(id).block != &bb) {
                error(blockRef(&bb) + " lists " + phiRef(id) + ", which belongs to " +
                      blockRef(ssa_.phi(id).block));
            }
            ++listedCount[id];
        }
    }

    for (const SSAPhi& phi : ssa_.phis()) {
        const uint32_t count = listedCount.count(phi.id) != 0 ? listedCount.at(phi.id) : 0;
        if (count != 1) {
            error(phiRef(phi.id) + " appears " + std::to_string(count) +
                  " time(s) in the block phi index (expected 1)");
        }
    }

    if (ssa_.blockPhiListCount() > reachableLists) {
        error(std::to_string(ssa_.blockPhiListCount() - reachableLists) +
              " phi list(s) reference blocks outside the function");
    }
}

}  // namespace

CanonicalSSAVerificationResult verifyCanonicalSSA(const Function& function,
                                                  const CanonicalSSA& ssa) {
    return Verifier(function, ssa, /*dominance=*/nullptr).run();
}

CanonicalSSAVerificationResult verifyCanonicalSSA(const Function& function, const CanonicalSSA& ssa,
                                                  const DominanceInfo& dominance) {
    return Verifier(function, ssa, &dominance).run();
}

}  // namespace stinkytofu
