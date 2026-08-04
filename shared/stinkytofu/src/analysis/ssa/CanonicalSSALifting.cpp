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
#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "stinkytofu/analysis/controlflow/Dominance.hpp"
#include "stinkytofu/analysis/ssa/CanonicalSSA.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace stinkytofu {
namespace {

/// How one physical operand participates in allocator SSA.
enum class OperandKind {
    /// Not allocatable: literal, special register, or pseudo register.
    Ignored,
    /// Allocatable full-DWORD range.
    AllocatableRange,
    /// Recognised but out of scope; `reason` explains why.
    Unsupported,
};

struct OperandClass {
    OperandKind kind = OperandKind::Ignored;
    size_t units = 0;
    std::string reason;
};

/// Classes whose registers become SSA values.
///
/// VCC and EXEC are their own register types here rather than SGPR indices, so
/// widening to SGPRs cannot make a scalar operand alias a special register.
/// Accumulators stay out until their VGPR aliasing rules are modelled: on some
/// architectures an AGPR and a VGPR name the same storage, and two SSA values
/// over one physical register would be unsound.
bool isLiftableClass(RegType type) {
    return type == RegType::V || type == RegType::S;
}

OperandClass classifyOperand(const StinkyRegister& reg) {
    if (!reg.isRegister()) return {OperandKind::Ignored, 0, {}};
    if (reg.isVirtualReg())
        return {OperandKind::Unsupported, 0,
                "unresolved template virtual register; resolve it before lifting"};
    if (isPseudoReg(reg)) return {OperandKind::Ignored, 0, {}};
    if (!isAllocatableReg(reg.reg.type)) return {OperandKind::Ignored, 0, {}};
    if (!isLiftableClass(reg.reg.type))
        return {OperandKind::Unsupported, 0,
                "register class '" + regTypeToString(reg.reg.type) +
                    "' is not lifted yet; VGPRs and SGPRs are supported"};
    return {OperandKind::AllocatableRange, reg.reg.num, {}};
}

/// True when any operand selects a True16 half, which needs sub-DWORD units.
bool usesTrue16Halves(const StinkyInstruction& instruction) {
    const auto* modifier = instruction.getModifier<True16Modifiers>();
    if (modifier == nullptr) return false;
    if (modifier->getDst0() != HighBitSel::NONE) return true;
    if (modifier->getDst1() != HighBitSel::NONE) return true;
    for (size_t src = 0; src < modifier->getSrcCount(); ++src) {
        if (modifier->getSrc(src) != HighBitSel::NONE) return true;
    }
    return false;
}

/// Total order on register keys, so PHI placement and live-in creation visit
/// keys in a stable order and produce identical graphs across runs.
bool regKeyLess(const RegKey& lhs, const RegKey& rhs) {
    if (lhs.type != rhs.type) return lhs.type < rhs.type;
    if (lhs.idx != rhs.idx) return lhs.idx < rhs.idx;
    return lhs.half < rhs.half;
}

std::vector<RegKey> sortedKeys(const RegKeySet& keys) {
    std::vector<RegKey> sorted(keys.begin(), keys.end());
    std::sort(sorted.begin(), sorted.end(), regKeyLess);
    return sorted;
}

class Lifter {
   public:
    Lifter(Function& function, const DominanceInfo& dominance,
           const LiftAsmRegistersToSSAOptions& options)
        : function_(function), dominance_(dominance), options_(options) {}

    Expected<CanonicalSSA> run();

   private:
    using Result = Expected<CanonicalSSA>;

    /// Per-block register facts gathered before any SSA value exists.
    struct BlockFacts {
        RegKeySet defs;
        /// Keys read before being written in this block, so their value must
        /// arrive from a predecessor.
        RegKeySet upwardExposed;
        RegKeySet liveIn;
    };

    /// Where a key was first read without a local definition, for diagnostics.
    struct ExposedUse {
        uint32_t instructionIndex = 0;
        uint32_t operand = 0;
    };

    // Setup and validation.
    void indexInstructions();
    bool checkReachability();
    bool gatherBlockFacts();
    bool validateInstruction(const StinkyInstruction& instruction, uint32_t index);

    // SSA construction.
    void computeLiveness();
    bool createEntryLiveIns(CanonicalSSABuilder& builder);
    void placePhis(CanonicalSSABuilder& builder);
    bool rename(CanonicalSSABuilder& builder);
    void renameBlock(CanonicalSSABuilder& builder, unsigned block, std::vector<RegKey>& pushedKeys);

    // Diagnostics.
    bool fail(const std::string& location, const std::string& message) {
        error_ = "@" + function_.getName() + location + ": " + message;
        return false;
    }
    bool fail(const std::string& message) {
        return fail("", message);
    }
    bool failAt(uint32_t instructionIndex, const std::string& message) {
        return fail(" #" + std::to_string(instructionIndex), message);
    }
    bool failAtOperand(uint32_t instructionIndex, bool isDestination, size_t operand,
                       const std::string& message) {
        const std::string role = isDestination ? " dst" : " src";
        return fail(" #" + std::to_string(instructionIndex) + role + std::to_string(operand),
                    message);
    }

    /// Keys with a PHI at \p slot, ordered so diagnostics are reproducible.
    std::vector<RegKey> sortedPhiKeys(unsigned slot) const {
        std::vector<RegKey> keys;
        keys.reserve(phisAt_[slot].size());
        for (const auto& [key, phiID] : phisAt_[slot]) keys.push_back(key);
        std::sort(keys.begin(), keys.end(), regKeyLess);
        return keys;
    }

    unsigned slotOf(const BasicBlock* block) const {
        auto it = dominance_.rpoIndex.find(block);
        return it == dominance_.rpoIndex.end() ? DominanceInfo::kUndef : it->second;
    }

    uint32_t indexOf(const StinkyInstruction* instruction) const {
        auto it = instructionIndex_.find(instruction);
        return it == instructionIndex_.end() ? 0 : it->second;
    }

    /// Instructions of a block that carry dataflow, in program order.
    static std::vector<StinkyInstruction*> dataflowInstructions(BasicBlock& block) {
        std::vector<StinkyInstruction*> instructions;
        for (IRBase& ir : block) {
            auto* instruction = dyn_cast<StinkyInstruction>(&ir);
            if (instruction == nullptr) continue;
            if (instruction->getUnifiedOpcode() == GFX::LABEL) continue;
            instructions.push_back(instruction);
        }
        return instructions;
    }

    Function& function_;
    const DominanceInfo& dominance_;
    const LiftAsmRegistersToSSAOptions& options_;

    std::unordered_map<const StinkyInstruction*, uint32_t> instructionIndex_;
    std::vector<BlockFacts> facts_;
    std::vector<std::vector<unsigned>> domChildren_;
    std::vector<RegKeyMap<SSAPhiID>> phisAt_;
    RegKeyMap<ExposedUse> firstExposedUse_;
    RegKeyMap<std::vector<SSAValueID>> stacks_;

    std::string error_;
};

void Lifter::indexInstructions() {
    uint32_t index = 0;
    for (BasicBlock& bb : function_) {
        for (IRBase& ir : bb) {
            if (const auto* instruction = dyn_cast<StinkyInstruction>(&ir))
                instructionIndex_.emplace(instruction, index++);
        }
    }
}

bool Lifter::checkReachability() {
    for (BasicBlock& bb : function_) {
        if (slotOf(&bb) == DominanceInfo::kUndef) {
            const std::string label = bb.getLabel().empty() ? "<unlabelled>" : bb.getLabel();
            return fail("block ^" + label +
                        " is unreachable from the entry; dominance is undefined there, so "
                        "unreachable components are not lifted yet");
        }
    }

    // A live-in value arrives at the entry block without travelling along a CFG
    // edge. If the entry is also a loop header, its incoming values merge the
    // live-in with the back edge, and a PHI cannot express that: there is no
    // predecessor slot for "function entry". Such a PHI would only reference
    // itself. Requiring a distinct preheader keeps the model sound.
    const BasicBlock& entry = *function_.begin();
    if (!entry.getPredecessors().empty()) {
        return fail("the entry block ^" + entry.getLabel() +
                    " has incoming edges; a live-in reaching a loop header has no predecessor "
                    "edge to merge on, so the entry must not be a loop header");
    }
    return true;
}

bool Lifter::validateInstruction(const StinkyInstruction& instruction, uint32_t index) {
    if (instruction.getHwInstDesc() == nullptr)
        return failAt(index, "instruction has no hardware descriptor");

    if (instruction.getUnifiedOpcode() == GFX::PHI) {
        return failAt(index,
                      "analysis PHIs must be removed before lifting; canonical "
                      "PHIs live in the graph, not the instruction stream");
    }
    if (isCall(instruction)) {
        return failAt(index,
                      "call sites need a calling convention to describe argument, result, "
                      "and clobbered registers");
    }
    if (usesTrue16Halves(instruction))
        return failAt(index, "True16 half operands need sub-DWORD SSA units");

    for (size_t operand = 0; operand < instruction.getSrcRegs().size(); ++operand) {
        const OperandClass operandClass = classifyOperand(instruction.getSrcRegs()[operand]);
        if (operandClass.kind == OperandKind::Unsupported)
            return failAtOperand(index, /*isDestination=*/false, operand, operandClass.reason);
    }

    RegKeySet definedHere;
    for (size_t operand = 0; operand < instruction.getDestRegs().size(); ++operand) {
        const StinkyRegister& reg = instruction.getDestRegs()[operand];
        const OperandClass operandClass = classifyOperand(reg);
        if (operandClass.kind == OperandKind::Unsupported)
            return failAtOperand(index, /*isDestination=*/true, operand, operandClass.reason);

        for (size_t unit = 0; unit < operandClass.units; ++unit) {
            const RegKey key = toRegKey(reg, static_cast<unsigned>(unit));
            if (!definedHere.insert(key).second) {
                return failAtOperand(
                    index, /*isDestination=*/true, operand,
                    "defines " + regKeyToString(key) + " more than once in one instruction");
            }
        }
    }
    return true;
}

bool Lifter::gatherBlockFacts() {
    facts_.assign(dominance_.rpo.size(), BlockFacts{});

    // Function order, not RPO, so diagnostics report the earliest instruction.
    for (BasicBlock& bb : function_) {
        BlockFacts& facts = facts_[slotOf(&bb)];
        RegKeySet definedSoFar;

        for (StinkyInstruction* instruction : dataflowInstructions(bb)) {
            const uint32_t index = indexOf(instruction);
            if (!validateInstruction(*instruction, index)) return false;

            const std::vector<StinkyRegister>& srcRegs = instruction->getSrcRegs();
            for (size_t operand = 0; operand < srcRegs.size(); ++operand) {
                const OperandClass operandClass = classifyOperand(srcRegs[operand]);
                for (size_t unit = 0; unit < operandClass.units; ++unit) {
                    const RegKey key = toRegKey(srcRegs[operand], static_cast<unsigned>(unit));
                    if (definedSoFar.contains(key)) continue;
                    facts.upwardExposed.insert(key);
                    firstExposedUse_.emplace(key,
                                             ExposedUse{index, static_cast<uint32_t>(operand)});
                }
            }

            const std::vector<StinkyRegister>& destRegs = instruction->getDestRegs();
            for (size_t operand = 0; operand < destRegs.size(); ++operand) {
                const OperandClass operandClass = classifyOperand(destRegs[operand]);
                for (size_t unit = 0; unit < operandClass.units; ++unit) {
                    const RegKey key = toRegKey(destRegs[operand], static_cast<unsigned>(unit));
                    definedSoFar.insert(key);
                    facts.defs.insert(key);
                }
            }
        }
    }
    return true;
}

void Lifter::computeLiveness() {
    // Backward fixpoint: liveIn[B] = upwardExposed[B] + (liveOut[B] - defs[B]).
    // Liveness is what prunes PHI placement, so no dead PHI is ever created.
    for (size_t slot = 0; slot < facts_.size(); ++slot)
        facts_[slot].liveIn = facts_[slot].upwardExposed;

    bool changed = true;
    while (changed) {
        changed = false;
        // Reverse RPO converges quickly for reducible CFGs and still terminates
        // for irreducible ones.
        for (size_t reverse = facts_.size(); reverse > 0; --reverse) {
            const unsigned slot = static_cast<unsigned>(reverse - 1);
            BlockFacts& facts = facts_[slot];
            for (const BasicBlock* successor : dominance_.rpo[slot]->getSuccessors()) {
                const unsigned successorSlot = slotOf(successor);
                if (successorSlot == DominanceInfo::kUndef) continue;
                for (const RegKey& key : facts_[successorSlot].liveIn) {
                    if (facts.defs.contains(key)) continue;
                    if (facts.liveIn.insert(key).second) changed = true;
                }
            }
        }
    }
}

bool Lifter::createEntryLiveIns(CanonicalSSABuilder& builder) {
    const std::vector<RegKey> keys = sortedKeys(facts_[0].liveIn);
    if (!keys.empty() && !options_.allowInferredLiveIns) {
        const RegKey& key = keys.front();
        auto it = firstExposedUse_.find(key);
        const std::string where = it == firstExposedUse_.end()
                                      ? std::string{}
                                      : " #" + std::to_string(it->second.instructionIndex) +
                                            " src" + std::to_string(it->second.operand);
        return fail(where, "reads " + regKeyToString(key) + " with no reaching definition");
    }

    for (const RegKey& key : keys) {
        SSAValue liveIn;
        liveIn.kind = SSAValueKind::LiveIn;
        liveIn.origin = key;
        stacks_[key].push_back(builder.addValue(std::move(liveIn)));
    }
    return true;
}

void Lifter::placePhis(CanonicalSSABuilder& builder) {
    const unsigned blockCount = static_cast<unsigned>(facts_.size());
    phisAt_.assign(blockCount, RegKeyMap<SSAPhiID>{});

    // Definition sites per key, including the entry for live-in keys so their
    // value participates in merges.
    RegKeyMap<std::vector<unsigned>> defSites;
    for (unsigned slot = 0; slot < blockCount; ++slot) {
        for (const RegKey& key : sortedKeys(facts_[slot].defs)) defSites[key].push_back(slot);
    }
    for (const RegKey& key : sortedKeys(facts_[0].liveIn)) {
        std::vector<unsigned>& sites = defSites[key];
        if (sites.empty() || sites.front() != 0) sites.insert(sites.begin(), 0);
    }

    RegKeySet allKeys;
    for (const auto& [key, sites] : defSites) allKeys.insert(key);

    std::vector<unsigned> worklist;
    std::unordered_set<unsigned> queued;
    for (const RegKey& key : sortedKeys(allKeys)) {
        const std::vector<unsigned>& sites = defSites[key];
        worklist.assign(sites.begin(), sites.end());
        queued.clear();
        queued.insert(sites.begin(), sites.end());

        while (!worklist.empty()) {
            const unsigned block = worklist.back();
            worklist.pop_back();

            for (unsigned frontier : dominance_.df[block]) {
                if (phisAt_[frontier].count(key) != 0) continue;
                // Pruned SSA: a merge only matters where the value is live.
                if (facts_[frontier].liveIn.contains(key)) {
                    SSAValue result;
                    result.kind = SSAValueKind::Phi;
                    result.origin = key;
                    const SSAValueID resultID = builder.addValue(std::move(result));

                    SSAPhi phi;
                    phi.block = dominance_.rpo[frontier];
                    phi.origin = key;
                    phi.result = resultID;
                    for (BasicBlock* predecessor : phi.block->getPredecessors())
                        phi.incoming.push_back(SSAPhiIncoming{predecessor, kInvalidSSAValueID});

                    const SSAPhiID phiID = builder.addPhi(std::move(phi));
                    builder.value(resultID).definingPhi = phiID;
                    builder.addPhiToBlock(*dominance_.rpo[frontier], phiID);
                    phisAt_[frontier].emplace(key, phiID);
                }
                if (queued.insert(frontier).second) worklist.push_back(frontier);
            }
        }
    }
}

void Lifter::renameBlock(CanonicalSSABuilder& builder, unsigned slot,
                         std::vector<RegKey>& pushedKeys) {
    BasicBlock* block = dominance_.rpo[slot];

    // PHI results are the values arriving at block entry.
    for (const auto& [key, phiID] : phisAt_[slot]) {
        stacks_[key].push_back(builder.phi(phiID).result);
        pushedKeys.push_back(key);
    }

    for (StinkyInstruction* instruction : dataflowInstructions(*block)) {
        SSAInstructionInfo info;

        // Sources first, so a read-modify-write operand reads the old value.
        const std::vector<StinkyRegister>& srcRegs = instruction->getSrcRegs();
        info.sources.resize(srcRegs.size());
        for (size_t operand = 0; operand < srcRegs.size(); ++operand) {
            const OperandClass operandClass = classifyOperand(srcRegs[operand]);
            for (size_t unit = 0; unit < operandClass.units; ++unit) {
                const RegKey key = toRegKey(srcRegs[operand], static_cast<unsigned>(unit));
                std::vector<SSAValueID>& stack = stacks_[key];
                // Liveness guaranteed an entry value for anything read without a
                // definition, so the stack cannot be empty here.
                const SSAValueID id = stack.empty() ? kInvalidSSAValueID : stack.back();
                info.sources[operand].units.push_back(id);
                if (id == kInvalidSSAValueID) continue;

                SSAUse use;
                use.instruction = instruction;
                use.operand = static_cast<uint32_t>(operand);
                use.unit = static_cast<uint32_t>(unit);
                builder.value(id).uses.push_back(use);
            }
        }

        const std::vector<StinkyRegister>& destRegs = instruction->getDestRegs();
        info.destinations.resize(destRegs.size());
        for (size_t operand = 0; operand < destRegs.size(); ++operand) {
            const OperandClass operandClass = classifyOperand(destRegs[operand]);
            for (size_t unit = 0; unit < operandClass.units; ++unit) {
                const RegKey key = toRegKey(destRegs[operand], static_cast<unsigned>(unit));

                SSAValue defined;
                defined.kind = SSAValueKind::InstructionDef;
                defined.origin = key;
                defined.definingInstruction = instruction;
                defined.definingOperand = static_cast<uint32_t>(operand);
                defined.definingUnit = static_cast<uint32_t>(unit);
                const SSAValueID id = builder.addValue(std::move(defined));

                info.destinations[operand].units.push_back(id);
                stacks_[key].push_back(id);
                pushedKeys.push_back(key);
            }
        }

        builder.setInstructionInfo(*instruction, std::move(info));
    }

    // Hand this block's exit values to the PHIs of every successor edge. A block
    // can appear as a predecessor more than once, so fill every matching slot.
    for (const BasicBlock* successor : block->getSuccessors()) {
        const unsigned successorSlot = slotOf(successor);
        if (successorSlot == DominanceInfo::kUndef) continue;

        for (const auto& [key, phiID] : phisAt_[successorSlot]) {
            const std::vector<SSAValueID>& stack = stacks_[key];
            if (stack.empty()) continue;
            const SSAValueID id = stack.back();

            SSAPhi& phi = builder.phi(phiID);
            for (size_t edge = 0; edge < phi.incoming.size(); ++edge) {
                if (phi.incoming[edge].predecessor != block) continue;
                if (phi.incoming[edge].value != kInvalidSSAValueID) continue;
                phi.incoming[edge].value = id;

                SSAUse use;
                use.phi = phiID;
                use.predecessor = block;
                builder.value(id).uses.push_back(use);
            }
        }
    }
}

bool Lifter::rename(CanonicalSSABuilder& builder) {
    const unsigned blockCount = static_cast<unsigned>(facts_.size());
    domChildren_.assign(blockCount, {});
    for (unsigned slot = 1; slot < blockCount; ++slot) {
        const unsigned parent = dominance_.idom[slot];
        if (parent != slot && parent < blockCount) domChildren_[parent].push_back(slot);
    }

    // Explicit stack rather than recursion: dominator trees can be as deep as
    // the block count in long straight-line kernels.
    struct Frame {
        unsigned block;
        size_t nextChild = 0;
        std::vector<RegKey> pushedKeys;
    };

    std::vector<Frame> frames;
    frames.push_back(Frame{0});
    renameBlock(builder, 0, frames.back().pushedKeys);

    while (!frames.empty()) {
        const size_t top = frames.size() - 1;
        const unsigned block = frames[top].block;
        if (frames[top].nextChild < domChildren_[block].size()) {
            const unsigned child = domChildren_[block][frames[top].nextChild++];
            frames.push_back(Frame{child});
            renameBlock(builder, child, frames.back().pushedKeys);
            continue;
        }

        const std::vector<RegKey>& pushed = frames[top].pushedKeys;
        for (auto key = pushed.rbegin(); key != pushed.rend(); ++key) stacks_[*key].pop_back();
        frames.pop_back();
    }

    // Every reachable predecessor edge is visited exactly once, so an unfilled
    // slot would mean the dominator walk missed a block.
    for (unsigned slot = 0; slot < blockCount; ++slot) {
        for (const RegKey& key : sortedPhiKeys(slot)) {
            const SSAPhi& phi = builder.phi(phisAt_[slot].at(key));
            for (size_t edge = 0; edge < phi.incoming.size(); ++edge) {
                if (phi.incoming[edge].value != kInvalidSSAValueID) continue;
                return fail("phi for " + regKeyToString(key) + " in ^" +
                            dominance_.rpo[slot]->getLabel() + " has no value on edge " +
                            std::to_string(edge));
            }
        }
    }
    return true;
}

Expected<CanonicalSSA> Lifter::run() {
    // Stamped even when empty, so every lifted graph can be checked against the
    // program it came from and only hand-built graphs are exempt.
    if (function_.empty()) {
        CanonicalSSABuilder empty;
        empty.setShape(computeFunctionShape(function_));
        return empty.take();
    }

    indexInstructions();
    if (!checkReachability()) return Result::Error(error_);
    if (!gatherBlockFacts()) return Result::Error(error_);

    computeLiveness();

    CanonicalSSABuilder builder;
    if (!createEntryLiveIns(builder)) return Result::Error(error_);
    placePhis(builder);
    if (!rename(builder)) return Result::Error(error_);
    builder.setShape(computeFunctionShape(function_));

    CanonicalSSA ssa = builder.take();
    if (options_.verify) {
        const CanonicalSSAVerificationResult verification =
            verifyCanonicalSSA(function_, ssa, dominance_);
        if (!verification.ok()) {
            fail("canonical SSA verification failed:\n" + verification.toString());
            return Result::Error(error_);
        }
    }
    return ssa;
}

}  // namespace

Expected<CanonicalSSA> liftAsmRegistersToSSA(Function& function, const DominanceInfo& dominance,
                                             const LiftAsmRegistersToSSAOptions& options) {
    return Lifter(function, dominance, options).run();
}

Expected<CanonicalSSA> liftAsmRegistersToSSA(Function& function,
                                             const LiftAsmRegistersToSSAOptions& options) {
    const DominanceInfo dominance = computeDominanceInfo(function);
    return liftAsmRegistersToSSA(function, dominance, options);
}

}  // namespace stinkytofu
