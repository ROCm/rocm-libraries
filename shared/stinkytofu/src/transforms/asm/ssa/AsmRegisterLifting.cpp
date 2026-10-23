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
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "stinkytofu/analysis/asm/ssa/SSAFunctionShape.hpp"
#include "stinkytofu/analysis/controlflow/Dominance.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/RegisterKey.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/ir/asm/ssa/AttachedSSAVerifier.hpp"
#include "stinkytofu/ir/asm/ssa/SSAOperandUnits.hpp"
#include "stinkytofu/ir/asm/ssa/StinkyOpOperand.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/ssa/LiftAsmRegistersToSSAPass.hpp"

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

OperandClass classifyOperand(const StinkyRegister& reg, const RegClassSet& classes) {
    if (!reg.isRegister()) return {OperandKind::Ignored, 0, {}};
    if (reg.isVirtualReg())
        return {OperandKind::Unsupported, 0,
                "unresolved template virtual register; resolve it before lifting"};
    if (isPseudoReg(reg)) return {OperandKind::Ignored, 0, {}};
    if (!isAllocatableReg(reg.reg.type)) return {OperandKind::Ignored, 0, {}};
    if (!isLiftableRegClass(reg.reg.type))
        return {OperandKind::Unsupported, 0,
                "register class '" + regTypeToString(reg.reg.type) +
                    "' is not lifted yet; VGPRs and SGPRs are supported"};
    // Liftable, but this lift was not asked for it. Deliberately leaving a class
    // physical is not an error: the operand carries an immediate payload and SSA
    // destruction never rewrites it, so it survives exactly as written.
    if (!classes.contains(reg.reg.type)) return {OperandKind::Ignored, 0, {}};
    return {OperandKind::AllocatableRange, liftedSSAUnits(reg, classes), {}};
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

/// Name of a register operand this lift covers, or nullopt when it covers none.
/// Any witness will do; only existence decides. Not the operand the True16
/// selector names, which is indexed by printed position: coarse but never unsound.
std::optional<std::string> inScopeOperandName(const StinkyInstruction& instruction,
                                              const RegClassSet& classes) {
    const std::vector<StinkyRegister>& destRegs = instruction.getDestRegs();
    for (size_t operand = 0; operand < destRegs.size(); ++operand) {
        if (classifyOperand(destRegs[operand], classes).kind == OperandKind::AllocatableRange)
            return "dst" + std::to_string(operand);
    }
    const std::vector<StinkyRegister>& srcRegs = instruction.getSrcRegs();
    for (size_t operand = 0; operand < srcRegs.size(); ++operand) {
        if (classifyOperand(srcRegs[operand], classes).kind == OperandKind::AllocatableRange)
            return "src" + std::to_string(operand);
    }
    return std::nullopt;
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

std::string formatPhysicalRegister(const StinkyRegister& reg) {
    if (!reg.isRegister()) return "<non-register>";
    std::string text = regTypeToString(reg.reg.type) + std::to_string(reg.reg.idx);
    if (reg.reg.num > 1) {
        const uint32_t last = reg.reg.idx + reg.reg.num - 1;
        text += ":" + std::to_string(last);
    }
    if (reg.reg.offset != 0) text += "@" + std::to_string(reg.reg.offset);
    if (reg.reg.isAbs) text = "|" + text + "|";
    if (reg.reg.isMinus) text = "-" + text;
    return text;
}

LegacyImmPayload asImmediateOperand(const StinkyRegister& reg) {
    switch (reg.dataType) {
        case StinkyRegister::Type::LiteralInt:
            return static_cast<int32_t>(reg.getLiteralInt());
        case StinkyRegister::Type::LiteralDouble:
            return reg.getLiteralDouble();
        case StinkyRegister::Type::LiteralString:
            return reg.getLiteralString();
        case StinkyRegister::Type::HwReg:
            return HwRegPayload{reg.hwreg.id, reg.hwreg.offset, reg.hwreg.size};
        case StinkyRegister::Type::Register:
            return formatPhysicalRegister(reg);
        case StinkyRegister::Type::Invalid:
            return std::monostate{};
    }
    return std::monostate{};
}

StinkySSAValue::PhysicalBinding bindingFromReg(const StinkyRegister& reg, uint32_t unit) {
    StinkySSAValue::PhysicalBinding binding;
    binding.type = reg.reg.type;
    binding.idx = reg.reg.idx + unit;
    binding.num = 1;
    binding.offset = reg.reg.offset;
    binding.isVirtual = reg.isVirtualReg();
    binding.isMinus = reg.reg.isMinus;
    binding.isAbs = reg.reg.isAbs;
    return binding;
}

StinkySSAValue::PhysicalBinding bindingFromKey(const RegKey& key) {
    StinkySSAValue::PhysicalBinding binding;
    binding.type = key.type;
    binding.idx = key.idx;
    binding.num = 1;
    return binding;
}

class Lifter {
   public:
    Lifter(Function& function, const DominanceInfo& dominance,
           const LiftAsmRegistersToSSAOptions& options)
        : function_(function), dominance_(dominance), options_(options) {}

    Expected<LiftAttachedSSAResult> run();

   private:
    using Result = Expected<LiftAttachedSSAResult>;

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

    struct PhiPlacement {
        StinkySSAValue* value = nullptr;
        size_t argIndex = 0;
    };

    // Setup and validation.
    void indexInstructions();
    bool checkReachability();
    bool gatherBlockFacts();
    bool validateInstruction(const StinkyInstruction& instruction, uint32_t index);

    // SSA construction.
    void computeLiveness();
    bool createEntryLiveIns();
    void placePhis();
    bool rename();
    void renameBlock(unsigned block, std::vector<RegKey>& pushedKeys);

    /// Record what this arena describes: the program it was built from, and the
    /// classes it was built for. Only on success, so a rejected lift leaves no
    /// claim behind. Both are needed, because the shape hashes the physical
    /// program, which is identical whichever classes were lifted.
    void stamp() {
        function_.ssaArena().setShape(computeFunctionShape(function_));
        function_.ssaArena().setLiftedClasses(options_.classes);
    }

    LiftAttachedSSAResult counts() const {
        LiftAttachedSSAResult result;
        result.valueCount = function_.ssaArena().valueCount();
        for (const BasicBlock& bb : function_)
            result.blockArgumentCount += bb.ssaArguments().size();
        return result;
    }

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
        for (const auto& [key, phi] : phisAt_[slot]) keys.push_back(key);
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
    std::vector<RegKeyMap<PhiPlacement>> phisAt_;
    RegKeyMap<ExposedUse> firstExposedUse_;
    RegKeyMap<std::vector<StinkySSAValue*>> stacks_;

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
                      "analysis PHIs must be removed before lifting; SSA merges "
                      "are block arguments, not the instruction stream");
    }
    if (isCall(instruction)) {
        return failAt(index,
                      "call sites need a calling convention to describe argument, result, "
                      "and clobbered registers");
    }
    // The selector records no register class, so it cannot say whether it is in
    // scope. An operand left physical keeps its register through destruction and
    // its selector on the modifier, so a half there cannot reach any SSA value.
    if (usesTrue16Halves(instruction)) {
        if (const std::optional<std::string> inScope =
                inScopeOperandName(instruction, options_.classes)) {
            return failAt(index, "True16 half operands need sub-DWORD SSA units; " + *inScope +
                                     " is in the lift scope");
        }
    }

    for (size_t operand = 0; operand < instruction.getSrcRegs().size(); ++operand) {
        const OperandClass operandClass =
            classifyOperand(instruction.getSrcRegs()[operand], options_.classes);
        if (operandClass.kind == OperandKind::Unsupported)
            return failAtOperand(index, /*isDestination=*/false, operand, operandClass.reason);
    }

    RegKeySet definedHere;
    for (size_t operand = 0; operand < instruction.getDestRegs().size(); ++operand) {
        const StinkyRegister& reg = instruction.getDestRegs()[operand];
        const OperandClass operandClass = classifyOperand(reg, options_.classes);
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
                const OperandClass operandClass =
                    classifyOperand(srcRegs[operand], options_.classes);
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
                const OperandClass operandClass =
                    classifyOperand(destRegs[operand], options_.classes);
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

bool Lifter::createEntryLiveIns() {
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

    BasicBlock& entry = *function_.begin();
    SSAArena& arena = function_.ssaArena();
    for (const RegKey& key : keys) {
        StinkySSAValue* liveIn = arena.createBlockArgument(key.type, 1);
        liveIn->setPhysicalBinding(bindingFromKey(key));
        entry.addSSAArgument(liveIn);
        stacks_[key].push_back(liveIn);
    }
    return true;
}

void Lifter::placePhis() {
    const unsigned blockCount = static_cast<unsigned>(facts_.size());
    phisAt_.assign(blockCount, RegKeyMap<PhiPlacement>{});

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

    SSAArena& arena = function_.ssaArena();
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
                    BasicBlock* join = dominance_.rpo[frontier];
                    StinkySSAValue* result = arena.createBlockArgument(key.type, 1);
                    result->setPhysicalBinding(bindingFromKey(key));
                    const size_t argIndex = join->ssaArguments().size();
                    join->addSSAArgument(result);
                    phisAt_[frontier].emplace(key, PhiPlacement{result, argIndex});
                }
                if (queued.insert(frontier).second) worklist.push_back(frontier);
            }
        }
    }
}

void Lifter::renameBlock(unsigned slot, std::vector<RegKey>& pushedKeys) {
    BasicBlock* block = dominance_.rpo[slot];
    SSAArena& arena = function_.ssaArena();

    // PHI results are the values arriving at block entry.
    for (const auto& [key, phi] : phisAt_[slot]) {
        stacks_[key].push_back(phi.value);
        pushedKeys.push_back(key);
    }

    for (StinkyInstruction* instruction : dataflowInstructions(*block)) {
        AttachedSSA attached;

        // Sources first, so a read-modify-write operand reads the old value.
        const std::vector<StinkyRegister>& srcRegs = instruction->getSrcRegs();
        for (size_t operand = 0; operand < srcRegs.size(); ++operand) {
            const OperandClass operandClass = classifyOperand(srcRegs[operand], options_.classes);
            if (operandClass.kind != OperandKind::AllocatableRange) {
                attached.operands.push_back(
                    makeSSAImmOperand(asImmediateOperand(srcRegs[operand])));
                continue;
            }
            for (size_t unit = 0; unit < operandClass.units; ++unit) {
                const RegKey key = toRegKey(srcRegs[operand], static_cast<unsigned>(unit));
                std::vector<StinkySSAValue*>& stack = stacks_[key];
                // Liveness guaranteed an entry value for anything read without a
                // definition, so the stack cannot be empty here.
                StinkySSAValue* value = stack.empty() ? nullptr : stack.back();
                attached.operands.push_back(makeSSAValueOperand(value));
            }
        }

        const std::vector<StinkyRegister>& destRegs = instruction->getDestRegs();
        for (size_t operand = 0; operand < destRegs.size(); ++operand) {
            const OperandClass operandClass = classifyOperand(destRegs[operand], options_.classes);
            for (size_t unit = 0; unit < operandClass.units; ++unit) {
                const RegKey key = toRegKey(destRegs[operand], static_cast<unsigned>(unit));
                StinkySSAValue* defined = arena.createRegister(key.type, 1);
                defined->setPhysicalBinding(
                    bindingFromReg(destRegs[operand], static_cast<uint32_t>(unit)));
                if (destRegs[operand].hasSymbolicName())
                    defined->setSymbol(destRegs[operand].getSymbolicName());
                attached.results.push_back(defined);
                stacks_[key].push_back(defined);
                pushedKeys.push_back(key);
            }
        }

        instruction->attachSSA(std::move(attached));
    }

    // Hand this block's exit values to the PHIs of every successor edge. A block
    // can appear as a predecessor more than once, so fill every matching slot.
    for (const BasicBlock* successor : block->getSuccessors()) {
        const unsigned successorSlot = slotOf(successor);
        if (successorSlot == DominanceInfo::kUndef) continue;
        BasicBlock* successorBlock = dominance_.rpo[successorSlot];

        size_t edgeCount = 0;
        for (const BasicBlock* predecessor : successorBlock->getPredecessors()) {
            if (predecessor == block) ++edgeCount;
        }
        if (edgeCount == 0) continue;

        for (const auto& [key, phi] : phisAt_[successorSlot]) {
            const std::vector<StinkySSAValue*>& stack = stacks_[key];
            if (stack.empty()) continue;
            StinkySSAValue* value = stack.back();
            const SSABlockArgument& arg = successorBlock->ssaArguments()[phi.argIndex];
            size_t already = 0;
            for (const SSABlockIncoming& incoming : arg.incoming) {
                if (incoming.predecessor == block) ++already;
            }
            for (size_t edge = already; edge < edgeCount; ++edge)
                successorBlock->setSSAArgumentIncoming(phi.argIndex, block, value);
        }
    }
}

bool Lifter::rename() {
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
    renameBlock(0, frames.back().pushedKeys);

    while (!frames.empty()) {
        const size_t top = frames.size() - 1;
        const unsigned block = frames[top].block;
        if (frames[top].nextChild < domChildren_[block].size()) {
            const unsigned child = domChildren_[block][frames[top].nextChild++];
            frames.push_back(Frame{child});
            renameBlock(child, frames.back().pushedKeys);
            continue;
        }

        const std::vector<RegKey>& pushed = frames[top].pushedKeys;
        for (auto key = pushed.rbegin(); key != pushed.rend(); ++key) stacks_[*key].pop_back();
        frames.pop_back();
    }

    // Every reachable predecessor edge is visited exactly once, so an unfilled
    // slot would mean the dominator walk missed a block.
    for (unsigned slot = 0; slot < blockCount; ++slot) {
        BasicBlock* block = dominance_.rpo[slot];
        const size_t predCount = block->getPredecessors().size();
        for (const RegKey& key : sortedPhiKeys(slot)) {
            const PhiPlacement& phi = phisAt_[slot].at(key);
            const size_t incoming = block->ssaArguments()[phi.argIndex].incoming.size();
            if (incoming == predCount) continue;
            return fail("phi for " + regKeyToString(key) + " in ^" + block->getLabel() +
                        " has no value on edge " + std::to_string(incoming));
        }
    }
    return true;
}

Expected<LiftAttachedSSAResult> Lifter::run() {
    function_.clearAttachedSSA();

    if (options_.classes.empty()) {
        fail("no register classes to lift; narrowing the scope to nothing is not a lift");
        return Result::Error(error_);
    }

    if (function_.empty()) {
        stamp();
        return counts();
    }

    indexInstructions();
    if (!checkReachability()) return Result::Error(error_);
    if (!gatherBlockFacts()) return Result::Error(error_);

    computeLiveness();
    if (!createEntryLiveIns()) {
        function_.clearAttachedSSA();
        return Result::Error(error_);
    }
    placePhis();
    if (!rename()) {
        function_.clearAttachedSSA();
        return Result::Error(error_);
    }

    stamp();
    if (options_.verify) {
        const AttachedSSAVerificationResult verification = verifyAttachedSSA(function_);
        if (!verification.ok()) {
            fail("attached SSA verification failed:\n" + verification.toString());
            function_.clearAttachedSSA();
            return Result::Error(error_);
        }
    }
    return counts();
}

}  // namespace

Expected<LiftAttachedSSAResult> liftAsmRegistersToAttachedSSA(
    Function& function, const DominanceInfo& dominance,
    const LiftAsmRegistersToSSAOptions& options) {
    return Lifter(function, dominance, options).run();
}

Expected<LiftAttachedSSAResult> liftAsmRegistersToAttachedSSA(
    Function& function, const LiftAsmRegistersToSSAOptions& options) {
    const DominanceInfo dominance = computeDominanceInfo(function);
    return liftAsmRegistersToAttachedSSA(function, dominance, options);
}

}  // namespace stinkytofu
