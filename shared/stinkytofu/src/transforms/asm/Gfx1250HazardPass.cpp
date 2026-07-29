// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/Gfx1250HazardPass.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace {
using namespace stinkytofu;

enum class MemoryGroupKind {
    None,
    SMEM,
    VMEM,
    TDM,
};

struct GroupState {
    MemoryGroupKind kind = MemoryGroupKind::None;
    bool hasMemory = false;
    bool hasNonAtomic = false;
    std::vector<StinkyRegister> sources;

    void clear() {
        kind = MemoryGroupKind::None;
        hasMemory = false;
        hasNonAtomic = false;
        sources.clear();
    }
};

MemoryGroupKind getMemoryGroupKind(const StinkyInstruction& inst) {
    if (isSMemLoad(inst) || isSMemStore(inst) || inst.is(InstFlag::IF_SMemAtomic))
        return MemoryGroupKind::SMEM;
    if (isMUBUFLoad(inst) || isMUBUFStore(inst) || isMUBUFAtomic(inst) ||
        isFLATLoad(inst) || isFLATStore(inst) || isFLATAtomic(inst) ||
        isGLOBALLoad(inst) || isGLOBALStore(inst) || isGLOBALAtomic(inst))
        return MemoryGroupKind::VMEM;
    if (isTensorLoad(inst)) return MemoryGroupKind::TDM;
    return MemoryGroupKind::None;
}

bool isAtomic(const StinkyInstruction& inst) {
    return isGlobalMemAtomic(inst);
}

bool isFlat(const StinkyInstruction& inst) {
    return isFLATLoad(inst) || isFLATStore(inst) || isFLATAtomic(inst);
}

bool hasDestSourceOverlap(const StinkyInstruction& inst,
                          const std::vector<StinkyRegister>& sources) {
    for (const StinkyRegister& dest : inst.getDestRegs()) {
        if (!dest.isRegister()) continue;
        for (const StinkyRegister& src : sources)
            if (dest.isOverlap(src)) return true;
    }
    return false;
}

bool hasSelfDestSourceOverlap(const StinkyInstruction& inst) {
    return hasDestSourceOverlap(inst, inst.getSrcRegs());
}

bool isFullXcntDrain(const StinkyInstruction& inst) {
    if (inst.getUnifiedOpcode() != GFX::s_wait_xcnt) return false;
    const auto& srcs = inst.getSrcRegs();
    return srcs.size() == 1 && srcs.front().dataType == StinkyRegister::Type::LiteralInt &&
           srcs.front().getLiteralInt() == 0;
}

bool isForeverSleep(const StinkyInstruction& inst) {
    if (inst.getUnifiedOpcode() != GFX::s_sleep) return false;
    const auto& srcs = inst.getSrcRegs();
    if (srcs.size() != 1 || srcs.front().dataType != StinkyRegister::Type::LiteralInt) return false;
    return (static_cast<uint16_t>(srcs.front().getLiteralInt()) & 0x8000U) != 0;
}

bool isImmediateMemorySuccessor(BasicBlock::iterator it, BasicBlock& bb) {
    for (auto next = std::next(it); next != bb.end(); ++next) {
        auto* inst = dyn_cast<StinkyInstruction>(next.getNodePtr());
        if (inst == nullptr || isPseudoInst(inst)) continue;
        return getMemoryGroupKind(*inst) != MemoryGroupKind::None;
    }
    return false;
}

void addSources(GroupState& state, const StinkyInstruction& inst) {
    for (const StinkyRegister& src : inst.getSrcRegs())
        if (src.isRegister()) state.sources.push_back(src);
}

void assertFallthrough(const BasicBlock& previous, const BasicBlock& next) {
    const auto& successors = previous.getSuccessors();
    const auto& predecessors = next.getPredecessors();

    assert(successors.size() == 1 && successors.front() == &next &&
           "an open replay group must reach the next physical block by fall-through");
    assert(std::find(predecessors.begin(), predecessors.end(), &previous) != predecessors.end() &&
           "fall-through successor is missing its predecessor edge");
}

class Gfx1250HazardPass : public Pass {
   public:
    static char ID;

    explicit Gfx1250HazardPass(std::vector<Function*> functions) : functions(std::move(functions)) {}

    const char* getName() const override {
        return "Gfx1250HazardPass";
    }

    Pass::ID getPassID() const override {
        return &Gfx1250HazardPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        const auto arch = passCtx.getGemmTileConfig().arch;
        if (arch[0] != 12 || arch[1] != 5 || arch[2] != 0) return preserveCFGAnalyses();

        const GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);
        if (!functions.empty()) {
            for (Function* f : functions)
                if (f) runOnFunction(*f, archId);
        } else {
            runOnFunction(func, archId);
        }
        return preserveCFGAnalyses();
    }

   private:
    static void insertXcntDrain(AsmIRBuilder& builder, GfxArchID archId,
                                StinkyInstruction* anchor, GroupState& state) {
        StinkyInstruction* wait = builder.create(getMCIDByUOp(GFX::s_wait_xcnt, archId), anchor);
        wait->addSrcReg(StinkyRegister(0));
        state.clear();
    }

    static void runOnFunction(Function& func, GfxArchID archId) {
        GroupState state;
        BasicBlock* previous = nullptr;

        // CFG blocks can be split solely by a fall-through label. Labels do not
        // emit instructions or break a single-group replay group, so preserve
        // state while walking the physical block layout.
        for (BasicBlock& bb : func) {
            if (state.hasMemory && previous) assertFallthrough(*previous, bb);

            AsmIRBuilder builder(bb, archId);
            for (auto it = bb.begin(); it != bb.end(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (inst == nullptr || isPseudoInst(inst)) continue;

                if (isFullXcntDrain(*inst)) {
                    state.clear();
                    continue;
                }

                if (isForeverSleep(*inst)) {
                    if (state.hasMemory) insertXcntDrain(builder, archId, inst, state);
                    // s_sleep is a non-memory single-group boundary.
                    state.clear();
                    continue;
                }

                if (inst->getUnifiedOpcode() == GFX::s_set_vgpr_msb) {
                    if (!isImmediateMemorySuccessor(it, bb) && state.hasMemory)
                        insertXcntDrain(builder, archId, inst, state);
                    // s_set_vgpr_msb is a non-memory single-group boundary.
                    state.clear();
                    continue;
                }

                const MemoryGroupKind kind = getMemoryGroupKind(*inst);
                if (kind == MemoryGroupKind::None) {
                    // In single-group mode hardware drains XCNT before every
                    // real non-memory instruction, including control flow.
                    state.clear();
                    continue;
                }

                if (state.hasMemory && state.kind != kind) state.clear();

                const bool atomic = isAtomic(*inst);
                if (atomic && state.hasNonAtomic) insertXcntDrain(builder, archId, inst, state);

                // SMEM groups replay as a unit. An individual SMEM instruction
                // with a self-overlapping destination is not repairable here:
                // its register allocation must already be valid. We can repair
                // only an overwrite of an earlier group member's source.
                if (kind == MemoryGroupKind::SMEM && state.hasMemory &&
                    hasDestSourceOverlap(*inst, state.sources)) {
                    insertXcntDrain(builder, archId, inst, state);
                }

                // A FLAT in a multi-instruction VMEM group must not overwrite
                // any group source, including its own source. Consecutive
                // atomics are exempt: the XNACK scoreboard guarantees their
                // exactly-once execution after the drain before the first one.
                const bool atomicOnlyGroup = state.hasMemory && !state.hasNonAtomic;
                if (kind == MemoryGroupKind::VMEM && isFlat(*inst) &&
                    !(atomic && atomicOnlyGroup)) {
                    const bool overwritesPriorSource =
                        state.hasMemory && hasDestSourceOverlap(*inst, state.sources);
                    const bool overwritesOwnSource =
                        state.hasMemory && hasSelfDestSourceOverlap(*inst);
                    if (overwritesPriorSource || overwritesOwnSource)
                        insertXcntDrain(builder, archId, inst, state);
                }

                state.kind = kind;
                state.hasMemory = true;
                state.hasNonAtomic |= !atomic;
                addSources(state, *inst);
            }
            previous = &bb;
        }
    }

    std::vector<Function*> functions;
};

char Gfx1250HazardPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createGfx1250HazardPass(std::vector<Function*> functions) {
    return std::make_unique<Gfx1250HazardPass>(std::move(functions));
}
}  // namespace stinkytofu
