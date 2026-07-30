// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/Gfx1250HazardPass.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/RegisterKey.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace {
using namespace stinkytofu;

enum class MemoryGroupKind {
    None,
    SMEM,
    VMEM,
    Other,
};

enum class ReplayMode {
    SingleGroup,
    MultiGroup,
};

// This pass currently implements only the single-group policy. Keep the
// mapping explicit because gfx1250 categorizes tensor memory differently in
// multi-group mode.
constexpr ReplayMode kReplayMode = ReplayMode::SingleGroup;

// A group is consecutive memory instructions of the same hardware type:
// SMEM, VMEM (global/flat/scratch/buffer), or Other (TDM in single-group
// mode). GroupState records the type, whether the group has non-atomic memory,
// and source register DWORDs that must remain intact until it drains.
struct GroupState {
    MemoryGroupKind kind = MemoryGroupKind::None;
    bool hasMemory = false;
    bool hasNonAtomic = false;
    RegKeySet sources;

    void clear() {
        kind = MemoryGroupKind::None;
        hasMemory = false;
        hasNonAtomic = false;
        sources.clear();
    }
};

MemoryGroupKind getMemoryGroupKind(const StinkyInstruction& inst, ReplayMode replayMode) {
    if (isSMemLoad(inst) || isSMemStore(inst) || inst.is(InstFlag::IF_SMemAtomic))
        return MemoryGroupKind::SMEM;
    if (isMUBUFLoad(inst) || isMUBUFStore(inst) || isMUBUFAtomic(inst) || isFLATLoad(inst) ||
        isFLATStore(inst) || isFLATAtomic(inst) || isGLOBALLoad(inst) || isGLOBALStore(inst) ||
        isGLOBALAtomic(inst))
        return MemoryGroupKind::VMEM;
    if (isTensorLoad(inst))
        return replayMode == ReplayMode::SingleGroup ? MemoryGroupKind::Other
                                                     : MemoryGroupKind::VMEM;
    return MemoryGroupKind::None;
}

bool isAtomic(const StinkyInstruction& inst) {
    return isGlobalMemAtomic(inst);
}

bool isFlat(const StinkyInstruction& inst) {
    return isFLATLoad(inst) || isFLATStore(inst) || isFLATAtomic(inst);
}

bool hasDestSourceOverlap(const StinkyInstruction& inst, const RegKeySet& sources) {
    for (const StinkyRegister& dest : inst.getDestRegs()) {
        bool overlaps = false;
        forEachRegUnit(dest, [&](RegKey key) { overlaps |= sources.contains(key); });
        if (overlaps) return true;
    }
    return false;
}

void addSources(RegKeySet& sources, const StinkyInstruction& inst) {
    for (const StinkyRegister& src : inst.getSrcRegs())
        forEachRegUnit(src, [&](RegKey key) { sources.insert(key); });
}

bool hasSelfDestSourceOverlap(const StinkyInstruction& inst) {
    RegKeySet sources;
    addSources(sources, inst);
    return hasDestSourceOverlap(inst, sources);
}

bool violatesFlatSourceRule(const StinkyInstruction& inst, const GroupState& state) {
    if (!state.hasMemory) return false;  // A single-instruction group may self-overlap.
    return hasDestSourceOverlap(inst, state.sources) || hasSelfDestSourceOverlap(inst);
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

bool isScalarPrefetch(const StinkyInstruction& inst) {
    return inst.getUnifiedOpcode() == GFX::s_prefetch_inst_pc_rel;
}

bool isImmediateMemorySuccessor(BasicBlock::iterator it, BasicBlock& bb, ReplayMode replayMode) {
    for (auto next = std::next(it); next != bb.end(); ++next) {
        auto* inst = dyn_cast<StinkyInstruction>(next.getNodePtr());
        if (inst == nullptr || isPseudoInst(inst)) continue;
        return getMemoryGroupKind(*inst, replayMode) != MemoryGroupKind::None;
    }
    return false;
}

void addSources(GroupState& state, const StinkyInstruction& inst) {
    addSources(state.sources, inst);
}

void warnUnrepairableSmemSelfOverlap(const BasicBlock& bb, const StinkyInstruction& inst) {
    std::cerr << "[Gfx1250HazardPass] warning: SMEM instruction \""
              << inst.getHwInstDesc()->mnemonic << "\" in block \"" << bb.getLabel()
              << "\" overwrites one of its own source registers; "
                 "s_wait_xcnt cannot repair this register allocation\n";
}

void assertFallthrough(const BasicBlock& previous, const BasicBlock& next) {
    const auto& successors = previous.getSuccessors();
    const auto& predecessors = next.getPredecessors();

    // `next` may also have branch predecessors; those paths arrive with XCNT
    // drained. Only the physical predecessor carrying this group's state must
    // be an unconditional fall-through to `next`.
    assert(successors.size() == 1 && successors.front() == &next &&
           "an open replay group must reach the next physical block by fall-through");
    assert(std::find(predecessors.begin(), predecessors.end(), &previous) != predecessors.end() &&
           "fall-through successor is missing its predecessor edge");
}

class Gfx1250HazardPass : public Pass {
   public:
    static char ID;

    explicit Gfx1250HazardPass(std::vector<Function*> functions)
        : functions(std::move(functions)) {}

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
    static void insertXcntDrain(AsmIRBuilder& builder, GfxArchID archId, StinkyInstruction* anchor,
                                GroupState& state) {
        StinkyInstruction* wait = builder.create(getMCIDByUOp(GFX::s_wait_xcnt, archId), anchor);
        wait->addSrcReg(StinkyRegister(0));
        state.clear();
    }

    // UTC translates virtual addresses to physical addresses. During page
    // migration, an unavailable translation may stall or return XNACK-Retry.
    // Retry lets the wave keep issuing instructions before earlier memory
    // translations finish, then replays the faulting instruction. This fix
    // preserves replay sources and inserts required drains.
    static void applySingleGroupXnackReplayFix(BasicBlock& bb, BasicBlock::iterator it,
                                               AsmIRBuilder& builder, GfxArchID archId,
                                               GroupState& state) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr || isPseudoInst(inst)) return;

        if (isFullXcntDrain(*inst)) {
            state.clear();
            return;
        }

        if (isForeverSleep(*inst)) {
            if (state.hasMemory) insertXcntDrain(builder, archId, inst, state);
            // s_sleep is a non-memory single-group boundary.
            state.clear();
            return;
        }

        if (isScalarPrefetch(*inst)) {
            if (state.hasMemory) insertXcntDrain(builder, archId, inst, state);
            state.clear();
            return;
        }

        if (inst->getUnifiedOpcode() == GFX::s_set_vgpr_msb) {
            if (!isImmediateMemorySuccessor(it, bb, kReplayMode) && state.hasMemory)
                insertXcntDrain(builder, archId, inst, state);
            // s_set_vgpr_msb is a non-memory single-group boundary.
            state.clear();
            return;
        }

        const MemoryGroupKind kind = getMemoryGroupKind(*inst, kReplayMode);
        if (kind == MemoryGroupKind::None) {
            // In single-group mode hardware drains XCNT before every real
            // non-memory instruction, including control flow.
            state.clear();
            return;
        }

        // Single-group hardware drains XCNT at an SMEM/VMEM/Other type
        // switch, so no replay sources survive into the new group.
        //
        // Note: Multi-group only fences VMEM <-> SMEM and needs cross-group
        //       dataflow analysis (Future TODO if frontend enables
        //       Multi-group mode).
        if (state.hasMemory && state.kind != kind) state.clear();

        const bool atomic = isAtomic(*inst);
        if (atomic && state.hasNonAtomic) insertXcntDrain(builder, archId, inst, state);

        // Single-group replay-source rules:
        //
        // 1. Global / Buffer / Scratch / Image:
        //    (a) in order; an op may overwrite its own or a peer's source.
        // 2. FLAT:
        //    (a) must not overwrite a group source;
        //    (b) a single instruction FLAT group may overwrite its own source.
        // 3. SMEM:
        //    (a) no instruction may overwrite any group source.
        // 4. Atomic / RMW:
        //    (a) drain before the first atomic after non-atomic memory;
        //    (b) consecutive atomics need no intervening drain due to the
        //        XNACK scoreboard.

        // Apply rule 3:
        if (kind == MemoryGroupKind::SMEM && hasSelfDestSourceOverlap(*inst))
            warnUnrepairableSmemSelfOverlap(bb, *inst);

        if (kind == MemoryGroupKind::SMEM && state.hasMemory &&
            hasDestSourceOverlap(*inst, state.sources)) {
            insertXcntDrain(builder, archId, inst, state);
        }

        // Rule 2: a FLAT must not clobber a source in a multi-op group.
        const bool violatesRule2 =
            kind == MemoryGroupKind::VMEM && isFlat(*inst) && violatesFlatSourceRule(*inst, state);

        // Rule 4(b): do not drain between consecutive atomics.
        const bool exemptByRule4b = atomic && state.hasMemory && !state.hasNonAtomic;
        if (violatesRule2 && !exemptByRule4b) {
            insertXcntDrain(builder, archId, inst, state);
        }

        state.kind = kind;
        state.hasMemory = true;
        state.hasNonAtomic |= !atomic;
        addSources(state, *inst);
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
                applySingleGroupXnackReplayFix(bb, it, builder, archId, state);
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
