// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/Gfx1250HazardPass.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/ModulePassManager.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/RegisterKey.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/transforms/asm/Gfx1250HazardProfile.hpp"

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

bool hasSelfDestSourceOverlap(const StinkyInstruction& inst) {
    RegKeySet sources;
    addSources(sources, inst);
    return hasDestSourceOverlap(inst, sources);
}

bool violatesFlatSourceRule(const StinkyInstruction& inst, const GroupState& state) {
    if (!state.hasMemory) return false;  // A single-instruction group may self-overlap.
    return hasDestSourceOverlap(inst, state.sources) || hasSelfDestSourceOverlap(inst);
}

bool violatesSmemSourceRule(const StinkyInstruction& inst, const GroupState& state) {
    if (!state.hasMemory) return false;  // A single-instruction clause may self-overlap.
    return hasDestSourceOverlap(inst, state.sources) || hasSelfDestSourceOverlap(inst);
}

// An SMEM load returning more than one DWORD can complete partially: some
// DWORDs return while others XNACK. Such a load is unreplayable once its source
// is gone, whereas a single-DWORD load returns all or nothing.
bool returnsMultipleDwords(const StinkyInstruction& inst) {
    unsigned dwords = 0;
    for (const StinkyRegister& dest : inst.getDestRegs())
        forEachRegUnit(dest, [&](RegKey) { ++dwords; });
    return dwords > 1;
}

bool hasZeroWaitImmediate(const StinkyInstruction& inst) {
    const auto& srcs = inst.getSrcRegs();
    return srcs.size() == 1 && srcs.front().dataType == StinkyRegister::Type::LiteralInt &&
           srcs.front().getLiteralInt() == 0;
}

// A zero-count data wait guarantees translation for the corresponding memory
// operations. Nonzero waits leave work in flight and are not full drains.
bool isFullTranslationDrain(const StinkyInstruction& inst) {
    switch (inst.getUnifiedOpcode()) {
        case GFX::s_wait_xcnt:
        case GFX::s_wait_loadcnt:
        case GFX::s_wait_kmcnt:
        case GFX::s_wait_dscnt:
        case GFX::s_wait_loadcnt_dscnt:
        case GFX::s_wait_tensorcnt:
        case GFX::s_wait_storecnt:
        case GFX::s_wait_storecnt_dscnt:
        case GFX::s_wait_asynccnt:
            return hasZeroWaitImmediate(inst);
        default:
            return false;
    }
}

bool isForeverSleep(const StinkyInstruction& inst) {
    if (inst.getUnifiedOpcode() != GFX::s_sleep) {
        return false;
    }
    const auto& srcs = inst.getSrcRegs();
    if (srcs.size() != 1 || srcs.front().dataType != StinkyRegister::Type::LiteralInt) {
        return false;
    }
    return (static_cast<uint16_t>(srcs.front().getLiteralInt()) & 0x8000U) != 0;
}

bool isScalarPrefetch(const StinkyInstruction& inst) {
    return inst.getUnifiedOpcode() == GFX::s_prefetch_inst_pc_rel;
}

bool isImmediateMemorySuccessor(BasicBlock::iterator it, BasicBlock& bb, ReplayMode replayMode) {
    for (auto next = std::next(it); next != bb.end(); ++next) {
        auto* inst = dyn_cast<StinkyInstruction>(next.getNodePtr());
        if (inst == nullptr || isPseudoInst(inst)) {
            continue;
        }
        return getMemoryGroupKind(*inst, replayMode) != MemoryGroupKind::None;
    }
    return false;
}

// Rules 2 and 3 are repaired by cutting the replay group short, which costs
// memory-level parallelism. Report it: a register allocation without the
// overlap would have kept the group intact.
void warnGroupBreak(const BasicBlock& bb, const StinkyInstruction& inst, const char* memKind) {
    std::cerr << "[Gfx1250HazardPass] warning: " << memKind << " instruction in block \""
              << bb.getLabel()
              << "\" overwrites a replay source of its group; "
                 "inserting s_wait_xcnt 0 to break the group:\n";
    inst.dump(std::cerr);
}

// A drain only separates the overlap from an earlier instruction. A multi-DWORD
// SMEM load that overwrites its own source has nothing to separate: it can
// partially complete, and the XNACKed DWORDs then replay with the source
// already clobbered. Only a different register allocation can fix it.
void reportUnrepairableSmemSelfOverlap(const BasicBlock& bb, const StinkyInstruction& inst) {
    std::cerr << "[Gfx1250HazardPass] error: multi-DWORD SMEM instruction in block \""
              << bb.getLabel()
              << "\" overwrites one of its own source registers; "
                 "s_wait_xcnt cannot repair this register allocation:\n";
    inst.dump(std::cerr);
    assert(false && "SMEM instruction overwrites its own XNACK replay source");
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

    explicit Gfx1250HazardPass(bool enableXcntDrainProfile)
        : enableXcntDrainProfile(enableXcntDrainProfile) {}

    const char* getName() const override {
        return "Gfx1250HazardPass";
    }

    Pass::ID getPassID() const override {
        return &Gfx1250HazardPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        const auto& caps = passCtx.getAsmCapsConfig();
        // Run if either flag is set: `RequiresXCntForVolatileVMEM` (atomics only) or
        // `EnableXnackReplay` (full replay protection including atomics).
        if (!caps.requiresXCntForVolatileVMEM && !caps.enableXnackReplay) {
            return preserveCFGAnalyses();
        }

        const auto arch = passCtx.getGemmTileConfig().arch;

        const GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);
        auto profile = makeXcntDrainProfile(enableXcntDrainProfile);
        runOnFunction(func, archId, *profile, caps.enableXnackReplay);
        profile->print();
        return preserveCFGAnalyses();
    }

   private:
    static void insertXcntDrain(AsmIRBuilder& builder, GfxArchID archId, StinkyInstruction* anchor,
                                GroupState& state, XcntDrainProfileBase& profile,
                                XcntDrainReason reason) {
        StinkyInstruction* wait = builder.create(getMCIDByUOp(GFX::s_wait_xcnt, archId), anchor);
        wait->addSrcReg(StinkyRegister(0));
        profile.record(reason, anchor);
        state.clear();
    }

    // UTC translates virtual addresses to physical addresses. During page
    // migration, an unavailable translation may stall or return XNACK-Retry.
    // Retry lets the wave keep issuing instructions before earlier memory
    // translations finish, then replays the faulting instruction. This fix
    // preserves replay sources and inserts required drains.
    static void applySingleGroupXnackReplayFix(BasicBlock& bb, BasicBlock::iterator it,
                                               AsmIRBuilder& builder, GfxArchID archId,
                                               GroupState& state, XcntDrainProfileBase& profile,
                                               bool enableXnackReplay) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr || isPseudoInst(inst)) return;

        if (isTensorLoad(*inst)) profile.noteTensorLoad();

        if (isFullTranslationDrain(*inst)) {
            state.clear();
            return;
        }
        if (enableXnackReplay) {
            if (isForeverSleep(*inst)) {
                if (state.hasMemory)
                    insertXcntDrain(builder, archId, inst, state, profile,
                                    XcntDrainReason::ForeverSleep);
                // s_sleep is a non-memory single-group boundary.
                state.clear();
                return;
            }

            // SW-prefetch runs later and owns its hints' XCnt waits; handle only
            // pre-existing scalar prefetches here.
            if (isScalarPrefetch(*inst)) {
                if (state.hasMemory)
                    insertXcntDrain(builder, archId, inst, state, profile,
                                    XcntDrainReason::ScalarPrefetch);
                state.clear();
                return;
            }

            if (inst->getUnifiedOpcode() == GFX::s_set_vgpr_msb) {
                if (!isImmediateMemorySuccessor(it, bb, kReplayMode) && state.hasMemory)
                    insertXcntDrain(builder, archId, inst, state, profile,
                                    XcntDrainReason::VgprMsb);
                // s_set_vgpr_msb is a non-memory single-group boundary.
                state.clear();
                return;
            }
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
        if (state.hasMemory && state.kind != kind) {
            state.clear();
        }

        // Single-group replay-source rules:
        //
        // 1. Global / Buffer / Scratch / Image:
        //    (a) in order; an op may overwrite its own or a peer's source.
        // 2. FLAT:
        //    (a) must not overwrite a group source;
        //    (b) a single instruction FLAT group may overwrite its own source.
        //        Otherwise, XNACK replay may re-execute a successful FLAT after its
        //        source has been overwritten. The XNACKed FLAT is safe because its
        //        source is still intact.
        // 3. SMEM:
        //    (a) must not overwrite a group source;
        //    (b) a single instruction SMEM clause may overwrite its own source
        //        only if it returns one DWORD or less. A multi-DWORD load can
        //        return some DWORDs and XNACK on others, so it cannot be
        //        replayed once its source is gone.
        // 4. Atomic / RMW:
        //    (a) drain before the first atomic after non-atomic memory;
        //    (b) consecutive atomics need no intervening drain due to the
        //        XNACK scoreboard.

        const bool atomic = isAtomic(*inst);

        // Rule 4(a): the first atomic after non-atomic memory must start
        // with XCNT == 0. This drain clears state before Rule 2 runs below.
        // Active under both `RequiresXCntForVolatileVMEM` and `EnableXnackReplay`
        // (the pass only runs when at least one is set).
        const bool needsRule4aDrain = atomic && state.hasNonAtomic;
        if (needsRule4aDrain) {
            insertXcntDrain(builder, archId, inst, state, profile, XcntDrainReason::AtomicRule4a);
        }

        // Rules 2 and 3 (source-clobber protection) require full replay support.
        // `RequiresXCntForVolatileVMEM` alone only protects atomics (Rule 4).
        if (enableXnackReplay) {
            if (kind == MemoryGroupKind::SMEM && returnsMultipleDwords(*inst) &&
                hasSelfDestSourceOverlap(*inst)) {
                reportUnrepairableSmemSelfOverlap(bb, *inst);
            }

            if (kind == MemoryGroupKind::SMEM && violatesSmemSourceRule(*inst, state)) {
                warnGroupBreak(bb, *inst, "SMEM");
                insertXcntDrain(builder, archId, inst, state, profile, XcntDrainReason::SmemRule3);
            }

            // Rule 2 applies only to non-atomic FLAT. Rule 4 already handled the
            // first atomic and permits a consecutive atomic run.
            const bool violatesRule2 = kind == MemoryGroupKind::VMEM && isFlat(*inst) && !atomic &&
                                       violatesFlatSourceRule(*inst, state);
            if (violatesRule2) {
                warnGroupBreak(bb, *inst, "FLAT");
                insertXcntDrain(builder, archId, inst, state, profile, XcntDrainReason::FlatRule2);
            }
        }

        // Only VMEM and SMEM retain replay state.
        if (kind == MemoryGroupKind::Other) {
            state.clear();
            return;
        }

        state.kind = kind;
        state.hasMemory = true;
        state.hasNonAtomic |= !atomic;
        addSources(state.sources, *inst);
    }

   public:
    static void runOnFunction(Function& func, GfxArchID archId, XcntDrainProfileBase& profile,
                              bool enableXnackReplay) {
        profile.beginFunction(func);

        GroupState state;
        BasicBlock* previous = nullptr;

        // CFG blocks can be split solely by a fall-through label. Labels do not
        // emit instructions or break a single-group replay group, so preserve
        // state while walking the physical block layout.
        for (BasicBlock& bb : func) {
            if (state.hasMemory && previous) assertFallthrough(*previous, bb);

            AsmIRBuilder builder(bb, archId);
            for (auto it = bb.begin(); it != bb.end(); ++it) {
                applySingleGroupXnackReplayFix(bb, it, builder, archId, state, profile,
                                               enableXnackReplay);
            }
            previous = &bb;
        }
    }

   private:
    bool enableXcntDrainProfile = false;
};

char Gfx1250HazardPass::ID = 0;

// Runs the per-function hazard fix under one shared profile so an enabled profile
// aggregates all functions. Only needed for profiling.
class Gfx1250HazardModulePass : public ModulePass {
   public:
    explicit Gfx1250HazardModulePass(bool enableXcntDrainProfile)
        : enableXcntDrainProfile(enableXcntDrainProfile) {}

    const char* getName() const override {
        return "Gfx1250HazardModulePass";
    }

    PreservedAnalyses run(StinkyAsmModule& M, PassContext& passCtx,
                          ModuleAnalysisManager& /*MAM*/) override {
        const auto& caps = passCtx.getAsmCapsConfig();
        if (!caps.requiresXCntForVolatileVMEM && !caps.enableXnackReplay)
            return PreservedAnalyses::all();

        const auto arch = passCtx.getGemmTileConfig().arch;
        const GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);
        auto profile = makeXcntDrainProfile(enableXcntDrainProfile);
        for (Function* f : M.getFunctions())
            if (f && !f->empty())
                Gfx1250HazardPass::runOnFunction(*f, archId, *profile, caps.enableXnackReplay);
        profile->print();
        return PreservedAnalyses::all();
    }

   private:
    bool enableXcntDrainProfile;
};
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createGfx1250HazardPass(bool enableXcntDrainProfile) {
    return std::make_unique<Gfx1250HazardPass>(enableXcntDrainProfile);
}

std::unique_ptr<ModulePass> createGfx1250HazardModulePass(bool enableXcntDrainProfile) {
    return std::make_unique<Gfx1250HazardModulePass>(enableXcntDrainProfile);
}
}  // namespace stinkytofu
