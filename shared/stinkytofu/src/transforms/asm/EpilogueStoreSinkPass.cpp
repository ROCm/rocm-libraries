// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/EpilogueStoreSinkPass.hpp"

#include <iostream>
#include <set>
#include <utility>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

#define DEBUG_TYPE "EpilogueStoreSinkPass"

namespace {
using namespace stinkytofu;

// A single physical register unit: (type, index). b128 store data spans 4 units.
using RegUnit = std::pair<RegType, unsigned>;

// Expand every (isRegister) operand in \p regs into individual (type, idx) units.
static void collectUnits(const std::vector<StinkyRegister>& regs, std::set<RegUnit>& out) {
    for (const StinkyRegister& r : regs) {
        if (!r.isRegister()) continue;
        for (unsigned off = 0; off < r.reg.num; ++off) {
            out.insert({r.reg.type, r.reg.idx + off});
        }
    }
}

// Does \p inst write any unit in \p units? (WAW/WAR/RAW hazard against a set.)
static bool writesAny(const StinkyInstruction& inst, const std::set<RegUnit>& units) {
    for (const StinkyRegister& d : inst.getDestRegs()) {
        if (!d.isRegister()) continue;
        for (unsigned off = 0; off < d.reg.num; ++off) {
            if (units.count({d.reg.type, d.reg.idx + off})) return true;
        }
    }
    return false;
}

// Does this instruction bump the VA_VDST counter? Must match InsertWaitAluPass's
// producer classification (isVectorALU || isTranscendental || isMatrixInstruction)
// so our sink distance equals the va_vdst(N) the wait pass will emit.
static bool bumpsVaVdst(const StinkyInstruction& inst) {
    return isVectorALU(inst) || isTranscendental(inst) || isMatrixInstruction(inst);
}

// Sink one buffer_store within its block.
// storeIt points at the store; the [msb?/wait] preceding it are left in place
// (regenerated later by InsertVgprMsb / InsertWaitAlu).
//
// Returns the number of VALU ops the store was sunk past (0 = not moved).
static unsigned sinkOneStore(BasicBlock& bb, BasicBlock::iterator storeIt, unsigned targetValu) {
    StinkyInstruction& store = getStinkyInst(storeIt);

    // The store's dependency footprint:
    //  - data + address regs it READS: nothing that writes them may be crossed
    //    (RAW producer is behind us; a later writer would be WAR — reg reuse,
    //     e.g. a next-batch buffer_load into the store's data regs).
    //  - it must not cross a writer of any reg it reads (covers SGPR SRD advance
    //    s_add_u32 sgprSrd*, which the store reads as an address base).
    std::set<RegUnit> readUnits;
    collectUnits(store.getSrcRegs(), readUnits);

    BasicBlock::iterator it = std::next(storeIt);
    BasicBlock::iterator dest = storeIt;  // last legal insertion point (before `it`)
    unsigned valuPassed = 0;

    while (it != bb.end() && valuPassed < targetValu) {
        IRBase* node = it.getNodePtr();
        auto* instPtr = dyn_cast<StinkyInstruction>(node);
        if (!instPtr) break;  // label / directive — hard boundary
        StinkyInstruction& cand = *instPtr;

        // Hard boundaries: other side-effecting insts (branch, another store,
        // waitcnt, barrier, the dwordx4 s_nop wait-state). Stop before them.
        if (hasSideEffect(cand)) break;

        // WAR/WAW: candidate writes a reg the store reads → cannot sink past it.
        if (writesAny(cand, readUnits)) break;

        // Legal to cross this instruction.
        if (bumpsVaVdst(cand)) ++valuPassed;
        ++it;
        dest = std::prev(it);
    }

    if (valuPassed == 0) return 0;  // no room / nothing to gain

    // Move the store to just AFTER `dest` (i.e. before std::next(dest)).
    BasicBlock::iterator insertPos = std::next(dest);
    bb.removeIR(&store);
    bb.insertIR(insertPos, &store);
    return valuPassed;
}

size_t sinkStoresInBlock(BasicBlock& bb, unsigned targetValu) {
    size_t moved = 0;
    // Snapshot store iterators first: moving one store must not disturb the walk.
    std::vector<BasicBlock::iterator> stores;
    for (auto it = bb.begin(); it != bb.end(); ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst && isGlobalMemStore(*inst)) stores.push_back(it);
    }
    for (auto storeIt : stores) {
        if (sinkOneStore(bb, storeIt, targetValu) > 0) ++moved;
    }
    return moved;
}

class EpilogueStoreSinkPass : public StinkyInstPass {
   public:
    static char ID;
    explicit EpilogueStoreSinkPass(unsigned targetValu) : targetValu_(targetValu) {}

    const char* getName() const override {
        return "EpilogueStoreSinkPass";
    }

    PassID getPassID() const override {
        return &EpilogueStoreSinkPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        for (BasicBlock& bb : func) {
            if (!passCtx.shouldProcessBasicBlock(bb)) continue;
            const size_t moved = sinkStoresInBlock(bb, targetValu_);
            PASS_DEBUG(std::cerr << "[EpilogueStoreSinkPass] bb=\"" << bb.getLabel()
                                 << "\" sunk_stores=" << moved << " target=" << targetValu_
                                 << "\n");
        }
        return preserveCFGAnalyses();
    }

   private:
    unsigned targetValu_;
};

char EpilogueStoreSinkPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createEpilogueStoreSinkPass(unsigned targetValu) {
    return std::make_unique<EpilogueStoreSinkPass>(targetValu);
}
}  // namespace stinkytofu
