// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/EpilogueStoreSinkPass.hpp"

#include <iostream>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/ir/asm/RegisterKey.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"

#define DEBUG_TYPE "EpilogueStoreSinkPass"

namespace {
using namespace stinkytofu;

// VA_VDST producer — must match InsertWaitAluPass's classification.
static bool bumpsVaVdst(const StinkyInstruction& inst) {
    return isVectorALU(inst) || isTranscendental(inst) || isMatrixInstruction(inst);
}

// Returns true if `store` may legally cross the wait `cand` without perturbing it.
// A store only bumps its own counter; waits on other counters are safe to cross.
static bool isCrossableWait(const StinkyInstruction& store, const StinkyInstruction& cand,
                            bool crossLoadcnt, bool crossAsyncCnt) {
    const auto op = cand.getUnifiedOpcode();
    if (op == GFX::s_wait_loadcnt) return crossLoadcnt;
    if (isGlobalStoreAsyncFromLds(store)) {
        // asynccnt store: may cross storecnt wait when counters are independent.
        if (op == GFX::s_wait_storecnt) return crossAsyncCnt;
    } else {
        // storecnt store: may cross asynccnt wait when counters are independent.
        if (op == GFX::s_wait_asynccnt) return crossAsyncCnt;
    }
    return false;
}

static int requiredMsbOf(const StinkyInstruction& inst) {
    auto [setVal, hasVgpr] = computeRequiredMsb(&inst);
    return hasVgpr ? setVal : -1;
}

// Sink one store past following VALU ops; returns how many were crossed.
// msbGuard: stop before straddling an s_set_vgpr_msb/frames flip.
static unsigned sinkOneStore(BasicBlock& bb, BasicBlock::iterator storeIt, unsigned targetValu,
                             bool crossLoadcnt, bool crossAsyncCnt, bool msbGuard) {
    StinkyInstruction& store = getStinkyInst(storeIt);

    // Regs the store reads: it may not cross any later writer of them (WAR), which
    // also covers the SGPR SRD advance s_add_u32 sgprSrd*.
    RegKeySet readUnits;
    addSources(readUnits, store);

    // MSB at the store's current position; stop before a flip to a different value.
    int curMsb = -1;
    if (msbGuard) {
        for (BasicBlock::iterator b = storeIt; b != bb.begin();) {
            --b;
            auto* p = dyn_cast<StinkyInstruction>(b.getNodePtr());
            if (!p) break;
            int m = requiredMsbOf(*p);
            if (m >= 0) {
                curMsb = m;
                break;
            }
        }
    }

    BasicBlock::iterator it = std::next(storeIt);
    BasicBlock::iterator dest = storeIt;       // last legal insertion point (before `it`)
    BasicBlock::iterator cleanDest = storeIt;  // last msb-clean landing (guard on)
    unsigned valuPassed = 0;
    unsigned valuAtCleanDest = 0;  // VALU passed as of the last msb-clean landing

    while (it != bb.end() && valuPassed < targetValu) {
        IRBase* node = it.getNodePtr();
        auto* instPtr = dyn_cast<StinkyInstruction>(node);
        if (!instPtr) break;
        StinkyInstruction& cand = *instPtr;

        if (hasSideEffect(cand) && !isCrossableWait(store, cand, crossLoadcnt, crossAsyncCnt))
            break;

        // WAR/WAW: candidate writes a reg the store reads → cannot sink past it.
        if (hasDestSourceOverlap(cand, readUnits)) break;

        // Legal to cross this instruction.
        if (bumpsVaVdst(cand)) ++valuPassed;
        ++it;
        dest = std::prev(it);

        // Track the furthest MSB-clean landing; stop at the first flip.
        if (msbGuard) {
            int m = requiredMsbOf(cand);
            if (m >= 0 && m != curMsb) break;
            cleanDest = dest;
            valuAtCleanDest = valuPassed;
        }
    }

    if (msbGuard) {
        dest = cleanDest;
        valuPassed = valuAtCleanDest;
    }

    if (valuPassed == 0) return 0;  // no room / nothing to gain

    // Move the store to just after `dest`.
    BasicBlock::iterator insertPos = std::next(dest);
    bb.removeIR(&store);
    bb.insertIR(insertPos, &store);
    return valuPassed;
}

size_t sinkStoresInBlock(BasicBlock& bb, unsigned targetValu, bool crossLoadcnt, bool crossAsyncCnt,
                         bool msbGuard) {
    size_t moved = 0;
    // Snapshot store iterators first: moving one store must not disturb the walk.
    std::vector<BasicBlock::iterator> stores;
    for (auto it = bb.begin(); it != bb.end(); ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst && isVmemTexStore(*inst)) stores.push_back(it);
    }
    // Bottom-up so each store sinks into the slot the one below vacated.
    for (size_t k = 0; k < stores.size(); ++k) {
        const size_t i = stores.size() - 1 - k;
        if (sinkOneStore(bb, stores[i], targetValu, crossLoadcnt, crossAsyncCnt, msbGuard) > 0)
            ++moved;
    }
    return moved;
}

// Pull `store` up to sit right after `anchor` if every inst between is hoppable.
static bool groupOneStore(BasicBlock& bb, BasicBlock::iterator anchorIt,
                          BasicBlock::iterator storeIt, bool crossLoadcnt, bool crossAsyncCnt) {
    StinkyInstruction& store = getStinkyInst(storeIt);
    RegKeySet readUnits;
    addSources(readUnits, store);

    // Verify the whole span (anchor, store) is hoppable before moving anything.
    for (BasicBlock::iterator it = std::next(anchorIt); it != storeIt; ++it) {
        auto* p = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (!p) return false;
        StinkyInstruction& cand = *p;
        if (hasSideEffect(cand) && !isCrossableWait(store, cand, crossLoadcnt, crossAsyncCnt))
            return false;
        if (hasDestSourceOverlap(cand, readUnits))
            return false;  // WAR: store reads a reg cand writes
    }

    // Move the store to just after the anchor (before anchor's current next).
    BasicBlock::iterator insertPos = std::next(anchorIt);
    if (insertPos == storeIt) return false;  // already adjacent
    bb.removeIR(&store);
    bb.insertIR(insertPos, &store);
    return true;
}

// Pack stores into groups of up to `maxStoreGroupSize` so one xcnt drain covers each group.
static size_t groupStoresInBlock(BasicBlock& bb, unsigned maxStoreGroupSize, bool crossLoadcnt,
                                 bool crossAsyncCnt) {
    if (maxStoreGroupSize < 2) return 0;
    std::vector<BasicBlock::iterator> stores;
    for (auto it = bb.begin(); it != bb.end(); ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst && isVmemTexStore(*inst)) stores.push_back(it);
    }
    const size_t storeCount = stores.size();
    size_t moved = 0;
    size_t i = 0;
    while (i < storeCount) {
        // stores[i] anchors a group; pull the next (maxStoreGroupSize-1) up behind it.
        BasicBlock::iterator anchor = stores[i];
        size_t g = 1;
        for (; g < maxStoreGroupSize && (i + g) < storeCount; ++g) {
            if (!groupOneStore(bb, anchor, stores[i + g], crossLoadcnt, crossAsyncCnt)) break;
            anchor = stores[i + g];  // chain: next store packs behind the one just moved
            ++moved;
        }
        i += g;  // start a fresh group (fresh xcnt) after the packed run
    }
    return moved;
}

class EpilogueStoreSinkPass : public StinkyInstPass {
   public:
    static char ID;
    explicit EpilogueStoreSinkPass(EpilogueStoreSinkOptions options) : options_(options) {}

    const char* getName() const override {
        return "EpilogueStoreSinkPass";
    }

    PassID getPassID() const override {
        return &EpilogueStoreSinkPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        const HWModel& hw = passCtx.getHWModel();
        const bool crossLoadcnt = hw.counters.hasSplitLoadStoreCnt;
        const bool crossAsyncCnt = hw.counters.hasSplitStoreCntAsyncCnt;
        // Without replay protection no drain is emitted, so the guard buys nothing.
        const bool msbGuard =
            options_.avoidMsbXcntDrain && passCtx.getAsmCapsConfig().enableXnackReplay;
        for (BasicBlock& bb : func) {
            if (!passCtx.shouldProcessBasicBlock(bb)) continue;
            const size_t moved =
                sinkStoresInBlock(bb, options_.targetValu, crossLoadcnt, crossAsyncCnt, msbGuard);
            const size_t grouped =
                groupStoresInBlock(bb, options_.maxStoreGroupSize, crossLoadcnt, crossAsyncCnt);
            PASS_DEBUG(std::cerr << "[EpilogueStoreSinkPass] bb=\"" << bb.getLabel()
                                 << "\" sunk_stores=" << moved << " grouped=" << grouped
                                 << " target=" << options_.targetValu
                                 << " crossLoadcnt=" << crossLoadcnt
                                 << " crossAsyncCnt=" << crossAsyncCnt << " msbGuard=" << msbGuard
                                 << " maxStoreGroupSize=" << options_.maxStoreGroupSize << "\n");
        }
        return preserveCFGAnalyses();
    }

   private:
    EpilogueStoreSinkOptions options_;
};

char EpilogueStoreSinkPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createEpilogueStoreSinkPass(EpilogueStoreSinkOptions options) {
    return std::make_unique<EpilogueStoreSinkPass>(options);
}
}  // namespace stinkytofu
