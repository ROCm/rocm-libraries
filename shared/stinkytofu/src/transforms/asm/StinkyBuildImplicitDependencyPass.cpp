/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the Software), to deal
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
#include "stinkytofu/transforms/asm/StinkyBuildImplicitDependencyPass.hpp"

#include <cassert>
#include <iostream>
#include <unordered_set>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/LoopAnalysis.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

#define DEBUG_TYPE "StinkyBuildImplicitDependencyPass"

// Implicit LDS dependency pass
// ============================
// Attaches RegType::LDS pseudo-registers to instructions based on their MemTokenData
// token IDs.  The instruction type determines src vs dest placement:
//
//   tensor_load / ds_write  →  LDS token to dest  (LDS producer)
//   ds_read                 →  LDS token to src   (LDS consumer)
//   barrier / signal / wait →  LDS token to both  (synchronization point)
//
// The def-use chain builder then sees:
//   producer(def LDS[t]) → barrier(use+def LDS[t]) → consumer(use LDS[t])
// which forces the scheduler to respect: producers → barrier → consumers.

namespace {
using namespace stinkytofu;

static void addUniqueLdsDest(StinkyInstruction& inst, int tokenId) {
    for (const StinkyRegister& d : inst.getDestRegs())
        if (d.reg.type == RegType::LDS && d.reg.idx == static_cast<uint32_t>(tokenId)) return;
    inst.addDestReg(StinkyRegister(RegType::LDS, tokenId, 1));
}

static void addUniqueLdsSrc(StinkyInstruction& inst, int tokenId) {
    for (const StinkyRegister& s : inst.getSrcRegs())
        if (s.reg.type == RegType::LDS && s.reg.idx == static_cast<uint32_t>(tokenId)) return;
    inst.addSrcReg(StinkyRegister(RegType::LDS, tokenId, 1));
}

// Barrier: LDS tokens to both src and dest.
// Returns false if this is a non-all-wave split barrier that should be skipped.
static bool processBarrier(StinkyInstruction& inst, const MemTokenData& mt,
                           [[maybe_unused]] const std::string& bbLabel) {
    if ((isBarrierSignal(inst) || isBarrierWait(inst)) && !isSplitBarrierAllWave(inst)) {
        PASS_DEBUG(std::cerr << "[BuildImplicitDep] non-all-wave split barrier BB label=\""
                             << bbLabel << "\" — skipping (workgroup-sync mode)\n");
        return false;
    }

    PASS_DEBUG(std::cerr << "[BuildImplicitDep] barrier BB label=\"" << bbLabel << "\" tokens=[";
               for (int t
                    : mt.tokens) std::cerr
               << t << " ";
               std::cerr << "]\n");

    for (int tokenId : mt.tokens) {
        addUniqueLdsDest(inst, tokenId);
        addUniqueLdsSrc(inst, tokenId);
    }
    return true;
}

// LDS writers (tensor_load, ds_write): LDS tokens to dest.
static void processLdsWriter(StinkyInstruction& inst, const MemTokenData& mt,
                             [[maybe_unused]] const std::string& bbLabel) {
    PASS_DEBUG(std::cerr << "[BuildImplicitDep] LDS writer ("
                         << (isTensorLoad(inst) ? "tensor_load" : "ds_write") << ") tokens=[";
               for (int t
                    : mt.tokens) std::cerr
               << t << " ";
               std::cerr << "] -> dest\n");

    for (int tokenId : mt.tokens) addUniqueLdsDest(inst, tokenId);
}

// LDS readers (ds_read): LDS tokens to src.
static void processLdsReader(StinkyInstruction& inst, const MemTokenData& mt,
                             [[maybe_unused]] const std::string& bbLabel) {
    PASS_DEBUG(std::cerr << "[BuildImplicitDep] LDS reader (ds_read) tokens=[";
               for (int t
                    : mt.tokens) std::cerr
               << t << " ";
               std::cerr << "] -> src\n");

    for (int tokenId : mt.tokens) addUniqueLdsSrc(inst, tokenId);
}

static bool isMemTokenCandidate(const StinkyInstruction& inst) {
    return isTensorLoad(inst) || isDSWrite(inst) || isDSRead(inst);
}

static const char* memTokenCandidateKind(const StinkyInstruction& inst) {
    if (isTensorLoad(inst)) return "tensor_load";
    if (isDSWrite(inst)) return "ds_store";
    if (isDSRead(inst)) return "ds_load";
    return "unknown";
}

static std::unordered_set<const BasicBlock*> collectOptLevel3MemTokenCheckBlocks(
    const std::vector<Loop>& loops) {
    std::unordered_set<const BasicBlock*> checkBlocks;

    for (const Loop& loop : loops) {
        // loop BBs
        for (const BasicBlock* bodyBB : loop.bodyBBs) {
            checkBlocks.insert(bodyBB);
        }

        // preloop BBs: predecessors of loop header that are outside loop body.
        if (loop.headerBB) {
            for (const BasicBlock* pred : loop.headerBB->getPredecessors()) {
                if (!loop.contains(pred)) checkBlocks.insert(pred);
            }
        }

        // postloop BBs: successors of loop body that are outside loop body.
        for (const BasicBlock* bodyBB : loop.bodyBBs) {
            for (const BasicBlock* succ : bodyBB->getSuccessors()) {
                if (succ && !loop.contains(succ)) checkBlocks.insert(succ);
            }
        }
    }

    return checkBlocks;
}

static void checkConsistentMemTokens(const BasicBlock& bb,
                                     const std::unordered_set<const BasicBlock*>& checkBlocks) {
    if (!checkBlocks.count(&bb)) return;

    bool hasWithToken = false;
    bool hasWithoutToken = false;

    for (auto it = bb.begin(); it != bb.end(); ++it) {
        const auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (!inst || !isMemTokenCandidate(*inst)) continue;

        if (inst->getModifier<MemTokenData>())
            hasWithToken = true;
        else
            hasWithoutToken = true;
    }

    if (!hasWithToken || !hasWithoutToken) return;

    std::cerr << "[BuildImplicitDep] ERROR: BB \"" << bb.getLabel()
              << "\" has inconsistent memory tokens — some ds_load/ds_store/tensor_load"
                 " instructions have MemTokenData while others do not:\n";

    for (auto it = bb.begin(); it != bb.end(); ++it) {
        const auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (!inst || !isMemTokenCandidate(*inst)) continue;

        const bool hasToken = inst->getModifier<MemTokenData>() != nullptr;
        std::cerr << "  " << (hasToken ? "[has token]   " : "[NO TOKEN]    ")
                  << memTokenCandidateKind(*inst) << " (" << inst->getHwInstDesc()->mnemonic
                  << ")\n";
    }

    assert(false && "inconsistent MemTokenData across ds_load/ds_store/tensor_load in basic block");
}

void setPseudoRegistersInBlock(BasicBlock& bb, PassContext& passCtx,
                               const std::unordered_set<const BasicBlock*>& checkBlocks) {
    if (!passCtx.getPassFeatureConfig().barrierConfig.unrollMovableBarrier) {
        PASS_DEBUG(std::cerr << "[BuildImplicitDep] skip BB label=\"" << bb.getLabel()
                             << "\" (unrollMovableBarrier=false)\n");
        return;
    }

    checkConsistentMemTokens(bb, checkBlocks);

    for (auto it = bb.begin(); it != bb.end(); ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (!inst) continue;

        const MemTokenData* mt = inst->getModifier<MemTokenData>();
        if (!mt) continue;
        assert(!mt->tokens.empty() && "MemTokenData with empty tokens");

        if (isBarrier(*inst))
            processBarrier(*inst, *mt, bb.getLabel());
        else if (isTensorLoad(*inst) || isDSWrite(*inst))
            processLdsWriter(*inst, *mt, bb.getLabel());
        else if (isDSRead(*inst))
            processLdsReader(*inst, *mt, bb.getLabel());
        else
            assert(false &&
                   "instruction has MemTokenData but is not a barrier, fence, "
                   "tensor_load, ds_write, or ds_read");
    }
}

class StinkyBuildImplicitDependencyPass : public StinkyInstPass {
   public:
    static char ID;

    const char* getName() const override {
        return "StinkyBuildImplicitDependencyPass";
    }

    PassID getPassID() const override {
        return &StinkyBuildImplicitDependencyPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& AM) override {
        const auto& loops = AM.getResult<LoopAnalysis>(func);
        const auto checkBlocks = collectOptLevel3MemTokenCheckBlocks(loops);

        PASS_DEBUG(std::cerr << "[BuildImplicitDep] mem-token consistency checks on "
                             << checkBlocks.size() << " BBs (preloop/loop/postloop derived)\n");
        PASS_DEBUG(for (const BasicBlock* bb
                        : checkBlocks) {
            if (bb) std::cerr << "  [BuildImplicitDep] check BB: " << bb->getLabel() << "\n";
        });

        for (BasicBlock& bb : func) {
            if (passCtx.shouldProcessBasicBlock(bb))
                setPseudoRegistersInBlock(bb, passCtx, checkBlocks);
        }
        return preserveCFGAnalyses();
    }
};

char StinkyBuildImplicitDependencyPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createStinkyBuildImplicitDependencyPass() {
    return std::make_unique<StinkyBuildImplicitDependencyPass>();
}
}  // namespace stinkytofu
