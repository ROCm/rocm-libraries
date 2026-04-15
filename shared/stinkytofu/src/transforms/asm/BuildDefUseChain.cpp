/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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
#include "stinkytofu/transforms/asm/BuildDefUseChain.hpp"

#include <cassert>
#include <vector>

#include "stinkytofu/analysis/controlflow/Dominance.hpp"
#include "stinkytofu/analysis/controlflow/DominanceAnalysis.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/DefUseChainUpdater.hpp"
#include "stinkytofu/ir/asm/RegisterKey.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/PhiPlacement.hpp"

namespace stinkytofu {
/// Chain construction helpers is private to BuildDefUseChain.cpp because
/// it directly modify sources and users.
class DefUseChainBuilder {
   public:
    static void addLink(StinkyInstruction* user, StinkyInstruction* def) {
        if (!def) return;
        user->sources.push_back(def);
        def->users.push_back(user);
    }

    static void addPhiOperand(StinkyInstruction* phi, StinkyInstruction* def) {
        phi->sources.push_back(def);
        if (def) def->users.push_back(phi);
    }

    static void clearChains(StinkyInstruction* inst) {
        inst->sources.clear();
        inst->users.clear();
    }
};

}  // namespace stinkytofu

namespace {
using namespace stinkytofu;

//----------------------------------------------------------------------
// Chain management helpers
//----------------------------------------------------------------------

void clearAllChains(Function& func) {
    for (BasicBlock& bb : func) {
        for (IRBase& ir : bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            DefUseChainBuilder::clearChains(cast<StinkyInstruction>(&ir));
        }
    }
}

void removeUnusedPhis(BasicBlock& bb) {
    for (auto it = bb.begin(); it != bb.end();) {
        if (it->getType() != IRBase::IRType::StinkyTofu) break;
        auto* inst = cast<StinkyInstruction>(&*it);
        if (inst->getUnifiedOpcode() != GFX::PHI) break;
        if (inst->getUsers().empty()) {
            it = DefUseChainUpdater::eraseAndUnlink(bb, it);
            continue;
        }
        ++it;
    }
}

//----------------------------------------------------------------------
// Build all def-use chains (PHI and non-PHI)
//
// Two passes over the RPO:
//   Pass 1 — compute reachOut[block] and link non-PHI sources.
//   Pass 2 — link PHI sources from predecessors' reachOut.
//
// The two-pass split is necessary because back-edge predecessors
// are not yet processed when the loop header is visited in pass 1.
//
// Time: O(N * R + I), N = blocks, R = register keys, I = instructions.
//----------------------------------------------------------------------

using ReachMap = RegKeyMap<StinkyInstruction*>;

void buildChains(Function& func, const DominanceInfo& domInfo) {
    const auto& rpo = domInfo.rpo;
    const unsigned N = rpo.size();
    if (N == 0) return;

    std::vector<ReachMap> reachOut(N);

    // --- Pass 1: reachOut + non-PHI chains ---

    for (unsigned i = 0; i < N; ++i) {
        ReachMap currentDef;
        if (i > 0) currentDef = reachOut[domInfo.idom[i]];

        for (IRBase& ir : *rpo[i]) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            auto* inst = cast<StinkyInstruction>(&ir);

            if (inst->getUnifiedOpcode() == GFX::PHI) {
                currentDef[toRegKey(inst->getDestReg(0))] = inst;
                continue;
            }

            if (inst->getUnifiedOpcode() == GFX::LABEL) continue;

            std::unordered_set<StinkyInstruction*> addedDefs;
            for (const auto& src : inst->getSrcRegs()) {
                if (!src.isRegister() || isPseudoReg(src)) continue;

                for (unsigned s = 0; s < src.reg.num; ++s) {
                    auto it = currentDef.find(toRegKey(src, s));
                    if (it != currentDef.end() && it->second != nullptr &&
                        addedDefs.insert(it->second).second)
                        DefUseChainBuilder::addLink(inst, it->second);
                }
            }

            for (const auto& dest : inst->getDestRegs()) {
                if (!dest.isRegister() || isPseudoReg(dest)) continue;
                for (unsigned d = 0; d < dest.reg.num; ++d) currentDef[toRegKey(dest, d)] = inst;
            }
        }

        reachOut[i] = std::move(currentDef);
    }

    // --- Pass 2: link PHI sources ---

    for (unsigned i = 0; i < N; ++i) {
        BasicBlock* bb = rpo[i];
        const std::vector<BasicBlock*>& preds = bb->getPredecessors();
        if (preds.empty()) continue;

        for (IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) break;
            auto* inst = cast<StinkyInstruction>(&ir);
            if (inst->getUnifiedOpcode() != GFX::PHI) break;

            RegKey key = toRegKey(inst->getDestReg(0));

            for (size_t j = 0; j < preds.size(); ++j) {
                StinkyInstruction* def = nullptr;

                auto predIt = domInfo.rpoIndex.find(preds[j]);
                if (predIt != domInfo.rpoIndex.end()) {
                    auto defIt = reachOut[predIt->second].find(key);
                    if (defIt != reachOut[predIt->second].end()) def = defIt->second;
                }

                DefUseChainBuilder::addPhiOperand(inst, def);
            }
        }
    }
}

//----------------------------------------------------------------------
// Pass registration
//----------------------------------------------------------------------

class BuildUseDefChainPass : public Pass {
    bool clearExisting_;

   public:
    static char ID;

    explicit BuildUseDefChainPass(bool clearExisting) : clearExisting_(clearExisting) {}

    const char* getName() const override {
        return "BuildUseDefChainPass";
    }

    Pass::ID getPassID() const override {
        return &BuildUseDefChainPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext&, AnalysisManager& AM) override {
        const auto& domInfo = AM.getResult<DominanceAnalysis>(func);
        buildUseDefChain(func, domInfo, clearExisting_);
        return PreservedAnalyses::none();
    }
};

char BuildUseDefChainPass::ID = 0;

}  // anonymous namespace

namespace stinkytofu {
// Time: O(N*E + R*(N + F) + I), dominated by PHI insertion.
//       N = blocks, E = edges, R = register keys, F = Sigma|DF[i]|, I = instructions.
void buildUseDefChain(Function& func, const DominanceInfo& domInfo, bool clearExisting) {
    if (func.empty()) return;

    // Phase 1: Insert PHIs at correct CFG join points (dominance-frontier based).
    insertPhiInstructions(func, domInfo, clearExisting);

    // Phase 2: Clear all existing chains when rebuilding, so buildChains
    // starts from a clean slate via DefUseChainBuilder.
    if (clearExisting) clearAllChains(func);

    // Phase 3: Build all def-use chains (PHI and non-PHI) in a single
    //          RPO traversal with dominator-inherited reaching definitions.
    buildChains(func, domInfo);

    // Phase 4: Remove PHIs that ended up with no users.
    for (BasicBlock& bb : func) removeUnusedPhis(bb);
}

void buildUseDefChain(Function& func, bool clearExisting) {
    if (func.empty()) return;
    DominanceInfo domInfo = computeDominanceInfo(func);
    buildUseDefChain(func, domInfo, clearExisting);
}

std::unique_ptr<Pass> createBuildUseDefChainPass(bool clearExisting) {
    return std::make_unique<BuildUseDefChainPass>(clearExisting);
}

}  // namespace stinkytofu
