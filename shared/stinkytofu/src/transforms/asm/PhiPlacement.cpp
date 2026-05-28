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
#include "stinkytofu/transforms/asm/PhiPlacement.hpp"

#include <cassert>
#include <unordered_set>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/controlflow/Dominance.hpp"
#include "stinkytofu/analysis/controlflow/DominanceAnalysis.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/RegisterKey.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace {
using namespace stinkytofu;

//----------------------------------------------------------------------
// Per-block register definitions
//----------------------------------------------------------------------

struct BlockDefs {
    RegKeySet keys;
    RegKeyMap<StinkyInstruction*> lastDef;
};

std::vector<BlockDefs> gatherDefs(const std::vector<BasicBlock*>& rpo) {
    const unsigned N = rpo.size();
    std::vector<BlockDefs> info(N);

    for (unsigned i = 0; i < N; ++i) {
        for (IRBase& ir : *rpo[i]) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            auto* inst = cast<StinkyInstruction>(&ir);
            if (isPseudoInst(inst)) continue;

            for (const auto& dest : inst->getDestRegs()) {
                if (!dest.isRegister() || isPseudoReg(dest)) continue;
                for (unsigned d = 0; d < dest.reg.num; ++d) {
                    RegKey key = toRegKey(dest, d);
                    info[i].keys.insert(key);
                    info[i].lastDef[key] = inst;
                }
            }
        }
    }

    return info;
}

//----------------------------------------------------------------------
// Globally-used registers  (semi-pruned SSA)
//
// Collect every register key that appears as a source operand in any
// real instruction.  Registers that are defined but never read need
// no PHIs — they would be dead on arrival.
//----------------------------------------------------------------------

RegKeySet gatherUsedRegs(const std::vector<BasicBlock*>& rpo) {
    RegKeySet used;
    for (BasicBlock* bb : rpo) {
        for (IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            auto* inst = cast<StinkyInstruction>(&ir);
            if (isPseudoInst(inst)) continue;
            for (const auto& src : inst->getSrcRegs()) {
                if (!src.isRegister() || isPseudoReg(src)) continue;
                for (unsigned s = 0; s < src.reg.num; ++s) used.insert(toRegKey(src, s));
            }
        }
    }
    return used;
}

//----------------------------------------------------------------------
// PHI-site computation  (iterated dominance frontier)
//
// For each register key K that is globally used, place a PHI at every
// block in the iterated dominance frontier of K's definition sites.
//
// Time: O(R * (N + F)), R = globally-used register keys, F = Sigma|DF[i]|.
//----------------------------------------------------------------------

std::vector<RegKeySet> computePhiSites(const std::vector<BlockDefs>& blockDefs,
                                       const std::vector<std::vector<unsigned>>& df,
                                       const RegKeySet& usedRegs, unsigned N) {
    RegKeyMap<std::vector<unsigned>> defSites;
    defSites.reserve(usedRegs.size());
    for (unsigned i = 0; i < N; ++i)
        for (const auto& key : blockDefs[i].keys)
            if (usedRegs.contains(key)) defSites[key].push_back(i);

    std::vector<RegKeySet> sites(N);

    std::vector<unsigned> wl;
    std::unordered_set<unsigned> inWL;
    std::unordered_set<unsigned> placed;
    wl.reserve(N);
    inWL.reserve(N);
    placed.reserve(N);

    for (const auto& [key, blocks] : defSites) {
        wl.assign(blocks.begin(), blocks.end());
        inWL.clear();
        inWL.insert(blocks.begin(), blocks.end());
        placed.clear();

        while (!wl.empty()) {
            unsigned b = wl.back();
            wl.pop_back();

            for (unsigned d : df[b]) {
                if (placed.insert(d).second) {
                    sites[d].insert(key);
                    if (inWL.insert(d).second) wl.push_back(d);
                }
            }
        }
    }

    return sites;
}

//----------------------------------------------------------------------
// Reaching definitions at block exits
//
// Computed in a single RPO pass by inheriting from the immediate
// dominator.  PHIs (just created) act as definitions at block entry;
// real instructions override them within the block.
//
// Time: O(N * R), N = blocks, R = unique register keys.
//----------------------------------------------------------------------

using ReachMap = RegKeyMap<StinkyInstruction*>;

std::vector<ReachMap> computeReachOut(const std::vector<BasicBlock*>& rpo,
                                      const std::vector<unsigned>& idom,
                                      const std::vector<BlockDefs>& blockDefs,
                                      const std::vector<RegKeyMap<StinkyInstruction*>>& phiInsts) {
    const unsigned N = rpo.size();
    std::vector<ReachMap> out(N);

    for (unsigned i = 0; i < N; ++i) {
        if (i > 0) out[i] = out[idom[i]];

        for (const auto& [key, phi] : phiInsts[i]) out[i][key] = phi;

        for (const auto& [key, inst] : blockDefs[i].lastDef) out[i][key] = inst;
    }

    return out;
}

//----------------------------------------------------------------------
// Remove existing PHI instructions (from a previous run)
//----------------------------------------------------------------------

void removeExistingPhis(Function& func) {
    for (BasicBlock& bb : func) {
        while (!bb.empty()) {
            IRBase& ir = *bb.begin();
            if (ir.getType() != IRBase::IRType::StinkyTofu) break;
            auto* inst = cast<StinkyInstruction>(&ir);
            if (inst->getUnifiedOpcode() != GFX::PHI) break;
            ir.erase();
        }
    }
}

//----------------------------------------------------------------------
// Pass wrapper
//----------------------------------------------------------------------

class InsertPhiPass : public Pass {
   public:
    static char ID;

    const char* getName() const override {
        return "InsertPhiPass";
    }

    Pass::ID getPassID() const override {
        return &InsertPhiPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext&, AnalysisManager& AM) override {
        const auto& domInfo = AM.getResult<DominanceAnalysis>(func);
        insertPhiInstructions(func, domInfo, true);
        return preserveCFGAnalyses();
    }
};

char InsertPhiPass::ID = 0;

}  // anonymous namespace

namespace stinkytofu {
// Time: O(N*E + R*(N + F) + I), N = blocks, E = CFG edges,
//       R = register keys, F = Sigma|DF[i]|, I = instructions.
void insertPhiInstructions(Function& func, const DominanceInfo& domInfo, bool clearExisting) {
    if (func.empty()) return;

    if (clearExisting) removeExistingPhis(func);

    const auto& rpo = domInfo.rpo;
    const unsigned N = rpo.size();
    if (N == 0) return;

    // --- 1. Per-block register definitions ---

    auto blockDefs = gatherDefs(rpo);

    // --- 2. Globally-used registers (semi-pruned SSA) ---

    auto usedRegs = gatherUsedRegs(rpo);

    // --- 3. PHI-placement sites (iterated DF, only for used registers) ---

    auto phiSites = computePhiSites(blockDefs, domInfo.df, usedRegs, N);

    // --- 4. Create PHI instructions (operands initially nullptr) ---

    std::vector<RegKeyMap<StinkyInstruction*>> phiInsts(N);

    const auto& archArr = func.getGemmTileConfig().arch;
    GfxArchID arch = getGfxArchID(archArr[0], archArr[1], archArr[2]);

    for (unsigned i = 0; i < N; ++i) {
        if (phiSites[i].empty()) continue;

        AsmIRBuilder builder(*rpo[i], arch);
        IRBase* anchor = rpo[i]->empty() ? nullptr : &*rpo[i]->begin();
        for (const auto& key : phiSites[i]) {
            StinkyInstruction* phi = builder.createPhi(key.type, key.idx, anchor);
            phiInsts[i][key] = phi;
        }
    }

    // --- 5. Reaching definitions at block exits ---

    auto reachOut = computeReachOut(rpo, domInfo.idom, blockDefs, phiInsts);

    // --- 6. Resolve PHI operands from predecessor reaching defs ---

    for (unsigned i = 0; i < N; ++i) {
        if (phiInsts[i].empty()) continue;

        const auto& preds = rpo[i]->getPredecessors();

        for (auto& [key, phi] : phiInsts[i]) {
            for (size_t j = 0; j < preds.size(); ++j) {
                auto predIt = domInfo.rpoIndex.find(preds[j]);
                if (predIt == domInfo.rpoIndex.end()) continue;

                auto defIt = reachOut[predIt->second].find(key);
                StinkyInstruction* def =
                    (defIt != reachOut[predIt->second].end()) ? defIt->second : nullptr;

                if (def != nullptr) phi->setSrcReg(j, def->getDestReg(0));
            }
        }
    }
}

void insertPhiInstructions(Function& func, bool clearExisting) {
    if (func.empty()) return;
    DominanceInfo domInfo = computeDominanceInfo(func);
    insertPhiInstructions(func, domInfo, clearExisting);
}

std::unique_ptr<Pass> createInsertPhiPass() {
    return std::make_unique<InsertPhiPass>();
}

}  // namespace stinkytofu
