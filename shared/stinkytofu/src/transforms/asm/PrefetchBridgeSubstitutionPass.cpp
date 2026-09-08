// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/PrefetchBridgeSubstitutionPass.hpp"

#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"

#define DEBUG_TYPE "PrefetchBridgeSubstitutionPass"

namespace {
using namespace stinkytofu;

bool isLdsFifoOp(const StinkyInstruction& inst) {
    return isDSRead(inst) || isDSWrite(inst) || isDSAtomic(inst);
}

void collectVgprs(const std::vector<StinkyRegister>& regs, std::vector<unsigned>& out) {
    for (const StinkyRegister& r : regs) {
        if (!r.isRegister() || isPseudoReg(r) || r.reg.type != RegType::V) continue;
        for (unsigned off = 0; off < r.reg.num; ++off) out.push_back(r.reg.idx + off);
    }
}

class PrefetchBridgeSubstitutionPass : public Pass {
   public:
    static char ID;

    const char* getName() const override {
        return "Prefetch Bridge Substitution";
    }

    Pass::ID getPassID() const override {
        return &PrefetchBridgeSubstitutionPass::ID;
    }

    // A prefetch, or a write that overwrites a prefetch address. Only these matter, so the
    // CFG search walks a handful of entries per block instead of every instruction.
    struct Ev {
        StinkyInstruction* inst;
        std::vector<unsigned> regs;
        unsigned lds;
        bool isPrefetch;
    };

    // Prefetches that can still be the last reader of \p reg on some path into (\p bb,
    // \p idx). A path is cut by an earlier write to the same register.
    void reaching(BasicBlock* bb, int idx, unsigned reg,
                  const std::unordered_map<BasicBlock*, std::vector<Ev>>& evs,
                  std::unordered_set<BasicBlock*>& seen, std::vector<const Ev*>& out) const {
        auto it = evs.find(bb);
        if (it != evs.end()) {
            for (int i = idx; i >= 0; --i) {
                const Ev& e = it->second[i];
                bool hit = false;
                for (unsigned r : e.regs) hit = hit || r == reg;
                if (!hit) continue;
                if (e.isPrefetch) out.push_back(&e);
                return;
            }
        }
        if (!seen.insert(bb).second) return;
        for (BasicBlock* pred : bb->getPredecessors()) {
            auto pit = evs.find(pred);
            reaching(pred, pit == evs.end() ? -1 : static_cast<int>(pit->second.size()) - 1, reg,
                     evs, seen, out);
        }
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        const int required = passCtx.getHWModel().waitHide.vmVsrcBridge;
        if (required <= 0) return preserveCFGAnalyses();

        const auto archTriple = passCtx.getGemmTileConfig().arch;
        const GfxArchID arch = getGfxArchID(archTriple[0], archTriple[1], archTriple[2]);
        const HwInstDesc* flatDesc = getMCIDByUOp(GFX::flat_prefetch_b8, arch);
        if (flatDesc == nullptr) return preserveCFGAnalyses();

        std::unordered_set<unsigned> addrRegs;
        for (BasicBlock& bb : func) {
            for (IRBase& node : bb) {
                auto* inst = dyn_cast<StinkyInstruction>(&node);
                if (inst == nullptr || !isGlobalPrefetch(*inst)) continue;
                std::vector<unsigned> regs;
                collectVgprs(inst->getSrcRegs(), regs);
                addrRegs.insert(regs.begin(), regs.end());
            }
        }
        if (addrRegs.empty()) return preserveCFGAnalyses();

        // LDS ops are counted in layout order, which is what the gap is stated in. A group
        // is a run of prefetches with no intervening overwrite; only its LAST member needs
        // to become the anchor, since the order FIFO puts it behind the whole group.
        std::unordered_map<BasicBlock*, std::vector<Ev>> evs;
        std::unordered_map<StinkyInstruction*, StinkyInstruction*> anchorOf;
        std::unordered_map<StinkyInstruction*, unsigned> ldsOf;
        std::vector<StinkyInstruction*> group;
        unsigned ldsCount = 0;

        auto closeGroup = [&]() {
            for (StinkyInstruction* p : group) anchorOf[p] = group.back();
            group.clear();
        };

        for (BasicBlock& bb : func) {
            for (IRBase& node : bb) {
                auto* inst = dyn_cast<StinkyInstruction>(&node);
                if (inst == nullptr) continue;
                if (isLdsFifoOp(*inst)) ++ldsCount;
                std::vector<unsigned> regs;
                if (isGlobalPrefetch(*inst)) {
                    collectVgprs(inst->getSrcRegs(), regs);
                    if (regs.empty()) continue;
                    evs[&bb].push_back({inst, regs, ldsCount, true});
                    ldsOf[inst] = ldsCount;
                    group.push_back(inst);
                    continue;
                }
                collectVgprs(inst->getDestRegs(), regs);
                std::vector<unsigned> hit;
                for (unsigned r : regs)
                    if (addrRegs.count(r) != 0) hit.push_back(r);
                if (hit.empty()) continue;
                evs[&bb].push_back({inst, hit, ldsCount, false});
                closeGroup();
            }
        }
        closeGroup();

        // Every path into a consumer must carry an anchor or the join cannot rely on one,
        // so a consumer qualifies only when EVERY group reaching it is far enough away.
        // Each of those groups then contributes exactly one flat: its last prefetch.
        std::unordered_set<StinkyInstruction*> chosen;
        for (auto& [bb, list] : evs) {
            for (size_t i = 0; i < list.size(); ++i) {
                if (list[i].isPrefetch) continue;
                for (unsigned reg : list[i].regs) {
                    std::unordered_set<BasicBlock*> seen;
                    std::vector<const Ev*> pfs;
                    reaching(bb, static_cast<int>(i) - 1, reg, evs, seen, pfs);
                    if (pfs.empty()) continue;
                    bool ok = true;
                    unsigned minGap = ~0u;
                    std::vector<StinkyInstruction*> anchors;
                    for (const Ev* pf : pfs) {
                        auto a = anchorOf.find(pf->inst);
                        StinkyInstruction* anchor = a != anchorOf.end() ? a->second : pf->inst;
                        anchors.push_back(anchor);
                        // An anchor laid out after its consumer only reaches it around a
                        // back edge, where a layout gap says nothing. Decline.
                        if (ldsOf[anchor] > list[i].lds)
                            ok = false;
                        else
                            minGap = std::min(minGap, list[i].lds - ldsOf[anchor]);
                    }
                    if (!ok || minGap < static_cast<unsigned>(required)) {
                        PASS_DEBUG(std::cerr << "[PrefetchBridge] declined v" << reg << " ("
                                             << anchors.size() << " groups reach it, gap="
                                             << (ok ? std::to_string(minGap) : "backedge") << " < "
                                             << required << ")\n");
                        continue;
                    }
                    chosen.insert(anchors.begin(), anchors.end());
                    PASS_DEBUG(std::cerr << "[PrefetchBridge] anchored v" << reg << " ("
                                         << anchors.size() << " groups reach it, gap=" << minGap
                                         << " >= " << required << ")\n");
                }
            }
        }

        int substituted = 0;
        for (StinkyInstruction* pf : chosen) {
            pf->updateHwInstDesc(flatDesc);
            // A null saddr is spelled "off" in the GLOBAL syntax and omitted in the FLAT
            // one, so the operand has to go with the opcode. The encoding is the same;
            // only the spelling differs.
            std::vector<StinkyRegister> srcs;
            for (const StinkyRegister& src : pf->getSrcRegs()) {
                if (src.dataType == StinkyRegister::Type::LiteralString &&
                    src.literalValue == "off")
                    continue;
                srcs.push_back(src);
            }
            pf->setSrcRegs(srcs);
            ++substituted;
        }

        PASS_DEBUG(std::cerr << "[PrefetchBridge] substituted " << substituted
                             << " prefetch(es)\n");
        return preserveCFGAnalyses();
    }
};

char PrefetchBridgeSubstitutionPass::ID = 0;

}  // namespace

namespace stinkytofu {

std::unique_ptr<Pass> createPrefetchBridgeSubstitutionPass() {
    return std::make_unique<PrefetchBridgeSubstitutionPass>();
}

}  // namespace stinkytofu
