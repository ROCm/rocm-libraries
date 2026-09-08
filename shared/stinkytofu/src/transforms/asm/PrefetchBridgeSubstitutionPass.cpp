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

// Ops taking an LDS FIFO ticket. A flat_* takes one in both FIFOs, so it counts here too.
bool isLdsFifoOp(const StinkyInstruction& inst) {
    return isDSRead(inst) || isDSWrite(inst) || isDSAtomic(inst) || isFLATLoad(inst) ||
           isFLATStore(inst) || isFLATAtomic(inst) || isFLATPrefetch(inst);
}

void collectVgprs(const std::vector<StinkyRegister>& regs, std::vector<unsigned>& out) {
    for (const StinkyRegister& r : regs) {
        if (!r.isRegister() || isPseudoReg(r) || r.reg.type != RegType::V) continue;
        for (unsigned off = 0; off < r.reg.num; ++off) out.push_back(r.reg.idx + off);
    }
}

// A prefetch, or a write that overwrites a prefetch address. Only these matter, so the
// CFG search walks a handful of entries per block instead of every instruction.
struct Ev {
    StinkyInstruction* inst;
    std::vector<unsigned> regs;
    unsigned lds;
    bool isPrefetch;
};

// The prefetch groups of one function, in the form the anchor decision needs them.
struct Groups {
    std::unordered_map<BasicBlock*, std::vector<Ev>> evs;
    std::unordered_map<StinkyInstruction*, StinkyInstruction*> anchorOf;
    std::unordered_map<StinkyInstruction*, unsigned> ldsOf;
};

// Prefetches that can still be the last reader of reg on some path into (bb, idx).
// A path is cut by an earlier write to the same register.
void reaching(BasicBlock* bb, int idx, unsigned reg,
              const std::unordered_map<BasicBlock*, std::vector<Ev>>& evs,
              std::unordered_set<BasicBlock*>& seen, std::vector<const Ev*>& out) {
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
        reaching(pred, pit == evs.end() ? -1 : static_cast<int>(pit->second.size()) - 1, reg, evs,
                 seen, out);
    }
}

// Address registers of every global prefetch in the function.
std::unordered_set<unsigned> collectPrefetchAddrRegs(Function& func) {
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
    return addrRegs;
}

// LDS ops are counted in layout order, which is what the gap is stated in. A group is a
// run of prefetches with no intervening overwrite; only its LAST member needs to become
// the anchor, since the order FIFO puts it behind the whole group.
Groups buildGroups(Function& func, const std::unordered_set<unsigned>& addrRegs) {
    Groups g;
    std::vector<StinkyInstruction*> group;
    unsigned ldsCount = 0;

    auto closeGroup = [&]() {
        for (StinkyInstruction* p : group) g.anchorOf[p] = group.back();
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
                g.evs[&bb].push_back({inst, regs, ldsCount, true});
                g.ldsOf[inst] = ldsCount;
                group.push_back(inst);
                continue;
            }
            collectVgprs(inst->getDestRegs(), regs);
            std::vector<unsigned> hit;
            for (unsigned r : regs)
                if (addrRegs.count(r) != 0) hit.push_back(r);
            if (hit.empty()) continue;
            g.evs[&bb].push_back({inst, hit, ldsCount, false});
            closeGroup();
        }
    }
    closeGroup();
    return g;
}

// Anchors elected by one consumer register, or empty when the consumer does not qualify.
// Every group reaching it must be far enough away, since a join cannot rely on an anchor
// that only some paths carry.
std::vector<StinkyInstruction*> electAnchorsForReg(BasicBlock* bb, size_t idx, unsigned reg,
                                                   unsigned consumerLds, const Groups& g,
                                                   int required) {
    std::unordered_set<BasicBlock*> seen;
    std::vector<const Ev*> pfs;
    reaching(bb, static_cast<int>(idx) - 1, reg, g.evs, seen, pfs);
    if (pfs.empty()) return {};

    bool ok = true;
    unsigned minGap = ~0u;
    std::vector<StinkyInstruction*> anchors;
    for (const Ev* pf : pfs) {
        auto a = g.anchorOf.find(pf->inst);
        StinkyInstruction* anchor = a != g.anchorOf.end() ? a->second : pf->inst;
        anchors.push_back(anchor);
        const unsigned anchorLds = g.ldsOf.at(anchor);
        // An anchor laid out after its consumer only reaches it around a back edge,
        // where a layout gap says nothing. Decline.
        if (anchorLds > consumerLds)
            ok = false;
        else
            minGap = std::min(minGap, consumerLds - anchorLds);
    }
    if (!ok || minGap < static_cast<unsigned>(required)) {
        PASS_DEBUG(std::cerr << "[PrefetchBridge] declined v" << reg << " (" << anchors.size()
                             << " groups reach it, gap="
                             << (ok ? std::to_string(minGap) : "backedge") << " < " << required
                             << ")\n");
        return {};
    }
    PASS_DEBUG(std::cerr << "[PrefetchBridge] anchored v" << reg << " (" << anchors.size()
                         << " groups reach it, gap=" << minGap << " >= " << required << ")\n");
    return anchors;
}

// The last prefetch of every group that some consumer elected.
std::unordered_set<StinkyInstruction*> chooseAnchors(const Groups& g, int required) {
    std::unordered_set<StinkyInstruction*> chosen;
    for (const auto& [bb, list] : g.evs) {
        for (size_t i = 0; i < list.size(); ++i) {
            if (list[i].isPrefetch) continue;
            for (unsigned reg : list[i].regs) {
                std::vector<StinkyInstruction*> anchors =
                    electAnchorsForReg(bb, i, reg, list[i].lds, g, required);
                chosen.insert(anchors.begin(), anchors.end());
            }
        }
    }
    return chosen;
}

int rewriteToFlat(const std::unordered_set<StinkyInstruction*>& chosen,
                  const HwInstDesc* flatDesc) {
    int substituted = 0;
    for (StinkyInstruction* pf : chosen) {
        pf->updateHwInstDesc(flatDesc);
        // A null saddr is spelled "off" in the GLOBAL syntax and omitted in the FLAT
        // one, so the operand has to go with the opcode. The encoding is the same;
        // only the spelling differs.
        std::vector<StinkyRegister> srcs;
        for (const StinkyRegister& src : pf->getSrcRegs()) {
            if (src.dataType == StinkyRegister::Type::LiteralString && src.literalValue == "off")
                continue;
            srcs.push_back(src);
        }
        pf->setSrcRegs(srcs);
        ++substituted;
    }
    return substituted;
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

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        const int required = passCtx.getHWModel().waitHide.vmVsrcBridge;
        if (required <= 0) return preserveCFGAnalyses();

        const auto archTriple = passCtx.getGemmTileConfig().arch;
        const GfxArchID arch = getGfxArchID(archTriple[0], archTriple[1], archTriple[2]);
        const HwInstDesc* flatDesc = getMCIDByUOp(GFX::flat_prefetch_b8, arch);
        if (flatDesc == nullptr) return preserveCFGAnalyses();

        const std::unordered_set<unsigned> addrRegs = collectPrefetchAddrRegs(func);
        if (addrRegs.empty()) return preserveCFGAnalyses();

        const Groups groups = buildGroups(func, addrRegs);
        const int substituted = rewriteToFlat(chooseAnchors(groups, required), flatDesc);

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
