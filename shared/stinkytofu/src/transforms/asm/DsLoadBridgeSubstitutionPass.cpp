// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/DsLoadBridgeSubstitutionPass.hpp"

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

#define DEBUG_TYPE "DsLoadBridgeSubstitutionPass"

namespace {
using namespace stinkytofu;

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

// The FLAT counterpart of a ds_load width, or nullopt when the width has none.
// Only the plain widths convert; tr/2addr/sub-dword forms have no 1:1 FLAT form.
GFX dsLoadToFlatLoad(uint16_t uop, bool& ok) {
    ok = true;
    switch (static_cast<GFX>(uop)) {
        case GFX::ds_load_b32:
            return GFX::flat_load_b32;
        case GFX::ds_load_b64:
            return GFX::flat_load_b64;
        case GFX::ds_load_b96:
            return GFX::flat_load_b96;
        case GFX::ds_load_b128:
            return GFX::flat_load_b128;
        default:
            ok = false;
            return GFX::ds_load_b32;
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
// anchorOf maps a prefetch to the ds_load elected to carry its bridge, so unlike the
// prefetch-side pass the anchor is never the prefetch itself.
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

// LDS ops are counted in layout order, which is what the gap is stated in. Prefetches
// accumulate until a convertible ds_load follows them; that ds_load becomes their shared
// anchor, since TEX order puts it behind the whole run. A run cut by an overwrite before
// any ds_load arrives gets no anchor and is therefore never substituted for.
Groups buildGroups(Function& func, const std::unordered_set<unsigned>& addrRegs) {
    Groups g;
    std::vector<StinkyInstruction*> pending;
    unsigned ldsCount = 0;

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
                pending.push_back(inst);
                continue;
            }

            // First convertible ds_load after a run of prefetches: elect it for all of them.
            if (!pending.empty() && isDSRead(*inst)) {
                bool convertible = false;
                dsLoadToFlatLoad(inst->getUnifiedOpcode(), convertible);
                if (convertible) {
                    for (StinkyInstruction* p : pending) g.anchorOf[p] = inst;
                    g.ldsOf[inst] = ldsCount;
                    pending.clear();
                    continue;
                }
            }

            collectVgprs(inst->getDestRegs(), regs);
            std::vector<unsigned> hit;
            for (unsigned r : regs)
                if (addrRegs.count(r) != 0) hit.push_back(r);
            if (hit.empty()) continue;
            g.evs[&bb].push_back({inst, hit, ldsCount, false});
            // The run ended without a ds_load to anchor it; those prefetches stay unanchored.
            pending.clear();
        }
    }
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
        // No ds_load followed this prefetch before its consumer: nothing to anchor with.
        if (a == g.anchorOf.end()) return {};
        StinkyInstruction* anchor = a->second;
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
        PASS_DEBUG(std::cerr << "[DsLoadBridge] declined v" << reg << " (" << anchors.size()
                             << " groups reach it, gap="
                             << (ok ? std::to_string(minGap) : "backedge") << " < " << required
                             << ")\n");
        return {};
    }
    PASS_DEBUG(std::cerr << "[DsLoadBridge] anchored v" << reg << " (" << anchors.size()
                         << " groups reach it, gap=" << minGap << " >= " << required << ")\n");
    return anchors;
}

// The ds_load elected by some consumer to carry a prefetch group's bridge.
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

// ds_load_bN -> flat_load_bN, keeping the 32-bit vaddr and appending the aperture saddr.
// Widening the address to a 64-bit vaddr pair instead would cost an extra address
// register per lane, so the saddr form is the only one worth emitting.
int rewriteToFlat(const std::unordered_set<StinkyInstruction*>& chosen, GfxArchID arch,
                  int apertureSgpr) {
    int substituted = 0;
    for (StinkyInstruction* ds : chosen) {
        bool convertible = false;
        const GFX flatUop = dsLoadToFlatLoad(ds->getUnifiedOpcode(), convertible);
        if (!convertible) continue;
        const HwInstDesc* flatDesc = getMCIDByUOp(flatUop, arch);
        if (flatDesc == nullptr) continue;

        std::vector<StinkyRegister> srcs = ds->getSrcRegs();
        srcs.push_back(StinkyRegister("s", static_cast<uint32_t>(apertureSgpr), 2));
        ds->updateHwInstDesc(flatDesc);
        ds->setSrcRegs(srcs);
        ++substituted;
    }
    return substituted;
}

// A FLAT op only reaches LDS through the aperture, so the pair must hold src_shared_base.
// Written once at kernel entry, which dominates every substituted load.
bool insertApertureInit(Function& func, GfxArchID arch, int apertureSgpr) {
    const HwInstDesc* movDesc = getMCIDByUOp(GFX::s_mov_b64, arch);
    if (movDesc == nullptr) return false;
    for (BasicBlock& bb : func) {
        for (auto it = bb.begin(); it != bb.end(); ++it) {
            auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if (inst == nullptr || isPseudoInst(inst)) continue;
            AsmIRBuilder irBuilder(bb, arch);
            StinkyInstruction* mov = irBuilder.create(movDesc, it.getNodePtr());
            mov->addDestReg(StinkyRegister(RegType::S, static_cast<uint32_t>(apertureSgpr), 2));
            mov->addSrcReg(StinkyRegister(RegType::SRC_SHARED_BASE, 0, 2));
            return true;
        }
    }
    return false;
}

class DsLoadBridgeSubstitutionPass : public Pass {
   public:
    static char ID;

    explicit DsLoadBridgeSubstitutionPass(int apertureSgpr) : apertureSgpr_(apertureSgpr) {}

    const char* getName() const override {
        return "Ds Load Bridge Substitution";
    }

    Pass::ID getPassID() const override {
        return &DsLoadBridgeSubstitutionPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        const int required = passCtx.getHWModel().waitHide.vmVsrcBridge;
        if (required <= 0) return preserveCFGAnalyses();
        // No SGPR pair holds the LDS aperture base, so no legal FLAT address can be built.
        if (apertureSgpr_ < 0) return preserveCFGAnalyses();

        const auto archTriple = passCtx.getGemmTileConfig().arch;
        const GfxArchID arch = getGfxArchID(archTriple[0], archTriple[1], archTriple[2]);

        const std::unordered_set<unsigned> addrRegs = collectPrefetchAddrRegs(func);
        if (addrRegs.empty()) return preserveCFGAnalyses();

        const Groups groups = buildGroups(func, addrRegs);
        const std::unordered_set<StinkyInstruction*> chosen = chooseAnchors(groups, required);
        if (chosen.empty()) return preserveCFGAnalyses();

        // Without the aperture write there is no legal FLAT address, so substitute nothing.
        if (!insertApertureInit(func, arch, apertureSgpr_)) return preserveCFGAnalyses();

        const int substituted = rewriteToFlat(chosen, arch, apertureSgpr_);

        PASS_DEBUG(std::cerr << "[DsLoadBridge] substituted " << substituted << " ds_load(s)\n");
        return preserveCFGAnalyses();
    }

   private:
    int apertureSgpr_;
};

char DsLoadBridgeSubstitutionPass::ID = 0;

}  // namespace

namespace stinkytofu {

std::unique_ptr<Pass> createDsLoadBridgeSubstitutionPass(int apertureSgpr) {
    return std::make_unique<DsLoadBridgeSubstitutionPass>(apertureSgpr);
}

}  // namespace stinkytofu
