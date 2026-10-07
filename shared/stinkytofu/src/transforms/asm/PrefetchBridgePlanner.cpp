// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/PrefetchBridgePlanner.hpp"

#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

#define DEBUG_TYPE "PrefetchBridgeSubstitutionPass"

namespace stinkytofu {
namespace {

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
    const StinkyInstruction* inst;
    std::vector<unsigned> regs;
    unsigned lds;
    bool isPrefetch;
};

// The prefetch groups of one function, in the form the anchor decision needs them.
struct Groups {
    std::unordered_map<const BasicBlock*, std::vector<Ev>> evs;
    std::unordered_map<const StinkyInstruction*, const StinkyInstruction*> anchorOf;
    std::unordered_map<const StinkyInstruction*, unsigned> ldsOf;
};

// Prefetches that can still be the last reader of reg on some path into (bb, idx).
// A path is cut by an earlier write to the same register.
void reaching(const BasicBlock* bb, int idx, unsigned reg,
              const std::unordered_map<const BasicBlock*, std::vector<Ev>>& evs,
              std::unordered_set<const BasicBlock*>& seen, std::vector<const Ev*>& out) {
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
std::unordered_set<unsigned> collectPrefetchAddrRegs(const std::vector<BlockOrder>& layout) {
    std::unordered_set<unsigned> addrRegs;
    for (const BlockOrder& block : layout) {
        for (const StinkyInstruction* inst : *block.insts) {
            if (!isGlobalPrefetch(*inst)) continue;
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
Groups buildGroups(const std::vector<BlockOrder>& layout,
                   const std::unordered_set<unsigned>& addrRegs) {
    Groups g;
    std::vector<const StinkyInstruction*> group;
    unsigned ldsCount = 0;

    auto closeGroup = [&]() {
        for (const StinkyInstruction* p : group) g.anchorOf[p] = group.back();
        group.clear();
    };

    for (const BlockOrder& block : layout) {
        for (const StinkyInstruction* inst : *block.insts) {
            if (isLdsFifoOp(*inst)) ++ldsCount;
            std::vector<unsigned> regs;
            if (isGlobalPrefetch(*inst)) {
                collectVgprs(inst->getSrcRegs(), regs);
                if (regs.empty()) continue;
                g.evs[block.bb].push_back({inst, regs, ldsCount, true});
                g.ldsOf[inst] = ldsCount;
                group.push_back(inst);
                continue;
            }
            collectVgprs(inst->getDestRegs(), regs);
            std::vector<unsigned> hit;
            for (unsigned r : regs)
                if (addrRegs.count(r) != 0) hit.push_back(r);
            if (hit.empty()) continue;
            g.evs[block.bb].push_back({inst, hit, ldsCount, false});
            closeGroup();
        }
    }
    closeGroup();
    return g;
}

// Anchors elected by one consumer register, or empty when the consumer does not qualify.
// Every group reaching it must be far enough away, since a join cannot rely on an anchor
// that only some paths carry.
std::vector<const StinkyInstruction*> electAnchorsForReg(const BasicBlock* bb, size_t idx,
                                                         unsigned reg, unsigned consumerLds,
                                                         const Groups& g, int required) {
    std::unordered_set<const BasicBlock*> seen;
    std::vector<const Ev*> pfs;
    reaching(bb, static_cast<int>(idx) - 1, reg, g.evs, seen, pfs);
    if (pfs.empty()) return {};

    bool ok = true;
    unsigned minGap = ~0u;
    std::vector<const StinkyInstruction*> anchors;
    for (const Ev* pf : pfs) {
        auto a = g.anchorOf.find(pf->inst);
        const StinkyInstruction* anchor = a != g.anchorOf.end() ? a->second : pf->inst;
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

}  // namespace

std::unordered_set<const StinkyInstruction*> planPrefetchBridge(
    const std::vector<BlockOrder>& layout, int required) {
    std::unordered_set<const StinkyInstruction*> chosen;
    if (required <= 0) return chosen;
    const std::unordered_set<unsigned> addrRegs = collectPrefetchAddrRegs(layout);
    if (addrRegs.empty()) return chosen;

    // The last prefetch of every group that some consumer elected.
    const Groups g = buildGroups(layout, addrRegs);
    for (const auto& [bb, list] : g.evs) {
        for (size_t i = 0; i < list.size(); ++i) {
            if (list[i].isPrefetch) continue;
            for (unsigned reg : list[i].regs) {
                std::vector<const StinkyInstruction*> anchors =
                    electAnchorsForReg(bb, i, reg, list[i].lds, g, required);
                chosen.insert(anchors.begin(), anchors.end());
            }
        }
    }
    return chosen;
}

}  // namespace stinkytofu
