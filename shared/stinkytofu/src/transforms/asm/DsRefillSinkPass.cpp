// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/DsRefillSinkPass.hpp"

#include <algorithm>
#include <climits>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "stinkytofu/analysis/asm/DependencyExport.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"
#include "stinkytofu/pipeline/IRCheckpoint.hpp"
#include "stinkytofu/transforms/asm/InsertWaitAluPass.hpp"
#include "stinkytofu/transforms/asm/dag/WmmaBudget.hpp"

#define DEBUG_TYPE "DsRefillSinkPass"

namespace {
using namespace stinkytofu;

constexpr size_t kNoPos = static_cast<size_t>(-1);

std::string regText(const StinkyRegister& reg) {
    if (!reg.isRegister()) return "?";
    const std::string prefix = reg.reg.type == RegType::V ? "v" : "r";
    if (reg.reg.num <= 1) return prefix + std::to_string(reg.reg.idx);
    return prefix + "[" + std::to_string(reg.reg.idx) + ":" +
           std::to_string(reg.reg.idx + reg.reg.num - 1) + "]";
}

std::string slotText(int slot) {
    return slot == INT_MAX ? "-" : std::to_string(slot);
}

// True when `reader` reads a VGPR `load` writes.
bool readsWhatItLoads(const StinkyInstruction& reader, const StinkyInstruction& load) {
    for (const StinkyRegister& dst : load.getDestRegs()) {
        if (!dst.isRegister() || dst.reg.type != RegType::V) continue;
        for (const StinkyRegister& src : reader.getSrcRegs())
            if (src.isRegister() && src.reg.type == RegType::V && dst.isOverlap(src)) return true;
    }
    return false;
}

int requiredMsbOf(const StinkyInstruction& inst) {
    const auto [setVal, hasVgpr] = computeRequiredMsb(&inst);
    return hasVgpr ? setVal : -1;
}

// What a move may not break: the dependency edges and pinned nodes of the IR checkpoint editor.
// A node the export does not know is pinned.
struct Constraints {
    std::unordered_map<const IRBase*, size_t> index;
    std::vector<bool> pinned;
    std::vector<std::vector<size_t>> succs;

    bool isPinned(const IRBase* node) const {
        auto it = index.find(node);
        return it == index.end() || pinned[it->second];
    }
    // A refill stays inside its stage: it crosses neither a pinned node nor a barrier, even one
    // the export does not pin.
    bool endsStage(const IRBase* node) const {
        const auto* inst = dyn_cast<StinkyInstruction>(node);
        return isPinned(node) || (inst && isBarrier(*inst));
    }
    // True when `node` must stay before `later`.
    bool mustPrecede(const IRBase* node, const IRBase* later) const {
        auto from = index.find(node);
        auto to = index.find(later);
        if (from == index.end() || to == index.end()) return true;
        const std::vector<size_t>& out = succs[from->second];
        return std::find(out.begin(), out.end(), to->second) != out.end();
    }
};

Constraints buildConstraints(Function& func) {
    Constraints c;
    const IRCheckpointRendering rendering = renderCheckpointRegion(func, /*withIds=*/false);
    const RegionDependencies deps = exportRegionDependencies(func);
    for (size_t i = 0; i < rendering.nodes.size(); ++i) c.index[rendering.nodes[i]] = i;
    c.pinned.assign(rendering.nodes.size(), true);
    c.succs.assign(rendering.nodes.size(), {});
    if (!deps.error.empty() || deps.insts.size() != rendering.nodes.size()) return c;
    for (size_t i = 0; i < deps.insts.size(); ++i) c.pinned[i] = deps.insts[i].pinned;
    for (const DependencyEdge& edge : deps.edges)
        c.succs[static_cast<size_t>(edge.src)].push_back(static_cast<size_t>(edge.dst));
    return c;
}

struct DsSite {
    StinkyInstruction* inst = nullptr;
    size_t pos = 0;
    int slot = 0;
    // The va_vdst it needs where it is; -1 for none.
    int wait = -1;
    // WMMAs still to issue before that wait is hidden; 0 for no wait, -1 when none hide it.
    int untilFree = 0;
};

// One block replayed through InsertWaitAlu's scoreboard from its entry state.
struct BlockScan {
    std::vector<IRBase*> nodes;
    // WMMAs of the block issued before each node.
    std::vector<int> slot;
    std::vector<size_t> wmmaPos;
    std::vector<DsSite> ds;
    int vaWaits = 0;
    // WMMAs still to issue, summed over the va_vdst waits WMMAs hide.
    int deficit = 0;
    int unhideable = 0;
    int wmmaLatency = 0;

    // Where gap `slot` starts: right after the WMMA before it.
    size_t gapStart(int s) const {
        return s <= 0 ? 0 : wmmaPos[static_cast<size_t>(s) - 1] + 1;
    }
    size_t gapEnd(int s) const {
        return static_cast<size_t>(s) < wmmaPos.size() ? wmmaPos[static_cast<size_t>(s)]
                                                       : nodes.size();
    }
};

BlockScan scanBlock(BasicBlock& bb, const WaitAluEntryStates& states) {
    BlockScan s;
    WaitAluTracker tracker(states, bb);
    int slot = 0;
    for (IRBase& node : bb) {
        const size_t pos = s.nodes.size();
        s.nodes.push_back(&node);
        s.slot.push_back(slot);
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (!inst) continue;
        const WaitAluNeed need = tracker.query(*inst);
        int untilFree = 0;
        if (need.vaVdst >= 0) {
            ++s.vaWaits;
            untilFree = tracker.untilFree(*inst, WaitAluHideOp::Wmma);
            if (untilFree > 0)
                s.deficit += untilFree;
            else if (untilFree < 0)
                ++s.unhideable;
        }
        if (isDSRead(*inst)) s.ds.push_back({inst, pos, slot, need.vaVdst, untilFree});
        tracker.commit(*inst);
        if (isMatrixInstruction(*inst)) {
            if (s.wmmaPos.empty()) s.wmmaLatency = static_cast<int>(inst->latencyCycles);
            s.wmmaPos.push_back(pos);
            ++slot;
        }
    }
    return s;
}

// The ds_loads in each gap between two WMMAs; a gap is never filled past the fullest one.
std::vector<int> loadsPerGap(const BlockScan& s) {
    std::vector<int> loads(s.wmmaPos.size() + 1, 0);
    for (const DsSite& site : s.ds) ++loads[static_cast<size_t>(site.slot)];
    return loads;
}

// True when the refills at `members` (positions in program order) may all move to just before
// node `insertBefore`: nothing they cross ends their stage or has to stay after them.
bool canMove(const std::vector<size_t>& members, size_t insertBefore, const BlockScan& s,
             const Constraints& c) {
    const std::unordered_set<size_t> isMember(members.begin(), members.end());
    for (size_t x = members.front() + 1; x < insertBefore; ++x) {
        if (isMember.count(x)) continue;
        if (c.endsStage(s.nodes[x])) return false;
        for (size_t m : members)
            if (m < x && c.mustPrecede(s.nodes[m], s.nodes[x])) return false;
    }
    return true;
}

struct RefillPlan {
    int free = 0;
    int due = INT_MAX;
    int stageEnd = 0;
    // Where the refill goes and its slot there; kNoPos and -1 when it stays.
    size_t insertBefore = kNoPos;
    int target = -1;
    std::string why;
};

struct PlanInputs {
    const BlockScan& scan;
    const Constraints& constraints;
    const std::vector<int>& gapLoads;
    int gapCap;
    int joinReach;
    int drainCycles;
};

// The window of the refill at `site` and the place inside it the policy picks.
RefillPlan planRefill(const DsSite& site, const PlanInputs& in) {
    const BlockScan& s = in.scan;
    RefillPlan p;
    p.free = site.slot + site.untilFree;
    const int drain =
        in.drainCycles > 0 ? in.drainCycles : static_cast<int>(site.inst->latencyCycles);
    const int lead = latencyBudgetWmmas(drain, s.wmmaLatency);
    for (size_t w : s.wmmaPos) {
        if (w > site.pos && readsWhatItLoads(*cast<StinkyInstruction>(s.nodes[w]), *site.inst)) {
            p.due = s.slot[w] - lead;
            break;
        }
    }
    size_t stageEndPos = s.nodes.size();
    for (size_t k = site.pos + 1; k < s.nodes.size(); ++k) {
        if (in.constraints.endsStage(s.nodes[k])) {
            stageEndPos = k;
            break;
        }
    }
    p.stageEnd =
        stageEndPos < s.nodes.size() ? s.slot[stageEndPos] : static_cast<int>(s.wmmaPos.size());

    if (site.untilFree < 0) {
        p.why = "no WMMA count hides it";
        return p;
    }
    if (in.constraints.isPinned(site.inst)) {
        p.why = "pinned";
        return p;
    }
    if (stageEndPos < s.nodes.size()) {
        const auto* end = dyn_cast<StinkyInstruction>(s.nodes[stageEndPos]);
        bool loadBetween = false;
        for (const DsSite& other : s.ds)
            loadBetween |= other.pos > site.pos && other.pos < stageEndPos;
        if (end && isBarrier(*end) && !loadBetween) {
            p.why = "last ds_load before a barrier";
            return p;
        }
    }
    const int limit = std::min(p.due, p.stageEnd);
    if (limit <= site.slot) {
        p.why = p.due <= site.slot ? "its data is due" : "its stage ends";
        return p;
    }

    const bool hides = p.free <= limit;
    struct Candidate {
        size_t insertBefore;
        int slot;
        bool joins;
    };
    std::vector<Candidate> candidates;
    if (hides) {
        const int msb = requiredMsbOf(*site.inst);
        for (int sl = p.free; sl <= std::min(limit, p.free + in.joinReach); ++sl) {
            for (size_t k = s.gapStart(sl); k < s.gapEnd(sl); ++k) {
                const auto* inst = dyn_cast<StinkyInstruction>(s.nodes[k]);
                if (inst && k != site.pos && isDSRead(*inst) && requiredMsbOf(*inst) == msb) {
                    candidates.push_back({k, sl, true});
                    break;
                }
            }
        }
        for (int sl = p.free; sl <= limit; ++sl) candidates.push_back({s.gapStart(sl), sl, false});
    } else {
        for (int sl = limit; sl > site.slot; --sl)
            candidates.push_back({s.gapStart(sl), sl, false});
    }
    for (const Candidate& cand : candidates) {
        if (in.gapLoads[static_cast<size_t>(cand.slot)] >= in.gapCap) continue;
        if (!canMove({site.pos}, cand.insertBefore, s, in.constraints)) continue;
        p.insertBefore = cand.insertBefore;
        p.target = cand.slot;
        p.why = !hides ? "weaker wait" : cand.joins ? "joins a ds run" : "wait hidden";
        return p;
    }
    p.why = "no legal slot";
    return p;
}

class DsRefillSinkPass : public StinkyInstPass {
   public:
    static char ID;

    const char* getName() const override {
        return "DsRefillSinkPass";
    }

    PassID getPassID() const override {
        return &DsRefillSinkPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        const auto& feats = passCtx.getPassFeatureConfig().dagFeatures;
        const WaitAluEntryStates states(func, passCtx,
                                        gfx1250InsertWaitAluOptions(feats.enableESM2TrackValuVsrc));
        const Constraints constraints = buildConstraints(func);
        for (BasicBlock& bb : func) {
            if (!passCtx.shouldProcessBasicBlock(bb) || !states.has(bb)) continue;
            const BlockScan scan = scanBlock(bb, states);
            const std::vector<int> gapLoads = loadsPerGap(scan);
            const PlanInputs inputs{scan,
                                    constraints,
                                    gapLoads,
                                    *std::max_element(gapLoads.begin(), gapLoads.end()),
                                    std::max(1, feats.dsBatchWmmas),
                                    feats.dsReadDrainLatency};
            int refills = 0;
            int movable = 0;
            for (const DsSite& site : scan.ds) {
                if (site.wait < 0) continue;
                ++refills;
                const RefillPlan plan = planRefill(site, inputs);
                if (plan.target >= 0) ++movable;
                PASS_DEBUG(std::cerr << "[DsRefillSink] bb=\"" << bb.getLabel() << "\" slot "
                                     << site.slot << " " << site.inst->getHwInstDesc()->mnemonic
                                     << " " << regText(site.inst->getDestRegs().front())
                                     << " N=" << site.wait << " D=" << site.untilFree << " window=["
                                     << plan.free << "," << slotText(plan.due)
                                     << "] stage-end=" << plan.stageEnd << " -> "
                                     << (plan.target >= 0 ? "slot " + std::to_string(plan.target)
                                                          : std::string("stays"))
                                     << " (" << plan.why << ")\n");
            }
            PASS_DEBUG(std::cerr << "[DsRefillSink] bb=\"" << bb.getLabel()
                                 << "\" va_vdst waits=" << scan.vaWaits
                                 << " deficit=" << scan.deficit << " unhideable=" << scan.unhideable
                                 << " ds_load waits=" << refills << " movable=" << movable << "\n");
        }
        return PreservedAnalyses::all();
    }
};

char DsRefillSinkPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createDsRefillSinkPass() {
    return std::make_unique<DsRefillSinkPass>();
}
}  // namespace stinkytofu
