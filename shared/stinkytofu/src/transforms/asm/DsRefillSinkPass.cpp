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

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
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
    // The vm_vsrc wait of each node that needs one.
    std::unordered_map<const IRBase*, int> vmWait;

    int dsIndexOf(const StinkyInstruction* inst) const {
        for (size_t i = 0; i < ds.size(); ++i)
            if (ds[i].inst == inst) return static_cast<int>(i);
        return -1;
    }

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
        if (need.vmVsrc >= 0) s.vmWait[inst] = need.vmVsrc;
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
    // Where the group goes and its slot there; kNoPos and -1 when it stays.
    size_t insertBefore = kNoPos;
    int target = -1;
    std::string why;
};

struct PlanInputs {
    const Constraints& constraints;
    int gapCap;
    int joinReach;
    int drainCycles;
};

// The slot by which `load`'s data must be issued: the LDS return time before the first later
// WMMA of the block that reads it, or INT_MAX when none does.
int dueSlot(const BlockScan& s, const DsSite& load, int drainCycles) {
    const int drain = drainCycles > 0 ? drainCycles : static_cast<int>(load.inst->latencyCycles);
    const int lead = latencyBudgetWmmas(drain, s.wmmaLatency);
    for (size_t w : s.wmmaPos)
        if (w > load.pos && readsWhatItLoads(*cast<StinkyInstruction>(s.nodes[w]), *load.inst))
            return s.slot[w] - lead;
    return INT_MAX;
}

int loadsInGap(const BlockScan& s, int slot, const std::unordered_set<size_t>& members) {
    int loads = 0;
    for (const DsSite& site : s.ds)
        if (site.slot == slot && !members.count(site.pos)) ++loads;
    return loads;
}

// The window of the refills in `group`, in program order, the first being the one that waits,
// and the place inside it the policy picks. The group moves together and keeps its order.
RefillPlan planGroup(const std::vector<DsSite>& group, const BlockScan& s, const PlanInputs& in) {
    const DsSite& first = group.front();
    RefillPlan p;
    std::vector<size_t> members;
    for (const DsSite& m : group) {
        members.push_back(m.pos);
        p.free = std::max(p.free, m.slot + std::max(0, m.untilFree));
        p.due = std::min(p.due, dueSlot(s, m, in.drainCycles));
    }
    const std::unordered_set<size_t> isMember(members.begin(), members.end());

    // The stage ends at the next pinned node or barrier. Before a barrier it ends at the last
    // other ds_load, since the barrier's threshold is counted from that load.
    size_t endPos = s.nodes.size();
    for (size_t k = first.pos + 1; k < s.nodes.size(); ++k) {
        if (!isMember.count(k) && in.constraints.endsStage(s.nodes[k])) {
            endPos = k;
            break;
        }
    }
    const auto* endInst =
        endPos < s.nodes.size() ? dyn_cast<StinkyInstruction>(s.nodes[endPos]) : nullptr;
    if (endInst && isBarrier(*endInst)) {
        size_t lastLoad = kNoPos;
        for (const DsSite& other : s.ds)
            if (other.pos > first.pos && other.pos < endPos && !isMember.count(other.pos))
                lastLoad = other.pos;
        if (lastLoad == kNoPos) {
            p.stageEnd = s.slot[endPos];
            p.why = "last ds_load before a barrier";
            return p;
        }
        endPos = lastLoad;
    }
    p.stageEnd = endPos < s.nodes.size() ? s.slot[endPos] : static_cast<int>(s.wmmaPos.size());

    for (const DsSite& m : group) {
        if (m.untilFree < 0) {
            p.why = "no WMMA count hides it";
            return p;
        }
        if (in.constraints.isPinned(m.inst)) {
            p.why = "pinned";
            return p;
        }
    }
    const int limit = std::min(p.due, p.stageEnd);
    if (limit <= first.slot) {
        p.why = p.due <= first.slot ? "its data is due" : "its stage ends";
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
        const int msb = requiredMsbOf(*first.inst);
        for (int sl = p.free; sl <= std::min(limit, p.free + in.joinReach); ++sl) {
            for (size_t k = s.gapStart(sl); k < s.gapEnd(sl); ++k) {
                const auto* inst = dyn_cast<StinkyInstruction>(s.nodes[k]);
                if (inst && !isMember.count(k) && isDSRead(*inst) && requiredMsbOf(*inst) == msb) {
                    candidates.push_back({k, sl, true});
                    break;
                }
            }
        }
        for (int sl = p.free; sl <= limit; ++sl) candidates.push_back({s.gapStart(sl), sl, false});
    } else {
        for (int sl = limit; sl > first.slot; --sl)
            candidates.push_back({s.gapStart(sl), sl, false});
    }
    for (const Candidate& cand : candidates) {
        if (cand.insertBefore > endPos) continue;
        if (loadsInGap(s, cand.slot, isMember) + static_cast<int>(group.size()) > in.gapCap)
            continue;
        if (!canMove(members, cand.insertBefore, s, in.constraints)) continue;
        p.insertBefore = cand.insertBefore;
        p.target = cand.slot;
        p.why = !hides ? "weaker wait" : cand.joins ? "joins a ds run" : "wait hidden";
        return p;
    }
    p.why = "no legal slot";
    return p;
}

std::vector<IRBase*> nodesOf(BasicBlock& bb) {
    std::vector<IRBase*> nodes;
    for (IRBase& node : bb) nodes.push_back(&node);
    return nodes;
}

void restoreOrder(BasicBlock& bb, const std::vector<IRBase*>& order) {
    for (IRBase* node : order) {
        bb.removeIR(node);
        bb.insertIR(bb.end(), node);
    }
}

// Move the nodes at `members` (positions in `nodes`, in order) to just before node
// `insertBefore`, keeping their order.
void moveGroup(BasicBlock& bb, const std::vector<IRBase*>& nodes,
               const std::vector<size_t>& members, size_t insertBefore) {
    IRBase* anchor = insertBefore < nodes.size() ? nodes[insertBefore] : nullptr;
    for (size_t m : members) {
        bb.removeIR(nodes[m]);
        bb.insertIR(anchor ? IRList::iterator(anchor) : bb.end(), nodes[m]);
    }
}

// Empty when `after` owes fewer WMMAs than `before`, leaves no new unhideable va_vdst wait, and
// keeps every vm_vsrc wait at least as weak; otherwise what it does worse.
std::string whyNotNetWin(const BlockScan& before, const BlockScan& after) {
    if (after.unhideable > before.unhideable) return "a va_vdst wait no WMMA count hides";
    for (const auto& [node, wait] : after.vmWait) {
        auto it = before.vmWait.find(node);
        if (it == before.vmWait.end() || wait < it->second) {
            const auto* inst = cast<StinkyInstruction>(node);
            return std::string("a stricter vm_vsrc wait before ") +
                   inst->getHwInstDesc()->mnemonic + " (" +
                   (it == before.vmWait.end() ? std::string("none") : std::to_string(it->second)) +
                   " -> " + std::to_string(wait) + ")";
        }
    }
    if (after.deficit >= before.deficit)
        return "WMMAs short " + std::to_string(before.deficit) + " -> " +
               std::to_string(after.deficit);
    return {};
}

struct Totals {
    int vaWaits = 0;
    int deficit = 0;
    int unhideable = 0;
    int vmWaits = 0;
};

Totals totalsOf(Function& func, const WaitAluEntryStates& states, PassContext& passCtx) {
    Totals t;
    for (BasicBlock& bb : func) {
        if (!passCtx.shouldProcessBasicBlock(bb) || !states.has(bb)) continue;
        const BlockScan s = scanBlock(bb, states);
        t.vaWaits += s.vaWaits;
        t.deficit += s.deficit;
        t.unhideable += s.unhideable;
        t.vmWaits += static_cast<int>(s.vmWait.size());
    }
    return t;
}

// A move that settles in this many rounds of picking up the loads it uncovers, or not at all.
constexpr int kMaxGroupRounds = 8;

class DsRefillSinkPass : public StinkyInstPass {
   public:
    static char ID;

    explicit DsRefillSinkPass(DsRefillSinkOptions options) : options_(options) {}

    const char* getName() const override {
        return "DsRefillSinkPass";
    }

    PassID getPassID() const override {
        return &DsRefillSinkPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        const auto& feats = passCtx.getPassFeatureConfig().dagFeatures;
        const InsertWaitAluOptions waitOptions =
            gfx1250InsertWaitAluOptions(feats.enableESM2TrackValuVsrc);
        auto states = std::make_unique<WaitAluEntryStates>(func, passCtx, waitOptions);
        const Constraints constraints = buildConstraints(func);
        const Totals before = totalsOf(func, *states, passCtx);

        std::vector<std::pair<BasicBlock*, std::vector<IRBase*>>> original;
        int moved = 0;
        for (BasicBlock& bb : func) {
            if (!passCtx.shouldProcessBasicBlock(bb) || !states->has(bb)) continue;
            original.emplace_back(&bb, nodesOf(bb));
            moved += refillBlock(bb, *states, constraints, feats);
        }
        if (moved == 0) return PreservedAnalyses::all();

        // Moves change the exit state of their block, and so the entry of its successors.
        states = std::make_unique<WaitAluEntryStates>(func, passCtx, waitOptions);
        const Totals after = totalsOf(func, *states, passCtx);
        const bool better = after.deficit < before.deficit &&
                            after.unhideable <= before.unhideable &&
                            after.vmWaits <= before.vmWaits;
        PASS_DEBUG(std::cerr << "[DsRefillSink] va_vdst waits " << before.vaWaits << " -> "
                             << after.vaWaits << ", WMMAs short " << before.deficit << " -> "
                             << after.deficit << ", vm_vsrc waits " << before.vmWaits << " -> "
                             << after.vmWaits << ": "
                             << (better ? "kept " + std::to_string(moved) + " moved loads"
                                        : std::string("reverted"))
                             << "\n");
        if (!better) {
            for (const auto& [bb, order] : original) restoreOrder(*bb, order);
            return PreservedAnalyses::all();
        }
        return preserveCFGAnalyses();
    }

   private:
    // Plan, and unless reporting move, every refill of `bb` that waits; returns the loads moved.
    int refillBlock(BasicBlock& bb, const WaitAluEntryStates& states,
                    const Constraints& constraints,
                    const PassFeatureConfig::DagFeatures& feats) const {
        BlockScan scan = scanBlock(bb, states);
        const BlockScan start = scan;
        const std::vector<int> gapLoads = loadsPerGap(scan);
        const PlanInputs in{constraints, *std::max_element(gapLoads.begin(), gapLoads.end()),
                            std::max(1, feats.dsBatchWmmas), feats.dsReadDrainLatency};
        std::unordered_set<const StinkyInstruction*> planned;
        int refills = 0;
        int movable = 0;
        int moved = 0;
        for (;;) {
            const DsSite* next = nullptr;
            for (const DsSite& site : scan.ds)
                if (site.wait >= 0 && !planned.count(site.inst)) {
                    next = &site;
                    break;
                }
            if (!next) break;
            const DsSite first = *next;
            planned.insert(first.inst);
            ++refills;
            std::vector<DsSite> group{first};
            RefillPlan plan = planGroup(group, scan, in);
            if (plan.target >= 0) ++movable;
            std::string outcome;
            if (!options_.reportOnly && plan.target >= 0) {
                outcome = moveRefill(bb, states, group, plan, scan, in);
                if (outcome.starts_with("moved")) moved += static_cast<int>(group.size());
            }
            PASS_DEBUG(std::cerr << "[DsRefillSink] bb=\"" << bb.getLabel() << "\" slot "
                                 << first.slot << " " << first.inst->getHwInstDesc()->mnemonic
                                 << " " << regText(first.inst->getDestRegs().front())
                                 << " N=" << first.wait << " D=" << first.untilFree << " window=["
                                 << plan.free << "," << slotText(plan.due)
                                 << "] stage-end=" << plan.stageEnd << " -> "
                                 << (plan.target >= 0 ? "slot " + std::to_string(plan.target)
                                                      : std::string("stays"))
                                 << " (" << plan.why << ")"
                                 << (outcome.empty() ? std::string() : ", " + outcome) << "\n");
        }
        if (options_.reportOnly) {
            PASS_DEBUG(std::cerr << "[DsRefillSink] bb=\"" << bb.getLabel() << "\" va_vdst waits="
                                 << start.vaWaits << " deficit=" << start.deficit
                                 << " unhideable=" << start.unhideable
                                 << " ds_load waits=" << refills << " movable=" << movable << "\n");
        } else {
            PASS_DEBUG(std::cerr << "[DsRefillSink] bb=\"" << bb.getLabel() << "\" va_vdst waits "
                                 << start.vaWaits << " -> " << scan.vaWaits << ", deficit "
                                 << start.deficit << " -> " << scan.deficit << ", moved " << moved
                                 << " loads\n");
        }
        return moved;
    }

    // Move `group` as `plan` says, together with the loads its first load's wait used to cover,
    // and keep the move only if the block comes out ahead. On success `scan` is the new block.
    std::string moveRefill(BasicBlock& bb, const WaitAluEntryStates& states,
                           std::vector<DsSite>& group, RefillPlan& plan, BlockScan& scan,
                           const PlanInputs& in) const {
        BlockScan after;
        for (int round = 0; round < kMaxGroupRounds; ++round) {
            std::vector<size_t> members;
            for (const DsSite& m : group) members.push_back(m.pos);
            moveGroup(bb, scan.nodes, members, plan.insertBefore);
            after = scanBlock(bb, states);
            std::vector<DsSite> uncovered;
            for (const DsSite& load : after.ds) {
                if (load.wait < 0) continue;
                const int was = scan.dsIndexOf(load.inst);
                if (was < 0 || scan.ds[static_cast<size_t>(was)].wait >= 0) continue;
                DsSite site = scan.ds[static_cast<size_t>(was)];
                if (site.pos <= group.front().pos || site.pos >= plan.insertBefore) continue;
                site.wait = load.wait;
                site.untilFree = load.untilFree;
                uncovered.push_back(site);
            }
            if (uncovered.empty()) {
                const std::string worse = whyNotNetWin(scan, after);
                if (!worse.empty()) {
                    restoreOrder(bb, scan.nodes);
                    return "kept: " + worse;
                }
                scan = std::move(after);
                return "moved " + std::to_string(group.size()) + " loads";
            }
            restoreOrder(bb, scan.nodes);
            group.insert(group.end(), uncovered.begin(), uncovered.end());
            std::sort(group.begin(), group.end(),
                      [](const DsSite& a, const DsSite& b) { return a.pos < b.pos; });
            plan = planGroup(group, scan, in);
            if (plan.target < 0) return "kept: the loads it covers cannot move (" + plan.why + ")";
        }
        return "kept: the group does not settle";
    }

    DsRefillSinkOptions options_;
};

char DsRefillSinkPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createDsRefillSinkPass(DsRefillSinkOptions options) {
    return std::make_unique<DsRefillSinkPass>(options);
}
}  // namespace stinkytofu
