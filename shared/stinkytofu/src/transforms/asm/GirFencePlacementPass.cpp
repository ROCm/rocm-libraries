/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "stinkytofu/transforms/asm/GirFencePlacementPass.hpp"

#include <algorithm>
#include <climits>
#include <deque>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/asm/GirFrameAnalysis.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"
#include "stinkytofu/transforms/asm/waitcnt/GirFrameCounterFlow.hpp"

#define DEBUG_TYPE "GirFencePlacementPass"

namespace {
using namespace stinkytofu;

constexpr int kWorkgroupBarrierId = -1;

/// Producer occurrences that have executed with no fence crossed since.  A fence publishes
/// EVERYTHING, so the kill is total: tracking discharge per-storage instead is what invents
/// redundant fences, because each one then looks like it covers only its own hazard.
using Live = std::set<size_t>;

/// `(block, frame)` is the finite quotient the frame model already repeats to, which is what
/// bounds this fixpoint exactly as it bounds the wait flow.
struct Node {
    BasicBlock* block = nullptr;
    GirFrame frame;

    bool operator<(const Node& other) const {
        if (block != other.block) return std::less<BasicBlock*>{}(block, other.block);
        return frame < other.frame;
    }
    bool operator==(const Node& other) const = default;
};

/// Instructions a fence has been DECIDED to precede.  Holding the plan here instead of in the IR
/// is what lets the fixpoint, the coverage check and the shrink all run over one unchanging
/// program: no index goes stale, ownership is intrinsic, and an undo is a set erase.
using Markers = std::set<const StinkyInstruction*>;
using HazardKinds = std::set<GirHazardKind>;
using MarkerKinds = std::map<const StinkyInstruction*, HazardKinds>;

struct MarkerOwnership {
    HazardKinds kinds;
    bool allRawAreTensor = true;
};

using MarkerOwnershipMap = std::map<const StinkyInstruction*, MarkerOwnership>;

const char* hazardKindName(GirHazardKind kind) {
    switch (kind) {
        case GirHazardKind::RAW:
            return "RAW";
        case GirHazardKind::WAR:
            return "WAR";
        case GirHazardKind::WAW:
            return "WAW";
    }
    STINKY_UNREACHABLE("unknown GIR hazard kind");
}

std::string fenceStamp(const HazardKinds& kinds) {
    std::string result = "GIR fence (";
    bool first = true;
    for (GirHazardKind kind : kinds) {
        if (!first) result += ",";
        result += hazardKindName(kind);
        first = false;
    }
    result += ")";
    return result;
}

bool publishes(const StinkyInstruction& inst, const Markers& markers) {
    return isBarrier(inst) || markers.count(&inst) != 0;
}

/// Walk one node, applying gen at producers and the total kill at fences.  `report` sees every
/// consumer whose producer is still live -- an undischarged hazard.
template <class Report>
Live transfer(const Node& node, const std::vector<StinkyInstruction*>& body,
              const GirFrameHazardAnalysis::Result& hazards, const Live& in, const Markers& markers,
              Report report) {
    Live live = in;
    for (size_t i = 0; i < body.size(); ++i) {
        StinkyInstruction* inst = body[i];
        // A marker names the instruction the fence is inserted BEFORE, so it kills here, exactly
        // where a materialized barrier at this slot would.
        if (publishes(*inst, markers)) live.clear();
        for (size_t h = 0; h < hazards.hazards.size(); ++h) {
            const GirFrameHazard& hazard = hazards.hazards[h];
            if (!hazard.crossAgent) continue;
            if (hazard.consumer == inst && hazard.consumerFrame == node.frame && live.count(h))
                report(i, h);
        }
        for (size_t h = 0; h < hazards.hazards.size(); ++h) {
            const GirFrameHazard& hazard = hazards.hazards[h];
            if (!hazard.crossAgent) continue;
            if (hazard.producer == inst && hazard.producerFrame == node.frame) live.insert(h);
        }
    }
    return live;
}

struct FrameCFG {
    std::vector<Node> nodes;
    std::map<Node, size_t> id;
    std::vector<std::vector<size_t>> preds;
    std::vector<std::vector<StinkyInstruction*>> body;
};

/// Every block of the region, not just the schedulable ones: GIR's own CFG roles are spread over
/// `entry`/`skipPGR`/`LoopEndL` after ScaffoldMapPass renames them, and dropping one severs the
/// path a loop-carried producer travels to reach its consumer.
FrameCFG buildFrameCFG(Function& function, const GirFrameAnalysis::Result& frames) {
    FrameCFG cfg;
    for (BasicBlock& block : function) {
        for (const GirFrame& frame : frames.frames(&block)) {
            cfg.id[Node{&block, frame}] = cfg.nodes.size();
            cfg.nodes.push_back(Node{&block, frame});
            std::vector<StinkyInstruction*> body;
            for (IRBase& node : block)
                if (auto* inst = dyn_cast<StinkyInstruction>(&node)) body.push_back(inst);
            cfg.body.push_back(std::move(body));
        }
    }
    // A `(block, frame)` that appears only in the edge set still carries liveness; dropping its
    // edge for want of a node is what leaves a hazard undischarged.
    const auto nodeId = [&cfg](const GirFrameNode& node) {
        auto found = cfg.id.find(Node{node.block, node.frame});
        if (found != cfg.id.end()) return found->second;
        const size_t id = cfg.nodes.size();
        cfg.id[Node{node.block, node.frame}] = id;
        cfg.nodes.push_back(Node{node.block, node.frame});
        std::vector<StinkyInstruction*> body;
        for (IRBase& inner : *node.block)
            if (auto* inst = dyn_cast<StinkyInstruction>(&inner)) body.push_back(inst);
        cfg.body.push_back(std::move(body));
        return id;
    };
    std::vector<std::pair<size_t, size_t>> edges;
    for (const auto& [from, tos] : frames.edges) {
        const size_t source = nodeId(from);
        for (const GirFrameNode& to : tos) edges.emplace_back(source, nodeId(to));
    }
    cfg.preds.resize(cfg.nodes.size());
    for (const auto& [source, sink] : edges) cfg.preds[sink].push_back(source);
    return cfg;
}

/// Least fixpoint of the may-reach set; meet is union, so a producer unfenced on ANY path in is
/// unfenced here.  Finite frames plus a monotone transfer bound the iteration.
std::vector<Live> solve(const FrameCFG& cfg, const GirFrameHazardAnalysis::Result& hazards,
                        const Markers& markers) {
    std::vector<Live> in(cfg.nodes.size());
    std::vector<Live> out(cfg.nodes.size());
    std::deque<size_t> work;
    for (size_t i = 0; i < cfg.nodes.size(); ++i) work.push_back(i);

    while (!work.empty()) {
        const size_t n = work.front();
        work.pop_front();
        Live merged;
        for (size_t p : cfg.preds[n]) merged.insert(out[p].begin(), out[p].end());
        if (merged == in[n] && !out[n].empty()) continue;
        in[n] = merged;
        Live next =
            transfer(cfg.nodes[n], cfg.body[n], hazards, in[n], markers, [](size_t, size_t) {});
        if (next == out[n]) continue;
        out[n] = std::move(next);
        for (size_t m = 0; m < cfg.nodes.size(); ++m)
            if (std::find(cfg.preds[m].begin(), cfg.preds[m].end(), n) != cfg.preds[m].end())
                work.push_back(m);
    }
    return in;
}

/// The earliest consumer still seeing its own producer, and every hazard violating there.
bool firstViolation(const FrameCFG& cfg, const GirFrameHazardAnalysis::Result& hazards,
                    const std::vector<Live>& in, const Markers& markers, size_t& node, size_t& slot,
                    std::vector<size_t>& violating) {
    for (size_t n = 0; n < cfg.nodes.size(); ++n) {
        size_t found = cfg.body[n].size();
        std::vector<size_t> here;
        transfer(cfg.nodes[n], cfg.body[n], hazards, in[n], markers, [&](size_t i, size_t h) {
            if (i < found) {
                found = i;
                here.clear();
            }
            if (i == found) here.push_back(h);
        });
        if (found < cfg.body[n].size()) {
            node = n;
            slot = found;
            violating = std::move(here);
            return true;
        }
    }
    return false;
}

enum class Counter { None, Ds, Tensor };

Counter counterOfProducer(const StinkyInstruction& inst) {
    if (isTensorLoad(inst)) return Counter::Tensor;
    if (isDSRead(inst) || isDSWrite(inst)) return Counter::Ds;
    return Counter::None;
}

bool countsFor(const StinkyInstruction& inst, Counter counter) {
    if (counter == Counter::Ds) return isDSRead(inst) || isDSWrite(inst);
    if (counter == Counter::Tensor) return isTensorLoad(inst);
    return false;
}

/// Give every tensor/DS RAW its own latest physical publication point. RAWs from one logical
/// producer action may share the earliest of their deadlines only when that slot is legal for
/// every edge and no issue on the producer's counter lies between them; each edge's FIFO rank is
/// then unchanged. Different groups that land on the same instruction naturally share one marker.
Markers rawMarkers(const FrameCFG& cfg, const GirFrameHazardAnalysis::Result& hazards) {
    using Group = std::tuple<BasicBlock*, uint64_t, Counter>;
    using ByDeadline = std::map<size_t, std::vector<const GirFrameHazard*>>;
    std::map<Group, ByDeadline> deadlines;
    for (const GirFrameHazard& hazard : hazards.hazards) {
        if (!hazard.crossAgent || hazard.kind != GirHazardKind::RAW) continue;
        const Counter counter = counterOfProducer(*hazard.producer);
        if (counter == Counter::None) continue;
        deadlines[{hazard.consumerBlock, hazard.producerAction, counter}][hazard.consumerIndex]
            .push_back(&hazard);
    }

    std::map<BasicBlock*, const std::vector<StinkyInstruction*>*, std::less<BasicBlock*>> bodies;
    for (size_t n = 0; n < cfg.nodes.size(); ++n) bodies.emplace(cfg.nodes[n].block, &cfg.body[n]);

    Markers markers;
    for (const auto& [group, byDeadline] : deadlines) {
        BasicBlock* block = std::get<0>(group);
        const Counter counter = std::get<2>(group);
        auto body = bodies.find(block);
        if (body == bodies.end() || byDeadline.empty())
            report_fatal_error("GirFencePlacementPass: RAW consumer block has no frame body");
        const std::vector<StinkyInstruction*>& instructions = *body->second;

        const auto legalAt = [block](const GirFrameHazard& hazard, size_t slot) {
            if (slot > hazard.consumerIndex) return false;
            return hazard.producerBlock != block || hazard.gap != 0 || slot > hazard.producerIndex;
        };
        for (const auto& [slot, edges] : byDeadline) {
            if (slot >= instructions.size() ||
                !std::all_of(edges.begin(), edges.end(),
                             [&](const GirFrameHazard* hazard) { return legalAt(*hazard, slot); }))
                report_fatal_error("GirFencePlacementPass: RAW has no legal consumer deadline");
        }

        auto deadline = byDeadline.rbegin();
        size_t clusterHigh = deadline->first;
        size_t clusterLow = deadline->first;
        std::vector<const GirFrameHazard*> cluster = deadline->second;
        const auto flush = [&] {
            if (clusterLow >= instructions.size())
                report_fatal_error("GirFencePlacementPass: RAW deadline is outside its block");
            markers.insert(instructions[clusterLow]);
        };
        for (++deadline; deadline != byDeadline.rend(); ++deadline) {
            const size_t candidate = deadline->first;
            bool crossedIssue =
                candidate >= instructions.size() || clusterHigh > instructions.size();
            for (size_t i = candidate; !crossedIssue && i < clusterHigh; ++i) {
                StinkyInstruction& inst = *instructions[i];
                crossedIssue = countsFor(inst, counter) || isBarrier(inst);
                int observed[waitcnt::CK_Count];
                if (!crossedIssue && waitcnt::observedWaitDrains(inst, observed)) {
                    const waitcnt::CounterKind kind =
                        counter == Counter::Ds ? waitcnt::CK_DS : waitcnt::CK_Tensor;
                    crossedIssue = observed[kind] >= 0;
                }
            }
            const bool legal =
                std::all_of(
                    cluster.begin(), cluster.end(),
                    [&](const GirFrameHazard* hazard) { return legalAt(*hazard, candidate); }) &&
                std::all_of(
                    deadline->second.begin(), deadline->second.end(),
                    [&](const GirFrameHazard* hazard) { return legalAt(*hazard, candidate); });
            if (crossedIssue || !legal) {
                flush();
                clusterHigh = candidate;
                cluster = deadline->second;
            } else {
                cluster.insert(cluster.end(), deadline->second.begin(), deadline->second.end());
            }
            clusterLow = candidate;
        }
        flush();
    }
    return markers;
}

MarkerOwnershipMap collectMarkerOwnership(const GirFrameAnalysis::Result& frames,
                                          const GirFrameHazardAnalysis::Result& hazards,
                                          const Markers& markers,
                                          const std::function<bool(const BasicBlock&)>& covers) {
    const auto marked = [&markers](const StinkyInstruction& inst) {
        return markers.count(&inst) != 0;
    };
    MarkerOwnershipMap result;
    for (const GirFrameHazard& hazard : hazards.hazards) {
        if (!hazard.crossAgent || !covers(*hazard.consumerBlock)) continue;
        const GirHazardFenceCover cover = enclosingFences(frames, hazard, marked);
        for (const GirHazardFenceOwner& owner : cover.owners) {
            if (!owner.anchor || !markers.count(owner.anchor)) continue;
            MarkerOwnership& summary = result[owner.anchor];
            summary.kinds.insert(hazard.kind);
            if (hazard.kind == GirHazardKind::RAW)
                summary.allRawAreTensor &=
                    waitcnt::classifyMemOp(*hazard.producer) == waitcnt::CK_Tensor;
        }
    }
    return result;
}

/// What is unfenced just before each instruction, so a candidate slot knows what it would
/// discharge -- and therefore which residuals its one wait must be the minimum of.
std::vector<Live> liveBefore(const Node& node, const std::vector<StinkyInstruction*>& body,
                             const GirFrameHazardAnalysis::Result& hazards, const Live& in,
                             const Markers& markers) {
    std::vector<Live> out(body.size() + 1);
    Live live = in;
    for (size_t i = 0; i < body.size(); ++i) {
        if (publishes(*body[i], markers)) live.clear();
        out[i] = live;
        for (size_t h = 0; h < hazards.hazards.size(); ++h) {
            const GirFrameHazard& hazard = hazards.hazards[h];
            if (hazard.crossAgent && hazard.producer == body[i] &&
                hazard.producerFrame == node.frame)
                live.insert(h);
        }
    }
    out[body.size()] = live;
    return out;
}

waitcnt::CounterKind kindOf(Counter counter) {
    return counter == Counter::Ds ? waitcnt::CK_DS : waitcnt::CK_Tensor;
}

/// Every `GirFrameNode` sharing a `(block, frame)`: an occurrence names the pair, but the frame
/// graph distinguishes nodes by the action that entered them, so all of them carry it.
using NodesByKey = std::map<std::pair<BasicBlock*, GirFrame>, std::vector<GirFrameNode>>;

NodesByKey collectNodes(const GirFrameAnalysis::Result& frames) {
    NodesByKey byKey;
    const auto remember = [&byKey](const GirFrameNode& node) {
        auto& list = byKey[{node.block, node.frame}];
        for (const GirFrameNode& seen : list)
            if (seen == node) return;
        list.push_back(node);
    };
    for (const auto& [node, successors] : frames.edges) {
        remember(node);
        for (const GirFrameNode& successor : successors) remember(successor);
    }
    return byKey;
}

/// What each live hazard already has in flight when this node BEGINS: `base` issues counted over
/// the frame span from its producer, plus the index its in-block tail resumes from.  A residual is
/// then `base` plus the same-counter issues in `[from, slot)`, which is the count the wait pass
/// will measure for a fence at `slot` -- and the part before the block is exactly what a
/// block-local count cannot see.
struct Reach {
    Counter counter = Counter::None;
    int base = 0;
    size_t from = 0;
};

std::optional<Reach> reachAtBlockStart(
    const GirFrameAnalysis::Result& frames, const NodesByKey& byKey, const Node& node,
    const GirFrameHazard& hazard,
    const std::unordered_map<const StinkyInstruction*, size_t>& index) {
    const Counter counter = counterOfProducer(*hazard.producer);
    if (counter == Counter::None) return std::nullopt;

    // A producer in this very node is already positioned; the span is its own distance forward.
    auto producer = index.find(hazard.producer);
    if (hazard.gap == 0 && hazard.producerFrame == node.frame && producer != index.end())
        return Reach{counter, 1, producer->second + 1};

    auto producerNodes = byKey.find({hazard.producerBlock, hazard.producerFrame});
    auto anchorNodes = byKey.find({node.block, node.frame});
    if (producerNodes == byKey.end() || anchorNodes == byKey.end()) return std::nullopt;

    int base = -1;
    for (const GirFrameNode& anchorNode : anchorNodes->second)
        for (const GirFrameNode& producerNode : producerNodes->second) {
            // The hazard's own gap is the frame's answer wherever it lands in this node; a hazard
            // merely passing through is measured to here instead.
            const int span =
                hazard.consumerBlock == node.block && hazard.consumerFrame == node.frame
                    ? hazard.gap
                    : waitcnt::girFrameDistance(frames, producerNode, anchorNode);
            if (span < 0) continue;
            const int count = waitcnt::girFrameIssuesAcrossSpan(
                frames, kindOf(counter), producerNode, hazard.producerIndex, span, anchorNode, 0);
            if (count > 0 && (base < 0 || count < base)) base = count;
        }
    if (base < 0) return std::nullopt;
    return Reach{counter, base, 0};
}

/// The slot the cut goes in.  The dataflow fixes how MANY cuts and which slots are legal; inside
/// that freedom the deepest wait wins, because one barrier carries one wait per counter and that
/// wait is the minimum over everything it discharges.
size_t chooseSlot(const GirFrameAnalysis::Result& frames, const NodesByKey& byKey, const Node& node,
                  const std::vector<StinkyInstruction*>& body,
                  const GirFrameHazardAnalysis::Result& hazards, const std::vector<Live>& live,
                  const std::vector<size_t>& violating, size_t consumerSlot,
                  const Markers& markers) {
    std::unordered_map<const StinkyInstruction*, size_t> index;
    for (size_t i = 0; i < body.size(); ++i) index[body[i]] = i;

    size_t low = 0;
    for (size_t i = consumerSlot; i-- > 0;)
        if (isBranch(*body[i]) || isLabel(*body[i]) || publishes(*body[i], markers)) {
            low = i + 1;
            break;
        }
    // A same-trip producer must already have run, or the cut kills nothing.
    for (size_t h : violating) {
        const GirFrameHazard& hazard = hazards.hazards[h];
        if (hazard.gap != 0) continue;
        auto producer = index.find(hazard.producer);
        if (producer != index.end() && producer->second < consumerSlot)
            low = std::max(low, producer->second + 1);
    }

    std::unordered_map<size_t, std::optional<Reach>> reach;
    const auto reachOf = [&](size_t h) -> const std::optional<Reach>& {
        auto found = reach.find(h);
        if (found == reach.end())
            found =
                reach.emplace(h, reachAtBlockStart(frames, byKey, node, hazards.hazards[h], index))
                    .first;
        return found->second;
    };

    size_t chosen = consumerSlot;
    int best = -1;
    for (size_t slot = low; slot <= consumerSlot; ++slot) {
        int worst = INT_MAX;
        for (size_t h : live[slot]) {
            const std::optional<Reach>& here = reachOf(h);
            if (!here) continue;
            int residual = here->base;
            for (size_t k = here->from; k < slot; ++k)
                if (countsFor(*body[k], here->counter)) ++residual;
            worst = std::min(worst, residual);
        }
        if (worst != INT_MAX && worst > best) {
            best = worst;
            chosen = slot;
        }
    }
    return chosen;
}

class GirFencePlacementPass final : public StinkyInstPass {
   public:
    static char ID;

    const char* getName() const override {
        return "GirFencePlacementPass";
    }
    PassID getPassID() const override {
        return &ID;
    }

    PreservedAnalyses run(Function& function, PassContext& passCtx, AnalysisManager& AM) override {
        const auto& frames = AM.getResult<GirFrameAnalysis>(function);
        const auto& hazards = AM.getResult<GirFrameHazardAnalysis>(function);
        if (frames.empty() || hazards.empty()) return PreservedAnalyses::all();
        if (std::none_of(hazards.hazards.begin(), hazards.hazards.end(),
                         [](const GirFrameHazard& hazard) { return hazard.crossAgent; }))
            return PreservedAnalyses::all();

        const GfxArchID archId =
            getGfxArchID(passCtx.getGemmTileConfig().arch[0], passCtx.getGemmTileConfig().arch[1],
                         passCtx.getGemmTileConfig().arch[2]);
        const HwInstDesc* signalDesc = getMCIDByUOp(GFX::s_barrier_signal, archId);
        const HwInstDesc* waitDesc = getMCIDByUOp(GFX::s_barrier_wait, archId);
        if (!signalDesc || !waitDesc)
            report_fatal_error("GirFencePlacementPass: no workgroup barrier on this architecture");

        const NodesByKey byKey = collectNodes(frames);
        // ONE program, decided over once.  Every phase below reads this CFG and writes only
        // `markers`; nothing touches the IR until the plan is final, so no hazard index goes
        // stale under an insertion and an undo costs a set erase instead of a block rebuild.
        const FrameCFG cfg = buildFrameCFG(function, frames);
        if (cfg.nodes.empty()) return PreservedAnalyses::all();
        const Markers mandatoryRawMarkers = rawMarkers(cfg, hazards);
        Markers markers = mandatoryRawMarkers;
        MarkerKinds markerKinds;
        for (const StinkyInstruction* marker : mandatoryRawMarkers)
            markerKinds[marker].insert(GirHazardKind::RAW);
        const auto marked = [&markers](const StinkyInstruction& inst) {
            return markers.count(&inst) != 0;
        };
        const auto covers = [&passCtx](const BasicBlock& block) {
            return passCtx.shouldProcessBasicBlock(const_cast<BasicBlock&>(block));
        };
        std::unordered_map<const StinkyInstruction*, size_t> position;
        for (BasicBlock& block : function)
            for (IRBase& node : block)
                if (auto* inst = dyn_cast<StinkyInstruction>(&node))
                    position.emplace(inst, position.size());
        const auto inProgramOrder = [&position](const Markers& plan) {
            std::vector<const StinkyInstruction*> out(plan.begin(), plan.end());
            std::sort(out.begin(), out.end(), [&position](const auto* lhs, const auto* rhs) {
                auto left = position.find(lhs), right = position.find(rhs);
                const size_t l = left == position.end() ? SIZE_MAX : left->second;
                const size_t r = right == position.end() ? SIZE_MAX : right->second;
                return l < r;
            });
            return out;
        };

        // Close the fence cut after every wait-aware removal. A mandatory RAW marker can also
        // happen to cut a WAR/WAW path. If its tensor wait later proves redundant, deleting that
        // marker must expose the other hazard to Phase 1 instead of turning the incomplete
        // ownership summary into a fatal error.
        const auto closeFenceCut = [&] {
            // Phase 1: cut every violating hazard, greedily, at the slot with the deepest
            // residual.
            for (;;) {
                const std::vector<Live> in = solve(cfg, hazards, markers);
                size_t node = 0, slot = 0;
                std::vector<size_t> violating;
                if (!firstViolation(cfg, hazards, in, markers, node, slot, violating)) break;
                slot = chooseSlot(
                    frames, byKey, cfg.nodes[node], cfg.body[node], hazards,
                    liveBefore(cfg.nodes[node], cfg.body[node], hazards, in[node], markers),
                    violating, slot, markers);
                const StinkyInstruction* marker = cfg.body[node][slot];
                if (!markers.insert(marker).second)
                    report_fatal_error("GirFencePlacementPass failed to converge");
                for (size_t h : violating) markerKinds[marker].insert(hazards.hazards[h].kind);
            }

            // Phase 2: the cut fixpoint answers LIVENESS; the wait pass asks POSITION -- does a
            // barrier stand before this consumer. Cover what the first question does not imply,
            // asking `lastBarrierBefore` about the plan so both phases speak of the same fences.
            for (const GirFrameHazard& hazard : hazards.hazards) {
                if (!hazard.crossAgent) continue;
                if (!passCtx.shouldProcessBasicBlock(*hazard.consumerBlock)) continue;
                const GirHazardFenceCover cover = enclosingFences(frames, hazard, marked);
                if (!cover.hasHazardPath)
                    report_fatal_error(
                        "GirFencePlacementPass found no exact frame path for a GIR hazard");
                if (!cover.hasUnfencedPath) continue;
                size_t index = 0;
                for (IRBase& node : *hazard.consumerBlock) {
                    auto* inst = dyn_cast<StinkyInstruction>(&node);
                    if (!inst) continue;
                    if (index++ == hazard.consumerIndex) {
                        markers.insert(inst);
                        markerKinds[inst].insert(hazard.kind);
                        break;
                    }
                }
            }

            // Phase 3: greedy can over-place auxiliary WAR/WAW cuts, so drop any the rest cover.
            // RAW markers are ownership points, not hitting-set candidates: deleting a later one
            // would bind that RAW's wait to an earlier fence and shorten its in-flight interval.
            for (bool shrinking = true; shrinking;) {
                shrinking = false;
                for (const StinkyInstruction* candidate : inProgramOrder(markers)) {
                    if (mandatoryRawMarkers.count(candidate)) continue;
                    markers.erase(candidate);
                    size_t n = 0, sl = 0;
                    std::vector<size_t> v;
                    if (!firstViolation(cfg, hazards, solve(cfg, hazards, markers), markers, n, sl,
                                        v)) {
                        markerKinds.erase(candidate);
                        shrinking = true;
                        break;
                    }
                    markers.insert(candidate);
                }
            }
        };
        closeFenceCut();

        // Phase 3b: plan waits against the still-virtual fence set. Tensor RAW ownership without a
        // tensor wait publishes no new completion, so drop that ownership. If the marker then owns
        // nothing else, retire it and close the fence cut again: another hazard that happened to
        // rely on this marker then receives its own marker and counter wait. Fence placement and
        // wait ownership therefore reach one fixpoint before IR materialization.
        bool waitAwareShrinkConverged = false;
        const size_t markerBound = position.size();
        Markers waitRetiredMarkers;
        waitcnt::WaitInsertionPlan finalVirtualWaits;
        for (size_t round = 0; round <= markerBound; ++round) {
            MarkerOwnershipMap ownership = collectMarkerOwnership(frames, hazards, markers, covers);
            waitcnt::WaitInsertionPlan virtualWaits =
                waitcnt::buildGirFrameWaitPlan(function, frames, hazards, covers, marked);

            bool removed = false;
            for (const StinkyInstruction* candidate : inProgramOrder(markers)) {
                auto owned = ownership.find(candidate);
                auto wait =
                    virtualWaits.anchorWaits.find(const_cast<StinkyInstruction*>(candidate));
                const bool hasTensorWait =
                    wait != virtualWaits.anchorWaits.end() &&
                    wait->second.tensorCount != waitcnt::WaitCountSpec::kUnused;

                const bool dropsUnwaitedRaw = owned != ownership.end() && !hasTensorWait &&
                                              owned->second.kinds.contains(GirHazardKind::RAW) &&
                                              owned->second.allRawAreTensor;
                if (dropsUnwaitedRaw) owned->second.kinds.erase(GirHazardKind::RAW);
                const bool retiresUnwaitedRaw =
                    dropsUnwaitedRaw && !waitRetiredMarkers.count(candidate);
                if (!retiresUnwaitedRaw && owned != ownership.end() && !owned->second.kinds.empty())
                    continue;

                markers.erase(candidate);
                markerKinds.erase(candidate);
                waitRetiredMarkers.insert(candidate);
                removed = true;
                break;
            }
            if (removed) {
                closeFenceCut();
                for (const StinkyInstruction* retired : waitRetiredMarkers) {
                    if (!markers.count(retired)) continue;
                    auto kinds = markerKinds.find(retired);
                    const bool reintroducedForNonRaw =
                        kinds != markerKinds.end() &&
                        std::any_of(kinds->second.begin(), kinds->second.end(),
                                    [](GirHazardKind kind) { return kind != GirHazardKind::RAW; });
                    if (!reintroducedForNonRaw)
                        report_fatal_error(
                            "GirFencePlacementPass reintroduced an unwaited/unowned marker");
                }
                continue;
            }

            markerKinds.clear();
            for (const auto& [marker, summary] : ownership) markerKinds[marker] = summary.kinds;
            finalVirtualWaits = std::move(virtualWaits);
            waitAwareShrinkConverged = true;
            break;
        }
        if (!waitAwareShrinkConverged)
            report_fatal_error("GirFencePlacementPass wait-aware shrink failed to converge");

        for (const auto& [marker, kinds] : markerKinds) {
            if (!kinds.contains(GirHazardKind::RAW)) continue;
            auto wait = finalVirtualWaits.anchorWaits.find(const_cast<StinkyInstruction*>(marker));
            if (wait == finalVirtualWaits.anchorWaits.end() ||
                wait->second.tensorCount == waitcnt::WaitCountSpec::kUnused)
                report_fatal_error("GirFencePlacementPass stamped RAW without a tensorcnt");
        }

        // A pure WAR marker is movable toward its tensor consumer. If the next WAR-owning marker
        // in the same block has no tensor issue before it, Phase 3 must have removed the earlier
        // one. Keeping both would add an earlier dscnt/barrier without ordering any extra write.
        for (BasicBlock& block : function) {
            const StinkyInstruction* earlierPureWar = nullptr;
            bool tensorIssued = false;
            for (IRBase& node : block) {
                auto* inst = dyn_cast<StinkyInstruction>(&node);
                if (!inst) continue;
                if (markers.count(inst)) {
                    const HazardKinds& kinds = markerKinds.at(inst);
                    if (kinds.contains(GirHazardKind::WAR)) {
                        if (earlierPureWar && !tensorIssued)
                            report_fatal_error("GirFencePlacementPass left mergeable WAR barriers");
                        earlierPureWar = kinds.size() == 1
                                             ? inst
                                             : static_cast<const StinkyInstruction*>(nullptr);
                        tensorIssued = false;
                    }
                }
                if (isTensorLoad(*inst)) tensorIssued = true;
            }
        }

        // Phase 4: and only now does the plan become instructions.
        const size_t placed = markers.size();
        const size_t erased = 0;
        for (const StinkyInstruction* target : inProgramOrder(markers)) {
            auto* anchor = const_cast<StinkyInstruction*>(target);
            AsmIRBuilder builder(*anchor->getParent(), archId);
            StinkyInstruction* signal = builder.create(signalDesc, anchor);
            signal->addSrcReg(StinkyRegister(kWorkgroupBarrierId));
            StinkyInstruction* wait = builder.create(waitDesc, anchor);
            wait->addSrcReg(StinkyRegister(kWorkgroupBarrierId));
            auto kinds = markerKinds.find(target);
            if (kinds == markerKinds.end() || kinds->second.empty())
                report_fatal_error("GirFencePlacementPass: barrier has no hazard-kind stamp");
            wait->addModifier<CommentData>(CommentData{fenceStamp(kinds->second)});
        }

        if (placed == 0 && erased == 0) return PreservedAnalyses::all();
        AM.invalidate(function, preserveCFGAnalyses());
        return preserveCFGAnalyses();
    }
};

char GirFencePlacementPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createGirFencePlacementPass() {
    return std::make_unique<GirFencePlacementPass>();
}
}  // namespace stinkytofu
