/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "stinkytofu/transforms/asm/waitcnt/GirFrameCounterFlow.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <deque>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitDataflow.hpp"

// `StinkyTofuDebugPass: "GirFrameWaits"` traces every hazard the counter flow accepted
// and the wait each anchor ended on.
#define DEBUG_TYPE "GirFrameWaits"

namespace stinkytofu::waitcnt {

namespace {

struct IssueKey {
    StinkyInstruction* inst = nullptr;
    GirFrame frame;

    bool operator==(const IssueKey&) const = default;
    bool operator<(const IssueKey& other) const {
        if (inst != other.inst) return std::less<StinkyInstruction*>{}(inst, other.inst);
        return frame < other.frame;
    }
};

struct StaticSite {
    StinkyInstruction* anchor = nullptr;
    CounterKind counter = CK_Count;

    bool operator==(const StaticSite&) const = default;
    bool operator<(const StaticSite& other) const {
        if (anchor != other.anchor) return std::less<StinkyInstruction*>{}(anchor, other.anchor);
        return counter < other.counter;
    }
};

struct TailSite {
    BasicBlock* block = nullptr;
    CounterKind counter = CK_Count;

    bool operator==(const TailSite&) const = default;
    bool operator<(const TailSite& other) const {
        if (block != other.block) return std::less<BasicBlock*>{}(block, other.block);
        return counter < other.counter;
    }
};

struct DynamicSite {
    StinkyInstruction* anchor = nullptr;
    GirFrame frame;
    CounterKind counter = CK_Count;

    bool operator<(const DynamicSite& other) const {
        if (anchor != other.anchor) return std::less<StinkyInstruction*>{}(anchor, other.anchor);
        if (!(frame == other.frame)) return frame < other.frame;
        return counter < other.counter;
    }
};

struct Potential {
    IssueKey producer;
    size_t producerIndex = 0;
    int anchorGap = 0;
    GirHazardKind kind = GirHazardKind::RAW;
};

struct RequirementSummary {
    int strictest = WaitCountSpec::kUnused;
    std::map<BasicBlock*, int, std::less<BasicBlock*>> byPred;
    std::set<IssueKey> producers;
    std::map<BasicBlock*, std::set<IssueKey>, std::less<BasicBlock*>> producersByPred;
    std::set<GirHazardKind> kinds;

    void record(BasicBlock* pred, int wait, const IssueKey& producer, GirHazardKind kind) {
        if (wait < 0) return;
        if (strictest == WaitCountSpec::kUnused || wait < strictest) strictest = wait;
        auto [it, inserted] = byPred.emplace(pred, wait);
        if (!inserted) it->second = std::min(it->second, wait);
        producers.insert(producer);
        producersByPred[pred].insert(producer);
        kinds.insert(kind);
    }
};

using DecisionMap = std::map<StaticSite, int>;
using TailDecisionMap = std::map<TailSite, int>;
using TailProducerMap = std::map<TailSite, std::set<IssueKey>>;
using PotentialMap = std::map<DynamicSite, std::vector<Potential>>;
using RequirementMap = std::map<StaticSite, RequirementSummary>;

CounterKind counterFor(const GirFrameHazard& hazard) {
    if (hazard.kind == GirHazardKind::RAW) return classifyMemOp(*hazard.producer);
    if (classifyMemOp(*hazard.producer) == CK_DS) return CK_DS;
    if (classifyMemOp(*hazard.consumer) == CK_DS) return CK_DS;
    return CK_Tensor;
}

/// Which hazard a fatal frame-graph check is about; without it the abort names a rule and no case.
std::string hazardText(const GirFrameHazard& hazard) {
    std::ostringstream out;
    out << (hazard.kind == GirHazardKind::RAW   ? "RAW"
            : hazard.kind == GirHazardKind::WAR ? "WAR"
                                                : "WAW")
        << " LDS" << hazard.storage << " gap=" << hazard.gap
        << " prod=" << hazard.producer->getParent()->getLabel() << "#" << hazard.producerIndex
        << "/act" << girExternalActionId(hazard.producerAction)
        << " cons=" << hazard.consumer->getParent()->getLabel() << "#" << hazard.consumerIndex
        << "/act" << girExternalActionId(hazard.consumerAction);
    return out.str();
}

size_t indexInBlock(StinkyInstruction& inst);

PotentialMap buildPotentials(Function& function, const GirFrameAnalysis::Result& frames,
                             const GirFrameHazardAnalysis::Result& hazards,
                             const std::function<bool(const BasicBlock&)>& covers,
                             const std::function<bool(const StinkyInstruction&)>& alsoFences) {
    PotentialMap result;
    std::set<std::tuple<StinkyInstruction*, GirFrame, CounterKind, StinkyInstruction*, GirFrame,
                        size_t, int, int>>
        seen;
    std::set<const BasicBlock*> modeled;
    for (BasicBlock& block : function) modeled.insert(&block);
    const GirPlanFenceCoverage plan = buildGirPlanFenceCoverage(frames, hazards, alsoFences);
    for (size_t h = 0; h < hazards.hazards.size(); ++h) {
        const GirFrameHazard& hazard = hazards.hazards[h];
        // GIR models exactly the scheduling region, so an endpoint outside this function is not a
        // hazard of it -- it is a tag riding on a fork that the region never modeled.
        if (!modeled.count(hazard.producer->getParent()) ||
            !modeled.count(hazard.consumer->getParent()))
            continue;
        if (!covers(*hazard.consumer->getParent())) continue;
        CounterKind counter = counterFor(hazard);
        if (counter != CK_DS && counter != CK_Tensor) continue;
        const auto addPotential = [&](StinkyInstruction* anchor, const GirFrame& anchorFrame,
                                      int anchorGap) {
            auto key =
                std::make_tuple(anchor, anchorFrame, counter, hazard.producer, hazard.producerFrame,
                                hazard.producerIndex, hazard.gap, anchorGap);
            if (!seen.insert(key).second) return;
            PASS_DEBUG(std::cerr << "[gir-haz] fn=" << function.getName()
                                 << " counter=" << (counter == CK_DS ? "ds" : "tensor")
                                 << " kind=" << static_cast<int>(hazard.kind)
                                 << " crossAgent=" << hazard.crossAgent << " gap=" << hazard.gap
                                 << " prod=" << hazard.producer->getParent()->getLabel() << "#"
                                 << hazard.producerIndex << "/act"
                                 << girExternalActionId(hazard.producerAction)
                                 << " cons=" << hazard.consumer->getParent()->getLabel() << "#"
                                 << hazard.consumerIndex << "/act"
                                 << girExternalActionId(hazard.consumerAction)
                                 << " anchor=" << anchor->getParent()->getLabel() << "#"
                                 << indexInBlock(*anchor) << " anchorGap=" << anchorGap << "\n");
            result[{anchor, anchorFrame, counter}].push_back(
                {{hazard.producer, hazard.producerFrame},
                 hazard.producerIndex,
                 anchorGap,
                 hazard.kind});
        };

        if (!hazard.crossAgent) {
            addPotential(hazard.consumer, hazard.consumerFrame, hazard.gap);
            continue;
        }

        const GirHazardFenceCover& cover = plan.byHazard[h];
        if (!cover.hasHazardPath)
            STINKY_UNREACHABLE(
                ("Cross-agent GIR hazard has no exact frame-graph path -- " + hazardText(hazard))
                    .c_str());
        if (cover.hasUnfencedPath)
            STINKY_UNREACHABLE(
                ("Cross-agent GIR hazard has an unfenced exact frame-graph path -- " +
                 hazardText(hazard))
                    .c_str());
        for (const GirHazardFenceOwner& owner : cover.owners)
            addPotential(owner.anchor, owner.anchorNode.frame, owner.producerToAnchorGap);
    }
    return result;
}

/// Decode from the opcode and the literal, never from SWaitCntData alone: on gfx1250 `dlcnt` is
/// never set, so reading it credited nothing and a second wait was emitted in front of every
/// anchor.  `observedWaitDrains` is the one implementation of this rule.
int observedWait(const StinkyInstruction& inst, CounterKind counter) {
    int counts[CK_Count];
    if (!observedWaitDrains(inst, counts)) return WaitCountSpec::kUnused;
    return counts[counter];
}

int decisionFor(const DecisionMap& decisions, StinkyInstruction* anchor, CounterKind counter) {
    auto found = decisions.find({anchor, counter});
    if (found == decisions.end() || found->second == WaitCountSpec::kUnused) return -1;
    return found->second;
}

size_t indexInBlock(StinkyInstruction& inst) {
    size_t index = 0;
    for (IRBase& node : *inst.getParent()) {
        auto* candidate = dyn_cast<StinkyInstruction>(&node);
        if (!candidate) continue;
        if (candidate == &inst) break;
        ++index;
    }
    return index;
}

/// What a span walk concluded.  `count` is the issue count; `unaccounted` records that at least
/// one path ended without either reaching the anchor or retiring, so the walk did not cover the
/// whole frame graph and `count` may be missing a constraint.
struct SpanResult {
    int count = -1;
    bool unaccounted = false;
};

/// How many same-counter issues stand at or after `potential`'s producer when control reaches the
/// anchor: 1 is the producer alone, so `waitToDrain` retires it at `n - 1`.
///
/// Walked over the producer-to-ANCHOR span rather than read out of a per-node state. A
/// cross-agent anchor may precede its consumer: a loop-tail fence is in the producer trip while
/// the overwrite it guards is at the next trip's head. A state keyed by `(instruction, frame)`
/// also cannot hold an occurrence a full ring period back -- with ring 2 the read two trips ago
/// and this trip's read are the same key, and the newer one overwrites the older, so the lookup
/// answers 1 for a producer a full loop of issues has since buried. The walk crosses the exact
/// producer-to-fence distance and counts those issues.
///
/// `count` is -1 when no path reaches the anchor with the producer still outstanding.  That is a
/// discharge proof ONLY when every path got there by retiring; a path that instead ran out of span
/// or off the end of the frame graph was never modelled at all, and `unaccounted` says so.  The
/// two must not share a return value: treating "I could not model this" as "no constraint" is how
/// a missing wait becomes silent.
///
/// `perPred` collects the MINIMUM over the paths arriving via each predecessor, which is the
/// quantity a per-edge requirement is about.  Reporting instead the predecessor of the single
/// cheapest path makes the answer depend on which of several equal paths the walk happened to
/// reach first -- and that order follows heap addresses, so the same kernel compiled twice got
/// different waits.
template <class RetireAt, class RetireAtEnd>
SpanResult walkSpan(const GirFrameAnalysis::Result& frames, CounterKind counter,
                    const GirFrameNode& producerNode, size_t producerIndex, int gap,
                    const StinkyInstruction* anchor, const GirFrameNode& anchorNode,
                    size_t anchorIndex, RetireAt retireAt, RetireAtEnd retireAtEnd,
                    std::map<BasicBlock*, int, std::less<BasicBlock*>>* perPred) {
    struct Step {
        GirFrameNode node;
        size_t index = 0;
        int steps = 0;
        int count = 0;
        BasicBlock* pred = nullptr;
        // Discharged by a wait already standing in the IR. The hazard model measured its gap
        // without seeing that wait, so such a path must still be WALKED -- it just stops
        // contributing a count.
        bool spent = false;
    };

    const int span = std::max(0, gap);
    int best = -1;
    bool reached = false;  // the traversal got to the anchor, discharged or not
    bool unaccounted = false;
    std::deque<Step> work{{producerNode, producerIndex + 1, 0, 1, nullptr, false}};
    // The frame graph is a graph, so the same state is reachable many ways and an unmemoised walk
    // re-expands it exponentially.  Truncating that with a budget is what made the answer depend
    // on exploration order; deduplicating the state makes the walk finite AND complete, because a
    // state already enqueued can contribute nothing a second time.
    std::unordered_set<size_t> visited;
    const auto stateKey = [](const Step& step) {
        size_t key = GirFrameNodeHash{}(step.node);
        for (const size_t part :
             {step.index, static_cast<size_t>(step.steps), static_cast<size_t>(step.count),
              static_cast<size_t>(step.spent), std::hash<BasicBlock*>{}(step.pred)})
            key = key * 1099511628211ULL ^ part;
        return key;
    };
    visited.insert(stateKey(work.front()));

    while (!work.empty()) {
        Step step = std::move(work.front());
        work.pop_front();

        bool retired = false;
        size_t position = 0;
        for (IRBase& node : *step.node.block) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (!inst) continue;
            const size_t here = position++;
            if (here < step.index) continue;

            // Match the chosen dynamic anchor occurrence exactly. The same static fence can occur
            // more than once within a ring period; accepting another occurrence would either bind
            // a RAW to an unrelated publication point or let a WAR overwrite before its drain.
            if (inst == anchor && step.steps == span && here == anchorIndex &&
                step.node == anchorNode) {
                reached = true;  // the model's own path, found
                if (!step.spent) {
                    if (best < 0 || step.count < best) best = step.count;
                    if (perPred) {
                        auto [entry, inserted] = perPred->emplace(step.pred, step.count);
                        if (!inserted) entry->second = std::min(entry->second, step.count);
                    }
                }
                retired = true;  // this path is answered; do not walk past its own anchor
                break;
            }

            // A wait already standing here discharges the producer, so this path asks for nothing
            // more. It keeps SCANNING, though: the hazard model measured its gap without seeing
            // this wait, and the anchor it measured to is usually further down this very block.
            if (!step.spent && retireAt(*inst, step.count)) step.spent = true;

            if (!step.spent && classifyMemOp(*inst) == counter) ++step.count;
        }
        if (retired) continue;

        if (retireAtEnd(step.node.block, step.count)) step.spent = true;

        auto successors = frames.edges.find(step.node);
        // Ran out of span, or off the end of the frame graph, without ever reaching the anchor:
        // this path was not modelled, so it proves nothing about the producer being retired.
        // Span is counted in GIR blocks, so an edge within one spends none of it.
        const auto costOf = [&](const GirFrameNode& successor) {
            return frames.insideOneGirBlock(step.node, successor) ? 0 : 1;
        };
        // Every state the graph holds owns an edge entry, so a successor WITHOUT one was dropped
        // for having no feasible exit. It is not a path: following it reads a state that does not
        // exist, and finding nothing there reads as running off the end of the model.
        const auto holds = [&](const GirFrameNode& successor) {
            return frames.edges.find(successor) != frames.edges.end();
        };
        if (successors == frames.edges.end() ||
            std::none_of(successors->second.begin(), successors->second.end(),
                         [&](const GirFrameNode& successor) {
                             return holds(successor) && step.steps + costOf(successor) <= span;
                         })) {
            unaccounted = true;
            continue;
        }
        for (const GirFrameNode& successor : successors->second) {
            if (!holds(successor)) continue;
            const int advance = costOf(successor);
            if (step.steps + advance > span) continue;
            Step next{successor, 0, step.steps + advance, step.count, step.node.block, step.spent};
            if (visited.insert(stateKey(next)).second) work.push_back(std::move(next));
        }
    }
    // The hazard model asserted a path at this distance. If the traversal reached the anchor at
    // all -- even on a path a standing wait had already discharged -- the model is reproduced.
    return {best, unaccounted && !reached};
}

SpanResult countAcrossSpan(const GirFrameAnalysis::Result& frames, const DecisionMap& decisions,
                           const TailDecisionMap& tailDecisions, CounterKind counter,
                           const GirFrameNode& producerNode, const Potential& potential,
                           StinkyInstruction* anchor, const GirFrameNode& anchorNode,
                           size_t anchorIndex,
                           std::map<BasicBlock*, int, std::less<BasicBlock*>>* perPred) {
    auto retireAt = [&](const StinkyInstruction& inst, int count) {
        for (const int keep :
             {decisionFor(decisions, const_cast<StinkyInstruction*>(&inst), counter),
              observedWait(inst, counter)})
            if (keep >= 0 && count > keep) return true;
        return false;
    };
    auto retireAtEnd = [&](BasicBlock* block, int count) {
        auto tail = tailDecisions.find({block, counter});
        return tail != tailDecisions.end() && count > tail->second;
    };

    return walkSpan(frames, counter, producerNode, potential.producerIndex, potential.anchorGap,
                    anchor, anchorNode, anchorIndex, retireAt, retireAtEnd, perPred);
}

RequirementMap simulate(Function& function, const GirFrameAnalysis::Result& frames,
                        const PotentialMap& potentials, const DecisionMap& decisions,
                        const TailDecisionMap& tailDecisions) {
    RequirementMap requirements;

    // A `(block, frame)` pair can name several nodes, which differ only by the action that entered
    // them; a producer occurrence names the pair, so every node carrying it is a starting point.
    std::map<std::pair<BasicBlock*, GirFrame>, std::vector<GirFrameNode>> nodesByKey;
    auto remember = [&](const GirFrameNode& node) {
        auto& list = nodesByKey[{node.block, node.frame}];
        for (const GirFrameNode& seen : list)
            if (seen == node) return;
        list.push_back(node);
    };
    for (const auto& [node, successors] : frames.edges) {
        remember(node);
        for (const GirFrameNode& successor : successors) remember(successor);
    }

    for (const auto& [site, list] : potentials) {
        const size_t anchorIndex = indexInBlock(*site.anchor);
        for (const GirFrameNode& anchorNode : nodesByKey[{site.anchor->getParent(), site.frame}]) {
            for (const Potential& potential : list) {
                auto producers = nodesByKey.find(
                    {potential.producer.inst->getParent(), potential.producer.frame});
                if (producers == nodesByKey.end()) continue;
                for (const GirFrameNode& producerNode : producers->second) {
                    std::map<BasicBlock*, int, std::less<BasicBlock*>> perPred;
                    const SpanResult span = countAcrossSpan(
                        frames, decisions, tailDecisions, site.counter, producerNode, potential,
                        site.anchor, anchorNode, anchorIndex, &perPred);
                    // Only a walk that reached the anchor on NO path and ended some path by
                    // exhaustion is a modelling failure.  In a branching graph most paths simply
                    // lead elsewhere, so counting every one of those measures the graph, not a bug.
                    // The hazard analysis asserted this producer-consumer pair and measured its
                    // distance. Reproducing no path at all, while one ran out short, means the
                    // counter flow disagrees with the pair it was handed -- and it then records
                    // no requirement, so a wait that is needed can simply go missing.
                    if (span.count < 0 && span.unaccounted)
                        STINKY_UNREACHABLE(
                            ("GIR counter flow found no path for a hazard it was given: anchor=" +
                             site.anchor->getParent()->getLabel() + "#" +
                             std::to_string(anchorIndex) +
                             " producer=" + producerNode.block->getLabel() + "#" +
                             std::to_string(potential.producerIndex) +
                             " span=" + std::to_string(potential.anchorGap) +
                             " counter=" + (site.counter == CK_DS ? "ds" : "tensor"))
                                .c_str());
                    if (span.count <= 0) continue;
                    for (const auto& [pred, arrived] : perPred)
                        requirements[{site.anchor, site.counter}].record(
                            pred, waitToDrain(site.counter, arrived), potential.producer,
                            potential.kind);
                }
            }
        }
    }
    return requirements;
}

int& counterField(WaitCountSpec& spec, CounterKind counter) {
    if (counter == CK_DS) return spec.dsCount;
    if (counter == CK_Tensor) return spec.tensorCount;
    STINKY_UNREACHABLE("GIR frame wait plan asked for an unsupported counter field");
}

int counterField(const WaitCountSpec& spec, CounterKind counter) {
    if (counter == CK_DS) return spec.dsCount;
    if (counter == CK_Tensor) return spec.tensorCount;
    STINKY_UNREACHABLE("GIR frame wait plan asked for an unsupported counter field");
}

void tightenField(WaitCountSpec& spec, CounterKind counter, int wait) {
    int& field = counterField(spec, counter);
    if (field == WaitCountSpec::kUnused || wait < field) field = wait;
}

void attachTensorTokens(WaitCountSpec& spec, const std::set<IssueKey>& producers) {
    if (spec.tensorCount == WaitCountSpec::kUnused) return;
    for (const IssueKey& producer : producers) {
        const auto* tokens = producer.inst->getModifier<MemTokenData>();
        if (tokens)
            spec.tensorTokens.insert(spec.tensorTokens.end(), tokens->tokens.begin(),
                                     tokens->tokens.end());
    }
    std::sort(spec.tensorTokens.begin(), spec.tensorTokens.end());
    spec.tensorTokens.erase(std::unique(spec.tensorTokens.begin(), spec.tensorTokens.end()),
                            spec.tensorTokens.end());
}

void attachTensorTokens(WaitCountSpec& spec, const RequirementSummary& summary) {
    attachTensorTokens(spec, summary.producers);
}

std::pair<int, bool> prefixIssuesBefore(StinkyInstruction* anchor, CounterKind counter) {
    int issues = 0;
    for (IRBase& node : *anchor->getParent()) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (!inst) continue;
        if (inst == anchor) break;
        if (observedWait(*inst, counter) >= 0) return {issues, true};
        if (classifyMemOp(*inst) == counter) ++issues;
    }
    return {issues, false};
}

using TailKindMap = std::map<TailSite, std::set<GirHazardKind>>;

struct PromotionPlan {
    DecisionMap anchors;
    TailDecisionMap tails;
    TailProducerMap tailProducers;
    TailKindMap tailKinds;
};

PromotionPlan derivePromotions(const RequirementMap& requirements, const DecisionMap& decisions,
                               const std::function<bool(const BasicBlock&)>& covers) {
    PromotionPlan result;
    for (const auto& [site, summary] : requirements) {
        auto current = decisions.find(site);
        if (current == decisions.end() || summary.strictest == WaitCountSpec::kUnused) continue;
        BasicBlock* block = site.anchor->getParent();
        const auto& predecessors = block->getPredecessors();
        size_t constrainedPredecessors = 0;
        for (const auto& [pred, wait] : summary.byPred) {
            (void)wait;
            if (pred) ++constrainedPredecessors;
        }
        if (predecessors.size() <= 1 || constrainedPredecessors < 2) continue;

        const auto [prefixIssues, prefixHasWait] = prefixIssuesBefore(site.anchor, site.counter);
        if (prefixHasWait) continue;

        int common = WaitCountSpec::kUnused;
        for (const auto& [pred, wait] : summary.byPred)
            if (pred) common = std::max(common, wait);
        if (common <= current->second) continue;

        PromotionPlan local;
        bool valid = true;
        for (const auto& [pred, need] : summary.byPred) {
            if (!pred || need >= common) continue;
            if (!covers(*pred) || pred->getSuccessors().size() != 1 ||
                pred->getSuccessors().front() != block) {
                valid = false;
                break;
            }
            const int edgeWait = need - prefixIssues;
            if (edgeWait < 0) {
                valid = false;
                break;
            }
            TailSite edge{pred, site.counter};
            auto [tail, inserted] = local.tails.emplace(edge, edgeWait);
            if (!inserted) tail->second = std::min(tail->second, edgeWait);
            auto producers = summary.producersByPred.find(pred);
            if (producers != summary.producersByPred.end())
                local.tailProducers[edge].insert(producers->second.begin(),
                                                 producers->second.end());
            local.tailKinds[edge].insert(summary.kinds.begin(), summary.kinds.end());
        }
        if (!valid || local.tails.empty()) continue;
        result.anchors[site] = common;
        for (const auto& [edge, wait] : local.tails) {
            auto [tail, inserted] = result.tails.emplace(edge, wait);
            if (!inserted) tail->second = std::min(tail->second, wait);
        }
        for (const auto& [edge, producers] : local.tailProducers)
            result.tailProducers[edge].insert(producers.begin(), producers.end());
        for (const auto& [edge, edgeKinds] : local.tailKinds)
            result.tailKinds[edge].insert(edgeKinds.begin(), edgeKinds.end());
    }
    return result;
}

/// Closes the decisions by STRENGTHENING only, from "no wait" downwards.
///
/// Both directions reach a fixpoint, but only this one reaches the weakest safe assignment.
/// Seeding every anchor at 0 drains the queue at each anchor before the queue is ever read, so a
/// loop-carried producer is never in the FIFO, its rank is never observed, and 0 re-derives itself
/// -- self-consistent and maximally strong.  Seeding at kUnused leaves the pipeline at its real
/// depth, so each hazard reads the rank its own frame actually holds.
RequirementMap closeDecisions(Function& function, const GirFrameAnalysis::Result& frames,
                              const PotentialMap& potentials, DecisionMap& decisions,
                              const TailDecisionMap& tailDecisions) {
    // Every round that does not close strictly lowers one site, and a site falls at most from its
    // first finite rank to 0, so the depth of the tracked queue bounds the descent.
    size_t depth = 0;
    for (BasicBlock& block : function)
        for (IRBase& node : block) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (inst && (classifyMemOp(*inst) == CK_DS || classifyMemOp(*inst) == CK_Tensor))
                ++depth;
        }
    const size_t rounds = decisions.size() * (depth + 2) + 1;

    for (size_t round = 0; round < rounds; ++round) {
        RequirementMap requirements =
            simulate(function, frames, potentials, decisions, tailDecisions);
        DecisionMap next = decisions;
        for (auto& [site, wait] : next) {
            auto found = requirements.find(site);
            if (found == requirements.end()) continue;
            const int need = found->second.strictest;
            if (need == WaitCountSpec::kUnused) continue;
            if (wait == WaitCountSpec::kUnused || need < wait) wait = need;
        }
        if (next == decisions) return requirements;
        decisions = std::move(next);
    }
    STINKY_UNREACHABLE("GIR finite-frame counter flow did not close at its frame-node bound");
}

WaitInsertionPlan materializePlan(const DecisionMap& decisions,
                                  const TailDecisionMap& tailDecisions,
                                  const TailProducerMap& tailProducers,
                                  const RequirementMap& requirements, const TailKindMap& tailKinds,
                                  GirWaitKinds* kinds) {
    WaitInsertionPlan plan;
    for (const auto& [site, wait] : decisions) {
        if (wait == WaitCountSpec::kUnused) continue;
        WaitCountSpec& spec = plan.anchorWaits[site.anchor];
        tightenField(spec, site.counter, wait);
        auto summary = requirements.find(site);
        if (site.counter == CK_Tensor && summary != requirements.end())
            attachTensorTokens(spec, summary->second);
        if (kinds && summary != requirements.end())
            kinds->anchors[{site.anchor, site.counter}].insert(summary->second.kinds.begin(),
                                                               summary->second.kinds.end());
    }

    std::map<BasicBlock*, WaitCountSpec, std::less<BasicBlock*>> tailSpecs;
    for (const auto& [site, wait] : tailDecisions) {
        WaitCountSpec& spec = tailSpecs[site.block];
        tightenField(spec, site.counter, wait);
        auto producers = tailProducers.find(site);
        if (site.counter == CK_Tensor && producers != tailProducers.end())
            attachTensorTokens(spec, producers->second);
        if (kinds)
            if (auto edge = tailKinds.find(site); edge != tailKinds.end())
                kinds->tails[{site.block, site.counter}].insert(edge->second.begin(),
                                                                edge->second.end());
    }
    for (auto& [block, spec] : tailSpecs) plan.tailDrains.push_back({block, std::move(spec)});
    return plan;
}

/// Prove every planned wait can change its counter and identify existing dscnt-only waits that the
/// plan makes redundant. A redundant planned wait is a counter-model defect; an existing wait came
/// from an earlier producer and is returned for reconciliation by the caller.
std::vector<StinkyInstruction*> analyzeCombinedWaitStream(
    Function& function, const WaitInsertionPlan& plan,
    const std::function<bool(const BasicBlock&)>& removableCoverage) {
    using Bounds = std::array<int, CK_Count>;
    const std::array<CounterKind, 2> counters{CK_DS, CK_Tensor};
    constexpr int kBeyondEncodableWait = 64;
    std::vector<StinkyInstruction*> redundantObserved;
    const auto unknownBounds = [] {
        Bounds result;
        result.fill(-1);
        return result;
    };

    const auto scanBlock = [&](BasicBlock& block, Bounds upperBound, bool finalScan) {
        const auto applyPlanned = [&](const WaitCountSpec& spec, CounterKind counter,
                                      const StinkyInstruction* anchor) {
            const int wait = counterField(spec, counter);
            if (wait == WaitCountSpec::kUnused) return;
            int& bound = upperBound[counter];
            if (bound >= 0 && wait >= bound) {
                if (finalScan)
                    STINKY_UNREACHABLE(
                        (std::string("GIR counter flow produced redundant ") +
                         (counter == CK_DS ? "dscnt" : "tensorcnt") +
                         " wait=" + std::to_string(wait) + " after bound=" + std::to_string(bound) +
                         " in " + (anchor ? anchor->getParent()->getLabel() : block.getLabel()))
                            .c_str());
                return;
            }
            if (bound < 0 || wait < bound) bound = wait;
        };

        for (IRBase& node : block) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (!inst) continue;

            auto planned = plan.anchorWaits.find(inst);
            if (planned != plan.anchorWaits.end())
                for (CounterKind counter : counters) applyPlanned(planned->second, counter, inst);

            int observed[CK_Count];
            if (observedWaitDrains(*inst, observed)) {
                for (CounterKind counter : counters) {
                    const int wait = observed[counter];
                    if (wait < 0) continue;
                    int& bound = upperBound[counter];
                    if (bound >= 0 && wait >= bound) {
                        const bool dscntOnlyWait =
                            counter == CK_DS && inst->getUnifiedOpcode() == GFX::s_wait_dscnt;
                        if (finalScan && dscntOnlyWait && removableCoverage &&
                            removableCoverage(block))
                            redundantObserved.push_back(inst);
                        continue;
                    }
                    bound = wait;
                }
            }

            const CounterKind issue = classifyMemOp(*inst);
            if ((issue == CK_DS || issue == CK_Tensor) && upperBound[issue] >= 0)
                upperBound[issue] = std::min(kBeyondEncodableWait, upperBound[issue] + 1);
        }

        for (const TailDrain& drain : plan.tailDrains) {
            if (drain.predBB != &block) continue;
            for (CounterKind counter : counters) applyPlanned(drain.spec, counter, nullptr);
        }
        return upperBound;
    };

    std::map<BasicBlock*, Bounds, std::less<BasicBlock*>> out;
    std::deque<BasicBlock*> work;
    std::unordered_set<BasicBlock*> queued;
    for (BasicBlock& block : function) {
        out[&block] = unknownBounds();
        work.push_back(&block);
        queued.insert(&block);
    }
    const auto mergedInput = [&](BasicBlock& block) {
        Bounds merged = unknownBounds();
        const auto& predecessors = block.getPredecessors();
        if (predecessors.empty()) return merged;
        for (CounterKind counter : counters) {
            bool known = true;
            int bound = 0;
            for (BasicBlock* pred : predecessors) {
                auto found = out.find(pred);
                if (found == out.end() || found->second[counter] < 0) {
                    known = false;
                    break;
                }
                bound = std::max(bound, found->second[counter]);
            }
            if (known) merged[counter] = bound;
        }
        return merged;
    };

    while (!work.empty()) {
        BasicBlock* block = work.front();
        work.pop_front();
        queued.erase(block);
        Bounds next = scanBlock(*block, mergedInput(*block), false);
        if (next == out[block]) continue;
        out[block] = next;
        for (BasicBlock* successor : block->getSuccessors())
            if (out.contains(successor) && queued.insert(successor).second)
                work.push_back(successor);
    }

    for (BasicBlock& block : function) (void)scanBlock(block, mergedInput(block), true);
    return redundantObserved;
}

/// A planned wait that cannot change its counter is a solver defect. Existing waits are checked by
/// the same walk but left to the insertion pass, which owns whether that earlier producer's IR may
/// be rewritten.
void validateNoRedundantWaits(Function& function, const WaitInsertionPlan& plan) {
    (void)analyzeCombinedWaitStream(function, plan, {});
}

StinkyInstruction* instructionAt(BasicBlock* block, size_t index) {
    size_t position = 0;
    for (IRBase& node : *block) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (!inst) continue;
        if (position++ == index) return inst;
    }
    return nullptr;
}

}  // namespace

void removeObservedDsWaitsRedundantWithPlan(Function& function, const WaitInsertionPlan& plan,
                                            const std::function<bool(const BasicBlock&)>& covers) {
    const std::vector<StinkyInstruction*> redundant =
        analyzeCombinedWaitStream(function, plan, covers);
    if (redundant.empty()) return;

    const std::unordered_set<StinkyInstruction*> remove(redundant.begin(), redundant.end());
    for (BasicBlock& block : function) {
        for (auto it = block.begin(); it != block.end();) {
            auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if (!inst || !remove.contains(inst)) {
                ++it;
                continue;
            }
            PASS_DEBUG(std::cerr << "[gir-wait-reconcile] remove redundant observed wait in "
                                 << block.getLabel() << "#" << indexInBlock(*inst) << "\n");
            it = block.eraseIR(it);
        }
    }
}

int girFrameIssuesAcrossSpan(const GirFrameAnalysis::Result& frames, CounterKind counter,
                             const GirFrameNode& producerNode, size_t producerIndex, int span,
                             const GirFrameNode& anchorNode, size_t anchorIndex) {
    StinkyInstruction* anchor = instructionAt(anchorNode.block, anchorIndex);
    if (!anchor) return -1;
    // No planned decisions yet -- a fence is placed before any wait is filled in, so the only
    // thing that can retire the producer early is a wait already standing in the IR.
    const SpanResult result = walkSpan(
        frames, counter, producerNode, producerIndex, span, anchor, anchorNode, anchorIndex,
        [&](const StinkyInstruction& inst, int count) {
            const int keep = observedWait(inst, counter);
            return keep >= 0 && count > keep;
        },
        [](BasicBlock*, int) { return false; }, nullptr);
    if (result.count < 0 && result.unaccounted)
        STINKY_UNREACHABLE(("GIR fence placement found no path for a hazard it was given: anchor=" +
                            anchorNode.block->getLabel() + "#" + std::to_string(anchorIndex) +
                            " producer=" + producerNode.block->getLabel() + "#" +
                            std::to_string(producerIndex) + " span=" + std::to_string(span))
                               .c_str());
    return result.count;
}

int girFrameDistance(const GirFrameAnalysis::Result& frames, const GirFrameNode& from,
                     const GirFrameNode& to) {
    if (from == to) return 0;
    // Distance is counted in GIR blocks, so an edge within one costs nothing and a plain queue no
    // longer pops in distance order: this is a 0-1 BFS, and a node may be relaxed after its visit.
    std::unordered_map<GirFrameNode, int, GirFrameNodeHash> best{{from, 0}};
    std::deque<GirFrameNode> work{from};
    while (!work.empty()) {
        GirFrameNode node = std::move(work.front());
        work.pop_front();
        const int steps = best[node];
        auto successors = frames.edges.find(node);
        if (successors == frames.edges.end()) continue;
        for (const GirFrameNode& successor : successors->second) {
            const int advance = frames.insideOneGirBlock(node, successor) ? 0 : 1;
            auto [slot, fresh] = best.try_emplace(successor, steps + advance);
            if (!fresh) {
                if (slot->second <= steps + advance) continue;
                slot->second = steps + advance;
            }
            if (advance == 0)
                work.push_front(successor);
            else
                work.push_back(successor);
        }
    }
    auto found = best.find(to);
    return found == best.end() ? -1 : found->second;
}

WaitInsertionPlan buildGirFrameWaitPlan(
    Function& function, const GirFrameAnalysis::Result& frames,
    const GirFrameHazardAnalysis::Result& hazards,
    const std::function<bool(const BasicBlock&)>& covers,
    const std::function<bool(const StinkyInstruction&)>& alsoFences, GirWaitKinds* kinds) {
    const PotentialMap potentials = buildPotentials(function, frames, hazards, covers, alsoFences);
    if (potentials.empty()) return {};

    DecisionMap decisions;
    for (const auto& [dynamic, _] : potentials)
        decisions[{dynamic.anchor, dynamic.counter}] = WaitCountSpec::kUnused;

    TailDecisionMap tailDecisions;
    TailProducerMap tailProducers;
    TailKindMap tailKinds;
    RequirementMap requirements =
        closeDecisions(function, frames, potentials, decisions, tailDecisions);

    size_t joinSites = 0;
    for (const auto& [site, wait] : decisions) {
        (void)wait;
        if (site.anchor->getParent()->getPredecessors().size() > 1) ++joinSites;
    }
    for (size_t round = 0; round <= joinSites; ++round) {
        PromotionPlan promotions = derivePromotions(requirements, decisions, covers);
        bool changed = false;
        for (const auto& [site, wait] : promotions.anchors) {
            auto found = decisions.find(site);
            if (found == decisions.end() || found->second != wait) {
                decisions[site] = wait;
                changed = true;
            }
        }
        for (const auto& [site, wait] : promotions.tails) {
            auto [found, inserted] = tailDecisions.emplace(site, wait);
            if (inserted || wait < found->second) {
                found->second = std::min(found->second, wait);
                changed = true;
            }
        }
        for (const auto& [site, producers] : promotions.tailProducers)
            tailProducers[site].insert(producers.begin(), producers.end());
        for (const auto& [site, edgeKinds] : promotions.tailKinds)
            tailKinds[site].insert(edgeKinds.begin(), edgeKinds.end());
        if (!changed) break;
        requirements = closeDecisions(function, frames, potentials, decisions, tailDecisions);
    }

    // Strengthening closure can make a later site unnecessary after an earlier wait is added.
    // Relax one site at a time: accepting all absent sites as one batch can make closure re-add
    // an alternating subset forever. A trial is accepted only when it and every previously
    // accepted removal remain kUnused after re-closing.
    std::vector<StaticSite> sites;
    for (BasicBlock& block : function)
        for (IRBase& node : block) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (!inst) continue;
            for (CounterKind counter : {CK_DS, CK_Tensor}) {
                StaticSite site{inst, counter};
                if (decisions.contains(site)) sites.push_back(site);
            }
        }

    std::set<StaticSite> removedSites;
    std::set<StaticSite> rejectedSites;
    for (;;) {
        requirements = simulate(function, frames, potentials, decisions, tailDecisions);
        bool accepted = false;
        for (const StaticSite& site : sites) {
            if (removedSites.contains(site) || rejectedSites.contains(site)) continue;
            auto decision = decisions.find(site);
            if (decision == decisions.end() || decision->second == WaitCountSpec::kUnused) continue;
            auto found = requirements.find(site);
            if (found != requirements.end() && found->second.strictest != WaitCountSpec::kUnused)
                continue;

            DecisionMap trial = decisions;
            trial[site] = WaitCountSpec::kUnused;
            RequirementMap trialRequirements =
                closeDecisions(function, frames, potentials, trial, tailDecisions);
            bool stable = trial.at(site) == WaitCountSpec::kUnused;
            for (const StaticSite& removed : removedSites)
                stable &= trial.at(removed) == WaitCountSpec::kUnused;
            if (!stable) {
                rejectedSites.insert(site);
                continue;
            }

            PASS_DEBUG(std::cerr << "[gir-relax] anchor=" << site.anchor->getParent()->getLabel()
                                 << "#" << indexInBlock(*site.anchor) << " counter="
                                 << (site.counter == CK_DS ? "ds" : "tensor") << "\n");
            decisions = std::move(trial);
            requirements = std::move(trialRequirements);
            removedSites.insert(site);
            rejectedSites.clear();
            accepted = true;
            break;
        }
        if (!accepted) break;
    }

    // kUnused is "emit nothing", which is weaker than any count -- never a pass for a site the
    // final state still constrains.
    requirements = simulate(function, frames, potentials, decisions, tailDecisions);
    for (const auto& [site, summary] : requirements) {
        if (summary.strictest == WaitCountSpec::kUnused) continue;
        auto decision = decisions.find(site);
        if (decision == decisions.end() || decision->second == WaitCountSpec::kUnused ||
            decision->second > summary.strictest)
            STINKY_UNREACHABLE("GIR finite-frame counter flow produced an unsafe wait decision");
    }
    for (const auto& [site, wait] : decisions) {
        if (wait == WaitCountSpec::kUnused) continue;
        PASS_DEBUG({
            std::cerr << "[gir-wait] fn=" << function.getName()
                      << " anchor=" << site.anchor->getParent()->getLabel() << "#"
                      << indexInBlock(*site.anchor)
                      << " counter=" << (site.counter == CK_DS ? "ds" : "tensor")
                      << " wait=" << wait;
            auto summary = requirements.find(site);
            if (summary != requirements.end())
                for (const IssueKey& producer : summary->second.producers)
                    std::cerr << " <- " << producer.inst->getParent()->getLabel() << "#"
                              << indexInBlock(*producer.inst);
            std::cerr << "\n";
        });
    }
    WaitInsertionPlan plan =
        materializePlan(decisions, tailDecisions, tailProducers, requirements, tailKinds, kinds);
    validateNoRedundantWaits(function, plan);
    return plan;
}

}  // namespace stinkytofu::waitcnt
