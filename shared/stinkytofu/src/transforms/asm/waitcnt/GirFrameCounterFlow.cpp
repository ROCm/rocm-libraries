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

// `StinkyTofuDebugPass: "GirWaitCntInsertionPass"` traces every hazard the counter flow accepted
// and the wait each anchor ended on.
#define DEBUG_TYPE "GirWaitCntInsertionPass"

namespace stinkytofu::waitcnt {

/// Span walks that ended with a path neither reaching the anchor nor retiring -- a hazard the
/// counter flow did not model and then treated as discharged.  Should be 0; `ST_GIR_SPAN_STATS=1`
/// reports it per function.
unsigned girUnaccountedSpans = 0;

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
};

struct RequirementSummary {
    int strictest = WaitCountSpec::kUnused;
    std::map<BasicBlock*, int, std::less<BasicBlock*>> byPred;
    std::set<IssueKey> producers;
    std::map<BasicBlock*, std::set<IssueKey>, std::less<BasicBlock*>> producersByPred;

    void record(BasicBlock* pred, int wait, const IssueKey& producer) {
        if (wait < 0) return;
        if (strictest == WaitCountSpec::kUnused || wait < strictest) strictest = wait;
        auto [it, inserted] = byPred.emplace(pred, wait);
        if (!inserted) it->second = std::min(it->second, wait);
        producers.insert(producer);
        producersByPred[pred].insert(producer);
    }
};

using DecisionMap = std::map<StaticSite, int>;
using TailDecisionMap = std::map<TailSite, int>;
using TailProducerMap = std::map<TailSite, std::set<IssueKey>>;
using PotentialMap = std::map<DynamicSite, std::vector<Potential>>;
using RequirementMap = std::map<StaticSite, RequirementSummary>;
using FeasibleDomainMap = std::unordered_map<GirFrameNode, std::set<int>, GirFrameNodeHash>;

CounterKind counterFor(const GirFrameHazard& hazard) {
    if (hazard.kind == GirHazardKind::RAW) return classifyMemOp(*hazard.producer);
    if (classifyMemOp(*hazard.producer) == CK_DS) return CK_DS;
    if (classifyMemOp(*hazard.consumer) == CK_DS) return CK_DS;
    return CK_Tensor;
}

size_t indexInBlock(StinkyInstruction& inst);
FeasibleDomainMap computeFeasibleDomains(Function& function,
                                         const GirFrameAnalysis::Result& frames);
bool hazardFeasible(Function& function, const GirFrameAnalysis::Result& frames,
                    const FeasibleDomainMap& feasible, const GirFrameHazard& hazard);

PotentialMap buildPotentials(Function& function, const GirFrameAnalysis::Result& frames,
                             const GirFrameHazardAnalysis::Result& hazards,
                             const std::function<bool(const BasicBlock&)>& covers,
                             const std::function<bool(const StinkyInstruction&)>& alsoFences) {
    PotentialMap result;
    const FeasibleDomainMap feasible = computeFeasibleDomains(function, frames);
    std::set<std::tuple<StinkyInstruction*, GirFrame, CounterKind, StinkyInstruction*, GirFrame,
                        size_t, int, int>>
        seen;
    std::set<const BasicBlock*> modeled;
    for (BasicBlock& block : function) modeled.insert(&block);
    for (const GirFrameHazard& hazard : hazards.hazards) {
        // GIR models exactly the scheduling region, so an endpoint outside this function is not a
        // hazard of it -- it is a tag riding on a fork that the region never modeled.
        if (!modeled.count(hazard.producerBlock) || !modeled.count(hazard.consumerBlock)) continue;
        if (!covers(*hazard.consumerBlock)) continue;
        if (!hazardFeasible(function, frames, feasible, hazard)) continue;
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
                                 << " prod=" << hazard.producerBlock->getLabel() << "#"
                                 << hazard.producerIndex << "/act" << hazard.producerAction
                                 << " cons=" << hazard.consumerBlock->getLabel() << "#"
                                 << hazard.consumerIndex << "/act" << hazard.consumerAction
                                 << " anchor=" << anchor->getParent()->getLabel() << "#"
                                 << indexInBlock(*anchor) << " anchorGap=" << anchorGap << "\n");
            result[{anchor, anchorFrame, counter}].push_back(
                {{hazard.producer, hazard.producerFrame}, hazard.producerIndex, anchorGap});
        };

        if (!hazard.crossAgent) {
            addPotential(hazard.consumer, hazard.consumerFrame, hazard.gap);
            continue;
        }

        const GirHazardFenceCover cover = enclosingFences(frames, hazard, alsoFences);
        if (!cover.hasHazardPath)
            report_fatal_error("Cross-agent GIR hazard has no exact frame-graph path");
        if (cover.hasUnfencedPath)
            report_fatal_error("Cross-agent GIR hazard has an unfenced exact frame-graph path");
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

bool isTripCompare(const StinkyInstruction& inst) {
    const auto opcode = inst.getUnifiedOpcode();
    return opcode == GFX::s_cmp_eq_u32 || opcode == GFX::s_cmp_eq_i32 ||
           opcode == GFX::s_cmp_lg_u32 || opcode == GFX::s_cmp_lg_i32 ||
           opcode == GFX::s_cmp_lt_u32 || opcode == GFX::s_cmp_lt_i32 ||
           opcode == GFX::s_cmp_le_u32 || opcode == GFX::s_cmp_le_i32 ||
           opcode == GFX::s_cmp_gt_u32 || opcode == GFX::s_cmp_gt_i32 ||
           opcode == GFX::s_cmp_ge_u32 || opcode == GFX::s_cmp_ge_i32;
}

const StinkyInstruction* definingTripCompare(const StinkyInstruction& branch) {
    for (StinkyInstruction* source : branch.getSources())
        if (source && isTripCompare(*source)) return source;

    const BasicBlock* block = branch.getParent();
    for (auto it = block->rbegin(); it != block->rend(); ++it) {
        auto* candidate = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (candidate && candidate != &branch && isTripCompare(*candidate)) return candidate;
    }
    return nullptr;
}

std::optional<int> compareLiteral(const StinkyInstruction& compare) {
    for (const StinkyRegister& operand : compare.getSrcRegs()) {
        if (operand.dataType == StinkyRegister::Type::LiteralInt) return operand.getLiteralInt();
        if (operand.dataType == StinkyRegister::Type::LiteralString) {
            const std::string text = operand.getLiteralString();
            char* end = nullptr;
            const long value = std::strtol(text.c_str(), &end, 0);
            if (end == text.c_str() + text.size()) return static_cast<int>(value);
        }
    }
    return std::nullopt;
}

bool isTripControlTarget(const std::string& label) {
    return label.find("skipPGR") != std::string::npos || label.find("toPGR") != std::string::npos ||
           label.find("LoopEndL") != std::string::npos ||
           label.find("NoGlobalLoadLoop") != std::string::npos;
}

std::optional<bool> evaluateTripCompare(const StinkyInstruction& compare, int trip) {
    const std::optional<int> literal = compareLiteral(compare);
    if (!literal) return std::nullopt;
    const auto opcode = compare.getUnifiedOpcode();
    if (opcode == GFX::s_cmp_eq_u32 || opcode == GFX::s_cmp_eq_i32) return trip == *literal;
    if (opcode == GFX::s_cmp_lg_u32 || opcode == GFX::s_cmp_lg_i32) return trip != *literal;
    if (opcode == GFX::s_cmp_lt_u32 || opcode == GFX::s_cmp_lt_i32) return trip < *literal;
    if (opcode == GFX::s_cmp_le_u32 || opcode == GFX::s_cmp_le_i32) return trip <= *literal;
    if (opcode == GFX::s_cmp_gt_u32 || opcode == GFX::s_cmp_gt_i32) return trip > *literal;
    if (opcode == GFX::s_cmp_ge_u32 || opcode == GFX::s_cmp_ge_i32) return trip >= *literal;
    return std::nullopt;
}

std::set<int> edgeTripDomain(BasicBlock* block, BasicBlock* successor,
                             const std::set<int>& domain) {
    IRBase* terminator = block->getTerminator();
    auto* branch = terminator ? dyn_cast<StinkyInstruction>(terminator) : nullptr;
    if (!branch || !isConditionalBranch(*branch)) return domain;
    const std::vector<std::string> targets = getBranchTargets(*branch);
    if (targets.size() != 1 || !isTripControlTarget(targets.front())) return domain;
    const StinkyInstruction* compare = definingTripCompare(*branch);
    if (!compare) return domain;

    const bool taken = successor->getLabel() == targets.front();
    bool branchOnTrue = false;
    if (branch->getUnifiedOpcode() == GFX::s_cbranch_scc1)
        branchOnTrue = true;
    else if (branch->getUnifiedOpcode() != GFX::s_cbranch_scc0)
        return domain;

    std::set<int> allowed;
    for (int trip : domain) {
        const std::optional<bool> comparison = evaluateTripCompare(*compare, trip);
        if (!comparison) return domain;
        const bool takesBranch = *comparison == branchOnTrue;
        if (takesBranch == taken) allowed.insert(trip);
    }
    return allowed;
}

std::set<int> initialTripDomain(Function& function) {
    int limit = 4;
    for (BasicBlock& block : function) {
        IRBase* terminator = block.getTerminator();
        auto* branch = terminator ? dyn_cast<StinkyInstruction>(terminator) : nullptr;
        if (!branch || !isConditionalBranch(*branch)) continue;
        const std::vector<std::string> targets = getBranchTargets(*branch);
        if (targets.size() != 1 || !isTripControlTarget(targets.front())) continue;
        const StinkyInstruction* compare = definingTripCompare(*branch);
        if (!compare) continue;
        const std::optional<int> literal = compareLiteral(*compare);
        if (literal) limit = std::max(limit, *literal + 3);
    }
    std::set<int> result;
    for (int trip = 1; trip <= limit; ++trip) result.insert(trip);
    return result;
}

FeasibleDomainMap computeFeasibleDomains(Function& function,
                                         const GirFrameAnalysis::Result& frames) {
    FeasibleDomainMap feasible;
    std::deque<GirFrameNode> work;
    BasicBlock* entry = function.getEntryBlock();
    if (!entry) return feasible;
    const std::set<int> initial = initialTripDomain(function);
    for (const GirFrame& frame : frames.frames(entry)) {
        GirFrameNode node{entry, frame};
        feasible[node] = initial;
        work.push_back(node);
    }

    while (!work.empty()) {
        GirFrameNode node = std::move(work.front());
        work.pop_front();
        auto successors = frames.edges.find(node);
        if (successors == frames.edges.end()) continue;
        for (const GirFrameNode& successor : successors->second) {
            const std::set<int> allowed =
                edgeTripDomain(node.block, successor.block, feasible[node]);
            if (allowed.empty()) continue;
            std::set<int>& destination = feasible[successor];
            const size_t before = destination.size();
            destination.insert(allowed.begin(), allowed.end());
            if (destination.size() != before) work.push_back(successor);
        }
    }
    return feasible;
}

bool domainsIntersect(const std::set<int>& lhs, const std::set<int>& rhs) {
    auto a = lhs.begin();
    auto b = rhs.begin();
    while (a != lhs.end() && b != rhs.end()) {
        if (*a == *b) return true;
        if (*a < *b)
            ++a;
        else
            ++b;
    }
    return false;
}

bool hazardFeasible(Function& function, const GirFrameAnalysis::Result& frames,
                    const FeasibleDomainMap& feasible, const GirFrameHazard& hazard) {
    (void)function;
    const GirFrameNode source{hazard.producerBlock, hazard.producerFrame};
    const GirFrameNode target{hazard.consumerBlock, hazard.consumerFrame};
    auto sourceDomain = feasible.find(source);
    auto targetDomain = feasible.find(target);
    if (sourceDomain == feasible.end() || targetDomain == feasible.end()) return false;
    if (source == target && hazard.producerIndex < hazard.consumerIndex)
        return domainsIntersect(sourceDomain->second, targetDomain->second);

    std::deque<std::pair<GirFrameNode, std::set<int>>> work;
    auto first = frames.edges.find(source);
    if (first == frames.edges.end()) return false;
    for (const GirFrameNode& successor : first->second) {
        std::set<int> domain = edgeTripDomain(source.block, successor.block, sourceDomain->second);
        if (!domain.empty()) work.push_back({successor, std::move(domain)});
    }

    FeasibleDomainMap seen;
    while (!work.empty()) {
        auto [node, domain] = std::move(work.front());
        work.pop_front();
        std::set<int> fresh;
        std::set<int>& prior = seen[node];
        std::set_difference(domain.begin(), domain.end(), prior.begin(), prior.end(),
                            std::inserter(fresh, fresh.end()));
        if (fresh.empty()) continue;
        prior.insert(fresh.begin(), fresh.end());
        if (node == target && domainsIntersect(fresh, targetDomain->second)) return true;
        auto successors = frames.edges.find(node);
        if (successors == frames.edges.end()) continue;
        for (const GirFrameNode& successor : successors->second) {
            std::set<int> next = edgeTripDomain(node.block, successor.block, fresh);
            if (!next.empty()) work.push_back({successor, std::move(next)});
        }
    }
    return false;
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
                    std::map<BasicBlock*, int, std::less<BasicBlock*>>* perPred,
                    const FeasibleDomainMap* feasible = nullptr) {
    struct Step {
        GirFrameNode node;
        size_t index = 0;
        int steps = 0;
        int count = 0;
        BasicBlock* pred = nullptr;
        // Trip counts this path is still possible for. A short prefetch arm reaches the steady
        // loop for no trip count, so the rank along it is not a constraint on the steady wait.
        std::set<int> trips;
    };

    const int span = std::max(0, gap);
    int best = -1;
    bool unaccounted = false;
    std::set<int> producerTrips;
    if (feasible) {
        auto found = feasible->find(producerNode);
        if (found != feasible->end()) producerTrips = found->second;
    }
    std::deque<Step> work{{producerNode, producerIndex + 1, 0, 1, nullptr, producerTrips}};
    // The frame graph is a graph, so the same state is reachable many ways and an unmemoised walk
    // re-expands it exponentially.  Truncating that with a budget is what made the answer depend
    // on exploration order; deduplicating the state makes the walk finite AND complete, because a
    // state already enqueued can contribute nothing a second time.
    std::unordered_set<size_t> visited;
    const auto stateKey = [](const Step& step) {
        size_t key = GirFrameNodeHash{}(step.node);
        for (const size_t part :
             {step.index, static_cast<size_t>(step.steps), static_cast<size_t>(step.count),
              std::hash<BasicBlock*>{}(step.pred)})
            key = key * 1099511628211ULL ^ part;
        key = key * 1099511628211ULL ^ step.trips.size();
        for (int trip : step.trips) key = key * 1099511628211ULL ^ static_cast<size_t>(trip);
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
                if (best < 0 || step.count < best) best = step.count;
                if (perPred) {
                    auto [entry, inserted] = perPred->emplace(step.pred, step.count);
                    if (!inserted) entry->second = std::min(entry->second, step.count);
                }
                retired = true;  // this path is answered; do not walk past its own anchor
                break;
            }

            if (retireAt(*inst, step.count)) {
                retired = true;
                break;
            }

            if (classifyMemOp(*inst) == counter) ++step.count;
        }
        if (retired) continue;

        if (retireAtEnd(step.node.block, step.count)) continue;
        // Ran out of span, or off the end of the frame graph, without ever reaching the anchor:
        // this path was not modelled, so it proves nothing about the producer being retired.
        if (step.steps >= span) {
            unaccounted = true;
            continue;
        }

        auto successors = frames.edges.find(step.node);
        if (successors == frames.edges.end()) {
            unaccounted = true;
            continue;
        }
        for (const GirFrameNode& successor : successors->second) {
            std::set<int> trips = step.trips;
            if (feasible && !trips.empty()) {
                trips = edgeTripDomain(step.node.block, successor.block, trips);
                if (trips.empty()) continue;  // no trip count takes this edge onward
            }
            Step next{successor, 0, step.steps + 1, step.count, step.node.block, std::move(trips)};
            if (visited.insert(stateKey(next)).second) work.push_back(std::move(next));
        }
    }
    return {best, unaccounted};
}

SpanResult countAcrossSpan(const GirFrameAnalysis::Result& frames, const DecisionMap& decisions,
                           const TailDecisionMap& tailDecisions, CounterKind counter,
                           const GirFrameNode& producerNode, const Potential& potential,
                           StinkyInstruction* anchor, const GirFrameNode& anchorNode,
                           size_t anchorIndex,
                           std::map<BasicBlock*, int, std::less<BasicBlock*>>* perPred,
                           const FeasibleDomainMap* feasible) {
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
                    anchor, anchorNode, anchorIndex, retireAt, retireAtEnd, perPred, feasible);
}

RequirementMap simulate(Function& function, const GirFrameAnalysis::Result& frames,
                        const PotentialMap& potentials, const DecisionMap& decisions,
                        const TailDecisionMap& tailDecisions) {
    const FeasibleDomainMap feasible = computeFeasibleDomains(function, frames);
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
                        site.anchor, anchorNode, anchorIndex, &perPred, &feasible);
                    // Only a walk that reached the anchor on NO path and ended some path by
                    // exhaustion is a modelling failure.  In a branching graph most paths simply
                    // lead elsewhere, so counting every one of those measures the graph, not a bug.
                    if (span.count < 0 && span.unaccounted) ++girUnaccountedSpans;
                    if (span.count <= 0) continue;
                    for (const auto& [pred, arrived] : perPred)
                        requirements[{site.anchor, site.counter}].record(
                            pred, waitToDrain(site.counter, arrived), potential.producer);
                }
            }
        }
    }
    return requirements;
}

int& counterField(WaitCountSpec& spec, CounterKind counter) {
    if (counter == CK_DS) return spec.dsCount;
    if (counter == CK_Tensor) return spec.tensorCount;
    report_fatal_error("GIR frame wait plan asked for an unsupported counter field");
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

struct PromotionPlan {
    DecisionMap anchors;
    TailDecisionMap tails;
    TailProducerMap tailProducers;
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
        }
        if (!valid || local.tails.empty()) continue;
        result.anchors[site] = common;
        for (const auto& [edge, wait] : local.tails) {
            auto [tail, inserted] = result.tails.emplace(edge, wait);
            if (!inserted) tail->second = std::min(tail->second, wait);
        }
        for (const auto& [edge, producers] : local.tailProducers)
            result.tailProducers[edge].insert(producers.begin(), producers.end());
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
    report_fatal_error("GIR finite-frame counter flow did not close at its frame-node bound");
}

WaitInsertionPlan materializePlan(const DecisionMap& decisions,
                                  const TailDecisionMap& tailDecisions,
                                  const TailProducerMap& tailProducers,
                                  const RequirementMap& requirements) {
    WaitInsertionPlan plan;
    for (const auto& [site, wait] : decisions) {
        if (wait == WaitCountSpec::kUnused) continue;
        WaitCountSpec& spec = plan.anchorWaits[site.anchor];
        tightenField(spec, site.counter, wait);
        auto summary = requirements.find(site);
        if (site.counter == CK_Tensor && summary != requirements.end())
            attachTensorTokens(spec, summary->second);
    }

    std::map<BasicBlock*, WaitCountSpec, std::less<BasicBlock*>> tailSpecs;
    for (const auto& [site, wait] : tailDecisions) {
        WaitCountSpec& spec = tailSpecs[site.block];
        tightenField(spec, site.counter, wait);
        auto producers = tailProducers.find(site);
        if (site.counter == CK_Tensor && producers != tailProducers.end())
            attachTensorTokens(spec, producers->second);
    }
    for (auto& [block, spec] : tailSpecs) plan.tailDrains.push_back({block, std::move(spec)});
    return plan;
}

/// Prove every planned wait can change its counter. A weaker/equal wait after the queue is already
/// bounded (accounting for every CFG path and intervening issue) is a counter-model defect, not an
/// optimization opportunity: fail generation instead of hiding it in emitted assembly.
void validateNoRedundantWaits(Function& function, WaitInsertionPlan& plan) {
    using Bounds = std::array<int, CK_Count>;
    const std::array<CounterKind, 2> counters{CK_DS, CK_Tensor};
    constexpr int kBeyondEncodableWait = 64;
    const auto unknownBounds = [] {
        Bounds result;
        result.fill(-1);
        return result;
    };

    const auto scanBlock = [&](BasicBlock& block, Bounds upperBound, bool validate) {
        const auto applyPlanned = [&](WaitCountSpec& spec, CounterKind counter,
                                      const StinkyInstruction* anchor) {
            int& wait = counterField(spec, counter);
            if (wait == WaitCountSpec::kUnused) return;
            int& bound = upperBound[counter];
            if (bound >= 0 && wait >= bound) {
                if (validate)
                    report_fatal_error(
                        std::string("GIR counter flow produced redundant ") +
                        (counter == CK_DS ? "dscnt" : "tensorcnt") +
                        " wait=" + std::to_string(wait) + " after bound=" + std::to_string(bound) +
                        " in " + (anchor ? anchor->getParent()->getLabel() : block.getLabel()));
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
            if (observedWaitDrains(*inst, observed))
                for (CounterKind counter : counters)
                    if (observed[counter] >= 0 &&
                        (upperBound[counter] < 0 || observed[counter] < upperBound[counter]))
                        upperBound[counter] = observed[counter];

            const CounterKind issue = classifyMemOp(*inst);
            if ((issue == CK_DS || issue == CK_Tensor) && upperBound[issue] >= 0)
                upperBound[issue] = std::min(kBeyondEncodableWait, upperBound[issue] + 1);
        }

        for (TailDrain& drain : plan.tailDrains) {
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
    if (result.count < 0 && result.unaccounted) ++girUnaccountedSpans;
    return result.count;
}

int girFrameDistance(const GirFrameAnalysis::Result& frames, const GirFrameNode& from,
                     const GirFrameNode& to) {
    if (from == to) return 0;
    std::unordered_set<GirFrameNode, GirFrameNodeHash> seen{from};
    std::deque<std::pair<GirFrameNode, int>> work{{from, 0}};
    while (!work.empty()) {
        auto [node, steps] = std::move(work.front());
        work.pop_front();
        auto successors = frames.edges.find(node);
        if (successors == frames.edges.end()) continue;
        for (const GirFrameNode& successor : successors->second) {
            if (successor == to) return steps + 1;
            if (!seen.insert(successor).second) continue;
            work.push_back({successor, steps + 1});
        }
    }
    return -1;
}

WaitInsertionPlan buildGirFrameWaitPlan(
    Function& function, const GirFrameAnalysis::Result& frames,
    const GirFrameHazardAnalysis::Result& hazards,
    const std::function<bool(const BasicBlock&)>& covers,
    const std::function<bool(const StinkyInstruction&)>& alsoFences) {
    const PotentialMap potentials = buildPotentials(function, frames, hazards, covers, alsoFences);
    if (potentials.empty()) return {};

    DecisionMap decisions;
    for (const auto& [dynamic, _] : potentials)
        decisions[{dynamic.anchor, dynamic.counter}] = WaitCountSpec::kUnused;

    TailDecisionMap tailDecisions;
    TailProducerMap tailProducers;
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
            report_fatal_error("GIR finite-frame counter flow produced an unsafe wait decision");
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
    if (girUnaccountedSpans && std::getenv("ST_GIR_SPAN_STATS"))
        std::cerr << "[gir-span] " << function.getName() << ": " << girUnaccountedSpans
                  << " unaccounted span walk(s)\n";
    WaitInsertionPlan plan = materializePlan(decisions, tailDecisions, tailProducers, requirements);
    validateNoRedundantWaits(function, plan);
    return plan;
}

}  // namespace stinkytofu::waitcnt
