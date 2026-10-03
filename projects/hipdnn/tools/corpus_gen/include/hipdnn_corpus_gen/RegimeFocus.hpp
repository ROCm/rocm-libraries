// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_corpus_gen/FeasibleShapeSet.hpp>
#include <hipdnn_corpus_gen/OperationMetadata.hpp>
#include <hipdnn_corpus_gen/ProblemSpace.hpp>
#include <hipdnn_corpus_gen/RegimeLabel.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

/// @file RegimeFocus.hpp
/// @brief Generating problems in one named regime, for a corpus asked to fill a quota there.
///
/// The exploration in ProblemSpace.hpp finds what an engine serves, and a regime is whatever
/// population those points happen to land in. That leaves the regimes defined by an *equality*
/// almost empty: `decode` is `seqlen_q == 1`, `prefill` is `seqlen_q == seqlen_k`, `mha` is
/// `heads_kv == heads`, and a walk over independent extents meets an equality with probability
/// close to zero. A model trained on that corpus has seen a handful of multi-head decode shapes,
/// and its errors there are the worst it has. Sampling more of the same does not fix it; asking
/// for them does.
///
/// A focus is a regime label read back into the relations that produce it. An `==` clause
/// against a constant is a pin, and against another parameter is a tie; the search then walks
/// only the parameters left free, so every point it proposes already satisfies the equalities.
/// Everything else a label requires -- an inequality, or the negation an `otherwise` label
/// implies -- is not compiled at all: every candidate is labelled by the same `regimeLabel` the
/// manifest uses and kept only if it matches, so a focus can make a regime reachable but can
/// never put a point in the wrong one.
namespace hipdnn_corpus_gen
{

/// @brief The relations a regime label pins, and the label itself as the final word.
struct RegimeFocus
{
    std::string label;

    /// Numeric parameters a clause fixes to a constant (`seqlen_q == 1`).
    std::map<std::string, int64_t> pins;

    /// Categorical parameters a clause fixes to a value (`is_causal == true`). A combination
    /// holding another value cannot be in the regime, so it is not searched.
    std::map<std::string, ParameterValue> categoricalPins;

    /// Numeric parameters a clause sets equal to another (`heads_kv == heads`): dependent to
    /// source. The dependent is not searched; it is copied from its source.
    std::map<std::string, std::string> ties;
};

namespace detail
{

/// Which declared label each axis would have to take for the joined @p label, or nullopt when
/// no assignment of the axes' labels spells it. Tried rather than split on `_`, because a
/// declared label may itself contain one.
inline std::optional<std::vector<std::string>>
    splitRegimeLabel(const std::vector<RegimeAxis>& axes, const std::string& label, size_t axis = 0)
{
    if(axis == axes.size())
    {
        return label.empty() ? std::optional<std::vector<std::string>>(std::vector<std::string>{})
                             : std::nullopt;
    }
    std::vector<std::string> options = axes[axis].labels;
    options.push_back(axes[axis].otherwise);
    for(const auto& option : options)
    {
        if(option.empty() || label.rfind(option, 0) != 0)
        {
            continue;
        }
        auto rest = label.substr(option.size());
        if(axis + 1 < axes.size())
        {
            if(rest.empty() || rest.front() != '_')
            {
                continue;
            }
            rest.erase(0, 1);
        }
        if(auto tail = splitRegimeLabel(axes, rest, axis + 1))
        {
            tail->insert(tail->begin(), option);
            return tail;
        }
    }
    return std::nullopt;
}

/// A declared constant as a parameter value, or nullopt for anything that is not one.
inline std::optional<ParameterValue> constantValue(const nlohmann::json& operand)
{
    if(operand.is_boolean())
    {
        return ParameterValue(operand.get<bool>());
    }
    if(operand.is_number_integer())
    {
        return ParameterValue(operand.get<int64_t>());
    }
    if(operand.is_number_float())
    {
        return ParameterValue(operand.get<double>());
    }
    if(operand.is_string() && queryReference(operand.get<std::string>()).empty())
    {
        return ParameterValue(operand.get<std::string>());
    }
    return std::nullopt;
}

inline bool isNumeric(const OperationMetadata& metadata, const std::string& name)
{
    const auto* parameter = metadata.find(name);
    return parameter != nullptr
           && (parameter->type == ParameterType::INT64
               || parameter->type == ParameterType::FLOAT64);
}

/// The equalities @p when asserts, folded into @p focus. Only `==` and a conjunction of them
/// are read; any other relation is left to the label check, which is exact where this is not.
inline void foldEqualities(const OperationMetadata& metadata,
                           const nlohmann::json& when,
                           RegimeFocus& focus)
{
    if(!when.is_object() || when.size() != 1)
    {
        return;
    }
    const auto entry = when.begin();
    const auto& op = entry.key();
    const auto& operands = entry.value();
    if(op == "and" && operands.is_array())
    {
        for(const auto& inner : operands)
        {
            foldEqualities(metadata, inner, focus);
        }
        return;
    }
    if(op != "==" || !operands.is_array() || operands.size() != 2)
    {
        return;
    }
    const auto reference = [](const nlohmann::json& operand) {
        return operand.is_string() ? queryReference(operand.get<std::string>()) : std::string();
    };
    auto left = reference(operands[0]);
    auto right = reference(operands[1]);
    if(left.empty() && right.empty())
    {
        return;
    }
    if(!left.empty() && !right.empty())
    {
        if(!isNumeric(metadata, left) || !isNumeric(metadata, right) || left == right)
        {
            return;
        }
        // The dependent is the left operand, as written: `heads_kv == heads` reads as "heads_kv
        // takes heads' value". A tie that would close a cycle, or re-tie something already fixed,
        // is dropped; the label check still holds the point to both.
        std::string source = right;
        for(auto hop = focus.ties.find(source); hop != focus.ties.end();
            hop = focus.ties.find(source))
        {
            source = hop->second;
        }
        if(source == left || focus.ties.count(left) > 0 || focus.pins.count(left) > 0)
        {
            return;
        }
        focus.ties[left] = right;
        return;
    }
    const auto& name = left.empty() ? right : left;
    const auto value = constantValue(left.empty() ? operands[0] : operands[1]);
    if(!value.has_value() || metadata.find(name) == nullptr)
    {
        return;
    }
    if(isNumeric(metadata, name))
    {
        if(const auto* integer = std::get_if<int64_t>(&*value))
        {
            focus.ties.erase(name);
            focus.pins[name] = *integer;
        }
        return;
    }
    focus.categoricalPins[name] = *value;
}

} // namespace detail

/// @brief The focus for @p label under @p metadata's declared facets, or nullopt with @p error
/// set when no assignment of the declared labels spells it.
///
/// A label nothing can produce is refused here rather than searched for: the search would spend
/// its whole budget and report the regime as saturated, which reads as "the engine serves none
/// of these" when the truth is "this regime does not exist".
inline std::optional<RegimeFocus> compileRegimeFocus(const OperationMetadata& metadata,
                                                     const std::string& label,
                                                     std::string& error)
{
    const auto chosen = detail::splitRegimeLabel(metadata.regimeLabel, label);
    if(!chosen.has_value())
    {
        error = "'" + label + "' is not a regime " + metadata.operation
                + " declares: no assignment of its regime_label facets spells it";
        return std::nullopt;
    }
    RegimeFocus focus;
    focus.label = label;
    for(size_t axis = 0; axis < chosen->size(); ++axis)
    {
        const auto& declared = metadata.regimeLabel[axis];
        std::vector<size_t> clauses;
        for(size_t i = 0; i < declared.labels.size(); ++i)
        {
            if(declared.labels[i] == (*chosen)[axis])
            {
                clauses.push_back(i);
            }
        }
        // One clause names the label: its equalities hold for every point carrying it. Several
        // clauses naming one label is a disjunction -- none of them is required -- and
        // `otherwise` requires only negations; neither pins anything.
        if(clauses.size() == 1 && clauses.front() < declared.whens.size())
        {
            detail::foldEqualities(metadata, declared.whens[clauses.front()], focus);
        }
    }
    return focus;
}

/// @brief Every label @p metadata's regime facets can spell: each facet's declared labels and its
/// `otherwise`, crossed in declaration order and joined as `regimeLabel` joins them.
///
/// A label here may still name nothing the engine serves, or nothing a declared constraint
/// admits -- cross attention, under a constraint that rules it out. That is for a focused search
/// to find, and to report as a shortfall, not for this list to guess.
inline std::vector<std::string> declaredRegimeLabels(const OperationMetadata& metadata)
{
    std::vector<std::string> joined;
    if(metadata.regimeLabel.empty())
    {
        return joined;
    }
    joined.emplace_back();
    for(const auto& axis : metadata.regimeLabel)
    {
        std::vector<std::string> options;
        for(const auto& label : axis.labels)
        {
            if(std::find(options.begin(), options.end(), label) == options.end())
            {
                options.push_back(label);
            }
        }
        if(!axis.otherwise.empty()
           && std::find(options.begin(), options.end(), axis.otherwise) == options.end())
        {
            options.push_back(axis.otherwise);
        }
        std::vector<std::string> next;
        for(const auto& prefix : joined)
        {
            for(const auto& option : options)
            {
                next.push_back(prefix.empty() ? option : prefix + "_" + option);
            }
        }
        joined = std::move(next);
    }
    return joined;
}

/// @brief A floor of @p floor problems for every regime @p metadata spells that @p quotas does not
/// already name, as a focus per regime (`--regime-floor`).
///
/// A combination of facet labels no single assignment spells is left out rather than refused:
/// the list is the cross product, and a floor is asked of what exists.
inline std::map<std::string, RegimeFocus> regimeFloors(const OperationMetadata& metadata,
                                                       const std::map<std::string, int64_t>& quotas)
{
    std::map<std::string, RegimeFocus> floors;
    for(const auto& regime : declaredRegimeLabels(metadata))
    {
        if(quotas.count(regime) > 0)
        {
            continue;
        }
        std::string error;
        auto focus = compileRegimeFocus(metadata, regime, error);
        if(focus.has_value())
        {
            floors.emplace(regime, std::move(*focus));
        }
    }
    return floors;
}

/// @brief What a focused search delivered for one regime.
struct RegimeSearchResult
{
    std::vector<ProblemPoint> problems;

    /// Every combination searched stopped because growing found no new point in the regime.
    bool saturated = false;

    /// A combination stopped at the budget limit while still finding points. More exist.
    bool searchCapped = false;

    /// Candidates proposed and labelled, and how many of those were in the regime. A focus
    /// whose candidates almost never land in it is a declaration the compiler read too little
    /// of, and this is where that shows.
    int64_t proposed = 0;
    int64_t inRegime = 0;
};

/// @brief Up to @p wanted problems in @p focus's regime that @p admits accepts, none of them
/// @p held.
///
/// The walk is `buildFeasibleShapeSet` over the parameters the focus leaves free, each proposal
/// completed with the pins and ties and then held to the declared constraints, to the exact
/// label, and to the engine. @p anchors -- points already known to be served, in any regime --
/// seed the walk: projected onto the free parameters, a served point is usually still served
/// once its dependents are set, and a walk that starts inside the region finds far more than one
/// that has to hunt for it.
///
/// Grown the way `exploreProblemSpace` grows a combination toward a corpus target: at its budget
/// first, then doubled while it keeps finding new points, and stopped either saturated or at the
/// growth limit. The two stops mean different things and are reported apart.
inline RegimeSearchResult exploreRegime(const OperationMetadata& metadata,
                                        const RegimeFocus& focus,
                                        const ExplorationRequest& request,
                                        int64_t wanted,
                                        const ProblemOracle& admits,
                                        const ProblemOracle& held,
                                        const std::vector<ProblemPoint>& anchors)
{
    RegimeSearchResult outcome;
    if(wanted <= 0)
    {
        return outcome;
    }

    const auto window = detail::numericWindow(metadata, request);
    std::vector<ShapeDimension> free;
    for(const auto& dimension : window)
    {
        if(focus.pins.count(dimension.name) == 0 && focus.ties.count(dimension.name) == 0)
        {
            free.push_back(dimension);
        }
    }

    size_t totalCombinations = 0;
    std::vector<ProblemPoint> combinations;
    for(auto& combination : detail::categoricalCombinations(
            metadata.parameters, request.maxCombinations, totalCombinations))
    {
        bool matches = true;
        for(const auto& [name, value] : focus.categoricalPins)
        {
            const auto found = combination.find(name);
            matches = matches && found != combination.end() && found->second == value;
        }
        if(matches)
        {
            combinations.push_back(std::move(combination));
        }
    }
    // Only combinations something is known to serve: a dtype the engine declines costs a whole
    // budget to prove again, and the first pass already proved it. With no anchors at all there
    // is nothing to go on, so every combination is tried.
    std::vector<ProblemPoint> served;
    for(const auto& combination : combinations)
    {
        const bool anchored = std::any_of(anchors.begin(), anchors.end(), [&](const auto& point) {
            return std::all_of(combination.begin(), combination.end(), [&](const auto& value) {
                const auto found = point.find(value.first);
                return found != point.end() && found->second == value.second;
            });
        });
        if(anchored)
        {
            served.push_back(combination);
        }
    }
    if(served.empty())
    {
        served = combinations;
    }
    if(served.empty())
    {
        return outcome;
    }

    const auto complete = [&](const ProblemPoint& categorical, const Shape& shape) {
        auto point = categorical;
        for(size_t i = 0; i < free.size(); ++i)
        {
            point[free[i].name] = shape[i];
        }
        for(const auto& [name, value] : focus.pins)
        {
            point[name] = value;
        }
        // Resolved in as many passes as there are ties, so a chain (a := b := c) settles
        // whatever order the map iterates in.
        for(size_t pass = 0; pass < focus.ties.size(); ++pass)
        {
            for(const auto& [dependent, source] : focus.ties)
            {
                const auto found = point.find(source);
                if(found != point.end())
                {
                    point[dependent] = found->second;
                }
            }
        }
        return point;
    };
    const auto project = [&](const ProblemPoint& point) {
        Shape shape;
        shape.reserve(free.size());
        for(const auto& dimension : free)
        {
            const auto value = detail::integerAt(point, dimension.name).value_or(dimension.low);
            shape.push_back(std::clamp(value, dimension.low, dimension.high));
        }
        return shape;
    };

    std::set<std::string> taken;
    if(free.empty())
    {
        // Everything numeric is pinned or tied: one candidate per combination.
        for(const auto& categorical : served)
        {
            const auto point = complete(categorical, Shape{});
            ++outcome.proposed;
            if(regimeLabel(metadata, point) == focus.label)
            {
                ++outcome.inRegime;
                if(detail::satisfiesConstraints(metadata, point) && admits(point)
                   && !(held && held(point)) && taken.insert(detail::describe(point)).second)
                {
                    outcome.problems.push_back(point);
                }
            }
        }
        outcome.saturated = static_cast<int64_t>(outcome.problems.size()) < wanted;
        return outcome;
    }

    // One lane per combination, kept across rounds: its search request (whose budget only
    // grows), its memo of oracle answers -- so a longer walk retracing a shorter one costs
    // lookups, not engine queries -- and what it has found.
    struct Lane
    {
        ProblemPoint categorical;
        FeasibleSetRequest search;
        std::map<Shape, bool> memo;
        std::vector<ProblemPoint> fresh;
        int64_t held = 0;
        int64_t reached = 0;
        bool saturated = false;
        bool capped = false;
    };
    std::vector<Lane> lanes(served.size());

    std::vector<std::string> names;
    names.reserve(window.size());
    for(const auto& dimension : window)
    {
        names.push_back(dimension.name);
    }
    size_t skeletonTotal = 0;
    const auto skeleton
        = detail::regimeSkeleton(metadata, window, names, request.maxSkeleton, skeletonTotal);

    for(size_t index = 0; index < lanes.size(); ++index)
    {
        auto& lane = lanes[index];
        lane.categorical = served[index];
        lane.search.dimensions = free;
        lane.search.oracleBudget = request.budgetPerCombination;
        lane.search.restarts = request.restarts;
        lane.search.stepsPerStart = request.stepsPerStart;
        // Offset from the first pass's seeds, so a focus does not retrace the walk that missed
        // this regime; still fixed by `seed`, so the corpus reproduces.
        lane.search.seed = request.seed + 0x5eed + index;
        for(const auto& anchor : anchors)
        {
            const bool here = std::all_of(
                lane.categorical.begin(), lane.categorical.end(), [&](const auto& value) {
                    const auto found = anchor.find(value.first);
                    return found != anchor.end() && found->second == value.second;
                });
            if(here)
            {
                lane.search.seeds.push_back(project(anchor));
            }
        }
        for(const auto& shape : skeleton)
        {
            ProblemPoint point = lane.categorical;
            for(size_t i = 0; i < names.size(); ++i)
            {
                point[names[i]] = shape[i];
            }
            lane.search.seeds.push_back(project(point));
        }
    }

    const auto run = [&](Lane& lane) {
        const ShapeOracle oracle = [&](const Shape& shape) {
            const auto known = lane.memo.find(shape);
            if(known != lane.memo.end())
            {
                return known->second;
            }
            const auto point = complete(lane.categorical, shape);
            ++outcome.proposed;
            bool verdict = false;
            if(regimeLabel(metadata, point) == focus.label)
            {
                ++outcome.inRegime;
                verdict = detail::satisfiesConstraints(metadata, point) && admits(point);
            }
            lane.memo.emplace(shape, verdict);
            return verdict;
        };
        // Held points are returned by the search and then set aside, so they are budgeted for:
        // a target they could meet on their own would add nothing.
        lane.search.targetCount += lane.held;
        const auto found = buildFeasibleShapeSet(oracle, lane.search);
        lane.search.targetCount -= lane.held;
        lane.fresh.clear();
        lane.held = 0;
        for(const auto& shape : found.shapes)
        {
            auto point = complete(lane.categorical, shape);
            if(held && held(point))
            {
                ++lane.held;
                continue;
            }
            lane.fresh.push_back(std::move(point));
        }
        return found.stats.distinct;
    };

    const auto total = [&lanes]() {
        int64_t sum = 0;
        for(const auto& lane : lanes)
        {
            sum += static_cast<int64_t>(lane.fresh.size());
        }
        return sum;
    };

    // Rounds, as `exploreProblemSpace` grows toward a corpus target: what is still wanted is
    // split over the lanes not yet spent, so a combination that cannot serve the regime at all
    // -- an alignment a declared constraint rules out -- hands its share to those that can.
    while(true)
    {
        const auto need = wanted - total();
        std::vector<Lane*> open;
        for(auto& lane : lanes)
        {
            if(!lane.saturated && !lane.capped)
            {
                open.push_back(&lane);
            }
        }
        if(need <= 0 || open.empty())
        {
            break;
        }
        const auto share
            = (need + static_cast<int64_t>(open.size()) - 1) / static_cast<int64_t>(open.size());
        for(auto* lane : open)
        {
            const auto target = static_cast<int64_t>(lane->fresh.size()) + share;
            lane->search.targetCount = target;
            // At the budget it already has first: a larger target alone re-tessellates what the
            // walk observed. Then doubled while it keeps finding new points.
            lane->reached = run(*lane);
            while(static_cast<int64_t>(lane->fresh.size()) < target)
            {
                if(lane->search.oracleBudget
                   >= request.budgetPerCombination * request.budgetGrowthLimit)
                {
                    lane->capped = true;
                    break;
                }
                const auto before = lane->reached;
                lane->search.oracleBudget *= 2;
                lane->search.stepsPerStart *= 2;
                lane->reached = run(*lane);
                if(lane->reached <= before)
                {
                    lane->saturated = true;
                    break;
                }
            }
        }
    }

    for(auto& lane : lanes)
    {
        outcome.searchCapped = outcome.searchCapped || lane.capped;
        for(auto& point : lane.fresh)
        {
            if(taken.insert(detail::describe(point)).second)
            {
                outcome.problems.push_back(std::move(point));
            }
        }
    }
    outcome.saturated
        = static_cast<int64_t>(outcome.problems.size()) < wanted && !outcome.searchCapped;
    return outcome;
}

} // namespace hipdnn_corpus_gen
