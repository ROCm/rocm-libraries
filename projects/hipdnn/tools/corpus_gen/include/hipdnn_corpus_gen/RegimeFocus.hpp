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
/// Regimes defined by an equality (`decode` is `seqlen_q == 1`) are almost never hit by a free
/// walk, so a focus pins/ties those parameters and walks the rest. Every candidate is still
/// checked with `regimeLabel`, so a focus never puts a point in the wrong regime.
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

/// Which declared label each axis would take for the joined @p label, or nullopt when none
/// spells it. Tried rather than split on `_`, because a declared label may contain one.
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
        // The left operand is the dependent. A tie that would close a cycle or re-tie a fixed
        // parameter is dropped; the label check still enforces it.
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
/// An impossible label is refused rather than searched, which would report it as saturated.
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
        // Only a label named by exactly one clause pins anything: several clauses are a
        // disjunction, and `otherwise` implies only negations.
        if(clauses.size() == 1 && clauses.front() < declared.whens.size())
        {
            detail::foldEqualities(metadata, declared.whens[clauses.front()], focus);
        }
    }
    return focus;
}

/// @brief What a focused search delivered for one regime.
struct RegimeSearchResult
{
    std::vector<ProblemPoint> problems;

    /// Every combination searched stopped because growing found no new point in the regime.
    bool saturated = false;

    /// A combination stopped at the budget limit while still finding points. More exist.
    bool searchCapped = false;

    /// Candidates proposed and labelled, and how many landed in the regime. A low ratio means
    /// the focus compiled too little of the declaration.
    int64_t proposed = 0;
    int64_t inRegime = 0;
};

/// @brief Up to @p wanted problems in @p focus's regime that @p admits accepts, none of them
/// @p held.
///
/// Walks the parameters the focus leaves free, seeded by @p anchors (served points in any
/// regime) projected onto them. Grows like `exploreProblemSpace`, stopping saturated or at the
/// growth limit; the two are reported apart.
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
    // Only combinations some anchor shows are served; a declined dtype would burn a whole
    // budget again. With no anchors, every combination is tried.
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

    // One lane per combination, kept across rounds. The memo lets a longer walk that retraces
    // a shorter one reuse answers instead of querying the engine again.
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
        // Offset from the first pass's seeds so the focus does not retrace the walk that missed
        // this regime; still deterministic in `seed`.
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
        // Held points come back from the search and are set aside, so the target includes them.
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

    // Split what is still wanted over the lanes not yet spent, so a combination that cannot
    // serve the regime hands its share to those that can.
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
            // Run at the current budget first (a larger target re-tessellates existing
            // observations), then double while new points keep appearing.
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
