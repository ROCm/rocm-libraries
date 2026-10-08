// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_corpus_gen/ArgumentResolver.hpp>
#include <hipdnn_corpus_gen/FeasibleShapeSet.hpp>
#include <hipdnn_corpus_gen/OperationMetadata.hpp>
#include <hipdnn_corpus_gen/WorkloadSampling.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

/// @file ProblemSpace.hpp
/// @brief The single, engine-independent exploration of a problem space (RFC 0019.13 §4, §5).
///
/// Categorical axes (`enum`, `bool`) are enumerated from the declaration; numeric axes have no
/// declared bounds (§4.3.2) and are searched. Engines only filter the resulting points.
namespace hipdnn_corpus_gen
{

/// Whether a problem point is admissible; in production, whether the engine accepts it.
using ProblemOracle = std::function<bool(const ProblemPoint&)>;

/// A point every oracle admits, i.e. a corpus on which several engines can be compared.
/// Short-circuits at the first refusal.
inline ProblemOracle allOf(std::vector<ProblemOracle> oracles)
{
    return [oracles = std::move(oracles)](const ProblemPoint& point) {
        for(const auto& admits : oracles)
        {
            if(!admits(point))
            {
                return false;
            }
        }
        return true;
    };
}

/// Search limits, identical in meaning for every operation.
struct ExplorationRequest
{
    /// Problem points wanted per categorical combination, so one dtype cannot take the whole
    /// budget.
    int64_t pointsPerCombination = 50;

    /// Oracle calls per categorical combination.
    int64_t budgetPerCombination = 20000;

    /// Ceiling on every numeric parameter: the largest extent worth benchmarking (§4.3.2).
    int64_t numericCeiling = 4096;

    /// Independent footholds per combination (see FeasibleShapeSet).
    int64_t restarts = 8;

    /// Hit-and-run steps per walk. Together with @ref restarts this bounds how many distinct
    /// problems the search can produce.
    int64_t stepsPerStart = 400;

    /// Reproducibility seed (§5.8).
    uint64_t seed = 0;

    /// Largest number of categorical combinations to explore; reported when it binds.
    size_t maxCombinations = 1024;

    /// Oracle calls first spent probing a combination with no admitted archetype. If the probe
    /// finds nothing, the combination is declined without spending @ref budgetPerCombination.
    int64_t probeBudget = 2000;

    /// Largest enumerated regime skeleton to try before sampling; each point costs an oracle
    /// call.
    size_t maxSkeleton = 512;

    /// Problems wanted from the whole operation; 0 means just the per-combination pass. Short of
    /// it, combinations that found something are searched again with growing budgets;
    /// combinations that found nothing are not grown.
    int64_t corpusTarget = 0;

    /// Maximum oracle budget, as a multiple of @ref budgetPerCombination, a combination may grow
    /// to. Hitting it is a search limit, reported as such.
    int64_t budgetGrowthLimit = 64;
};

/// One categorical assignment and what the numeric search found under it.
struct CombinationResult
{
    /// How many of this combination's problems came from each source.
    int64_t fromArchetypes = 0;
    int64_t fromNeighbourhood = 0;
    int64_t fromExploration = 0;

    ProblemPoint categorical;
    std::vector<ProblemPoint> problems;
    FeasibleSetStats stats;

    /// Growth toward @ref ExplorationRequest::corpusTarget stopped because doubling the search
    /// found no new point.
    bool saturated = false;

    /// Growth stopped at @ref ExplorationRequest::budgetGrowthLimit while still finding new
    /// points; more exist.
    bool searchCapped = false;

    /// Per numeric parameter (in @ref ProblemCorpus::numericParameters order), the served point
    /// with its smallest and largest value over every point the engine accepted here, selected
    /// or not. Empty when nothing was served. Ties go to the lexicographically first shape, so a
    /// seed reproduces them.
    std::vector<ProblemPoint> lowest;
    std::vector<ProblemPoint> highest;
};

/// The problem corpus and an account of how it was produced.
struct ProblemCorpus
{
    std::string operation;

    /// Parameter names in the numeric search's index order, mapping a shape to a point.
    std::vector<std::string> numericParameters;

    std::vector<CombinationResult> combinations;

    /// Combinations and skeleton points not explored because a cap bound, with explanations.
    std::vector<std::string> skippedCombinations;

    /// Candidates a declared constraint refused and admitted. An over-strong constraint empties
    /// the corpus, which would otherwise look like an engine that serves nothing.
    int64_t constraintRejections = 0;
    int64_t constraintAdmissions = 0;

    /// Why the corpus holds fewer than @ref ExplorationRequest::corpusTarget problems, one line
    /// per combination. Empty when the target was met or not set.
    std::vector<std::string> shortfall;

    /// Every problem point found, flattened.
    std::vector<ProblemPoint> problems() const
    {
        std::vector<ProblemPoint> all;
        for(const auto& combination : combinations)
        {
            all.insert(all.end(), combination.problems.begin(), combination.problems.end());
        }
        return all;
    }
};

namespace detail
{

/// The cross product of every categorical parameter's declared values, truncated to @p limit;
/// @p total receives the untruncated size. Declaration order keeps the result reproducible.
inline std::vector<ProblemPoint>
    categoricalCombinations(const std::vector<Parameter>& parameters, size_t limit, size_t& total)
{
    std::vector<ProblemPoint> combinations{ProblemPoint{}};
    total = 1;

    for(const auto& parameter : parameters)
    {
        const auto values = parameter.enumerable();
        if(values.empty())
        {
            continue;
        }
        total *= values.size();

        std::vector<ProblemPoint> expanded;
        for(const auto& base : combinations)
        {
            for(const auto& value : values)
            {
                auto point = base;
                point[parameter.name] = value;
                expanded.push_back(std::move(point));
            }
        }
        combinations = std::move(expanded);

        if(combinations.size() > limit)
        {
            combinations.resize(limit);
        }
    }
    return combinations;
}

/// Whether @p point satisfies every relation @p metadata declares. A constraint that fails to
/// evaluate counts as unsatisfied rather than ignored.
inline bool satisfiesConstraints(const OperationMetadata& metadata, const ProblemPoint& point)
{
    if(metadata.constraints.size() == 0)
    {
        return true;
    }

    const auto context = detail::contextFor(point);
    for(size_t i = 0; i < metadata.constraints.size(); ++i)
    {
        const auto value = metadata.constraints.evaluate(i, context);
        if(!value.isBool() || !value.asBool())
        {
            return false;
        }
    }
    return true;
}

/// The cross product of declared regime buckets and common values (§4.3.2, §4.3.5), used to
/// seed the search: these combinations are realistic by construction, unlike independent draws.
/// Parameters with neither take their floor and are left to the walk. Strided to @p limit;
/// @p total receives the full size.
inline std::vector<Shape> regimeSkeleton(const OperationMetadata& metadata,
                                         const std::vector<ShapeDimension>& window,
                                         const std::vector<std::string>& numericParameters,
                                         size_t limit,
                                         size_t& total)
{
    std::vector<std::vector<int64_t>> choices;
    choices.reserve(numericParameters.size());
    total = 1;

    for(size_t i = 0; i < numericParameters.size(); ++i)
    {
        std::vector<int64_t> values;
        for(const auto& entry : metadata.regimes)
        {
            if(entry.second.parameter != numericParameters[i] || !entry.second.derived.empty())
            {
                continue;
            }
            for(const auto& bucket : entry.second.buckets)
            {
                if(bucket.is_number_integer())
                {
                    const auto value = bucket.get<int64_t>();
                    if(value >= window[i].low && value <= window[i].high)
                    {
                        values.push_back(value);
                    }
                }
            }
        }

        // §4.3.2 common_values: representative values from recorded usage.
        if(const auto* parameter = metadata.find(numericParameters[i]))
        {
            for(const auto& value : parameter->commonValues)
            {
                if(const auto* held = std::get_if<int64_t>(&value))
                {
                    if(*held >= window[i].low && *held <= window[i].high)
                    {
                        values.push_back(*held);
                    }
                }
            }
        }

        if(values.empty())
        {
            values.push_back(window[i].low);
        }
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        total *= values.size();
        choices.push_back(std::move(values));
    }

    std::vector<Shape> skeleton{Shape{}};
    for(const auto& values : choices)
    {
        std::vector<Shape> expanded;
        for(const auto& base : skeleton)
        {
            for(const auto value : values)
            {
                auto point = base;
                point.push_back(value);
                expanded.push_back(std::move(point));
            }
        }
        skeleton = std::move(expanded);
    }

    if(skeleton.size() > limit && limit > 0)
    {
        // Strided, not truncated: a prefix would keep only the smallest early parameters.
        std::vector<Shape> sampled;
        sampled.reserve(limit);
        const auto step = static_cast<double>(skeleton.size()) / static_cast<double>(limit);
        for(size_t i = 0; i < limit; ++i)
        {
            sampled.push_back(skeleton[static_cast<size_t>(static_cast<double>(i) * step)]);
        }
        skeleton = std::move(sampled);
    }
    return skeleton;
}

/// A one-line `name=value,...` rendering of @p point.
inline std::string describe(const ProblemPoint& point)
{
    std::string text;
    for(const auto& [name, value] : point)
    {
        if(!text.empty())
        {
            text += ",";
        }
        text += name + "=";
        std::visit(
            [&text](const auto& held) {
                using Held = std::decay_t<decltype(held)>;
                if constexpr(std::is_same_v<Held, std::string>)
                {
                    text += held;
                }
                else if constexpr(std::is_same_v<Held, bool>)
                {
                    text += held ? "true" : "false";
                }
                else
                {
                    text += std::to_string(held);
                }
            },
            value);
    }
    return text;
}

/// Every numeric parameter's search window: the declared range clipped to the ceiling, or the
/// ceiling alone. Shared with RegimeFocus.hpp so both walk the same box.
inline std::vector<ShapeDimension> numericWindow(const OperationMetadata& metadata,
                                                 const ExplorationRequest& request)
{
    std::vector<ShapeDimension> window;
    for(const auto& parameter : metadata.parameters)
    {
        if(parameter.type != ParameterType::INT64 && parameter.type != ParameterType::FLOAT64)
        {
            continue;
        }
        ShapeDimension dimension{parameter.name, 1, request.numericCeiling};
        if(parameter.range.has_value())
        {
            // The declared floor may be 0: some engines accept only unpadded convolutions.
            dimension.low = parameter.range->first;
            dimension.high = std::min(request.numericCeiling, parameter.range->second);
        }
        window.push_back(dimension);
    }
    return window;
}

} // namespace detail

/// @brief Explores the space @p metadata declares, keeping the points @p admits accepts.
///
/// Categorical combinations are enumerated; each gets its own oracle budget for the numeric
/// search, so the first one searched cannot exhaust a shared budget.
///
/// @p held marks points the caller already has. They stay feasible to the walk and count as
/// served when deciding what to grow, but are left out of the result.
inline ProblemCorpus exploreProblemSpace(const OperationMetadata& metadata,
                                         const ExplorationRequest& request,
                                         const ProblemOracle& admits,
                                         const ProblemOracle& held = {})
{
    ProblemCorpus corpus;
    corpus.operation = metadata.operation;

    const auto numericWindow = detail::numericWindow(metadata, request);
    for(const auto& dimension : numericWindow)
    {
        corpus.numericParameters.push_back(dimension.name);
    }

    size_t totalCombinations = 0;
    const auto combinations = detail::categoricalCombinations(
        metadata.parameters, request.maxCombinations, totalCombinations);

    if(totalCombinations > combinations.size())
    {
        corpus.skippedCombinations.push_back(
            std::to_string(totalCombinations - combinations.size()) + " of "
            + std::to_string(totalCombinations)
            + " categorical combinations not explored (maxCombinations bound)");
    }

    // Per-combination state kept for the growth pass. The memo makes growing affordable: a
    // longer walk from the same seed retraces the shorter one, answered from here.
    std::vector<FeasibleSetRequest> searches(combinations.size());
    std::vector<size_t> anchoredCounts(combinations.size(), 0);
    std::vector<std::map<Shape, bool>> answered(combinations.size());
    /// Held points the search reached, per combination.
    std::vector<int64_t> heldFound(combinations.size(), 0);
    const auto isHeld = [&held](const ProblemPoint& point) { return held && held(point); };

    const auto oracleFor = [&](size_t index) -> ShapeOracle {
        return [&, index](const Shape& shape) {
            auto& memo = answered[index];
            const auto known = memo.find(shape);
            if(known != memo.end())
            {
                return known->second;
            }
            auto point = combinations[index];
            for(size_t i = 0; i < corpus.numericParameters.size(); ++i)
            {
                point[corpus.numericParameters[i]] = shape[i];
            }
            // Declared constraints first, so a non-problem never costs a graph build.
            bool verdict = false;
            if(!detail::satisfiesConstraints(metadata, point))
            {
                ++corpus.constraintRejections;
            }
            else
            {
                ++corpus.constraintAdmissions;
                verdict = admits(point);
            }
            memo.emplace(shape, verdict);
            return verdict;
        };
    };

    for(size_t index = 0; index < combinations.size(); ++index)
    {
        const auto& categorical = combinations[index];

        CombinationResult result;
        result.categorical = categorical;

        if(numericWindow.empty())
        {
            // All-categorical operation: one problem per combination.
            if(admits(categorical))
            {
                result.problems.push_back(categorical);
            }
            corpus.combinations.push_back(std::move(result));
            continue;
        }

        const auto oracle = oracleFor(index);

        size_t skeletonTotal = 0;
        const auto skeleton = detail::regimeSkeleton(
            metadata, numericWindow, corpus.numericParameters, request.maxSkeleton, skeletonTotal);
        if(skeletonTotal > skeleton.size() && index == 0)
        {
            corpus.skippedCombinations.push_back(
                std::to_string(skeletonTotal - skeleton.size()) + " of "
                + std::to_string(skeletonTotal)
                + " declared regime combinations not tried (maxSkeleton bound)");
        }

        FeasibleSetRequest search;
        search.seeds = skeleton;
        search.dimensions = numericWindow;
        search.targetCount = request.pointsPerCombination;
        search.oracleBudget = request.budgetPerCombination;
        search.restarts = request.restarts;
        search.stepsPerStart = request.stepsPerStart;
        // Varied per combination so dtypes do not retrace each other; still reproducible.
        search.seed = request.seed + index;

        // Workload half (§5.2): most points are drawn near recorded shapes; the remainder is left
        // to the region search.
        std::mt19937_64 rng(request.seed + index + 1);
        std::set<std::string> seen;
        std::vector<ProblemPoint> anchored;

        const auto accept = [&](const ProblemPoint& point) {
            if(!seen.insert(detail::describe(point)).second)
            {
                return false;
            }
            if(!detail::satisfiesConstraints(metadata, point))
            {
                ++corpus.constraintRejections;
                return false;
            }
            ++corpus.constraintAdmissions;
            if(!admits(point))
            {
                return false;
            }
            if(isHeld(point))
            {
                ++heldFound[index];
                return false;
            }
            return true;
        };

        size_t archetypeQuota = 0;
        size_t neighbourhoodQuota = 0;
        if(!metadata.mixture.isExplorationOnly())
        {
            const auto total = static_cast<double>(request.pointsPerCombination);
            archetypeQuota
                = static_cast<size_t>(std::max(0.0, metadata.mixture.archetypes * total));
            neighbourhoodQuota
                = static_cast<size_t>(std::max(0.0, metadata.mixture.neighbourhood * total));
        }

        // Capped so archetypes the engine declines cannot consume the whole budget.
        const size_t archetypeAttempts = archetypeQuota * 20;
        for(size_t attempt = 0; attempt < archetypeAttempts && anchored.size() < archetypeQuota;
            ++attempt)
        {
            const auto& archetype = metadata.archetypes[attempt % metadata.archetypes.size()];
            const auto drawn = detail::drawFromArchetype(metadata, archetype, categorical, rng);
            if(!drawn.has_value())
            {
                continue;
            }
            if(accept(*drawn))
            {
                anchored.push_back(*drawn);
                result.problems.push_back(*drawn);
                ++result.fromArchetypes;
            }
        }

        // Perturb only admitted anchors; with none, the budget goes to exploration.
        const size_t neighbourhoodAttempts = anchored.empty() ? 0 : neighbourhoodQuota * 20;
        size_t drawnNearby = 0;
        for(size_t attempt = 0; attempt < neighbourhoodAttempts && drawnNearby < neighbourhoodQuota;
            ++attempt)
        {
            const auto& anchor = anchored[attempt % anchored.size()];
            const auto moved = detail::perturbWithinNeighbourhood(metadata, anchor, rng);
            if(accept(moved))
            {
                result.problems.push_back(moved);
                ++result.fromNeighbourhood;
                ++drawnNearby;
            }
        }

        if(!metadata.archetypes.empty() && anchored.empty())
        {
            // Reported so it is not mistaken for an operation with no declared workloads.
            corpus.skippedCombinations.push_back("no declared archetype was admitted for "
                                                 + detail::describe(categorical)
                                                 + "; that combination is exploration only");
        }

        // The search fills whatever the anchored draws did not.
        search.targetCount = std::max<int64_t>(
            0, request.pointsPerCombination - static_cast<int64_t>(result.problems.size()));

        // Admitted points seed the walk inside the served region.
        for(const auto& point : result.problems)
        {
            Shape shape;
            shape.reserve(corpus.numericParameters.size());
            for(const auto& name : corpus.numericParameters)
            {
                shape.push_back(detail::integerAt(point, name).value_or(1));
            }
            search.seeds.push_back(std::move(shape));
        }
        searches[index] = search;
        anchoredCounts[index] = result.problems.size();

        if(search.targetCount > 0)
        {
            auto found = [&]() {
                if(!result.problems.empty() || request.probeBudget >= search.oracleBudget)
                {
                    return buildFeasibleShapeSet(oracle, search);
                }
                // Nothing anchored: probe first, and run the full search only if the probe found
                // something. The full search replays the probe's walk from the memo.
                auto probe = search;
                probe.oracleBudget = request.probeBudget;
                auto probed = buildFeasibleShapeSet(oracle, probe);
                return probed.stats.distinct == 0 ? probed : buildFeasibleShapeSet(oracle, search);
            }();
            result.stats = found.stats;
            for(const auto& shape : found.shapes)
            {
                auto point = categorical;
                for(size_t i = 0; i < corpus.numericParameters.size(); ++i)
                {
                    point[corpus.numericParameters[i]] = shape[i];
                }
                if(isHeld(point))
                {
                    ++heldFound[index];
                    continue;
                }
                if(seen.insert(detail::describe(point)).second)
                {
                    result.problems.push_back(std::move(point));
                    ++result.fromExploration;
                }
            }
        }

        corpus.combinations.push_back(std::move(result));
    }

    // Growth toward corpusTarget: the first pass gave each combination a modest fixed target
    // so declined ones stay cheap; combinations that turned out to be served are searched
    // again for the remainder.
    const auto supplied = [&corpus]() {
        int64_t total = 0;
        for(const auto& combination : corpus.combinations)
        {
            total += static_cast<int64_t>(combination.problems.size());
        }
        return total;
    };

    std::vector<bool> spent(combinations.size(), false);
    while(request.corpusTarget > 0 && !numericWindow.empty())
    {
        const auto before = supplied();
        const auto need = request.corpusTarget - before;
        if(need <= 0)
        {
            break;
        }

        std::vector<size_t> growable;
        for(size_t index = 0; index < corpus.combinations.size(); ++index)
        {
            if(!spent[index]
               && (!corpus.combinations[index].problems.empty() || heldFound[index] > 0))
            {
                growable.push_back(index);
            }
        }
        if(growable.empty())
        {
            break;
        }

        const auto share = (need + static_cast<int64_t>(growable.size()) - 1)
                           / static_cast<int64_t>(growable.size());
        for(const auto index : growable)
        {
            auto& result = corpus.combinations[index];
            auto& search = searches[index];
            const auto oracle = oracleFor(index);
            // Held points and anchored duplicates fill cells but add nothing, so they are
            // excluded from the fresh selection. Anchored draws are always kept.
            std::set<std::string> anchoredSeen;
            for(size_t i = 0; i < anchoredCounts[index]; ++i)
            {
                anchoredSeen.insert(detail::describe(result.problems[i]));
            }
            struct Selection
            {
                std::vector<ProblemPoint> fresh;
                int64_t held = 0;
            };
            const auto select = [&](const FeasibleShapeSet& set) {
                Selection selection;
                for(const auto& shape : set.shapes)
                {
                    auto point = result.categorical;
                    for(size_t i = 0; i < corpus.numericParameters.size(); ++i)
                    {
                        point[corpus.numericParameters[i]] = shape[i];
                    }
                    if(isHeld(point))
                    {
                        ++selection.held;
                    }
                    else if(anchoredSeen.count(detail::describe(point)) == 0)
                    {
                        selection.fresh.push_back(std::move(point));
                    }
                }
                return selection;
            };

            // The new search replaces the old one's contribution, so it must resupply it plus
            // this round's share.
            const auto wanted
                = static_cast<int64_t>(result.problems.size() - anchoredCounts[index]) + share;
            // Sized to include anchored and held points, which the search also returns.
            search.targetCount
                = static_cast<int64_t>(result.problems.size()) + heldFound[index] + share;

            // Run at the current budget first (a larger target re-tessellates what the walk
            // observed). Then grow: by a quarter for sparse regions, so saturation is cheap to
            // establish, or doubling once the region proves productive.
            auto found = buildFeasibleShapeSet(oracle, search);
            auto selection = select(found);
            // At least one new point per hundred queries counts as productive.
            bool productive = found.stats.oracleCalls > 0
                              && found.stats.distinct * 100 >= found.stats.oracleCalls;
            while(static_cast<int64_t>(selection.fresh.size()) < wanted)
            {
                const auto cells = static_cast<int64_t>(found.shapes.size());
                if(cells < found.stats.distinct)
                {
                    // Selection, not search, was the limit: held or anchored points took some
                    // cells. Ask for more cells over the same observations before spending
                    // more budget.
                    const auto fresh = static_cast<int64_t>(selection.fresh.size());
                    const auto deficit = wanted - fresh;
                    const auto extra
                        = fresh > 0 ? (deficit * cells + fresh - 1) / fresh : found.stats.distinct;
                    search.targetCount = std::min(cells + extra, found.stats.distinct);
                    found = buildFeasibleShapeSet(oracle, search);
                    selection = select(found);
                    continue;
                }
                if(search.oracleBudget >= request.budgetPerCombination * request.budgetGrowthLimit)
                {
                    result.searchCapped = true;
                    break;
                }
                const auto reached = found.stats.distinct;
                const auto grow = [](int64_t value, bool twice) {
                    return twice ? value * 2 : value + std::max<int64_t>(1, value / 4);
                };
                search.oracleBudget = grow(search.oracleBudget, productive);
                search.stepsPerStart = grow(search.stepsPerStart, productive);
                found = buildFeasibleShapeSet(oracle, search);
                selection = select(found);
                if(found.stats.distinct <= reached)
                {
                    result.saturated = true;
                    break;
                }
                productive = true;
            }
            spent[index] = result.saturated || result.searchCapped;

            result.problems.resize(anchoredCounts[index]);
            result.fromExploration = static_cast<int64_t>(selection.fresh.size());
            heldFound[index] = selection.held;
            result.stats = found.stats;
            result.problems.insert(result.problems.end(),
                                   std::make_move_iterator(selection.fresh.begin()),
                                   std::make_move_iterator(selection.fresh.end()));
        }

        if(supplied() <= before)
        {
            // No combination added a point: each is now saturated or capped.
            break;
        }
    }

    // The served edges, from every answer the engine gave: the memo holds all each walk reached,
    // and anchored draws were asked outside it.
    for(size_t index = 0; index < corpus.combinations.size() && !numericWindow.empty(); ++index)
    {
        auto& result = corpus.combinations[index];
        const auto dimensions = corpus.numericParameters.size();
        std::vector<const Shape*> low(dimensions, nullptr);
        std::vector<const Shape*> high(dimensions, nullptr);
        std::vector<Shape> anchoredShapes;
        anchoredShapes.reserve(result.problems.size());
        for(const auto& point : result.problems)
        {
            Shape shape;
            shape.reserve(dimensions);
            for(const auto& name : corpus.numericParameters)
            {
                shape.push_back(detail::integerAt(point, name).value_or(1));
            }
            anchoredShapes.push_back(std::move(shape));
        }
        const auto consider = [&](const Shape& shape) {
            for(size_t d = 0; d < dimensions; ++d)
            {
                if(low[d] == nullptr || shape[d] < (*low[d])[d]
                   || (shape[d] == (*low[d])[d] && shape < *low[d]))
                {
                    low[d] = &shape;
                }
                if(high[d] == nullptr || shape[d] > (*high[d])[d]
                   || (shape[d] == (*high[d])[d] && shape < *high[d]))
                {
                    high[d] = &shape;
                }
            }
        };
        for(const auto& [shape, verdict] : answered[index])
        {
            if(verdict)
            {
                consider(shape);
            }
        }
        for(const auto& shape : anchoredShapes)
        {
            consider(shape);
        }
        if(dimensions == 0 || low.front() == nullptr)
        {
            continue;
        }
        const auto pointOf = [&](const Shape& shape) {
            auto point = result.categorical;
            for(size_t d = 0; d < dimensions; ++d)
            {
                point[corpus.numericParameters[d]] = shape[d];
            }
            return point;
        };
        result.lowest.reserve(dimensions);
        result.highest.reserve(dimensions);
        for(size_t d = 0; d < dimensions; ++d)
        {
            result.lowest.push_back(pointOf(*low[d]));
            result.highest.push_back(pointOf(*high[d]));
        }
    }

    if(request.corpusTarget > 0 && supplied() < request.corpusTarget)
    {
        const auto total = supplied();
        corpus.shortfall.push_back(std::to_string(total) + " of "
                                   + std::to_string(request.corpusTarget) + " problems requested");
        for(size_t index = 0; index < corpus.combinations.size(); ++index)
        {
            const auto& result = corpus.combinations[index];
            if(result.problems.empty() && heldFound[index] == 0)
            {
                continue;
            }
            const auto who = detail::describe(result.categorical) + ": "
                             + std::to_string(result.problems.size()) + " new problems ("
                             + std::to_string(heldFound[index]) + " more already held), ";
            if(result.searchCapped)
            {
                corpus.shortfall.push_back(
                    who + "search stopped at its budget limit ("
                    + std::to_string(searches[index].oracleBudget)
                    + " oracle calls) while still finding new points -- more exist; raise "
                      "--budget");
            }
            else if(result.saturated)
            {
                corpus.shortfall.push_back(
                    who + "saturated: " + std::to_string(result.stats.distinct)
                    + " distinct served points observed, and doubling the search found no more");
            }
            else
            {
                corpus.shortfall.push_back(who
                                           + "grown, but its new points repeated ones it "
                                             "already held");
            }
        }
    }

    return corpus;
}

} // namespace hipdnn_corpus_gen
