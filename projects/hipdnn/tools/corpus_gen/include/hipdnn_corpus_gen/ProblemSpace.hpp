// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_corpus_gen/ArgumentResolver.hpp>
#include <hipdnn_corpus_gen/FeasibleShapeSet.hpp>
#include <hipdnn_corpus_gen/OperationMetadata.hpp>
#include <hipdnn_corpus_gen/PointFilter.hpp>
#include <hipdnn_corpus_gen/WorkloadSampling.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <utility>
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
    /// budget. When the declaration weights the values of a parameter (Parameter::weights) the
    /// weighted parameters split a budget rather than multiply it: the first pass is this many
    /// points for each distinct combination of the unweighted parameters, divided among the
    /// combinations by weight, so it is the mean per unweighted combination, not per combination.
    /// A combination may get none: the growth toward @ref corpusTarget, the regime floors and the
    /// quotas reach it. Every declared value that some combination can hold is drawn at least
    /// once, so the first pass is more than this budget only when there are more such values
    /// than points. See detail::firstPassTargets.
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

    /// The caller's `--keep` clauses. Those on an enum or bool parameter narrow its axis before
    /// the combinations are counted and sampled, so @ref maxCombinations bounds the combinations
    /// a keep admits and not the full product; clauses on a numeric parameter do not apply here,
    /// since the caller's oracle filters them point by point.
    std::vector<KeepClause> keep;

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

    /// Scales the archetype and neighbourhood quotas with @ref corpusTarget instead of fixing
    /// them at the mixture's share of the combination's first-pass count: each combination's
    /// anchored quota is the mixture share of max(its first-pass count, corpusTarget / the
    /// explored combinations that can hold a problem), times its relative size when values are
    /// weighted, so the quotas sum to the target and the declared mixture holds at a large
    /// --count rather than collapsing into the exploration share. Off by default (the fixed
    /// quotas); without a corpusTarget it changes nothing.
    bool anchorScaling = false;
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

    /// The categorical combinations the declaration holds (the product of each enum's and bool's
    /// values the request's `keep` clauses admit), how many of them the declared constraints
    /// leave unable to hold a problem (see detail::excludedByCategoricals), and how many of
    /// the rest ExplorationRequest::maxCombinations left unexplored: the combinations that
    /// could hold a problem and were not sampled. Without a bound that binds, none is left
    /// unexplored; constraint-empty ones are counted either way.
    size_t categoricalTotal = 0;
    size_t categoricalConstraintEmpty = 0;
    size_t categoricalNotExplored = 0;

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

    /// Problems drawn from archetypes or their neighbourhood, over every combination.
    int64_t anchoredProblems() const
    {
        int64_t anchored = 0;
        for(const auto& combination : combinations)
        {
            anchored += combination.fromArchetypes + combination.fromNeighbourhood;
        }
        return anchored;
    }

    /// Every problem drawn from archetypes or their neighbourhood, over every combination.
    std::vector<ProblemPoint> anchoredPoints() const
    {
        std::vector<ProblemPoint> anchored;
        for(const auto& combination : combinations)
        {
            const auto count
                = static_cast<size_t>(combination.fromArchetypes + combination.fromNeighbourhood);
            anchored.insert(
                anchored.end(),
                combination.problems.begin(),
                combination.problems.begin()
                    + static_cast<std::ptrdiff_t>(std::min(count, combination.problems.size())));
        }
        return anchored;
    }

    /// Anchored problems over every problem found, 0 for an empty corpus: the mixture the
    /// corpus actually realised, as against the one the declaration asks for.
    double anchoredShare() const
    {
        const auto all = problems().size();
        return all == 0 ? 0.0 : static_cast<double>(anchoredProblems()) / static_cast<double>(all);
    }
};

namespace detail
{

/// The cross product of every categorical parameter's declared values that @p keep admits, at
/// most @p limit of it plus whatever keeps every admitted value present; @p total receives the
/// full size of the admitted product, 0 when @p keep admits no value of some parameter.
///
/// Past the limit the product is sampled, not cut to a prefix (a prefix keeps only the first
/// values of the earliest parameters), and each value the sample missed gets a combination of
/// its own, so a model trained on the corpus has seen every value (§5.2). The sample steps
/// through the product by a stride coprime to its size, so it visits distinct combinations in a
/// fixed, scattered order: an even stride aliases with the mixed-radix index and leaves whole
/// value triples out (sdpa_fwd's 6912 combinations at a stride of 6.75 have no kv_bhsd, jitter,
/// local_4096 one). @p live, when given, says which combinations can hold a problem (see
/// excludedByCategoricals): only those are sampled, so the limit is spent on combinations that
/// count. Declaration order keeps the result reproducible.
///
/// @p constraintEmpty, when given, receives how many combinations of the admitted product @p
/// live refuses, counted over the whole product and not only the part the sample walks: past
/// the limit the walk goes on counting after the sample is full.
///
/// @p keep narrows each axis to the values it admits (admitsValue) before anything is counted,
/// so the limit and the per-value cover are spent on combinations the caller can keep rather
/// than on a sample the caller then discards.
inline std::vector<ProblemPoint>
    categoricalCombinations(const std::vector<Parameter>& parameters,
                            size_t limit,
                            size_t& total,
                            const std::function<bool(const ProblemPoint&)>& live = {},
                            const std::vector<KeepClause>& keep = {},
                            size_t* constraintEmpty = nullptr)
{
    if(constraintEmpty != nullptr)
    {
        *constraintEmpty = 0;
    }
    std::vector<std::pair<std::string, std::vector<ParameterValue>>> axes;
    total = 1;
    for(const auto& parameter : parameters)
    {
        auto values = parameter.enumerable();
        if(values.empty())
        {
            continue;
        }
        values.erase(std::remove_if(values.begin(),
                                    values.end(),
                                    [&](const ParameterValue& value) {
                                        return !admitsValue(keep, parameter.name, value);
                                    }),
                     values.end());
        if(values.empty())
        {
            total = 0;
            return {};
        }
        total = total > std::numeric_limits<size_t>::max() / values.size()
                    ? std::numeric_limits<size_t>::max()
                    : total * values.size();
        axes.emplace_back(parameter.name, std::move(values));
    }

    // Combination `index`, the first parameter most significant.
    const auto at = [&axes](size_t index) {
        ProblemPoint point;
        for(size_t axis = axes.size(); axis > 0; --axis)
        {
            const auto& values = axes[axis - 1].second;
            point[axes[axis - 1].first] = values[index % values.size()];
            index /= values.size();
        }
        return point;
    };

    std::vector<ProblemPoint> combinations;
    if(total <= limit || limit == 0)
    {
        combinations.reserve(total);
        for(size_t index = 0; index < total; ++index)
        {
            combinations.push_back(at(index));
            if(constraintEmpty != nullptr && live && !live(combinations.back()))
            {
                ++*constraintEmpty;
            }
        }
        return combinations;
    }

    // The stride nearest the golden ratio of the product that shares no factor with it, so the
    // walk index * stride % total is a permutation of the product.
    size_t stride
        = std::max<size_t>(1, static_cast<size_t>(static_cast<double>(total) * 0.6180339887));
    while(std::gcd(stride, total) != 1)
    {
        ++stride;
    }
    // The walk's product index after @p step strides, by repeated addition modulo the total so
    // a large product cannot overflow.
    const auto next = [&](size_t index) {
        return index >= total - stride ? index - (total - stride) : index + stride;
    };
    const auto counts = [&live](const ProblemPoint& point) { return !live || live(point); };

    combinations.reserve(limit);
    size_t visited = 0;
    size_t cursor = 0;
    size_t empty = 0;
    for(; visited < total && combinations.size() < limit; ++visited, cursor = next(cursor))
    {
        auto point = at(cursor);
        if(counts(point))
        {
            combinations.push_back(std::move(point));
        }
        else
        {
            ++empty;
        }
    }
    if(constraintEmpty != nullptr)
    {
        auto ahead = cursor;
        for(size_t step = visited; step < total; ++step, ahead = next(ahead))
        {
            empty += counts(at(ahead)) ? 0 : 1;
        }
        *constraintEmpty = empty;
    }
    // A value the sample missed gets the next combination of the walk that holds it, if any
    // can; failing that, the first combination with the value swapped in.
    for(const auto& axis : axes)
    {
        const auto& name = axis.first;
        for(const auto& value : axis.second)
        {
            const bool present
                = std::any_of(combinations.begin(), combinations.end(), [&](const auto& point) {
                      return point.at(name) == value;
                  });
            if(present)
            {
                continue;
            }
            bool found = false;
            auto ahead = cursor;
            for(size_t step = visited; step < total && !found; ++step, ahead = next(ahead))
            {
                auto point = at(ahead);
                if(point.at(name) == value && counts(point))
                {
                    combinations.push_back(std::move(point));
                    found = true;
                }
            }
            // With no combination at all (a keep that every constraint empties) there is none to
            // swap the value into.
            if(!found && !combinations.empty())
            {
                auto point = combinations.front();
                point[name] = value;
                combinations.push_back(std::move(point));
            }
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

/// Whether any parameter weights its values (Parameter::weights).
inline bool anyWeighted(const std::vector<Parameter>& parameters)
{
    return std::any_of(parameters.begin(), parameters.end(), [](const Parameter& parameter) {
        return !parameter.weights.empty();
    });
}

/// Whether the categorical values of @p combination alone make some declared constraint false.
/// The language is three-valued: a constraint that reads a numeric parameter (not bound here)
/// stays unresolved unless its other arms decide it, so only a definite false counts. This is
/// how (alignment bottom_right, non-causal, no window) is found empty without a shape.
inline bool excludedByCategoricals(const OperationMetadata& metadata,
                                   const ProblemPoint& combination)
{
    if(metadata.constraints.size() == 0)
    {
        return false;
    }
    const auto context = detail::contextFor(combination);
    for(size_t i = 0; i < metadata.constraints.size(); ++i)
    {
        const auto value = metadata.constraints.evaluate(i, context);
        if(!value.containsUnresolved() && value.isBool() && !value.asBool())
        {
            return true;
        }
    }
    return false;
}

/// How large each of @p combinations is relative to the average over those that can hold a
/// problem: 1 for every combination that can when nothing is weighted, else sized so that each
/// weighted parameter's values hold their declared weights' shares of the whole.
///
/// A combination its categorical values make empty (excludedByCategoricals) has size 0, whether
/// or not anything is weighted. Dropping one value's combinations from the sum would otherwise
/// leave that value short of its weight (a quarter of the window-none combinations are empty,
/// none of the windowed ones are), so the sizes start as the product of the values' relative
/// weights and are raked: each weighted parameter's values are rescaled in turn to their
/// weights' shares, over the values that have any combination left, until all agree. The one
/// definition the main search and the regime focus share, so both realise the declared shares.
///
/// The unweighted enum and bool parameters named in @p even are raked too, each value to an
/// equal share, so that a value fewer combinations can hold (a form that the constraints leave
/// in a few combinations) is not smaller than the others. When a constraint ties one value to
/// another, the shares may be reachable only by taking some combinations to 0, which the raking
/// approaches without reaching; @p converged (when given) is false then (see weighGroups).
inline std::vector<double> combinationSizes(const OperationMetadata& metadata,
                                            const std::vector<ProblemPoint>& combinations,
                                            const std::set<std::string>& even = {},
                                            bool* converged = nullptr)
{
    if(converged != nullptr)
    {
        *converged = true;
    }
    std::vector<double> sizes(combinations.size(), 1.0);
    const bool raked = !even.empty() || anyWeighted(metadata.parameters);
    size_t live = 0;
    for(size_t index = 0; index < combinations.size(); ++index)
    {
        if(excludedByCategoricals(metadata, combinations[index]))
        {
            sizes[index] = 0.0;
            continue;
        }
        ++live;
        for(const auto& parameter : metadata.parameters)
        {
            const auto found = combinations[index].find(parameter.name);
            if(!parameter.weights.empty() && found != combinations[index].end())
            {
                sizes[index] *= parameter.relativeWeight(std::get<std::string>(found->second));
            }
        }
    }
    if(!raked)
    {
        return sizes;
    }

    // The raked parameters: the weighted ones and the unweighted ones named in `even`.
    std::vector<const Parameter*> raking;
    for(const auto& parameter : metadata.parameters)
    {
        if(!parameter.weights.empty()
           || (!parameter.enumerable().empty() && even.count(parameter.name) > 0))
        {
            raking.push_back(&parameter);
        }
    }
    const auto keyOf = [](const ParameterValue& value) {
        if(const auto* text = std::get_if<std::string>(&value))
        {
            return *text;
        }
        const auto* flag = std::get_if<bool>(&value);
        return std::string(flag != nullptr && *flag ? "true" : "false");
    };

    const auto normalise = [&sizes, live]() {
        double sum = 0.0;
        for(const auto size : sizes)
        {
            sum += size;
        }
        if(sum > 0.0)
        {
            for(auto& size : sizes)
            {
                size *= static_cast<double>(live) / sum;
            }
        }
    };
    normalise();
    // A raking that settles is the shares met; one still moving after the last round is
    // approaching an unreachable boundary, a combination shrinking toward 0.
    double change = 0.0;
    for(int round = 0; round < 200; ++round)
    {
        change = 0.0;
        for(const auto* parameter : raking)
        {
            std::map<std::string, double> mass;
            for(size_t index = 0; index < combinations.size(); ++index)
            {
                const auto found = combinations[index].find(parameter->name);
                if(found != combinations[index].end())
                {
                    mass[keyOf(found->second)] += sizes[index];
                }
            }
            double total = 0.0;
            double weight = 0.0;
            for(const auto& [value, held] : mass)
            {
                if(held > 0.0)
                {
                    total += held;
                    weight += parameter->weightOf(value);
                }
            }
            if(!(total > 0.0))
            {
                continue;
            }
            for(size_t index = 0; index < combinations.size(); ++index)
            {
                const auto found = combinations[index].find(parameter->name);
                if(found == combinations[index].end())
                {
                    continue;
                }
                const auto value = keyOf(found->second);
                if(mass[value] > 0.0)
                {
                    const auto factor = total * parameter->weightOf(value) / weight / mass[value];
                    change = std::max(change, std::abs(factor - 1.0));
                    sizes[index] *= factor;
                }
            }
        }
        normalise();
        if(change < 1e-9)
        {
            break;
        }
    }
    if(converged != nullptr)
    {
        *converged = change < 1e-6;
    }
    return sizes;
}

/// A reproducible pseudo-random key for part @p index under @p seed (splitmix64): the tie-break
/// of apportion, so equal remainders are not always settled in favour of the earliest parts,
/// which the enumeration order would turn into a bias toward the first value of the leading
/// parameters.
inline uint64_t tieKey(uint64_t seed, size_t index)
{
    uint64_t z = seed + 0x9e3779b97f4a7c15ULL * (static_cast<uint64_t>(index) + 1);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/// Splits @p total into whole parts in proportion to @p sizes. The parts sum to @p total exactly
/// and each is its ideal share rounded down or up (a per-part ceiling or rounding would not sum,
/// and over a few points the error is the share).
///
/// The rounded-up units are not the largest remainders: when every ideal is below one (a need
/// of 60 over 864 parts), "largest remainder" hands every unit to the biggest parts and no
/// small part ever gets one. They are drawn by systematic sampling over the remainders instead,
/// in the order of @p tieKeys (index order when none are given), from a start the keys fix:
/// each part gets its extra unit with probability equal to its remainder, so the expected share
/// of every part, and of every value the parts group, is its size's.
inline std::vector<int64_t> apportion(int64_t total,
                                      const std::vector<double>& sizes,
                                      const std::vector<uint64_t>& tieKeys = {})
{
    std::vector<int64_t> parts(sizes.size(), 0);
    double sum = 0.0;
    for(const auto size : sizes)
    {
        sum += size;
    }
    if(total <= 0 || sizes.empty() || !(sum > 0.0))
    {
        return parts;
    }
    std::vector<double> remainder(sizes.size(), 0.0);
    int64_t given = 0;
    for(size_t i = 0; i < sizes.size(); ++i)
    {
        const double ideal = static_cast<double>(total) * sizes[i] / sum;
        parts[i] = static_cast<int64_t>(std::floor(ideal));
        given += parts[i];
        remainder[i] = ideal - static_cast<double>(parts[i]);
    }

    std::vector<size_t> order(sizes.size());
    for(size_t i = 0; i < order.size(); ++i)
    {
        order[i] = i;
    }
    uint64_t mixed = 0x9e3779b97f4a7c15ULL;
    if(!tieKeys.empty())
    {
        std::stable_sort(order.begin(), order.end(), [&tieKeys](size_t left, size_t right) {
            return tieKeys[left] < tieKeys[right];
        });
        for(const auto key : tieKeys)
        {
            mixed = (mixed ^ key) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    // The start in [0, 1): the keys' own, or the middle (plain rounding of the running total).
    const double start
        = tieKeys.empty() ? 0.5 : static_cast<double>(mixed >> 11) / 9007199254740992.0;

    std::vector<bool> extra(sizes.size(), false);
    double running = 0.0;
    auto lastCount = static_cast<int64_t>(std::floor(start));
    for(const auto index : order)
    {
        running += remainder[index];
        const auto count = static_cast<int64_t>(std::floor(running + start));
        if(count > lastCount && given < total)
        {
            ++parts[index];
            extra[index] = true;
            ++given;
        }
        lastCount = count;
    }
    // Rounding in the running sum can leave one unit unplaced: give it to the largest remainder
    // not yet raised.
    while(given < total)
    {
        size_t best = sizes.size();
        for(const auto index : order)
        {
            if(!extra[index] && (best == sizes.size() || remainder[index] > remainder[best]))
            {
                best = index;
            }
        }
        if(best == sizes.size())
        {
            break;
        }
        ++parts[best];
        extra[best] = true;
        ++given;
    }
    return parts;
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

    // Shape `index` of the product, first parameter most significant, decoded without
    // materialising the product (which reaches hundreds of thousands of shapes).
    const auto shapeAt = [&choices](size_t index) {
        Shape shape(choices.size());
        for(size_t d = choices.size(); d > 0; --d)
        {
            const auto& values = choices[d - 1];
            shape[d - 1] = values[index % values.size()];
            index /= values.size();
        }
        return shape;
    };

    std::vector<Shape> skeleton;
    if(total <= limit || limit == 0)
    {
        skeleton.reserve(total);
        for(size_t i = 0; i < total; ++i)
        {
            skeleton.push_back(shapeAt(i));
        }
    }
    else
    {
        // Strided, not truncated: a prefix would keep only the smallest early parameters.
        skeleton.reserve(limit);
        const auto step = static_cast<double>(total) / static_cast<double>(limit);
        for(size_t i = 0; i < limit; ++i)
        {
            skeleton.push_back(shapeAt(static_cast<size_t>(static_cast<double>(i) * step)));
        }
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

/// Makes sure the first pass draws every declared value of every enumerable parameter, weighted
/// or not, that some live combination holds. A value @p targets leaves with no point (a light
/// value spread over combinations whose shares each round to nothing) gets one in its live
/// combination of largest size, ties to the lowest key, and that point is taken from the
/// combination with the largest surplus over its exact share (@p budget split by @p sizes), so
/// the total stays the budget. A combination is only a donor if giving the point up leaves every
/// value it holds with a point. When none can give one the total grows by the points owed, which
/// happens only when there are more uncovered values than points.
inline void coverEveryValue(const OperationMetadata& metadata,
                            const std::vector<ProblemPoint>& combinations,
                            const std::vector<double>& sizes,
                            const std::vector<uint64_t>& keys,
                            double budget,
                            std::vector<int64_t>& targets)
{
    std::vector<std::vector<ParameterValue>> axes;
    std::vector<std::string> names;
    for(const auto& parameter : metadata.parameters)
    {
        auto values = parameter.enumerable();
        if(!values.empty())
        {
            names.push_back(parameter.name);
            axes.push_back(std::move(values));
        }
    }

    // Which value of each axis a combination holds (-1: it names none).
    std::vector<std::vector<int>> holds(combinations.size(), std::vector<int>(axes.size(), -1));
    std::vector<std::vector<int64_t>> points(axes.size());
    for(size_t axis = 0; axis < axes.size(); ++axis)
    {
        points[axis].assign(axes[axis].size(), 0);
    }
    double total = 0.0;
    for(const auto size : sizes)
    {
        total += size;
    }
    for(size_t index = 0; index < combinations.size(); ++index)
    {
        for(size_t axis = 0; axis < axes.size(); ++axis)
        {
            const auto found = combinations[index].find(names[axis]);
            if(found == combinations[index].end())
            {
                continue;
            }
            const auto& values = axes[axis];
            const auto at = std::find(values.begin(), values.end(), found->second);
            if(at != values.end())
            {
                holds[index][axis] = static_cast<int>(at - values.begin());
                points[axis][static_cast<size_t>(holds[index][axis])] += targets[index];
            }
        }
    }
    if(!(total > 0.0))
    {
        return;
    }

    const auto shift = [&](size_t index, int64_t by) {
        targets[index] += by;
        for(size_t axis = 0; axis < axes.size(); ++axis)
        {
            if(holds[index][axis] >= 0)
            {
                points[axis][static_cast<size_t>(holds[index][axis])] += by;
            }
        }
    };
    const auto keepsEveryValue = [&](size_t index) {
        if(targets[index] > 1)
        {
            return true;
        }
        for(size_t axis = 0; axis < axes.size(); ++axis)
        {
            if(holds[index][axis] >= 0 && points[axis][static_cast<size_t>(holds[index][axis])] < 2)
            {
                return false;
            }
        }
        return true;
    };

    constexpr auto NONE = std::numeric_limits<size_t>::max();
    for(size_t axis = 0; axis < axes.size(); ++axis)
    {
        for(size_t value = 0; value < axes[axis].size(); ++value)
        {
            if(points[axis][value] > 0)
            {
                continue;
            }
            auto taker = NONE;
            for(size_t index = 0; index < combinations.size(); ++index)
            {
                if(sizes[index] > 0.0 && holds[index][axis] == static_cast<int>(value)
                   && (taker == NONE || sizes[index] > sizes[taker]
                       || (sizes[index] == sizes[taker] && keys[index] < keys[taker])))
                {
                    taker = index;
                }
            }
            if(taker == NONE)
            {
                continue;
            }
            shift(taker, 1);
            auto donor = NONE;
            double donorSurplus = 0.0;
            for(size_t index = 0; index < combinations.size(); ++index)
            {
                if(index == taker || targets[index] <= 0 || !keepsEveryValue(index))
                {
                    continue;
                }
                const auto surplus
                    = static_cast<double>(targets[index]) - budget * sizes[index] / total;
                if(donor == NONE || surplus > donorSurplus
                   || (surplus == donorSurplus && keys[index] < keys[donor]))
                {
                    donor = index;
                    donorSurplus = surplus;
                }
            }
            if(donor != NONE)
            {
                shift(donor, -1);
            }
        }
    }
}

/// The first pass's problem count for each of @p combinations, @p sizes being their
/// combinationSizes.
///
/// Nothing weighted: pointsPerCombination for each. Weighted: the weighted parameters split a
/// budget and do not multiply it. The budget is pointsPerCombination for each distinct live
/// combination (one that can hold a problem) of the parameters that have no weights, so
/// declaring a weight on an enum such as layout, scale or window reshapes the corpus without
/// growing it by the number of their values. It is divided by size, the units placed as
/// apportion places them, so each weighted value holds its declared share of the first pass. No
/// combination is owed a point: one whose share rounds to nothing is left for the growth, the
/// regime floors and the quotas to reach. What is owed is every declared value that some live
/// combination holds (coverEveryValue): the first pass is the budget, or more only when there
/// are more uncovered values than points. A combination the declared constraints make empty
/// gets none.
inline std::vector<int64_t> firstPassTargets(const OperationMetadata& metadata,
                                             const ExplorationRequest& request,
                                             const std::vector<ProblemPoint>& combinations,
                                             const std::vector<double>& sizes)
{
    std::vector<int64_t> targets(combinations.size(), request.pointsPerCombination);
    if(!anyWeighted(metadata.parameters))
    {
        return targets;
    }

    std::vector<std::string> unweighted;
    for(const auto& parameter : metadata.parameters)
    {
        if(parameter.weights.empty() && !parameter.enumerable().empty())
        {
            unweighted.push_back(parameter.name);
        }
    }
    std::set<ProblemPoint> distinct;
    std::vector<uint64_t> keys(combinations.size());
    for(size_t index = 0; index < combinations.size(); ++index)
    {
        keys[index] = tieKey(request.seed, index);
        if(!(sizes[index] > 0.0))
        {
            continue;
        }
        ProblemPoint projected;
        for(const auto& name : unweighted)
        {
            const auto found = combinations[index].find(name);
            if(found != combinations[index].end())
            {
                projected.insert(*found);
            }
        }
        distinct.insert(std::move(projected));
    }

    const auto budget = request.pointsPerCombination
                        * std::max<int64_t>(1, static_cast<int64_t>(distinct.size()));

    targets = apportion(budget, sizes, keys);
    coverEveryValue(metadata, combinations, sizes, keys, static_cast<double>(budget), targets);
    return targets;
}

/// How a growth round splits the @p need still missing over the parts that can supply it,
/// whose sizes are @p sizes. Unweighted, equally, each part's share rounded up; weighted, by
/// size as apportion splits it, so the shares sum to the need instead of giving every small
/// part a whole point a round. @p keys are the tie-break of apportion.
inline std::vector<int64_t> splitNeed(int64_t need,
                                      bool weighted,
                                      const std::vector<double>& sizes,
                                      const std::vector<uint64_t>& keys)
{
    if(weighted)
    {
        return apportion(need, sizes, keys);
    }
    const auto parts = static_cast<int64_t>(sizes.size());
    return std::vector<int64_t>(sizes.size(), parts == 0 ? 0 : (need + parts - 1) / parts);
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
    size_t constraintEmpty = 0;
    const auto combinations = detail::categoricalCombinations(
        metadata.parameters,
        request.maxCombinations,
        totalCombinations,
        [&metadata](const ProblemPoint& combination) {
            return !detail::excludedByCategoricals(metadata, combination);
        },
        request.keep,
        &constraintEmpty);

    corpus.categoricalTotal = totalCombinations;
    corpus.categoricalConstraintEmpty = constraintEmpty;
    // The sample holds only combinations that can hold a problem, but for the value cover's
    // swapped-in ones, which the constraints may empty: count the live ones it has.
    const auto sampled = static_cast<size_t>(std::count_if(
        combinations.begin(), combinations.end(), [&metadata](const ProblemPoint& combination) {
            return !detail::excludedByCategoricals(metadata, combination);
        }));
    const auto holders = totalCombinations - constraintEmpty;
    corpus.categoricalNotExplored = holders > sampled ? holders - sampled : 0;
    if(corpus.categoricalNotExplored > 0)
    {
        auto line = std::to_string(corpus.categoricalNotExplored) + " of " + std::to_string(holders)
                    + " categorical combinations that can hold a problem not explored "
                      "(maxCombinations bound)";
        if(constraintEmpty > 0)
        {
            line += "; the other " + std::to_string(constraintEmpty) + " of "
                    + std::to_string(totalCombinations)
                    + " hold none under the declared constraints";
        }
        corpus.skippedCombinations.push_back(std::move(line));
    }
    else if(totalCombinations > 0 && constraintEmpty == totalCombinations)
    {
        corpus.skippedCombinations.push_back(
            "every one of the " + std::to_string(totalCombinations)
            + " categorical combinations is empty under the declared constraints, so no problem "
              "can be generated");
    }

    // Per-combination state kept for the growth pass. The memo makes growing affordable: a
    // longer walk from the same seed retraces the shorter one, answered from here.
    std::vector<FeasibleSetRequest> searches(combinations.size());
    // How large each combination is relative to the average (Parameter::weights); 1 for every
    // combination that can hold a problem when nothing is weighted, 0 for one that cannot, and
    // then the counts below are the unweighted ones.
    const bool weighted = detail::anyWeighted(metadata.parameters);
    const auto sizes = detail::combinationSizes(metadata, combinations);
    // The first pass's per-combination problem count: pointsPerCombination for each, or, when
    // weighted, pointsPerCombination for each distinct unweighted combination split by size,
    // with every declared value drawn at least once. A live combination left with none is
    // untried: the growth treats it as open, not as one that found nothing.
    const auto firstPass = detail::firstPassTargets(metadata, request, combinations, sizes);
    std::vector<bool> untried(combinations.size(), false);
    for(size_t i = 0; i < untried.size(); ++i)
    {
        untried[i] = sizes[i] > 0.0 && firstPass[i] == 0;
    }
    // The combinations that can hold a problem, weighted or not.
    const auto live = std::max<std::ptrdiff_t>(
        1, std::count_if(sizes.begin(), sizes.end(), [](double size) { return size > 0.0; }));
    const auto pointsFor = [&firstPass](size_t index) { return firstPass[index]; };
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

    // The skeleton depends on the numeric window only, which no categorical value changes.
    size_t skeletonTotal = 0;
    const auto skeleton = numericWindow.empty() ? std::vector<Shape>{}
                                                : detail::regimeSkeleton(metadata,
                                                                         numericWindow,
                                                                         corpus.numericParameters,
                                                                         request.maxSkeleton,
                                                                         skeletonTotal);
    if(skeletonTotal > skeleton.size())
    {
        corpus.skippedCombinations.push_back(
            std::to_string(skeletonTotal - skeleton.size()) + " of " + std::to_string(skeletonTotal)
            + " declared regime combinations not tried (maxSkeleton bound)");
    }

    for(size_t index = 0; index < combinations.size(); ++index)
    {
        const auto& categorical = combinations[index];

        CombinationResult result;
        result.categorical = categorical;

        if(numericWindow.empty())
        {
            // All-categorical operation: one problem per combination, through the same
            // constraint and held-out checks as a searched point.
            if(!detail::satisfiesConstraints(metadata, categorical))
            {
                ++corpus.constraintRejections;
            }
            else if(!isHeld(categorical))
            {
                ++corpus.constraintAdmissions;
                if(admits(categorical))
                {
                    result.problems.push_back(categorical);
                }
            }
            corpus.combinations.push_back(std::move(result));
            continue;
        }

        const auto oracle = oracleFor(index);

        FeasibleSetRequest search;
        search.seeds = skeleton;
        search.dimensions = numericWindow;
        search.targetCount = pointsFor(index);
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
            // Problems per combination the mixture shares apply to.
            auto total = static_cast<double>(pointsFor(index));
            if(request.anchorScaling && request.corpusTarget > 0)
            {
                total = std::max(total,
                                 static_cast<double>(request.corpusTarget) * sizes[index]
                                     / static_cast<double>(live));
            }
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

        if(!metadata.archetypes.empty() && anchored.empty() && pointsFor(index) > 0)
        {
            // Reported so it is not mistaken for an operation with no declared workloads.
            corpus.skippedCombinations.push_back("no declared archetype was admitted for "
                                                 + detail::describe(categorical)
                                                 + "; that combination is exploration only");
        }

        // The search fills whatever the anchored draws did not.
        search.targetCount
            = std::max<int64_t>(0, pointsFor(index) - static_cast<int64_t>(result.problems.size()));

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
        const auto spentBefore = std::count(spent.begin(), spent.end(), true);
        const auto need = request.corpusTarget - before;
        if(need <= 0)
        {
            break;
        }

        std::vector<size_t> growable;
        for(size_t index = 0; index < corpus.combinations.size(); ++index)
        {
            if(!spent[index]
               && (untried[index] || !corpus.combinations[index].problems.empty()
                   || heldFound[index] > 0))
            {
                growable.push_back(index);
            }
        }
        if(growable.empty())
        {
            break;
        }

        // Each combination's share of what is missing.
        std::vector<double> growableSizes;
        std::vector<uint64_t> growableKeys;
        for(const auto index : growable)
        {
            growableSizes.push_back(sizes[index]);
            growableKeys.push_back(detail::tieKey(request.seed, index));
        }
        const auto shares = detail::splitNeed(need, weighted, growableSizes, growableKeys);
        for(size_t slot = 0; slot < growable.size(); ++slot)
        {
            const auto index = growable[slot];
            const auto share = shares[slot];
            if(share == 0)
            {
                continue;
            }
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
            auto found = [&]() {
                if(!untried[index] || request.probeBudget >= search.oracleBudget)
                {
                    return buildFeasibleShapeSet(oracle, search);
                }
                // Never searched: probe first, as the first pass does, so a combination the
                // engine declines costs the probe, not the whole budget.
                auto probe = search;
                probe.oracleBudget = request.probeBudget;
                auto probed = buildFeasibleShapeSet(oracle, probe);
                return probed.stats.distinct == 0 ? probed : buildFeasibleShapeSet(oracle, search);
            }();
            if(untried[index] && found.stats.distinct == 0)
            {
                // Declined, like a combination the first pass found nothing in: not grown.
                result.stats = found.stats;
                result.saturated = true;
                spent[index] = true;
                continue;
            }
            untried[index] = false;
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

        // Only the combinations that got a share were searched, and a small need leaves most
        // without one. A round that added no point but spent a combination (its search proved
        // saturated or capped) has not shown the others empty: the next splits the need over
        // those still open. Every round adds a point or spends one, so this ends.
        if(supplied() <= before && std::count(spent.begin(), spent.end(), true) == spentBefore)
        {
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
