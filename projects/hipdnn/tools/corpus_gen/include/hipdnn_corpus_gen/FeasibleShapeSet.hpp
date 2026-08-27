// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

/// @file FeasibleShapeSet.hpp
/// @brief Builds a spread set of shapes an engine accepts (RFC 0019.13 §5.3).
///
/// The problem in one line: draw a well-spread sample from a set that nobody has described.
/// An operation parameter carries no declared range, the engine advertises none, and the only
/// thing that knows is `is_applicable` answering one whole shape at a time.
///
/// Three properties of that set defeat the obvious approaches:
///
///  - **It is high-dimensional.** Convolution has a dozen parameters. Rejection sampling from
///    a box fails here for the usual reason: the feasible fraction of a box falls off roughly
///    geometrically in the number of dimensions, so nearly every draw is wasted.
///  - **The parameters are coupled.** Padding, stride, dilation and groups do not have bounds
///    of their own; together they decide whether a shape is degenerate. A bound found for one
///    dimension with the others pinned describes a slice, not the space, so probing each
///    dimension separately and taking the product overstates the region.
///  - **It has holes, and may be disconnected.** Alignment and divisibility rules exclude
///    interior values; an engine with two kernel families can accept two separated islands.
///
/// The approach that survives all three is to stop describing the region and move around
/// inside it. Find a feasible point, then take steps, keeping the ones the oracle accepts.
/// Coupling is handled because every step is tested as a whole shape. Dimensionality is
/// handled because the walk does not have to find the region by luck more than once. What a
/// walk cannot do is cross a gap, so it is restarted from independent draws, and what it
/// reaches is reported rather than assumed to be everything.
///
/// Everything is done on a logarithmic scale. These dimensions span orders of magnitude, and
/// a step of +1 means something entirely different at 8 than at 8192.
namespace hipdnn_corpus_gen
{

/// One value per declared parameter, in the order the caller declared them.
using Shape = std::vector<int64_t>;

/// Whether the engine accepts this shape. In production: build the graph and call
/// `is_applicable`. Injected so the search can be tested against regions already known, and
/// because building a graph is the caller's business, not this file's.
using ShapeOracle = std::function<bool(const Shape&)>;

/// A parameter and the range worth asking about.
///
/// The bounds are a search window, not a claim about the engine. `high` is what the memory
/// ceiling permits -- nothing above it can be benchmarked, so nothing above it is worth
/// proposing. `low` is 1 by default because degenerate shapes are real workloads: `M = 1` is
/// single-token decode, a regime in its own right.
struct ShapeDimension
{
    std::string name;
    int64_t low = 1;
    int64_t high = 1;
};

/// What the search reached, and what it did not.
struct FeasibleSetStats
{
    /// Oracle calls. The cost that matters: each is a graph build and a predicate call.
    int64_t oracleCalls = 0;

    /// Draws made while looking for a foothold, and how many landed. Their ratio is the
    /// feasible fraction of the search box, and the reason the walk exists -- a low number
    /// here is exactly the cost that rejection sampling alone would pay for every shape.
    int64_t seedAttempts = 0;
    int64_t seedsFound = 0;

    /// Independent footholds whose walks never met. A lower bound on the number of
    /// disconnected components, not a count: two walks that stayed apart may still have been
    /// in one region and simply not have crossed.
    int64_t isolatedStarts = 0;

    /// Distinct shapes accepted before thinning. The gap between this and the returned set is
    /// how much spread cost.
    int64_t accepted = 0;

    /// True when the target count was not reached. The caller MUST NOT read a short set as a
    /// small region: it may equally be a budget that ran out.
    bool budgetExhausted = false;
};

struct FeasibleShapeSet
{
    std::vector<Shape> shapes;
    FeasibleSetStats stats;
};

/// How hard to look, and for how much.
struct FeasibleSetRequest
{
    std::vector<ShapeDimension> dimensions;

    /// Shapes wanted. The search stops early when it has them.
    int64_t targetCount = 100;

    /// Ceiling on oracle calls. Present because the feasible fraction of the box is unknown
    /// before the search starts, so "how long will this take" cannot be answered in advance --
    /// only bounded.
    int64_t oracleBudget = 100000;

    /// Independent footholds to look for. More restarts find more components and cost more
    /// rejection draws; one restart on a disconnected region silently samples one island.
    int64_t restarts = 8;

    /// Steps attempted per foothold before giving up on it.
    int64_t stepsPerStart = 400;

    /// Reproducibility, per §5.8. Randomised against the region's structure, not across runs.
    uint64_t seed = 0;
};

namespace detail
{

inline double toLog(int64_t value)
{
    return std::log(static_cast<double>(value < 1 ? 1 : value));
}

inline int64_t fromLog(double value, int64_t low, int64_t high)
{
    // Clamped in log space before exponentiating. A walk that steps upward repeatedly can
    // push the exponent past what a double can represent, and llround of an infinity is
    // undefined -- clamping after the fact would be too late.
    const double ceiling = toLog(high);
    if(value >= ceiling)
    {
        return high;
    }
    const auto rounded = static_cast<int64_t>(std::llround(std::exp(value)));
    return std::clamp(rounded, low, high);
}

/// Log-uniform draw across the box. Log-uniform rather than uniform because performance
/// regimes scale multiplicatively: a uniform draw over [1, 2^20] puts almost every sample in
/// the top octave and never visits the small shapes that are their own regime.
inline Shape drawLogUniform(const std::vector<ShapeDimension>& dimensions, std::mt19937_64& rng)
{
    Shape shape;
    shape.reserve(dimensions.size());
    for(const auto& dimension : dimensions)
    {
        std::uniform_real_distribution<double> distribution(toLog(dimension.low),
                                                            toLog(dimension.high));
        shape.push_back(fromLog(distribution(rng), dimension.low, dimension.high));
    }
    return shape;
}

/// Squared distance in log space, which is the space spread is judged in for the same reason
/// sampling happens there: 64 and 128 are as far apart as 4096 and 8192.
inline double logDistanceSquared(const Shape& a, const Shape& b)
{
    double total = 0.0;
    for(size_t i = 0; i < a.size(); ++i)
    {
        const double delta = toLog(a[i]) - toLog(b[i]);
        total += delta * delta;
    }
    return total;
}

/// Log-space distance from @p shape to the nearest member of @p known. Infinite for an empty
/// set, so the first foothold is always novel.
inline double distanceToSet(const Shape& shape, const std::vector<Shape>& known)
{
    double nearest = std::numeric_limits<double>::max();
    for(const auto& other : known)
    {
        nearest = std::min(nearest, logDistanceSquared(shape, other));
    }
    return nearest;
}

/// How far a draw must be from everything already found to count as unexplored territory
/// rather than more of what is known.
///
/// A quarter of the box's diagonal in log space. Scale-free by construction, so it means the
/// same thing for a two-parameter box as for a twelve-parameter one, and expressed as a
/// squared distance to match logDistanceSquared.
inline double noveltyRadiusSquared(const std::vector<ShapeDimension>& dimensions)
{
    double diagonal = 0.0;
    for(const auto& dimension : dimensions)
    {
        const double span = toLog(dimension.high) - toLog(dimension.low);
        diagonal += span * span;
    }
    return diagonal * 0.25 * 0.25;
}

/// Greedy farthest-point selection: repeatedly take the candidate furthest from everything
/// already chosen.
///
/// A walk's output is not a sample of the region -- consecutive steps are neighbours, so the
/// raw trace is clustered along whatever path it happened to take. Thinning by distance turns
/// that into coverage. Greedy rather than optimal because the optimal choice is combinatorial
/// and the greedy one is within a constant factor, which is far inside the noise of everything
/// else here.
inline std::vector<Shape> spreadSelect(const std::vector<Shape>& candidates, int64_t count)
{
    if(candidates.empty() || count <= 0)
    {
        return {};
    }
    if(static_cast<int64_t>(candidates.size()) <= count)
    {
        return candidates;
    }

    std::vector<Shape> chosen;
    chosen.reserve(static_cast<size_t>(count));
    std::vector<bool> taken(candidates.size(), false);

    // Seeded from the first accepted shape rather than an arbitrary one, so the selection is
    // reproducible for a given walk.
    chosen.push_back(candidates.front());
    taken[0] = true;

    std::vector<double> nearest(candidates.size(), std::numeric_limits<double>::max());
    while(static_cast<int64_t>(chosen.size()) < count)
    {
        size_t best = 0;
        double bestDistance = -1.0;
        for(size_t i = 0; i < candidates.size(); ++i)
        {
            if(taken[i])
            {
                continue;
            }
            nearest[i] = std::min(nearest[i], logDistanceSquared(candidates[i], chosen.back()));
            if(nearest[i] > bestDistance)
            {
                bestDistance = nearest[i];
                best = i;
            }
        }
        if(bestDistance < 0.0)
        {
            break;
        }
        taken[best] = true;
        chosen.push_back(candidates[best]);
    }
    return chosen;
}

} // namespace detail

/// @brief Builds a spread set of shapes @p oracle accepts.
///
/// Three phases, each answering a failure of the one before:
///
///  1. **Seed.** Log-uniform draws across the box until one is accepted. This is rejection
///     sampling, and it is used only to find footholds -- a handful of them -- rather than to
///     produce the corpus, because its cost per shape is what makes it unusable in many
///     dimensions.
///  2. **Walk.** From each foothold, multiply one randomly chosen dimension by a random
///     factor and keep the result if the oracle accepts it. The step size adapts: it shrinks
///     on rejection, so a walk against a boundary works its way along rather than hammering
///     it, and grows on acceptance, so an open region is crossed rather than crept through.
///     Every step is a whole shape put to the oracle, which is what makes coupled parameters
///     no harder than independent ones.
///  3. **Thin.** Farthest-point selection over everything accepted, because a walk's trace is
///     clustered and a corpus wants coverage.
///
/// What this does not do: guarantee uniformity over the feasible set, or find every component
/// of a disconnected one. Restarts are the mitigation and @ref FeasibleSetStats::isolatedStarts
/// is the evidence; neither is a proof.
inline FeasibleShapeSet buildFeasibleShapeSet(const ShapeOracle& oracle,
                                              const FeasibleSetRequest& request)
{
    FeasibleShapeSet result;
    if(request.dimensions.empty() || request.targetCount <= 0)
    {
        return result;
    }

    // Seed hits kept per restart. Small: enough to contribute the every-coordinate variation
    // a walk cannot, without letting a dense region's pool grow without bound.
    constexpr int64_t SEED_RETENTION_QUOTA = 8;

    std::mt19937_64 rng(request.seed);
    std::vector<Shape> accepted;
    std::uniform_int_distribution<size_t> pickDimension(0, request.dimensions.size() - 1);

    const auto ask = [&](const Shape& shape) {
        ++result.stats.oracleCalls;
        return oracle(shape);
    };

    for(int64_t restart = 0; restart < request.restarts; ++restart)
    {
        if(result.stats.oracleCalls >= request.oracleBudget)
        {
            break;
        }

        // Seed. Draws continue past the first hit until one lands in unexplored territory,
        // and every hit is kept.
        //
        // Stopping at the first hit would be cheaper and would be wrong for the job. Uniform
        // draws find a component in proportion to its measure, so a region in two pieces --
        // one of them a fortieth the size of the other -- hands the foothold to the larger
        // piece forty times in forty-one. Restarts then re-explore what is already known and
        // the small piece goes unfound however many are spent. Requiring novelty is what
        // makes a restart worth its budget.
        //
        // Two allowances, because the seed phase does two jobs that cost differently and
        // bounding them together makes each wrong.
        //
        // Reaching the first hit costs about one draw per feasible fraction of the box, and
        // that fraction is exactly what nobody knows before the search starts. So it gets the
        // generous share, and a box with no feasible region at all is what the bound is for.
        //
        // Hunting for novelty afterwards is bounded by what a corpus of this size can
        // resolve. After n draws turn up nothing unexplored, a component still undiscovered
        // occupies less than roughly 1/n of the box; at ten draws per requested shape that is
        // a component too small to contribute even one shape to what was asked for. Spending
        // past that buys resolution the caller did not request -- and in a region that is
        // simply one connected piece it would be spent in full, every restart, to prove what
        // the first draws already indicated.
        Shape current;
        bool seeded = false;
        double bestNovelty = -1.0;
        int64_t retained = 0;
        int64_t sinceFirstHit = 0;
        const double radius = detail::noveltyRadiusSquared(request.dimensions);
        const int64_t footholdAllowance
            = std::max<int64_t>(1, request.oracleBudget / (request.restarts * 2));
        const int64_t noveltyAllowance = std::max<int64_t>(64, request.targetCount * 10);
        for(int64_t attempt = 0; attempt < footholdAllowance; ++attempt)
        {
            if(result.stats.oracleCalls >= request.oracleBudget
               || (seeded && sinceFirstHit >= noveltyAllowance))
            {
                break;
            }
            if(seeded)
            {
                ++sinceFirstHit;
            }
            ++result.stats.seedAttempts;
            const Shape candidate = detail::drawLogUniform(request.dimensions, rng);
            if(!ask(candidate))
            {
                continue;
            }
            ++result.stats.seedsFound;

            // Kept whether or not it becomes the foothold. It is a feasible shape that has
            // already been paid for, and unlike a walk step it differs from its neighbours in
            // every coordinate -- which is most of what per-dimension coverage comes from.
            //
            // Capped, because retention is a side benefit of the search rather than its
            // purpose, and the novelty check is linear in what is already held. In a region
            // where most draws are accepted an uncapped pool would grow by the whole
            // allowance each restart and make the seed phase quadratic.
            const double novelty = detail::distanceToSet(candidate, accepted);
            if(retained < SEED_RETENTION_QUOTA)
            {
                accepted.push_back(candidate);
                ++retained;
            }

            if(novelty > bestNovelty)
            {
                bestNovelty = novelty;
                current = candidate;
                seeded = true;
            }
            if(novelty > radius)
            {
                break;
            }
        }
        if(!seeded)
        {
            continue;
        }

        // The foothold is kept whether or not the retention quota is spent. It is the most
        // novel thing the phase found, which in a region of unequal pieces is the only point
        // from the small one -- draws from the large piece fill the quota long before a rare
        // one lands, so a quota applied here would discard precisely what the hunt was for.
        accepted.push_back(current);

        // Territory the walk has not seen is worth walking even when enough shapes are
        // already held. Otherwise a component discovered late contributes the single shape
        // that found it, and reports as a speck rather than as the region it is.
        const bool novelFoothold = bestNovelty > radius;
        int64_t walkAccepted = 0;

        // Walk. One dimension at a time: a single-coordinate move is likelier to stay inside
        // a region shaped by per-dimension rules than a move in every coordinate at once,
        // which in high dimensions almost always leaves it.
        double stepScale = 2.0;
        for(int64_t step = 0; step < request.stepsPerStart; ++step)
        {
            if(result.stats.oracleCalls >= request.oracleBudget
               || (!novelFoothold
                   && static_cast<int64_t>(accepted.size()) >= request.targetCount * 4))
            {
                break;
            }

            const size_t dimension = pickDimension(rng);
            std::uniform_real_distribution<double> factor(1.0 / stepScale, stepScale);

            Shape candidate = current;
            candidate[dimension] = detail::fromLog(detail::toLog(current[dimension])
                                                       + std::log(factor(rng)),
                                                   request.dimensions[dimension].low,
                                                   request.dimensions[dimension].high);
            if(candidate == current)
            {
                continue;
            }

            if(ask(candidate))
            {
                current = candidate;
                accepted.push_back(current);
                ++walkAccepted;
                // Widen, but not without limit: an unbounded step degenerates into the
                // log-uniform draw the seed phase already does, and loses the walk's one
                // advantage of starting from somewhere feasible.
                stepScale = std::min(stepScale * 1.1, 8.0);
            }
            else
            {
                // Narrow toward 1.0, where every proposal is a small move. A walk pressed
                // against a boundary ends up sliding along it rather than repeatedly asking
                // to cross.
                stepScale = std::max(1.0 + ((stepScale - 1.0) * 0.7), 1.05);
            }
        }

        if(walkAccepted == 0)
        {
            // A foothold that went nowhere: either an isolated point or a walk that could not
            // get out of a corner. Either way it is not evidence of a region.
            ++result.stats.isolatedStarts;
        }
    }

    std::sort(accepted.begin(), accepted.end());
    accepted.erase(std::unique(accepted.begin(), accepted.end()), accepted.end());
    result.stats.accepted = static_cast<int64_t>(accepted.size());

    result.shapes = detail::spreadSelect(accepted, request.targetCount);
    result.stats.budgetExhausted
        = static_cast<int64_t>(result.shapes.size()) < request.targetCount;
    return result;
}

} // namespace hipdnn_corpus_gen
