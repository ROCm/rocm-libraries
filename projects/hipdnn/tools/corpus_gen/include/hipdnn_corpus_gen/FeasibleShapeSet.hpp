// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
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

    /// Cells the corpus is partitioned into, and how many hold a problem. Coverage is
    /// occupied over total -- a measurement rather than a claim, which post-hoc thinning
    /// could not provide.
    int64_t cells = 0;
    int64_t cellsOccupied = 0;

    /// Distinct feasible points the search reached, before selection. The corpus is chosen
    /// from these, so it can never exceed them -- and when it falls short, this says whether
    /// the search or the selection was the limit.
    int64_t distinct = 0;

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

    /// Shapes to try before drawing any, and to walk from when accepted.
    ///
    /// Rejection sampling finds a region in proportion to its measure, and for an operation
    /// whose parameters constrain each other that measure is nearly zero: independent draws
    /// over a convolution's thirteen parameters produce a filter larger than its input almost
    /// every time, and the search maps the frontend's validator instead of the engine.
    ///
    /// A caller that knows the structure -- declared regime buckets, recorded workload shapes
    /// -- supplies it here. This is not a shortcut around the search: seeds are put to the
    /// oracle like anything else, and the walk still discovers what no list contains.
    std::vector<Shape> seeds;
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

/// Log space cannot represent zero, and some parameters legitimately start there. A
/// convolution's padding is the common case: an engine that requires unpadded input --
/// hip-kernel-provider's conv pack does -- accepts nothing a search floored at 1 can propose,
/// and reports as serving no convolutions at all.
///
/// So the search runs over `value - low + 1`, which is at least 1 for any floor, and
/// translates at the boundary. A dimension with `low = 1` is unchanged, which is most of them.
inline double toLogFrom(int64_t value, int64_t low)
{
    return toLog(value - low + 1);
}

inline int64_t fromLogFrom(double value, int64_t low, int64_t high)
{
    const auto offset = fromLog(value, 1, high - low + 1);
    return std::clamp(offset + low - 1, low, high);
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
        std::uniform_real_distribution<double> distribution(
            toLog(1), toLogFrom(dimension.high, dimension.low));
        shape.push_back(fromLogFrom(distribution(rng), dimension.low, dimension.high));
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

/// Well-spread cell centres over the points the search actually reached.
///
/// Not over the declared box, which was the first attempt and is wrong for the same reason a
/// uniform prior is wrong here: the feasible set is usually a thin slice of the box. An engine
/// requiring unit stride, unit dilation and no padding pins six of a convolution's thirteen
/// parameters, so the region is seven-dimensional and every cell placed off that slice is
/// unreachable by construction. Measured: 184 occupied of 10000 box cells, with the search
/// having found more than a million feasible points to put in them.
///
/// Tessellating the observed set instead means every cell can be filled and the corpus spreads
/// over the region that exists. How much of the *declared* space turned out to be reachable is
/// a separate question, and is reported separately rather than folded into coverage.
///
/// k-centre over the observations, which is the practical stand-in for the centroidal Voronoi
/// tessellation CVT-MAP-Elites uses to avoid a grid exponential in the parameter count.
inline std::vector<std::vector<double>> buildCentroids(
    const std::vector<std::vector<double>>& observed, size_t count)
{
    std::vector<std::vector<double>> centroids;
    if(observed.empty())
    {
        return centroids;
    }

    centroids.push_back(observed.front());
    std::vector<double> nearest(observed.size(), std::numeric_limits<double>::max());

    while(centroids.size() < count && centroids.size() < observed.size())
    {
        size_t best = 0;
        double bestDistance = -1.0;
        for(size_t i = 0; i < observed.size(); ++i)
        {
            double distance = 0.0;
            for(size_t d = 0; d < observed[i].size(); ++d)
            {
                const double delta = observed[i][d] - centroids.back()[d];
                distance += delta * delta;
            }
            nearest[i] = std::min(nearest[i], distance);
            if(nearest[i] > bestDistance)
            {
                bestDistance = nearest[i];
                best = i;
            }
        }
        if(bestDistance <= 0.0)
        {
            break; // Every remaining observation coincides with a centre already chosen.
        }
        centroids.push_back(observed[best]);
    }
    return centroids;
}

/// One problem per cell, kept by proximity to the cell's centre.
///
/// This replaces selecting a spread subset after the fact. Post-hoc selection can only choose
/// among what the search produced, so when a declared skeleton dominates the accepted set it
/// faithfully returns skeleton points and the corpus contains nothing the declaration did not
/// already name. An archive cannot collapse that way: a seed occupies its own cell and no
/// more, and coverage becomes a number -- occupied cells over total -- rather than a claim.
class CellArchive
{
public:
    CellArchive(std::vector<std::vector<double>> centroids,
                const std::vector<ShapeDimension>& dimensions)
        : _centroids(std::move(centroids))
        , _dimensions(dimensions)
        , _occupants(_centroids.size())
        , _distances(_centroids.size(), std::numeric_limits<double>::max())
    {
    }

    /// Places @p shape in its cell. Returns true when it took an empty cell, which is the
    /// signal that the search reached somewhere new.
    bool insert(const Shape& shape)
    {
        std::vector<double> position;
        position.reserve(shape.size());
        for(size_t d = 0; d < shape.size(); ++d)
        {
            position.push_back(toLogFrom(shape[d], _dimensions[d].low));
        }

        size_t cell = 0;
        double best = std::numeric_limits<double>::max();
        for(size_t c = 0; c < _centroids.size(); ++c)
        {
            double distance = 0.0;
            for(size_t d = 0; d < position.size(); ++d)
            {
                const double delta = position[d] - _centroids[c][d];
                distance += delta * delta;
            }
            if(distance < best)
            {
                best = distance;
                cell = c;
            }
        }

        const bool wasEmpty = !_occupants[cell].has_value();
        if(wasEmpty || best < _distances[cell])
        {
            _occupants[cell] = shape;
            _distances[cell] = best;
        }
        return wasEmpty;
    }

    std::vector<Shape> contents() const
    {
        std::vector<Shape> shapes;
        for(const auto& occupant : _occupants)
        {
            if(occupant.has_value())
            {
                shapes.push_back(*occupant);
            }
        }
        return shapes;
    }

    size_t occupied() const
    {
        return static_cast<size_t>(
            std::count_if(_occupants.begin(), _occupants.end(), [](const auto& o) {
                return o.has_value();
            }));
    }

    size_t cells() const
    {
        return _centroids.size();
    }

private:
    std::vector<std::vector<double>> _centroids;
    const std::vector<ShapeDimension>& _dimensions;
    std::vector<std::optional<Shape>> _occupants;
    std::vector<double> _distances;
};

/// One discrete hit-and-run step from @p current (Baumert et al., *Operations Research* 57(3)).
///
/// A uniform random direction, then a uniform draw along the chord that direction cuts through
/// the box -- shrinking the interval toward the current point on each refusal, which is what
/// makes the step legal for a region that is neither convex nor connected.
///
/// This replaces a coordinate-direction walk with an adaptive step size. That walk is the one
/// the literature singles out as failing to converge because it becomes trapped in isolated
/// regions, and its step size was a hand-rolled substitute for the chord this samples directly.
inline std::optional<Shape> hitAndRunStep(const ShapeOracle& oracle,
                                          const Shape& current,
                                          const std::vector<ShapeDimension>& dimensions,
                                          std::mt19937_64& rng,
                                          int64_t maxShrinks,
                                          int64_t& calls,
                                          int64_t budget)
{
    // Half the steps move along one axis, half in a random direction.
    //
    // Neither alone is enough. A full-dimensional direction changes every coordinate at once,
    // which is fatal for a region defined per coordinate -- a set admitting only multiples of
    // eight in each of three dimensions is left by almost every diagonal step, and the walk
    // stalls. A coordinate direction preserves the other coordinates and moves happily inside
    // such a region, but cannot cross a diagonal ridge. Coordinate Hit-and-Run exists as a
    // named variant for the first reason; keeping both is what covers the second.
    std::vector<double> direction(dimensions.size(), 0.0);
    std::bernoulli_distribution axisAligned(0.5);
    if(axisAligned(rng))
    {
        std::uniform_int_distribution<size_t> pick(0, dimensions.size() - 1);
        std::bernoulli_distribution sign(0.5);
        direction[pick(rng)] = sign(rng) ? 1.0 : -1.0;
    }
    else
    {
        std::normal_distribution<double> gaussian(0.0, 1.0);
        double norm = 0.0;
        for(auto& component : direction)
        {
            component = gaussian(rng);
            norm += component * component;
        }
        norm = std::sqrt(norm);
        if(norm <= 0.0)
        {
            return std::nullopt;
        }
        for(auto& component : direction)
        {
            component /= norm;
        }
    }

    // The chord's extent within the box, in log space.
    double low = -std::numeric_limits<double>::max();
    double high = std::numeric_limits<double>::max();
    std::vector<double> position(dimensions.size());
    for(size_t d = 0; d < dimensions.size(); ++d)
    {
        position[d] = toLogFrom(current[d], dimensions[d].low);
        const double ceiling = toLogFrom(dimensions[d].high, dimensions[d].low);
        if(std::abs(direction[d]) < 1e-12)
        {
            continue;
        }
        const double toFloor = (0.0 - position[d]) / direction[d];
        const double toCeiling = (ceiling - position[d]) / direction[d];
        low = std::max(low, std::min(toFloor, toCeiling));
        high = std::min(high, std::max(toFloor, toCeiling));
    }
    if(!(low < high))
    {
        return std::nullopt;
    }

    for(int64_t shrink = 0; shrink < maxShrinks && calls < budget; ++shrink)
    {
        std::uniform_real_distribution<double> along(low, high);
        const double t = along(rng);

        Shape candidate(current.size());
        for(size_t d = 0; d < dimensions.size(); ++d)
        {
            candidate[d] = fromLogFrom(position[d] + (t * direction[d]),
                                       dimensions[d].low,
                                       dimensions[d].high);
        }

        if(candidate == current)
        {
            // The draw landed back on the lattice point it started from; shrinking here would
            // narrow the interval without having learned anything.
            (t < 0.0 ? low : high) = t;
            continue;
        }

        ++calls;
        if(oracle(candidate))
        {
            return candidate;
        }
        // Refused: pull that side of the interval in to the refused point, which is the
        // accept/reject a disconnected region makes unavoidable.
        (t < 0.0 ? low : high) = t;
    }
    return std::nullopt;
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

    std::mt19937_64 rng(request.seed);

    const auto ask = [&](const Shape& shape) {
        ++result.stats.oracleCalls;
        return oracle(shape);
    };

    // Every feasible point the search reaches. Held because the cells are placed over these
    // rather than over the declared box -- the region is typically a thin slice of the box,
    // and cells placed off it can never be filled.
    std::vector<Shape> observed;
    const size_t observationCap = static_cast<size_t>(request.targetCount) * 200;

    const auto record = [&](const Shape& shape) {
        if(observed.size() < observationCap)
        {
            observed.push_back(shape);
        }
    };

    std::vector<Shape> footholds;
    for(const auto& seed : request.seeds)
    {
        if(result.stats.oracleCalls >= request.oracleBudget
           || seed.size() != request.dimensions.size())
        {
            continue;
        }
        ++result.stats.seedAttempts;
        if(ask(seed))
        {
            ++result.stats.seedsFound;
            record(seed);
            footholds.push_back(seed);
        }
    }

    size_t nextFoothold = 0;
    for(int64_t restart = 0; restart < request.restarts; ++restart)
    {
        if(result.stats.oracleCalls >= request.oracleBudget)
        {
            break;
        }

        Shape current;
        bool started = false;

        if(nextFoothold < footholds.size())
        {
            current = footholds[nextFoothold++];
            started = true;
        }
        else
        {
            const auto allowance
                = std::max<int64_t>(1, request.oracleBudget / (request.restarts * 2));
            for(int64_t attempt = 0;
                attempt < allowance && result.stats.oracleCalls < request.oracleBudget;
                ++attempt)
            {
                ++result.stats.seedAttempts;
                Shape candidate = detail::drawLogUniform(request.dimensions, rng);
                if(ask(candidate))
                {
                    ++result.stats.seedsFound;
                    record(candidate);
                    current = std::move(candidate);
                    started = true;
                    break;
                }
            }
        }

        if(!started)
        {
            continue;
        }

        int64_t moved = 0;
        for(int64_t step = 0; step < request.stepsPerStart; ++step)
        {
            if(result.stats.oracleCalls >= request.oracleBudget
               || observed.size() >= observationCap)
            {
                break;
            }

            const auto next = detail::hitAndRunStep(oracle,
                                                    current,
                                                    request.dimensions,
                                                    rng,
                                                    /*maxShrinks=*/16,
                                                    result.stats.oracleCalls,
                                                    request.oracleBudget);
            if(!next.has_value())
            {
                continue;
            }
            current = *next;
            ++result.stats.accepted;
            record(current);
            ++moved;
        }

        if(moved == 0)
        {
            ++result.stats.isolatedStarts;
        }
    }

    // Deduplicate before tessellating: a walk revisits points, and duplicates would pull
    // centres toward wherever it lingered rather than toward where it reached.
    std::sort(observed.begin(), observed.end());
    observed.erase(std::unique(observed.begin(), observed.end()), observed.end());
    result.stats.distinct = static_cast<int64_t>(observed.size());

    if(observed.empty())
    {
        result.stats.budgetExhausted = true;
        return result;
    }

    std::vector<std::vector<double>> positions;
    positions.reserve(observed.size());
    for(const auto& shape : observed)
    {
        std::vector<double> point;
        point.reserve(shape.size());
        for(size_t d = 0; d < shape.size(); ++d)
        {
            point.push_back(detail::toLogFrom(shape[d], request.dimensions[d].low));
        }
        positions.push_back(std::move(point));
    }

    const auto centroids
        = detail::buildCentroids(positions, static_cast<size_t>(request.targetCount));

    detail::CellArchive archive(centroids, request.dimensions);
    for(const auto& shape : observed)
    {
        archive.insert(shape);
    }

    result.shapes = archive.contents();
    result.stats.cells = static_cast<int64_t>(archive.cells());
    result.stats.cellsOccupied = static_cast<int64_t>(archive.occupied());
    result.stats.budgetExhausted
        = static_cast<int64_t>(result.shapes.size()) < request.targetCount;
    return result;
}

} // namespace hipdnn_corpus_gen
