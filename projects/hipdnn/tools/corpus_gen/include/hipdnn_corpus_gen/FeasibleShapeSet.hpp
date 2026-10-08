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
/// The region is known only through a whole-shape oracle, so the search walks inside it,
/// in log space, from restarted footholds.
namespace hipdnn_corpus_gen
{

/// One value per declared parameter, in declaration order.
using Shape = std::vector<int64_t>;

/// Whether the engine accepts this shape (in production: build the graph, call `is_applicable`).
using ShapeOracle = std::function<bool(const Shape&)>;

/// A parameter's search window, not a claim about the engine. `high` is the memory ceiling;
/// `low` defaults to 1 because degenerate shapes (e.g. `M = 1` decode) are real workloads.
struct ShapeDimension
{
    std::string name;
    int64_t low = 1;
    int64_t high = 1;
};

/// What the search reached, and what it did not.
struct FeasibleSetStats
{
    /// Oracle calls; each is a graph build plus a predicate call.
    int64_t oracleCalls = 0;

    /// Draws made seeking a foothold, and how many were accepted.
    int64_t seedAttempts = 0;
    int64_t seedsFound = 0;

    /// Footholds whose walk never moved. Evidence of isolated components, not a count.
    int64_t isolatedStarts = 0;

    /// Walk steps accepted, before selection.
    int64_t accepted = 0;

    /// Cells the corpus is partitioned into, and how many hold a shape (coverage = ratio).
    int64_t cells = 0;
    int64_t cellsOccupied = 0;

    /// Distinct feasible points reached; the corpus is selected from these.
    int64_t distinct = 0;

    /// Target count not reached. MUST NOT be read as a small region: the budget may have run out.
    bool budgetExhausted = false;
};

struct FeasibleShapeSet
{
    std::vector<Shape> shapes;
    FeasibleSetStats stats;
};

/// Search inputs and limits.
struct FeasibleSetRequest
{
    std::vector<ShapeDimension> dimensions;

    /// Shapes wanted; also the number of cells.
    int64_t targetCount = 100;

    /// Ceiling on oracle calls; the feasible fraction is unknown up front, so cost is bounded.
    int64_t oracleBudget = 100000;

    /// Independent footholds. Too few on a disconnected region silently samples one island.
    int64_t restarts = 8;

    /// Steps attempted per foothold before giving up on it.
    int64_t stepsPerStart = 400;

    /// RNG seed, for reproducibility (§5.8).
    uint64_t seed = 0;

    /// Shapes tried first, and walked from when accepted (e.g. regime buckets, recorded
    /// workloads). Coupled parameters make random draws almost never feasible. Seeds still go
    /// through the oracle.
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
    // Clamp in log space: exp of a large exponent overflows, and llround(inf) is undefined.
    const double ceiling = toLog(high);
    if(value >= ceiling)
    {
        return high;
    }
    const auto rounded = static_cast<int64_t>(std::llround(std::exp(value)));
    return std::clamp(rounded, low, high);
}

/// Log space cannot represent zero, yet some parameters (e.g. padding) start there, so the
/// search runs over `value - low + 1`.
inline double toLogFrom(int64_t value, int64_t low)
{
    return toLog(value - low + 1);
}

inline int64_t fromLogFrom(double value, int64_t low, int64_t high)
{
    const auto offset = fromLog(value, 1, high - low + 1);
    return std::clamp(offset + low - 1, low, high);
}

/// Log-uniform draw across the box; a uniform draw would almost never visit small shapes.
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

/// Squared distance in log space, where spread is judged.
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

/// Squared log-space distance from @p shape to the nearest member of @p known; max double if
/// @p known is empty.
inline double distanceToSet(const Shape& shape, const std::vector<Shape>& known)
{
    double nearest = std::numeric_limits<double>::max();
    for(const auto& other : known)
    {
        nearest = std::min(nearest, logDistanceSquared(shape, other));
    }
    return nearest;
}

/// Squared novelty threshold: a quarter of the box's log-space diagonal, so it is scale-free.
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

/// Up to @p count well-spread cell centres (greedy k-centre) over the reached points, not the
/// declared box: the feasible set is usually a thin slice of it, and off-slice cells never fill.
inline std::vector<std::vector<double>>
    buildCentroids(const std::vector<std::vector<double>>& observed, size_t count)
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

/// One shape per cell, kept by proximity to the cell's centre. Unlike post-hoc thinning, a
/// dominant seed cluster fills only its own cells, and coverage is measurable.
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

    /// Places @p shape in its nearest cell; returns true if that cell was empty.
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
        return static_cast<size_t>(std::count_if(
            _occupants.begin(), _occupants.end(), [](const auto& o) { return o.has_value(); }));
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
/// The chord interval shrinks toward @p current on each refusal, so non-convex regions work.
inline std::optional<Shape> hitAndRunStep(const ShapeOracle& oracle,
                                          const Shape& current,
                                          const std::vector<ShapeDimension>& dimensions,
                                          std::mt19937_64& rng,
                                          int64_t maxShrinks,
                                          int64_t& calls,
                                          int64_t budget)
{
    // Half axis-aligned, half random: random directions rarely stay inside per-coordinate
    // constraints (e.g. multiples of 8); axis steps cannot cross a diagonal ridge.
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
            candidate[d] = fromLogFrom(
                position[d] + (t * direction[d]), dimensions[d].low, dimensions[d].high);
        }

        if(candidate == current)
        {
            // Rounded back to the start point: shrink without spending an oracle call.
            (t < 0.0 ? low : high) = t;
            continue;
        }

        ++calls;
        if(oracle(candidate))
        {
            return candidate;
        }
        // Refused: pull that side of the interval in to the refused point.
        (t < 0.0 ? low : high) = t;
    }
    return std::nullopt;
}

} // namespace detail

/// @brief Builds a spread set of shapes @p oracle accepts.
///
/// Finds footholds (caller seeds, then log-uniform draws), walks from each by hit-and-run, then
/// keeps one shape per cell over the points reached. Neither uniformity nor discovery of every
/// disconnected component is guaranteed; see @ref FeasibleSetStats.
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

    // Every feasible point reached (capped); cells are placed over these.
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

    // Deduplicate so revisits don't pull centres toward where the walk lingered.
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
    result.stats.budgetExhausted = static_cast<int64_t>(result.shapes.size()) < request.targetCount;
    return result;
}

} // namespace hipdnn_corpus_gen
