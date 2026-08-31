// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestFeasibleShapeSet.cpp
 * @brief Covers shape-set generation against regions whose shape is known in advance.
 *
 * Each case defines a region, so what the search should find is arithmetic rather than
 * whatever it happened to produce. The cases are chosen to be the ones that defeat the
 * approaches this algorithm exists to replace: a region too sparse for rejection sampling, a
 * region whose parameters constrain each other, and a region in two pieces.
 */

#include <gtest/gtest.h>

#include <hipdnn_corpus_gen/FeasibleShapeSet.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <vector>

namespace hipdnn_corpus_gen
{
namespace
{

std::vector<ShapeDimension> matmulLikeDimensions(int64_t high = 8192)
{
    return {{"M", 1, high}, {"N", 1, high}, {"K", 1, high}};
}

FeasibleSetRequest requestFor(std::vector<ShapeDimension> dimensions, int64_t target = 60)
{
    FeasibleSetRequest request;
    request.dimensions = std::move(dimensions);
    request.targetCount = target;
    request.seed = 20260827;
    return request;
}

/// Distinct values seen in one dimension, as a crude read on whether a set spans its range
/// rather than huddling around wherever the walk started.
size_t distinctValues(const std::vector<Shape>& shapes, size_t dimension)
{
    std::set<int64_t> values;
    for(const auto& shape : shapes)
    {
        values.insert(shape[dimension]);
    }
    return values.size();
}

} // namespace

TEST(TestFeasibleShapeSet, EveryShapeReturnedIsAccepted)
{
    // The floor. A set containing one shape the engine would refuse is worse than an empty
    // one: it is benchmarked, fails, and the failure is attributed to the kernel.
    const auto region = [](const Shape& shape) {
        return shape[0] * shape[1] <= 1000000 && shape[2] % 8 == 0;
    };

    const auto result = buildFeasibleShapeSet(region, requestFor(matmulLikeDimensions()));

    ASSERT_FALSE(result.shapes.empty());
    for(const auto& shape : result.shapes)
    {
        EXPECT_TRUE(region(shape)) << "returned a shape the oracle rejects";
    }
}

TEST(TestFeasibleShapeSet, FindsARegionTooSparseForRejectionSampling)
{
    // Roughly one box point in 500 qualifies. Rejection sampling would need ~30000 draws for
    // 60 shapes; the walk pays that rate only until it has a foothold.
    const auto region = [](const Shape& shape) {
        return shape[0] % 8 == 0 && shape[1] % 8 == 0 && shape[2] % 8 == 0;
    };

    auto request = requestFor(matmulLikeDimensions());
    const auto result = buildFeasibleShapeSet(region, request);

    // The corpus is the occupied cells, so it is not required to reach the requested count:
    // this region covers about one box point in 512, and many cells contain no feasible point
    // at all. A cell that cannot be filled is a fact about the region, and reporting coverage
    // is more useful than topping the corpus up with duplicates of what was reachable.
    EXPECT_GT(result.stats.cellsOccupied, request.targetCount / 2)
        << "occupied " << result.stats.cellsOccupied << " of " << result.stats.cells;
    EXPECT_EQ(static_cast<int64_t>(result.shapes.size()), result.stats.cellsOccupied);

    // What the walk is for, stated as the cost it beats: rejection sampling pays about 512
    // oracle calls per shape in this region however many it wants.
    const double callsPerShape = static_cast<double>(result.stats.oracleCalls)
                                 / static_cast<double>(result.stats.accepted);
    EXPECT_LT(callsPerShape, 150.0) << "oracle calls " << result.stats.oracleCalls
                                    << " for " << result.stats.accepted << " accepted";
}

TEST(TestFeasibleShapeSet, HandlesParametersThatConstrainEachOther)
{
    // The convolution-shaped difficulty in miniature: no dimension has a bound of its own,
    // and validity is a relation between them. Probing each dimension with the others pinned
    // would describe a slice; the walk tests whole shapes, so coupling costs it nothing.
    const auto region = [](const Shape& shape) {
        const int64_t input = shape[0];
        const int64_t filter = shape[1];
        const int64_t stride = shape[2];
        // A filter must fit the input, and the output must be at least one element.
        return filter <= input && stride <= filter && ((input - filter) / stride) + 1 >= 1;
    };

    const auto result = buildFeasibleShapeSet(
        region, requestFor({{"input", 1, 4096}, {"filter", 1, 512}, {"stride", 1, 32}}));

    ASSERT_FALSE(result.shapes.empty());
    for(const auto& shape : result.shapes)
    {
        EXPECT_LE(shape[1], shape[0]);
        EXPECT_LE(shape[2], shape[1]);
    }
}

TEST(TestFeasibleShapeSet, ReachesBothHalvesOfADisconnectedRegion)
{
    // Two islands with a gap no single step crosses. One walk samples one island and reports
    // a region half the size it is; restarts from independent draws are what reach the other.
    const auto region = [](const Shape& shape) {
        return (shape[0] <= 100 && shape[1] <= 100) || (shape[0] >= 4000 && shape[1] >= 4000);
    };

    auto request = requestFor({{"M", 1, 8192}, {"N", 1, 8192}});
    request.restarts = 16;
    const auto result = buildFeasibleShapeSet(region, request);

    const bool reachedSmall = std::any_of(result.shapes.begin(),
                                          result.shapes.end(),
                                          [](const Shape& s) { return s[0] <= 100; });
    const bool reachedLarge = std::any_of(result.shapes.begin(),
                                          result.shapes.end(),
                                          [](const Shape& s) { return s[0] >= 4000; });

    EXPECT_TRUE(reachedSmall);
    EXPECT_TRUE(reachedLarge) << "restarts did not reach the second island";
}

TEST(TestFeasibleShapeSet, SpreadsAcrossTheRangeRatherThanClustering)
{
    // A walk's trace is a path: consecutive shapes are neighbours. Without thinning the set
    // would be a huddle around wherever it started, which is a corpus that covers one corner
    // and reports as though it covered the space.
    const auto region = [](const Shape&) { return true; };

    const auto result = buildFeasibleShapeSet(region, requestFor(matmulLikeDimensions(), 40));

    ASSERT_EQ(result.shapes.size(), 40U);
    EXPECT_GT(distinctValues(result.shapes, 0), 20U);

    // Spanning the range, not just varied: something small and something large in each.
    const auto minMax = std::minmax_element(
        result.shapes.begin(), result.shapes.end(), [](const Shape& a, const Shape& b) {
            return a[0] < b[0];
        });
    EXPECT_LT((*minMax.first)[0], 100);
    EXPECT_GT((*minMax.second)[0], 1000);
}

TEST(TestFeasibleShapeSet, ReportsAnEmptyRegionRatherThanInventingOne)
{
    // Nothing is acceptable. The honest answer is nothing, with budgetExhausted set so the
    // caller can tell "no region" from "not enough budget" -- which it cannot from the count.
    const auto region = [](const Shape&) { return false; };

    const auto result = buildFeasibleShapeSet(region, requestFor(matmulLikeDimensions()));

    EXPECT_TRUE(result.shapes.empty());
    EXPECT_EQ(result.stats.seedsFound, 0);
    EXPECT_TRUE(result.stats.budgetExhausted);
    EXPECT_GT(result.stats.oracleCalls, 0) << "gave up without asking";
}

TEST(TestFeasibleShapeSet, RespectsTheOracleBudget)
{
    // The budget is what makes an unknown feasible fraction survivable: a region this sparse
    // cannot be characterised in advance, so the search is bounded rather than estimated.
    const auto region = [](const Shape& shape) { return shape[0] == 4096 && shape[1] == 4096; };

    auto request = requestFor(matmulLikeDimensions());
    request.oracleBudget = 500;
    const auto result = buildFeasibleShapeSet(region, request);

    EXPECT_LE(result.stats.oracleCalls, 500 + static_cast<int64_t>(request.dimensions.size()));
    EXPECT_TRUE(result.stats.budgetExhausted);
}

TEST(TestFeasibleShapeSet, IsReproducibleFromItsSeed)
{
    // §5.8: a corpus must be reproducible from its seed, and this is the step that decides
    // which shapes are in it.
    const auto region = [](const Shape& shape) { return shape[0] * shape[1] <= 100000; };

    const auto first = buildFeasibleShapeSet(region, requestFor(matmulLikeDimensions()));
    const auto second = buildFeasibleShapeSet(region, requestFor(matmulLikeDimensions()));

    auto third = requestFor(matmulLikeDimensions());
    third.seed = 7;
    const auto other = buildFeasibleShapeSet(region, third);

    EXPECT_EQ(first.shapes, second.shapes);
    EXPECT_NE(first.shapes, other.shapes);
}

TEST(TestFeasibleShapeSet, StaysInsideTheDeclaredSearchWindow)
{
    // The window is the memory ceiling: nothing above it can be benchmarked, so proposing
    // there wastes oracle calls on shapes that could never enter a corpus.
    const auto region = [](const Shape&) { return true; };

    const auto result
        = buildFeasibleShapeSet(region, requestFor({{"M", 16, 512}, {"N", 16, 512}}));

    ASSERT_FALSE(result.shapes.empty());
    for(const auto& shape : result.shapes)
    {
        EXPECT_GE(shape[0], 16);
        EXPECT_LE(shape[0], 512);
        EXPECT_GE(shape[1], 16);
        EXPECT_LE(shape[1], 512);
    }
}

} // namespace hipdnn_corpus_gen
