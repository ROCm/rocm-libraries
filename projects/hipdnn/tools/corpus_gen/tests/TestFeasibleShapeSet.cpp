// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestFeasibleShapeSet.cpp
 * @brief Covers shape-set generation against regions whose shape is known in advance.
 *
 * Each case defines its region, so the expected result is arithmetic: sparse, coupled, and
 * disconnected regions.
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

/// Distinct values seen in one dimension, as a crude check of spread.
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
    // A refused shape in the set would be benchmarked and blamed on the kernel.
    const auto region
        = [](const Shape& shape) { return shape[0] * shape[1] <= 1000000 && shape[2] % 8 == 0; };

    const auto result = buildFeasibleShapeSet(region, requestFor(matmulLikeDimensions()));

    ASSERT_FALSE(result.shapes.empty());
    for(const auto& shape : result.shapes)
    {
        EXPECT_TRUE(region(shape)) << "returned a shape the oracle rejects";
    }
}

TEST(TestFeasibleShapeSet, FindsARegionTooSparseForRejectionSampling)
{
    // About one box point in 512 qualifies.
    const auto region = [](const Shape& shape) {
        return shape[0] % 8 == 0 && shape[1] % 8 == 0 && shape[2] % 8 == 0;
    };

    auto request = requestFor(matmulLikeDimensions());
    const auto result = buildFeasibleShapeSet(region, request);

    // The corpus is the occupied cells; many cells hold no feasible point, so the target count
    // need not be reached and is not padded with duplicates.
    EXPECT_GT(result.stats.cellsOccupied, request.targetCount / 2)
        << "occupied " << result.stats.cellsOccupied << " of " << result.stats.cells;
    EXPECT_EQ(static_cast<int64_t>(result.shapes.size()), result.stats.cellsOccupied);

    // Rejection sampling would pay about 512 calls per shape here.
    const double callsPerShape = static_cast<double>(result.stats.oracleCalls)
                                 / static_cast<double>(result.stats.accepted);
    EXPECT_LT(callsPerShape, 150.0) << "oracle calls " << result.stats.oracleCalls << " for "
                                    << result.stats.accepted << " accepted";
}

TEST(TestFeasibleShapeSet, HandlesParametersThatConstrainEachOther)
{
    // No dimension has its own bound; validity is a relation between them.
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
    // No single step crosses the gap; restarts must reach the other island.
    const auto region = [](const Shape& shape) {
        return (shape[0] <= 100 && shape[1] <= 100) || (shape[0] >= 4000 && shape[1] >= 4000);
    };

    auto request = requestFor({{"M", 1, 8192}, {"N", 1, 8192}});
    request.restarts = 16;
    const auto result = buildFeasibleShapeSet(region, request);

    const bool reachedSmall = std::any_of(
        result.shapes.begin(), result.shapes.end(), [](const Shape& s) { return s[0] <= 100; });
    const bool reachedLarge = std::any_of(
        result.shapes.begin(), result.shapes.end(), [](const Shape& s) { return s[0] >= 4000; });

    EXPECT_TRUE(reachedSmall);
    EXPECT_TRUE(reachedLarge) << "restarts did not reach the second island";
}

TEST(TestFeasibleShapeSet, SpreadsAcrossTheRangeRatherThanClustering)
{
    // Without thinning, a walk's consecutive shapes would cluster near its start.
    const auto region = [](const Shape&) { return true; };

    const auto result = buildFeasibleShapeSet(region, requestFor(matmulLikeDimensions(), 40));

    ASSERT_EQ(result.shapes.size(), 40U);
    EXPECT_GT(distinctValues(result.shapes, 0), 20U);

    const auto minMax
        = std::minmax_element(result.shapes.begin(),
                              result.shapes.end(),
                              [](const Shape& a, const Shape& b) { return a[0] < b[0]; });
    EXPECT_LT((*minMax.first)[0], 100);
    EXPECT_GT((*minMax.second)[0], 1000);
}

TEST(TestFeasibleShapeSet, ReportsAnEmptyRegionRatherThanInventingOne)
{
    // budgetExhausted distinguishes "no region" from "not enough budget".
    const auto region = [](const Shape&) { return false; };

    const auto result = buildFeasibleShapeSet(region, requestFor(matmulLikeDimensions()));

    EXPECT_TRUE(result.shapes.empty());
    EXPECT_EQ(result.stats.seedsFound, 0);
    EXPECT_TRUE(result.stats.budgetExhausted);
    EXPECT_GT(result.stats.oracleCalls, 0) << "gave up without asking";
}

TEST(TestFeasibleShapeSet, RespectsTheOracleBudget)
{
    const auto region = [](const Shape& shape) { return shape[0] == 4096 && shape[1] == 4096; };

    auto request = requestFor(matmulLikeDimensions());
    request.oracleBudget = 500;
    const auto result = buildFeasibleShapeSet(region, request);

    EXPECT_LE(result.stats.oracleCalls, 500 + static_cast<int64_t>(request.dimensions.size()));
    EXPECT_TRUE(result.stats.budgetExhausted);
}

TEST(TestFeasibleShapeSet, IsReproducibleFromItsSeed)
{
    // §5.8: a corpus must be reproducible from its seed.
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
    // The window is the memory ceiling; nothing above it can be benchmarked.
    const auto region = [](const Shape&) { return true; };

    const auto result = buildFeasibleShapeSet(region, requestFor({{"M", 16, 512}, {"N", 16, 512}}));

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
