// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestDeterministicRandom.cpp
 * @brief Pins sampled values for fixed seeds.
 *
 * A corpus is reproducible from its seed only if the draws are, and the `<random>`
 * distributions differ between standard libraries. These values were produced from the raw
 * `std::mt19937_64` output, which the standard fixes, so they hold on every platform; a draw
 * that went back to a library distribution would change them.
 */

#include <gtest/gtest.h>

#include <hipdnn_corpus_gen/DeterministicRandom.hpp>
#include <hipdnn_corpus_gen/FeasibleShapeSet.hpp>

#include <cstdint>
#include <random>
#include <vector>

namespace hipdnn_corpus_gen
{

TEST(TestDeterministicRandom, IntegerDrawsForASeedArePinned)
{
    std::mt19937_64 rng(42);
    std::vector<uint64_t> drawn;
    drawn.reserve(6);
    for(int i = 0; i < 6; ++i)
    {
        drawn.push_back(detail::uniformBelow(rng, 1000));
    }

    EXPECT_EQ(drawn, (std::vector<uint64_t>{406, 824, 450, 662, 381, 428}));
}

TEST(TestDeterministicRandom, ABiasedRawOutputIsRefused)
{
    // For this bound, the first raw output of seed 0 (2947667278772165694) falls in the
    // `2^64 mod bound` sliver that a plain modulo would over-represent; the draw must move on
    // to the next raw output instead of returning it.
    std::mt19937_64 rng(0);

    EXPECT_EQ(detail::uniformBelow(rng, (uint64_t{1} << 63) + 1), 9078476729143589258ULL);
}

TEST(TestDeterministicRandom, ZeroBoundDrawsNothing)
{
    std::mt19937_64 rng(1);
    std::mt19937_64 untouched(1);

    EXPECT_EQ(detail::uniformBelow(rng, 0), 0U);
    EXPECT_EQ(rng(), untouched());
}

TEST(TestDeterministicRandom, RealDrawsForASeedArePinned)
{
    std::mt19937_64 unit(42);
    EXPECT_EQ(detail::uniformUnit(unit), 0.75515553295453897);
    EXPECT_EQ(detail::uniformUnit(unit), 0.63903139385469743);
    EXPECT_EQ(detail::uniformUnit(unit), 0.7521452007480266);

    std::mt19937_64 ranged(42);
    EXPECT_NEAR(detail::uniformReal(ranged, -2.0, 6.0), 4.0412442636363117, 1e-14);
    EXPECT_NEAR(detail::uniformReal(ranged, -2.0, 6.0), 3.1122511508375794, 1e-14);
}

TEST(TestDeterministicRandom, BernoulliDrawsForASeedArePinned)
{
    std::mt19937_64 rng(42);
    std::vector<bool> drawn;
    drawn.reserve(10);
    for(int i = 0; i < 10; ++i)
    {
        drawn.push_back(detail::bernoulli(rng, 0.5));
    }

    EXPECT_EQ(drawn,
              (std::vector<bool>{false, false, false, true, false, true, false, true, true, true}));
}

TEST(TestDeterministicRandom, NormalDrawsForASeedArePinned)
{
    std::mt19937_64 rng(42);

    EXPECT_NEAR(detail::standardNormal(rng), -1.0771745442782885, 1e-12);
    EXPECT_NEAR(detail::standardNormal(rng), 1.0945198485006107, 1e-12);
    EXPECT_NEAR(detail::standardNormal(rng), 1.7947316657951717, 1e-12);
    EXPECT_NEAR(detail::standardNormal(rng), -0.91212515564414098, 1e-12);
}

TEST(TestDeterministicRandom, ASeededShapeSearchIsPinned)
{
    // The whole search: footholds, hit-and-run steps and cell selection all draw from the
    // seeded engine, so a seed reproduces the same shapes everywhere.
    FeasibleSetRequest request;
    request.dimensions = {{"M", 1, 8192}, {"N", 1, 8192}, {"K", 1, 8192}};
    request.targetCount = 6;
    request.seed = 20260827;
    const auto region
        = [](const Shape& shape) { return shape[0] * shape[1] <= 1000000 && shape[2] % 8 == 0; };

    const auto result = buildFeasibleShapeSet(region, request);

    const std::vector<Shape> expected{{1, 2, 40},
                                      {123, 5523, 6264},
                                      {7440, 2, 256},
                                      {2, 5016, 8},
                                      {1017, 542, 8},
                                      {48, 11, 6384}};
    EXPECT_EQ(result.shapes, expected);
}

} // namespace hipdnn_corpus_gen
