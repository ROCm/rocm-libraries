// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestTensorFillers.cpp
 * @brief Covers the contents a problem declares.
 *
 * A routing that assigns no tokens to any expert produces a fast, stable, entirely meaningless
 * measurement — and looks in every column exactly like a fast kernel. So the properties pinned
 * here are the ones a benchmark cannot check for itself: that the offsets account for every
 * token, that the per-token indices agree with the offsets, and that a skewed routing is
 * actually skewed.
 */

#include <gtest/gtest.h>

#include <hipdnn_bench/TensorFillers.hpp>

#include <cstdint>
#include <cstring>
#include <vector>

namespace hipdnn_bench
{
namespace
{

std::vector<int32_t> asInt32(const std::vector<uint8_t>& bytes, size_t count)
{
    std::vector<int32_t> values(count, 0);
    std::memcpy(values.data(), bytes.data(), count * sizeof(int32_t));
    return values;
}

} // namespace

TEST(TestTensorFillers, BalancedRoutingGivesEveryExpertAnEqualShare)
{
    const auto routing = computeRouting(/*numTokens=*/64, /*numExperts=*/8, /*imbalanced=*/false);

    ASSERT_EQ(routing.offsets.size(), 9U);
    EXPECT_EQ(routing.offsets.front(), 0);
    EXPECT_EQ(routing.offsets.back(), 64) << "the offsets must account for every token";
    for(size_t e = 0; e + 1 < routing.offsets.size(); ++e)
    {
        EXPECT_EQ(routing.offsets[e + 1] - routing.offsets[e], 8);
    }
}

TEST(TestTensorFillers, ARemainderIsDistributedRatherThanDropped)
{
    // 10 tokens over 4 experts. Dropping the remainder would leave two tokens unassigned and
    // the last offset short -- a routing the kernel would read as a smaller problem.
    const auto routing = computeRouting(10, 4, false);

    EXPECT_EQ(routing.offsets.back(), 10);
    EXPECT_EQ(routing.assignment.size(), 10U);
}

TEST(TestTensorFillers, AnImbalancedRoutingIsActuallyImbalanced)
{
    // The case a heuristic is least likely to have been tuned for: one expert carrying most of
    // the work while the rest idle. A corpus of only balanced routings never shows it.
    const auto routing = computeRouting(1000, 8, /*imbalanced=*/true);

    ASSERT_EQ(routing.offsets.size(), 9U);
    EXPECT_EQ(routing.offsets.back(), 1000);

    const auto first = routing.offsets[1] - routing.offsets[0];
    const auto second = routing.offsets[2] - routing.offsets[1];
    EXPECT_GT(first, second * 2) << "the head expert must carry materially more";
    EXPECT_GT(first, 1000 / 8) << "and more than an equal share";
}

TEST(TestTensorFillers, AssignmentAgreesWithTheOffsets)
{
    // The invariant that makes the two fills one computation. An offset table saying expert 3
    // owns tokens [40, 60) beside an index table that never names expert 3 describes no
    // coherent routing, and the kernel reads one while the corpus describes the other.
    for(const bool imbalanced : {false, true})
    {
        const auto routing = computeRouting(97, 5, imbalanced);
        ASSERT_EQ(routing.assignment.size(), 97U);

        for(size_t e = 0; e + 1 < routing.offsets.size(); ++e)
        {
            for(int64_t t = routing.offsets[e]; t < routing.offsets[e + 1]; ++t)
            {
                ASSERT_EQ(routing.assignment[static_cast<size_t>(t)], static_cast<int64_t>(e))
                    << "token " << t << " with imbalanced=" << imbalanced;
            }
        }
    }
}

TEST(TestTensorFillers, RoutingIsDeterministic)
{
    // Reproducibility is the whole point of computing this host-side: the same problem point
    // must produce the same bytes on any machine, or a corpus cannot be re-measured.
    EXPECT_EQ(computeRouting(500, 7, true).offsets, computeRouting(500, 7, true).offsets);
    EXPECT_NE(computeRouting(500, 7, true).offsets, computeRouting(500, 7, false).offsets);
}

TEST(TestTensorFillers, WritesOffsetsAsInt32)
{
    const auto routing = computeRouting(64, 4, false);
    const auto bytes = fillRoutingOffsets(5 * sizeof(int32_t), FillElement::INT32, routing);

    EXPECT_EQ(asInt32(bytes, 5), (std::vector<int32_t>{0, 16, 32, 48, 64}));
}

TEST(TestTensorFillers, SequenceStaysInRangeWhenGivenAModulo)
{
    // An index tensor filled with 0..n-1 past the number of experts names experts that do not
    // exist, which the kernel may read as an out-of-bounds gather.
    const auto bytes = fillSequence(6 * sizeof(int32_t), FillElement::INT32, /*modulo=*/4);
    EXPECT_EQ(asInt32(bytes, 6), (std::vector<int32_t>{0, 1, 2, 3, 0, 1}));
}

TEST(TestTensorFillers, UniformIsBoundedAndReproducible)
{
    const auto first = fillUniform(32 * sizeof(int32_t), FillElement::INT32, 8, /*seed=*/5);
    const auto second = fillUniform(32 * sizeof(int32_t), FillElement::INT32, 8, /*seed=*/5);
    const auto other = fillUniform(32 * sizeof(int32_t), FillElement::INT32, 8, /*seed=*/6);

    EXPECT_EQ(first, second);
    EXPECT_NE(first, other);
    for(const auto value : asInt32(first, 32))
    {
        EXPECT_GE(value, 0);
        EXPECT_LT(value, 8);
    }
}

TEST(TestTensorFillers, RefusesNonsenseRatherThanProducingIt)
{
    EXPECT_TRUE(computeRouting(64, 0, false).offsets.empty()) << "no experts is not a routing";
    EXPECT_TRUE(computeRouting(-1, 4, false).offsets.empty());
}

} // namespace hipdnn_bench
