// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "Gfx942AttentionUnified3dTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx942AttentionUnified3dGeometry.hpp"

/**
 * @file TestGfx942AttentionUnified3dScorer.cpp
 * @brief The cold ranking of hipkernel:Gfx942AttentionUnified3d, separate from admission:
 *        the scorer orders kernels kernel_match already admitted and never removes one.
 *
 * Order: block_size 16 first, then num_segments nearest gfx942SplitRuleSegments(...) at
 * block 16 with the device's CU count, ties to the smaller num_segments. Any num_segments
 * is correct (the reduce combines every segment), so this is performance only.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing::gfx942_unified_3d
{
namespace
{

/// A sample of the shipped num_segments values (2 to 128).
const std::vector<int64_t> SHIPPED_SEGMENTS{2, 3, 7, 8, 14, 16, 28, 32, 57, 64, 128};

double score(const GraphSpec& graphSpec, const KernelSpec& kernelSpec, int cus = 304)
{
    const auto graph = fixture(graphSpec, cus);
    const auto bound = matchesGraph(PACK, graph.context());
    EXPECT_TRUE(bound.has_value());
    return scoreKernel(
        PACK, graph.context(), makeKernel(kernelSpec), bound.value_or(BoundTokens{}));
}

KernelSpec kernelWith(int64_t blockSize, int64_t segments)
{
    KernelSpec kernel;
    kernel.blockSize = blockSize;
    kernel.numSegments = segments;
    return kernel;
}

/// The highest-scoring (block, segments) among the shipped values for @p graph.
std::pair<int64_t, int64_t> best(const GraphSpec& graph, int cus = 304)
{
    std::pair<int64_t, int64_t> winner{0, 0};
    double top = -1.0;
    for(const int64_t block : {16, 32, 64})
    {
        for(const int64_t segments : SHIPPED_SEGMENTS)
        {
            const double s = score(graph, kernelWith(block, segments), cus);
            if(s > top)
            {
                top = s;
                winner = {block, segments};
            }
        }
    }
    return winner;
}

int64_t splitRule(const GraphSpec& g, int cus)
{
    return gfx942SplitRuleSegments(
        g.batch, g.numQueryHeads, g.numKvHeads, g.headSize, 16, g.seqLenKv, cus);
}

TEST(TestGfx942AttentionUnified3dScorer, PrefersBlock16OverAnyOtherBlock)
{
    const auto graph = GraphSpec{};
    const int64_t target = splitRule(graph, 304);
    // The worst block-16 candidate still beats the best block-32/64 one.
    const double worst16 = score(graph, kernelWith(16, target == 128 ? 2 : 128));
    for(const int64_t block : {32, 64})
    {
        EXPECT_GT(worst16, score(graph, kernelWith(block, target))) << block;
    }
}

TEST(TestGfx942AttentionUnified3dScorer, PicksTheSegmentCountNearestTheSplitRule)
{
    for(const int64_t batch : {1, 4, 64})
    {
        for(const int64_t seqLenKv : {64, 2048, 8192, 65536})
        {
            GraphSpec graph;
            graph.batch = batch;
            graph.seqLenKv = seqLenKv;
            const int64_t target = splitRule(graph, 304);
            int64_t nearest = SHIPPED_SEGMENTS.front();
            for(const int64_t s : SHIPPED_SEGMENTS)
            {
                if(std::llabs(s - target) < std::llabs(nearest - target))
                {
                    nearest = s; // strict <: ties keep the smaller, earlier value
                }
            }
            const auto [block, segments] = best(graph);
            EXPECT_EQ(block, 16);
            EXPECT_EQ(segments, nearest)
                << "B" << batch << " Skv" << seqLenKv << " target " << target;
        }
    }
}

TEST(TestGfx942AttentionUnified3dScorer, BreaksADistanceTieTowardFewerSegments)
{
    const auto graph = GraphSpec{};
    const int64_t target = splitRule(graph, 304);
    ASSERT_GT(target, 1);
    EXPECT_GT(score(graph, kernelWith(16, target - 1)), score(graph, kernelWith(16, target + 1)));
}

TEST(TestGfx942AttentionUnified3dScorer, FollowsTheDevicesCuCount)
{
    // B1 Hkv8 D128 Skv2048 at block 16 takes the decode rule: min(4 * cus / 8, 64).
    GraphSpec graph;
    graph.batch = 1;
    graph.seqLenKv = 2048;
    EXPECT_EQ(splitRule(graph, 304), 64);
    EXPECT_EQ(best(graph, 304).second, 64);
    EXPECT_EQ(splitRule(graph, 80), 40);
    EXPECT_EQ(best(graph, 80).second, 32); // nearest sampled value to 40
}

TEST(TestGfx942AttentionUnified3dScorer, FallsBackToRocKesOffBoxCuCount)
{
    GraphSpec graph;
    graph.batch = 1;
    graph.seqLenKv = 2048;
    // multiProcessorCount 0: rocKE's _resolve_num_cus off-box default, 120 -> 60 -> 57.
    EXPECT_EQ(best(graph, 0).second, 57);
    EXPECT_EQ(best(graph, 0), best(graph, 120));
}

TEST(TestGfx942AttentionUnified3dScorer, NeverRemovesAnAdmittedKernel)
{
    // kernel_match admits every knob value, and every admitted kernel scores above zero
    // (the "unsupported" floor): the scorer orders, it does not filter.
    const auto graph = GraphSpec{};
    for(const int64_t block : {16, 32, 64})
    {
        for(const int64_t segments : SHIPPED_SEGMENTS)
        {
            const auto kernel = kernelWith(block, segments);
            EXPECT_TRUE(kernelMatchesGraph(graph, kernel)) << block << "/" << segments;
            EXPECT_GT(score(graph, kernel), 0.0) << block << "/" << segments;
        }
    }
}

TEST(TestGfx942AttentionUnified3dScorer, GivesEveryKnobPairADistinctScore)
{
    const auto graph = GraphSpec{};
    std::vector<double> scores;
    for(const int64_t block : {16, 32, 64})
    {
        for(const int64_t segments : SHIPPED_SEGMENTS)
        {
            scores.push_back(score(graph, kernelWith(block, segments)));
        }
    }
    std::sort(scores.begin(), scores.end());
    EXPECT_EQ(std::adjacent_find(scores.begin(), scores.end()), scores.end());
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::testing::gfx942_unified_3d

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
