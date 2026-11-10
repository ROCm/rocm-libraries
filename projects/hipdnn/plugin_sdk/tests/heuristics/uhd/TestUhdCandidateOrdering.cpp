// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestUhdCandidateOrdering.cpp
 * @brief How a ranking places candidates an early-exit model pruned (uhd::BatchScores)
 *        relative to fully scored ones and to each other.
 */

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/UhdKernelHeuristic.hpp>

#include <cstdint>
#include <limits>
#include <vector>

namespace
{
using hipdnn_plugin_sdk::ingestor::DescriptorId;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::detail::orderRanked;
using hipdnn_plugin_sdk::ingestor::detail::prunedCandidate;
using hipdnn_plugin_sdk::ingestor::detail::Ranked;

constexpr double INF = std::numeric_limits<double>::infinity();

KernelDefinition kernel(uint8_t tag, int64_t priority = 0)
{
    KernelDefinition definition;
    definition.kernelId = DescriptorId{};
    definition.kernelId[0] = tag;
    definition.priority = priority;
    return definition;
}

Ranked full(const KernelDefinition& entry, double ordering)
{
    return Ranked{{ordering, ordering}, &entry};
}

std::vector<uint8_t> tagsOf(const std::vector<Ranked>& ranked)
{
    std::vector<uint8_t> tags;
    tags.reserve(ranked.size());
    for(const auto& candidate : ranked)
    {
        tags.push_back(candidate.entry->kernelId[0]);
    }
    return tags;
}

TEST(TestUhdCandidateOrdering, UsableFullScoresPrecedePrunedWhichPrecedeUnusableOnes)
{
    const auto a = kernel(1);
    const auto b = kernel(2);
    const auto c = kernel(3);
    const auto d = kernel(4);
    // A declined or out-of-range candidate (-infinity) ranks after every pruned one, however
    // early that one was pruned; a usable full score ranks ahead of every pruned one.
    std::vector<Ranked> ranked
        = {prunedCandidate(&a, 0.0, 3, 1e300, 1.0), prunedCandidate(&b, 0.0, 1, -1e300, 1.0)};
    ranked.push_back(full(c, -INF));
    ranked.push_back(full(d, 1.0));
    orderRanked(ranked);
    EXPECT_EQ(tagsOf(ranked), (std::vector<uint8_t>{4, 1, 2, 3}));
}

TEST(TestUhdCandidateOrdering, PrunedCandidatesOrderByStageThenByObjectiveDirectionOfPartialSum)
{
    const auto a = kernel(1);
    const auto b = kernel(2);
    const auto c = kernel(3);
    const auto d = kernel(4);

    // Maximizing: a larger partial sum is better, but a later stage always wins.
    std::vector<Ranked> maximize = {prunedCandidate(&a, 0.0, 1, 100.0, 1.0),
                                    prunedCandidate(&b, 0.0, 2, -5.0, 1.0),
                                    prunedCandidate(&c, 0.0, 1, 200.0, 1.0),
                                    prunedCandidate(&d, 0.0, 2, 3.0, 1.0)};
    orderRanked(maximize);
    EXPECT_EQ(tagsOf(maximize), (std::vector<uint8_t>{4, 2, 3, 1}));

    // Minimizing (objectiveSign -1): a smaller partial sum is better.
    std::vector<Ranked> minimize = {prunedCandidate(&a, 0.0, 1, 100.0, -1.0),
                                    prunedCandidate(&b, 0.0, 2, -5.0, -1.0),
                                    prunedCandidate(&c, 0.0, 1, 200.0, -1.0),
                                    prunedCandidate(&d, 0.0, 2, 3.0, -1.0)};
    orderRanked(minimize);
    EXPECT_EQ(tagsOf(minimize), (std::vector<uint8_t>{2, 4, 1, 3}));
}

TEST(TestUhdCandidateOrdering, ATiedPartialSumFallsToPriorityThenKernelId)
{
    const auto lowId = kernel(1, 0);
    const auto highId = kernel(9, 0);
    const auto preferred = kernel(5, 7);
    std::vector<Ranked> ranked = {prunedCandidate(&highId, 0.0, 2, 4.0, 1.0),
                                  prunedCandidate(&lowId, 0.0, 2, 4.0, 1.0),
                                  prunedCandidate(&preferred, 0.0, 2, 4.0, 1.0)};
    orderRanked(ranked);
    EXPECT_EQ(tagsOf(ranked), (std::vector<uint8_t>{5, 1, 9}));
}

TEST(TestUhdCandidateOrdering, ANaNPartialSumRanksLastInItsStageWithoutBreakingTheOrder)
{
    const auto a = kernel(1);
    const auto b = kernel(2);
    const auto c = kernel(3);
    std::vector<Ranked> ranked
        = {prunedCandidate(&a, 0.0, 2, std::numeric_limits<double>::quiet_NaN(), 1.0),
           prunedCandidate(&b, 0.0, 2, -4.0, 1.0),
           prunedCandidate(&c, 0.0, 2, 4.0, 1.0)};
    orderRanked(ranked);
    EXPECT_EQ(tagsOf(ranked), (std::vector<uint8_t>{3, 2, 1}));
}

TEST(TestUhdCandidateOrdering, APrunedCandidateReportsNoMeasurement)
{
    const auto a = kernel(1);
    const auto pruned = prunedCandidate(&a, 12.5, 1, 42.0, 1.0);
    EXPECT_TRUE(pruned.pruned);
    EXPECT_EQ(pruned.score.reported, 0.0);
    EXPECT_EQ(pruned.score.ordering, -INF);
    EXPECT_EQ(pruned.group, 12.5);
}

TEST(TestUhdCandidateOrdering, FullyScoredCandidatesKeepTheirScoreThenPriorityThenIdOrder)
{
    const auto a = kernel(1, 0);
    const auto b = kernel(2, 5);
    const auto c = kernel(3, 5);
    const auto d = kernel(4, 0);
    std::vector<Ranked> ranked = {full(a, 1.0), full(b, 1.0), full(c, 1.0), full(d, 3.0)};
    orderRanked(ranked);
    EXPECT_EQ(tagsOf(ranked), (std::vector<uint8_t>{4, 2, 3, 1}));
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
