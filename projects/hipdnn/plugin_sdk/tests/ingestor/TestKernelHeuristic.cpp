// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <memory>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Catalog.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelHeuristic.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "KernelIngestorTestFixtures.hpp"

/**
 * @file TestKernelHeuristic.cpp
 * @brief Unit tests for IKernelHeuristic.hpp: NativeKernelHeuristic's lazy symbol
 *        resolution, the ranking/tie-break order every IKernelHeuristic gets for free
 *        from rank(), and the makeKernelHeuristic() factory.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_plugin_sdk::ingestor::testing;

TEST(TestIngestorKernelHeuristic, ResolvesItsSymbolOnFirstUseRatherThanAtConstruction)
{
    // RFC 0017 section 8.1 admits the heuristic at applicability only as a name, and
    // section 3 generalizes it: a heuristic model is not read until something needs the
    // catalog ranked. An engine whose matchers reject a graph must never pay for its
    // selector, so constructing one against a symbol that is not registered yet has to
    // be legal, and only scoring may fail.
    const NativeKernelHeuristic heuristic("hipdnn.kernel_ingestor.test.not_yet_registered");

    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};
    const auto kernel = makeDefinition(testId(0x01), 64);

    EXPECT_THROW(heuristic.score(kernel, context), std::runtime_error);
}

TEST(TestIngestorKernelHeuristic, RanksHigherScoringKernelsFirst)
{
    const ScopedTestSymbols symbols;
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    Catalog catalog;
    const auto lowId = testId(0x01);
    const auto highId = testId(0x02);
    catalog.entries = {makeDefinition(lowId, 64), makeDefinition(highId, 256)};

    const NativeKernelHeuristic heuristic(SCORE_SYMBOL);
    const auto ranked = heuristic.rank(catalog, context);

    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, highId);
}

TEST(TestIngestorKernelHeuristic, BreaksScoreTiesOnPriority)
{
    const ScopedConstantScore constantScore;
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    Catalog catalog;
    const auto lowPriorityId = testId(0x01);
    const auto highPriorityId = testId(0x02);
    catalog.entries = {makeDefinition(lowPriorityId, 64, 1), makeDefinition(highPriorityId, 64, 5)};

    const NativeKernelHeuristic heuristic(CONSTANT_SCORE_SYMBOL);
    const auto ranked = heuristic.rank(catalog, context);

    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, highPriorityId);
}

TEST(TestIngestorKernelHeuristic, BreaksRemainingTiesOnKernelIdForStabilityAcrossRuns)
{
    const ScopedConstantScore constantScore;
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    Catalog catalog;
    // Inserted highest-id first: the result must not depend on load order.
    const auto lowerId = testId(0x01);
    const auto higherId = testId(0x02);
    catalog.entries = {makeDefinition(higherId, 64), makeDefinition(lowerId, 64)};

    const NativeKernelHeuristic heuristic(CONSTANT_SCORE_SYMBOL);
    const auto ranked = heuristic.rank(catalog, context);

    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, lowerId);
}

// ---------------------------------------------------------------------------
// makeKernelHeuristic(): the UHD -> IKernelHeuristic adapter factory.
// ---------------------------------------------------------------------------

TEST(TestIngestorKernelHeuristic, MakeKernelHeuristicBuildsANativeHeuristicForNativeKind)
{
    const ScopedTestSymbols symbols;

    HeuristicDescriptor descriptor;
    descriptor.id = HEURISTIC_ID;
    descriptor.name = "test heuristic";
    descriptor.kind = HeuristicKind::NATIVE;
    descriptor.payload = SCORE_SYMBOL;

    const auto heuristic = makeKernelHeuristic(descriptor);

    ASSERT_NE(heuristic, nullptr);
    // Proves the factory actually wired the payload symbol through: scoring against the
    // registered scorer must succeed and return what it computes.
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};
    EXPECT_EQ(heuristic->score(makeDefinition(testId(0x01), 128), context), 128.0);
}

TEST(TestIngestorKernelHeuristic, MakeKernelHeuristicThrowsForAKindWithNoAdapter)
{
    // HeuristicKind::MODEL has no adapter yet: RFC 0017's UHD follow-up owns it. Failing
    // at descriptor-assembly time, not at the first rank(), matches the other
    // cross-reference checks the state manager runs eagerly.
    HeuristicDescriptor descriptor;
    descriptor.id = HEURISTIC_ID;
    descriptor.name = "model heuristic";
    descriptor.kind = HeuristicKind::MODEL;
    descriptor.payload = "some/model/artifact.bin";

    EXPECT_THROW(makeKernelHeuristic(descriptor), std::invalid_argument);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
