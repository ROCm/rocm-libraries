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
 * @brief Tests for IKernelHeuristic.hpp: lazy symbol resolution, ranking/tie-break
 *        order, and the makeKernelHeuristic() factory.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_plugin_sdk::ingestor::testing;

TEST(TestIngestorKernelHeuristic, RefusesToConstructAgainstAnUnregisteredSymbol)
{
    // The inverse of what this asserted before eager resolution. Resolving on first
    // score() means a UHD naming a scorer this build does not ship survives load and
    // survives isApplicable(), then throws at plan build, past the point RFC 0017
    // §8.6 makes applicability a promise the engine must keep. Failing here instead
    // turns that into a load-time exclusion.
    EXPECT_THROW(NativeKernelHeuristic("hipdnn.kernel_ingestor.test.not_yet_registered"),
                 std::runtime_error);
}

TEST(TestIngestorKernelHeuristic, NamesTheDescriptorThatCouldNotResolve)
{
    // The whole diagnostic for a typo'd symbol, so it must name the descriptor to fix
    // rather than only the symbol that was missing.
    HeuristicDescriptor descriptor;
    descriptor.id = HEURISTIC_ID;
    descriptor.name = "misspelled selector";
    descriptor.kind = HeuristicKind::NATIVE;
    descriptor.payload = "hipdnn.kernel_ingestor.test.misspelled";

    try
    {
        makeKernelHeuristic(descriptor);
        FAIL() << "expected an unresolved-symbol failure";
    }
    catch(const std::runtime_error& error)
    {
        const std::string message = error.what();
        EXPECT_NE(message.find("hipdnn.kernel_ingestor.test.misspelled"), std::string::npos);
        EXPECT_NE(message.find("misspelled selector"), std::string::npos);
        EXPECT_NE(message.find(toString(HEURISTIC_ID)), std::string::npos);
    }
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
    // Inserted highest-id first: result must not depend on load order.
    const auto lowerId = testId(0x01);
    const auto higherId = testId(0x02);
    catalog.entries = {makeDefinition(higherId, 64), makeDefinition(lowerId, 64)};

    const NativeKernelHeuristic heuristic(CONSTANT_SCORE_SYMBOL);
    const auto ranked = heuristic.rank(catalog, context);

    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, lowerId);
}

// makeKernelHeuristic(): the UHD -> IKernelHeuristic adapter factory.

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
    // Confirms the factory wired the payload symbol through.
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};
    EXPECT_EQ(heuristic->score(makeDefinition(testId(0x01), 128), context), 128.0);
}

TEST(TestIngestorKernelHeuristic, MakeKernelHeuristicThrowsForAKindWithNoAdapter)
{
    // HeuristicKind::MODEL has no adapter yet (RFC 0017's UHD follow-up); fails at
    // descriptor-assembly time, not at first rank().
    HeuristicDescriptor descriptor;
    descriptor.id = HEURISTIC_ID;
    descriptor.name = "model heuristic";
    descriptor.kind = HeuristicKind::MODEL;
    descriptor.payload = "some/model/artifact.bin";

    EXPECT_THROW(makeKernelHeuristic(descriptor), std::invalid_argument);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
