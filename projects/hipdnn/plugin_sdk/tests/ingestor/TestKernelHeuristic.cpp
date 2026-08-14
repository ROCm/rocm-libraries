// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Catalog.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelHeuristic.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include "KernelIngestorTestFixtures.hpp"

/**
 * @file TestKernelHeuristic.cpp
 * @brief Tests for IKernelHeuristic.hpp: eager symbol resolution, ranking/tie-break
 *        order, and the makeKernelHeuristic() factory.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_plugin_sdk::ingestor::testing;

TEST(TestIngestorKernelHeuristic, RefusesToConstructAgainstAnUnregisteredSymbol)
{
    // Eager resolution turns an unshipped scorer symbol into a load-time exclusion,
    // instead of surviving to throw at plan build.
    EXPECT_THROW(NativeKernelHeuristic("hipdnn.kernel_ingestor.test.not_yet_registered"),
                 std::runtime_error);
}

TEST(TestIngestorKernelHeuristic, NamesTheDescriptorThatCouldNotResolve)
{
    // Must name the descriptor to fix, not only the missing symbol.
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
    const auto lowerId = testId(0x01);
    const auto higherId = testId(0x02);
    catalog.entries = {makeDefinition(higherId, 64), makeDefinition(lowerId, 64)};

    const NativeKernelHeuristic heuristic(CONSTANT_SCORE_SYMBOL);
    const auto ranked = heuristic.rank(catalog, context);

    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, lowerId);
}

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
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};
    EXPECT_EQ(heuristic->score(makeDefinition(testId(0x01), 128), context), 128.0);
}

TEST(TestIngestorKernelHeuristic, MakeKernelHeuristicThrowsForAKindWithNoAdapter)
{
    // HeuristicKind::MODEL has no adapter yet; fails at assembly time, not first rank().
    HeuristicDescriptor descriptor;
    descriptor.id = HEURISTIC_ID;
    descriptor.name = "model heuristic";
    descriptor.kind = HeuristicKind::MODEL;
    descriptor.payload = "some/model/artifact.bin";

    EXPECT_THROW(makeKernelHeuristic(descriptor), std::invalid_argument);
}

TEST(TestIngestorKernelHeuristic, MakeKernelHeuristicFallsBackWhenNoDescriptorIsSupplied)
{
    // An engine shipping no UHD still gets a usable scorer rather than a null or a
    // throw: absence is a supported state, not a load failure.
    const auto heuristic = makeKernelHeuristic(std::nullopt);

    ASSERT_NE(heuristic, nullptr);
}

TEST(TestIngestorKernelHeuristic, WarnsNamingTheEngineWhenNoHeuristicIsSupplied)
{
    // The warning is the whole point of allowing a missing UHD: it is what separates an
    // engine that meant to declare its order from one still waiting on a model. An
    // unnamed warning cannot tell an operator which engine to go look at.
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_WARN);

    const auto heuristic = makeKernelHeuristic(std::nullopt, "engine 'test:unranked'");

    ASSERT_NE(heuristic, nullptr);
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, "test:unranked"))
        << "warning did not name the engine:\n"
        << recorder.getRecordedLogsAsString();
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, "ships no heuristic"))
        << "warning did not say what was missing:\n"
        << recorder.getRecordedLogsAsString();
}

TEST(TestIngestorKernelHeuristic, DeclaredOrderRanksOnPriorityWhenNoHeuristicIsSupplied)
{
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    Catalog catalog;
    const auto lowPriorityId = testId(0x01);
    const auto highPriorityId = testId(0x02);
    // Declared low-first, so insertion order cannot make this pass. The block sizes
    // differ and favour the loser, so a fallback that scored on kernel metadata instead
    // of returning a constant would outrank priority and fail here.
    catalog.entries
        = {makeDefinition(lowPriorityId, 4096, 1), makeDefinition(highPriorityId, 64, 5)};

    const auto heuristic = makeKernelHeuristic(std::nullopt);
    const auto ranked = heuristic->rank(catalog, context);

    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, highPriorityId);
}

TEST(TestIngestorKernelHeuristic, DeclaredOrderFallsToKernelIdWhenPriorityTies)
{
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    Catalog catalog;
    const auto lowerId = testId(0x01);
    const auto higherId = testId(0x02);
    // Equal priority, declared higher-id first, block sizes differing and favouring the
    // loser: only the id tie-break can produce the expected order, and any metadata-
    // sensitive score would break it.
    catalog.entries = {makeDefinition(higherId, 4096), makeDefinition(lowerId, 64)};

    const auto heuristic = makeKernelHeuristic(std::nullopt);
    const auto ranked = heuristic->rank(catalog, context);

    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, lowerId);
}

TEST(TestIngestorKernelHeuristic, DeclaredOrderRanksEveryKernelEqually)
{
    // The fallback must contribute no ordering of its own: any score spread would
    // outrank priority, which is the one signal an engine without a model still has.
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    const DeclaredOrderKernelHeuristic heuristic;

    EXPECT_EQ(heuristic.score(makeDefinition(testId(0x01), 64), context),
              heuristic.score(makeDefinition(testId(0x02), 4096), context));
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
