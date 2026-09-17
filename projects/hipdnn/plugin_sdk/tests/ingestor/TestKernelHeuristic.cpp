// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

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

/// Scoring sees what the engine's graph match bound, so a heuristic can rank on graph
/// facts and not only on `$kernel.*`. Ranking inverts on the token alone here: the
/// kernels are otherwise identical.
TEST(TestIngestorKernelHeuristic, ScoresFromTheTokensTheGraphMatchBound)
{
    constexpr const char* TOKEN_SCORE_SYMBOL = "hipdnn.kernel_ingestor.test.token_score";
    ScoreRegistry::registerSymbol(
        TOKEN_SCORE_SYMBOL,
        +[](const MatchContext&, const BoundTokens& bound, const KernelDefinition& kernel) {
            const auto preferred = tryGetBoundInt(bound, "test.preferred_block_size");
            return preferred.has_value()
                           && kernel.getIntMetadata(std::string(BLOCK_SIZE)) == *preferred
                       ? 1.0
                       : 0.0;
        });

    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    Catalog catalog;
    const auto smallId = testId(0x01);
    const auto largeId = testId(0x02);
    catalog.entries = {makeDefinition(smallId, 64), makeDefinition(largeId, 256)};
    catalog.bound["test.preferred_block_size"] = int64_t{64};

    const NativeKernelHeuristic heuristic(TOKEN_SCORE_SYMBOL);
    const auto ranked = heuristic.rank(catalog, context);

    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, smallId);

    ScoreRegistry::unregisterSymbol(TOKEN_SCORE_SYMBOL);
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

TEST(TestIngestorKernelHeuristic, RanksNanScoringKernelsBelowEveryFiniteScore)
{
    // A pack's score() is arbitrary code, so NaN is reachable. NaN compares false against
    // everything, which would make it read as equivalent to every kernel while finite
    // scores stayed ordered -- not a strict weak ordering, and UB for stable_sort.
    const ScopedNanScore nanScore;
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    Catalog catalog;
    const auto nanId = testId(0x01);
    const auto smallId = testId(0x02);
    const auto largeId = testId(0x03);
    catalog.entries
        = {makeDefinition(nanId, 4096), makeDefinition(smallId, 64), makeDefinition(largeId, 256)};

    const NativeKernelHeuristic heuristic(NAN_SCORE_SYMBOL);
    const auto ranked = heuristic.rank(catalog, context);

    ASSERT_EQ(ranked.size(), 3U);
    // The finite kernels keep their own order, and the NaN one sinks to the back.
    EXPECT_EQ(ranked[0].kernelId, largeId);
    EXPECT_EQ(ranked[1].kernelId, smallId);
    EXPECT_EQ(ranked[2].kernelId, nanId);
}

TEST(TestIngestorKernelHeuristic, KeepsFiniteScoresOrderedWhenAScorerReturnsNan)
{
    // The damage a NaN does is to the *other* kernels: it reads as equivalent to every
    // one of them, so finite scores get separated by it and stop being sorted among
    // themselves. Interleave NaN and finite so a broken comparator cannot look sorted by
    // accident, then assert the finite subsequence is still descending.
    //
    // Determinism alone is too weak an assertion to make here: stable_sort against a
    // broken comparator is reliably *wrong* rather than random, so a repeat-and-compare
    // check passes even with the bug present.
    const ScopedNanScore nanScore;
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    Catalog catalog;
    std::vector<DescriptorId> nanIds;
    for(uint8_t seed = 1; seed <= 6; ++seed)
    {
        const bool scoresNan = (seed % 2 == 0);
        catalog.entries.push_back(makeDefinition(testId(seed), scoresNan ? 4096 : 64 * seed));
        if(scoresNan)
        {
            nanIds.push_back(testId(seed));
        }
    }

    const NativeKernelHeuristic heuristic(NAN_SCORE_SYMBOL);
    const auto ranked = heuristic.rank(catalog, context);
    ASSERT_EQ(ranked.size(), catalog.entries.size());

    const auto scoresNan = [&nanIds](const DescriptorId& id) {
        return std::find(nanIds.begin(), nanIds.end(), id) != nanIds.end();
    };

    int64_t previousBlockSize = std::numeric_limits<int64_t>::max();
    bool seenNan = false;
    for(const auto& entry : ranked)
    {
        if(scoresNan(entry.kernelId))
        {
            seenNan = true;
            continue;
        }
        // Every finite kernel must outrank every NaN one, and stay ordered among its peers.
        EXPECT_FALSE(seenNan) << "a finite score ranked below a NaN score";
        const int64_t blockSize = entry.getIntMetadata(BLOCK_SIZE);
        EXPECT_LE(blockSize, previousBlockSize) << "finite scores are no longer descending";
        previousBlockSize = blockSize;
    }

    // And the result is reproducible, which is the promise the fallback ordering makes.
    const auto repeated = heuristic.rank(catalog, context);
    ASSERT_EQ(repeated.size(), ranked.size());
    for(size_t i = 0; i < ranked.size(); ++i)
    {
        EXPECT_EQ(ranked[i].kernelId, repeated[i].kernelId) << "ranking diverged at index " << i;
    }
}

TEST(TestIngestorKernelHeuristic, BreaksTiesAmongNanScoringKernelsOnPriorityThenKernelId)
{
    // NaN kernels collapse to one score, so the existing tie-break chain must still
    // order them -- otherwise "ranks last" would be an unordered heap at the back.
    const ScopedNanScore nanScore;
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    Catalog catalog;
    const auto lowPriorityId = testId(0x01);
    const auto tiedLowerId = testId(0x02);
    const auto tiedHigherId = testId(0x03);
    // Listed with the higher id first, so passing requires the tie-break to reorder them
    // rather than merely preserving input order.
    catalog.entries = {makeDefinition(lowPriorityId, 4096, 1),
                       makeDefinition(tiedHigherId, 4096, 5),
                       makeDefinition(tiedLowerId, 4096, 5)};

    const NativeKernelHeuristic heuristic(NAN_SCORE_SYMBOL);
    const auto ranked = heuristic.rank(catalog, context);

    ASSERT_EQ(ranked.size(), 3U);
    EXPECT_EQ(ranked[0].kernelId, tiedLowerId); // priority 5, lower id wins the tie
    EXPECT_EQ(ranked[1].kernelId, tiedHigherId); // priority 5
    EXPECT_EQ(ranked[2].kernelId, lowPriorityId); // priority 1 sinks despite the id order
}

TEST(TestIngestorKernelHeuristic, TreatsInfiniteScoresAsOrdinaryExtremes)
{
    // Infinities are already a valid strict weak ordering; they must keep ranking
    // normally rather than being lumped in with NaN.
    const ScopedTestSymbols symbols;
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    ScoreRegistry::registerSymbol(
        "hipdnn.kernel_ingestor.test.infinite_score",
        +[](const MatchContext&, const BoundTokens&, const KernelDefinition& kernel) -> double {
            return kernel.getIntMetadata(BLOCK_SIZE) == 4096
                       ? std::numeric_limits<double>::infinity()
                       : -std::numeric_limits<double>::infinity();
        });

    Catalog catalog;
    const auto positiveInfinityId = testId(0x01);
    const auto negativeInfinityId = testId(0x02);
    catalog.entries
        = {makeDefinition(negativeInfinityId, 64), makeDefinition(positiveInfinityId, 4096)};

    {
        const NativeKernelHeuristic heuristic("hipdnn.kernel_ingestor.test.infinite_score");
        const auto ranked = heuristic.rank(catalog, context);

        ASSERT_EQ(ranked.size(), 2U);
        EXPECT_EQ(ranked.front().kernelId, positiveInfinityId);
        EXPECT_EQ(ranked.back().kernelId, negativeInfinityId);
    }

    ScoreRegistry::unregisterSymbol("hipdnn.kernel_ingestor.test.infinite_score");
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
    EXPECT_EQ(heuristic->score(context, BoundTokens{}, makeDefinition(testId(0x01), 128)), 128.0);
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

TEST(TestIngestorKernelHeuristic, UnrankedFallsToPriorityWhenNoHeuristicIsSupplied)
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

TEST(TestIngestorKernelHeuristic, UnrankedFallsToKernelIdWhenPriorityTies)
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

TEST(TestIngestorKernelHeuristic, UnrankedRanksEveryKernelEqually)
{
    // The fallback must contribute no ordering of its own: any score spread would
    // outrank priority, which is the one signal an engine without a model still has.
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    const UnrankedKernelHeuristic heuristic;

    EXPECT_EQ(heuristic.score(context, BoundTokens{}, makeDefinition(testId(0x01), 64)),
              heuristic.score(context, BoundTokens{}, makeDefinition(testId(0x02), 4096)));
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
