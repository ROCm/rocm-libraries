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

#include <cmath>

#include <hipdnn_plugin_sdk/ingestor/Catalog.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelHeuristicFactory.hpp>
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
    descriptor.adapter = UhdAdapter::NATIVE;
    descriptor.nativeSymbol = "hipdnn.kernel_ingestor.test.misspelled";

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
    descriptor.adapter = UhdAdapter::NATIVE;
    descriptor.nativeSymbol = SCORE_SYMBOL;

    const auto heuristic = makeKernelHeuristic(descriptor);

    ASSERT_NE(heuristic, nullptr);
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};
    EXPECT_EQ(heuristic->score(context, BoundTokens{}, makeDefinition(testId(0x01), 128)), 128.0);
}

/// A native cost scorer: milliseconds by block size, 256 the fast kernel. Registered for one
/// test's duration, because the registry is process-wide.
constexpr const char* MILLISECONDS_SYMBOL = "hipdnn.kernel_ingestor.test.milliseconds";

class ScopedMillisecondsScorer
{
public:
    ScopedMillisecondsScorer()
    {
        ScoreRegistry::registerSymbol(
            MILLISECONDS_SYMBOL,
            +[](const MatchContext&, const BoundTokens&, const KernelDefinition& kernel) {
                switch(kernel.getIntMetadata(BLOCK_SIZE))
                {
                case 256:
                    return 1.0;
                case 64:
                    return 10.0;
                default:
                    return 0.0; // no measurement
                }
            });
    }
    ~ScopedMillisecondsScorer()
    {
        ScoreRegistry::unregisterSymbol(MILLISECONDS_SYMBOL);
    }
    ScopedMillisecondsScorer(const ScopedMillisecondsScorer&) = delete;
    ScopedMillisecondsScorer& operator=(const ScopedMillisecondsScorer&) = delete;
};

HeuristicDescriptor millisecondsDescriptor(const std::string& metric)
{
    HeuristicDescriptor descriptor;
    descriptor.id = HEURISTIC_ID;
    descriptor.name = "native cost scorer";
    descriptor.adapter = UhdAdapter::NATIVE;
    descriptor.nativeSymbol = MILLISECONDS_SYMBOL;
    descriptor.objective = "min";
    descriptor.score = {metric, !metric.empty(), "identity"};
    return descriptor;
}

/// Regression (R4). A signature-less native scorer never saw the UHD's objective, and
/// rankScored orders higher-first, so a `min` scorer returning milliseconds ranked the 10 ms
/// kernel first. Both routes to a direct scorer are covered: the metric-less ranker the
/// factory builds itself, and a metric's ranker, which UhdKernelHeuristic wraps.
TEST(TestIngestorKernelHeuristic, ANativeMinScorerRanksTheCheapestKernelFirst)
{
    const ScopedMillisecondsScorer scorer;
    const TestGraph graph;
    const auto properties = testDeviceProperties();

    Catalog catalog;
    const auto slowId = testId(0x01);
    const auto fastId = testId(0x02);
    catalog.entries = {makeDefinition(slowId, 64), makeDefinition(fastId, 256)};

    for(const std::string metric : {"", "time"})
    {
        SCOPED_TRACE("metric '" + metric + "'");
        const auto heuristic = makeKernelHeuristic(millisecondsDescriptor(metric));
        ASSERT_NE(heuristic, nullptr);
        const MatchContext context{graph, 0, properties, metric.empty() ? "tflops" : "time"};

        const auto ranked = heuristic->rank(catalog, context);
        ASSERT_EQ(ranked.size(), 2U);
        EXPECT_EQ(ranked.front().kernelId, fastId) << "the 10 ms kernel outranked the 1 ms one";
    }
}

/// Once a direct scorer is ordered in its objective's direction, a calibrated `time` native
/// ranker's figure of merit is usable: ascending milliseconds, the physical values -- not the
/// negated ordering key, which a caller comparing engines would read as negative time.
TEST(TestIngestorKernelHeuristic, ACalibratedNativeTimeScorerReportsAscendingMilliseconds)
{
    const ScopedMillisecondsScorer scorer;
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties, "time"};

    Catalog catalog;
    catalog.entries = {makeDefinition(testId(0x01), 64), makeDefinition(testId(0x02), 256)};

    const auto heuristic = makeKernelHeuristic(millisecondsDescriptor("time"));
    std::string modelId;
    const auto calibrated = heuristic->calibratedRanking(catalog, context, modelId);

    ASSERT_EQ(calibrated.size(), 2U);
    EXPECT_EQ(calibrated.front().kernelId, testId(0x02));
    EXPECT_DOUBLE_EQ(calibrated.front().score, 1.0);
    EXPECT_DOUBLE_EQ(calibrated.back().score, 10.0);
    EXPECT_EQ(modelId, toString(HEURISTIC_ID));
}

/// Regression (S2). A direct scorer's value was reported in its transform's space and the
/// inverse applied afterwards, past a zero test meant for "no measurement" -- so a `log` time
/// scorer pricing a kernel at exactly 1 ms (log 1 = 0) read as unmeasured, and the whole
/// calibrated ranking was withheld because its winner looked unpriced.
TEST(TestIngestorKernelHeuristic, ATransformedZeroIsAMeasurementNotItsAbsence)
{
    constexpr const char* LOG_MILLISECONDS_SYMBOL = "hipdnn.kernel_ingestor.test.log_milliseconds";
    ScoreRegistry::registerSymbol(
        LOG_MILLISECONDS_SYMBOL,
        +[](const MatchContext&, const BoundTokens&, const KernelDefinition& kernel) {
            return std::log(kernel.getIntMetadata(BLOCK_SIZE) == 256 ? 1.0 : 10.0);
        });
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties, "time"};

    Catalog catalog;
    catalog.entries = {makeDefinition(testId(0x01), 64), makeDefinition(testId(0x02), 256)};

    auto descriptor = millisecondsDescriptor("time");
    descriptor.nativeSymbol = LOG_MILLISECONDS_SYMBOL;
    descriptor.score.transform = "log";
    const auto heuristic = makeKernelHeuristic(descriptor);
    std::string modelId;
    const auto calibrated = heuristic->calibratedRanking(catalog, context, modelId);

    ASSERT_EQ(calibrated.size(), 2U) << "a 1 ms winner was taken for an unmeasured one";
    EXPECT_EQ(calibrated.front().kernelId, testId(0x02));
    EXPECT_DOUBLE_EQ(calibrated.front().score, 1.0);
    EXPECT_NEAR(calibrated.back().score, 10.0, 1e-12);
    ScoreRegistry::unregisterSymbol(LOG_MILLISECONDS_SYMBOL);
}

/// Under `min` a zero cost is no measurement. Negated it would be -0, above every real
/// candidate's negated cost, so the one kernel the scorer could not price would win.
TEST(TestIngestorKernelHeuristic, AZeroCostDoesNotWinUnderANativeMinScorer)
{
    const ScopedMillisecondsScorer scorer;
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    Catalog catalog;
    const auto unpricedId = testId(0x01);
    const auto slowId = testId(0x02);
    catalog.entries = {makeDefinition(unpricedId, 128), makeDefinition(slowId, 64)};

    const auto heuristic = makeKernelHeuristic(millisecondsDescriptor(""));
    const auto ranked = heuristic->rankScored(catalog, context);

    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, slowId);
    EXPECT_DOUBLE_EQ(ranked.back().score, 0.0) << "the unpriced kernel reported a figure of merit";
}

TEST(TestIngestorKernelHeuristic, MakeKernelHeuristicDegradesWhenAModelCannotBeBroughtUp)
{
    // A MODEL naming an artifact that is not there degrades rather than throwing, which is
    // the difference between it and NATIVE: an unregistered symbol is a build fact and the
    // engine could never score, while a missing artifact is a deployment fact and the
    // engine still selects, by declared order (RFC 0019 §5).
    HeuristicDescriptor descriptor;
    descriptor.id = HEURISTIC_ID;
    descriptor.name = "model heuristic";
    descriptor.adapter = UhdAdapter::TREE_DATA;
    descriptor.modelArtifactPath = "some/model/artifact.bin";
    descriptor.featuresSignature = {R"("$kernel.tile_m")"};
    descriptor.featuresHash = "sha256:whatever";

    std::shared_ptr<IKernelHeuristic> heuristic;
    ASSERT_NO_THROW(heuristic = makeKernelHeuristic(descriptor));
    ASSERT_NE(heuristic, nullptr);

    // Ranking still works, and gives the order an engine with no model would give.
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    Catalog catalog;
    const auto lowPriorityId = testId(0x01);
    const auto highPriorityId = testId(0x02);
    catalog.entries = {makeDefinition(lowPriorityId, 64, 1), makeDefinition(highPriorityId, 64, 5)};

    const auto ranked = heuristic->rank(catalog, context);

    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, highPriorityId);
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
    // It reports 0 -- RFC 0019 §5 step 7's value for "no measurement" -- so a fallback and a
    // model that scored zero describe themselves the same way. traceDecidedBy() is what tells
    // them apart, and calibratedRanking needs one rule rather than two sentinels.
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    const UnrankedKernelHeuristic heuristic;

    EXPECT_DOUBLE_EQ(heuristic.score(context, BoundTokens{}, makeDefinition(testId(0x01), 64)),
                     0.0);
    EXPECT_DOUBLE_EQ(heuristic.score(context, BoundTokens{}, makeDefinition(testId(0x02), 4096)),
                     0.0);

    // Equal-in-ordering is what the fallback owes. These two definitions carry the same
    // priority, so the documented tiebreak -- ascending descriptor id -- has to decide, and
    // a NaN score must not have leaked into the comparator to decide it instead.
    Catalog catalog;
    catalog.entries.push_back(makeDefinition(testId(0x02), 4096));
    catalog.entries.push_back(makeDefinition(testId(0x01), 64));
    const auto ranked = heuristic.rankScored(catalog, context);
    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, testId(0x01)) << "the id tiebreak did not decide";
    EXPECT_DOUBLE_EQ(ranked.front().score, 0.0) << "the fallback invented a figure of merit";
}
/// RFC 0019 §5 step 7: "No model, or the scorer errors -> rank by static_order (priority + id)",
/// under the heading "A failure degrades the result; it never fails the request" -- which spells
/// out that a malformed descriptor set "must not fail after the engine has already claimed
/// applicability."
///
/// The model path honoured this from the start; the native path did not, and every UHD shipped
/// today is native. It is reachable from descriptor data alone: the shipped scorer reads
/// kernel.getIntMetadata("block_size"), which throws std::out_of_range when a KDP joins the
/// engine with a kernel that omits the knob.
class ThrowingHeuristic : public IKernelHeuristic
{
public:
    double score(const MatchContext& /*context*/,
                 const BoundTokens& /*bound*/,
                 const KernelDefinition& /*kernel*/) const override
    {
        throw std::out_of_range("kernel has no metadata field 'block_size'");
    }
};

TEST(TestIngestorKernelHeuristic, AThrowingScorerDegradesInsteadOfFailingTheRequest)
{
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    Catalog catalog;
    catalog.entries.push_back(makeDefinition(testId(0x02), 4096));
    catalog.entries.push_back(makeDefinition(testId(0x01), 64));

    const ThrowingHeuristic heuristic;

    // The request survives at all -- this is the whole of step 7's guarantee.
    std::vector<ScoredKernel> ranked;
    ASSERT_NO_THROW(ranked = heuristic.rankScored(catalog, context));

    // And it is a usable answer: static_order, which step 5 defines as priority then id. These
    // two carry equal priority, so ascending id decides.
    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, testId(0x01));
    EXPECT_DOUBLE_EQ(ranked.front().score, 0.0) << "a failed ranking reported a figure of merit";

    // The whole ranking degrades, not the candidates that happened to throw: a mix of real
    // scores and sentinels is neither order.
    EXPECT_DOUBLE_EQ(ranked.back().score, 0.0);
}

TEST(TestIngestorKernelHeuristic, AThrowingScorerIsReportedRatherThanSwallowed)
{
    // Degrading silently would leave an engine that looks like it ranks on a model while
    // ranking on declared order -- the failure RFC 0019 §12 exists to make visible.
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    Catalog catalog;
    catalog.entries.push_back(makeDefinition(testId(0x01), 64));

    const ThrowingHeuristic heuristic;
    (void)heuristic.rankScored(catalog, context);

    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "scorer threw while ranking"));
    EXPECT_TRUE(recorder.hasLogContaining("block_size")) << "the cause was not carried through";

    // Once: the cause is a property of the descriptor set, so it recurs for every graph.
    const auto after = recorder.countLogsAtLevel(HIPDNN_SEV_ERROR);
    (void)heuristic.rankScored(catalog, context);
    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_ERROR), after) << "the report repeated";
}

TEST(TestIngestorKernelHeuristic, RankAlsoSurvivesAThrowingScorer)
{
    // rank() is what production calls (KernelIngestorStateManager), and it is derived from
    // rankScored -- so the guard has to reach it without a second implementation.
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    Catalog catalog;
    catalog.entries.push_back(makeDefinition(testId(0x02), 4096));
    catalog.entries.push_back(makeDefinition(testId(0x01), 64));

    const ThrowingHeuristic heuristic;
    std::vector<KernelDefinition> ordered;
    ASSERT_NO_THROW(ordered = heuristic.rank(catalog, context));
    ASSERT_EQ(ordered.size(), 2U);
    EXPECT_EQ(ordered.front().kernelId, testId(0x01));
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
