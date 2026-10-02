// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestUhdKernelHeuristic.cpp
 * @brief Tests that a trained UHD's score reaches kernel ranking, built from both the
 *        problem and the kernel, and that every model failure leaves declared order.
 */

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <optional>

#include "../KernelIngestorTestFixtures.hpp"

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/FeatureExtractor.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/NativeScorerRegistry.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/ScoreTransform.hpp>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelHeuristicFactory.hpp>
#include <hipdnn_plugin_sdk/ingestor/MakeEngine.hpp>
#include <hipdnn_plugin_sdk/ingestor/UhdKernelHeuristic.hpp>

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/GbdtModelTestBuilder.hpp>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <exception>
#include <filesystem>
#include <map>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace hipdnn_plugin_sdk::ingestor
{
namespace
{

// Slot 0 is a kernel knob, slot 1 a problem token.
const std::vector<nlohmann::json> SIGNATURE = {"$kernel.tile_m", "$attention.seqlen"};

/// Kernel-only, for cases ranking through a whole engine: the fixture graph matcher binds no
/// `$q.*` tokens, so a problem slot would make the model fail closed.
const std::vector<nlohmann::json> ENGINE_SIGNATURE = {"$kernel.tile_m"};

/// The knobs a conformant UED exposes for SIGNATURE (RFC 0019 §6.3 check 2).
const std::vector<std::string> KNOBS = {"tile_m"};

/// The fields a conformant KMD declares for SIGNATURE (§6.3 check 2: `F ⊆ KMD.fields`).
const std::unordered_set<std::string> FIELDS = {"tile_m"};

/// `split_k` as a knob must also be a KMD field (§3.2).
const std::unordered_set<std::string> FIELDS_WITH_SPLIT_K = {"tile_m", "split_k"};

DescriptorId testId(uint8_t tag)
{
    DescriptorId id{};
    id.fill(0);
    id[0] = tag;
    return id;
}

KernelDefinition kernelWith(uint8_t tag, int64_t tileM, int64_t priority)
{
    KernelDefinition kernel;
    kernel.kernelId = testId(tag);
    kernel.priority = priority;
    kernel.metadata["tile_m"] = tileM;
    return kernel;
}

/// One usable leaf and one out-of-range leaf, so a single ranking contains both kinds.
hipdnn_test_sdk::utilities::GbdtModelTestBuilder::TreeSpec oneUsableOneOutOfRange()
{
    hipdnn_test_sdk::utilities::GbdtModelTestBuilder::TreeSpec spec;
    spec.featureIndices = {0, 0, 0};
    spec.thresholds = {96.0, 0.0, 0.0};
    spec.leftChildren = {1, -1, -1};
    spec.rightChildren = {2, -1, -1};
    spec.leafValues = {0.0, 4.0, -1.0};
    spec.defaultLeft = {1, 1, 1};
    return spec;
}

/// One usable leaf and one leaf predicting exactly zero, which RFC 0019 §8.3 rejects.
hipdnn_test_sdk::utilities::GbdtModelTestBuilder::TreeSpec oneUsableOneZero()
{
    auto spec = oneUsableOneOutOfRange();
    spec.leafValues = {0.0, 4.0, 0.0};
    return spec;
}

/// Every leaf negative: ordinary for a GBDT, out of domain under an "exp" transform.
hipdnn_test_sdk::utilities::GbdtModelTestBuilder::TreeSpec preferNegativeScores()
{
    hipdnn_test_sdk::utilities::GbdtModelTestBuilder::TreeSpec spec;
    spec.featureIndices = {0, 0, 0};
    spec.thresholds = {96.0, 0.0, 0.0};
    spec.leftChildren = {1, -1, -1};
    spec.rightChildren = {2, -1, -1};
    spec.leafValues = {0.0, -2.0, -0.5};
    spec.defaultLeft = {1, 1, 1};
    return spec;
}

/// Splits on slot 0 (`$kernel.tile_m`): tile_m <= 96 scores 1.0, above scores 9.0.
hipdnn_test_sdk::utilities::GbdtModelTestBuilder::TreeSpec preferLargeTiles()
{
    hipdnn_test_sdk::utilities::GbdtModelTestBuilder::TreeSpec spec;
    spec.featureIndices = {0, 0, 0};
    spec.thresholds = {96.0, 0.0, 0.0};
    spec.leftChildren = {1, -1, -1};
    spec.rightChildren = {2, -1, -1};
    spec.leafValues = {0.0, 1.0, 9.0};
    spec.defaultLeft = {1, 1, 1};
    return spec;
}

/// The inverse of preferLargeTiles, so the winner identifies which model ranked.
hipdnn_test_sdk::utilities::GbdtModelTestBuilder::TreeSpec preferSmallTiles()
{
    hipdnn_test_sdk::utilities::GbdtModelTestBuilder::TreeSpec spec;
    spec.featureIndices = {0, 0, 0};
    spec.thresholds = {96.0, 0.0, 0.0};
    spec.leftChildren = {1, -1, -1};
    spec.rightChildren = {2, -1, -1};
    spec.leafValues = {0.0, 9.0, 1.0};
    spec.defaultLeft = {1, 1, 1};
    return spec;
}

/// Splits on slot 1 (seqlen) first, so the same catalog ranks differently per problem.
hipdnn_test_sdk::utilities::GbdtModelTestBuilder::TreeSpec preferLargeTilesOnLongSequences()
{
    hipdnn_test_sdk::utilities::GbdtModelTestBuilder::TreeSpec spec;
    // Root splits on seqlen; each side then splits on tile_m, in opposite directions.
    spec.featureIndices = {1, 0, 0, 0, 0, 0, 0};
    spec.thresholds = {1024.0, 96.0, 96.0, 0.0, 0.0, 0.0, 0.0};
    spec.leftChildren = {1, 3, 5, -1, -1, -1, -1};
    spec.rightChildren = {2, 4, 6, -1, -1, -1, -1};
    //                     short seq: small tile wins   long seq: large tile wins
    spec.leafValues = {0.0, 0.0, 0.0, 9.0, 1.0, 1.0, 9.0};
    spec.defaultLeft = {1, 1, 1, 1, 1, 1, 1};
    return spec;
}

struct Fixture
{
    /// Relative to the directory writeFixture wrote it in.
    std::string modelFileName;
    /// The hash SIGNATURE really computes to, which is what a descriptor must declare.
    std::string featuresHash;
    /// Score fields the fixture was built for, carried so the descriptor matches.
    std::string objective;
    bool calibrated;
    std::string scoreTransform;
    /// The signature the artifact was built for, so the descriptor declares the same one.
    std::vector<nlohmann::json> signature;
};

/// Writes a GBDT artifact into @p dir and reports what a descriptor over it must say.
///
/// @param modelHash Written into the artifact; defaults to the signature's real hash.
Fixture writeFixture(const std::filesystem::path& dir,
                     const hipdnn_test_sdk::utilities::GbdtModelTestBuilder::TreeSpec& tree,
                     const std::string& objective = "max",
                     const std::string& modelHash = {},
                     // RFC 0019 §4.4: any registered metric may be calibrated.
                     bool calibrated = true,
                     /// "exp" inverts as a logarithm: out of domain for a negative prediction.
                     const std::string& scoreTransform = "identity",
                     /// The artifact's feature count follows this.
                     const std::vector<nlohmann::json>& signature = SIGNATURE)
{
    const std::string signatureHash = uhd::FeatureExtractor::computeHash(signature);

    hipdnn_test_sdk::utilities::GbdtModelTestBuilder model;
    model.setFeaturesHash(modelHash.empty() ? signatureHash : modelHash)
        .setNumFeatures(static_cast<int32_t>(signature.size()))
        .setTrainingArches({"gfx942"})
        .addTree(tree);
    model.buildToFile((dir / "model.bin").string());

    return {"model.bin", signatureHash, objective, calibrated, scoreTransform, signature};
}

/// The registered metric @p objective implies; the parser refuses a mismatch (RFC 0019 §4.4).
std::string metricFor(const std::string& objective)
{
    return objective == "min" ? "time" : "tflops";
}

/// The descriptor the loader would produce for a tree_data UHD in @p dir.
HeuristicDescriptor modelDescriptor(const std::filesystem::path& dir,
                                    const std::string& artifact,
                                    const std::string& objective = "max",
                                    bool calibrated = true,
                                    const std::string& scoreTransform = "identity",
                                    const std::string& featuresHash = {},
                                    const std::vector<nlohmann::json>& signature = SIGNATURE)
{
    HeuristicDescriptor descriptor;
    descriptor.id = testId(0xEE);
    descriptor.name = "test model heuristic";
    descriptor.adapter = UhdAdapter::TREE_DATA;
    descriptor.featuresSignature = signature;
    descriptor.featuresHash
        = featuresHash.empty() ? uhd::FeatureExtractor::computeHash(signature) : featuresHash;
    descriptor.objective = objective;
    descriptor.score.metric = metricFor(objective);
    descriptor.score.calibrated = calibrated;
    descriptor.score.transform = scoreTransform;
    descriptor.modelArtifactPath = artifact;
    descriptor.baseDir = dir;
    return descriptor;
}

/// The descriptor over the artifact @p fixture wrote, carrying the score fields it was built for.
HeuristicDescriptor modelDescriptor(const std::filesystem::path& dir, const Fixture& fixture)
{
    return modelDescriptor(dir,
                           fixture.modelFileName,
                           fixture.objective,
                           fixture.calibrated,
                           fixture.scoreTransform,
                           {},
                           fixture.signature);
}

/// A `.uhd.json` naming @p artifact, for rules only the parser enforces (e.g. §4.4).
///
/// @param metric Omitted from the document when empty (a metric-less ranker).
nlohmann::json uhdDocument(const std::string& artifact,
                           const std::string& objective,
                           bool calibrated,
                           const std::string& metric = "tflops")
{
    nlohmann::json document;
    document["version"] = "1.0";
    document["id"] = "ee000000-0000-0000-0000-000000000000";
    document["name"] = "test model heuristic";
    document["adapter"] = "tree_data";
    document["features_signature"] = SIGNATURE;
    document["features_hash"] = uhd::FeatureExtractor::computeHash(SIGNATURE);
    document["objective"] = objective;
    document["score"] = {{"calibrated", calibrated}, {"transform", "identity"}};
    if(!metric.empty())
    {
        document["score"]["metric"] = metric;
    }
    document["tree_data"] = {{"artifact", artifact}};
    document["trained_against"]
        = {{"ued", {{"id", "11000000-0000-0000-0000-000000000000"}, {"revision", "1.0"}}},
           {"kmd", {{"id", "12000000-0000-0000-0000-000000000000"}, {"revision", "1.0"}}},
           {"umd", nlohmann::json::array()}};
    return document;
}

/// Parses @p document as descriptor discovery would; nullopt when the loader refuses it.
std::optional<HeuristicDescriptor> parseUhd(const nlohmann::json& document,
                                            const std::filesystem::path& path)
{
    try
    {
        return detail::parseHeuristicDescriptor(document, path);
    }
    catch(const std::exception&)
    {
        return std::nullopt;
    }
}

/// The fixtures' training architecture, so the out-of-distribution warning stays out.
DeviceProperties gfx942()
{
    auto properties = testing::testDeviceProperties();
    properties.gcnArchName = "gfx942";
    properties.multiProcessorCount = 304;
    return properties;
}

/// Declared order puts the small tile first; every model here prefers the large one.
Catalog catalogAgainstPriority(int64_t seqlen)
{
    Catalog catalog;
    catalog.entries = {kernelWith(0x01, 64, 100), kernelWith(0x02, 128, 1)};
    catalog.bound["attention.seqlen"] = seqlen;
    return catalog;
}

/// Priorities swapped: declared order puts the large tile first.
Catalog catalogAlongPriority(int64_t seqlen)
{
    Catalog catalog;
    catalog.entries = {kernelWith(0x01, 64, 1), kernelWith(0x02, 128, 100)};
    catalog.bound["attention.seqlen"] = seqlen;
    return catalog;
}

/// An engine's figure of merit (RFC 0019 §11.1): the calibrated top value, else 0.
double engineEstimate(const IKernelHeuristic& heuristic,
                      const Catalog& catalog,
                      const MatchContext& context)
{
    std::string modelId;
    const auto calibrated = heuristic.calibratedRanking(catalog, context, modelId);
    return calibrated.empty() ? 0.0 : calibrated.front().score;
}

/// Rankers keyed by metric (`""` for metric-less), then architecture key.
using RankersByArch = std::map<std::string, HeuristicDescriptor>;
using RankersByMetric = std::map<std::string, RankersByArch>;

} // namespace

/// Slot 0 is an inline expression; both slots read `tile_m`, so the pair stays §6.3-conformant
/// while only the expression varies.
const std::vector<nlohmann::json> EXPRESSION_SIGNATURE
    = {nlohmann::json::parse(R"({"ceil_div":["$attention.seqlen","$kernel.tile_m"]})"),
       "$kernel.tile_m"};
/// Same references and arity as EXPRESSION_SIGNATURE; only the operator differs.
const std::vector<nlohmann::json> RESPELLED_SIGNATURE
    = {nlohmann::json::parse(R"({"*":["$attention.seqlen","$kernel.tile_m"]})"), "$kernel.tile_m"};

TEST(TestIngestorUhdKernelHeuristic, ADescriptorWhoseInlineExpressionChangedIsRefused)
{
    // §6.3 check 1: expression bodies are in the hash, so the swap is caught. tryCreate, not
    // makeKernelHeuristic: the factory degrades to declared order rather than returning null.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_inline_expression_changed");
    const auto fixture = writeFixture(dir.path(),
                                      preferLargeTiles(),
                                      "max",
                                      {},
                                      /*calibrated=*/true,
                                      "identity",
                                      EXPRESSION_SIGNATURE);

    const auto descriptor = modelDescriptor(dir.path(),
                                            fixture.modelFileName,
                                            fixture.objective,
                                            fixture.calibrated,
                                            fixture.scoreTransform,
                                            fixture.featuresHash,
                                            RESPELLED_SIGNATURE);

    EXPECT_EQ(UhdKernelHeuristic::tryCreate(descriptor, "test", KNOBS, FIELDS), nullptr);
}

TEST(TestIngestorUhdKernelHeuristic, AnExpressionCarryingDescriptorLoadsWhenItsHashAgrees)
{
    // Control: the test above refuses the swap, not inline expressions as such.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_inline_expression_agrees");
    const auto fixture = writeFixture(dir.path(),
                                      preferLargeTiles(),
                                      "max",
                                      {},
                                      /*calibrated=*/true,
                                      "identity",
                                      EXPRESSION_SIGNATURE);

    EXPECT_NE(
        UhdKernelHeuristic::tryCreate(modelDescriptor(dir.path(), fixture), "test", KNOBS, FIELDS),
        nullptr);
}

TEST(TestIngestorUhdKernelHeuristic, RanksByTheModelRatherThanByPriority)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_kernel_heuristic_happy");
    const auto fixture = writeFixture(dir.path(), preferLargeTiles());

    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};
    const auto ranked = heuristic->rank(catalogAgainstPriority(2048), context);

    ASSERT_EQ(ranked.size(), 2U);
    // The large tile is last by priority and first by score.
    EXPECT_EQ(ranked.front().kernelId, testId(0x02));
}

TEST(TestIngestorUhdKernelHeuristic, TheProblemChangesTheRanking)
{
    // The policy path binds no query variables, so this distinguishes the model path.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_kernel_heuristic_problem");
    const auto fixture = writeFixture(dir.path(), preferLargeTilesOnLongSequences());

    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};

    const auto longSeq = heuristic->rank(catalogAgainstPriority(4096), context);
    const auto shortSeq = heuristic->rank(catalogAgainstPriority(128), context);

    ASSERT_EQ(longSeq.size(), 2U);
    ASSERT_EQ(shortSeq.size(), 2U);
    EXPECT_EQ(longSeq.front().kernelId, testId(0x02)); // long sequence: large tile
    EXPECT_EQ(shortSeq.front().kernelId, testId(0x01)); // short sequence: small tile
}

/// A kernel axis plus `$graph.flops`, which the work model publishes, not any matcher. Under
/// preferLargeTilesOnLongSequences(), work <= 1024 favours the small tile, above it the large.
const std::vector<nlohmann::json> WORK_SIGNATURE = {"$kernel.tile_m", "$graph.flops"};

TEST(TestIngestorUhdKernelHeuristic, TheGraphsLogicalWorkChangesTheRankingThroughTheLiveSelector)
{
    // The winner follows `graph.flops`, which no matcher binds.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_kernel_heuristic_graph_work");
    const auto fixture = writeFixture(dir.path(),
                                      preferLargeTilesOnLongSequences(),
                                      "max",
                                      {},
                                      /*calibrated=*/true,
                                      "identity",
                                      WORK_SIGNATURE);
    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    const testing::MatmulTestGraph small(4, 4, 4); // 128 flops
    const testing::MatmulTestGraph large(64, 64, 64); // 524288 flops
    const auto properties = gfx942();
    const auto catalog = catalogAgainstPriority(2048);

    const auto onSmall = heuristic->rankScored(catalog, MatchContext{small.graph(), 0, properties});
    const auto onLarge = heuristic->rankScored(catalog, MatchContext{large.graph(), 0, properties});

    ASSERT_EQ(onSmall.size(), 2U);
    ASSERT_EQ(onLarge.size(), 2U);
    EXPECT_EQ(onSmall.front().kernelId, testId(0x01)) << "small problem: small tile";
    EXPECT_EQ(onLarge.front().kernelId, testId(0x02)) << "large problem: large tile";
    // Declared order also puts the small tile first; the leaf value proves the model ranked.
    EXPECT_DOUBLE_EQ(onSmall.front().score, 9.0);
    EXPECT_DOUBLE_EQ(onLarge.front().score, 9.0);
}

TEST(TestIngestorUhdKernelHeuristic, AGraphMatchTokenCannotStandInForTheCanonicalWork)
{
    // A matcher may publish new names, never a reserved one like `graph.flops`.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_kernel_heuristic_reserved_token");
    const auto fixture = writeFixture(dir.path(),
                                      preferLargeTilesOnLongSequences(),
                                      "max",
                                      {},
                                      /*calibrated=*/true,
                                      "identity",
                                      WORK_SIGNATURE);
    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    const testing::MatmulTestGraph large(64, 64, 64);
    const auto properties = gfx942();
    const MatchContext context{large.graph(), 0, properties};
    for(const auto* token : {"graph.flops", "$graph.flops"})
    {
        auto catalog = catalogAgainstPriority(2048);
        catalog.bound[token] = 1.0; // a "small problem" if it were believed
        const auto ranked = heuristic->rankScored(catalog, context);
        ASSERT_EQ(ranked.size(), 2U);
        EXPECT_EQ(ranked.front().kernelId, testId(0x02)) << token << " overrode graph.flops";
        EXPECT_DOUBLE_EQ(ranked.front().score, 9.0);
        EXPECT_DOUBLE_EQ(heuristic->score(context, catalog.bound, catalog.entries[1]), 9.0)
            << "the single-kernel path binds a different problem half from the ranking";
    }
}

TEST(TestIngestorUhdKernelHeuristic, AMinimisingObjectiveReversesTheOrder)
{
    // A `min` model declares the `time` metric, so it must be asked for a time ranking.
    const hipdnn_test_sdk::utilities::ScopedDirectory maxDir("uhd_kernel_heuristic_max");
    const hipdnn_test_sdk::utilities::ScopedDirectory minDir("uhd_kernel_heuristic_min");
    const auto maxFixture = writeFixture(maxDir.path(), preferLargeTiles(), "max");
    const auto minFixture = writeFixture(minDir.path(), preferLargeTiles(), "min");

    const auto maximising
        = makeKernelHeuristic(modelDescriptor(maxDir.path(), maxFixture), {}, KNOBS, FIELDS);
    const auto minimising
        = makeKernelHeuristic(modelDescriptor(minDir.path(), minFixture), {}, KNOBS, FIELDS);
    ASSERT_NE(maximising, nullptr);
    ASSERT_NE(minimising, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};
    const MatchContext timeContext{graph, 0, properties, "time"};

    // Along priority, so declared order does not already put the small tile first.
    const auto catalog = catalogAlongPriority(2048);

    const auto maxRanked = maximising->rank(catalog, context);
    const auto minRanked = minimising->rank(catalog, timeContext);

    ASSERT_EQ(maxRanked.size(), 2U);
    ASSERT_EQ(minRanked.size(), 2U);
    EXPECT_EQ(maxRanked.front().kernelId, testId(0x02)); // larger tile scores higher
    EXPECT_EQ(minRanked.front().kernelId, testId(0x01)); // ... so it loses when minimising
}

TEST(TestIngestorUhdKernelHeuristic, AnAbsentArtifactDegradesToDeclaredOrder)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_kernel_heuristic_absent");

    // KNOBS, so the model is refused for the missing artifact, not the §6.3 knob check.
    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), "not_written.bin"), {}, KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};
    const auto ranked = heuristic->rank(catalogAgainstPriority(2048), context);

    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, testId(0x01)); // highest priority
}

TEST(TestIngestorUhdKernelHeuristic, AFeaturesHashMismatchDegradesToDeclaredOrder)
{
    // RFC 0019 §6.3 check 1.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_kernel_heuristic_hash");
    const auto fixture
        = writeFixture(dir.path(), preferLargeTiles(), "max", "sha256:not_the_real_hash");

    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};
    const auto ranked = heuristic->rank(catalogAgainstPriority(2048), context);

    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, testId(0x01));
}

TEST(TestIngestorUhdKernelHeuristic, AKernelMissingAFeatureDegradesTheWholeRanking)
{
    // One kernel omits `tile_m`: the whole ranking falls back rather than mixing model and
    // fallback order.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_kernel_heuristic_partial");
    const auto fixture = writeFixture(dir.path(), preferLargeTiles());

    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    Catalog catalog;
    KernelDefinition incomplete;
    incomplete.kernelId = testId(0x03);
    incomplete.priority = 50; // between the two well-formed kernels
    catalog.entries = {kernelWith(0x01, 64, 100), incomplete, kernelWith(0x02, 128, 1)};
    catalog.bound["attention.seqlen"] = int64_t{2048};

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};
    const auto ranked = heuristic->rank(catalog, context);

    ASSERT_EQ(ranked.size(), 3U);
    EXPECT_EQ(ranked[0].kernelId, testId(0x01)); // priority 100
    EXPECT_EQ(ranked[1].kernelId, testId(0x03)); // priority 50
    EXPECT_EQ(ranked[2].kernelId, testId(0x02)); // priority 1
}

TEST(TestIngestorUhdKernelHeuristic, AListValuedTokenIsSkippedRatherThanFatal)
{
    // MetadataValue admits vector<int64_t>, which the feature extractor does not; the binding
    // is skipped.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_kernel_heuristic_list");
    const auto fixture = writeFixture(dir.path(), preferLargeTiles());

    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    auto catalog = catalogAgainstPriority(2048);
    catalog.bound["dims"] = std::vector<int64_t>{4, 8, 2048};

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};
    const auto ranked = heuristic->rank(catalog, context);

    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, testId(0x02)); // still the model's order
}

/// RFC 0019 §6.3 check 2 is `set($kernel.* axes) ⊆ set(UED.knobs)`, not equality: training
/// drops constant knobs, but the engine must still expose them.
TEST(TestIngestorUhdKernelHeuristic, AKnobTheModelDoesNotReadIsWarnedAboutAndRanksAnyway)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_kernel_heuristic_extra_knob");
    const auto fixture = writeFixture(dir.path(), preferLargeTiles());
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_WARN);
    const auto heuristic = makeKernelHeuristic(
        modelDescriptor(dir.path(), fixture), {}, {"tile_m", "split_k"}, FIELDS_WITH_SPLIT_K);
    ASSERT_NE(heuristic, nullptr);
    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};
    const auto ranked = heuristic->rank(catalogAgainstPriority(2048), context);
    ASSERT_EQ(ranked.size(), 2U);
    // At 2048 the model prefers the large tile, which declared order puts last.
    EXPECT_EQ(ranked.front().kernelId, testId(0x02));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, "split_k"))
        << "an unread dial has to be visible, even though it is not fatal";
}

TEST(TestIngestorUhdKernelHeuristic, AnAxisWithNoKnobIsRefused)
{
    // The model ranks on tile_m while the engine exposes no knobs.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_kernel_heuristic_no_knob");
    const auto fixture = writeFixture(dir.path(), preferLargeTiles());
    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, {}, FIELDS);
    ASSERT_NE(heuristic, nullptr);
    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};
    const auto ranked = heuristic->rank(catalogAgainstPriority(2048), context);
    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, testId(0x01));
}

/// RFC 0019 §6.3 check 2's other assertion, `F ⊆ KMD.fields`, re-checked at load: a field no
/// kernel carries leaves the slot unbound, silently.
TEST(TestIngestorUhdKernelHeuristic, AModelReadingAFieldTheKmdDoesNotDeclareIsRefusedAtLoad)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_kernel_heuristic_undeclared_field");
    const auto fixture = writeFixture(dir.path(), preferLargeTiles());
    const auto descriptor = modelDescriptor(dir.path(), fixture);
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);

    // KNOBS includes `tile_m`, so only the KMD check can fail.
    EXPECT_EQ(UhdKernelHeuristic::tryCreate(descriptor, "test-engine", KNOBS, {"warp_n"}), nullptr);

    // §5 step 8: the failure must name the UHD, not just the engine.
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "test model heuristic"))
        << "the refusal did not name the UHD that was refused";
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "test-engine"));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "tile_m"));

    // Control: a KMD declaring the field loads.
    EXPECT_NE(UhdKernelHeuristic::tryCreate(descriptor, "test-engine", KNOBS, FIELDS), nullptr);
}

/// RFC 0019 §6.3 check 4. A short row would hit TreeDataAdapter's missing-value branch and
/// still score; the features hash cannot catch a width mismatch.
TEST(TestIngestorUhdKernelHeuristic, AModelWhoseFeatureCountDisagreesWithItsSignatureIsRefused)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_kernel_heuristic_width");

    // Not writeFixture, which derives num_features from the signature.
    hipdnn_test_sdk::utilities::GbdtModelTestBuilder model;
    model.setFeaturesHash(uhd::FeatureExtractor::computeHash(SIGNATURE))
        .setNumFeatures(static_cast<int32_t>(SIGNATURE.size()) + 1)
        .setTrainingArches({"gfx942"})
        .addTree(preferLargeTiles());
    ASSERT_TRUE(model.buildToFile((dir.path() / "model.bin").string()));

    EXPECT_EQ(UhdKernelHeuristic::tryCreate(
                  modelDescriptor(dir.path(), "model.bin"), "test-engine", KNOBS, FIELDS),
              nullptr);
}

/// RFC 0019 §8.3: exact gcnArchName, then `default`; resolved at first rank() (§9.2).
TEST(TestIngestorUhdKernelHeuristic, AnArchSpecificModelOutranksTheDefaultOne)
{
    // The default prefers small tiles and gfx942 large, so the winner identifies the model.
    const hipdnn_test_sdk::utilities::ScopedDirectory defaultDir("uhd_arch_specific_default");
    const hipdnn_test_sdk::utilities::ScopedDirectory archDir("uhd_arch_specific_gfx942");
    const auto fallback = writeFixture(defaultDir.path(), preferSmallTiles());
    const auto specific = writeFixture(archDir.path(), preferLargeTiles());

    const std::map<std::string, HeuristicDescriptor> byArch{
        {"default", modelDescriptor(defaultDir.path(), fallback)},
        {"gfx942", modelDescriptor(archDir.path(), specific)}};

    const auto heuristic = makeKernelHeuristic(modelDescriptor(defaultDir.path(), fallback),
                                               {},
                                               KNOBS,
                                               FIELDS,
                                               RankersByMetric{{"tflops", byArch}});
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};
    const auto ranked = heuristic->rank(catalogAgainstPriority(2048), context);

    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, testId(0x02))
        << "the gfx942 model should have ranked, not the default one";
}

TEST(TestIngestorUhdKernelHeuristic, AnUnnamedArchFallsBackToDefault)
{
    // §8.3: an unnamed device takes `default`, which prefers small tiles.
    const hipdnn_test_sdk::utilities::ScopedDirectory defaultDir("uhd_arch_fallback_default");
    const hipdnn_test_sdk::utilities::ScopedDirectory archDir("uhd_arch_fallback_gfx942");
    const auto fallback = writeFixture(defaultDir.path(), preferSmallTiles());
    const auto specific = writeFixture(archDir.path(), preferLargeTiles());

    const std::map<std::string, HeuristicDescriptor> byArch{
        {"default", modelDescriptor(defaultDir.path(), fallback)},
        {"gfx942", modelDescriptor(archDir.path(), specific)}};

    const auto heuristic = makeKernelHeuristic(modelDescriptor(defaultDir.path(), fallback),
                                               {},
                                               KNOBS,
                                               FIELDS,
                                               RankersByMetric{{"tflops", byArch}});
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    auto properties = gfx942();
    properties.gcnArchName = "gfx1100"; // named by neither entry
    const MatchContext context{graph, 0, properties};
    const auto ranked = heuristic->rank(catalogAgainstPriority(2048), context);

    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, testId(0x01))
        << "an unnamed architecture should rank by the default model";
}

TEST(TestIngestorUhdKernelHeuristic, ArchResolutionIsStableAcrossCalls)
{
    // §9.2 caches what it loads; the cache must not change answers or cross architectures.
    const hipdnn_test_sdk::utilities::ScopedDirectory defaultDir("uhd_arch_cached_default");
    const hipdnn_test_sdk::utilities::ScopedDirectory archDir("uhd_arch_cached_gfx942");
    const auto fallback = writeFixture(defaultDir.path(), preferSmallTiles());
    const auto specific = writeFixture(archDir.path(), preferLargeTiles());

    const std::map<std::string, HeuristicDescriptor> byArch{
        {"default", modelDescriptor(defaultDir.path(), fallback)},
        {"gfx942", modelDescriptor(archDir.path(), specific)}};

    const auto heuristic = makeKernelHeuristic(modelDescriptor(defaultDir.path(), fallback),
                                               {},
                                               KNOBS,
                                               FIELDS,
                                               RankersByMetric{{"tflops", byArch}});
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto onGfx942 = gfx942();
    auto onOther = gfx942();
    onOther.gcnArchName = "gfx1100";

    const MatchContext gfx942Context{graph, 0, onGfx942};
    const MatchContext otherContext{graph, 0, onOther};

    const auto first = heuristic->rank(catalogAgainstPriority(2048), gfx942Context);
    const auto other = heuristic->rank(catalogAgainstPriority(2048), otherContext);
    const auto again = heuristic->rank(catalogAgainstPriority(2048), gfx942Context);

    ASSERT_EQ(first.size(), 2U);
    EXPECT_EQ(first.front().kernelId, again.front().kernelId) << "the cache changed its answer";
    EXPECT_NE(first.front().kernelId, other.front().kernelId)
        << "two architectures were served the same model";
}

/// RFC 0019.13 §15.2: "an ordered sequence of `(UKD id, score)`, winner first". Engine
/// selection reads the top score as the engine's figure of merit.
TEST(TestIngestorUhdKernelHeuristic, SelectionReturnsIdsWithScoresWinnerFirst)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_scored_form");
    const auto fixture = writeFixture(dir.path(), preferLargeTiles());

    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};
    const auto scored = heuristic->rankScored(catalogAgainstPriority(2048), context);

    ASSERT_EQ(scored.size(), 2U);
    EXPECT_EQ(scored.front().kernelId, testId(0x02)) << "winner is not first";
    // Real scores, not placeholders: the top one is a figure of merit.
    EXPECT_GT(scored.front().score, scored.back().score);
    EXPECT_TRUE(std::isfinite(scored.front().score));

    // rank() and rankScored() share one ordering.
    const auto ordered = heuristic->rank(catalogAgainstPriority(2048), context);
    ASSERT_EQ(ordered.size(), scored.size());
    for(size_t i = 0; i < ordered.size(); ++i)
    {
        EXPECT_EQ(ordered[i].kernelId, scored[i].kernelId) << "views disagree at " << i;
    }
}

TEST(TestIngestorUhdKernelHeuristic, ADegradedRankingReportsTheZeroTheRfcPrescribes)
{
    // RFC 0019 §5 step 7: declared order carries no model score, so it reports 0.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_scored_degraded");
    const auto fixture = writeFixture(dir.path(), preferLargeTiles());

    // An axis the UED does not expose degrades ranking (§6.3); an ignored knob would not.
    const auto heuristic = makeKernelHeuristic(
        modelDescriptor(dir.path(), fixture), {}, {"split_k"}, FIELDS_WITH_SPLIT_K);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};
    const auto scored = heuristic->rankScored(catalogAgainstPriority(2048), context);

    ASSERT_EQ(scored.size(), 2U);
    EXPECT_DOUBLE_EQ(scored.front().score, 0.0)
        << "a fallback ordering invented a score it did not compute";
}

TEST(TestIngestorUhdKernelHeuristic, AnObjectiveContradictingItsMetricIsRefusedAtParse)
{
    // RFC 0019 §4.4: a registered metric fixes its direction and `objective` only restates
    // it. The refusal happens at parse, so this asserts on the document.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_objective_contradicts_metric");
    const auto fixture = writeFixture(dir.path(), preferLargeTiles(), "min");
    const auto path = dir.path() / "test.uhd.json";

    EXPECT_FALSE(
        parseUhd(uhdDocument(fixture.modelFileName, "min", /*calibrated=*/true, "tflops"), path)
            .has_value())
        << "a minimising tflops UHD loaded";
    EXPECT_FALSE(
        parseUhd(uhdDocument(fixture.modelFileName, "max", /*calibrated=*/true, "time"), path)
            .has_value())
        << "a maximising time UHD loaded";

    // A calibrated value needs a named metric.
    EXPECT_FALSE(parseUhd(uhdDocument(fixture.modelFileName, "min", /*calibrated=*/true, ""), path)
                     .has_value())
        << "a calibrated UHD naming no metric loaded";

    // Controls: a consistent cost model and a metric-less uncalibrated ranker both load.
    EXPECT_TRUE(
        parseUhd(uhdDocument(fixture.modelFileName, "min", /*calibrated=*/true, "time"), path)
            .has_value());
    EXPECT_TRUE(parseUhd(uhdDocument(fixture.modelFileName, "min", /*calibrated=*/false, ""), path)
                    .has_value());
}

TEST(TestIngestorUhdKernelHeuristic, ATransformTheRuntimeCannotInvertIsRefusedAtLoad)
{
    // RFC 0019 §4, §11.3: an uninvertible transform would report values in the wrong units
    // while still ordering correctly, so the vocabulary is closed at parse.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_unknown_transform");
    const auto fixture = writeFixture(dir.path(), preferLargeTiles());

    auto document = uhdDocument(fixture.modelFileName, "max", /*calibrated=*/true);
    document["score"]["transform"] = "zscore";
    EXPECT_FALSE(parseUhd(document, dir.path() / "test.uhd.json").has_value())
        << "a UHD naming a transform with no inverse loaded";

    // Every name score_transform advertises still loads.
    for(const auto* known : uhd::score_transform::SUPPORTED_TRANSFORMS)
    {
        if(*known == '\0')
        {
            continue; // "no transform" is spelled as an absent key, which the schema requires
        }
        document["score"]["transform"] = known;
        EXPECT_TRUE(parseUhd(document, dir.path() / "test.uhd.json").has_value())
            << "a supported transform was refused: " << known;
    }
}

TEST(TestIngestorUhdKernelHeuristic, ACalibratedModelReportsItsTopScoreAsTheEngineEstimate)
{
    // RFC 0019 §11.1: the estimate must be the score of the same kernel the ranking put first.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_engine_estimate");
    const auto fixture
        = writeFixture(dir.path(), preferLargeTiles(), "max", {}, /*calibrated=*/true);

    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};

    std::string modelId;
    const auto calibrated
        = heuristic->calibratedRanking(catalogAgainstPriority(2048), context, modelId);
    const auto selected = heuristic->rankScored(catalogAgainstPriority(2048), context);
    ASSERT_FALSE(calibrated.empty());
    ASSERT_FALSE(selected.empty());
    EXPECT_EQ(calibrated.front().kernelId, selected.front().kernelId)
        << "the estimate describes a kernel selection would not run";
    EXPECT_DOUBLE_EQ(calibrated.front().score, selected.front().score);
    EXPECT_GT(calibrated.front().score, 0.0)
        << "a real estimate must outrank the 0 a declining engine reports";
    EXPECT_FALSE(modelId.empty()) << "a calibrated ranking did not name the model it came from";
}

TEST(TestIngestorUhdKernelHeuristic, APerArchCalibratedModelEstimatesOnAnArchitectureItNames)
{
    // §8.3 exact match with no `default`: the estimate must come from the per-arch model.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_estimate_per_arch");
    const auto onNine42
        = writeFixture(dir.path(), preferLargeTiles(), "max", {}, /*calibrated=*/true);

    const std::map<std::string, HeuristicDescriptor> byArch{
        {"gfx942", modelDescriptor(dir.path(), onNine42)}};

    // No descriptor: there is no `default` for the loader to have resolved.
    const auto heuristic = makeKernelHeuristic(
        std::nullopt, "test-engine", KNOBS, FIELDS, RankersByMetric{{"tflops", byArch}});
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};

    EXPECT_GT(engineEstimate(*heuristic, catalogAgainstPriority(2048), context), 0.0)
        << "an engine holding a calibrated model for this architecture declined to estimate";

    // An architecture the UED does not name still reports §5 step 7's zero.
    auto unnamed = gfx942();
    unnamed.gcnArchName = "gfx1100";
    EXPECT_DOUBLE_EQ(
        engineEstimate(*heuristic, catalogAgainstPriority(2048), MatchContext{graph, 0, unnamed}),
        0.0);
}

TEST(TestIngestorUhdKernelHeuristic, AnUncalibratedArchModelIsNotReportedAsCalibrated)
{
    // A calibrated `default` must not make the uncalibrated gfx942 model's score count as
    // calibrated (§11.3).
    const hipdnn_test_sdk::utilities::ScopedDirectory defaultDir("uhd_estimate_mixed_default");
    const hipdnn_test_sdk::utilities::ScopedDirectory archDir("uhd_estimate_mixed_gfx942");
    const auto fallback
        = writeFixture(defaultDir.path(), preferSmallTiles(), "max", {}, /*calibrated=*/true);
    const auto specific
        = writeFixture(archDir.path(), preferLargeTiles(), "max", {}, /*calibrated=*/false);

    const std::map<std::string, HeuristicDescriptor> byArch{
        {"default", modelDescriptor(defaultDir.path(), fallback)},
        {"gfx942", modelDescriptor(archDir.path(), specific)}};

    const auto heuristic = makeKernelHeuristic(modelDescriptor(defaultDir.path(), fallback),
                                               "test-engine",
                                               KNOBS,
                                               FIELDS,
                                               RankersByMetric{{"tflops", byArch}});
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};

    // The winner identifies the gfx942 model as the one ranking.
    const auto ranked = heuristic->rankScored(catalogAgainstPriority(2048), context);
    ASSERT_EQ(ranked.size(), 2U);
    ASSERT_EQ(ranked.front().kernelId, testId(0x02)) << "the default model ranked, not gfx942's";

    EXPECT_DOUBLE_EQ(engineEstimate(*heuristic, catalogAgainstPriority(2048), context), 0.0)
        << "an uncalibrated model's score was reported as a calibrated value";

    // gfx1100 falls through to the calibrated `default`, but that model was trained for gfx942
    // only: it still ranks (§9.3) but withholds the estimate (§5 step 8).
    auto unnamed = gfx942();
    unnamed.gcnArchName = "gfx1100";
    const MatchContext unnamedContext{graph, 0, unnamed};
    EXPECT_DOUBLE_EQ(engineEstimate(*heuristic, catalogAgainstPriority(2048), unnamedContext), 0.0);
    EXPECT_FALSE(heuristic->rankScored(catalogAgainstPriority(2048), unnamedContext).empty())
        << "withholding the estimate stopped the engine selecting";
}

TEST(TestIngestorUhdKernelHeuristic, AnUncalibratedModelEstimatesZero)
{
    // §11.3: an uncalibrated score is not on a cross-engine scale, so the estimate is 0
    // (§5 step 7); it still ranks.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_no_estimate");
    const auto fixture
        = writeFixture(dir.path(), preferLargeTiles(), "max", {}, /*calibrated=*/false);

    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};

    EXPECT_DOUBLE_EQ(engineEstimate(*heuristic, catalogAgainstPriority(2048), context), 0.0);
    EXPECT_FALSE(heuristic->rankScored(catalogAgainstPriority(2048), context).empty())
        << "reporting a zero estimate must not stop it selecting";
}

TEST(TestIngestorKernelHeuristicEstimate, AnEngineWithNoModelOffersNoCalibratedRanking)
{
    // Reporting the priority it sorted by would put an arbitrary integer on a metric's scale.
    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};

    const UnrankedKernelHeuristic heuristic;
    std::string modelId;
    EXPECT_TRUE(
        heuristic.calibratedRanking(catalogAgainstPriority(2048), context, modelId).empty());
    EXPECT_TRUE(modelId.empty()) << "no model ranked, so none may be named";
}

TEST(TestIngestorUhdKernelHeuristic, AModelWhoseTransformGoesOutOfDomainDoesNotCorruptTheSort)
{
    // `exp` inverts as log(raw), so a negative prediction yields NaN, which would break
    // std::stable_sort's strict weak ordering.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_out_of_domain");
    const auto fixture = writeFixture(dir.path(), preferNegativeScores(), "max", {}, false, "exp");

    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};

    const auto scored = heuristic->rankScored(catalogAgainstPriority(2048), context);
    ASSERT_EQ(scored.size(), 2U);

    // NaN must surface as the 0 that means "no measurement".
    for(const auto& entry : scored)
    {
        EXPECT_TRUE(std::isfinite(entry.score)) << "a non-finite score reached the caller";
        EXPECT_DOUBLE_EQ(entry.score, 0.0);
    }

    // The engine estimate uses the same sentinel.
    EXPECT_DOUBLE_EQ(engineEstimate(*heuristic, catalogAgainstPriority(2048), context), 0.0);
}

TEST(TestIngestorUhdKernelHeuristic, ACalibratedModelCannotReportANegativeThroughput)
{
    // expm1 of a negative log1p prediction is finite but negative, which a throughput cannot
    // be (§11.3); it is bounded to §5 step 7's 0 so it cannot rank beneath "no measurement".
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_negative_tflops");
    const auto fixture
        = writeFixture(dir.path(), preferNegativeScores(), "max", {}, /*calibrated=*/true, "log1p");

    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};

    const auto scored = heuristic->rankScored(catalogAgainstPriority(2048), context);
    ASSERT_EQ(scored.size(), 2U);
    for(const auto& entry : scored)
    {
        EXPECT_GE(entry.score, 0.0) << "a negative throughput reached the caller";
        EXPECT_DOUBLE_EQ(entry.score, 0.0);
    }

    EXPECT_DOUBLE_EQ(engineEstimate(*heuristic, catalogAgainstPriority(2048), context), 0.0)
        << "the engine estimate went below the RFC's zero";
}

TEST(TestIngestorUhdKernelHeuristic, AMinObjectiveScoresBelowZeroWithoutThatBeingAnError)
{
    // `objective: min` negates a cost, so negative *oriented* scores are normal; only a
    // negative recovered value is out of range.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_min_negative_oriented");
    const auto fixture
        = writeFixture(dir.path(), preferLargeTiles(), "min", {}, /*calibrated=*/false, "identity");

    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties, "time"};

    const auto scored = heuristic->rankScored(catalogAgainstPriority(2048), context);
    ASSERT_EQ(scored.size(), 2U);
    EXPECT_LT(scored.front().score, 0.0) << "a min objective's oriented scores were clamped";
    EXPECT_GT(scored.front().score, scored.back().score) << "the cheaper candidate did not win";
}

TEST(TestIngestorUhdKernelHeuristic, AnUnmeasuredCandidateSortsLastUnderAMinObjectiveToo)
{
    // The reported 0 for "no measurement" exceeds every oriented min score, so the ordering
    // key must differ from the reported score (RFC 0019 §5 step 7).
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_min_unmeasured_last");
    const auto fixture = writeFixture(
        dir.path(), oneUsableOneOutOfRange(), "min", {}, /*calibrated=*/false, "identity");

    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties, "time"};

    const auto scored = heuristic->rankScored(catalogAgainstPriority(2048), context);
    ASSERT_EQ(scored.size(), 2U);

    // The winner's reported score (-4.0, a negated cost) is below the unmeasured 0.
    EXPECT_LT(scored.front().score, 0.0) << "the measured candidate did not come first";
    EXPECT_DOUBLE_EQ(scored.back().score, 0.0) << "the unmeasured candidate is not reporting 0";
}

TEST(TestIngestorUhdKernelHeuristic, AZeroCostPredictionDoesNotWinUnderAMinObjective)
{
    // RFC 0019 §8.3 accepts only a strictly positive prediction; a zero cost oriented for
    // `min` would be -0 and outrank every real candidate.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_min_zero_last");
    const auto fixture
        = writeFixture(dir.path(), oneUsableOneZero(), "min", {}, /*calibrated=*/false, "identity");

    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties, "time"};

    const auto scored = heuristic->rankScored(catalogAgainstPriority(2048), context);
    ASSERT_EQ(scored.size(), 2U);
    EXPECT_LT(scored.front().score, 0.0) << "the zero-cost candidate outranked a measured one";
    EXPECT_DOUBLE_EQ(scored.back().score, 0.0);
}

TEST(TestIngestorUhdKernelHeuristic, AMetriclessRankerOrdersOnSignedScores)
{
    // RFC 0019 §8.3's positivity applies only to physical scores; a metric-less `identity`
    // ranker orders on its own scale, where -0.5 beats -2.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_metricless_signed");
    const auto fixture
        = writeFixture(dir.path(), preferNegativeScores(), "max", {}, /*calibrated=*/false);
    auto descriptor = modelDescriptor(dir.path(), fixture);
    descriptor.score.metric.clear();

    const auto heuristic = makeKernelHeuristic(descriptor, {}, KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};

    const auto scored = heuristic->rankScored(catalogAgainstPriority(2048), context);
    ASSERT_EQ(scored.size(), 2U);
    EXPECT_EQ(scored.front().kernelId, testId(0x02)) << "the model's preference was discarded";
    EXPECT_DOUBLE_EQ(scored.front().score, -0.5);
    EXPECT_DOUBLE_EQ(scored.back().score, -2.0);

    // Under log1p the inverse of a negative prediction is out of range again.
    const hipdnn_test_sdk::utilities::ScopedDirectory logDir("uhd_metricless_log1p");
    const auto logFixture = writeFixture(
        logDir.path(), preferNegativeScores(), "max", {}, /*calibrated=*/false, "log1p");
    auto logDescriptor = modelDescriptor(logDir.path(), logFixture);
    logDescriptor.score.metric.clear();
    const auto bounded = makeKernelHeuristic(logDescriptor, {}, KNOBS, FIELDS);
    ASSERT_NE(bounded, nullptr);
    const auto declared = bounded->rankScored(catalogAgainstPriority(2048), context);
    ASSERT_EQ(declared.size(), 2U);
    EXPECT_EQ(declared.front().kernelId, testId(0x01)) << "declared order did not decide";
    EXPECT_DOUBLE_EQ(declared.front().score, 0.0);
}

TEST(TestIngestorUhdKernelHeuristic, AModelReadingAListFieldElementIsAdmittedOnTheField)
{
    // List elements bind as `tile[0]`, `tile[1]`, ... while the KMD and UED declare `tile`.
    const std::vector<nlohmann::json> signature = {"$kernel.tile[1]", "$attention.seqlen"};
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_indexed_kernel_field");
    const auto fixture = writeFixture(
        dir.path(), preferLargeTiles(), "max", {}, /*calibrated=*/true, "identity", signature);

    const auto heuristic = makeKernelHeuristic(modelDescriptor(dir.path(), fixture),
                                               {},
                                               {"tile"},
                                               std::unordered_set<std::string>{"tile"});
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};
    Catalog catalog = catalogAgainstPriority(2048);
    for(auto& kernel : catalog.entries)
    {
        kernel.metadata["tile"]
            = std::vector<int64_t>{16, std::get<int64_t>(kernel.metadata.at("tile_m"))};
    }

    const auto scored = heuristic->rankScored(catalog, context);
    ASSERT_EQ(scored.size(), 2U);
    EXPECT_EQ(scored.front().kernelId, testId(0x02)) << "the model was refused admission";
    EXPECT_DOUBLE_EQ(scored.front().score, 9.0);
}

TEST(TestIngestorUhdKernelHeuristic, ANegativeThroughputIsReportedAsAnErrorNotSwallowed)
{
    // RFC 0019 §12: a discarded score must be visible. ERROR, not WARN: the number is wrong,
    // not merely out of distribution (§9.3).
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_negative_reported");
    const auto fixture
        = writeFixture(dir.path(), preferNegativeScores(), "max", {}, /*calibrated=*/true, "log1p");

    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};
    (void)heuristic->rankScored(catalogAgainstPriority(2048), context);

    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "cannot take"))
        << "a model predicting a negative throughput was discarded without a word";

    // A total failure is reported differently from losing one candidate.
    EXPECT_TRUE(recorder.hasLogContaining("Every candidate was affected"));

    // Once per heuristic: the condition is a property of the model and recurs on every graph.
    const auto after = recorder.countLogsAtLevel(HIPDNN_SEV_ERROR);
    (void)heuristic->rankScored(catalogAgainstPriority(2048), context);
    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_ERROR), after) << "the report repeated";
}

TEST(TestIngestorUhdKernelHeuristic, APartiallyAffectedRankingSaysTheModelStillDecidedTheRest)
{
    // One candidate out of range leaves a ranking that is still mostly the model's.
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_partial_out_of_range");
    const auto fixture = writeFixture(
        dir.path(), oneUsableOneOutOfRange(), "max", {}, /*calibrated=*/true, "identity");

    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};
    const auto scored = heuristic->rankScored(catalogAgainstPriority(2048), context);

    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "1 of 2 candidates"));
    EXPECT_TRUE(recorder.hasLogContaining("The remaining candidates ranked on the model."));

    // The usable candidate still won on its score.
    ASSERT_EQ(scored.size(), 2U);
    EXPECT_GT(scored.front().score, 0.0);
    EXPECT_DOUBLE_EQ(scored.back().score, 0.0);
}

TEST(TestIngestorUhdKernelHeuristic, PerArchModelsRankWithoutADefaultEntry)
{
    // RFC 0019 §8.3: exact gcnArchName comes first, so no `default` is needed for named arches.
    const hipdnn_test_sdk::utilities::ScopedDirectory gfx942Dir("uhd_no_default_942");
    const hipdnn_test_sdk::utilities::ScopedDirectory gfx950Dir("uhd_no_default_950");
    const auto onNine42 = writeFixture(gfx942Dir.path(), preferLargeTiles());
    const auto onNine50 = writeFixture(gfx950Dir.path(), preferSmallTiles());

    const std::map<std::string, HeuristicDescriptor> byArch{
        {"gfx942", modelDescriptor(gfx942Dir.path(), onNine42)},
        {"gfx950", modelDescriptor(gfx950Dir.path(), onNine50)}};

    // No descriptor: there is no `default` for the loader to have resolved.
    const auto heuristic = makeKernelHeuristic(
        std::nullopt, "test-engine", KNOBS, FIELDS, RankersByMetric{{"tflops", byArch}});
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};
    const auto ranked = heuristic->rank(catalogAgainstPriority(2048), context);

    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, testId(0x02))
        << "the gfx942 model did not rank; this is declared order";
}

TEST(TestIngestorUhdKernelHeuristic, AnArchNamedModelDoesNotRankAnArchitectureItDoesNotName)
{
    // §8.3: exact, then `default`, then unavailable. `{"gfx950": X}` does not make X universal.
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_wrong_arch_only");
    const auto onNine50 = writeFixture(dir.path(), preferLargeTiles());

    const std::map<std::string, HeuristicDescriptor> byArch{
        {"gfx950", modelDescriptor(dir.path(), onNine50)}};

    const auto heuristic = makeKernelHeuristic(
        std::nullopt, "test-engine", KNOBS, FIELDS, RankersByMetric{{"tflops", byArch}});
    ASSERT_NE(heuristic, nullptr);

    // Named by no entry.
    const testing::TestGraph graph;
    auto properties = gfx942();
    properties.gcnArchName = "gfx1100";
    const MatchContext context{graph, 0, properties};
    const auto ranked = heuristic->rank(catalogAgainstPriority(2048), context);

    // Declared order; the gfx950 model would have put 0x02 first.
    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, testId(0x01))
        << "a model named only for gfx950 ranked a gfx1100 device";

    // The log names the metric the request ranked by.
    EXPECT_TRUE(recorder.hasLogContaining("names no model for 'gfx1100' in metric 'tflops'"));
}

TEST(TestIngestorUhdKernelHeuristic, ADefaultStillCoversAnArchitectureNotNamedExplicitly)
{
    // §8.3's second step: a declared `default` still covers unnamed architectures.
    const hipdnn_test_sdk::utilities::ScopedDirectory defaultDir("uhd_default_covers");
    const hipdnn_test_sdk::utilities::ScopedDirectory archDir("uhd_default_covers_950");
    const auto fallback = writeFixture(defaultDir.path(), preferLargeTiles());
    const auto specific = writeFixture(archDir.path(), preferSmallTiles());

    const std::map<std::string, HeuristicDescriptor> byArch{
        {"default", modelDescriptor(defaultDir.path(), fallback)},
        {"gfx950", modelDescriptor(archDir.path(), specific)}};

    const auto heuristic = makeKernelHeuristic(modelDescriptor(defaultDir.path(), fallback),
                                               "test-engine",
                                               KNOBS,
                                               FIELDS,
                                               RankersByMetric{{"tflops", byArch}});
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    auto properties = gfx942();
    properties.gcnArchName = "gfx1100";
    const MatchContext context{graph, 0, properties};
    const auto ranked = heuristic->rank(catalogAgainstPriority(2048), context);

    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, testId(0x02)) << "the default model did not cover gfx1100";
}

/// RFC 0019 §12: the trace must say whether the model or a fallback decided. Every degraded path
/// is a legal ranking, so these assert the reported provenance, not the winner.
struct SelectionCondition
{
    const char* name;
    const char* expectedProvenance;
};

class TestIngestorUhdProvenance : public ::testing::TestWithParam<SelectionCondition>
{
};

/// A UHD well-formed except in the one way @p condition describes.
std::shared_ptr<IKernelHeuristic> heuristicForCondition( // NOLINT(misc-use-internal-linkage)
    const std::string& condition,
    const std::filesystem::path& dir)
{
    if(condition == "healthy_model")
    {
        const auto fixture = writeFixture(dir, preferLargeTiles());
        return makeKernelHeuristic(modelDescriptor(dir, fixture), "e", KNOBS, FIELDS);
    }
    if(condition == "features_hash_disagrees")
    {
        // §6.3 check 1.
        const auto fixture
            = writeFixture(dir, preferLargeTiles(), "max", "sha256:not_the_real_hash");
        return makeKernelHeuristic(modelDescriptor(dir, fixture), "e", KNOBS, FIELDS);
    }
    if(condition == "knobs_disagree_with_axes")
    {
        // §6.3 check 2: the model ranks on an axis the UED never exposed.
        const auto fixture = writeFixture(dir, preferLargeTiles());
        return makeKernelHeuristic(
            modelDescriptor(dir, fixture), "e", {"split_k"}, FIELDS_WITH_SPLIT_K);
    }
    if(condition == "objective_contradicts_metric")
    {
        // §4.4: a tflops model declaring `min` is refused at parse, so the factory gets no
        // descriptor.
        const auto fixture = writeFixture(dir, preferLargeTiles(), "min");
        return makeKernelHeuristic(
            parseUhd(uhdDocument(fixture.modelFileName, "min", /*calibrated=*/true, "tflops"),
                     dir / "test.uhd.json"),
            "e",
            KNOBS,
            FIELDS);
    }
    if(condition == "no_uhd_at_all")
    {
        // §5 step 6: shipping no heuristic is valid.
        return makeKernelHeuristic(std::nullopt, "e", KNOBS, FIELDS);
    }
    if(condition == "arch_not_covered")
    {
        // §8.3 third step: no model for this architecture and no default.
        const auto fixture = writeFixture(dir, preferLargeTiles());
        const std::map<std::string, HeuristicDescriptor> byArch{
            {"gfx950", modelDescriptor(dir, fixture)}};
        return makeKernelHeuristic(
            std::nullopt, "e", KNOBS, FIELDS, RankersByMetric{{"tflops", byArch}});
    }
    return nullptr;
}

TEST_P(TestIngestorUhdProvenance, TheRuntimeReportsWhichOfTheThreeDecided)
{
    const auto condition = GetParam();
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(std::string("uhd_prov_")
                                                          + condition.name);

    const auto heuristic = heuristicForCondition(condition.name, dir.path());
    ASSERT_NE(heuristic, nullptr) << "unhandled condition: " << condition.name;

    // §5 step 7: every condition still ranks.
    const testing::TestGraph graph;
    auto properties = gfx942();
    if(std::string(condition.name) == "arch_not_covered")
    {
        properties.gcnArchName = "gfx1100";
    }
    const MatchContext context{graph, 0, properties};

    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    const auto ranked = heuristic->rankScored(catalogAgainstPriority(2048), context);

    EXPECT_EQ(ranked.size(), 2U) << "a degraded ranking still has to answer";

    // The logged trace, not a context-free accessor: an arch resolver's provenance depends on
    // the device.
    EXPECT_TRUE(
        recorder.hasLogContaining(std::string("decided_by=") + condition.expectedProvenance))
        << "the trace did not report " << condition.expectedProvenance;
}

INSTANTIATE_TEST_SUITE_P(
    SelectionConditions,
    TestIngestorUhdProvenance,
    ::testing::Values(SelectionCondition{"healthy_model", "model"},
                      SelectionCondition{"features_hash_disagrees", "declared_order"},
                      SelectionCondition{"knobs_disagree_with_axes", "declared_order"},
                      SelectionCondition{"objective_contradicts_metric", "declared_order"},
                      SelectionCondition{"no_uhd_at_all", "declared_order"},
                      SelectionCondition{"arch_not_covered", "declared_order"}),
    [](const ::testing::TestParamInfo<SelectionCondition>& info) {
        return std::string(info.param.name);
    });

/// RFC 0019 §6.3 check 2 compares two sets; an emptied side compares vacuously, so the cases
/// below pin each side and then the comparison itself.
namespace
{

/// The shared StubDeviceResolver reports gfx000; this matches the fixtures' training arch.
class Gfx942DeviceResolver : public IDeviceResolver<testing::StubHandle>
{
public:
    DeviceId deviceId(const testing::StubHandle& /*handle*/) const override
    {
        return 0;
    }

    const DeviceProperties& deviceProperties(DeviceId /*deviceId*/) const override
    {
        return _properties;
    }

private:
    DeviceProperties _properties = gfx942();
};

KernelDescriptor kernelDescriptorWith(uint8_t tag, int64_t tileM, int64_t priority)
{
    KernelDescriptor kernel;
    kernel.id = testId(tag);
    kernel.name = "kernel_tile_" + std::to_string(tileM);
    kernel.source.sourceFile = "Test.cpp";
    kernel.source.entryPoint = "TestKernel";
    kernel.metadata = {{"tile_m", MetadataValue{tileM}}};
    kernel.priority = priority;
    return kernel;
}

/// A whole engine's descriptor set: a UED exposing @p knobs, the model in @p dir, and a
/// catalog whose declared order is the opposite of what that model prefers. `split_k` is in
/// the schema so a case can expose it as a knob (GenericEngine refuses undeclared knobs).
DescriptorSet engineSetRankingOnTileM(const std::filesystem::path& dir,
                                      const Fixture& fixture,
                                      std::vector<std::string> knobs)
{
    DescriptorSet set;
    set.engine.id = testing::ENGINE_ID;
    set.engine.name = "test:uhd_knob_contract";
    set.engine.heuristicId = testId(0xEE);
    set.engine.metadataSchemaId = testing::SCHEMA_ID;
    set.engine.knobs = std::move(knobs);
    set.engine.graphMatchNativeSymbol = testing::GRAPH_MATCH_SYMBOL;

    set.schema.id = testing::SCHEMA_ID;
    set.schema.name = "test schema";
    set.schema.fields = {{"tile_m", MetadataType::INT, MetadataValue{int64_t{64}}},
                         {"split_k", MetadataType::INT, MetadataValue{int64_t{1}}},
                         {testing::BLOCK_SIZE, MetadataType::INT, MetadataValue{int64_t{64}}}};

    set.heuristic = modelDescriptor(dir, fixture);
    set.dispatches = testing::makeStubDispatches();

    KernelDescriptorPack pack;
    pack.id = testing::PACK_ID;
    pack.name = "test pack";
    pack.engineId = testing::ENGINE_ID;
    pack.dispatchId = testing::DISPATCH_ID;
    // Declared order and the model disagree.
    pack.kernels = {kernelDescriptorWith(0x01, 64, 100), kernelDescriptorWith(0x02, 128, 1)};
    set.packs = {std::move(pack)};

    return set;
}

/// Builds the engine @p set describes via makeEngine() and ranks its catalog once, returning
/// what RFC 0019 §12's trace said decided. makeEngine() moves the UED, so it is where the knobs
/// must be read.
std::string provenanceOfEngineRanking(DescriptorSet set)
{
    const testing::ScopedTestSymbols symbols;
    const testing::StubWorkspaceHandler handler;
    const testing::ScopedDispatchRegistration<testing::StubHandle> dispatch(
        "hipdnn.kernel_ingestor.test.dispatch", handler);

    const Gfx942DeviceResolver resolver;
    auto engine = makeEngine<testing::StubHandle, testing::StubSettings, testing::StubContext>(
        std::move(set), resolver);
    if(engine == nullptr)
    {
        return "<no engine>";
    }

    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    const testing::StubHandle handle;
    const testing::TestGraph graph(testing::makeGraphId(0x71));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper emptyConfig(nullptr, 0);
    testing::StubContext context;
    engine->initializeExecutionContext(handle, graph, emptyConfig, context);

    if(recorder.hasLogContaining("decided_by=model"))
    {
        return "model";
    }
    if(recorder.hasLogContaining("decided_by=declared_order"))
    {
        return "declared_order";
    }
    return "<no trace>: " + recorder.getRecordedLogsAsString();
}

} // namespace

TEST(TestIngestorUhdKernelHeuristic, TheAxisSetReadsRfc0019sBareReferenceSpelling)
{
    // Canonical raw and inline entries both expose the axes consumed by the model.
    const auto axes = kernelAxesOf(uhd::FeatureExtractor(SIGNATURE));

    EXPECT_EQ(axes, (std::unordered_set<std::string>{"tile_m"}));

    // Problem namespaces do not become kernel axes.
    const auto mixed = kernelAxesOf(
        uhd::FeatureExtractor({"$kernel.tile_m",
                               "$attention.seqlen",
                               nlohmann::json::parse(R"({"*": ["$kernel.split_k", 2]})")}));
    EXPECT_EQ(mixed, (std::unordered_set<std::string>{"tile_m", "split_k"}));
}

TEST(TestIngestorUhdKernelHeuristic, ABareReferenceSignatureStillSatisfiesTheKnobAxisCheck)
{
    // A UED exposing exactly the knob its model ranks on must get its model.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_bare_signature");
    const auto fixture = writeFixture(
        dir.path(), preferLargeTiles(), "max", {}, /*calibrated=*/true, "identity", SIGNATURE);

    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), "e", KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};

    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    const auto ranked = heuristic->rankScored(catalogAgainstPriority(2048), context);

    EXPECT_EQ(ranked.size(), 2U);
    EXPECT_TRUE(recorder.hasLogContaining("decided_by=model"))
        << "a conformant bare-reference signature was refused its model";
}

TEST(TestIngestorUhdEngineKnobContract, MakeEngineCarriesTheUedsKnobsIntoTheModelCheck)
{
    // makeEngine moves the UED; the knobs must be read before the move or the exposed set is
    // empty.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_make_engine_knobs");
    const auto fixture = writeFixture(dir.path(),
                                      preferLargeTiles(),
                                      "max",
                                      {},
                                      /*calibrated=*/true,
                                      "identity",
                                      ENGINE_SIGNATURE);

    EXPECT_EQ(provenanceOfEngineRanking(engineSetRankingOnTileM(dir.path(), fixture, {"tile_m"})),
              "model");
}

TEST(TestIngestorUhdEngineKnobContract, AnEngineRankingOnAnAxisItDoesNotExposeIsRefused)
{
    // Control: the UED omits the knob the model ranks on (§6.3).
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_make_engine_knob_mismatch");
    const auto fixture = writeFixture(dir.path(),
                                      preferLargeTiles(),
                                      "max",
                                      {},
                                      /*calibrated=*/true,
                                      "identity",
                                      ENGINE_SIGNATURE);

    EXPECT_EQ(provenanceOfEngineRanking(engineSetRankingOnTileM(dir.path(), fixture, {"split_k"})),
              "declared_order");
}

TEST(TestIngestorUhdEngineKnobContract, AnEngineExposingAKnobItsModelIgnoresStillRanks)
{
    // Training drops a constant knob that the UED still exposes.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_make_engine_unread_knob");
    const auto fixture = writeFixture(dir.path(),
                                      preferLargeTiles(),
                                      "max",
                                      {},
                                      /*calibrated=*/true,
                                      "identity",
                                      ENGINE_SIGNATURE);
    // `split_k` is a legal knob the model does not rank on.
    EXPECT_EQ(provenanceOfEngineRanking(
                  engineSetRankingOnTileM(dir.path(), fixture, {"tile_m", "split_k"})),
              "model");
}

TEST(TestIngestorUhdKernelHeuristic, InlineFeaturesReachTheTreeScorer)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_inline_tree");
    auto tree = preferSmallTiles();
    tree.thresholds[0] = 48.0;
    const std::vector<nlohmann::json> signature
        = {nlohmann::json::parse(R"({"ceil_div":["$attention.seqlen","$kernel.tile_m"]})")};
    const auto fixture = writeFixture(dir.path(), tree, "max", {}, false, "identity", signature);
    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS, FIELDS);
    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};
    const auto ranked = heuristic->rankScored(catalogAgainstPriority(4096), context);
    ASSERT_EQ(ranked.size(), 2u);
    EXPECT_EQ(ranked.front().kernelId, testId(0x02));
    EXPECT_GT(ranked.front().score, ranked.back().score);
}

TEST(TestIngestorUhdKernelHeuristic, ASingleCandidateCarriesItsModelScore)
{
    // RFC 0019.13 §15.2: each candidate carries its score, even when the catalog has only one.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_single_candidate");
    const auto fixture = writeFixture(dir.path(), preferLargeTiles());
    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};

    auto single = catalogAgainstPriority(2048);
    single.entries.resize(1); // the small tile, which preferLargeTiles scores 1.0 rather than 0
    const auto sole = heuristic->rankScored(single, context);
    ASSERT_EQ(sole.size(), 1u);
    EXPECT_EQ(sole.front().kernelId, testId(0x01));
    EXPECT_GT(sole.front().score, 0.0) << "the sole candidate reported the no-measurement zero";

    // Same score as the full catalog gives that kernel.
    const auto both = heuristic->rankScored(catalogAgainstPriority(2048), context);
    ASSERT_EQ(both.size(), 2u);
    EXPECT_EQ(both.back().kernelId, testId(0x01));
    EXPECT_DOUBLE_EQ(sole.front().score, both.back().score);

    // An empty catalog has nothing to score.
    auto empty = catalogAgainstPriority(2048);
    empty.entries.clear();
    EXPECT_TRUE(heuristic->rankScored(empty, context).empty());
}

TEST(TestIngestorUhdKernelHeuristic, AnUnavailableExactArchitectureDoesNotUseDefault)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_blocked_exact");
    const auto fixture = writeFixture(dir.path(), preferLargeTiles());
    const auto fallback = modelDescriptor(dir.path(), fixture);
    const auto heuristic = makeKernelHeuristic(fallback,
                                               {},
                                               KNOBS,
                                               FIELDS,
                                               RankersByMetric{{"tflops", {{"default", fallback}}}},
                                               {{"tflops", {"gfx942"}}});
    const testing::TestGraph graph;
    const auto exact = gfx942();
    auto other = gfx942();
    other.gcnArchName = "gfx950";
    EXPECT_EQ(heuristic->rankScored(catalogAgainstPriority(2048), MatchContext{graph, 0, exact})
                  .front()
                  .kernelId,
              testId(0x01));
    EXPECT_EQ(heuristic->rankScored(catalogAgainstPriority(2048), MatchContext{graph, 0, other})
                  .front()
                  .kernelId,
              testId(0x02));
}

TEST(TestIngestorUhdKernelHeuristic, AFailedExactModelDoesNotUseDefault)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_failed_exact");
    const auto fixture = writeFixture(dir.path(), preferLargeTiles());
    const auto fallback = modelDescriptor(dir.path(), fixture);
    const auto heuristic = makeKernelHeuristic(
        fallback,
        {},
        KNOBS,
        FIELDS,
        RankersByMetric{
            {"tflops",
             {{"default", fallback}, {"gfx942", modelDescriptor(dir.path(), "missing.bin")}}}});
    const testing::TestGraph graph;
    const auto properties = gfx942();
    EXPECT_EQ(
        heuristic->rankScored(catalogAgainstPriority(2048), MatchContext{graph, 0, properties})
            .front()
            .kernelId,
        testId(0x01));
}

/// RFC 0019 §3.1: arch fallback stays inside the requested metric; (gfx942, time) falls back to
/// (default, time), never (gfx942, tflops), and no metric borrows another's `default`.
TEST(TestIngestorUhdKernelHeuristic, ArchFallbackStaysInsideTheRequestedMetric)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    const hipdnn_test_sdk::utilities::ScopedDirectory tflopsDir("uhd_metric_fallback_tflops");
    const hipdnn_test_sdk::utilities::ScopedDirectory timeDir("uhd_metric_fallback_time");
    // tflops prefers the large tile; time (1 ms small, 9 ms large, `min`) prefers the small.
    // Distinct ids let the calibrated ranking's model id say which answered.
    auto throughput
        = modelDescriptor(tflopsDir.path(), writeFixture(tflopsDir.path(), preferLargeTiles()));
    throughput.id = testId(0xA1);
    auto latency
        = modelDescriptor(timeDir.path(), writeFixture(timeDir.path(), preferLargeTiles(), "min"));
    latency.id = testId(0xB1);

    const RankersByMetric rankers{{"tflops", {{"gfx942", throughput}}},
                                  {"time", {{"default", latency}}}};
    const auto heuristic = makeKernelHeuristic(std::nullopt, "e", KNOBS, FIELDS, rankers);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto onGfx942 = gfx942();
    auto onGfx950 = gfx942();
    onGfx950.gcnArchName = "gfx950";
    // Declared order puts the large tile first.
    const auto catalog = catalogAlongPriority(2048);

    // (gfx942, time) falls back to (default, time).
    const MatchContext timeOnGfx942{graph, 0, onGfx942, "time"};
    const auto timeRanked = heuristic->rankScored(catalog, timeOnGfx942);
    ASSERT_EQ(timeRanked.size(), 2U);
    EXPECT_EQ(timeRanked.front().kernelId, testId(0x01))
        << "a time request on gfx942 was not ranked by the time model";
    std::string modelId;
    EXPECT_FALSE(heuristic->calibratedRanking(catalog, timeOnGfx942, modelId).empty());
    EXPECT_EQ(modelId, toString(latency.id));

    // Control: the gfx942 key belongs to tflops.
    modelId.clear();
    EXPECT_FALSE(
        heuristic->calibratedRanking(catalog, MatchContext{graph, 0, onGfx942}, modelId).empty());
    EXPECT_EQ(modelId, toString(throughput.id));

    // (gfx950, tflops) has no exact entry or tflops `default`, so declared order decides.
    const MatchContext tflopsOnGfx950{graph, 0, onGfx950};
    const auto unranked = heuristic->rankScored(catalog, tflopsOnGfx950);
    ASSERT_EQ(unranked.size(), 2U);
    EXPECT_EQ(unranked.front().kernelId, testId(0x02)) << "the time model ranked a tflops request";
    EXPECT_DOUBLE_EQ(unranked.front().score, 0.0);
    modelId.clear();
    EXPECT_TRUE(heuristic->calibratedRanking(catalog, tflopsOnGfx950, modelId).empty());
    EXPECT_TRUE(recorder.hasLogContaining("names no model for 'gfx950' in metric 'tflops'"));

    // (gfx950, time) reaches the time `default`.
    const auto timeOnGfx950
        = heuristic->rankScored(catalog, MatchContext{graph, 0, onGfx950, "time"});
    ASSERT_EQ(timeOnGfx950.size(), 2U);
    EXPECT_EQ(timeOnGfx950.front().kernelId, testId(0x01));
}

/// RFC 0019 §11.4: L2 values are physical units, best first. For `time` that is ascending
/// milliseconds, not the negated key `objective: min` sorts by internally.
TEST(TestIngestorUhdKernelHeuristic, ACalibratedTimeModelReportsAscendingMilliseconds)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_time_calibrated");
    // As a time, preferLargeTiles makes the small tile fast (1) and the large slow (9).
    const auto fixture
        = writeFixture(dir.path(), preferLargeTiles(), "min", {}, /*calibrated=*/true, "identity");
    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();

    // Declared order puts the slow kernel first.
    std::string modelId;
    const auto ranking = heuristic->calibratedRanking(
        catalogAlongPriority(2048), MatchContext{graph, 0, properties, "time"}, modelId);
    ASSERT_EQ(ranking.size(), 2U);
    EXPECT_EQ(ranking[0].kernelId, testId(0x01));
    EXPECT_DOUBLE_EQ(ranking[0].score, 1.0);
    EXPECT_EQ(ranking[1].kernelId, testId(0x02));
    EXPECT_DOUBLE_EQ(ranking[1].score, 9.0);
    EXPECT_FALSE(modelId.empty());

    // No substitution (§4.4): asked for tflops, a time model has no value to give.
    modelId.clear();
    EXPECT_TRUE(heuristic
                    ->calibratedRanking(
                        catalogAlongPriority(2048), MatchContext{graph, 0, properties}, modelId)
                    .empty());
    EXPECT_TRUE(modelId.empty());
}

/// RFC 0019 §3.1, §11.4: a metric with no ranker of its own picks kernels with the metric-less
/// UHD, else the `tflops` one, else declared order; its calibrated ranking stays empty (§4.4).
TEST(TestIngestorUhdKernelHeuristic, TheDefaultRankerDecidesForAMetricWithNoRankerOfItsOwn)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    const hipdnn_test_sdk::utilities::ScopedDirectory tflopsDir("uhd_default_ranker_tflops");
    const hipdnn_test_sdk::utilities::ScopedDirectory metriclessDir(
        "uhd_default_ranker_metricless");
    const hipdnn_test_sdk::utilities::ScopedDirectory timeDir("uhd_default_ranker_time");
    // tflops prefers the large tile, metric-less the small, each scoring 9; declared order puts
    // the large tile first with 0. Winner plus top score identify which decided.
    const auto throughput
        = modelDescriptor(tflopsDir.path(), writeFixture(tflopsDir.path(), preferLargeTiles()));
    auto metricless = modelDescriptor(
        metriclessDir.path(),
        writeFixture(metriclessDir.path(), preferSmallTiles(), "max", {}, /*calibrated=*/false));
    metricless.score.metric.clear();

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext tflopsContext{graph, 0, properties};
    const MatchContext timeContext{graph, 0, properties, "time"};
    const auto catalog = catalogAlongPriority(2048);

    // Only a tflops ranker: it decides the time request too.
    const auto tflopsOnly = makeKernelHeuristic(
        std::nullopt, "e", KNOBS, FIELDS, RankersByMetric{{"tflops", {{"default", throughput}}}});
    const auto asTflops = tflopsOnly->rankScored(catalog, tflopsContext);
    const auto asTime = tflopsOnly->rankScored(catalog, timeContext);
    ASSERT_EQ(asTime.size(), 2U);
    ASSERT_EQ(asTflops.size(), asTime.size());
    for(size_t i = 0; i < asTime.size(); ++i)
    {
        EXPECT_EQ(asTime[i].kernelId, asTflops[i].kernelId) << "orders differ at " << i;
        EXPECT_DOUBLE_EQ(asTime[i].score, asTflops[i].score);
    }
    EXPECT_DOUBLE_EQ(asTime.front().score, 9.0) << "declared order decided, not the tflops ranker";
    EXPECT_TRUE(recorder.hasLogContaining("metric=time ranker=default ranker_metric=tflops"));

    // But it cannot report a time.
    std::string modelId;
    EXPECT_TRUE(tflopsOnly->calibratedRanking(catalog, timeContext, modelId).empty())
        << "a tflops value was reported as a time";
    EXPECT_FALSE(tflopsOnly->calibratedRanking(catalog, tflopsContext, modelId).empty());

    // A metric-less ranker takes precedence over tflops for the time request.
    const auto withMetricless = makeKernelHeuristic(
        std::nullopt,
        "e",
        KNOBS,
        FIELDS,
        RankersByMetric{{"tflops", {{"default", throughput}}}, {"", {{"default", metricless}}}});
    const auto metriclessDecides = withMetricless->rankScored(catalog, timeContext);
    ASSERT_EQ(metriclessDecides.size(), 2U);
    EXPECT_EQ(metriclessDecides.front().kernelId, testId(0x01))
        << "the tflops ranker stood in ahead of the metric-less one";
    EXPECT_DOUBLE_EQ(metriclessDecides.front().score, 9.0);
    EXPECT_TRUE(recorder.hasLogContaining("metric=time ranker=default ranker_metric=(none)"));
    const auto tflopsKeepsItsOwn = withMetricless->rankScored(catalog, tflopsContext);
    ASSERT_EQ(tflopsKeepsItsOwn.size(), 2U);
    EXPECT_EQ(tflopsKeepsItsOwn.front().kernelId, testId(0x02));
    EXPECT_DOUBLE_EQ(tflopsKeepsItsOwn.front().score, 9.0);

    // Only a time model for another architecture: priority, then id, decides.
    const auto latency
        = modelDescriptor(timeDir.path(), writeFixture(timeDir.path(), preferLargeTiles(), "min"));
    const auto neither = makeKernelHeuristic(
        std::nullopt, "e", KNOBS, FIELDS, RankersByMetric{{"time", {{"gfx950", latency}}}});
    const auto declared = neither->rankScored(catalog, timeContext);
    ASSERT_EQ(declared.size(), 2U);
    EXPECT_EQ(declared.front().kernelId, testId(0x02)) << "priority did not decide";
    EXPECT_DOUBLE_EQ(declared.front().score, 0.0);
    EXPECT_TRUE(recorder.hasLogContaining("names no model for 'gfx942' in metric 'time'"));
}

// ---- Two-layer (grouped) selection ------------------------------------------------------

namespace
{
/// A grouped artifact over SIGNATURE, grouping on slot 0 (`$kernel.tile_m`). Layer 1 favours
/// group 128 under `max` and group 64 under `min`; layer 2 is a constant per group (64: 3.0,
/// 128: 7.0), so a score identifies which ensemble ran.
Fixture writeGroupedFixture(const std::filesystem::path& dir, const std::string& objective = "max")
{
    const std::string signatureHash = uhd::FeatureExtractor::computeHash(SIGNATURE);

    hipdnn_test_sdk::utilities::GbdtModelTestBuilder model;
    model.setFeaturesHash(signatureHash)
        .setNumFeatures(static_cast<int32_t>(SIGNATURE.size()))
        .setTrainingArches({"gfx942"})
        .setGroupByFeatureIndex(0);

    hipdnn_test_sdk::utilities::GbdtModelTestBuilder::TreeSpec layerOne;
    layerOne.featureIndices = {0, 0, 0};
    layerOne.thresholds = {96.0, 0.0, 0.0};
    layerOne.leftChildren = {1, -1, -1};
    layerOne.rightChildren = {2, -1, -1};
    layerOne.leafValues = {0.0, 1.0, 9.0};
    layerOne.defaultLeft = {1, 1, 1};
    model.addTree(layerOne);

    const auto constantTree = [](double value) {
        hipdnn_test_sdk::utilities::GbdtModelTestBuilder::TreeSpec spec;
        spec.featureIndices = {0};
        spec.thresholds = {0.0};
        spec.leftChildren = {-1};
        spec.rightChildren = {-1};
        spec.leafValues = {value};
        spec.defaultLeft = {1};
        return spec;
    };
    model.addGroup(64.0, {constantTree(3.0)});
    model.addGroup(128.0, {constantTree(7.0)});
    model.buildToFile((dir / "model.bin").string());

    return {"model.bin", signatureHash, objective, true, "identity", SIGNATURE};
}
} // namespace

TEST(TestIngestorUhdKernelHeuristicGrouped, EachCandidateReportsItsOwnGroup)
{
    // §15.2: the runner-up may be built if the winner fails, so it needs its own group.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_grouped_per_candidate");
    const auto fixture = writeGroupedFixture(dir.path());
    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};
    const auto scored = heuristic->rankScored(catalogAgainstPriority(2048), context);

    ASSERT_EQ(scored.size(), 2U);
    // Layer 1 prefers the large tile, so the 128 kernel wins and reports group 128.
    EXPECT_EQ(scored.front().kernelId, testId(0x02));
    EXPECT_DOUBLE_EQ(scored.front().group, 128.0);
    // The runner-up reports its own group, not the winner's.
    EXPECT_EQ(scored.back().kernelId, testId(0x01));
    EXPECT_DOUBLE_EQ(scored.back().group, 64.0);
}

TEST(TestIngestorUhdKernelHeuristicGrouped, TheReportedGroupIsTheOneThatScored)
{
    // The group must come from the row the model scored, not be re-derived from metadata.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_grouped_agrees");
    const auto fixture = writeGroupedFixture(dir.path());
    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};
    const auto scored = heuristic->rankScored(catalogAgainstPriority(2048), context);

    ASSERT_FALSE(scored.empty());
    EXPECT_DOUBLE_EQ(scored.front().score, 7.0);
    EXPECT_DOUBLE_EQ(scored.front().group, 128.0);
}

TEST(TestIngestorUhdKernelHeuristicGrouped, AMinObjectiveChoosesTheCheapestGroup)
{
    // Layer 1 of a `min` model predicts a cost, so the cheapest group must be chosen.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_grouped_min");
    const auto fixture = writeGroupedFixture(dir.path(), "min");
    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties, "time"};
    const auto scored = heuristic->rankScored(catalogAgainstPriority(2048), context);

    ASSERT_EQ(scored.size(), 2U);
    // Group 64's layer 2 predicts 3, negated for `min`: the model ranked, from the right group.
    EXPECT_EQ(scored.front().kernelId, testId(0x01));
    EXPECT_DOUBLE_EQ(scored.front().group, 64.0);
    EXPECT_DOUBLE_EQ(scored.front().score, -3.0) << "the winner was not scored by group 64";
    // Group 128 was declined, so it reports no measurement.
    EXPECT_DOUBLE_EQ(scored.back().group, 128.0);
    EXPECT_DOUBLE_EQ(scored.back().score, 0.0) << "the slower group was not excluded";
}

TEST(TestIngestorUhdKernelHeuristicGrouped, ExcludingAGroupIsNotReportedAsATrainingDefect)
{
    // An excluded group's -infinity is by design, not an out-of-range prediction; §12 reserves
    // ERROR for a broken model.
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_grouped_not_a_defect");
    const auto fixture = writeGroupedFixture(dir.path());
    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};
    const auto scored = heuristic->rankScored(catalogAgainstPriority(2048), context);

    // Proves grouping actually excluded a candidate.
    ASSERT_EQ(scored.size(), 2U);
    EXPECT_DOUBLE_EQ(scored.back().score, 0.0);
    EXPECT_DOUBLE_EQ(scored.back().group, 64.0);

    EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "cannot take"))
        << "a group layer 1 declined to pick was reported as a model predicting out of range";
}

TEST(TestIngestorUhdKernelHeuristicGrouped, TheGroupFeatureIsNamed)
{
    // The name comes from the adapter's group slot, not the descriptor's first entry.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_grouped_named");
    const auto fixture = writeGroupedFixture(dir.path());
    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    const auto feature = heuristic->groupFeature();
    ASSERT_TRUE(feature.has_value());
    EXPECT_EQ(*feature, "kernel.tile_m");
}

TEST(TestIngestorUhdKernelHeuristicGrouped, ASingleLayerModelReportsNoGroup)
{
    // NaN, because every real number is a legal group value.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_ungrouped_no_group");
    const auto fixture = writeFixture(dir.path(), preferLargeTiles());
    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};
    const auto scored = heuristic->rankScored(catalogAgainstPriority(2048), context);

    ASSERT_EQ(scored.size(), 2U);
    EXPECT_FALSE(heuristic->groupFeature().has_value());
    for(const auto& candidate : scored)
    {
        EXPECT_TRUE(std::isnan(candidate.group));
    }
}

TEST(TestIngestorUhdKernelHeuristicGrouped, ADegradedRankingReportsNoGroup)
{
    // The fallback decided no group, so none may be reported.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_grouped_degraded");
    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), "not_written.bin"), {}, KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};
    const auto scored = heuristic->rankScored(catalogAgainstPriority(2048), context);

    ASSERT_FALSE(scored.empty());
    for(const auto& candidate : scored)
    {
        EXPECT_TRUE(std::isnan(candidate.group));
    }
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
