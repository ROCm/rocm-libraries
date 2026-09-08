// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestUhdKernelHeuristic.cpp
 * @brief Covers the seam that lets a trained UHD rank kernels at plan build.
 *
 * The point of interest is not that a model produces a number -- TestTreeDataAdapter
 * already covers that -- but that the number reaches the ranking, that it is built from
 * the problem as well as the kernel, and that every way the model can fail leaves the
 * catalog in declared order rather than a partial one.
 */

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <optional>

#include "../KernelIngestorTestFixtures.hpp"

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelHeuristicFactory.hpp>
#include <hipdnn_plugin_sdk/ingestor/MakeEngine.hpp>
#include <hipdnn_plugin_sdk/ingestor/UhdKernelHeuristic.hpp>
#include <hipdnn_plugin_sdk/ingestor/uhd/FeatureExtractor.hpp>
#include <hipdnn_plugin_sdk/ingestor/uhd/NativeScorerRegistry.hpp>

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/GbdtModelTestBuilder.hpp>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <exception>
#include <filesystem>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace hipdnn_plugin_sdk::ingestor
{
namespace
{

// The signature both the descriptor and the tests agree on: slot 0 is a kernel knob, slot
// 1 a problem token. Two slots from two namespaces is the minimum that can show the
// interaction reaching the model.
const std::vector<std::string> SIGNATURE = {R"("$kernel.tile_m")", R"("$q.seqlen")"};

/// The same two slots in RFC 0019 §7.2's canonical spelling: a bare reference, which is what
/// tools/uhd_gen writes and what a hand-authored `.uhd.json` most naturally carries. It is
/// deliberately NOT valid JSON, so anything reaching for a raw `json::parse` instead of
/// FeatureExtractor::parseSignatureEntry drops every entry -- and the §6.3 axis set that
/// follows comes back empty rather than wrong, which no assertion on a ranking can see.
///
/// computeHash() canonicalizes both spellings, so this hashes identically to SIGNATURE and the
/// two are interchangeable everywhere except in the parse under test.
const std::vector<std::string> BARE_SIGNATURE = {"$kernel.tile_m", "$q.seqlen"};

/// One slot, read from the kernel, for the cases that rank through a whole engine. A `$q.*`
/// slot would bind nothing there -- the fixture graph matcher binds no tokens -- so the model
/// would fail closed for a reason unrelated to what those cases are about.
const std::vector<std::string> ENGINE_SIGNATURE = {R"("$kernel.tile_m")"};

/// The knobs a conformant UED would expose for SIGNATURE. RFC 0019 §6.3 check 2 requires
/// `set(UED.knobs) == set($kernel.* axes the model reads)`, so a fixture that omitted these
/// would describe a descriptor pair the loader is required to refuse.
const std::vector<std::string> KNOBS = {"tile_m"};

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

/// A tree splitting on slot 0 (`$kernel.tile_m`): tile_m <= 96 scores 1.0, above scores
/// 9.0. Larger tiles win, which lets a test set priority the other way round.
/// Every leaf negative. A GBDT raw score is unbounded, so this is an ordinary model -- it only
/// becomes interesting once a transform whose inverse is a logarithm is declared over it.
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

/// The inverse of preferLargeTiles: small tiles score higher. Paired with it, the winner
/// identifies which model ranked, so an arch-resolution test cannot pass by coincidence.
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

/// A tree splitting on slot 1 (`$q.seqlen`), so the same catalog ranks differently for
/// different problems -- which is the whole reason a UHD exists.
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
    /// The GBDT artifact `writeFixture` wrote, relative to the directory it wrote it in.
    std::string modelFileName;
    /// The hash SIGNATURE really computes to, which is what a descriptor must declare.
    std::string featuresHash;
    /// The score fields the fixture was built for. They live on the descriptor now rather
    /// than in a second file, so they have to travel back to the caller that will build it
    /// -- otherwise every call site would repeat a value it has already passed here, and a
    /// test varying the objective would hand the runtime a descriptor that does not.
    std::string objective;
    bool calibrated;
    std::string scoreTransform;
    /// The signature the artifact was built for. Carried so `modelDescriptor(dir, fixture)`
    /// describes the model that was actually written: a case varying the spelling of the
    /// signature would otherwise write one signature and declare another.
    std::vector<std::string> signature;
};

/// Writes a GBDT artifact into @p dir and reports what a descriptor over it must say.
///
/// There is no second file to write. The descriptor carries the header, so the only thing
/// on disk is the model the header names.
///
/// @param modelHash Written into the model artifact. Defaults to the signature's real
///        hash; a caller passing something else is testing the contract check, which
///        compares the model's own hash against the one the descriptor declares.
Fixture writeFixture(const std::filesystem::path& dir,
                     const hipdnn_test_sdk::utilities::GbdtModelTestBuilder::TreeSpec& tree,
                     const std::string& objective = "max",
                     const std::string& modelHash = {},
                     // RFC 0019.13 §15.1 binds these: a calibrated score is cross-engine
                     // TFLOPS and therefore ascending, so a descending objective defaults to
                     // uncalibrated. A caller passing the pair explicitly is testing the rule.
                     std::optional<bool> calibrated = std::nullopt,
                     /// The target transform the model was trained under; its inverse runs at
                     /// score time. "exp" inverts as a logarithm, which is out of domain for a
                     /// negative prediction.
                     const std::string& scoreTransform = "identity",
                     /// The feature signature to train against. Defaults to SIGNATURE; a caller
                     /// passing another is varying either the spelling or the slot count, and
                     /// the artifact's feature count follows it so the two cannot drift.
                     const std::vector<std::string>& signature = SIGNATURE)
{
    const std::string signatureHash = uhd::FeatureExtractor::computeHash(signature);

    hipdnn_test_sdk::utilities::GbdtModelTestBuilder model;
    model.setFeaturesHash(modelHash.empty() ? signatureHash : modelHash)
        .setNumFeatures(static_cast<int32_t>(signature.size()))
        .setTrainingArches({"gfx942"})
        .addTree(tree);
    model.buildToFile((dir / "model.bin").string());

    return {"model.bin",
            signatureHash,
            objective,
            calibrated.value_or(objective != "min"),
            scoreTransform,
            signature};
}

/// The descriptor the loader would produce for a tree_data UHD in @p dir.
HeuristicDescriptor modelDescriptor(const std::filesystem::path& dir,
                                    const std::string& artifact,
                                    const std::string& objective = "max",
                                    bool calibrated = true,
                                    const std::string& scoreTransform = "identity",
                                    const std::string& featuresHash = {},
                                    const std::vector<std::string>& signature = SIGNATURE)
{
    HeuristicDescriptor descriptor;
    descriptor.id = testId(0xEE);
    descriptor.name = "test model heuristic";
    descriptor.adapter = UhdAdapter::TREE_DATA;
    descriptor.featuresSignature = signature;
    descriptor.featuresHash
        = featuresHash.empty() ? uhd::FeatureExtractor::computeHash(signature) : featuresHash;
    descriptor.objective = objective;
    descriptor.score.units = "tflops";
    descriptor.score.calibrated = calibrated;
    descriptor.score.transform = scoreTransform;
    descriptor.modelArtifactPath = artifact;
    descriptor.baseDir = dir;
    return descriptor;
}

/// The descriptor over the artifact @p fixture wrote, carrying the score fields it was
/// built for. Every test that varies one of them varies it through this, so the variation
/// cannot be lost between writing the model and describing it.
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

/// A `.uhd.json` as an author would write it, naming @p artifact.
///
/// Only the tests that need a document the *parser* refuses build one of these. A
/// descriptor assembled in memory has already skipped every check the parse performs, so a
/// rule enforced there -- §15.1's calibrated/`min` pair is the one -- is only reachable
/// through the JSON.
nlohmann::json uhdDocument(const std::string& artifact,
                           const std::string& objective,
                           bool calibrated)
{
    nlohmann::json document;
    document["version"] = "1.0";
    document["id"] = "ee000000-0000-0000-0000-000000000000";
    document["name"] = "test model heuristic";
    document["adapter"] = "tree_data";
    document["features_signature"] = SIGNATURE;
    document["features_hash"] = uhd::FeatureExtractor::computeHash(SIGNATURE);
    document["objective"] = objective;
    document["score"]
        = {{"units", "tflops"}, {"calibrated", calibrated}, {"transform", "identity"}};
    document["tree_data"] = {{"artifact", artifact}};
    return document;
}

/// Parses @p document exactly as descriptor discovery would.
///
/// @returns nullopt when the loader refuses it, which is the state the factory then sees:
///          an engine whose UHD did not parse arrives with no descriptor at all.
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

/// The suite's shared device, with the architecture the fixtures train against so the
/// out-of-distribution warning stays out of these cases.
DeviceProperties gfx942()
{
    auto properties = testing::testDeviceProperties();
    properties.gcnArchName = "gfx942";
    properties.multiProcessorCount = 304;
    return properties;
}

/// Small tile at high priority, large tile at low priority. Declared order therefore puts
/// the small tile first, and every model here prefers the large one -- so a test that sees
/// the large tile first is seeing the model, not the fallback.
Catalog catalogAgainstPriority(int64_t seqlen)
{
    Catalog catalog;
    catalog.entries = {kernelWith(0x01, 64, 100), kernelWith(0x02, 128, 1)};
    catalog.bound["seqlen"] = seqlen;
    return catalog;
}

} // namespace

TEST(TestIngestorUhdKernelHeuristic, RanksByTheModelRatherThanByPriority)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_kernel_heuristic_happy");
    const auto fixture = writeFixture(dir.path(), preferLargeTiles());

    const auto heuristic = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS);
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
    // The bridge exists so a model can see the problem. The policy path binds no query
    // variables at all, so this is the case that distinguishes the two.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_kernel_heuristic_problem");
    const auto fixture = writeFixture(dir.path(), preferLargeTilesOnLongSequences());

    const auto heuristic = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS);
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

TEST(TestIngestorUhdKernelHeuristic, AMinimisingObjectiveReversesTheOrder)
{
    // Same model, same catalog; only `objective` differs. A UHD trained on a cost rather
    // than a rate has to rank ascending, and getting this wrong inverts it silently.
    const hipdnn_test_sdk::utilities::ScopedDirectory maxDir("uhd_kernel_heuristic_max");
    const hipdnn_test_sdk::utilities::ScopedDirectory minDir("uhd_kernel_heuristic_min");
    const auto maxFixture = writeFixture(maxDir.path(), preferLargeTiles(), "max");
    const auto minFixture = writeFixture(minDir.path(), preferLargeTiles(), "min");

    const auto maximising
        = makeKernelHeuristic(modelDescriptor(maxDir.path(), maxFixture), {}, KNOBS);
    const auto minimising
        = makeKernelHeuristic(modelDescriptor(minDir.path(), minFixture), {}, KNOBS);
    ASSERT_NE(maximising, nullptr);
    ASSERT_NE(minimising, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};

    // Priorities are chosen so neither answer coincides with declared order. With
    // catalogAgainstPriority the minimising case would pass on a silent degradation,
    // because declared order already puts the small tile first.
    Catalog catalog;
    catalog.entries = {kernelWith(0x01, 64, 1), kernelWith(0x02, 128, 100)};
    catalog.bound["seqlen"] = int64_t{2048};

    const auto maxRanked = maximising->rank(catalog, context);
    const auto minRanked = minimising->rank(catalog, context);

    ASSERT_EQ(maxRanked.size(), 2U);
    ASSERT_EQ(minRanked.size(), 2U);
    EXPECT_EQ(maxRanked.front().kernelId, testId(0x02)); // larger tile scores higher
    EXPECT_EQ(minRanked.front().kernelId, testId(0x01)); // ... so it loses when minimising
}

TEST(TestIngestorUhdKernelHeuristic, AnAbsentArtifactDegradesToDeclaredOrder)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_kernel_heuristic_absent");

    // KNOBS, so the degradation under test is the missing artifact and not the §6.3 knob
    // check -- with no knobs declared the heuristic would refuse before it ever looked for
    // the file, and this case would pass without exercising what it names.
    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), "not_written.bin"), {}, KNOBS);
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
    // The descriptor's signature and the model's training signature must agree; if they do
    // not, the pair came from two different runs and the model is being fed a row it was
    // not trained on (RFC 0019 §6.3).
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_kernel_heuristic_hash");
    const auto fixture
        = writeFixture(dir.path(), preferLargeTiles(), "max", "sha256:not_the_real_hash");

    const auto heuristic = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS);
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
    // One kernel omits `tile_m`, so its row cannot be built. The whole ranking falls back,
    // rather than scoring the kernels that worked and leaving the rest at a sentinel --
    // that mixed order would be neither the model's nor the fallback's. This is why rank()
    // is overridden instead of score().
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_kernel_heuristic_partial");
    const auto fixture = writeFixture(dir.path(), preferLargeTiles());

    const auto heuristic = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS);
    ASSERT_NE(heuristic, nullptr);

    Catalog catalog;
    KernelDefinition incomplete;
    incomplete.kernelId = testId(0x03);
    incomplete.priority = 50; // between the two well-formed kernels
    catalog.entries = {kernelWith(0x01, 64, 100), incomplete, kernelWith(0x02, 128, 1)};
    catalog.bound["seqlen"] = int64_t{2048};

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
    // MetadataValue admits vector<int64_t>, which the feature extractor's value type does
    // not. The binding is skipped, so a signature that does not reference it still ranks.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_kernel_heuristic_list");
    const auto fixture = writeFixture(dir.path(), preferLargeTiles());

    const auto heuristic = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS);
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

/// RFC 0019 §6.3 check 2: `set(UED.knobs) == set($kernel.* axes the model reads)`. Both
/// directions of a mismatch fail silently, which is why the RFC asks for set equality rather
/// than a subset test, and why both are pinned here.
///
/// The expected outcome is a degraded ranking, not a failure: §5 step 7 requires a broken
/// feature contract to leave the engine selecting, by declared order.
TEST(TestIngestorUhdKernelHeuristic, AKnobTheModelDoesNotReadIsRefused)
{
    // The caller can turn split_k and the heuristic will not react. Nothing in the output
    // distinguishes that from a knob the model happens to weigh lightly.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_kernel_heuristic_extra_knob");
    const auto fixture = writeFixture(dir.path(), preferLargeTiles());
    const auto heuristic = makeKernelHeuristic(
        modelDescriptor(dir.path(), fixture), {}, {"tile_m", "split_k"});
    ASSERT_NE(heuristic, nullptr);
    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};
    const auto ranked = heuristic->rank(catalogAgainstPriority(2048), context);
    ASSERT_EQ(ranked.size(), 2U);
    // Declared order: the high-priority kernel leads, which is exactly what the model would
    // have overturned had it been used.
    EXPECT_EQ(ranked.front().kernelId, testId(0x01));
}

TEST(TestIngestorUhdKernelHeuristic, AnAxisWithNoKnobIsRefused)
{
    // The reverse direction: the model ranks on tile_m while the engine exposes nothing, so
    // its scores turn on something no caller can influence.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_kernel_heuristic_no_knob");
    const auto fixture = writeFixture(dir.path(), preferLargeTiles());
    const auto heuristic = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, {});
    ASSERT_NE(heuristic, nullptr);
    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};
    const auto ranked = heuristic->rank(catalogAgainstPriority(2048), context);
    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, testId(0x01));
}

/// RFC 0019 §3.1 lets a UED name a UHD per architecture; §8.3 resolves exact gcnArchName then
/// `default`. It cannot happen at load -- descriptor discovery runs before any device exists --
/// so it happens at first rank(), which is also §9.2's load-on-demand.
TEST(TestIngestorUhdKernelHeuristic, AnArchSpecificModelOutranksTheDefaultOne)
{
    // Two models that disagree: the default prefers small tiles, the gfx942 one prefers large.
    // Whichever ranks first identifies which model was used, so the assertion cannot pass by
    // coincidence.
    const hipdnn_test_sdk::utilities::ScopedDirectory defaultDir("uhd_arch_specific_default");
    const hipdnn_test_sdk::utilities::ScopedDirectory archDir("uhd_arch_specific_gfx942");
    const auto fallback = writeFixture(defaultDir.path(), preferSmallTiles());
    const auto specific = writeFixture(archDir.path(), preferLargeTiles());

    const std::map<std::string, HeuristicDescriptor> byArch{
        {"default", modelDescriptor(defaultDir.path(), fallback)},
        {"gfx942", modelDescriptor(archDir.path(), specific)}};

    const auto heuristic = makeKernelHeuristic(
        modelDescriptor(defaultDir.path(), fallback), {}, KNOBS, byArch);
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
    // §8.3: exact match, then `default`, and a device the UED says nothing about takes the
    // second. The fallback prefers small tiles, so its winner differs from the gfx942 model's.
    const hipdnn_test_sdk::utilities::ScopedDirectory defaultDir("uhd_arch_fallback_default");
    const hipdnn_test_sdk::utilities::ScopedDirectory archDir("uhd_arch_fallback_gfx942");
    const auto fallback = writeFixture(defaultDir.path(), preferSmallTiles());
    const auto specific = writeFixture(archDir.path(), preferLargeTiles());

    const std::map<std::string, HeuristicDescriptor> byArch{
        {"default", modelDescriptor(defaultDir.path(), fallback)},
        {"gfx942", modelDescriptor(archDir.path(), specific)}};

    const auto heuristic = makeKernelHeuristic(
        modelDescriptor(defaultDir.path(), fallback), {}, KNOBS, byArch);
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
    // §9.2 caches what it loads. A cache that returned a different model on the second call --
    // or that served one arch's model to another -- would be worse than no cache at all.
    const hipdnn_test_sdk::utilities::ScopedDirectory defaultDir("uhd_arch_cached_default");
    const hipdnn_test_sdk::utilities::ScopedDirectory archDir("uhd_arch_cached_gfx942");
    const auto fallback = writeFixture(defaultDir.path(), preferSmallTiles());
    const auto specific = writeFixture(archDir.path(), preferLargeTiles());

    const std::map<std::string, HeuristicDescriptor> byArch{
        {"default", modelDescriptor(defaultDir.path(), fallback)},
        {"gfx942", modelDescriptor(archDir.path(), specific)}};

    const auto heuristic = makeKernelHeuristic(
        modelDescriptor(defaultDir.path(), fallback), {}, KNOBS, byArch);
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

/// RFC 0019.13 §15.2: "an ordered sequence of `(UKD id, score)`, winner first". The score
/// travels with the id because §15.2's named callers need it -- a knob query reports the
/// top-ranked value as its default, autotune walks the list, and engine selection reads the top
/// score as the engine's figure of merit. Returning order alone makes the third impossible.
TEST(TestIngestorUhdKernelHeuristic, SelectionReturnsIdsWithScoresWinnerFirst)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_scored_form");
    const auto fixture = writeFixture(dir.path(), preferLargeTiles());

    const auto heuristic = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};
    const auto scored = heuristic->rankScored(catalogAgainstPriority(2048), context);

    ASSERT_EQ(scored.size(), 2U);
    EXPECT_EQ(scored.front().kernelId, testId(0x02)) << "winner is not first";
    // Ordered by score, and the scores are real rather than placeholders -- a caller reading
    // the top one as a figure of merit has to get the model's number.
    EXPECT_GT(scored.front().score, scored.back().score);
    EXPECT_TRUE(std::isfinite(scored.front().score));

    // The whole-kernel view reports the same order, since one implementation decides it.
    const auto ordered = heuristic->rank(catalogAgainstPriority(2048), context);
    ASSERT_EQ(ordered.size(), scored.size());
    for(size_t i = 0; i < ordered.size(); ++i)
    {
        EXPECT_EQ(ordered[i].kernelId, scored[i].kernelId) << "views disagree at " << i;
    }
}

TEST(TestIngestorUhdKernelHeuristic, ADegradedRankingReportsTheZeroTheRfcPrescribes)
{
    // Declared order carries no model score, so it reports 0 -- RFC 0019 §5 step 7: "the engine
    // reports an estimated throughput of 0 so any engine with a real estimate outranks it...
    // and loses on merit rather than by exception."
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_scored_degraded");
    const auto fixture = writeFixture(dir.path(), preferLargeTiles());

    // A knob set the model does not read breaks the §6.3 contract, so ranking degrades.
    const auto heuristic = makeKernelHeuristic(
        modelDescriptor(dir.path(), fixture), {}, {"tile_m", "split_k"});
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};
    const auto scored = heuristic->rankScored(catalogAgainstPriority(2048), context);

    ASSERT_EQ(scored.size(), 2U);
    EXPECT_DOUBLE_EQ(scored.front().score, 0.0)
        << "a fallback ordering invented a score it did not compute";
}

TEST(TestIngestorUhdKernelHeuristic, ACalibratedScoreCannotAlsoBeMinimising)
{
    // RFC 0019.13 §15.1: §11.3 requires a cross-engine score to be an absolute metric on a
    // scale that means the same thing everywhere -- TFLOPS, which ascends. Accepting this pair
    // would have engine selection compare one engine's throughput against another's latency
    // and rank the faster engine last, with nothing in the output to show it happened. So the
    // load fails rather than picking a direction.
    //
    // The descriptor IS the UHD, so the refusal happens where the JSON is parsed rather
    // than where a second file was opened; a descriptor handed to the factory has already
    // been through it. The document is therefore what this asserts on.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_calibrated_min");
    const auto fixture
        = writeFixture(dir.path(), preferLargeTiles(), "min", {}, /*calibrated=*/true);

    EXPECT_FALSE(parseUhd(uhdDocument(fixture.modelFileName, "min", /*calibrated=*/true),
                          dir.path() / "test.uhd.json")
                     .has_value())
        << "a calibrated, minimising UHD loaded";

    // The supported pairing -- rank on a cost target, decline cross-engine comparison -- still
    // loads, so the check rejects the contradiction rather than the objective.
    const hipdnn_test_sdk::utilities::ScopedDirectory okDir("uhd_uncalibrated_min");
    const auto ok
        = writeFixture(okDir.path(), preferLargeTiles(), "min", {}, /*calibrated=*/false);
    EXPECT_TRUE(parseUhd(uhdDocument(ok.modelFileName, "min", /*calibrated=*/false),
                         okDir.path() / "test.uhd.json")
                    .has_value());
}

TEST(TestIngestorUhdKernelHeuristic, ACalibratedModelReportsItsTopScoreAsTheEngineEstimate)
{
    // RFC 0019 §11.1's stopgap for `predict_engine_tflops`: with no distinct estimate model,
    // the engine reports sort_kernel_catalog's best predicted score. The estimate must be the
    // *same* number the ranking put first -- an estimate derived from a second traversal could
    // disagree with the kernel actually selected, and the engine would be ranked on a plan it
    // is not going to run.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_engine_estimate");
    const auto fixture
        = writeFixture(dir.path(), preferLargeTiles(), "max", {}, /*calibrated=*/true);

    const auto heuristic = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};

    const auto estimate = heuristic->estimateTflops(catalogAgainstPriority(2048), context);
    EXPECT_DOUBLE_EQ(estimate,
                     heuristic->rankScored(catalogAgainstPriority(2048), context).front().score);
    EXPECT_GT(estimate, 0.0) << "a real estimate must outrank the 0 a declining engine reports";
}

TEST(TestIngestorUhdKernelHeuristic, AnUncalibratedModelEstimatesZero)
{
    // §11.3: a cross-engine score has to be an absolute metric on a shared scale. An
    // uncalibrated model ranks within its own engine and says nothing about how it compares to
    // another. RFC 0019 §5 step 7 fixes what it reports instead: "an estimated throughput of 0
    // so any engine with a real estimate outranks it... loses on merit rather than by
    // exception." It still ranks -- declining to estimate is not declining to select.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_no_estimate");
    const auto fixture
        = writeFixture(dir.path(), preferLargeTiles(), "max", {}, /*calibrated=*/false);

    const auto heuristic = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};

    EXPECT_DOUBLE_EQ(heuristic->estimateTflops(catalogAgainstPriority(2048), context), 0.0);
    EXPECT_FALSE(heuristic->rankScored(catalogAgainstPriority(2048), context).empty())
        << "reporting a zero estimate must not stop it selecting";
}

TEST(TestIngestorKernelHeuristicEstimate, AnEngineWithNoModelEstimatesZero)
{
    // The fallback ranks on declared order and computes no figure of merit, so it reports the 0
    // §5 step 7 prescribes. Reporting the priority it sorted by would put an arbitrary integer
    // on a TFLOPS scale, where a large priority would outrank a real throughput.
    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};

    const UnrankedKernelHeuristic heuristic;
    EXPECT_DOUBLE_EQ(heuristic.estimateTflops(catalogAgainstPriority(2048), context), 0.0);
}

TEST(TestIngestorUhdKernelHeuristic, AModelWhoseTransformGoesOutOfDomainDoesNotCorruptTheSort)
{
    // The regression this pair of defects needed. `exp` inverts as log(raw), and a GBDT raw
    // score is unbounded, so a negative prediction makes applyInverse return NaN on a wholly
    // legal descriptor -- no third party, no malformed artifact.
    //
    // NaN in a sort comparator is undefined behaviour rather than a wrong answer: it compares
    // false both ways, so it reads as "equivalent" to every element while real scores stay
    // ordered among themselves, which violates the strict weak ordering std::stable_sort
    // requires. The base class had always sanitized for this reason; overriding rankScored
    // bypassed that, and this case is what would have caught it.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_out_of_domain");
    const auto fixture = writeFixture(dir.path(), preferNegativeScores(), "max", {}, false, "exp");

    const auto heuristic = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};

    const auto scored = heuristic->rankScored(catalogAgainstPriority(2048), context);
    ASSERT_EQ(scored.size(), 2U);

    // Every reported score is a real number, and specifically the 0 that means "no measurement"
    // -- not a NaN leaking out to a caller that would compare it.
    for(const auto& entry : scored)
    {
        EXPECT_TRUE(std::isfinite(entry.score)) << "a non-finite score reached the caller";
        EXPECT_DOUBLE_EQ(entry.score, 0.0);
    }

    // And the engine estimate agrees, which is the point of using one sentinel at both layers.
    EXPECT_DOUBLE_EQ(heuristic->estimateTflops(catalogAgainstPriority(2048), context), 0.0);
}

TEST(TestIngestorUhdKernelHeuristic, ACalibratedModelCannotReportANegativeThroughput)
{
    // The case a finite-only guard misses, and the one most likely to occur: log1p is what
    // uhd_gen emits by default, and expm1 of a negative prediction is finite and negative. A
    // calibrated score is TFLOPS (§11.3) and a throughput cannot be negative, so that value is
    // the model predicting outside its target's range -- a training defect, not a slow kernel.
    //
    // It matters which way it fails. A negative score is *below* the 0 that RFC 0019 §5 step 7
    // gives "no measurement", so an engine emitting nonsense would rank beneath an engine that
    // honestly declined to estimate. Bounding it to 0 puts the two on the footing the RFC
    // describes: it loses on merit, not beneath merit.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_negative_tflops");
    const auto fixture
        = writeFixture(dir.path(), preferNegativeScores(), "max", {}, /*calibrated=*/true, "log1p");

    const auto heuristic = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS);
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

    EXPECT_DOUBLE_EQ(heuristic->estimateTflops(catalogAgainstPriority(2048), context), 0.0)
        << "the engine estimate went below the RFC's zero";
}

TEST(TestIngestorUhdKernelHeuristic, AMinObjectiveScoresBelowZeroWithoutThatBeingAnError)
{
    // The distinction the range check must not flatten. `objective: min` negates a cost, so
    // every *oriented* score is below zero while the underlying cost is perfectly ordinary.
    // A negative recovered value is meaningless; a negative oriented one is the normal case
    // for a cost target, and refusing it would refuse every candidate a min model ever ranks.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_min_negative_oriented");
    const auto fixture = writeFixture(
        dir.path(), preferLargeTiles(), "min", {}, /*calibrated=*/false, "identity");

    const auto heuristic = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};

    const auto scored = heuristic->rankScored(catalogAgainstPriority(2048), context);
    ASSERT_EQ(scored.size(), 2U);
    EXPECT_LT(scored.front().score, 0.0) << "a min objective's oriented scores were clamped";
    EXPECT_GT(scored.front().score, scored.back().score) << "the cheaper candidate did not win";
}

TEST(TestIngestorUhdKernelHeuristic, AnUnmeasuredCandidateSortsLastUnderAMinObjectiveToo)
{
    // Why the ordering key and the reported score have to be separate values. Reporting 0 for
    // "no measurement" is right -- RFC 0019 §5 step 7 -- but 0 is *greater* than every oriented
    // score a min objective produces, so using it to sort as well made an unmeasured candidate
    // outrank every measured one. It has to sort last whichever way the objective points.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_min_unmeasured_last");
    const auto fixture = writeFixture(
        dir.path(), oneUsableOneOutOfRange(), "min", {}, /*calibrated=*/false, "identity");

    const auto heuristic = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};

    const auto scored = heuristic->rankScored(catalogAgainstPriority(2048), context);
    ASSERT_EQ(scored.size(), 2U);

    // The measured candidate wins even though its reported score (-4.0, a negated cost) is
    // numerically below the 0 the unmeasured one reports.
    EXPECT_LT(scored.front().score, 0.0) << "the measured candidate did not come first";
    EXPECT_DOUBLE_EQ(scored.back().score, 0.0) << "the unmeasured candidate is not reporting 0";
}

TEST(TestIngestorUhdKernelHeuristic, ANegativeThroughputIsReportedAsAnErrorNotSwallowed)
{
    // A model predicting a value its target cannot take is broken, and the runtime's only
    // recourse is to discard the score. Discarding it silently would leave an engine that
    // looks like it ranks on a model while ranking on declared order -- which is precisely the
    // failure mode RFC 0019 §12's observability exists to make visible.
    //
    // ERROR rather than WARN: the sibling WARN on this path is "not trained for this arch",
    // where §9.3 says the model is still worth using. Here the number is wrong, not uncertain.
    auto recorder = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(
        HIPDNN_SEV_INFO);

    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_negative_reported");
    const auto fixture
        = writeFixture(dir.path(), preferNegativeScores(), "max", {}, /*calibrated=*/true, "log1p");

    const auto heuristic = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};
    (void)heuristic->rankScored(catalogAgainstPriority(2048), context);

    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "cannot take"))
        << "a model predicting a negative throughput was discarded without a word";

    // Every candidate was affected here, and the message has to say so: an engine whose model
    // contributed nothing to a ranking is a different report from one that lost a candidate.
    EXPECT_TRUE(recorder.hasLogContaining("Every candidate was affected"));

    // Once per heuristic. The condition is a property of the model, so it recurs on every
    // graph; repeating it per ranking would bury it.
    const auto after = recorder.countLogsAtLevel(HIPDNN_SEV_ERROR);
    (void)heuristic->rankScored(catalogAgainstPriority(2048), context);
    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_ERROR), after) << "the report repeated";
}

TEST(TestIngestorUhdKernelHeuristic, APartiallyAffectedRankingSaysTheModelStillDecidedTheRest)
{
    // The other half of the count, and why the count is worth carrying. One candidate
    // extrapolating badly leaves a ranking that is still mostly the model's; reporting it the
    // same way as a total failure would send someone hunting for the wrong problem.
    auto recorder = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(
        HIPDNN_SEV_INFO);

    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_partial_out_of_range");
    const auto fixture = writeFixture(
        dir.path(), oneUsableOneOutOfRange(), "max", {}, /*calibrated=*/true, "identity");

    const auto heuristic = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), {}, KNOBS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    const auto properties = gfx942();
    const MatchContext context{graph, 0, properties};
    const auto scored = heuristic->rankScored(catalogAgainstPriority(2048), context);

    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "1 of 2 candidates"));
    EXPECT_TRUE(recorder.hasLogContaining("The remaining candidates ranked on the model."));

    // And the usable candidate still won on its score rather than being dragged down with it.
    ASSERT_EQ(scored.size(), 2U);
    EXPECT_GT(scored.front().score, 0.0);
    EXPECT_DOUBLE_EQ(scored.back().score, 0.0);
}

TEST(TestIngestorUhdKernelHeuristic, PerArchModelsRankWithoutADefaultEntry)
{
    // RFC 0019 §8.3's first step is the exact gcnArchName, so a UED naming models per
    // architecture and no `default` must still rank by model on the architectures it names.
    // It did not: with no `default` the loader left heuristicId unset, makeKernelHeuristic
    // returned the unranked fallback before looking at the map, and the engine ranked by
    // declared order everywhere -- including on the very architectures it shipped a model for.
    //
    // Every pre-existing arch test passed a "default" key, which is why nothing caught this.
    const hipdnn_test_sdk::utilities::ScopedDirectory gfx942Dir("uhd_no_default_942");
    const hipdnn_test_sdk::utilities::ScopedDirectory gfx950Dir("uhd_no_default_950");
    const auto onNine42 = writeFixture(gfx942Dir.path(), preferLargeTiles());
    const auto onNine50 = writeFixture(gfx950Dir.path(), preferSmallTiles());

    const std::map<std::string, HeuristicDescriptor> byArch{
        {"gfx942", modelDescriptor(gfx942Dir.path(), onNine42)},
        {"gfx950", modelDescriptor(gfx950Dir.path(), onNine50)}};

    // No descriptor: there is no `default` for the loader to have resolved.
    const auto heuristic = makeKernelHeuristic(std::nullopt, "test-engine", KNOBS, byArch);
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
    // §8.3's third step, which had no implementation: exact, then `default`, then unavailable.
    // A single arch-named entry used to be promoted to the universal model on the reasoning
    // that one entry means one model. But `{"gfx950": X}` says X describes gfx950 -- not that
    // it describes everything -- so a gfx950-only UHD ranked every device, silently, on a model
    // trained for different hardware.
    auto recorder = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(
        HIPDNN_SEV_INFO);

    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_wrong_arch_only");
    const auto onNine50 = writeFixture(dir.path(), preferLargeTiles());

    const std::map<std::string, HeuristicDescriptor> byArch{
        {"gfx950", modelDescriptor(dir.path(), onNine50)}};

    const auto heuristic = makeKernelHeuristic(std::nullopt, "test-engine", KNOBS, byArch);
    ASSERT_NE(heuristic, nullptr);

    // A device the UED says nothing about.
    const testing::TestGraph graph;
    auto properties = gfx942();
    properties.gcnArchName = "gfx1100";
    const MatchContext context{graph, 0, properties};
    const auto ranked = heuristic->rank(catalogAgainstPriority(2048), context);

    // Declared order -- catalogAgainstPriority puts the small tile first -- not the gfx950
    // model, which prefers large tiles and would have put 0x02 first.
    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, testId(0x01))
        << "a model named only for gfx950 ranked a gfx1100 device";

    // And it says so. Without this the only symptom is a UHD-carrying engine quietly not
    // using its UHD on some machines and not others.
    EXPECT_TRUE(recorder.hasLogContaining("names no model for 'gfx1100'"));
}

TEST(TestIngestorUhdKernelHeuristic, ADefaultStillCoversAnArchitectureNotNamedExplicitly)
{
    // The compatibility half of the same rule. Tightening the third step must not break the
    // second: a UED that does declare a `default` still ranks every unnamed architecture with
    // it, which is what every model shipped so far relies on.
    const hipdnn_test_sdk::utilities::ScopedDirectory defaultDir("uhd_default_covers");
    const hipdnn_test_sdk::utilities::ScopedDirectory archDir("uhd_default_covers_950");
    const auto fallback = writeFixture(defaultDir.path(), preferLargeTiles());
    const auto specific = writeFixture(archDir.path(), preferSmallTiles());

    const std::map<std::string, HeuristicDescriptor> byArch{
        {"default", modelDescriptor(defaultDir.path(), fallback)},
        {"gfx950", modelDescriptor(archDir.path(), specific)}};

    const auto heuristic = makeKernelHeuristic(
        modelDescriptor(defaultDir.path(), fallback), "test-engine", KNOBS, byArch);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    auto properties = gfx942();
    properties.gcnArchName = "gfx1100";
    const MatchContext context{graph, 0, properties};
    const auto ranked = heuristic->rank(catalogAgainstPriority(2048), context);

    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, testId(0x02)) << "the default model did not cover gfx1100";
}

/// RFC 0019 §5 defines what selection does when things go wrong, and §12 requires the trace to
/// say *whether the model or a fallback decided*. Every degraded path is a legal ranking, so no
/// assertion on the chosen kernel can tell them apart -- the implementation could be entirely
/// broken and every functional test would still pass.
///
/// These are mock UHDs, one per condition, and they assert the provenance rather than the
/// winner. That distinction is the point: a test that writes the model can make any kernel win,
/// so "the winner is X" only proves the fixture works. "A model decided" is a claim the fixture
/// cannot fake, because it is the runtime that reports it.
struct SelectionCondition
{
    const char* name;
    const char* expectedProvenance;
};

class TestIngestorUhdProvenance : public ::testing::TestWithParam<SelectionCondition>
{
};

/// Builds the heuristic for one named condition. Each is a UHD that is well-formed except in the
/// one way the condition describes, so the branch under test is the only difference.
std::shared_ptr<IKernelHeuristic> heuristicForCondition(  // NOLINT(misc-use-internal-linkage)
    const std::string& condition,
    const std::filesystem::path& dir)
{
    if(condition == "healthy_model")
    {
        const auto fixture = writeFixture(dir, preferLargeTiles());
        return makeKernelHeuristic(modelDescriptor(dir, fixture), "e", KNOBS);
    }
    if(condition == "features_hash_disagrees")
    {
        // §6.3 check 1: the model was trained on a different signature than the one the
        // extractor will produce.
        const auto fixture
            = writeFixture(dir, preferLargeTiles(), "max", "sha256:not_the_real_hash");
        return makeKernelHeuristic(modelDescriptor(dir, fixture), "e", KNOBS);
    }
    if(condition == "knobs_disagree_with_axes")
    {
        // §6.3 check 2: the UED advertises a knob the model has no axis for.
        const auto fixture = writeFixture(dir, preferLargeTiles());
        return makeKernelHeuristic(modelDescriptor(dir, fixture), "e", {"tile_m", "split_k"});
    }
    if(condition == "calibrated_and_minimising")
    {
        // §15.1: the one combination the loader refuses outright. It is refused at the
        // parse now -- the descriptor IS the UHD -- so the engine reaches the factory with
        // no descriptor, and this goes through the document rather than round it.
        const auto fixture
            = writeFixture(dir, preferLargeTiles(), "min", {}, /*calibrated=*/true);
        return makeKernelHeuristic(
            parseUhd(uhdDocument(fixture.modelFileName, "min", /*calibrated=*/true),
                     dir / "test.uhd.json"),
            "e",
            KNOBS);
    }
    if(condition == "no_uhd_at_all")
    {
        // §5 step 6: shipping no heuristic is valid and is the starting state.
        return makeKernelHeuristic(std::nullopt, "e", KNOBS);
    }
    if(condition == "arch_not_covered")
    {
        // §8.3 third step: models exist, but none for this architecture and no default.
        const auto fixture = writeFixture(dir, preferLargeTiles());
        const std::map<std::string, HeuristicDescriptor> byArch{
            {"gfx950", modelDescriptor(dir, fixture)}};
        return makeKernelHeuristic(std::nullopt, "e", KNOBS, byArch);
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

    // Every condition still produces a usable ranking -- §5 step 7's "it never fails the
    // request" -- and says which of the three produced it.
    const testing::TestGraph graph;
    auto properties = gfx942();
    if(std::string(condition.name) == "arch_not_covered")
    {
        properties.gcnArchName = "gfx1100";
    }
    const MatchContext context{graph, 0, properties};

    auto recorder = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(
        HIPDNN_SEV_INFO);
    const auto ranked = heuristic->rankScored(catalogAgainstPriority(2048), context);

    EXPECT_EQ(ranked.size(), 2U) << "a degraded ranking still has to answer";

    // The logged trace, not the context-free accessor. Provenance is a fact about one ranking:
    // an arch resolver decides by model on the architectures it names and by declared order on
    // the ones it does not, so only the line emitted where the device is known can say which.
    EXPECT_TRUE(recorder.hasLogContaining(std::string("decided_by=")
                                          + condition.expectedProvenance))
        << "the trace did not report " << condition.expectedProvenance;
}

INSTANTIATE_TEST_SUITE_P(
    SelectionConditions,
    TestIngestorUhdProvenance,
    ::testing::Values(SelectionCondition{"healthy_model", "model"},
                      SelectionCondition{"features_hash_disagrees", "declared_order"},
                      SelectionCondition{"knobs_disagree_with_axes", "declared_order"},
                      SelectionCondition{"calibrated_and_minimising", "declared_order"},
                      SelectionCondition{"no_uhd_at_all", "declared_order"},
                      SelectionCondition{"arch_not_covered", "declared_order"}),
    [](const ::testing::TestParamInfo<SelectionCondition>& info) {
        return std::string(info.param.name);
    });

/// RFC 0019 §6.3 check 2 compares two sets, and the comparison is only as good as the two
/// sides. An empty set on either side compares equal to an empty set on the other, so a
/// defect that empties one passes vacuously on every engine whose model happens to read no
/// `$kernel.*` feature -- and rejects every engine whose model does. The three cases below
/// pin each side and then the comparison itself.
namespace
{

/// gfx942, matching the fixtures' training arch, so the out-of-distribution path stays out
/// of the engine-level cases. The shared StubDeviceResolver reports gfx000.
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
/// catalog whose declared order is the opposite of what that model prefers.
///
/// `split_k` is in the schema but on no kernel, purely so a case can expose it as a knob the
/// model has no axis for -- GenericEngine refuses a knob its schema does not declare, so a
/// mismatch has to be a field that exists.
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
    set.schema.fields
        = {{"tile_m", MetadataType::INT, MetadataValue{int64_t{64}}},
           {"split_k", MetadataType::INT, MetadataValue{int64_t{1}}},
           {testing::BLOCK_SIZE, MetadataType::INT, MetadataValue{int64_t{64}}}};

    set.heuristic = modelDescriptor(dir, fixture);
    set.dispatches = testing::makeStubDispatches();

    KernelDescriptorPack pack;
    pack.id = testing::PACK_ID;
    pack.name = "test pack";
    pack.engineId = testing::ENGINE_ID;
    pack.dispatchId = testing::DISPATCH_ID;
    // Small tile at high priority, large tile at low: declared order and the model disagree,
    // so the two provenances are distinguishable by more than the log line alone.
    pack.kernels = {kernelDescriptorWith(0x01, 64, 100), kernelDescriptorWith(0x02, 128, 1)};
    set.packs = {std::move(pack)};

    return set;
}

/// Builds the engine @p set describes through the production entry point and ranks its
/// catalog once, returning what RFC 0019 §12's trace said decided.
///
/// Through makeEngine() rather than makeStateManager(): makeEngine is where the UED is moved
/// from, and therefore the only place the knobs can be read after they are gone.
std::string provenanceOfEngineRanking(DescriptorSet set)
{
    const testing::ScopedTestSymbols symbols;
    const testing::StubWorkspaceHandler handler;
    const testing::ScopedDispatchRegistration<testing::StubHandle> dispatch(
        "hipdnn.kernel_ingestor.test.dispatch", handler);

    const Gfx942DeviceResolver resolver;
    auto engine
        = makeEngine<testing::StubHandle, testing::StubSettings, testing::StubContext>(
            std::move(set), resolver);
    if(engine == nullptr)
    {
        return "<no engine>";
    }

    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    const testing::StubHandle handle;
    const testing::TestGraph graph(testing::makeGraphId(0x71));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper emptyConfig(nullptr,
                                                                                        0);
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
    // One side of the §6.3 comparison, on its own. A bare reference is the canonical spelling
    // and is not valid JSON; reading the axes with a plain json::parse throws on every entry
    // and yields an empty set, which the comparison cannot distinguish from "the model reads
    // no kernel feature". Asserting the exact set, not merely that it is non-empty, so an
    // axis silently dropped from a mixed signature is caught too.
    const auto axes = kernelAxesOf(BARE_SIGNATURE);

    EXPECT_FALSE(axes.empty()) << "an empty axis set is indistinguishable from agreement";
    EXPECT_EQ(axes, (std::unordered_set<std::string>{"tile_m"}));

    // `$q.*` stays out, and an entry that really is a JsonLogic expression still contributes
    // the axes it reads -- both spellings have to work through the one parse.
    const auto mixed = kernelAxesOf({"$kernel.tile_m",
                                     R"("$q.seqlen")",
                                     R"({"*": [{"var": "$kernel.split_k"}, 2]})"});
    EXPECT_EQ(mixed, (std::unordered_set<std::string>{"tile_m", "split_k"}));
}

TEST(TestIngestorUhdKernelHeuristic, ABareReferenceSignatureStillSatisfiesTheKnobAxisCheck)
{
    // The same side, reached through the check that consumes it. A UED exposing exactly the
    // knob its model ranks on is conformant, and must get its model -- an axis set emptied by
    // the parse turns that into "exposes [tile_m], model ranks on <none>" and refuses it.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_bare_signature");
    const auto fixture = writeFixture(dir.path(),
                                      preferLargeTiles(),
                                      "max",
                                      {},
                                      std::nullopt,
                                      "identity",
                                      BARE_SIGNATURE);

    const auto heuristic = makeKernelHeuristic(modelDescriptor(dir.path(), fixture), "e", KNOBS);
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
    // The other side of the §6.3 comparison, through the production path. makeEngine moves
    // the UED into the engine; the knobs must be read before that move, or the check sees an
    // empty exposed set and refuses the model of every engine that declares a knob at all.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_make_engine_knobs");
    const auto fixture = writeFixture(dir.path(),
                                      preferLargeTiles(),
                                      "max",
                                      {},
                                      std::nullopt,
                                      "identity",
                                      ENGINE_SIGNATURE);

    EXPECT_EQ(provenanceOfEngineRanking(engineSetRankingOnTileM(dir.path(), fixture, {"tile_m"})),
              "model");
}

TEST(TestIngestorUhdEngineKnobContract, AnEngineExposingAKnobItsModelDoesNotRankOnIsRefused)
{
    // The comparison itself, so neither of the two cases above can be satisfied by a check
    // that has quietly stopped comparing anything. Same engine, same model; the UED adds a
    // knob the model has no axis for, which §6.3 requires be refused.
    const hipdnn_test_sdk::utilities::ScopedDirectory dir("uhd_make_engine_knob_mismatch");
    const auto fixture = writeFixture(dir.path(),
                                      preferLargeTiles(),
                                      "max",
                                      {},
                                      std::nullopt,
                                      "identity",
                                      ENGINE_SIGNATURE);

    EXPECT_EQ(provenanceOfEngineRanking(
                  engineSetRankingOnTileM(dir.path(), fixture, {"tile_m", "split_k"})),
              "declared_order");
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
