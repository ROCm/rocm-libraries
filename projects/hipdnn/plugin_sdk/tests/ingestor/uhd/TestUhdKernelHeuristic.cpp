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

#include <gtest/gtest.h>

#include "../KernelIngestorTestFixtures.hpp"

#include <hipdnn_plugin_sdk/ingestor/KernelHeuristicFactory.hpp>
#include <hipdnn_plugin_sdk/ingestor/UhdKernelHeuristic.hpp>
#include <hipdnn_plugin_sdk/ingestor/uhd/FeatureExtractor.hpp>
#include <hipdnn_plugin_sdk/ingestor/uhd/NativeScorerRegistry.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/uhd_generated.h>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/GbdtModelTestBuilder.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace hipdnn_plugin_sdk::ingestor
{
namespace
{

namespace fbs = hipdnn_flatbuffers_sdk::data_objects;

// The signature both the descriptor and the tests agree on: slot 0 is a kernel knob, slot
// 1 a problem token. Two slots from two namespaces is the minimum that can show the
// interaction reaching the model.
const std::vector<std::string> SIGNATURE = {R"("$kernel.tile_m")", R"("$q.seqlen")"};

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

/// Writes a `.uhd.fb` naming a model artifact, and returns its filename.
///
/// Local to this file rather than shared: the loader's own suites build their descriptors
/// inline too, and a builder covering every field none of them set would be a third
/// spelling of the schema to keep in step.
std::string writeUhd(const std::filesystem::path& dir,
                     const std::string& modelFileName,
                     const std::string& objective,
                     const std::string& featuresHash)
{
    flatbuffers::FlatBufferBuilder builder;

    auto id = builder.CreateString("test-uhd");
    auto name = builder.CreateString("Test UHD");
    auto hash = builder.CreateString(featuresHash);
    auto obj = builder.CreateString(objective);
    auto units = builder.CreateString("tflops");
    auto transform = builder.CreateString("identity");
    auto artifact = builder.CreateString(modelFileName);

    std::vector<flatbuffers::Offset<flatbuffers::String>> signature;
    signature.reserve(SIGNATURE.size());
    for(const auto& entry : SIGNATURE)
    {
        signature.push_back(builder.CreateString(entry));
    }
    auto signatureVec = builder.CreateVector(signature);
    auto score = fbs::CreateUhdScoreMetadata(builder, units, true, transform);

    auto uhd = fbs::CreateUHD(builder,
                              id,
                              name,
                              fbs::UhdAdapter::TREE_DATA,
                              0, // derived
                              signatureVec,
                              hash,
                              obj,
                              score,
                              artifact);
    builder.Finish(uhd, "HUHD");

    const std::string fileName = "test.uhd.fb";
    std::ofstream out(dir / fileName, std::ios::binary);
    out.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),
              static_cast<std::streamsize>(builder.GetSize()));
    out.close();
    return fileName;
}

struct Fixture
{
    std::string uhdFileName;
    std::string featuresHash;
};

/// Writes a matching `.uhd.fb` and GBDT artifact into @p dir.
///
/// @param modelHash Written into the model artifact. Defaults to the signature's real
///        hash; a caller passing something else is testing the contract check.
Fixture writeFixture(const std::filesystem::path& dir,
                     const hipdnn_test_sdk::utilities::GbdtModelTestBuilder::TreeSpec& tree,
                     const std::string& objective = "max",
                     const std::string& modelHash = {})
{
    const std::string signatureHash = uhd::FeatureExtractor::computeHash(SIGNATURE);

    hipdnn_test_sdk::utilities::GbdtModelTestBuilder model;
    model.setFeaturesHash(modelHash.empty() ? signatureHash : modelHash)
        .setNumFeatures(static_cast<int32_t>(SIGNATURE.size()))
        .setTrainingArches({"gfx942"})
        .addTree(tree);
    model.buildToFile((dir / "model.bin").string());

    return {writeUhd(dir, "model.bin", objective, signatureHash), signatureHash};
}

HeuristicDescriptor modelDescriptor(const std::filesystem::path& dir, const std::string& payload)
{
    HeuristicDescriptor descriptor;
    descriptor.id = testId(0xEE);
    descriptor.name = "test model heuristic";
    descriptor.kind = HeuristicKind::MODEL;
    descriptor.payload = payload;
    descriptor.baseDir = dir;
    return descriptor;
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

    const auto heuristic = makeKernelHeuristic(modelDescriptor(dir.path(), fixture.uhdFileName), {}, KNOBS);
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

    const auto heuristic = makeKernelHeuristic(modelDescriptor(dir.path(), fixture.uhdFileName), {}, KNOBS);
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
        = makeKernelHeuristic(modelDescriptor(maxDir.path(), maxFixture.uhdFileName), {}, KNOBS);
    const auto minimising
        = makeKernelHeuristic(modelDescriptor(minDir.path(), minFixture.uhdFileName), {}, KNOBS);
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

    const auto heuristic = makeKernelHeuristic(modelDescriptor(dir.path(), "not_written.uhd.fb"));
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

    const auto heuristic = makeKernelHeuristic(modelDescriptor(dir.path(), fixture.uhdFileName), {}, KNOBS);
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

    const auto heuristic = makeKernelHeuristic(modelDescriptor(dir.path(), fixture.uhdFileName), {}, KNOBS);
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

    const auto heuristic = makeKernelHeuristic(modelDescriptor(dir.path(), fixture.uhdFileName), {}, KNOBS);
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

#endif // HIPDNN_ENABLE_K
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
    const auto heuristic = makeKernelHeuristic(modelDescriptor(dir.path(), fixture.uhdFileName),
                                               {},
                                               {"tile_m", "split_k"});
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
    const auto heuristic
        = makeKernelHeuristic(modelDescriptor(dir.path(), fixture.uhdFileName), {}, {});
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
        {"default", modelDescriptor(defaultDir.path(), fallback.uhdFileName)},
        {"gfx942", modelDescriptor(archDir.path(), specific.uhdFileName)}};

    const auto heuristic = makeKernelHeuristic(
        modelDescriptor(defaultDir.path(), fallback.uhdFileName), {}, KNOBS, byArch);
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
        {"default", modelDescriptor(defaultDir.path(), fallback.uhdFileName)},
        {"gfx942", modelDescriptor(archDir.path(), specific.uhdFileName)}};

    const auto heuristic = makeKernelHeuristic(
        modelDescriptor(defaultDir.path(), fallback.uhdFileName), {}, KNOBS, byArch);
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
        {"default", modelDescriptor(defaultDir.path(), fallback.uhdFileName)},
        {"gfx942", modelDescriptor(archDir.path(), specific.uhdFileName)}};

    const auto heuristic = makeKernelHeuristic(
        modelDescriptor(defaultDir.path(), fallback.uhdFileName), {}, KNOBS, byArch);
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

} // namespace hipdnn_plugin_sdk::ingestor
