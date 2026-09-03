// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelHeuristicFactory.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../KernelIngestorTestFixtures.hpp"

/**
 * @file TestUhdGeneratedModel.cpp
 * @brief The seam between the training tool and the runtime.
 *
 * Everything else in the UHD suites builds its model in-process with the
 * generated C++ builder, which is why a Python writer whose vtable was two slots
 * short shipped green for as long as it did: the two sides never exchanged a
 * file. These load committed `uhd_gen` output instead.
 *
 * Three contracts hold this together, and each of them has already been broken:
 *
 *  - the descriptor JSON, parsed by `DescriptorLoader` before a field is read
 *  - `features_hash`, computed by Python and recomputed by C++ from the
 *    signature in the same file; a mismatch refuses the load
 *  - the artifact path in `tree_data`, relative to the `.uhd.json` that
 *    declares it
 *
 * What is deliberately NOT here: executing the chosen kernel. That needs a
 * packaged descriptor tree and a device, and lives with the provider's
 * integration suite. These prove the artifact loads and reorders a catalog,
 * which is the part no test covered.
 */
namespace hipdnn_plugin_sdk::ingestor
{
namespace
{

/// Where the committed `uhd_gen` output lives, relative to this source file.
///
/// The path is a compile definition rather than a runtime search: the fixture is
/// source, not a build output, and probing for it would turn a missing file into
/// a skip when it should be a failure.
std::filesystem::path fixtureDir()
{
    return std::filesystem::path(HIPDNN_UHD_GENERATED_FIXTURE_DIR);
}

/// The committed `tile_selector.uhd.json`, through the loader's own parser.
///
/// Parsed rather than retyped: a descriptor built here would pass whatever the
/// tool emitted through a second, hand-maintained spelling of the schema, and
/// the seam this suite exists to hold is exactly that the tool writes what the
/// parser reads.
HeuristicDescriptor generatedDescriptor()
{
    const auto path = fixtureDir() / "tile_selector.uhd.json";
    std::ifstream stream(path);
    auto descriptor = detail::parseHeuristicDescriptor(nlohmann::json::parse(stream), path);
    descriptor.treeRoot = fixtureDir();
    return descriptor;
}

DescriptorId testId(uint8_t tag)
{
    DescriptorId id{};
    id.fill(0);
    id[0] = tag;
    return id;
}

/// Two kernels differing only in tile, with `priority` set AGAINST the model.
///
/// The small tile is declared first and ranks first without a heuristic, so any
/// ordering the model does not produce is the fallback's, not a coincidence.
Catalog catalogAgainstPriority(int64_t seqlen)
{
    Catalog catalog;

    KernelDefinition small;
    small.kernelId = testId(0x01);
    small.priority = 10;
    small.metadata["tile_m"] = int64_t{64};

    KernelDefinition large;
    large.kernelId = testId(0x02);
    large.priority = 1;
    large.metadata["tile_m"] = int64_t{128};

    catalog.entries = {small, large};
    catalog.bound["seqlen"] = seqlen;
    return catalog;
}

/// The knobs an engine shipping this model would declare.
///
/// RFC 0019 §6.3 check 2 requires the engine's exposed knobs to be exactly the model's
/// `$kernel.*` axes, and the fixture's signature is `["$kernel.tile_m", "$q.seqlen"]`, so
/// `tile_m` is the whole set. Passing it is not test scaffolding: a caller that omits it
/// is describing an engine that varies nothing, and a model ranking on `tile_m` there is
/// the broken contract the check exists to refuse.
const std::vector<std::string> KNOBS = {"tile_m"};

} // namespace

TEST(TestIngestorUhdGeneratedModel, TheToolsOwnOutputLoads)
{
    // The whole chain in one assertion: the descriptor parses, its features_hash
    // matches what the C++ extractor recomputes from the signature beside it, and
    // model.bin resolves relative to the descriptor and loads. Any one of the
    // three failing returns nullptr here.
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);

    const auto heuristic = makeKernelHeuristic(generatedDescriptor(), {}, KNOBS);

    ASSERT_NE(heuristic, nullptr);
    EXPECT_EQ(recorder.getRecordedLogCount(), 0U)
        << "loading the tool's output must be silent: "
        << recorder.getRecordedLogsAsString();
}

TEST(TestIngestorUhdGeneratedModel, TheModelDecidesTheOrderRatherThanPriority)
{
    const auto heuristic = makeKernelHeuristic(generatedDescriptor(), {}, KNOBS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    auto properties = testing::testDeviceProperties();
    properties.gcnArchName = "gfx942";
    const MatchContext context{graph, 0, properties};

    // 4096 is in the region the training data says the large tile wins.
    const auto ranked = heuristic->rank(catalogAgainstPriority(4096), context);

    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, testId(0x02))
        << "the large tile is last by priority and must be first by score";
}

TEST(TestIngestorUhdGeneratedModel, TheSameCatalogRanksDifferentlyForADifferentProblem)
{
    // The case that separates a model from a static order. Both rankings above
    // could be produced by a constant; only a model that reads $q.seqlen flips
    // when the problem does, and that flip is the reason a UHD exists at all.
    const auto heuristic = makeKernelHeuristic(generatedDescriptor(), {}, KNOBS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    auto properties = testing::testDeviceProperties();
    properties.gcnArchName = "gfx942";
    const MatchContext context{graph, 0, properties};

    const auto longSequence = heuristic->rank(catalogAgainstPriority(4096), context);
    const auto shortSequence = heuristic->rank(catalogAgainstPriority(128), context);

    ASSERT_EQ(longSequence.size(), 2U);
    ASSERT_EQ(shortSequence.size(), 2U);
    EXPECT_EQ(longSequence.front().kernelId, testId(0x02)) << "long sequence wants tile 128";
    EXPECT_EQ(shortSequence.front().kernelId, testId(0x01)) << "short sequence wants tile 64";
}

TEST(TestIngestorUhdGeneratedModel, TheCommittedDescriptorNamesTheCommittedArtifact)
{
    // The artifact reference, checked as a file rather than inferred from a
    // successful load. A fixture regenerated with a different --descriptor-name
    // would still load while silently no longer matching what the docs describe.
    const auto descriptorPath = fixtureDir() / "tile_selector.uhd.json";
    ASSERT_TRUE(std::filesystem::exists(descriptorPath));

    std::ifstream stream(descriptorPath);
    const auto document = nlohmann::json::parse(stream);
    EXPECT_EQ(document.at("adapter"), "tree_data");
    EXPECT_EQ(document.at("tree_data").at("artifact"), "model.bin");

    EXPECT_TRUE(std::filesystem::exists(fixtureDir() / "model.bin"));
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
