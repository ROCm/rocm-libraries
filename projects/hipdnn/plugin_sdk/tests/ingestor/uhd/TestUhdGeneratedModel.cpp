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
    catalog.bound["q.seqlen"] = seqlen;
    return catalog;
}

/// The committed `dtype_selector` pair, whose signature reads a STRING field.
///
/// A second fixture rather than a richer first one: `tile_selector` is numeric
/// throughout, and it has to stay that way to keep proving that a signature with
/// no categorical field hashes exactly as it did before the field existed.
HeuristicDescriptor dtypeDescriptor()
{
    const auto dir = fixtureDir() / "dtype_selector";
    const auto path = dir / "dtype_selector.uhd.json";
    std::ifstream stream(path);
    auto descriptor = detail::parseHeuristicDescriptor(nlohmann::json::parse(stream), path);
    descriptor.treeRoot = dir;
    return descriptor;
}

/// The same two tiles, with the dtype the model actually splits on bound.
Catalog catalogForDtype(const std::string& dtype)
{
    Catalog catalog = catalogAgainstPriority(1024);
    for(auto& entry : catalog.entries)
    {
        entry.metadata["dtype"] = dtype;
    }
    return catalog;
}

const std::vector<std::string> DTYPE_KNOBS = {"dtype", "tile_m"};

/// The knobs an engine shipping this model would declare.
///
/// RFC 0019 §6.3 check 2 requires the engine's exposed knobs to be exactly the model's
/// `$kernel.*` axes, and the fixture's signature is `["$kernel.tile_m", "$q.seqlen"]`, so
/// `tile_m` is the whole set. Passing it is not test scaffolding: a caller that omits it
/// is describing an engine that varies nothing, and a model ranking on `tile_m` there is
/// the broken contract the check exists to refuse.
const std::vector<std::string> KNOBS = {"tile_m"};

} // namespace

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

// ---- A signature that reads a string (RFC 0019 §6.5) ---------------------------

TEST(TestIngestorUhdGeneratedModel, TheModelRanksOnTheStringItWasTrainedOn)
{
    // End to end: a category reaches the model as a number and changes the answer.
    // The two catalogs differ ONLY in dtype -- same tiles, same priorities, same
    // seqlen -- so a model that never saw the string cannot produce this flip, and a
    // model reading it through the wrong codes produces the flip backwards.
    const auto heuristic = makeKernelHeuristic(dtypeDescriptor(), {}, DTYPE_KNOBS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    auto properties = testing::testDeviceProperties();
    properties.gcnArchName = "gfx942";
    const MatchContext context{graph, 0, properties};

    const auto wide = heuristic->rank(catalogForDtype("BF16"), context);
    const auto narrow = heuristic->rank(catalogForDtype("FP16"), context);

    ASSERT_EQ(wide.size(), 2U);
    ASSERT_EQ(narrow.size(), 2U);
    EXPECT_EQ(wide.front().kernelId, testId(0x02)) << "BF16 was trained to want tile 128";
    EXPECT_EQ(narrow.front().kernelId, testId(0x01)) << "FP16 was trained to want tile 64";
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
