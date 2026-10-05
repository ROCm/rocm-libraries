// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelHeuristicFactory.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../../TestResourcePaths.hpp"
#include "../KernelIngestorTestFixtures.hpp"

/**
 * @file TestUhdGeneratedModel.cpp
 * @brief Loads committed `uhd_gen` output, rather than an in-process model, so the tool's
 * writer and the runtime's reader exchange a real file. Executing the chosen kernel needs a
 * device and lives in the provider's integration suite.
 */
namespace hipdnn_plugin_sdk::ingestor
{
namespace
{

/// Where the committed `uhd_gen` output is staged. A missing fixture fails rather than skips.
std::filesystem::path fixtureDir()
{
    return hipdnn_plugin_sdk::test::uhdGeneratedFixtureDir();
}

/// The committed `tile_selector.uhd.json`, parsed by the loader rather than retyped, so the
/// test checks that the parser reads what the tool writes.
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

/// Two kernels differing only in tile, with `priority` set against the model: without a
/// heuristic the small tile ranks first.
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

/// The committed `dtype_selector` pair, whose signature reads a string field. Kept separate
/// so `tile_selector` stays all-numeric and keeps its pre-categorical hash.
HeuristicDescriptor dtypeDescriptor()
{
    const auto dir = fixtureDir() / "dtype_selector";
    const auto path = dir / "dtype_selector.uhd.json";
    std::ifstream stream(path);
    auto descriptor = detail::parseHeuristicDescriptor(nlohmann::json::parse(stream), path);
    descriptor.treeRoot = dir;
    return descriptor;
}

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

/// RFC 0019 §6.3 check 2: exposed knobs must equal the model's `$kernel.*` axes, which for
/// this fixture is `tile_m` alone.
const std::vector<std::string> KNOBS = {"tile_m"};

/// KMD fields: the knobs plus any dispatch-only fields (§3.2, §6.3 check 2).
const std::unordered_set<std::string> FIELDS = {"tile_m"};
const std::unordered_set<std::string> DTYPE_FIELDS = {"dtype", "tile_m"};

} // namespace

TEST(TestIngestorUhdGeneratedModel, TheModelDecidesTheOrderRatherThanPriority)
{
    const auto heuristic = makeKernelHeuristic(generatedDescriptor(), {}, KNOBS, FIELDS);
    ASSERT_NE(heuristic, nullptr);

    const testing::TestGraph graph;
    auto properties = testing::testDeviceProperties();
    properties.gcnArchName = "gfx942";
    const MatchContext context{graph, 0, properties};

    // The training data has the large tile winning at seqlen 4096.
    const auto ranked = heuristic->rank(catalogAgainstPriority(4096), context);

    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, testId(0x02))
        << "the large tile is last by priority and must be first by score";
}

TEST(TestIngestorUhdGeneratedModel, TheSameCatalogRanksDifferentlyForADifferentProblem)
{
    // A constant order cannot produce this flip; only a model reading $q.seqlen can.
    const auto heuristic = makeKernelHeuristic(generatedDescriptor(), {}, KNOBS, FIELDS);
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
    // The catalogs differ only in dtype, so the flip comes from the categorical encoding;
    // wrong codes would flip it backwards.
    const auto heuristic = makeKernelHeuristic(dtypeDescriptor(), {}, DTYPE_KNOBS, DTYPE_FIELDS);
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
