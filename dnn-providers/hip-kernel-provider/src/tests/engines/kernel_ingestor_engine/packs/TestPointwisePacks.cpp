// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"

/**
 * @file TestPointwisePacks.cpp
 * @brief The engine's descriptor data: its shape, its cross-references, and its engine id.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

/// @brief The pack whose kernels compile @p operation, by source file.
const hipdnn_plugin_sdk::ingestor::KernelDescriptorPack&
    packFor(const hipdnn_plugin_sdk::ingestor::DescriptorSet& set, const std::string& operation)
{
    const auto match = std::find_if(set.packs.begin(), set.packs.end(), [&](const auto& pack) {
        return !pack.kernels.empty()
               && pack.kernels.front().source.sourceFile == "Pointwise" + operation + ".cpp";
    });
    if(match == set.packs.end())
    {
        throw std::runtime_error("no pack whose kernels compile Pointwise" + operation + ".cpp");
    }
    return *match;
}

TEST(TestPointwisePacks, EachPackShipsThreeKernelsCoveringTwoBlockSizesAndTwoDataTypes)
{
    const auto& set = loadedSet("hipkernel:Pointwise");

    ASSERT_EQ(set.packs.size(), 2U);
    for(const auto& pack : set.packs)
    {
        const auto& kernels = pack.kernels;
        ASSERT_EQ(kernels.size(), 3U) << pack.name;

        // The two FLOAT kernels differ only in block size; HALF is pruned on a FLOAT graph.
        const auto describes = [&kernels](int64_t blockSize, const std::string& dtype) {
            return std::any_of(kernels.begin(), kernels.end(), [&](const auto& kernel) {
                return std::get<int64_t>(kernel.metadata.at(std::string(BLOCK_SIZE_FIELD)))
                           == blockSize
                       && std::get<std::string>(kernel.metadata.at(std::string(DTYPE_FIELD)))
                              == dtype;
            });
        };

        EXPECT_TRUE(describes(64, "FLOAT")) << pack.name;
        EXPECT_TRUE(describes(256, "FLOAT")) << pack.name;
        EXPECT_TRUE(describes(64, "HALF")) << pack.name;
    }
}

TEST(TestPointwisePacks, EveryKernelNamesItsPacksEmbeddedSource)
{
    const auto& set = loadedSet("hipkernel:Pointwise");

    for(const std::string operation : {"Add", "Mul"})
    {
        for(const auto& kernel : packFor(set, operation).kernels)
        {
            EXPECT_EQ(kernel.source.kind,
                      hipdnn_plugin_sdk::ingestor::KernelSourceKind::EMBEDDED_SOURCE);
            EXPECT_EQ(kernel.source.sourceFile, "Pointwise" + operation + ".cpp");
            EXPECT_EQ(kernel.source.entryPoint, "Pointwise" + operation);
        }
    }
}

/// The point of two packs under one engine: everything but the operation matcher and the
/// kernels is one descriptor referenced twice, not two copies.
TEST(TestPointwisePacks, BothPacksShareTheEngineDispatchAndAllButOneMatcher)
{
    const auto& set = loadedSet("hipkernel:Pointwise");
    const auto& add = packFor(set, "Add");
    const auto& mul = packFor(set, "Mul");

    EXPECT_EQ(add.engineId, mul.engineId);
    EXPECT_EQ(add.dispatchId, mul.dispatchId);
    ASSERT_EQ(set.dispatches.size(), 1U);

    // Three matchers each: the shared applicability check, the shared kernel-scoped dtype
    // check, and one operation check of their own. Two shared ids is what makes the
    // expensive graph work run once per graph instead of once per pack.
    ASSERT_EQ(add.matcherIds.size(), 3U);
    ASSERT_EQ(mul.matcherIds.size(), 3U);

    // Counted rather than set_intersection'd: matcher ids are in the order the pack
    // authored them, not sorted, and a sorted-range algorithm would quietly under-count.
    EXPECT_EQ(std::count_if(add.matcherIds.begin(),
                            add.matcherIds.end(),
                            [&mul](const auto& matcherId) {
                                return std::find(mul.matcherIds.begin(),
                                                 mul.matcherIds.end(),
                                                 matcherId)
                                       != mul.matcherIds.end();
                            }),
              2);
}

TEST(TestPointwisePacks, ExposesBlockSizeAsAKnobAndDtypeAsInternal)
{
    const auto& set = loadedSet("hipkernel:Pointwise");

    // dtype is pinned by the graph rather than chosen.
    ASSERT_EQ(set.engine.knobs.size(), 1U);
    EXPECT_EQ(set.engine.knobs.front(), std::string(BLOCK_SIZE_FIELD));
}

TEST(TestPointwisePacks, MatchersCoverBothScopes)
{
    const auto& set = loadedSet("hipkernel:Pointwise");

    // Three graph-scoped: one shared applicability check both packs list, plus one
    // operation check each. One kernel-scoped, shared, pruning per candidate.
    EXPECT_EQ(std::count_if(set.matchers.begin(),
                            set.matchers.end(),
                            [](const auto& matcher) {
                                return matcher.scope
                                       == hipdnn_plugin_sdk::ingestor::MatchScope::GRAPH;
                            }),
              3);
    EXPECT_EQ(std::count_if(set.matchers.begin(),
                            set.matchers.end(),
                            [](const auto& matcher) {
                                return matcher.scope
                                       == hipdnn_plugin_sdk::ingestor::MatchScope::KERNEL;
                            }),
              1);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
