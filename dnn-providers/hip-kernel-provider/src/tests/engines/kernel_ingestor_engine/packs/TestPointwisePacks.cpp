// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

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

    ASSERT_EQ(set.packs.size(), 3U);
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

    for(const std::string operation : {"Add", "Mul", "Sub"})
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

/// The point of three packs under one engine: everything but the operation matcher and
/// the kernels is one descriptor referenced three times, not three copies.
TEST(TestPointwisePacks, EveryPackSharesTheEngineDispatchAndAllButOneMatcher)
{
    const auto& set = loadedSet("hipkernel:Pointwise");
    const auto& add = packFor(set, "Add");
    const auto& mul = packFor(set, "Mul");
    const auto& sub = packFor(set, "Sub");

    // Load-bearing form: packs land in set.packs by resolveDescriptorSets() selecting
    // engineId == set.engine.id, so comparing packs to each other proves nothing.
    EXPECT_EQ(add.engineId, set.engine.id);
    EXPECT_EQ(mul.engineId, set.engine.id);
    EXPECT_EQ(sub.engineId, set.engine.id);
    EXPECT_EQ(add.dispatchId, mul.dispatchId);
    EXPECT_EQ(add.dispatchId, sub.dispatchId);
    ASSERT_EQ(set.dispatches.size(), 1U);

    // Three matchers each: the shared applicability check, the shared kernel-scoped dtype
    // check, and one operation check of their own. Two shared ids is what makes the
    // expensive graph work run once per graph instead of once per pack.
    ASSERT_EQ(add.matcherIds.size(), 3U);
    ASSERT_EQ(mul.matcherIds.size(), 3U);
    ASSERT_EQ(sub.matcherIds.size(), 3U);

    // Counted rather than set_intersection'd: matcher ids are in the order the pack
    // authored them, not sorted, and a sorted-range algorithm would quietly under-count.
    const auto lists = {&mul.matcherIds, &sub.matcherIds};
    EXPECT_EQ(std::count_if(add.matcherIds.begin(),
                            add.matcherIds.end(),
                            [&lists](const auto& matcherId) {
                                return std::all_of(lists.begin(),
                                                   lists.end(),
                                                   [&matcherId](const auto* other) {
                                                       return std::find(other->begin(),
                                                                        other->end(),
                                                                        matcherId)
                                                              != other->end();
                                                   });
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

    // Four graph-scoped: one shared applicability check every pack lists, plus one
    // operation check each. One kernel-scoped, shared, pruning per candidate.
    EXPECT_EQ(std::count_if(set.matchers.begin(),
                            set.matchers.end(),
                            [](const auto& matcher) {
                                return matcher.scope
                                       == hipdnn_plugin_sdk::ingestor::MatchScope::GRAPH;
                            }),
              4);
    EXPECT_EQ(std::count_if(set.matchers.begin(),
                            set.matchers.end(),
                            [](const auto& matcher) {
                                return matcher.scope
                                       == hipdnn_plugin_sdk::ingestor::MatchScope::KERNEL;
                            }),
              1);
}

/// The GPU suite (ExecutesASubtractGraphThroughItsOwnPack) proves the kernel itself
/// computes a - b; add and mul are commutative so an operand swap there is invisible,
/// but sub is asymmetric, so this pins the fast, device-free half: binding never
/// swaps input_a/input_b before dispatch gets them.
TEST(TestPointwisePacks, SubtractsInTheRightDirection)
{
    const GraphFixture fixture(
        buildPointwiseGraph(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SUB));

    hipdnn_plugin_sdk::ingestor::BoundTokens bound;
    ASSERT_TRUE(matchesGraph(POINTWISE_SUB, fixture.context(), bound));

    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, POINTWISE_SUB.inputAToken),
              INPUT_A_UID);
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, POINTWISE_SUB.inputBToken),
              INPUT_B_UID);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
