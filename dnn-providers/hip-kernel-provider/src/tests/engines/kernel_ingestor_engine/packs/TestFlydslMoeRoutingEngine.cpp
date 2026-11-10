// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslMoeRoutingCommon.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslMoeRoutingTestGraphs.hpp"

/**
 * @file TestFlydslMoeRoutingEngine.cpp
 * @brief hipkernel:flydsl_moe_routing's matchers, scorer and launch split, on graphs built
 *        in-process.
 *
 * Host-only: no device, no shard. What the pack ships is censused in
 * TestFlydslMoeRoutingPacks.cpp, and every staged object is launched against the CPU
 * reference in TestFlydslMoeRoutingObjects.cpp.
 *
 * The graph match accepts one MoE routing node over packed INT32 topk_ids / token_index /
 * token_ks and INT32 or INT64 first_token_offset, in the op's shapes, and declines
 * everything else. The kernel match bounds num_experts by the object's `e_max`.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_moe_routing;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using data_objects::DataType;

std::optional<BoundTokens> match(const MoeRoutingGraphSpec& spec)
{
    const GraphFixture fixture(buildFlydslMoeRoutingGraph(spec));
    return matchesGraph(FLYDSL_MOE_ROUTING, fixture.context());
}

/// A kernel carrying the metadata the MoE routing KMD declares.
KernelDefinition makeMoeRoutingKernel(int64_t eMax = 8192, const std::string& dtype = "i32")
{
    KernelDefinition kernel;
    kernel.name = "moe_routing_" + dtype + "_emax" + std::to_string(eMax);
    kernel.priority = 100;
    kernel.metadata = {{std::string(FLYDSL_MOE_ROUTING_DTYPE_FIELD), dtype},
                       {std::string(FLYDSL_MOE_ROUTING_E_MAX_FIELD), eMax},
                       {std::string(FLYDSL_MOE_ROUTING_PREFIX_SYMBOL_FIELD),
                        std::string("moe_routing_prefix_kernel_1")},
                       {std::string(FLYDSL_MOE_ROUTING_SCATTER_SYMBOL_FIELD),
                        std::string("moe_routing_scatter_kernel_2")}};
    return kernel;
}

bool kernelMatches(const MoeRoutingGraphSpec& spec, const KernelDefinition& kernel)
{
    const GraphFixture fixture(buildFlydslMoeRoutingGraph(spec));
    const auto bound = matchesGraph(FLYDSL_MOE_ROUTING, fixture.context());
    return bound.has_value()
           && matchesKernel(FLYDSL_MOE_ROUTING, fixture.context(), kernel, *bound);
}

// ---------------------------------------------------------------------------
// graph match: accepted
// ---------------------------------------------------------------------------

TEST(TestFlydslMoeRoutingEngine, AcceptsTheOpAndBindsItsOperandsAndExtents)
{
    const auto bound = match({});
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_MOE_ROUTING.inputAToken),
              FLYDSL_MOE_ROUTING_TOPK_IDS_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_MOE_ROUTING.inputBToken),
              FLYDSL_MOE_ROUTING_OFFSETS_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_MOE_ROUTING.outputToken),
              FLYDSL_MOE_ROUTING_TOKEN_INDEX_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_MOE_ROUTING_TOKEN_KS_TOKEN),
              FLYDSL_MOE_ROUTING_TOKEN_KS_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_MOE_ROUTING_TOKENS_TOKEN), 16);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_MOE_ROUTING_TOP_K_TOKEN), 10);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_MOE_ROUTING_EXPERTS_TOKEN), 512);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_MOE_ROUTING_OFFSET_IS_I64_TOKEN), 0);
}

TEST(TestFlydslMoeRoutingEngine, AcceptsInt64OffsetsAndBindsTheWidth)
{
    MoeRoutingGraphSpec spec;
    spec.offsetDataType = DataType::INT64;
    const auto bound = match(spec);
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_MOE_ROUTING_OFFSET_IS_I64_TOKEN), 1);
}

TEST(TestFlydslMoeRoutingEngine, AcceptsOneTokenOneSlotAndOneExpert)
{
    MoeRoutingGraphSpec spec;
    spec.tokens = 1;
    spec.topK = 1;
    spec.experts = 1;
    EXPECT_TRUE(match(spec).has_value());
}

TEST(TestFlydslMoeRoutingEngine, AcceptsUnitAxesWithAnyStride)
{
    // An extent-1 axis is never stepped along, so its stride is not checked.
    MoeRoutingGraphSpec spec;
    spec.strides[FLYDSL_MOE_ROUTING_OFFSETS_UID] = {1, 0, 0};
    spec.strides[FLYDSL_MOE_ROUTING_TOKEN_INDEX_UID] = {0, 1, 7};
    EXPECT_TRUE(match(spec).has_value());
}

// ---------------------------------------------------------------------------
// graph match: declined
// ---------------------------------------------------------------------------

TEST(TestFlydslMoeRoutingEngine, DeclinesAGraphThatIsNotOneMoeRoutingNode)
{
    MoeRoutingGraphSpec twoNodes;
    twoNodes.secondNode = true;
    EXPECT_FALSE(match(twoNodes).has_value());

    MoeRoutingGraphSpec overrideShape;
    overrideShape.overrideShapeEnabled = true;
    EXPECT_FALSE(match(overrideShape).has_value());

    const GraphFixture rmsNorm(buildFlydslRmsNormGraph());
    EXPECT_FALSE(matchesGraph(FLYDSL_MOE_ROUTING, rmsNorm.context()).has_value());
}

TEST(TestFlydslMoeRoutingEngine, DeclinesDtypesNoObjectReadsOrWrites)
{
    for(const auto uid : {FLYDSL_MOE_ROUTING_TOPK_IDS_UID,
                          FLYDSL_MOE_ROUTING_TOKEN_INDEX_UID,
                          FLYDSL_MOE_ROUTING_TOKEN_KS_UID})
    {
        for(const auto type : {DataType::INT64, DataType::FLOAT, DataType::INT8})
        {
            MoeRoutingGraphSpec spec;
            spec.dataTypes[uid] = type;
            EXPECT_FALSE(match(spec).has_value())
                << "uid " << uid << " as " << data_objects::EnumNameDataType(type);
        }
    }
    for(const auto type : {DataType::FLOAT, DataType::INT8, DataType::UINT8})
    {
        MoeRoutingGraphSpec spec;
        spec.offsetDataType = type;
        EXPECT_FALSE(match(spec).has_value()) << data_objects::EnumNameDataType(type);
    }
}

TEST(TestFlydslMoeRoutingEngine, DeclinesShapesOtherThanTheOps)
{
    // topk_ids not rank 2.
    MoeRoutingGraphSpec rank3;
    rank3.dims[FLYDSL_MOE_ROUTING_TOPK_IDS_UID] = {16, 10, 1};
    EXPECT_FALSE(match(rank3).has_value());

    // first_token_offset not [E + 1, 1, 1].
    MoeRoutingGraphSpec shortOffsets;
    shortOffsets.dims[FLYDSL_MOE_ROUTING_OFFSETS_UID] = {512, 1, 1};
    EXPECT_FALSE(match(shortOffsets).has_value());
    MoeRoutingGraphSpec flatOffsets;
    flatOffsets.dims[FLYDSL_MOE_ROUTING_OFFSETS_UID] = {513};
    EXPECT_FALSE(match(flatOffsets).has_value());

    // Rows not [1, S * top_k, 1].
    for(const auto uid : {FLYDSL_MOE_ROUTING_TOKEN_INDEX_UID, FLYDSL_MOE_ROUTING_TOKEN_KS_UID})
    {
        MoeRoutingGraphSpec shortRows;
        shortRows.dims[uid] = {1, 159, 1};
        EXPECT_FALSE(match(shortRows).has_value()) << "uid " << uid;
        MoeRoutingGraphSpec flatRows;
        flatRows.dims[uid] = {160};
        EXPECT_FALSE(match(flatRows).has_value()) << "uid " << uid;
    }
}

TEST(TestFlydslMoeRoutingEngine, DeclinesNonPositiveExtentsAndExperts)
{
    MoeRoutingGraphSpec noExperts;
    noExperts.experts = 0;
    EXPECT_FALSE(match(noExperts).has_value());

    MoeRoutingGraphSpec negativeExperts;
    negativeExperts.experts = -3;
    negativeExperts.dims[FLYDSL_MOE_ROUTING_OFFSETS_UID] = {1, 1, 1};
    EXPECT_FALSE(match(negativeExperts).has_value());

    MoeRoutingGraphSpec noTokens;
    noTokens.tokens = 0;
    EXPECT_FALSE(match(noTokens).has_value());
}

TEST(TestFlydslMoeRoutingEngine, DeclinesMorePairsThanTheKernelsAddress)
{
    // S * top_k at the 32-bit byte-offset bound and one past it.
    MoeRoutingGraphSpec atBound;
    atBound.tokens = MAX_PAIRS;
    atBound.topK = 1;
    atBound.experts = 8;
    EXPECT_TRUE(match(atBound).has_value());

    MoeRoutingGraphSpec pastBound = atBound;
    pastBound.tokens = MAX_PAIRS + 1;
    EXPECT_FALSE(match(pastBound).has_value());
}

TEST(TestFlydslMoeRoutingEngine, DeclinesNonPackedOperands)
{
    MoeRoutingGraphSpec transposedIds;
    transposedIds.strides[FLYDSL_MOE_ROUTING_TOPK_IDS_UID] = {1, 16};
    EXPECT_FALSE(match(transposedIds).has_value());

    MoeRoutingGraphSpec stridedRows;
    stridedRows.strides[FLYDSL_MOE_ROUTING_TOKEN_KS_UID] = {320, 2, 1};
    EXPECT_FALSE(match(stridedRows).has_value());

    MoeRoutingGraphSpec stridedOffsets;
    stridedOffsets.strides[FLYDSL_MOE_ROUTING_OFFSETS_UID] = {2, 1, 1};
    EXPECT_FALSE(match(stridedOffsets).has_value());
}

TEST(TestFlydslMoeRoutingEngine, DeclinesMisalignedOperands)
{
    for(const auto uid : {FLYDSL_MOE_ROUTING_TOPK_IDS_UID,
                          FLYDSL_MOE_ROUTING_OFFSETS_UID,
                          FLYDSL_MOE_ROUTING_TOKEN_INDEX_UID,
                          FLYDSL_MOE_ROUTING_TOKEN_KS_UID})
    {
        MoeRoutingGraphSpec spec;
        spec.alignment[uid] = 2;
        EXPECT_FALSE(match(spec).has_value()) << "uid " << uid;
    }
    // INT64 offsets are written as 8-byte pairs.
    MoeRoutingGraphSpec wide;
    wide.offsetDataType = DataType::INT64;
    wide.alignment[FLYDSL_MOE_ROUTING_OFFSETS_UID] = 4;
    EXPECT_FALSE(match(wide).has_value());
    wide.alignment[FLYDSL_MOE_ROUTING_OFFSETS_UID] = 8;
    EXPECT_TRUE(match(wide).has_value());
}

TEST(TestFlydslMoeRoutingEngine, DeclinesVirtualAndAliasedOperands)
{
    for(const auto uid : {FLYDSL_MOE_ROUTING_TOPK_IDS_UID,
                          FLYDSL_MOE_ROUTING_OFFSETS_UID,
                          FLYDSL_MOE_ROUTING_TOKEN_INDEX_UID,
                          FLYDSL_MOE_ROUTING_TOKEN_KS_UID})
    {
        MoeRoutingGraphSpec spec;
        spec.virtualTensors.insert(uid);
        EXPECT_FALSE(match(spec).has_value()) << "uid " << uid;
    }
    // token_index written over token_ks.
    MoeRoutingGraphSpec aliased;
    aliased.tokenIndexUid = FLYDSL_MOE_ROUTING_TOKEN_KS_UID;
    EXPECT_FALSE(match(aliased).has_value());
}

// ---------------------------------------------------------------------------
// kernel match and score
// ---------------------------------------------------------------------------

TEST(TestFlydslMoeRoutingEngine, KernelMatchBoundsTheExpertsByTheObjectsTable)
{
    const auto kernel = makeMoeRoutingKernel(8192);
    MoeRoutingGraphSpec spec;
    spec.experts = 8192;
    EXPECT_TRUE(kernelMatches(spec, kernel));
    spec.experts = 8193;
    EXPECT_FALSE(kernelMatches(spec, kernel));

    spec.experts = 512;
    EXPECT_TRUE(kernelMatches(spec, kernel));
    EXPECT_FALSE(kernelMatches(spec, makeMoeRoutingKernel(256)));
    EXPECT_FALSE(kernelMatches(spec, makeMoeRoutingKernel(8192, "i64")));
}

TEST(TestFlydslMoeRoutingEngine, ScoresByPriority)
{
    const GraphFixture fixture(buildFlydslMoeRoutingGraph());
    const auto bound = matchesGraph(FLYDSL_MOE_ROUTING, fixture.context());
    ASSERT_TRUE(bound.has_value());
    auto kernel = makeMoeRoutingKernel();
    kernel.priority = 37;
    EXPECT_EQ(scoreKernel(FLYDSL_MOE_ROUTING, fixture.context(), kernel, *bound), 37.0);
}

// ---------------------------------------------------------------------------
// launch split and workspace
// ---------------------------------------------------------------------------

TEST(TestFlydslMoeRoutingEngine, SplitsPairsIntoWholeTilesWithinTheBounds)
{
    // Decode: one tile, one block.
    const auto decode = routingSplit(10, 512, 304);
    EXPECT_EQ(decode.blocks, 1);
    EXPECT_EQ(decode.span, 256);

    for(const int64_t pairs : {int64_t{1},
                               int64_t{255},
                               int64_t{256},
                               int64_t{257},
                               int64_t{81920},
                               int64_t{655360},
                               int64_t{5000000}})
    {
        for(const int64_t experts : {int64_t{1}, int64_t{64}, int64_t{512}, int64_t{8192}})
        {
            for(const int units : {0, 104, 304})
            {
                const auto split = routingSplit(pairs, experts, units);
                SCOPED_TRACE(std::to_string(pairs) + " pairs, " + std::to_string(experts)
                             + " experts, " + std::to_string(units) + " CUs");
                EXPECT_EQ(split.span % TILE_THREADS, 0);
                EXPECT_GE(split.blocks, 1);
                // Every pair covered, and no block empty.
                EXPECT_GE(split.blocks * split.span, pairs);
                EXPECT_LT((split.blocks - 1) * split.span, pairs);
                EXPECT_LE(split.blocks, units > 0 ? units : FALLBACK_COMPUTE_UNITS);
                EXPECT_LE(split.blocks, std::max<int64_t>(1, PREFIX_ENTRY_BUDGET / experts));
            }
        }
    }
}

TEST(TestFlydslMoeRoutingEngine, WorkspaceHoldsACountPerExpertAndBlockInWholeGroups)
{
    EXPECT_EQ(workspaceBytesFor(512, 3), size_t{512} * 3 * 4);
    EXPECT_EQ(workspaceBytesFor(5, 3), size_t{16} * 4);
    EXPECT_EQ(workspaceBytesFor(1, 1), size_t{4} * 4);

    const GraphFixture fixture(buildFlydslMoeRoutingGraph());
    const auto bound = matchesGraph(FLYDSL_MOE_ROUTING, fixture.context());
    ASSERT_TRUE(bound.has_value());
    const auto bytes = dispatchHandler(FLYDSL_MOE_ROUTING)
                           .workspaceBytes(fixture.context(), *bound, makeMoeRoutingKernel());
    // 160 pairs: one tile, one block.
    EXPECT_EQ(bytes, workspaceBytesFor(512, 1));
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
