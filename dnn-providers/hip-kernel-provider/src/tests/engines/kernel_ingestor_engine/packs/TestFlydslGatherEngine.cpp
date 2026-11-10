// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslGatherTestGraphs.hpp"

/**
 * @file TestFlydslGatherEngine.cpp
 * @brief hipkernel:flydsl_gather's matchers and scorer, on graphs built in-process.
 *
 * Host-only: no device, no shard. What the pack ships is censused in
 * TestFlydslGatherPacks.cpp, and every staged object is launched against the CPU
 * reference in TestFlydslGatherObjects.cpp.
 *
 * The graph match accepts a single gather node over packed operands -- any axis, any
 * index rank -- in bf16, f16 or f32 with INT32 or INT64 indices, and declines everything
 * else the kernel would not compute exactly. The kernel match pairs a graph with objects
 * of its dtype and index dtype, and admits a 16-byte vector object only for a row of a
 * multiple of 16 bytes with 16-byte aligned x and y.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using data_objects::DataType;

std::optional<BoundTokens> match(const GatherGraphSpec& spec)
{
    const GraphFixture fixture(buildFlydslGatherGraph(spec));
    return matchesGraph(FLYDSL_GATHER, fixture.context());
}

/// A kernel carrying the metadata the gather KMD declares. Source is left
/// default-constructed: the cases here read metadata only.
KernelDefinition makeGatherKernel(const std::string& dtype,
                                  const std::string& indexDtype,
                                  int64_t vecBytes,
                                  int64_t priority = 100)
{
    KernelDefinition kernel;
    kernel.name = "gather_" + dtype + "_idx" + indexDtype + "_vec" + std::to_string(vecBytes);
    kernel.priority = priority;
    kernel.metadata = {{std::string(FLYDSL_GATHER_DTYPE_FIELD), dtype},
                       {std::string(FLYDSL_GATHER_INDEX_DTYPE_FIELD), indexDtype},
                       {std::string(FLYDSL_GATHER_VEC_BYTES_FIELD), vecBytes},
                       {std::string(FLYDSL_GATHER_BLOCK_THREADS_FIELD), int64_t{256}}};
    return kernel;
}

bool kernelMatches(const GatherGraphSpec& spec, const KernelDefinition& kernel)
{
    const GraphFixture fixture(buildFlydslGatherGraph(spec));
    const auto bound = matchesGraph(FLYDSL_GATHER, fixture.context());
    return bound.has_value() && matchesKernel(FLYDSL_GATHER, fixture.context(), kernel, *bound);
}

// ---------------------------------------------------------------------------
// graph match: accepted
// ---------------------------------------------------------------------------

TEST(TestFlydslGatherEngine, AcceptsTheEmbeddingLookupAndBindsItsExtents)
{
    GatherGraphSpec spec;
    spec.xDims = {248320, 5120};
    spec.indexDims = {1, 7};
    const auto bound = match(spec);
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_GATHER.inputAToken), FLYDSL_GATHER_X_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_GATHER.inputBToken), FLYDSL_GATHER_INDICES_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_GATHER.outputToken), FLYDSL_GATHER_Y_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_GATHER_NUM_INDICES_TOKEN), 7);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_GATHER_AXIS_EXTENT_TOKEN), 248320);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_GATHER_INNER_TOKEN), 5120);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_GATHER_OUTER_TOKEN), 1);
}

TEST(TestFlydslGatherEngine, AcceptsEveryDtypeAndIndexDtype)
{
    for(const auto dataType : {DataType::BFLOAT16, DataType::HALF, DataType::FLOAT})
    {
        for(const auto indexType : {DataType::INT32, DataType::INT64})
        {
            GatherGraphSpec spec;
            spec.dataType = dataType;
            spec.indexDataType = indexType;
            EXPECT_TRUE(match(spec).has_value())
                << data_objects::EnumNameDataType(dataType) << " / "
                << data_objects::EnumNameDataType(indexType);
        }
    }
}

TEST(TestFlydslGatherEngine, AcceptsAnyAxisAndIndexRank)
{
    // x [3, 10, 4, 6] gathered along axis 1 by indices [2, 5]: outer 3, inner 24.
    GatherGraphSpec spec;
    spec.xDims = {3, 10, 4, 6};
    spec.indexDims = {2, 5};
    spec.axis = 1;
    const auto bound = match(spec);
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_GATHER_NUM_INDICES_TOKEN), 10);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_GATHER_AXIS_EXTENT_TOKEN), 10);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_GATHER_INNER_TOKEN), 24);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_GATHER_OUTER_TOKEN), 3);

    // The last axis: one element per row.
    spec.axis = 3;
    const auto last = match(spec);
    ASSERT_TRUE(last.has_value());
    EXPECT_EQ(tryGetBoundInt(*last, FLYDSL_GATHER_INNER_TOKEN), 1);
    EXPECT_EQ(tryGetBoundInt(*last, FLYDSL_GATHER_OUTER_TOKEN), 120);

    // A rank-1 x and a rank-1 index: take().
    GatherGraphSpec take;
    take.xDims = {100};
    take.indexDims = {9};
    EXPECT_TRUE(match(take).has_value());
}

TEST(TestFlydslGatherEngine, AcceptsUnitAxesWithAnyStride)
{
    // An extent-1 axis is never stepped along, so its stride is not checked.
    GatherGraphSpec spec;
    spec.indexDims = {1, 8};
    spec.indexStrides = std::vector<int64_t>{0, 1};
    EXPECT_TRUE(match(spec).has_value());
}

// ---------------------------------------------------------------------------
// graph match: declined
// ---------------------------------------------------------------------------

TEST(TestFlydslGatherEngine, DeclinesAGraphThatIsNotOneGatherNode)
{
    GatherGraphSpec twoNodes;
    twoNodes.secondNode = true;
    EXPECT_FALSE(match(twoNodes).has_value());

    GatherGraphSpec overrideShape;
    overrideShape.overrideShapeEnabled = true;
    EXPECT_FALSE(match(overrideShape).has_value());

    const GraphFixture rmsNorm(buildFlydslRmsNormGraph());
    EXPECT_FALSE(matchesGraph(FLYDSL_GATHER, rmsNorm.context()).has_value());
}

TEST(TestFlydslGatherEngine, DeclinesDtypesNoObjectIsBuiltFor)
{
    GatherGraphSpec int8Table;
    int8Table.dataType = DataType::INT8;
    EXPECT_FALSE(match(int8Table).has_value());

    GatherGraphSpec doubleTable;
    doubleTable.dataType = DataType::DOUBLE;
    EXPECT_FALSE(match(doubleTable).has_value());

    GatherGraphSpec floatIndices;
    floatIndices.indexDataType = DataType::FLOAT;
    EXPECT_FALSE(match(floatIndices).has_value());

    GatherGraphSpec int8Indices;
    int8Indices.indexDataType = DataType::INT8;
    EXPECT_FALSE(match(int8Indices).has_value());

    // Values are copied as raw bits: y must be in x's dtype.
    GatherGraphSpec convertingY;
    convertingY.yDataType = DataType::FLOAT;
    EXPECT_FALSE(match(convertingY).has_value());
}

TEST(TestFlydslGatherEngine, DeclinesAnAxisOutsideXsRank)
{
    GatherGraphSpec negative;
    negative.axis = -1;
    negative.yDims = std::vector<int64_t>{1, 8, 256};
    EXPECT_FALSE(match(negative).has_value());

    GatherGraphSpec pastRank;
    pastRank.axis = 2;
    pastRank.yDims = std::vector<int64_t>{1, 8, 256};
    EXPECT_FALSE(match(pastRank).has_value());
}

TEST(TestFlydslGatherEngine, DeclinesAYShapeOtherThanTheOps)
{
    GatherGraphSpec wrongRow;
    wrongRow.yDims = std::vector<int64_t>{1, 8, 128};
    EXPECT_FALSE(match(wrongRow).has_value());

    GatherGraphSpec flattenedIndices;
    flattenedIndices.yDims = std::vector<int64_t>{8, 256};
    EXPECT_FALSE(match(flattenedIndices).has_value());
}

TEST(TestFlydslGatherEngine, DeclinesOperandsThatAreNotPacked)
{
    GatherGraphSpec paddedTable;
    paddedTable.xStrides = std::vector<int64_t>{512, 1};
    EXPECT_FALSE(match(paddedTable).has_value());

    GatherGraphSpec stridedIndices;
    stridedIndices.indexDims = {2, 4};
    stridedIndices.indexStrides = std::vector<int64_t>{8, 2};
    EXPECT_FALSE(match(stridedIndices).has_value());

    GatherGraphSpec transposedY;
    transposedY.yStrides = std::vector<int64_t>{2048, 1, 8};
    EXPECT_FALSE(match(transposedY).has_value());

    GatherGraphSpec emptyIndices;
    emptyIndices.indexDims = {1, 0};
    EXPECT_FALSE(match(emptyIndices).has_value());
}

TEST(TestFlydslGatherEngine, DeclinesVirtualOperandsAndUnderAlignedBases)
{
    for(const auto uid : {FLYDSL_GATHER_X_UID, FLYDSL_GATHER_INDICES_UID, FLYDSL_GATHER_Y_UID})
    {
        GatherGraphSpec spec;
        spec.virtualTensors = {uid};
        EXPECT_FALSE(match(spec).has_value()) << "virtual uid " << uid;
    }

    // An element's alignment is the floor for every operand.
    GatherGraphSpec oddTable;
    oddTable.alignment = {{FLYDSL_GATHER_X_UID, 1}};
    EXPECT_FALSE(match(oddTable).has_value());

    GatherGraphSpec i64OnFour;
    i64OnFour.indexDataType = DataType::INT64;
    i64OnFour.alignment = {{FLYDSL_GATHER_INDICES_UID, 4}};
    EXPECT_FALSE(match(i64OnFour).has_value());

    GatherGraphSpec i32OnFour;
    i32OnFour.alignment = {{FLYDSL_GATHER_INDICES_UID, 4}};
    EXPECT_TRUE(match(i32OnFour).has_value());
}

TEST(TestFlydslGatherEngine, DeclinesExtentsPastTheKernelsArgumentsAndGrid)
{
    // One f32 row of 2^29 elements is 2 GiB: past the 32-bit buffer range.
    GatherGraphSpec wideRow;
    wideRow.dataType = DataType::FLOAT;
    wideRow.xDims = {2, int64_t{1} << 29};
    wideRow.indexDims = {1};
    EXPECT_FALSE(match(wideRow).has_value());

    // A table past 2^31 rows is past the kernel's int32 axis extent.
    GatherGraphSpec tallTable;
    tallTable.xDims = {int64_t{1} << 31, 2};
    tallTable.indexDims = {1};
    EXPECT_FALSE(match(tallTable).has_value());

    // An outer extent past the grid's z limit.
    GatherGraphSpec manyOuter;
    manyOuter.xDims = {65536, 4, 2};
    manyOuter.indexDims = {1};
    manyOuter.axis = 1;
    EXPECT_FALSE(match(manyOuter).has_value());
    manyOuter.xDims = {65535, 4, 2};
    EXPECT_TRUE(match(manyOuter).has_value());

    // INT64 indices past the 32-bit range of their descriptor.
    GatherGraphSpec manyIndices;
    manyIndices.indexDataType = DataType::INT64;
    manyIndices.indexDims = {int64_t{1} << 28};
    EXPECT_FALSE(match(manyIndices).has_value());
}

// ---------------------------------------------------------------------------
// kernel match and score
// ---------------------------------------------------------------------------

TEST(TestFlydslGatherEngine, PairsAGraphWithObjectsOfItsDtypeAndIndexDtype)
{
    GatherGraphSpec spec;
    EXPECT_TRUE(kernelMatches(spec, makeGatherKernel("bf16", "i32", 2)));
    EXPECT_FALSE(kernelMatches(spec, makeGatherKernel("f16", "i32", 2)));
    EXPECT_FALSE(kernelMatches(spec, makeGatherKernel("bf16", "i64", 2)));

    spec.dataType = DataType::FLOAT;
    spec.indexDataType = DataType::INT64;
    EXPECT_TRUE(kernelMatches(spec, makeGatherKernel("f32", "i64", 4)));
    EXPECT_FALSE(kernelMatches(spec, makeGatherKernel("f32", "i32", 4)));
    // An access width that is neither the element nor the vector names no object.
    EXPECT_FALSE(kernelMatches(spec, makeGatherKernel("f32", "i64", 8)));
}

TEST(TestFlydslGatherEngine, AdmitsTheVectorObjectOnlyForWholeAlignedVectors)
{
    const auto vector = makeGatherKernel("bf16", "i32", 16);
    const auto element = makeGatherKernel("bf16", "i32", 2, 10);

    GatherGraphSpec spec;
    EXPECT_TRUE(kernelMatches(spec, vector));
    EXPECT_TRUE(kernelMatches(spec, element));

    // 1001 bf16 elements are not a whole number of 16-byte vectors.
    spec.xDims = {64, 1001};
    EXPECT_FALSE(kernelMatches(spec, vector));
    EXPECT_TRUE(kernelMatches(spec, element));

    // Four f32 elements are one vector; an f32 row of 6 is not.
    GatherGraphSpec f32Rows;
    f32Rows.dataType = DataType::FLOAT;
    f32Rows.xDims = {64, 4};
    EXPECT_TRUE(kernelMatches(f32Rows, makeGatherKernel("f32", "i32", 16)));
    f32Rows.xDims = {64, 6};
    EXPECT_FALSE(kernelMatches(f32Rows, makeGatherKernel("f32", "i32", 16)));
    EXPECT_TRUE(kernelMatches(f32Rows, makeGatherKernel("f32", "i32", 4)));

    // A base aligned to less than a vector: the per-element object only.
    for(const auto uid : {FLYDSL_GATHER_X_UID, FLYDSL_GATHER_Y_UID})
    {
        GatherGraphSpec underAligned;
        underAligned.alignment = {{uid, 8}};
        EXPECT_FALSE(kernelMatches(underAligned, vector)) << "uid " << uid;
        EXPECT_TRUE(kernelMatches(underAligned, element)) << "uid " << uid;
    }
}

TEST(TestFlydslGatherEngine, DeclinesAnObjectWhoseRowChunksPassTheGrid)
{
    // 2^25 bf16 elements per row: 131072 chunks of 256 lanes for the per-element object,
    // past the grid's y limit; 16384 of 256 x 8 for the vector one.
    GatherGraphSpec spec;
    spec.xDims = {2, int64_t{1} << 25};
    spec.indexDims = {1};
    EXPECT_FALSE(kernelMatches(spec, makeGatherKernel("bf16", "i32", 2)));
    EXPECT_TRUE(kernelMatches(spec, makeGatherKernel("bf16", "i32", 16)));
}

TEST(TestFlydslGatherEngine, RanksTheVectorObjectAboveThePerElementOne)
{
    const GraphFixture fixture(buildFlydslGatherGraph());
    EXPECT_GT(
        scoreKernel(FLYDSL_GATHER, fixture.context(), makeGatherKernel("bf16", "i32", 16)),
        scoreKernel(FLYDSL_GATHER, fixture.context(), makeGatherKernel("bf16", "i32", 2, 10)));
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
