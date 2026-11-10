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

#include "engines/kernel_ingestor_engine/packs/FlydslTopkTestGraphs.hpp"

/**
 * @file TestFlydslTopkEngine.cpp
 * @brief hipkernel:flydsl_topk's matchers and scorer, on graphs built in-process.
 *
 * Host-only: no device, no shard. What the pack ships is censused in TestFlydslTopkPacks.cpp,
 * and every staged object is launched against the CPU reference in TestFlydslTopkObjects.cpp.
 *
 * The graph match accepts a single top-k node over packed operands of any rank in bf16,
 * f16 or f32 with INT32 or INT64 indices, with or without values, largest or smallest, any
 * k in [1, n]; it declines everything else the kernel would not compute exactly. The kernel
 * match pairs a graph with objects of its dtype and index dtype whose `k_max` covers k.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using data_objects::DataType;

std::optional<BoundTokens> match(const TopkGraphSpec& spec)
{
    const GraphFixture fixture(buildFlydslTopkGraph(spec));
    return matchesGraph(FLYDSL_TOPK, fixture.context());
}

/// A kernel carrying the metadata the top-k KMD declares. Source is left default-constructed:
/// the cases here read metadata only.
KernelDefinition makeTopkKernel(const std::string& dtype,
                                const std::string& indexDtype,
                                int64_t kMax,
                                int64_t priority = 10)
{
    KernelDefinition kernel;
    kernel.name = "topk_" + dtype + "_idx" + indexDtype + "_kmax" + std::to_string(kMax);
    kernel.priority = priority;
    kernel.metadata = {{std::string(FLYDSL_TOPK_DTYPE_FIELD), dtype},
                       {std::string(FLYDSL_TOPK_INDEX_DTYPE_FIELD), indexDtype},
                       {std::string(FLYDSL_TOPK_K_MAX_FIELD), kMax},
                       {std::string(FLYDSL_TOPK_BLOCK_THREADS_FIELD), int64_t{256}}};
    return kernel;
}

bool kernelMatches(const TopkGraphSpec& spec, const KernelDefinition& kernel)
{
    const GraphFixture fixture(buildFlydslTopkGraph(spec));
    const auto bound = matchesGraph(FLYDSL_TOPK, fixture.context());
    return bound.has_value() && matchesKernel(FLYDSL_TOPK, fixture.context(), kernel, *bound);
}

// ---------------------------------------------------------------------------
// graph match: accepted
// ---------------------------------------------------------------------------

TEST(TestFlydslTopkEngine, AcceptsTheRouterAndBindsItsExtents)
{
    TopkGraphSpec spec;
    spec.xDims = {2, 7, 512};
    spec.k = 8;
    const auto bound = match(spec);
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_TOPK.inputAToken), FLYDSL_TOPK_X_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_TOPK.inputBToken), FLYDSL_TOPK_VALUES_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_TOPK.outputToken), FLYDSL_TOPK_INDICES_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_TOPK_ROWS_TOKEN), 14);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_TOPK_COLUMNS_TOKEN), 512);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_TOPK_K_TOKEN), 8);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_TOPK_LARGEST_TOKEN), 1);
}

TEST(TestFlydslTopkEngine, AcceptsGreedyArgmaxOverTheVocabularyWithoutValues)
{
    TopkGraphSpec spec;
    spec.xDims = {4, 248320};
    spec.k = 1;
    spec.withValues = false;
    spec.indexDataType = DataType::INT64;
    const auto bound = match(spec);
    ASSERT_TRUE(bound.has_value());
    EXPECT_FALSE(tryGetBoundInt(*bound, FLYDSL_TOPK.inputBToken).has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_TOPK_ROWS_TOKEN), 4);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_TOPK_COLUMNS_TOKEN), 248320);
}

TEST(TestFlydslTopkEngine, AcceptsEveryDtypeIndexDtypeAndDirection)
{
    for(const auto dataType : {DataType::BFLOAT16, DataType::HALF, DataType::FLOAT})
    {
        for(const auto indexType : {DataType::INT32, DataType::INT64})
        {
            for(const bool largest : {true, false})
            {
                TopkGraphSpec spec;
                spec.dataType = dataType;
                spec.indexDataType = indexType;
                spec.largest = largest;
                const auto bound = match(spec);
                ASSERT_TRUE(bound.has_value()) << data_objects::EnumNameDataType(dataType) << " / "
                                               << data_objects::EnumNameDataType(indexType);
                EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_TOPK_LARGEST_TOKEN), largest ? 1 : 0);
            }
        }
    }
}

TEST(TestFlydslTopkEngine, AcceptsEveryKFromOneToTheRowAndARankOneX)
{
    for(const int64_t k : {int64_t{1}, int64_t{2}, int64_t{511}, int64_t{512}})
    {
        TopkGraphSpec spec;
        spec.k = k;
        EXPECT_TRUE(match(spec).has_value()) << "k " << k;
    }
    TopkGraphSpec rankOne;
    rankOne.xDims = {100};
    rankOne.k = 3;
    const auto bound = match(rankOne);
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_TOPK_ROWS_TOKEN), 1);
}

TEST(TestFlydslTopkEngine, AcceptsExactComputeTypes)
{
    for(const auto computeType : {DataType::UNSET, DataType::FLOAT, DataType::DOUBLE})
    {
        TopkGraphSpec spec;
        spec.computeDataType = computeType;
        EXPECT_TRUE(match(spec).has_value()) << data_objects::EnumNameDataType(computeType);
    }
    TopkGraphSpec own;
    own.dataType = DataType::HALF;
    own.computeDataType = DataType::HALF;
    EXPECT_TRUE(match(own).has_value());
}

TEST(TestFlydslTopkEngine, AcceptsUnitAxesWithAnyStride)
{
    TopkGraphSpec spec;
    spec.xDims = {1, 512};
    spec.xStrides = std::vector<int64_t>{0, 1};
    EXPECT_TRUE(match(spec).has_value());
}

// ---------------------------------------------------------------------------
// graph match: declined
// ---------------------------------------------------------------------------

TEST(TestFlydslTopkEngine, DeclinesAGraphThatIsNotOneTopkNode)
{
    TopkGraphSpec twoNodes;
    twoNodes.secondNode = true;
    EXPECT_FALSE(match(twoNodes).has_value());

    TopkGraphSpec overrideShape;
    overrideShape.overrideShapeEnabled = true;
    EXPECT_FALSE(match(overrideShape).has_value());

    const GraphFixture rmsNorm(buildFlydslRmsNormGraph());
    EXPECT_FALSE(matchesGraph(FLYDSL_TOPK, rmsNorm.context()).has_value());
}

TEST(TestFlydslTopkEngine, DeclinesDtypesNoObjectIsBuiltFor)
{
    TopkGraphSpec doubleX;
    doubleX.dataType = DataType::DOUBLE;
    EXPECT_FALSE(match(doubleX).has_value());

    TopkGraphSpec int8X;
    int8X.dataType = DataType::INT8;
    EXPECT_FALSE(match(int8X).has_value());

    TopkGraphSpec floatIndices;
    floatIndices.indexDataType = DataType::FLOAT;
    EXPECT_FALSE(match(floatIndices).has_value());

    TopkGraphSpec int8Indices;
    int8Indices.indexDataType = DataType::INT8;
    EXPECT_FALSE(match(int8Indices).has_value());

    // Values are copied bit for bit: they are in x's dtype.
    TopkGraphSpec convertingValues;
    convertingValues.valuesDataType = DataType::FLOAT;
    EXPECT_FALSE(match(convertingValues).has_value());
}

TEST(TestFlydslTopkEngine, DeclinesAComputeTypeThatRoundsBeforeComparing)
{
    // bf16 does not widen into f16 (nor f32 into either): ties would change.
    TopkGraphSpec halfOnBf16;
    halfOnBf16.computeDataType = DataType::HALF;
    EXPECT_FALSE(match(halfOnBf16).has_value());

    TopkGraphSpec bf16OnF32;
    bf16OnF32.dataType = DataType::FLOAT;
    bf16OnF32.computeDataType = DataType::BFLOAT16;
    EXPECT_FALSE(match(bf16OnF32).has_value());
}

TEST(TestFlydslTopkEngine, DeclinesKOutsideOneToTheRow)
{
    for(const int64_t k : {int64_t{0}, int64_t{-1}, int64_t{513}})
    {
        TopkGraphSpec spec;
        spec.k = k;
        spec.indicesDims = std::vector<int64_t>{16, 8};
        spec.valuesDims = std::vector<int64_t>{16, 8};
        EXPECT_FALSE(match(spec).has_value()) << "k " << k;
    }
}

TEST(TestFlydslTopkEngine, DeclinesOutputShapesOtherThanTheOps)
{
    TopkGraphSpec wrongIndices;
    wrongIndices.indicesDims = std::vector<int64_t>{16, 4};
    EXPECT_FALSE(match(wrongIndices).has_value());

    TopkGraphSpec wrongValues;
    wrongValues.valuesDims = std::vector<int64_t>{8, 8};
    EXPECT_FALSE(match(wrongValues).has_value());

    TopkGraphSpec flattened;
    flattened.xDims = {2, 8, 512};
    flattened.indicesDims = std::vector<int64_t>{16, 8};
    EXPECT_FALSE(match(flattened).has_value());
}

TEST(TestFlydslTopkEngine, DeclinesOperandsThatAreNotPacked)
{
    TopkGraphSpec paddedX;
    paddedX.xStrides = std::vector<int64_t>{520, 1};
    EXPECT_FALSE(match(paddedX).has_value());

    TopkGraphSpec transposedX;
    transposedX.xStrides = std::vector<int64_t>{1, 16};
    EXPECT_FALSE(match(transposedX).has_value());

    TopkGraphSpec stridedIndices;
    stridedIndices.indicesStrides = std::vector<int64_t>{16, 2};
    EXPECT_FALSE(match(stridedIndices).has_value());

    TopkGraphSpec stridedValues;
    stridedValues.valuesStrides = std::vector<int64_t>{1, 16};
    EXPECT_FALSE(match(stridedValues).has_value());

    TopkGraphSpec emptyRows;
    emptyRows.xDims = {0, 512};
    EXPECT_FALSE(match(emptyRows).has_value());
}

TEST(TestFlydslTopkEngine, DeclinesVirtualOperandsAndUnderAlignedBases)
{
    for(const auto uid : {FLYDSL_TOPK_X_UID, FLYDSL_TOPK_INDICES_UID, FLYDSL_TOPK_VALUES_UID})
    {
        TopkGraphSpec spec;
        spec.virtualTensors = {uid};
        EXPECT_FALSE(match(spec).has_value()) << "virtual uid " << uid;
    }

    TopkGraphSpec oddX;
    oddX.alignment = {{FLYDSL_TOPK_X_UID, 1}};
    EXPECT_FALSE(match(oddX).has_value());

    TopkGraphSpec i64OnFour;
    i64OnFour.indexDataType = DataType::INT64;
    i64OnFour.alignment = {{FLYDSL_TOPK_INDICES_UID, 4}};
    EXPECT_FALSE(match(i64OnFour).has_value());

    // An element's alignment is enough: the 128-bit row loads are chosen at launch.
    TopkGraphSpec elementAligned;
    elementAligned.alignment = {{FLYDSL_TOPK_X_UID, 2}, {FLYDSL_TOPK_VALUES_UID, 2}};
    EXPECT_TRUE(match(elementAligned).has_value());
}

TEST(TestFlydslTopkEngine, DeclinesExtentsPastTheKernelsArguments)
{
    // One f32 row of 2^29 elements is 2 GiB: past the 32-bit buffer range.
    TopkGraphSpec wideRow;
    wideRow.dataType = DataType::FLOAT;
    wideRow.xDims = {2, int64_t{1} << 29};
    EXPECT_FALSE(match(wideRow).has_value());

    // More than 2^31 - 1 rows.
    TopkGraphSpec manyRows;
    manyRows.xDims = {int64_t{1} << 16, int64_t{1} << 16, 4};
    manyRows.k = 2;
    EXPECT_FALSE(match(manyRows).has_value());
}

// ---------------------------------------------------------------------------
// kernel match and score
// ---------------------------------------------------------------------------

TEST(TestFlydslTopkEngine, PairsAGraphWithObjectsOfItsDtypeAndIndexDtype)
{
    TopkGraphSpec spec;
    EXPECT_TRUE(kernelMatches(spec, makeTopkKernel("bf16", "i32", 1024)));
    EXPECT_FALSE(kernelMatches(spec, makeTopkKernel("f16", "i32", 1024)));
    EXPECT_FALSE(kernelMatches(spec, makeTopkKernel("bf16", "i64", 1024)));

    spec.dataType = DataType::FLOAT;
    spec.indexDataType = DataType::INT64;
    EXPECT_TRUE(kernelMatches(spec, makeTopkKernel("f32", "i64", 1024)));
    EXPECT_FALSE(kernelMatches(spec, makeTopkKernel("f32", "i32", 1024)));
}

TEST(TestFlydslTopkEngine, AdmitsAnObjectOnlyForKUpToItsBound)
{
    const auto reduction = makeTopkKernel("bf16", "i32", 1, 100);
    const auto general = makeTopkKernel("bf16", "i32", FLYDSL_TOPK_GENERAL_K_MAX);

    TopkGraphSpec argmax;
    argmax.k = 1;
    EXPECT_TRUE(kernelMatches(argmax, reduction));
    EXPECT_TRUE(kernelMatches(argmax, general));

    const TopkGraphSpec router{};
    EXPECT_FALSE(kernelMatches(router, reduction));
    EXPECT_TRUE(kernelMatches(router, general));

    TopkGraphSpec atBound;
    atBound.xDims = {2, 4096};
    atBound.k = FLYDSL_TOPK_GENERAL_K_MAX;
    EXPECT_TRUE(kernelMatches(atBound, general));
    atBound.k = FLYDSL_TOPK_GENERAL_K_MAX + 1;
    EXPECT_FALSE(kernelMatches(atBound, general));
}

TEST(TestFlydslTopkEngine, RanksTheReductionAboveTheRadixSelect)
{
    TopkGraphSpec argmax;
    argmax.k = 1;
    const GraphFixture fixture(buildFlydslTopkGraph(argmax));
    EXPECT_GT(scoreKernel(FLYDSL_TOPK, fixture.context(), makeTopkKernel("bf16", "i32", 1, 100)),
              scoreKernel(FLYDSL_TOPK,
                          fixture.context(),
                          makeTopkKernel("bf16", "i32", FLYDSL_TOPK_GENERAL_K_MAX)));
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
