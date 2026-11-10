// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslGdnTestGraphs.hpp"

/**
 * @file TestFlydslGdnEngine.cpp
 * @brief hipkernel:flydsl_gdn's matchers and scorer, on graphs built in-process.
 *
 * Host-only: no device, no shard. What the pack ships is censused in TestFlydslGdnPacks.cpp,
 * and every staged object is launched against the CPU reference in TestFlydslGdnObjects.cpp.
 *
 * The graph match accepts a single GDN node in bf16 or f16 with an f32 state, any gate
 * form, beta form, L2 norm, scale, state_indices, head replication and strides the kernel
 * addresses, and declines checkpoints, other dtypes, rule-breaking attribute combinations
 * and layouts the kernel would not compute exactly. The kernel match pairs a graph with
 * objects of its dtype, K and V, and admits a vector-state object only for states with a
 * unit-stride K, whole-vector strides and 16-byte aligned bases.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using data_objects::DataType;
using data_objects::GateDomain;

std::optional<BoundTokens> match(const GdnGraphSpec& spec)
{
    const GraphFixture fixture(buildFlydslGdnGraph(spec));
    return matchesGraph(FLYDSL_GDN, fixture.context());
}

/// A kernel carrying the metadata the GDN KMD declares. Source is left default-constructed:
/// the cases here read metadata only.
KernelDefinition makeGdnKernel(const std::string& dtype,
                               int64_t keyDim,
                               int64_t valueDim,
                               int64_t stateStrided,
                               int64_t priority = 100)
{
    KernelDefinition kernel;
    kernel.name = "gdn_" + dtype + "_d" + std::to_string(keyDim) + "_vd" + std::to_string(valueDim)
                  + "_sstr" + std::to_string(stateStrided);
    kernel.priority = priority;
    kernel.metadata = {{std::string(FLYDSL_GDN_DTYPE_FIELD), dtype},
                       {std::string(FLYDSL_GDN_HEAD_DIM_FIELD), keyDim},
                       {std::string(FLYDSL_GDN_HEAD_DIM_V_FIELD), valueDim},
                       {std::string(FLYDSL_GDN_STATE_STRIDED_FIELD), stateStrided},
                       {std::string(FLYDSL_GDN_DV_SPLIT_FIELD), valueDim / 32},
                       {std::string(FLYDSL_GDN_BLOCK_THREADS_FIELD), int64_t{256}}};
    return kernel;
}

bool kernelMatches(const GdnGraphSpec& spec, const KernelDefinition& kernel)
{
    const GraphFixture fixture(buildFlydslGdnGraph(spec));
    const auto bound = matchesGraph(FLYDSL_GDN, fixture.context());
    return bound.has_value() && matchesKernel(FLYDSL_GDN, fixture.context(), kernel, *bound);
}

// ---------------------------------------------------------------------------
// graph match: accepted
// ---------------------------------------------------------------------------

TEST(TestFlydslGdnEngine, AcceptsQwenDecodeAndBindsItsExtents)
{
    const auto bound = match(GdnGraphSpec{});
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_GDN.inputAToken), FLYDSL_GDN_Q_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_GDN.inputBToken), FLYDSL_GDN_K_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_GDN.outputToken), FLYDSL_GDN_O_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_GDN_V_TOKEN), FLYDSL_GDN_V_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_GDN_INITIAL_STATE_TOKEN), FLYDSL_GDN_INITIAL_STATE_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_GDN_FINAL_STATE_TOKEN), FLYDSL_GDN_FINAL_STATE_UID);
    EXPECT_FALSE(tryGetBoundInt(*bound, FLYDSL_GDN_STATE_INDICES_TOKEN).has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_GDN_TOKENS_TOKEN), 2);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_GDN_SEQUENCES_TOKEN), 2);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_GDN_HEADS_OUT_TOKEN), 48);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_GDN_KEY_DIM_TOKEN), 128);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_GDN_VALUE_DIM_TOKEN), 128);
}

TEST(TestFlydslGdnEngine, AcceptsPrefillAndEveryOptionalOperandsAbsence)
{
    GdnGraphSpec prefill;
    prefill.tokens = 300;
    prefill.sequences = 3;
    EXPECT_TRUE(match(prefill).has_value());

    GdnGraphSpec bare;
    bare.initialState = false;
    bare.finalState = false;
    bare.aLog = false;
    bare.dtBias = false;
    EXPECT_TRUE(match(bare).has_value());
}

TEST(TestFlydslGdnEngine, AcceptsEveryGateAndBetaForm)
{
    GdnGraphSpec logGate;
    logGate.safeGate = false;
    logGate.aLog = false;
    logGate.dtBias = false;
    EXPECT_TRUE(match(logGate).has_value());

    GdnGraphSpec linearGate = logGate;
    linearGate.gateDomain = GateDomain::LINEAR;
    EXPECT_TRUE(match(linearGate).has_value());

    GdnGraphSpec rawBeta;
    rawBeta.useBetaSigmoid = false;
    rawBeta.useQkL2norm = false;
    rawBeta.scale = 0.25F;
    EXPECT_TRUE(match(rawBeta).has_value());

    GdnGraphSpec negEigval;
    negEigval.allowNegEigval = true;
    EXPECT_TRUE(match(negEigval).has_value());
}

TEST(TestFlydslGdnEngine, AcceptsEveryGateOperandDtype)
{
    for(const auto type : {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16})
    {
        GdnGraphSpec spec;
        spec.gType = type;
        spec.betaType = type;
        spec.paramType = type;
        EXPECT_TRUE(match(spec).has_value()) << data_objects::EnumNameDataType(type);
    }
    GdnGraphSpec f16;
    f16.ioType = DataType::HALF;
    EXPECT_TRUE(match(f16).has_value());
}

TEST(TestFlydslGdnEngine, AcceptsHeadReplicationFormsAndAStatePool)
{
    GdnGraphSpec kPerValueHead;
    kPerValueHead.headsK = 48;
    EXPECT_TRUE(match(kPerValueHead).has_value());

    GdnGraphSpec moreQueryHeads;
    moreQueryHeads.headsQ = 8;
    moreQueryHeads.headsK = 8;
    moreQueryHeads.headsV = 4;
    EXPECT_TRUE(match(moreQueryHeads).has_value());

    GdnGraphSpec pool;
    pool.stateIndices = true;
    pool.poolRows = 9;
    pool.overwriteInitialState = true;
    pool.batchInvariant = true;
    const auto bound = match(pool);
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_GDN_STATE_INDICES_TOKEN), FLYDSL_GDN_STATE_INDICES_UID);
}

TEST(TestFlydslGdnEngine, AcceptsStridedTokenRowsAndAKMajorState)
{
    // q and k as slices of a packed [T, 2 * 16 * 128 + 48 * 128] projection row.
    GdnGraphSpec sliced;
    sliced.strides[FLYDSL_GDN_Q_UID] = {10240, 128, 1};
    sliced.strides[FLYDSL_GDN_K_UID] = {10240, 128, 1};
    sliced.strides[FLYDSL_GDN_V_UID] = {10240, 128, 1};
    EXPECT_TRUE(match(sliced).has_value());

    GdnGraphSpec kMajor;
    kMajor.strides[FLYDSL_GDN_INITIAL_STATE_UID] = flydslGdnKMajorStateStrides(48, 128, 128);
    kMajor.strides[FLYDSL_GDN_FINAL_STATE_UID] = flydslGdnKMajorStateStrides(48, 128, 128);
    EXPECT_TRUE(match(kMajor).has_value());
}

// ---------------------------------------------------------------------------
// graph match: declined
// ---------------------------------------------------------------------------

TEST(TestFlydslGdnEngine, DeclinesAGraphThatIsNotOneGdnNode)
{
    GdnGraphSpec twoNodes;
    twoNodes.secondNode = true;
    EXPECT_FALSE(match(twoNodes).has_value());

    GdnGraphSpec overrideShape;
    overrideShape.overrideShapeEnabled = true;
    EXPECT_FALSE(match(overrideShape).has_value());

    const GraphFixture rmsNorm(buildFlydslRmsNormGraph());
    EXPECT_FALSE(matchesGraph(FLYDSL_GDN, rmsNorm.context()).has_value());
}

TEST(TestFlydslGdnEngine, DeclinesCheckpoints)
{
    GdnGraphSpec spec;
    spec.checkpoints = true;
    spec.checkpointEveryNTokens = 1;
    EXPECT_FALSE(match(spec).has_value());

    GdnGraphSpec countOnly;
    countOnly.checkpointEveryNTokens = 4;
    EXPECT_FALSE(match(countOnly).has_value());
}

TEST(TestFlydslGdnEngine, DeclinesAttributeCombinationsTheSchemaForbids)
{
    GdnGraphSpec linearSafe;
    linearSafe.gateDomain = GateDomain::LINEAR;
    EXPECT_FALSE(match(linearSafe).has_value());

    GdnGraphSpec aLogWithoutSafe;
    aLogWithoutSafe.safeGate = false;
    EXPECT_FALSE(match(aLogWithoutSafe).has_value());

    GdnGraphSpec negEigvalWithoutSigmoid;
    negEigvalWithoutSigmoid.useBetaSigmoid = false;
    negEigvalWithoutSigmoid.allowNegEigval = true;
    EXPECT_FALSE(match(negEigvalWithoutSigmoid).has_value());

    GdnGraphSpec finalWithoutFlag;
    finalWithoutFlag.outputFinalState = false;
    EXPECT_FALSE(match(finalWithoutFlag).has_value());

    GdnGraphSpec flagWithoutFinal;
    flagWithoutFinal.finalState = false;
    flagWithoutFinal.outputFinalState = true;
    EXPECT_FALSE(match(flagWithoutFinal).has_value());

    GdnGraphSpec indicesWithoutInitial;
    indicesWithoutInitial.stateIndices = true;
    indicesWithoutInitial.initialState = false;
    EXPECT_FALSE(match(indicesWithoutInitial).has_value());

    GdnGraphSpec overwriteWithoutFinal;
    overwriteWithoutFinal.finalState = false;
    overwriteWithoutFinal.overwriteInitialState = true;
    EXPECT_FALSE(match(overwriteWithoutFinal).has_value());

    GdnGraphSpec infiniteScale;
    infiniteScale.scale = std::numeric_limits<float>::infinity();
    EXPECT_FALSE(match(infiniteScale).has_value());
}

TEST(TestFlydslGdnEngine, DeclinesDtypesNoObjectIsBuiltFor)
{
    GdnGraphSpec f32Io;
    f32Io.ioType = DataType::FLOAT;
    EXPECT_FALSE(match(f32Io).has_value());

    GdnGraphSpec mixedO;
    mixedO.oType = DataType::FLOAT;
    EXPECT_FALSE(match(mixedO).has_value());

    GdnGraphSpec bf16State;
    bf16State.stateType = DataType::BFLOAT16;
    EXPECT_FALSE(match(bf16State).has_value());

    GdnGraphSpec doubleGate;
    doubleGate.gType = DataType::DOUBLE;
    EXPECT_FALSE(match(doubleGate).has_value());

    GdnGraphSpec int64Offsets;
    int64Offsets.cuSeqlensType = DataType::INT64;
    EXPECT_FALSE(match(int64Offsets).has_value());

    GdnGraphSpec int64Indices;
    int64Indices.stateIndices = true;
    int64Indices.indexType = DataType::INT64;
    EXPECT_FALSE(match(int64Indices).has_value());

    GdnGraphSpec doubleCompute;
    doubleCompute.computeType = DataType::DOUBLE;
    EXPECT_FALSE(match(doubleCompute).has_value());
}

TEST(TestFlydslGdnEngine, DeclinesShapesOtherThanTheOps)
{
    GdnGraphSpec badKeyHeads;
    badKeyHeads.headsK = 8;
    EXPECT_FALSE(match(badKeyHeads).has_value());

    GdnGraphSpec indivisible;
    indivisible.headsQ = 16;
    indivisible.headsK = 16;
    indivisible.headsV = 40;
    EXPECT_FALSE(match(indivisible).has_value());

    GdnGraphSpec wrongGate;
    wrongGate.dims[FLYDSL_GDN_G_UID] = {2, 16};
    EXPECT_FALSE(match(wrongGate).has_value());

    GdnGraphSpec wrongState;
    wrongState.dims[FLYDSL_GDN_INITIAL_STATE_UID] = {2, 48, 128, 64};
    EXPECT_FALSE(match(wrongState).has_value());

    GdnGraphSpec poolWithoutIndices;
    poolWithoutIndices.poolRows = 5;
    EXPECT_FALSE(match(poolWithoutIndices).has_value());

    GdnGraphSpec noSequence;
    noSequence.dims[FLYDSL_GDN_CU_SEQLENS_UID] = {1};
    EXPECT_FALSE(match(noSequence).has_value());
}

TEST(TestFlydslGdnEngine, DeclinesLayoutsTheKernelDoesNotAddress)
{
    // q's K must be unit-stride, its rows whole 4-element vectors from an 8-byte base.
    GdnGraphSpec kStrided;
    kStrided.strides[FLYDSL_GDN_Q_UID] = {int64_t{16} * 256, 256, 2};
    EXPECT_FALSE(match(kStrided).has_value());

    GdnGraphSpec oddRows;
    oddRows.strides[FLYDSL_GDN_K_UID] = {int64_t{16} * 130, 130, 1};
    EXPECT_FALSE(match(oddRows).has_value());

    GdnGraphSpec underAligned;
    underAligned.alignment[FLYDSL_GDN_Q_UID] = 4;
    EXPECT_FALSE(match(underAligned).has_value());

    // v and o along a unit-stride V; o must not overlap itself.
    GdnGraphSpec vStrided;
    vStrided.strides[FLYDSL_GDN_V_UID] = {int64_t{48} * 256, 256, 2};
    EXPECT_FALSE(match(vStrided).has_value());

    GdnGraphSpec overlappingO;
    overlappingO.strides[FLYDSL_GDN_O_UID] = {int64_t{48} * 128, 0, 1};
    EXPECT_FALSE(match(overlappingO).has_value());

    GdnGraphSpec overlappingFinal;
    overlappingFinal.strides[FLYDSL_GDN_FINAL_STATE_UID]
        = {int64_t{48} * 128 * 128, int64_t{128} * 128, 1, 1};
    EXPECT_FALSE(match(overlappingFinal).has_value());

    for(const auto uid : {FLYDSL_GDN_Q_UID, FLYDSL_GDN_O_UID, FLYDSL_GDN_INITIAL_STATE_UID})
    {
        GdnGraphSpec spec;
        spec.virtualTensors = {uid};
        EXPECT_FALSE(match(spec).has_value()) << "virtual uid " << uid;
    }
}

TEST(TestFlydslGdnEngine, DeclinesOperandsPastTheKernelsOffsets)
{
    // 2^20 tokens of 16 x 128 bf16 q is 4 GiB: past the 32-bit buffer range.
    GdnGraphSpec longPrefill;
    longPrefill.tokens = int64_t{1} << 20;
    EXPECT_FALSE(match(longPrefill).has_value());

    // More output heads than the grid's y extent.
    GdnGraphSpec manyHeads;
    manyHeads.headsQ = 65536;
    manyHeads.headsK = 65536;
    manyHeads.headsV = 65536;
    manyHeads.keyDim = 32;
    manyHeads.valueDim = 32;
    manyHeads.initialState = false;
    manyHeads.finalState = false;
    manyHeads.aLog = false;
    manyHeads.dtBias = false;
    manyHeads.tokens = 1;
    manyHeads.sequences = 1;
    EXPECT_FALSE(match(manyHeads).has_value());
}

// ---------------------------------------------------------------------------
// kernel match and score
// ---------------------------------------------------------------------------

TEST(TestFlydslGdnEngine, PairsAGraphWithObjectsOfItsDtypeAndHeadDims)
{
    const GdnGraphSpec spec;
    EXPECT_TRUE(kernelMatches(spec, makeGdnKernel("bf16", 128, 128, 0)));
    EXPECT_TRUE(kernelMatches(spec, makeGdnKernel("bf16", 128, 128, 1, 10)));
    EXPECT_FALSE(kernelMatches(spec, makeGdnKernel("f16", 128, 128, 0)));
    EXPECT_FALSE(kernelMatches(spec, makeGdnKernel("bf16", 64, 128, 0)));
    EXPECT_FALSE(kernelMatches(spec, makeGdnKernel("bf16", 128, 256, 0)));

    GdnGraphSpec rectangular;
    rectangular.keyDim = 64;
    rectangular.valueDim = 256;
    EXPECT_TRUE(kernelMatches(rectangular, makeGdnKernel("bf16", 64, 256, 0)));
}

TEST(TestFlydslGdnEngine, AdmitsTheVectorStateObjectOnlyForVectorStates)
{
    const auto vector = makeGdnKernel("bf16", 128, 128, 0);
    const auto strided = makeGdnKernel("bf16", 128, 128, 1, 10);

    for(const auto uid : {FLYDSL_GDN_INITIAL_STATE_UID, FLYDSL_GDN_FINAL_STATE_UID})
    {
        GdnGraphSpec kMajor;
        kMajor.strides[uid] = flydslGdnKMajorStateStrides(48, 128, 128);
        EXPECT_FALSE(kernelMatches(kMajor, vector)) << "uid " << uid;
        EXPECT_TRUE(kernelMatches(kMajor, strided)) << "uid " << uid;

        GdnGraphSpec underAligned;
        underAligned.alignment[uid] = 8;
        EXPECT_FALSE(kernelMatches(underAligned, vector)) << "uid " << uid;
        EXPECT_TRUE(kernelMatches(underAligned, strided)) << "uid " << uid;

        // A padded V row of 130 floats is not a whole number of vectors.
        GdnGraphSpec padded;
        padded.strides[uid] = {int64_t{2} * 48 * 128 * 130, int64_t{128} * 130, 130, 1};
        EXPECT_FALSE(kernelMatches(padded, vector)) << "uid " << uid;
        EXPECT_TRUE(kernelMatches(padded, strided)) << "uid " << uid;
    }

    // No state at all: either object.
    GdnGraphSpec stateless;
    stateless.initialState = false;
    stateless.finalState = false;
    EXPECT_TRUE(kernelMatches(stateless, vector));
    EXPECT_TRUE(kernelMatches(stateless, strided));
}

TEST(TestFlydslGdnEngine, RanksTheVectorStateObjectAboveTheStridedOne)
{
    const GraphFixture fixture(buildFlydslGdnGraph());
    EXPECT_GT(scoreKernel(FLYDSL_GDN, fixture.context(), makeGdnKernel("bf16", 128, 128, 0)),
              scoreKernel(FLYDSL_GDN, fixture.context(), makeGdnKernel("bf16", 128, 128, 1, 10)));
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
