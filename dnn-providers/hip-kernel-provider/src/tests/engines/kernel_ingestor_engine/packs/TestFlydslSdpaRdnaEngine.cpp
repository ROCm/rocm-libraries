// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose: a future edit that puts this file back on
// an ungated source list then yields an empty translation unit rather than a suite that
// reads a staged shard nothing produced.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>

#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaRdnaTestGraphs.hpp"

/**
 * @file TestFlydslSdpaRdnaEngine.cpp
 * @brief The hipkernel:flydsl_sdpa_rdna pack's seams: what its graph matcher accepts and
 *        declines, what it binds, which candidate a graph selects, and real launches of
 *        the staged objects checked against a host reference.
 *
 * The census over the staged shard lives in TestFlydslSdpaRdnaPacks.cpp, in the census
 * binary. Everything here builds its graph in-process.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hipdnn_flatbuffers_sdk::data_objects::DataType;

/// Runs the graph match over @p spec, which every binding case below needs to succeed.
BoundTokens bindingsFor(const SdpaGraphSpec& spec = {})
{
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    const auto bound = matchesGraph(FLYDSL_SDPA_RDNA, fixture.context());
    if(!bound.has_value())
    {
        throw std::runtime_error("flydsl_sdpa_rdna graph match declined a graph it must serve");
    }
    return *bound;
}

int64_t boundInt(const BoundTokens& bound, const char* token)
{
    const auto value = tryGetBoundInt(bound, token);
    if(!value.has_value())
    {
        throw std::runtime_error(std::string("no integer bound for ") + token);
    }
    return *value;
}

float boundScale(const BoundTokens& bound)
{
    const auto bits = static_cast<int32_t>(boundInt(bound, FLYDSL_SDPA_RDNA_SCALE_TOKEN));
    float scale = 0.0F;
    std::memcpy(&scale, &bits, sizeof(scale));
    return scale;
}

/// A kernel carrying exactly the metadata this pack's KMD declares.
KernelDefinition makeFlydslSdpaRdnaKernel(const std::string& dtype,
                                          int64_t headDim,
                                          int64_t causal,
                                          int64_t blockM = 128,
                                          int64_t blockN = 32,
                                          int64_t dvSplit = 1,
                                          int64_t hasBias = 0,
                                          std::optional<int64_t> headDimMax = std::nullopt,
                                          int64_t decode = 0,
                                          int64_t hdPair = 0,
                                          int64_t biasQ = 0)
{
    KernelDefinition kernel;
    kernel.name = "sdpa_" + dtype + "_d" + std::to_string(headDim);
    kernel.metadata = {{FLYDSL_SDPA_RDNA_DTYPE_FIELD, dtype},
                       {FLYDSL_SDPA_RDNA_HEAD_DIM_FIELD, headDim},
                       {FLYDSL_SDPA_RDNA_CAUSAL_FIELD, causal},
                       {FLYDSL_SDPA_RDNA_BLOCK_M_FIELD, blockM},
                       {FLYDSL_SDPA_RDNA_BLOCK_N_FIELD, blockN},
                       {FLYDSL_SDPA_RDNA_DV_SPLIT_FIELD, dvSplit},
                       {FLYDSL_SDPA_RDNA_HAS_BIAS_FIELD, hasBias},
                       {FLYDSL_SDPA_RDNA_HEAD_DIM_MAX_FIELD, headDimMax.value_or(headDim)},
                       {FLYDSL_SDPA_RDNA_DECODE_FIELD, decode},
                       {FLYDSL_SDPA_RDNA_MERGE_SYMBOL_FIELD,
                        std::string(decode != 0 ? "flash_attn_decode_merge_rdna_kernel_1" : "")},
                       {FLYDSL_SDPA_RDNA_HD_PAIR_FIELD, hdPair},
                       {FLYDSL_SDPA_RDNA_BIAS_Q_FIELD, biasQ}};
    kernel.priority = 100;
    return kernel;
}

// ---------------------------------------------------------------------------
// graph_match: what this pack accepts
// ---------------------------------------------------------------------------

/// A spec the graph match must accept, named for what it exercises.
struct AcceptedCase
{
    std::string name;
    SdpaGraphSpec spec;
};

class TestFlydslSdpaRdnaGraphAccepts : public ::testing::TestWithParam<AcceptedCase>
{
};

TEST_P(TestFlydslSdpaRdnaGraphAccepts, BindsTheGraph)
{
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(GetParam().spec));
    EXPECT_TRUE(matchesGraph(FLYDSL_SDPA_RDNA, fixture.context()).has_value());
}

SdpaGraphSpec accepted(void (*edit)(SdpaGraphSpec&))
{
    SdpaGraphSpec spec;
    edit(spec);
    return spec;
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestFlydslSdpaRdnaGraphAccepts,
    ::testing::Values(
        AcceptedCase{"Bf16Bshd", accepted([](SdpaGraphSpec&) {})},
        AcceptedCase{"F16Bhsd", accepted([](SdpaGraphSpec& s) {
                         s.dataType = DataType::HALF;
                         s.layout = SdpaLayout::BHSD;
                     })},
        AcceptedCase{"HeadDim128", accepted([](SdpaGraphSpec& s) { s.headDim = 128; })},
        // The kernel takes any head_dim that is a multiple of 32 from 64 up. Whether
        // an object was SHIPPED for one is kernel_match's question, not this one's.
        AcceptedCase{"HeadDim96", accepted([](SdpaGraphSpec& s) { s.headDim = 96; })},
        AcceptedCase{"HeadDim256", accepted([](SdpaGraphSpec& s) { s.headDim = 256; })},
        // Served by the generic tier, which reads head_dim at runtime.
        AcceptedCase{"HeadDim80", accepted([](SdpaGraphSpec& s) { s.headDim = 80; })},
        // K and V may have different head counts, each dividing Q's.
        AcceptedCase{"ValueHeadsDifferFromKeyHeads", accepted([](SdpaGraphSpec& s) {
                         s.heads = 8;
                         s.kvHeads = 2;
                         s.valueHeads = 4;
                     })},
        AcceptedCase{"PackedQkv", accepted([](SdpaGraphSpec& s) { s.packedQkv = true; })},
        // An additive f32 bias, broadcast the ways the reference reads it.
        AcceptedCase{"AdditiveBias", accepted([](SdpaGraphSpec& s) { s.withBias = true; })},
        AcceptedCase{"BiasPerBatchAndHead", accepted([](SdpaGraphSpec& s) {
                         s.withBias = true;
                         s.biasDims = {s.batch, s.heads, s.seqLenQ, s.seqLenKv};
                     })},
        AcceptedCase{"BiasRankTwo", accepted([](SdpaGraphSpec& s) {
                         s.withBias = true;
                         s.biasDims = {s.seqLenQ, s.seqLenKv};
                     })},
        AcceptedCase{"BiasInTheQueryDtype", accepted([](SdpaGraphSpec& s) {
                         s.withBias = true;
                         s.biasDataType = s.dataType;
                     })},
        AcceptedCase{"BiasOverKeysOnly", accepted([](SdpaGraphSpec& s) {
                         s.withBias = true;
                         s.biasDims = {s.seqLenKv};
                     })},
        AcceptedCase{"GroupedQueryHeads", accepted([](SdpaGraphSpec& s) {
                         s.heads = 8;
                         s.kvHeads = 2;
                     })},
        AcceptedCase{"MultiQuery", accepted([](SdpaGraphSpec& s) { s.kvHeads = 1; })},
        AcceptedCase{"SequenceNotAMultipleOfAnyTile", accepted([](SdpaGraphSpec& s) {
                         s.seqLenQ = 77;
                         s.seqLenKv = 77;
                     })},
        AcceptedCase{"CrossAttention", accepted([](SdpaGraphSpec& s) {
                         s.seqLenQ = 64;
                         s.seqLenKv = 300;
                     })},
        AcceptedCase{"SingleTokenDecode", accepted([](SdpaGraphSpec& s) {
                         s.seqLenQ = 1;
                         s.seqLenKv = 513;
                     })},
        AcceptedCase{"TopLeftCausal",
                     accepted([](SdpaGraphSpec& s) { s.mask = SdpaMask::TOP_LEFT_CAUSAL; })},
        AcceptedCase{"BottomRightCausalWithMoreKeys", accepted([](SdpaGraphSpec& s) {
                         s.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
                         s.seqLenQ = 64;
                         s.seqLenKv = 300;
                     })},
        // A row no key reaches gets O = 0 and LSE = -inf, as the reference defines it.
        AcceptedCase{"BottomRightCausalWithMoreQueries", accepted([](SdpaGraphSpec& s) {
                         s.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
                         s.seqLenQ = 300;
                         s.seqLenKv = 64;
                     })},
        AcceptedCase{"SlidingWindow",
                     accepted([](SdpaGraphSpec& s) { s.mask = SdpaMask::SLIDING_WINDOW; })},
        AcceptedCase{"LeftOnlyWindow", accepted([](SdpaGraphSpec& s) { s.leftBound = 32; })},
        AcceptedCase{"RightBoundBand", accepted([](SdpaGraphSpec& s) {
                         s.mask = SdpaMask::TOP_LEFT_CAUSAL;
                         s.rightBound = 16;
                     })},
        AcceptedCase{"SoftmaxStats", accepted([](SdpaGraphSpec& s) { s.stats = SdpaStats::LSE; })},
        AcceptedCase{"ScaleAbsent",
                     accepted([](SdpaGraphSpec& s) { s.scaleKind = SdpaScale::ABSENT; })},
        AcceptedCase{"ScaleConstantTensor",
                     accepted([](SdpaGraphSpec& s) { s.scaleKind = SdpaScale::CONSTANT_TENSOR; })},
        AcceptedCase{"ScaleRuntimeTensor",
                     accepted([](SdpaGraphSpec& s) { s.scaleKind = SdpaScale::RUNTIME_TENSOR; })}),
    [](const ::testing::TestParamInfo<AcceptedCase>& info) { return info.param.name; });

// ---------------------------------------------------------------------------
// graph_match: what this pack declines, each for the reason the kernel cannot serve it
// ---------------------------------------------------------------------------

class TestFlydslSdpaRdnaGraphDeclines : public ::testing::TestWithParam<AcceptedCase>
{
};

TEST_P(TestFlydslSdpaRdnaGraphDeclines, DeclinesTheGraph)
{
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(GetParam().spec));
    EXPECT_FALSE(matchesGraph(FLYDSL_SDPA_RDNA, fixture.context()).has_value());
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestFlydslSdpaRdnaGraphDeclines,
    ::testing::Values(
        // No f32 object exists, and the kernel's WMMA has no f32 operand form.
        AcceptedCase{"Fp32", accepted([](SdpaGraphSpec& s) { s.dataType = DataType::FLOAT; })},
        // One element type throughout the kernel.
        AcceptedCase{"MixedDtypes",
                     accepted([](SdpaGraphSpec& s) { s.keyDataType = DataType::HALF; })},
        // Each lane reads a row's head-dim values as one contiguous vector.
        AcceptedCase{"StridedHeadDim",
                     accepted([](SdpaGraphSpec& s) { s.queryHeadDimStride = 2; })},
        // One head_dim for Q, K and V.
        AcceptedCase{"ValueHeadDimDiffers",
                     accepted([](SdpaGraphSpec& s) { s.valueHeadDim = 128; })},
        // The kv head is the query head divided by the group size.
        AcceptedCase{"HeadsNotAMultipleOfKvHeads", accepted([](SdpaGraphSpec& s) {
                         s.heads = 6;
                         s.kvHeads = 4;
                     })},
        AcceptedCase{"HeadsNotAMultipleOfValueHeads", accepted([](SdpaGraphSpec& s) {
                         s.heads = 6;
                         s.kvHeads = 2;
                         s.valueHeads = 4;
                     })},
        // A bound below -1 has no meaning in the reference's convention.
        AcceptedCase{"LeftBoundBelowMinusOne",
                     accepted([](SdpaGraphSpec& s) { s.leftBound = -2; })},
        // The kernel reads a bias in f32 or the query's dtype, and only one that broadcasts
        // to [B, H, Sq, Skv].
        AcceptedCase{"BiasInAThirdDtype", accepted([](SdpaGraphSpec& s) {
                         s.withBias = true;
                         s.biasDataType = DataType::HALF;
                     })},
        AcceptedCase{"BiasThatDoesNotBroadcast", accepted([](SdpaGraphSpec& s) {
                         s.withBias = true;
                         s.biasDims = {1, 1, s.seqLenQ, s.seqLenKv + 1};
                     })},
        AcceptedCase{"BiasOfRankFive", accepted([](SdpaGraphSpec& s) {
                         s.withBias = true;
                         s.biasDims = {1, 1, 1, s.seqLenQ, s.seqLenKv};
                     })},
        // The LSE output the kernel writes is f32 [B, H, Sq, 1], and only when named.
        AcceptedCase{"StatsBf16",
                     accepted([](SdpaGraphSpec& s) { s.stats = SdpaStats::LSE_BF16; })},
        AcceptedCase{"StatsWrongShape",
                     accepted([](SdpaGraphSpec& s) { s.stats = SdpaStats::LSE_WRONG_SHAPE; })},
        AcceptedCase{"StatsRequestedWithoutTensor", accepted([](SdpaGraphSpec& s) {
                         s.stats = SdpaStats::REQUESTED_WITHOUT_TENSOR;
                     })},
        // A ragged (THD) operand would be read as one dense batch.
        AcceptedCase{"RaggedQuery", accepted([](SdpaGraphSpec& s) { s.raggedQuery = true; })},
        // The running max is taken over unscaled scores, which orders them like the
        // scaled ones only for a positive scale.
        AcceptedCase{"NegativeScale", accepted([](SdpaGraphSpec& s) { s.scale = -0.125F; })},
        AcceptedCase{"Dropout", accepted([](SdpaGraphSpec& s) { s.withDropout = true; })},
        // A scale behind a device pointer is not a kernel argument.
        AcceptedCase{"ScaleDeviceTensor",
                     accepted([](SdpaGraphSpec& s) { s.scaleKind = SdpaScale::DEVICE_TENSOR; })}),
    [](const ::testing::TestParamInfo<AcceptedCase>& info) { return info.param.name; });

// ---------------------------------------------------------------------------
// graph_match: what it binds
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaRdnaBinding, BindsTheFourOperandsByUid)
{
    const auto bound = bindingsFor();
    EXPECT_EQ(boundInt(bound, std::string(FLYDSL_SDPA_RDNA.inputAToken).c_str()),
              FLYDSL_SDPA_RDNA_Q_UID);
    EXPECT_EQ(boundInt(bound, std::string(FLYDSL_SDPA_RDNA.inputBToken).c_str()),
              FLYDSL_SDPA_RDNA_K_UID);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_RDNA_V_TOKEN), FLYDSL_SDPA_RDNA_V_UID);
    EXPECT_EQ(boundInt(bound, std::string(FLYDSL_SDPA_RDNA.outputToken).c_str()),
              FLYDSL_SDPA_RDNA_O_UID);
}

TEST(TestFlydslSdpaRdnaBinding, AnUnmaskedGraphSelectsTheNonCausalVariant)
{
    const auto bound = bindingsFor();
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_RDNA_CAUSAL_TOKEN), 0);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_RDNA_RIGHT_BOUND_TOKEN), -1);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_RDNA_LEFT_BOUND_TOKEN), -1);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_RDNA_STATS_TOKEN), FLYDSL_SDPA_RDNA_NO_STATS);
}

TEST(TestFlydslSdpaRdnaBinding, TopLeftCausalIsARightBoundOfZeroAtTheTopLeft)
{
    SdpaGraphSpec spec;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.seqLenQ = 64;
    spec.seqLenKv = 300;
    const auto bound = bindingsFor(spec);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_RDNA_CAUSAL_TOKEN), 1);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_RDNA_RIGHT_BOUND_TOKEN), 0);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_RDNA_ALIGN_BOTTOM_RIGHT_TOKEN), 0);
}

TEST(TestFlydslSdpaRdnaBinding, BottomRightCausalLeavesTheOffsetToTheKernel)
{
    SdpaGraphSpec spec;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.seqLenQ = 64;
    spec.seqLenKv = 300;
    const auto bound = bindingsFor(spec);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_RDNA_CAUSAL_TOKEN), 1);
    // The kernel derives Skv - Sq itself, so per-batch lengths can move it later.
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_RDNA_RIGHT_BOUND_TOKEN), 0);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_RDNA_ALIGN_BOTTOM_RIGHT_TOKEN), 1);
}

TEST(TestFlydslSdpaRdnaBinding, AWindowBindsItsLeftBoundAndClampsAnOversizedOne)
{
    SdpaGraphSpec spec;
    spec.mask = SdpaMask::SLIDING_WINDOW;
    EXPECT_EQ(boundInt(bindingsFor(spec), FLYDSL_SDPA_RDNA_LEFT_BOUND_TOKEN), 64);

    // Past Sq + Skv a window masks nothing more, and the kernel's int32 sums need the
    // bound kept in range.
    spec.leftBound = int64_t{1} << 40;
    EXPECT_EQ(boundInt(bindingsFor(spec), FLYDSL_SDPA_RDNA_LEFT_BOUND_TOKEN),
              spec.seqLenQ + spec.seqLenKv);
}

TEST(TestFlydslSdpaRdnaBinding, ABandSelectsTheCausalVariantWithItsRightBound)
{
    SdpaGraphSpec spec;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.rightBound = 16;
    const auto bound = bindingsFor(spec);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_RDNA_CAUSAL_TOKEN), 1);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_RDNA_RIGHT_BOUND_TOKEN), 16);
}

TEST(TestFlydslSdpaRdnaBinding, BindsTheStatsOutputByUid)
{
    SdpaGraphSpec spec;
    spec.stats = SdpaStats::LSE;
    EXPECT_EQ(boundInt(bindingsFor(spec), FLYDSL_SDPA_RDNA_STATS_TOKEN),
              FLYDSL_SDPA_RDNA_STATS_UID);
}

TEST(TestFlydslSdpaRdnaBinding, BindsAnExplicitScaleAsItsBitPattern)
{
    // As an integer, 0.0625 would read on the device as a denormal near zero.
    SdpaGraphSpec spec;
    spec.scale = 0.0625F;
    const auto bound = bindingsFor(spec);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_RDNA_SCALE_SOURCE_TOKEN), FLYDSL_SDPA_RDNA_SCALE_BAKED);
    EXPECT_EQ(boundScale(bound), 0.0625F);
}

TEST(TestFlydslSdpaRdnaBinding, AnAbsentScaleDefaultsToTheInverseRootOfTheHeadDim)
{
    // hipDNN's CPU reference applies this same default (CpuFpReferenceSdpa.hpp).
    SdpaGraphSpec spec;
    spec.scaleKind = SdpaScale::ABSENT;
    spec.headDim = 128;
    const auto bound = bindingsFor(spec);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_RDNA_SCALE_SOURCE_TOKEN), FLYDSL_SDPA_RDNA_SCALE_BAKED);
    EXPECT_FLOAT_EQ(boundScale(bound), 1.0F / std::sqrt(128.0F));
}

TEST(TestFlydslSdpaRdnaBinding, AScaleTensorIsBoundByUidForExecuteToResolve)
{
    SdpaGraphSpec spec;
    spec.scaleKind = SdpaScale::RUNTIME_TENSOR;
    const auto bound = bindingsFor(spec);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_RDNA_SCALE_SOURCE_TOKEN), FLYDSL_SDPA_RDNA_SCALE_TENSOR);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_RDNA_SCALE_TOKEN), FLYDSL_SDPA_RDNA_SCALE_UID);
}

// ---------------------------------------------------------------------------
// kernel_match: which candidate a graph selects
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaRdnaKernelMatch, AdmitsTheCandidateBuiltForTheGraphsClass)
{
    SdpaGraphSpec spec;
    spec.headDim = 128;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    const auto bound = bindingsFor(spec);
    EXPECT_TRUE(matchesKernel(
        FLYDSL_SDPA_RDNA, fixture.context(), makeFlydslSdpaRdnaKernel("bf16", 128, 1), bound));
}

TEST(TestFlydslSdpaRdnaKernelMatch, DeclinesEachBakedAxisThatDisagrees)
{
    SdpaGraphSpec spec;
    spec.headDim = 128;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    const auto bound = bindingsFor(spec);

    EXPECT_FALSE(matchesKernel(
        FLYDSL_SDPA_RDNA, fixture.context(), makeFlydslSdpaRdnaKernel("f16", 128, 1), bound));
    EXPECT_FALSE(matchesKernel(
        FLYDSL_SDPA_RDNA, fixture.context(), makeFlydslSdpaRdnaKernel("bf16", 64, 1), bound));
    EXPECT_FALSE(matchesKernel(
        FLYDSL_SDPA_RDNA, fixture.context(), makeFlydslSdpaRdnaKernel("bf16", 128, 0), bound));
}

TEST(TestFlydslSdpaRdnaKernelMatch, DeclinesACandidateWhoseTileIsNotOneTheKernelBuilds)
{
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph());
    const auto bound = bindingsFor();

    // block_m is whole waves of 16 rows; block_n whole 32-key sub-tiles.
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA,
                               fixture.context(),
                               makeFlydslSdpaRdnaKernel("bf16", 64, 0, 120, 32),
                               bound));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA,
                               fixture.context(),
                               makeFlydslSdpaRdnaKernel("bf16", 64, 0, 128, 48),
                               bound));
}

TEST(TestFlydslSdpaRdnaKernelMatch, AdmitsHeadDimPairsAndDeclinesAnyOtherWaveCount)
{
    SdpaGraphSpec spec;
    spec.headDim = 256;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    const auto bound = bindingsFor(spec);

    // hd_pair is 0 or 1: the dispatcher launches block_m / 16 waves, or twice that.
    EXPECT_TRUE(
        matchesKernel(FLYDSL_SDPA_RDNA,
                      fixture.context(),
                      makeFlydslSdpaRdnaKernel("bf16", 256, 0, 64, 32, 1, 0, std::nullopt, 0, 1),
                      bound));
    EXPECT_FALSE(
        matchesKernel(FLYDSL_SDPA_RDNA,
                      fixture.context(),
                      makeFlydslSdpaRdnaKernel("bf16", 256, 0, 64, 32, 1, 0, std::nullopt, 0, 2),
                      bound));
}

TEST(TestFlydslSdpaRdnaKernelMatch, AdmitsAColumnSplitThatDividesTheHead)
{
    SdpaGraphSpec spec;
    spec.headDim = 256;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    const auto bound = bindingsFor(spec);

    EXPECT_TRUE(matchesKernel(FLYDSL_SDPA_RDNA,
                              fixture.context(),
                              makeFlydslSdpaRdnaKernel("bf16", 256, 0, 128, 32, 2),
                              bound));
}

TEST(TestFlydslSdpaRdnaKernelMatch, DeclinesAColumnSplitThatDoesNotDivideTheHead)
{
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph());
    const auto bound = bindingsFor();

    // head_dim 64 into 3 column tiles, and a split of none at all.
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA,
                               fixture.context(),
                               makeFlydslSdpaRdnaKernel("bf16", 64, 0, 128, 32, 3),
                               bound));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA,
                               fixture.context(),
                               makeFlydslSdpaRdnaKernel("bf16", 64, 0, 128, 32, 0),
                               bound));
}

TEST(TestFlydslSdpaRdnaKernelMatch, AGenericObjectServesMultiplesOfEightUpToItsLargest)
{
    // head_dim 0 is the generic object's: it reads the real one at runtime.
    const auto generic128 = makeFlydslSdpaRdnaKernel("bf16", 0, 0, 128, 32, 1, 0, 128);
    const auto generic256 = makeFlydslSdpaRdnaKernel("bf16", 0, 0, 128, 32, 4, 0, 256);
    const auto matches = [](int64_t headDim, const KernelDefinition& kernel) {
        SdpaGraphSpec spec;
        spec.headDim = headDim;
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
        return matchesKernel(FLYDSL_SDPA_RDNA, fixture.context(), kernel, bindingsFor(spec));
    };

    EXPECT_TRUE(matches(8, generic128));
    EXPECT_TRUE(matches(80, generic128));
    EXPECT_TRUE(matches(128, generic128));
    EXPECT_FALSE(matches(84, generic128)) << "not a multiple of 8";
    EXPECT_FALSE(matches(136, generic128)) << "past its largest";
    EXPECT_TRUE(matches(136, generic256));
    EXPECT_TRUE(matches(248, generic256));
}

TEST(TestFlydslSdpaRdnaKernelMatch, ASpecializedObjectServesOnlyItsOwnHead)
{
    SdpaGraphSpec spec;
    spec.headDim = 80;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    // A record claiming head_dim 64 but a larger maximum is malformed, not generic.
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA,
                               fixture.context(),
                               makeFlydslSdpaRdnaKernel("bf16", 64, 0, 128, 32, 1, 0, 128),
                               bindingsFor(spec)));
}

TEST(TestFlydslSdpaRdnaKernelMatch, ADecodeObjectServesUpToItsRowLimit)
{
    const auto decode = makeFlydslSdpaRdnaKernel("bf16", 128, 1, 16, 32, 1, 0, 128, 1);
    const auto matches = [&decode](int64_t heads, int64_t kvHeads, int64_t seqLenQ) {
        SdpaGraphSpec spec;
        spec.headDim = 128;
        spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
        spec.heads = heads;
        spec.kvHeads = kvHeads;
        spec.seqLenQ = seqLenQ;
        spec.seqLenKv = 300;
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
        return matchesKernel(FLYDSL_SDPA_RDNA, fixture.context(), decode, bindingsFor(spec));
    };

    EXPECT_TRUE(matches(32, 8, 1)) << "group of 4, one position";
    EXPECT_TRUE(matches(16, 1, 1)) << "MQA, group of 16";
    EXPECT_TRUE(matches(8, 2, 4)) << "group of 4, four positions";
    EXPECT_TRUE(matches(32, 1, 1)) << "MQA, group of 32: two row tiles";
    EXPECT_TRUE(matches(8, 2, 32)) << "group of 4, 32 positions: 128 rows";
    EXPECT_FALSE(matches(8, 2, 33)) << "132 rows: the prefill objects are faster";
    EXPECT_FALSE(matches(64, 1, 3)) << "MQA, group of 64, three positions: 192 rows";
}

TEST(TestFlydslSdpaRdnaKernelMatch, ADecodeObjectDeclinesWhatItsTileCannotHold)
{
    const auto decode = makeFlydslSdpaRdnaKernel("bf16", 128, 1, 16, 32, 1, 0, 128, 1);
    SdpaGraphSpec spec;
    spec.headDim = 128;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.valueHeads = 4;
    spec.seqLenQ = 1;
    spec.seqLenKv = 300;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, fixture.context(), decode, bindingsFor(spec)))
        << "a tile shares one KV head, so K and V must group alike";

    auto unnamed = decode;
    unnamed.metadata[FLYDSL_SDPA_RDNA_MERGE_SYMBOL_FIELD] = std::string();
    spec.valueHeads.reset();
    const GraphFixture plain(buildFlydslSdpaRdnaGraph(spec));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, plain.context(), unnamed, bindingsFor(spec)))
        << "a decode object must name its merge kernel";
}

TEST(TestFlydslSdpaRdnaKernelMatch, ADecodeBiasObjectServesABiasItsOffsetsReach)
{
    const auto plain = makeFlydslSdpaRdnaKernel("bf16", 128, 1, 16, 32, 1, 0, 128, 1);
    const auto bias = makeFlydslSdpaRdnaKernel("bf16", 128, 1, 16, 32, 1, 1, 128, 1);
    SdpaGraphSpec spec;
    spec.headDim = 128;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 1;
    spec.seqLenKv = 300;
    spec.withBias = true;
    const GraphFixture biased(buildFlydslSdpaRdnaGraph(spec));
    EXPECT_TRUE(matchesKernel(FLYDSL_SDPA_RDNA, biased.context(), bias, bindingsFor(spec)));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, biased.context(), plain, bindingsFor(spec)));

    // A bias object is built for one 16-row tile: an MQA group of 32 goes to prefill.
    SdpaGraphSpec mqa = spec;
    mqa.heads = 32;
    mqa.kvHeads = 1;
    const GraphFixture wide(buildFlydslSdpaRdnaGraph(mqa));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, wide.context(), bias, bindingsFor(mqa)));

    // One tile spans the four heads of a group; a head stride that puts the last of them
    // past the 32-bit offsets declines, though each (batch, head) slice alone fits.
    spec.biasDims = {1, spec.heads, 1, spec.seqLenKv};
    spec.biasStrides = {spec.heads * (int64_t{1} << 29), int64_t{1} << 29, spec.seqLenKv, 1};
    const GraphFixture farHeads(buildFlydslSdpaRdnaGraph(spec));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, farHeads.context(), bias, bindingsFor(spec)));
}

TEST(TestFlydslSdpaRdnaKernelMatch, PairsTheBiasVariantWithAGraphThatHasABias)
{
    SdpaGraphSpec biased;
    biased.withBias = true;
    const GraphFixture withBias(buildFlydslSdpaRdnaGraph(biased));
    const auto biasBound = bindingsFor(biased);
    const GraphFixture without(buildFlydslSdpaRdnaGraph());
    const auto plainBound = bindingsFor();

    const auto plain = makeFlydslSdpaRdnaKernel("bf16", 64, 0);
    const auto bias = makeFlydslSdpaRdnaKernel("bf16", 64, 0, 128, 32, 1, 1);
    const auto biasQ = makeFlydslSdpaRdnaKernel("bf16", 64, 0, 128, 32, 1, 1, 64, 0, 0, 1);
    EXPECT_TRUE(matchesKernel(FLYDSL_SDPA_RDNA, withBias.context(), bias, biasBound));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, withBias.context(), plain, biasBound));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, withBias.context(), biasQ, biasBound));
    EXPECT_TRUE(matchesKernel(FLYDSL_SDPA_RDNA, without.context(), plain, plainBound));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, without.context(), bias, plainBound));

    // A bias in the query's dtype pairs with the `bias_q` object instead.
    biased.biasDataType = biased.dataType;
    const GraphFixture withBiasQ(buildFlydslSdpaRdnaGraph(biased));
    const auto biasQBound = bindingsFor(biased);
    EXPECT_TRUE(matchesKernel(FLYDSL_SDPA_RDNA, withBiasQ.context(), biasQ, biasQBound));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, withBiasQ.context(), bias, biasQBound));
}

TEST(TestFlydslSdpaRdnaKernelMatch, DeclinesARecordMissingAFieldRatherThanThrowing)
{
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph());
    const auto bound = bindingsFor();

    auto kernel = makeFlydslSdpaRdnaKernel("bf16", 64, 0);
    kernel.metadata.erase(FLYDSL_SDPA_RDNA_BLOCK_M_FIELD);
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, fixture.context(), kernel, bound));

    kernel = makeFlydslSdpaRdnaKernel("bf16", 64, 0);
    kernel.metadata[FLYDSL_SDPA_RDNA_HEAD_DIM_FIELD] = std::string("64");
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, fixture.context(), kernel, bound));

    kernel = makeFlydslSdpaRdnaKernel("bf16", 64, 0);
    kernel.metadata.erase(FLYDSL_SDPA_RDNA_DV_SPLIT_FIELD);
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, fixture.context(), kernel, bound));

    kernel = makeFlydslSdpaRdnaKernel("bf16", 64, 0);
    kernel.metadata.erase(FLYDSL_SDPA_RDNA_HAS_BIAS_FIELD);
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, fixture.context(), kernel, bound));

    kernel = makeFlydslSdpaRdnaKernel("bf16", 64, 0);
    kernel.metadata.erase(FLYDSL_SDPA_RDNA_HEAD_DIM_MAX_FIELD);
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, fixture.context(), kernel, bound));
}

TEST(TestFlydslSdpaRdnaKernelMatch, ScoresByDescriptorPriority)
{
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph());
    auto kernel = makeFlydslSdpaRdnaKernel("bf16", 64, 0);
    kernel.priority = 37;
    EXPECT_EQ(scoreKernel(FLYDSL_SDPA_RDNA, fixture.context(), kernel), 37.0);
}

// ---------------------------------------------------------------------------
// dispatch: the seams that need no device
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaRdnaDispatch, ResolvesFromTheDispatchRegistry)
{
    EXPECT_NO_THROW(static_cast<void>(dispatchHandler(FLYDSL_SDPA_RDNA)));
}

TEST(TestFlydslSdpaRdnaDispatch, NeedsNoWorkspace)
{
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph());
    EXPECT_EQ(dispatchHandler(FLYDSL_SDPA_RDNA)
                  .workspaceBytes(
                      fixture.context(), bindingsFor(), makeFlydslSdpaRdnaKernel("bf16", 64, 0)),
              0U);
}

TEST(TestFlydslSdpaRdnaDispatch, RefusesToPrepareWithoutTheMatcherSBindings)
{
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph());
    EXPECT_THROW(
        dispatchHandler(FLYDSL_SDPA_RDNA)
            .prepare(fixture.context(), BoundTokens{}, makeFlydslSdpaRdnaKernel("bf16", 64, 0)),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

// ---------------------------------------------------------------------------
// dispatch on device: a real launch out of the staged archive
// ---------------------------------------------------------------------------

/// The device's architecture with feature suffixes removed, which is what the shard and
/// its kpack entries are keyed on.
DeviceProperties packedArchDeviceProperties()
{
    auto properties = currentDeviceProperties();
    const auto colon = properties.gcnArchName.find(':');
    if(colon != std::string::npos)
    {
        properties.gcnArchName.resize(colon);
    }
    return properties;
}

/// The staged candidate for (@p dtype, @p headDim, @p causal, @p hasBias), if this build
/// packed one: the object built for that head_dim, else the narrowest generic one serving it.
std::optional<KernelDefinition> findStagedKernel(const std::string& dtype,
                                                 int64_t headDim,
                                                 int64_t causal,
                                                 int64_t hasBias,
                                                 int64_t decode = 0,
                                                 int64_t biasQ = 0)
{
    // A build for CDNA targets only stages no RDNA set at all.
    const auto* staged = findLoadedSet(FLYDSL_SDPA_RDNA.engineName);
    if(staged == nullptr)
    {
        return std::nullopt;
    }
    const auto& set = *staged;
    // A build for several arches loads every arch's shard: pass over another device's.
    const auto device = packedArchDeviceProperties().gcnArchName;
    std::optional<KernelDefinition> best;
    int64_t bestMax = 0;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto& arches = kernel.arch.empty() ? pack.arch : kernel.arch;
            if(!device.empty() && !arches.empty()
               && std::find(arches.begin(), arches.end(), device) == arches.end())
            {
                continue;
            }
            const auto* name
                = tryGetMetadataField<std::string>(kernel.metadata, FLYDSL_SDPA_RDNA_DTYPE_FIELD);
            const auto* dim
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SDPA_RDNA_HEAD_DIM_FIELD);
            const auto* isCausal
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SDPA_RDNA_CAUSAL_FIELD);
            const auto* bias
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SDPA_RDNA_HAS_BIAS_FIELD);
            const auto* dimMax = tryGetMetadataField<int64_t>(kernel.metadata,
                                                              FLYDSL_SDPA_RDNA_HEAD_DIM_MAX_FIELD);
            const auto* isDecode
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SDPA_RDNA_DECODE_FIELD);
            const auto* inQueryType
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SDPA_RDNA_BIAS_Q_FIELD);
            if(name == nullptr || dim == nullptr || isCausal == nullptr || bias == nullptr
               || dimMax == nullptr || isDecode == nullptr || inQueryType == nullptr
               || *name != dtype || *isCausal != causal || *bias != hasBias || *isDecode != decode
               || *inQueryType != biasQ)
            {
                continue;
            }
            const bool exact = *dim == headDim;
            const bool generic = *dim == 0 && headDim % 8 == 0 && headDim <= *dimMax;
            if(!exact && !generic)
            {
                continue;
            }
            KernelDefinition definition{kernel.id,
                                        pack.id,
                                        pack.dispatchId,
                                        kernel.source,
                                        kernel.metadata,
                                        kernel.priority,
                                        kernel.arch.empty() ? pack.arch : kernel.arch,
                                        kernel.originDirectory,
                                        kernel.name,
                                        kernel.treeRoot};
            if(exact)
            {
                return definition;
            }
            if(!best.has_value() || *dimMax < bestMax)
            {
                best = definition;
                bestMax = *dimMax;
            }
        }
    }
    return best;
}

/// One operand's host values and device copy, laid out per the graph spec's strides.
template <typename T>
class DeviceTensor
{
public:
    DeviceTensor(const std::vector<int64_t>& dims,
                 const std::vector<int64_t>& strides,
                 uint32_t seed,
                 bool fill)
        : _dims(dims)
        , _strides(strides)
    {
        int64_t span = 1;
        for(size_t axis = 0; axis < dims.size(); ++axis)
        {
            span += (dims[axis] - 1) * strides[axis];
        }
        _host.assign(static_cast<size_t>(span), T(0.0F));

        if(fill)
        {
            // k/32 for |k| <= 32: exact in both bf16 and f16, so the reference reads the
            // same values the device does and input rounding stays out of the budget.
            uint32_t state = seed;
            for(auto& value : _host)
            {
                state = state * 1664525U + 1013904223U;
                const auto k = static_cast<int32_t>((state >> 16) % 65U) - 32;
                value = T(static_cast<float>(k) / 32.0F);
            }
        }

        EXPECT_EQ(hipSuccess, hipMalloc(&_device, bytes()));
        EXPECT_EQ(hipSuccess, hipMemcpy(_device, _host.data(), bytes(), hipMemcpyHostToDevice));
    }

    ~DeviceTensor()
    {
        static_cast<void>(hipFree(_device));
    }

    DeviceTensor(const DeviceTensor&) = delete;
    DeviceTensor& operator=(const DeviceTensor&) = delete;

    void* device() const
    {
        return _device;
    }

    /// Element (b, h, s, d) as the reference reads it.
    double at(int64_t b, int64_t h, int64_t s, int64_t d) const
    {
        const auto index = b * _strides[0] + h * _strides[1] + s * _strides[2] + d * _strides[3];
        return static_cast<double>(static_cast<float>(_host[static_cast<size_t>(index)]));
    }

    /// Copies the device contents back over the host values.
    void readBack()
    {
        EXPECT_EQ(hipSuccess, hipMemcpy(_host.data(), _device, bytes(), hipMemcpyDeviceToHost));
    }

private:
    size_t bytes() const
    {
        return _host.size() * sizeof(T);
    }

    std::vector<int64_t> _dims;
    std::vector<int64_t> _strides;
    std::vector<T> _host;
    void* _device = nullptr;
};

/// Runs the staged kernel for @p spec through prepare() and launch(), and checks every
/// output element against a double-precision reference computed from the same inputs.
///
/// Skips when the staged set holds no kernel for the spec's class: on an arch this build
/// packed nothing for, there is no kernel to be wrong about.
template <typename T>
void expectSdpaMatchesReference(const SdpaGraphSpec& spec,
                                double absoluteTolerance,
                                bool decode = false)
{
    // The bounds the graph builder writes for this spec, in CpuFpReferenceSdpa's terms.
    int64_t left = -1;
    int64_t right = -1;
    if(spec.mask == SdpaMask::TOP_LEFT_CAUSAL || spec.mask == SdpaMask::BOTTOM_RIGHT_CAUSAL)
    {
        right = 0;
    }
    if(spec.mask == SdpaMask::SLIDING_WINDOW)
    {
        left = 64;
        right = 0;
    }
    left = spec.leftBound.value_or(left);
    right = spec.rightBound.value_or(right);
    const int64_t align
        = spec.mask == SdpaMask::BOTTOM_RIGHT_CAUSAL ? spec.seqLenKv - spec.seqLenQ : 0;
    const auto isMasked = [&](int64_t row, int64_t key) {
        return (right >= 0 && key >= row + 1 + align + right)
               || (left >= 0 && key < row + align - left);
    };

    const std::string dtype = spec.dataType == DataType::BFLOAT16 ? "bf16" : "f16";
    const bool causal = right >= 0;
    const auto kernel
        = findStagedKernel(dtype,
                           spec.headDim,
                           causal ? 1 : 0,
                           spec.withBias ? 1 : 0,
                           decode ? 1 : 0,
                           spec.withBias && spec.biasDataType != DataType::FLOAT ? 1 : 0);
    if(!kernel.has_value())
    {
        GTEST_SKIP() << "nothing packed for dtype=" << dtype << " head_dim=" << spec.headDim
                     << " causal=" << causal << " in this build's descriptor root";
    }

    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec), packedArchDeviceProperties());
    const auto bound = matchesGraph(FLYDSL_SDPA_RDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());
    ASSERT_TRUE(matchesKernel(FLYDSL_SDPA_RDNA, fixture.context(), *kernel, *bound));

    const auto& handler = dispatchHandler(FLYDSL_SDPA_RDNA);
    const auto prepared = handler.prepare(fixture.context(), *bound, *kernel);
    ASSERT_NE(prepared, nullptr);

    const int64_t d = spec.headDim;
    const int64_t valueHeads = spec.valueHeads.value_or(spec.kvHeads);
    // Packed: one [B, S, 3, H, D] buffer, read as a (B, 3H, S, D) tensor whose head ranges
    // [0, H), [H, 2H) and [2H, 3H) are Q, K and V.
    std::optional<DeviceTensor<T>> qkv;
    std::optional<DeviceTensor<T>> qTensor;
    std::optional<DeviceTensor<T>> kTensor;
    std::optional<DeviceTensor<T>> vTensor;
    if(spec.packedQkv)
    {
        qkv.emplace(std::vector<int64_t>{spec.batch, 3 * spec.heads, spec.seqLenQ, d},
                    sdpaPackedQkvStrides(spec),
                    1U,
                    true);
    }
    else
    {
        qTensor.emplace(std::vector<int64_t>{spec.batch, spec.heads, spec.seqLenQ, d},
                        sdpaStrides(spec.layout, spec.heads, spec.seqLenQ, d),
                        1U,
                        true);
        kTensor.emplace(std::vector<int64_t>{spec.batch, spec.kvHeads, spec.seqLenKv, d},
                        sdpaStrides(spec.layout, spec.kvHeads, spec.seqLenKv, d),
                        2U,
                        true);
        vTensor.emplace(std::vector<int64_t>{spec.batch, valueHeads, spec.seqLenKv, d},
                        sdpaStrides(spec.layout, valueHeads, spec.seqLenKv, d),
                        3U,
                        true);
    }
    const auto qAt = [&](int64_t b, int64_t h, int64_t s, int64_t e) {
        return spec.packedQkv ? qkv->at(b, h, s, e) : qTensor->at(b, h, s, e);
    };
    const auto kAt = [&](int64_t b, int64_t h, int64_t s, int64_t e) {
        return spec.packedQkv ? qkv->at(b, spec.heads + h, s, e) : kTensor->at(b, h, s, e);
    };
    const auto vAt = [&](int64_t b, int64_t h, int64_t s, int64_t e) {
        return spec.packedQkv ? qkv->at(b, 2 * spec.heads + h, s, e) : vTensor->at(b, h, s, e);
    };
    const auto third = static_cast<size_t>(spec.heads * d) * sizeof(T);
    void* qPtr = spec.packedQkv ? qkv->device() : qTensor->device();
    void* kPtr = spec.packedQkv ? static_cast<void*>(static_cast<char*>(qkv->device()) + third)
                                : kTensor->device();
    void* vPtr = spec.packedQkv ? static_cast<void*>(static_cast<char*>(qkv->device()) + 2 * third)
                                : vTensor->device();
    DeviceTensor<T> o({spec.batch, spec.heads, spec.seqLenQ, d},
                      sdpaStrides(spec.layout, spec.heads, spec.seqLenQ, d),
                      4U,
                      false);

    // f32 [B, H, Sq, 1], packed BHSD: the layout the graph builder declares.
    DeviceTensor<float> stats({spec.batch, spec.heads, spec.seqLenQ, 1},
                              {spec.heads * spec.seqLenQ, spec.seqLenQ, 1, 1},
                              5U,
                              false);
    const bool wantStats = spec.stats == SdpaStats::LSE;

    // The bias as the device holds it, indexed as [B, H, Sq, Skv]: the graph's right-aligned
    // dims padded with 1s, and stride 0 on every broadcast axis.
    std::vector<int64_t> biasDims4{1, 1, 1, 1};
    std::vector<int64_t> biasStrides4{0, 0, 0, 0};
    if(spec.withBias)
    {
        const auto [dims, strides] = sdpaBiasLayout(spec);
        for(size_t axis = 0; axis < dims.size(); ++axis)
        {
            const auto target = 4 - dims.size() + axis;
            biasDims4[target] = dims[axis];
            biasStrides4[target] = dims[axis] == 1 ? 0 : strides[axis];
        }
    }
    // In f32, or in the query's dtype (the `bias_q` objects); one of the two is bound.
    const bool biasInQueryType = spec.withBias && spec.biasDataType != DataType::FLOAT;
    const DeviceTensor<float> bias(biasDims4, biasStrides4, 6U, true);
    const DeviceTensor<T> biasQ(biasDims4, biasStrides4, 6U, true);
    const auto biasAt = [&](int64_t b, int64_t h, int64_t s, int64_t j) {
        return biasInQueryType ? biasQ.at(b, h, s, j) : bias.at(b, h, s, j);
    };

    float runtimeScale = spec.scale;
    std::vector<hipdnnPluginDeviceBuffer_t> buffers{{FLYDSL_SDPA_RDNA_Q_UID, qPtr},
                                                    {FLYDSL_SDPA_RDNA_K_UID, kPtr},
                                                    {FLYDSL_SDPA_RDNA_V_UID, vPtr},
                                                    {FLYDSL_SDPA_RDNA_O_UID, o.device()}};
    if(wantStats)
    {
        buffers.push_back({FLYDSL_SDPA_RDNA_STATS_UID, stats.device()});
    }
    if(spec.withBias)
    {
        buffers.push_back(
            {FLYDSL_SDPA_RDNA_BIAS_UID, biasInQueryType ? biasQ.device() : bias.device()});
    }
    if(spec.scaleKind == SdpaScale::RUNTIME_TENSOR)
    {
        buffers.push_back({FLYDSL_SDPA_RDNA_SCALE_UID, &runtimeScale});
    }

    // A decode launch with more than one KV split writes its partials to workspace.
    const auto workspaceBytes = handler.workspaceBytes(fixture.context(), *bound, *kernel);
    void* workspace = nullptr;
    if(workspaceBytes > 0)
    {
        ASSERT_EQ(hipSuccess, hipMalloc(&workspace, workspaceBytes));
    }
    const Handle handle;
    handler.launch(
        handle, *prepared, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace);
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
    static_cast<void>(hipFree(workspace));
    o.readBack();
    stats.readBack();

    const double scale = spec.scaleKind == SdpaScale::ABSENT
                             ? 1.0 / std::sqrt(static_cast<double>(d))
                             : static_cast<double>(spec.scale);
    const int64_t group = spec.heads / spec.kvHeads;

    double worst = 0.0;
    std::string where;
    double worstLse = 0.0;
    std::string whereLse;
    std::vector<double> scores(static_cast<size_t>(spec.seqLenKv));
    for(int64_t b = 0; b < spec.batch; ++b)
    {
        for(int64_t h = 0; h < spec.heads; ++h)
        {
            const int64_t kvHead = h / group;
            const int64_t vHead = h / (spec.heads / valueHeads);
            for(int64_t s = 0; s < spec.seqLenQ; ++s)
            {
                double maxScore = -std::numeric_limits<double>::infinity();
                for(int64_t j = 0; j < spec.seqLenKv; ++j)
                {
                    if(isMasked(s, j))
                    {
                        continue;
                    }
                    double dot = 0.0;
                    for(int64_t e = 0; e < d; ++e)
                    {
                        dot += qAt(b, h, s, e) * kAt(b, kvHead, j, e);
                    }
                    scores[static_cast<size_t>(j)]
                        = dot * scale + (spec.withBias ? biasAt(b, h, s, j) : 0.0);
                    maxScore = std::max(maxScore, scores[static_cast<size_t>(j)]);
                }
                double sum = 0.0;
                for(int64_t j = 0; j < spec.seqLenKv; ++j)
                {
                    scores[static_cast<size_t>(j)]
                        = isMasked(s, j) ? 0.0
                                         : std::exp(scores[static_cast<size_t>(j)] - maxScore);
                    sum += scores[static_cast<size_t>(j)];
                }
                // A row no key reaches: O = 0 and LSE = -inf, as the reference defines it.
                const bool hasKeys = sum > 0.0;
                if(wantStats)
                {
                    const double expectedLse = hasKeys ? maxScore + std::log(sum)
                                                       : -std::numeric_limits<double>::infinity();
                    const double actualLse = stats.at(b, h, s, 0);
                    // A keyless row must be exactly -inf; any other value is a full miss.
                    const bool negativeInfinity = std::isinf(actualLse) && actualLse < 0.0;
                    double lseError = std::numeric_limits<double>::max();
                    if(hasKeys)
                    {
                        lseError = std::abs(actualLse - expectedLse);
                    }
                    else if(negativeInfinity)
                    {
                        lseError = 0.0;
                    }
                    if(lseError > worstLse)
                    {
                        worstLse = lseError;
                        whereLse = "b=" + std::to_string(b) + " h=" + std::to_string(h) + " s="
                                   + std::to_string(s) + ": got " + std::to_string(actualLse)
                                   + ", expected " + std::to_string(expectedLse);
                    }
                }
                for(int64_t e = 0; e < d; ++e)
                {
                    double expected = 0.0;
                    for(int64_t j = 0; j < spec.seqLenKv; ++j)
                    {
                        expected += scores[static_cast<size_t>(j)] * vAt(b, vHead, j, e);
                    }
                    expected = hasKeys ? expected / sum : 0.0;
                    const double error = std::abs(o.at(b, h, s, e) - expected);
                    if(error > worst)
                    {
                        worst = error;
                        where = "b=" + std::to_string(b) + " h=" + std::to_string(h)
                                + " s=" + std::to_string(s) + " d=" + std::to_string(e) + ": got "
                                + std::to_string(o.at(b, h, s, e)) + ", expected "
                                + std::to_string(expected);
                    }
                }
            }
        }
    }
    EXPECT_LE(worst, absoluteTolerance) << where;
    // LSE is computed in f32 from f32 accumulators, so it is far tighter than O.
    EXPECT_LE(worstLse, 1e-3) << whereLse;
}

// P is rounded to the input dtype before the second product, so bf16's 8-bit mantissa
// sets its budget; f16's 11 bits allow an order of magnitude less.
constexpr double BF16_TOLERANCE = 3e-2;
constexpr double F16_TOLERANCE = 4e-3;

TEST(TestGpuFlydslSdpaRdnaDispatch, ComputesBf16Bshd)
{
    SKIP_IF_NO_DEVICES();
    const SdpaGraphSpec spec;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, ComputesF16BhsdAtHeadDim128)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.layout = SdpaLayout::BHSD;
    spec.headDim = 128;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, ComputesCausalGroupedQueryAttentionAtARaggedLength)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 77;
    spec.seqLenKv = 77;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, ComputesBottomRightCausalCrossAttention)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.headDim = 128;
    spec.seqLenQ = 64;
    spec.seqLenKv = 300;
    spec.kvHeads = 2;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, ComputesSingleTokenDecodeOverALongCache)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.seqLenQ = 1;
    spec.seqLenKv = 513;
    spec.kvHeads = 1;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, ReadsARuntimeScaleAtExecute)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.scaleKind = SdpaScale::RUNTIME_TENSOR;
    spec.scale = 0.07F;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, WritesTheLogSumExpOfEachRow)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 150;
    spec.seqLenKv = 150;
    spec.stats = SdpaStats::LSE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, AppliesASlidingWindowNarrowerThanATile)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.mask = SdpaMask::SLIDING_WINDOW;
    spec.leftBound = 37;
    spec.seqLenQ = 300;
    spec.seqLenKv = 300;
    spec.kvHeads = 2;
    spec.stats = SdpaStats::LSE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, AppliesALeftOnlyWindowWithTheNonCausalObject)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.leftBound = 64;
    spec.seqLenQ = 129;
    spec.seqLenKv = 129;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, AppliesABandToTheRightOfTheDiagonal)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.rightBound = 5;
    spec.seqLenQ = 200;
    spec.seqLenKv = 200;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, ZeroesRowsNoKeyReaches)
{
    SKIP_IF_NO_DEVICES();
    // Bottom-right causal with more queries than keys: the first Sq - Skv rows see none.
    SdpaGraphSpec spec;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.seqLenQ = 300;
    spec.seqLenKv = 64;
    spec.stats = SdpaStats::LSE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, ComputesHeadDim96)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.headDim = 96;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 113;
    spec.seqLenKv = 113;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
}

// head_dim 256 runs a schedule of its own: Q re-read per K-step and the output columns
// split across two workgroups, so the grid doubles. Cover both variants, GQA, a ragged
// length, the LSE output and a bottom-right decode step against the reference.
TEST(TestGpuFlydslSdpaRdnaDispatch, ComputesHeadDim256CausalGroupedQueryAttention)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.headDim = 256;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 150;
    spec.seqLenKv = 150;
    spec.stats = SdpaStats::LSE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, ComputesHeadDim256NonCausalBhsd)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.layout = SdpaLayout::BHSD;
    spec.headDim = 256;
    spec.seqLenQ = 70;
    spec.seqLenKv = 200;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, ComputesHeadDim256BottomRightDecode)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.headDim = 256;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.batch = 2;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 1;
    spec.seqLenKv = 500;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
}

// An additive f32 bias, added to the scaled scores before the mask, in each broadcast shape
// the reference reads: one [Sq, Skv] plane for every batch and head, one per head, a rank-2
// tensor read through a transposed view, and with a window and the LSE output.
TEST(TestGpuFlydslSdpaRdnaDispatch, AddsABiasBroadcastOverBatchAndHeads)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.withBias = true;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 77;
    spec.seqLenKv = 77;
    spec.stats = SdpaStats::LSE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, AddsABiasInTheQueryDtype)
{
    SKIP_IF_NO_DEVICES();
    // What PyTorch passes as a float attn_mask: the query's dtype. Prefill at head_dim 64 and
    // 128, and a decode.
    SdpaGraphSpec spec;
    spec.withBias = true;
    spec.biasDataType = DataType::BFLOAT16;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.seqLenQ = 77;
    spec.seqLenKv = 99;
    spec.biasDims = {1, spec.heads, spec.seqLenQ, spec.seqLenKv};
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
    spec.dataType = DataType::HALF;
    spec.biasDataType = DataType::HALF;
    spec.headDim = 128;
    spec.biasDims = {spec.seqLenKv};
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
    spec.seqLenQ = 1;
    spec.seqLenKv = 300;
    spec.biasDims = {spec.seqLenKv};
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, AddsAPerHeadBiasAtHeadDim128)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.withBias = true;
    spec.dataType = DataType::HALF;
    spec.layout = SdpaLayout::BHSD;
    spec.headDim = 128;
    spec.seqLenQ = 70;
    spec.seqLenKv = 150;
    spec.biasDims = {1, spec.heads, spec.seqLenQ, spec.seqLenKv};
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, AddsARankTwoTransposedBiasAtHeadDim256)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.withBias = true;
    spec.headDim = 256;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.seqLenQ = 90;
    spec.seqLenKv = 130;
    spec.biasDims = {spec.seqLenQ, spec.seqLenKv};
    spec.biasStrides = {1, spec.seqLenQ};
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, AddsABiasWithinASlidingWindow)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.withBias = true;
    spec.headDim = 96;
    spec.mask = SdpaMask::SLIDING_WINDOW;
    spec.seqLenQ = 200;
    spec.seqLenKv = 200;
    spec.biasDims = {spec.batch, spec.heads, spec.seqLenQ, spec.seqLenKv};
    spec.stats = SdpaStats::LSE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
}

// A head_dim no specialized object is built for runs on the generic tier, which reads it at
// runtime from tensors that really are that wide: any read past a row would show up here.
TEST(TestGpuFlydslSdpaRdnaDispatch, ComputesAGenericHeadDim80)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.headDim = 80;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 133;
    spec.seqLenKv = 133;
    spec.stats = SdpaStats::LSE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, ComputesAGenericHeadDim40Bhsd)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.layout = SdpaLayout::BHSD;
    spec.headDim = 40;
    spec.seqLenQ = 70;
    spec.seqLenKv = 190;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
}

// The up-to-160 tier: a 160-wide K/V row does not tile the block's threads, so each KV tile
// is loaded in passes whose last one overruns the tile and stores only the rows inside it.
TEST(TestGpuFlydslSdpaRdnaDispatch, ComputesAGenericHeadDim152)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.headDim = 152;
    spec.heads = 4;
    spec.kvHeads = 4;
    spec.seqLenQ = 150;
    spec.seqLenKv = 301;
    spec.stats = SdpaStats::LSE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, ComputesAGenericHeadDim192)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.headDim = 192;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.seqLenQ = 90;
    spec.seqLenKv = 150;
    spec.kvHeads = 2;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
}

// Past 224 the widest tier serves, its output columns split four ways.
TEST(TestGpuFlydslSdpaRdnaDispatch, ComputesAGenericHeadDim240)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.headDim = 240;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.heads = 4;
    spec.kvHeads = 2;
    spec.seqLenQ = 140;
    spec.seqLenKv = 140;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
}

// K and V with different head counts: each query head reads its own K head and V head.
TEST(TestGpuFlydslSdpaRdnaDispatch, ReadsKAndVThroughTheirOwnHeadGroups)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.valueHeads = 4;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.seqLenQ = 100;
    spec.seqLenKv = 100;
    spec.stats = SdpaStats::LSE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
}

// Q, K and V as strided views into one fused-QKV buffer: no copies, just strides.
TEST(TestGpuFlydslSdpaRdnaDispatch, ComputesPackedQkvViews)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.packedQkv = true;
    spec.headDim = 128;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.seqLenQ = 150;
    spec.seqLenKv = 150;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
}

// The decode objects: the GQA group packed into one 16-row tile, KV split across
// workgroups and merged. One KV head across a long cache splits several ways; short ones
// write O directly. Windows, several query positions and LSE run through both paths.
TEST(TestGpuFlydslSdpaRdnaDispatch, DecodesOneTokenAcrossSplits)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.headDim = 128;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.batch = 1;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 1;
    spec.seqLenKv = 2100;
    spec.stats = SdpaStats::LSE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, DecodesSeveralPositionsInOneSplit)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.headDim = 64;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 4;
    spec.seqLenKv = 300;
    spec.stats = SdpaStats::LSE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE, true);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, DecodesWithinASlidingWindow)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.headDim = 96;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.leftBound = 127;
    spec.batch = 1;
    spec.heads = 16;
    spec.kvHeads = 1;
    spec.seqLenQ = 1;
    spec.seqLenKv = 3000;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
}

// More rows than one tile holds: MQA's group of 32, and a speculative step of several
// positions, each split across row tiles along the grid's second dimension.
TEST(TestGpuFlydslSdpaRdnaDispatch, DecodesAnMqaGroupAcrossRowTiles)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.headDim = 128;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.batch = 2;
    spec.heads = 32;
    spec.kvHeads = 1;
    spec.seqLenQ = 1;
    spec.seqLenKv = 2100;
    spec.stats = SdpaStats::LSE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, DecodesASpeculativeStepAcrossRowTiles)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.headDim = 64;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 7;
    spec.seqLenKv = 900;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE, true);
}

// A head_dim off the specialized set on the decode path: columns past it are never read,
// and the split workspace's rows are that wide.
TEST(TestGpuFlydslSdpaRdnaDispatch, DecodesAGenericHeadDim80AcrossSplits)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.headDim = 80;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.batch = 1;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 1;
    spec.seqLenKv = 2100;
    spec.stats = SdpaStats::LSE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, DecodesAGenericHeadDim160)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.headDim = 160;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.heads = 8;
    spec.kvHeads = 4;
    spec.seqLenQ = 2;
    spec.seqLenKv = 700;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE, true);
}

// A bias on the decode path: each lane reads its own (head, position) row of the GQA
// group's slice, broadcast over the batch here, across splits and with LSE.
TEST(TestGpuFlydslSdpaRdnaDispatch, DecodesWithABiasAcrossSplits)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.headDim = 128;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.batch = 2;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 2;
    spec.seqLenKv = 2100;
    spec.stats = SdpaStats::LSE;
    spec.withBias = true;
    spec.biasDims = {1, spec.heads, spec.seqLenQ, spec.seqLenKv};
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
}

// d256 splits its output columns across two workgroups per KV split; LSE is written by
// one of them, and the merge combines the partials across the full head.
TEST(TestGpuFlydslSdpaRdnaDispatch, DecodesHeadDim256AcrossSplitsAndColumnTiles)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.headDim = 256;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.batch = 1;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 2;
    spec.seqLenKv = 2500;
    spec.stats = SdpaStats::LSE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE, true);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, AppliesTheDefaultScaleWhenTheGraphGivesNone)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.scaleKind = SdpaScale::ABSENT;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
