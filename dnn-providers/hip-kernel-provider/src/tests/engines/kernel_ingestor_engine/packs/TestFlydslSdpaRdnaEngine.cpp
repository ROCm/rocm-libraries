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
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
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
#include "engines/kernel_ingestor_engine/packs/FlydslNormDeviceTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaObjectTestSupport.hpp"
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

/// The engine whose objects a test's kernel stands for: the decode kernel's, or the prefill
/// kernel's. One kernel family each, sharing the graph matcher and its bindings.
const PackSymbols& engineFor(bool decode)
{
    return decode ? FLYDSL_SDPA_RDNA_DECODE : FLYDSL_SDPA_RDNA;
}

/// A kernel carrying exactly the metadata its engine's KMD declares: a decode object (@p decode
/// nonzero) names its merge kernel and has no hd_pair; a prefill object the reverse.
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
                                          int64_t biasQ = 0,
                                          std::optional<int64_t> headDimV = std::nullopt,
                                          int64_t hasSink = 0)
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
                       {FLYDSL_SDPA_RDNA_BIAS_Q_FIELD, biasQ},
                       {FLYDSL_SDPA_RDNA_HEAD_DIM_V_FIELD, headDimV.value_or(headDim)},
                       {FLYDSL_SDPA_RDNA_HAS_SINK_FIELD, hasSink}};
    if(decode != 0)
    {
        kernel.metadata[FLYDSL_SDPA_RDNA_MERGE_SYMBOL_FIELD]
            = std::string("flash_attn_decode_merge_rdna_kernel_1");
    }
    else
    {
        kernel.metadata[FLYDSL_SDPA_RDNA_HD_PAIR_FIELD] = hdPair;
    }
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
        // Multi-head latent attention, un-absorbed: V narrower than Q and K.
        AcceptedCase{"ValueHeadDimNarrower", accepted([](SdpaGraphSpec& s) {
                         s.headDim = 192;
                         s.valueHeadDim = 128;
                     })},
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
        // An attention sink per query head, rank 1 and rank 4.
        AcceptedCase{"AttentionSink", accepted([](SdpaGraphSpec& s) { s.withSink = true; })},
        AcceptedCase{"AttentionSinkRank4", accepted([](SdpaGraphSpec& s) {
                         s.withSink = true;
                         s.sinkRank4 = true;
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
        AcceptedCase{"DeprecatedTopLeftCausal",
                     accepted([](SdpaGraphSpec& s) { s.deprecatedCausalTopLeft = true; })},
        AcceptedCase{"DeprecatedBottomRightCausalWithItsOwnRightBound",
                     accepted([](SdpaGraphSpec& s) {
                         s.deprecatedCausalBottomRight = true;
                         s.rightBound = 0;
                     })},
        // A deprecated boolean with a left_bound keeps the window (gpt-oss SWA graphs), and
        // an explicit right_bound wins over the boolean's 0.
        AcceptedCase{"DeprecatedCausalWithAWindow", accepted([](SdpaGraphSpec& s) {
                         s.deprecatedCausalBottomRight = true;
                         s.leftBound = 32;
                     })},
        AcceptedCase{"DeprecatedCausalWithABand", accepted([](SdpaGraphSpec& s) {
                         s.deprecatedCausalTopLeft = true;
                         s.rightBound = 16;
                     })},
        AcceptedCase{"SoftmaxStats", accepted([](SdpaGraphSpec& s) { s.stats = SdpaStats::LSE; })},
        AcceptedCase{"ScaleAbsent",
                     accepted([](SdpaGraphSpec& s) { s.scaleKind = SdpaScale::ABSENT; })},
        AcceptedCase{"ScaleConstantTensor",
                     accepted([](SdpaGraphSpec& s) { s.scaleKind = SdpaScale::CONSTANT_TENSOR; })},
        AcceptedCase{"ScaleRuntimeTensor",
                     accepted([](SdpaGraphSpec& s) { s.scaleKind = SdpaScale::RUNTIME_TENSOR; })},
        // Every value a field can take that the kernels do handle, beside the declines.
        AcceptedCase{"ComputeTypeUnset",
                     accepted([](SdpaGraphSpec& s) { s.computeDataType = DataType::UNSET; })},
        AcceptedCase{"ComputeTypeHalf",
                     accepted([](SdpaGraphSpec& s) { s.computeDataType = DataType::HALF; })},
        AcceptedCase{"ComputeTypeBfloat16",
                     accepted([](SdpaGraphSpec& s) { s.computeDataType = DataType::BFLOAT16; })},
        AcceptedCase{"OperandsAlignedTo256", accepted([](SdpaGraphSpec& s) {
                         for(const auto uid : {FLYDSL_SDPA_RDNA_Q_UID,
                                               FLYDSL_SDPA_RDNA_K_UID,
                                               FLYDSL_SDPA_RDNA_V_UID,
                                               FLYDSL_SDPA_RDNA_O_UID})
                         {
                             s.alignment[uid] = 256;
                         }
                     })},
        AcceptedCase{"MmaCoreModeHalf",
                     accepted([](SdpaGraphSpec& s) { s.mmaCoreMode = DataType::HALF; })},
        AcceptedCase{"MmaCoreModeBfloat16",
                     accepted([](SdpaGraphSpec& s) { s.mmaCoreMode = DataType::BFLOAT16; })},
        AcceptedCase{"GenerateStatsFalseWithoutATensor",
                     accepted([](SdpaGraphSpec& s) { s.generateStats = false; })},
        AcceptedCase{"StatsTensorWithGenerateStatsUnset", accepted([](SdpaGraphSpec& s) {
                         s.stats = SdpaStats::LSE;
                         s.generateStats = std::nullopt;
                     })},
        AcceptedCase{"ScaleTensorOfRankFour", accepted([](SdpaGraphSpec& s) {
                         s.scaleKind = SdpaScale::RUNTIME_TENSOR;
                         s.scaleDims = {1, 1, 1, 1};
                     })},
        AcceptedCase{"ScaleTensorInHalf", accepted([](SdpaGraphSpec& s) {
                         s.scaleKind = SdpaScale::RUNTIME_TENSOR;
                         s.scaleDataType = DataType::HALF;
                     })}),
    [](const ::testing::TestParamInfo<AcceptedCase>& info) { return info.param.name; });

// ---------------------------------------------------------------------------
// graph_match: every field, each value no kernel handles declined by all three engines that
// share this matcher
// ---------------------------------------------------------------------------

class TestFlydslSdpaRdnaFieldDeclines : public ::testing::TestWithParam<SdpaFieldDecline>
{
};

TEST_P(TestFlydslSdpaRdnaFieldDeclines, DeclinesTheField)
{
    SdpaGraphSpec spec;
    {
        const GraphFixture base(buildFlydslSdpaRdnaGraph(spec));
        ASSERT_TRUE(matchesGraph(FLYDSL_SDPA_RDNA, base.context()).has_value())
            << "the spec the case edits must be accepted on its own";
    }
    if(GetParam().control)
    {
        auto control = spec;
        GetParam().control(control);
        const GraphFixture accepted(buildFlydslSdpaRdnaGraph(control));
        ASSERT_TRUE(matchesGraph(FLYDSL_SDPA_RDNA, accepted.context()).has_value())
            << "the case's feature without its defect must be accepted";
    }
    GetParam().apply(spec);
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    for(const auto* engine : {&FLYDSL_SDPA_RDNA, &FLYDSL_SDPA_RDNA_DECODE, &FLYDSL_SDPA_AIW})
    {
        EXPECT_FALSE(matchesGraph(*engine, fixture.context()).has_value()) << engine->engineName;
    }
}

INSTANTIATE_TEST_SUITE_P(,
                         TestFlydslSdpaRdnaFieldDeclines,
                         ::testing::ValuesIn(sdpaFieldDeclines()),
                         [](const auto& info) { return info.param.name; });

/// The shared pass-by-value table, plus the bias in the shape a host scalar has: rank 1, one
/// element. The RDNA engines accept that bias as a device tensor (it broadcasts over every
/// axis); the CDNA engines decline it for its broadcast key axis, so it is RDNA's case only.
std::vector<SdpaFieldDecline> passByValueDeclines()
{
    auto cases = sdpaPassByValueDeclines();
    const auto scalarBias = [](SdpaGraphSpec& s) {
        s.withBias = true;
        s.biasDataType = s.dataType;
        s.biasDims = {1};
        s.biasStrides = {1};
    };
    cases.push_back({"PassByValueScalarBias",
                     [scalarBias](SdpaGraphSpec& s) {
                         scalarBias(s);
                         s.passByValueTensors.insert(FLYDSL_SDPA_RDNA_BIAS_UID);
                     },
                     scalarBias});
    return cases;
}

INSTANTIATE_TEST_SUITE_P(PassByValue,
                         TestFlydslSdpaRdnaFieldDeclines,
                         ::testing::ValuesIn(passByValueDeclines()),
                         [](const auto& info) { return info.param.name; });

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
        // V's head dim may be narrower than Q/K's (MLA), never wider.
        AcceptedCase{"ValueHeadDimWider", accepted([](SdpaGraphSpec& s) { s.valueHeadDim = 128; })},
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
        AcceptedCase{"BothDeprecatedBooleans", accepted([](SdpaGraphSpec& s) {
                         s.deprecatedCausalTopLeft = true;
                         s.deprecatedCausalBottomRight = true;
                     })},
        AcceptedCase{"UnknownDiagonalAlignment", accepted([](SdpaGraphSpec& s) {
                         s.mask = SdpaMask::TOP_LEFT_CAUSAL;
                         s.diagonalAlignment
                             = static_cast<hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment>(
                                 7);
                     })},
        // A bound below -1 has no meaning in the reference's convention.
        AcceptedCase{"LeftBoundBelowMinusOne",
                     accepted([](SdpaGraphSpec& s) { s.leftBound = -2; })},
        // left_bound = 0 keeps no key at or before the diagonal (cuDNN's convention), and the
        // kernel's window, one less, would read it as unbounded.
        AcceptedCase{"LeftBoundZero", accepted([](SdpaGraphSpec& s) { s.leftBound = 0; })},
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
        // The kernels read a sink in the query's dtype.
        AcceptedCase{"SinkInAnotherDtype", accepted([](SdpaGraphSpec& s) {
                         s.withSink = true;
                         s.sinkDataType = DataType::FLOAT;
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

TEST(TestFlydslSdpaRdnaBinding, TheDeprecatedBottomRightBooleanIsCausalAtTheBottomRight)
{
    // The graph's alignment field says top-left; the boolean overrides it, as in the
    // reference.
    SdpaGraphSpec spec;
    spec.deprecatedCausalBottomRight = true;
    spec.seqLenQ = 64;
    spec.seqLenKv = 300;
    const auto bound = bindingsFor(spec);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_RDNA_CAUSAL_TOKEN), 1);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_RDNA_RIGHT_BOUND_TOKEN), 0);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_RDNA_ALIGN_BOTTOM_RIGHT_TOKEN), 1);
    // With a left_bound too, the window is kept (cuDNN and #13065), not dropped, as the
    // kernel's window: one less than the graph's bound.
    spec.leftBound = 32;
    EXPECT_EQ(boundInt(bindingsFor(spec), FLYDSL_SDPA_RDNA_LEFT_BOUND_TOKEN), 31);
}

TEST(TestFlydslSdpaRdnaBinding, TheDeprecatedTopLeftBooleanIsCausalAtTheStoredAlignment)
{
    // causal_mask is right 0 at the alignment the graph stores: top-left by default, and
    // bottom-right when diagonal_alignment says so, as the reference resolves it.
    SdpaGraphSpec spec;
    spec.deprecatedCausalTopLeft = true;
    spec.seqLenQ = 64;
    spec.seqLenKv = 300;
    const auto topLeft = bindingsFor(spec);
    EXPECT_EQ(boundInt(topLeft, FLYDSL_SDPA_RDNA_CAUSAL_TOKEN), 1);
    EXPECT_EQ(boundInt(topLeft, FLYDSL_SDPA_RDNA_RIGHT_BOUND_TOKEN), 0);
    EXPECT_EQ(boundInt(topLeft, FLYDSL_SDPA_RDNA_ALIGN_BOTTOM_RIGHT_TOKEN), 0);
    spec.diagonalAlignment = hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    const auto bottomRight = bindingsFor(spec);
    EXPECT_EQ(boundInt(bottomRight, FLYDSL_SDPA_RDNA_CAUSAL_TOKEN), 1);
    EXPECT_EQ(boundInt(bottomRight, FLYDSL_SDPA_RDNA_RIGHT_BOUND_TOKEN), 0);
    EXPECT_EQ(boundInt(bottomRight, FLYDSL_SDPA_RDNA_ALIGN_BOTTOM_RIGHT_TOKEN), 1);
    // A right_bound the graph sets wins over the boolean's 0: a band, not plain causal.
    spec.diagonalAlignment.reset();
    spec.rightBound = 16;
    const auto band = bindingsFor(spec);
    EXPECT_EQ(boundInt(band, FLYDSL_SDPA_RDNA_CAUSAL_TOKEN), 1);
    EXPECT_EQ(boundInt(band, FLYDSL_SDPA_RDNA_RIGHT_BOUND_TOKEN), 16);
    EXPECT_EQ(boundInt(band, FLYDSL_SDPA_RDNA_ALIGN_BOTTOM_RIGHT_TOKEN), 0);
}

TEST(TestFlydslSdpaRdnaBinding, AWindowBindsItsLeftBoundAndClampsAnOversizedOne)
{
    // left_bound = L keeps the L keys ending at the diagonal (cuDNN); the kernel keeps
    // left_bound + 1, so it is bound one less, and 1 binds 0, the diagonal alone.
    SdpaGraphSpec spec;
    spec.mask = SdpaMask::SLIDING_WINDOW;
    EXPECT_EQ(boundInt(bindingsFor(spec), FLYDSL_SDPA_RDNA_LEFT_BOUND_TOKEN), 63);
    spec.leftBound = 1;
    EXPECT_EQ(boundInt(bindingsFor(spec), FLYDSL_SDPA_RDNA_LEFT_BOUND_TOKEN), 0);

    // Past Sq + Skv a window masks nothing more, and the kernel's int32 sums need the
    // bound kept in range.
    spec.leftBound = int64_t{1} << 40;
    EXPECT_EQ(boundInt(bindingsFor(spec), FLYDSL_SDPA_RDNA_LEFT_BOUND_TOKEN),
              spec.seqLenQ + spec.seqLenKv);
}

TEST(TestFlydslSdpaRdnaBinding, ClampsAnOversizedRightBoundPastBothLengths)
{
    // Bottom-right with more queries than keys: the diagonal starts at Skv - Sq, so a
    // bound clamped to Skv would hide keys the first rows reach.
    SdpaGraphSpec spec;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.seqLenQ = 300;
    spec.seqLenKv = 64;
    spec.rightBound = 250;
    EXPECT_EQ(boundInt(bindingsFor(spec), FLYDSL_SDPA_RDNA_RIGHT_BOUND_TOKEN), 250);

    spec.rightBound = int64_t{1} << 40;
    EXPECT_EQ(boundInt(bindingsFor(spec), FLYDSL_SDPA_RDNA_RIGHT_BOUND_TOKEN),
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

TEST(TestFlydslSdpaRdnaBinding, AnAbsentScaleDefaultsToOne)
{
    // cuDNN's default, which hipDNN's CPU reference applies too (CpuFpReferenceSdpa.hpp).
    SdpaGraphSpec spec;
    spec.scaleKind = SdpaScale::ABSENT;
    spec.headDim = 128;
    const auto bound = bindingsFor(spec);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_RDNA_SCALE_SOURCE_TOKEN), FLYDSL_SDPA_RDNA_SCALE_BAKED);
    EXPECT_FLOAT_EQ(boundScale(bound), 1.0F);
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

TEST(TestFlydslSdpaRdnaKernelMatch, AnMlaObjectServesOnlyItsHeadDimPair)
{
    SdpaGraphSpec mla;
    mla.headDim = 192;
    mla.valueHeadDim = 128;
    const GraphFixture narrowV(buildFlydslSdpaRdnaGraph(mla));
    const auto narrowBound = bindingsFor(mla);
    SdpaGraphSpec square;
    square.headDim = 192;
    const GraphFixture sameV(buildFlydslSdpaRdnaGraph(square));
    const auto squareBound = bindingsFor(square);

    const auto mlaKernel
        = makeFlydslSdpaRdnaKernel("bf16", 192, 0, 64, 32, 1, 0, 192, 0, 1, 0, int64_t{128});
    const auto generic
        = makeFlydslSdpaRdnaKernel("bf16", 0, 0, 64, 32, 1, 0, int64_t{224}, 0, 1, 0, int64_t{0});
    EXPECT_TRUE(matchesKernel(FLYDSL_SDPA_RDNA, narrowV.context(), mlaKernel, narrowBound));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, sameV.context(), mlaKernel, squareBound));
    // A generic object's V is its runtime head_dim, so it serves only the square graph.
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, narrowV.context(), generic, narrowBound));
    EXPECT_TRUE(matchesKernel(FLYDSL_SDPA_RDNA, sameV.context(), generic, squareBound));
}

TEST(TestFlydslSdpaRdnaKernelMatch, ADecodeObjectServesEveryRowCountItsGridHolds)
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
        return matchesKernel(FLYDSL_SDPA_RDNA_DECODE, fixture.context(), decode, bindingsFor(spec));
    };

    EXPECT_TRUE(matches(32, 8, 1)) << "group of 4, one position";
    EXPECT_TRUE(matches(16, 1, 1)) << "MQA, group of 16";
    EXPECT_TRUE(matches(8, 2, 4)) << "group of 4, four positions";
    EXPECT_TRUE(matches(32, 1, 1)) << "MQA, group of 32: two row tiles";
    EXPECT_TRUE(matches(8, 2, 32)) << "group of 4, 32 positions: 128 rows";
    // Past a few tiles the prefill engine is usually faster, but which engine runs a graph
    // both serve is the heuristic's to say: this one declines only what it cannot compute.
    EXPECT_TRUE(matches(8, 2, 33)) << "132 rows";
    EXPECT_TRUE(matches(64, 1, 3)) << "MQA, group of 64, three positions: 192 rows";
    EXPECT_FALSE(matches(64, 1, 16384)) << "65536 row tiles: past the grid's second dimension";
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
    EXPECT_FALSE(
        matchesKernel(FLYDSL_SDPA_RDNA_DECODE, fixture.context(), decode, bindingsFor(spec)))
        << "a tile shares one KV head, so K and V must group alike";

    auto unnamed = decode;
    unnamed.metadata[FLYDSL_SDPA_RDNA_MERGE_SYMBOL_FIELD] = std::string();
    spec.valueHeads.reset();
    const GraphFixture plain(buildFlydslSdpaRdnaGraph(spec));
    EXPECT_FALSE(
        matchesKernel(FLYDSL_SDPA_RDNA_DECODE, plain.context(), unnamed, bindingsFor(spec)))
        << "a decode object must name its merge kernel";
}

TEST(TestFlydslSdpaRdnaKernelMatch, ADecodeObjectDeclinesAnOutputItsMergeCannotAddress)
{
    // The merge kernel stores O and the LSE at one 32-bit element offset over the whole
    // tensor: an O whose second batch starts past 2^31 elements is declined, one just
    // inside is served.
    const auto decode = makeFlydslSdpaRdnaKernel("bf16", 128, 1, 16, 32, 1, 0, 128, 1);
    SdpaGraphSpec spec;
    spec.headDim = 128;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 1;
    spec.seqLenKv = 300;
    spec.outputBatchStride = (int64_t{1} << 31) - 4096;
    {
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
        EXPECT_TRUE(
            matchesKernel(FLYDSL_SDPA_RDNA_DECODE, fixture.context(), decode, bindingsFor(spec)))
            << "O's last element just inside int32";
    }
    spec.outputBatchStride = int64_t{1} << 31;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    EXPECT_FALSE(
        matchesKernel(FLYDSL_SDPA_RDNA_DECODE, fixture.context(), decode, bindingsFor(spec)))
        << "O's second batch past int32";
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
    EXPECT_TRUE(matchesKernel(FLYDSL_SDPA_RDNA_DECODE, biased.context(), bias, bindingsFor(spec)));
    EXPECT_FALSE(
        matchesKernel(FLYDSL_SDPA_RDNA_DECODE, biased.context(), plain, bindingsFor(spec)));

    // A bias object is built for one 16-row tile, so it declines an MQA group of 32.
    SdpaGraphSpec mqa = spec;
    mqa.heads = 32;
    mqa.kvHeads = 1;
    const GraphFixture wide(buildFlydslSdpaRdnaGraph(mqa));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA_DECODE, wide.context(), bias, bindingsFor(mqa)));

    // One tile spans the four heads of a group; a head stride that puts the last of them
    // past the 32-bit offsets declines, though each (batch, head) slice alone fits.
    spec.biasDims = {1, spec.heads, 1, spec.seqLenKv};
    spec.biasStrides = {spec.heads * (int64_t{1} << 29), int64_t{1} << 29, spec.seqLenKv, 1};
    const GraphFixture farHeads(buildFlydslSdpaRdnaGraph(spec));
    EXPECT_FALSE(
        matchesKernel(FLYDSL_SDPA_RDNA_DECODE, farHeads.context(), bias, bindingsFor(spec)));
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

TEST(TestFlydslSdpaRdnaKernelMatch, PairsTheSinkVariantWithAGraphThatHasASink)
{
    SdpaGraphSpec sunk;
    sunk.withSink = true;
    const GraphFixture withSink(buildFlydslSdpaRdnaGraph(sunk));
    const auto sinkBound = bindingsFor(sunk);
    const GraphFixture without(buildFlydslSdpaRdnaGraph());
    const auto plainBound = bindingsFor();

    const auto plain = makeFlydslSdpaRdnaKernel("bf16", 64, 0);
    const auto sink
        = makeFlydslSdpaRdnaKernel("bf16", 64, 0, 128, 32, 1, 0, 64, 0, 0, 0, std::nullopt, 1);
    EXPECT_TRUE(matchesKernel(FLYDSL_SDPA_RDNA, withSink.context(), sink, sinkBound));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, withSink.context(), plain, sinkBound));
    EXPECT_TRUE(matchesKernel(FLYDSL_SDPA_RDNA, without.context(), plain, plainBound));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, without.context(), sink, plainBound));
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

    kernel = makeFlydslSdpaRdnaKernel("bf16", 64, 0);
    kernel.metadata.erase(FLYDSL_SDPA_RDNA_HAS_SINK_FIELD);
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

// The decode kernel's engine: the same seams, through its own symbols.
TEST(TestFlydslSdpaRdnaDecodeDispatch, ResolvesFromTheDispatchRegistry)
{
    EXPECT_NO_THROW(static_cast<void>(dispatchHandler(FLYDSL_SDPA_RDNA_DECODE)));
}

TEST(TestFlydslSdpaRdnaDecodeDispatch, SharesThePrefillEnginesGraphMatch)
{
    SdpaGraphSpec spec;
    spec.seqLenQ = 1;
    spec.seqLenKv = 300;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    const auto prefill = matchesGraph(FLYDSL_SDPA_RDNA, fixture.context());
    const auto decode = matchesGraph(FLYDSL_SDPA_RDNA_DECODE, fixture.context());
    ASSERT_TRUE(prefill.has_value());
    ASSERT_TRUE(decode.has_value());
    EXPECT_TRUE(*prefill == *decode) << "the two engines bind a graph alike";
}

TEST(TestFlydslSdpaRdnaDecodeDispatch, ScoresByDescriptorPriority)
{
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph());
    auto kernel = makeFlydslSdpaRdnaKernel("bf16", 128, 1, 16, 32, 1, 0, 128, 1);
    kernel.priority = 41;
    EXPECT_EQ(scoreKernel(FLYDSL_SDPA_RDNA_DECODE, fixture.context(), kernel), 41.0);
}

TEST(TestFlydslSdpaRdnaDecodeDispatch, RefusesToPrepareWithoutTheMatcherSBindings)
{
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph());
    EXPECT_THROW(dispatchHandler(FLYDSL_SDPA_RDNA_DECODE)
                     .prepare(fixture.context(),
                              BoundTokens{},
                              makeFlydslSdpaRdnaKernel("bf16", 128, 1, 16, 32, 1, 0, 128, 1)),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestFlydslSdpaRdnaDecodeDispatch, SizesTheSplitWorkspaceForEachPartialAndItsMaxAndSum)
{
    // The launch lays the split workspace out as the f32 partial O, Dv per (batch, query
    // head, query position, split), then each split row's max and sum: B * H * Sq * splits *
    // (Dv + 2) f32s, and the caller allocates what workspaceBytes reports. The split count is
    // the engine's own choice, so the size must be a whole number of splits of that row,
    // between 2 and 64. One stat short per row is not: Dv + 1 and Dv + 2 are coprime and
    // Dv + 2 > 64.
    struct Shape
    {
        int64_t batch;
        int64_t heads;
        int64_t kvHeads;
        int64_t seqLenQ;
        int64_t seqLenKv;
        int64_t headDim;
        int computeUnits;
    };
    for(const auto& shape : {Shape{1, 8, 2, 1, 4096, 128, 0},
                             Shape{2, 16, 2, 2, 20000, 64, 0},
                             Shape{1, 1, 1, 1, 2049, 128, 40},
                             Shape{3, 4, 4, 3, 70000, 64, 60}})
    {
        SCOPED_TRACE("B " + std::to_string(shape.batch) + ", H " + std::to_string(shape.heads)
                     + ", Sq " + std::to_string(shape.seqLenQ) + ", Skv "
                     + std::to_string(shape.seqLenKv) + ", D " + std::to_string(shape.headDim));
        SdpaGraphSpec spec;
        spec.batch = shape.batch;
        spec.heads = shape.heads;
        spec.kvHeads = shape.kvHeads;
        spec.seqLenQ = shape.seqLenQ;
        spec.seqLenKv = shape.seqLenKv;
        spec.headDim = shape.headDim;
        auto properties = testDeviceProperties();
        properties.multiProcessorCount = shape.computeUnits;
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec), properties);
        const auto bound = matchesGraph(FLYDSL_SDPA_RDNA_DECODE, fixture.context());
        ASSERT_TRUE(bound.has_value());
        const auto kernel
            = makeFlydslSdpaRdnaKernel("bf16", shape.headDim, 0, 16, 32, 1, 0, shape.headDim, 1);
        ASSERT_TRUE(matchesKernel(FLYDSL_SDPA_RDNA_DECODE, fixture.context(), kernel, *bound));
        const auto bytes = dispatchHandler(FLYDSL_SDPA_RDNA_DECODE)
                               .workspaceBytes(fixture.context(), *bound, kernel);
        const auto perSplit
            = static_cast<size_t>(shape.batch * shape.heads * shape.seqLenQ * (shape.headDim + 2))
              * sizeof(float);
        EXPECT_EQ(bytes % perSplit, 0U) << bytes << " bytes";
        EXPECT_GE(bytes / perSplit, 2U) << bytes << " bytes";
        EXPECT_LE(bytes / perSplit, 64U) << bytes << " bytes";
    }
}

// ---------------------------------------------------------------------------
// dispatch on device: a real launch out of the staged archive
// ---------------------------------------------------------------------------

// packedArchDeviceProperties() and reportUnstaged() come from FlydslNormDeviceTestSupport.hpp:
// a case on a device this build ships the engine's objects for fails when it finds none
// staged, rather than skipping.

/// The shipped-arch list (HIPKERNELPROVIDER_FLYDSL_ARCHS_<op>) each engine's objects are
/// built for, by the op name the build gives it.
const char* shippedOpFor(const PackSymbols& engine)
{
    if(&engine == &FLYDSL_SDPA_AIW)
    {
        return "sdpa_aiw";
    }
    return &engine == &FLYDSL_SDPA_RDNA_DECODE ? "sdpa_rdna_decode" : "sdpa_rdna";
}

/// The staged candidate for (@p dtype, @p headDim, @p causal, @p hasBias), if this build
/// packed one: the object built for that head_dim, else the narrowest generic one serving it.
std::optional<KernelDefinition> findStagedKernel(const std::string& dtype,
                                                 int64_t headDim,
                                                 int64_t causal,
                                                 int64_t hasBias,
                                                 int64_t decode = 0,
                                                 int64_t biasQ = 0,
                                                 std::optional<int64_t> headDimV = std::nullopt,
                                                 int64_t hasSink = 0)
{
    // A build for CDNA targets only stages no RDNA set at all.
    const auto* staged = findLoadedSet(engineFor(decode != 0).engineName);
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
            const auto* inQueryType
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SDPA_RDNA_BIAS_Q_FIELD);
            const auto* dimV
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SDPA_RDNA_HEAD_DIM_V_FIELD);
            const auto* sink
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SDPA_RDNA_HAS_SINK_FIELD);
            if(sink == nullptr || *sink != hasSink)
            {
                continue;
            }
            if(name == nullptr || dim == nullptr || isCausal == nullptr || bias == nullptr
               || dimMax == nullptr || inQueryType == nullptr || dimV == nullptr || *name != dtype
               || *isCausal != causal || *bias != hasBias || *inQueryType != biasQ
               || (*dimV == 0 ? headDimV.value_or(headDim) != headDim
                              : *dimV != headDimV.value_or(headDim)))
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
                                        kernel.treeRoot,
                                        kernel.sources};
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

    /// Sets element (b, h, s, d) on the host; upload() copies it to the device.
    void set(int64_t b, int64_t h, int64_t s, int64_t d, T value)
    {
        const auto index = b * _strides[0] + h * _strides[1] + s * _strides[2] + d * _strides[3];
        _host[static_cast<size_t>(index)] = value;
    }

    /// Sets every element, so an element the kernel never writes reads back as @p value.
    void fillWith(T value)
    {
        std::fill(_host.begin(), _host.end(), value);
        upload();
    }

    void upload()
    {
        EXPECT_EQ(hipSuccess, hipMemcpy(_device, _host.data(), bytes(), hipMemcpyHostToDevice));
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

/// The staged object of AOTriton's gfx1201 kernel for a graph, if this build packed one: the
/// object built for the head_dim, else the narrowest padded tier serving it. @p masked is
/// whether any bound applies (the object that applies a window), @p biasF32 whether the
/// mask is f32 rather than the query's dtype.
std::optional<KernelDefinition> findStagedAiwKernel(const std::string& dtype,
                                                    int64_t headDim,
                                                    int64_t masked,
                                                    int64_t hasBias,
                                                    int64_t biasF32,
                                                    std::optional<int64_t> headDimV,
                                                    int64_t hasSink)
{
    const auto* staged = findLoadedSet(FLYDSL_SDPA_AIW.engineName);
    if(staged == nullptr)
    {
        return std::nullopt;
    }
    const auto device = packedArchDeviceProperties().gcnArchName;
    std::optional<KernelDefinition> best;
    int64_t bestMax = 0;
    for(const auto& pack : staged->packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto& arches = kernel.arch.empty() ? pack.arch : kernel.arch;
            if(!device.empty() && !arches.empty()
               && std::find(arches.begin(), arches.end(), device) == arches.end())
            {
                continue;
            }
            const auto field = [&kernel](const char* name) {
                const auto* value = tryGetMetadataField<int64_t>(kernel.metadata, name);
                return value == nullptr ? int64_t{-1} : *value;
            };
            const auto* name
                = tryGetMetadataField<std::string>(kernel.metadata, FLYDSL_SDPA_RDNA_DTYPE_FIELD);
            const auto dim = field(FLYDSL_SDPA_RDNA_HEAD_DIM_FIELD);
            const auto dimMax = field(FLYDSL_SDPA_RDNA_HEAD_DIM_MAX_FIELD);
            const auto dimV = field(FLYDSL_SDPA_RDNA_HEAD_DIM_V_FIELD);
            if(name == nullptr || *name != dtype || field(FLYDSL_SDPA_RDNA_CAUSAL_FIELD) != masked
               || field(FLYDSL_SDPA_RDNA_HAS_BIAS_FIELD) != hasBias
               || (hasBias != 0 && field(FLYDSL_SDPA_AIW_BIAS_F32_FIELD) != biasF32)
               || field(FLYDSL_SDPA_RDNA_HAS_SINK_FIELD) != hasSink)
            {
                continue;
            }
            const bool exact = dim == headDim && dimV == headDimV.value_or(headDim);
            const bool padded
                = dim == 0 && !headDimV.has_value() && headDim % 8 == 0 && headDim <= dimMax;
            if(!exact && !padded)
            {
                continue;
            }
            KernelDefinition definition{kernel.id,
                                        pack.id,
                                        pack.dispatchId,
                                        kernel.source,
                                        kernel.metadata,
                                        kernel.priority,
                                        arches,
                                        kernel.originDirectory,
                                        kernel.name,
                                        kernel.treeRoot,
                                        kernel.sources};
            if(exact)
            {
                return definition;
            }
            if(!best.has_value() || dimMax < bestMax)
            {
                best = definition;
                bestMax = dimMax;
            }
        }
    }
    return best;
}

/// Which RDNA engine's staged object a GPU case runs.
enum class RdnaFamily
{
    PREFILL,
    DECODE,
    AIW,
};

/// Runs the staged kernel for @p spec through prepare() and launch(), and checks every
/// output element against a double-precision reference computed from the same inputs.
///
/// When the staged set holds no kernel for the spec's class this fails on a device the build
/// ships the engine's objects for, where the shard must hold one, and skips elsewhere: on an
/// arch this build packed nothing for, there is no kernel to be wrong about.
///
/// @p forced, when given, is launched instead of the object the metadata search picks: the
/// per-object suite (TestFlydslSdpaObjects.cpp) names each staged object itself.
template <typename T>
void expectSdpaMatchesReference(const SdpaGraphSpec& spec,
                                double absoluteTolerance,
                                RdnaFamily family,
                                const KernelDefinition* forced = nullptr);

template <typename T>
void expectSdpaMatchesReference(const SdpaGraphSpec& spec,
                                double absoluteTolerance,
                                bool decode = false)
{
    expectSdpaMatchesReference<T>(
        spec, absoluteTolerance, decode ? RdnaFamily::DECODE : RdnaFamily::PREFILL);
}

template <typename T>
void expectSdpaMatchesReference(const SdpaGraphSpec& spec,
                                double absoluteTolerance,
                                RdnaFamily family,
                                const KernelDefinition* forced)
{
    const bool decode = family == RdnaFamily::DECODE;
    // The bounds the graph builder writes for this spec, in CpuFpReferenceSdpa's terms: a
    // left bound L keeps the L keys ending at the diagonal, as cuDNN's does.
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
               || (left >= 0 && key <= row + align - left);
    };

    const std::string dtype = spec.dataType == DataType::BFLOAT16 ? "bf16" : "f16";
    const bool causal = right >= 0;
    const auto searched
        = family == RdnaFamily::AIW
              ? findStagedAiwKernel(dtype,
                                    spec.headDim,
                                    right >= 0 || left >= 0 ? 1 : 0,
                                    spec.withBias ? 1 : 0,
                                    spec.withBias && spec.biasDataType == DataType::FLOAT ? 1 : 0,
                                    spec.valueHeadDim,
                                    spec.withSink ? 1 : 0)
              : findStagedKernel(dtype,
                                 spec.headDim,
                                 causal ? 1 : 0,
                                 spec.withBias ? 1 : 0,
                                 decode ? 1 : 0,
                                 spec.withBias && spec.biasDataType != DataType::FLOAT ? 1 : 0,
                                 spec.valueHeadDim,
                                 spec.withSink ? 1 : 0);
    const auto kernel = forced != nullptr ? std::optional<KernelDefinition>(*forced) : searched;
    const auto& engine = family == RdnaFamily::AIW ? FLYDSL_SDPA_AIW : engineFor(decode);
    if(!kernel.has_value())
    {
        reportUnstaged(
            shippedOpFor(engine),
            "nothing staged for dtype=" + dtype + " head_dim=" + std::to_string(spec.headDim)
                + " causal=" + std::to_string(causal ? 1 : 0)
                + " bias=" + std::to_string(spec.withBias ? 1 : 0) + " sink="
                + std::to_string(spec.withSink ? 1 : 0) + " in this build's descriptor root");
        return;
    }

    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec), packedArchDeviceProperties());
    const auto bound = matchesGraph(engine, fixture.context());
    ASSERT_TRUE(bound.has_value());
    ASSERT_TRUE(matchesKernel(engine, fixture.context(), *kernel, *bound));

    const auto& handler = dispatchHandler(engine);
    const auto prepared = handler.prepare(fixture.context(), *bound, *kernel);
    ASSERT_NE(prepared, nullptr);

    const int64_t d = spec.headDim;
    // V's and O's head dim: Q/K's, or narrower (MLA).
    const int64_t dv = spec.valueHeadDim.value_or(d);
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
        vTensor.emplace(
            std::vector<int64_t>{spec.batch, valueHeads, spec.seqLenKv, dv},
            sdpaStrides(spec.valueLayout.value_or(spec.layout), valueHeads, spec.seqLenKv, dv),
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
    // Device-value overrides: a step in V, and NaN in Q or K (separate tensors only).
    if(spec.valueOnesBelowKey || spec.queryNanPosition || spec.keyNanPosition)
    {
        ASSERT_FALSE(spec.packedQkv);
        const T nan(std::numeric_limits<float>::quiet_NaN());
        for(int64_t b = 0; b < spec.batch; ++b)
        {
            if(spec.valueOnesBelowKey)
            {
                for(int64_t h = 0; h < valueHeads; ++h)
                {
                    for(int64_t j = 0; j < spec.seqLenKv; ++j)
                    {
                        for(int64_t e = 0; e < dv; ++e)
                        {
                            vTensor->set(b, h, j, e, T(j < *spec.valueOnesBelowKey ? 1.0F : 0.0F));
                        }
                    }
                }
            }
            for(int64_t h = 0; spec.queryNanPosition && h < spec.heads; ++h)
            {
                qTensor->set(b, h, *spec.queryNanPosition, 0, nan);
            }
            for(int64_t h = 0; spec.keyNanPosition && h < spec.kvHeads; ++h)
            {
                kTensor->set(b, h, *spec.keyNanPosition, 0, nan);
            }
        }
        qTensor->upload();
        kTensor->upload();
        vTensor->upload();
    }
    const auto third = static_cast<size_t>(spec.heads * d) * sizeof(T);
    void* qPtr = spec.packedQkv ? qkv->device() : qTensor->device();
    void* kPtr = spec.packedQkv ? static_cast<void*>(static_cast<char*>(qkv->device()) + third)
                                : kTensor->device();
    void* vPtr = spec.packedQkv ? static_cast<void*>(static_cast<char*>(qkv->device()) + 2 * third)
                                : vTensor->device();
    DeviceTensor<T> o({spec.batch, spec.heads, spec.seqLenQ, dv},
                      sdpaStrides(spec.layout, spec.heads, spec.seqLenQ, dv),
                      4U,
                      false);
    // NaN until written: an output row the kernel skips fails rather than reading as a zero.
    // A case that expects NaN fills with 7 instead, a value no O or LSE here takes.
    const bool expectsNan = spec.queryNanPosition.has_value() || spec.keyNanPosition.has_value();
    const float unwritten = expectsNan ? 7.0F : std::numeric_limits<float>::quiet_NaN();
    o.fillWith(T(unwritten));

    // f32 [B, H, Sq, 1], packed BHSD or laid out [B, Sq, H]: the layout the graph builder
    // declares.
    DeviceTensor<float> stats(
        {spec.batch, spec.heads, spec.seqLenQ, 1},
        spec.statsSequenceMajor
            ? std::vector<int64_t>{spec.heads * spec.seqLenQ, 1, spec.heads, 1}
            : std::vector<int64_t>{spec.heads * spec.seqLenQ, spec.seqLenQ, 1, 1},
        5U,
        false);
    stats.fillWith(unwritten);
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
    DeviceTensor<float> bias(biasDims4, biasStrides4, 6U, true);
    DeviceTensor<T> biasQ(biasDims4, biasStrides4, 6U, true);
    if(spec.biasMasksRows || spec.biasNegativeInfinityLeadingKeys > 0
       || spec.biasNegativeInfinityRows)
    {
        for(int64_t b = 0; b < biasDims4[0]; ++b)
        {
            for(int64_t h = 0; h < biasDims4[1]; ++h)
            {
                for(int64_t s = 0; s < biasDims4[2]; ++s)
                {
                    // Every third query row masked on every key with the dtype's lowest
                    // finite value, as a padding mask writes it: the row's softmax is uniform
                    // over its keys.
                    for(int64_t j = 0; spec.biasMasksRows && s % 3 == 0 && j < biasDims4[3]; ++j)
                    {
                        bias.set(b, h, s, j, std::numeric_limits<float>::lowest());
                        biasQ.set(b, h, s, j, std::numeric_limits<T>::lowest());
                    }
                    // -inf on the leading keys (a boolean key-padding mask cast to the bias),
                    // or on every key of every third row, a row no key reaches.
                    const int64_t masked
                        = spec.biasNegativeInfinityRows && s % 3 == 0
                              ? biasDims4[3]
                              : std::min(spec.biasNegativeInfinityLeadingKeys, biasDims4[3]);
                    for(int64_t j = 0; j < masked; ++j)
                    {
                        bias.set(b, h, s, j, -std::numeric_limits<float>::infinity());
                        biasQ.set(b, h, s, j, T(-std::numeric_limits<float>::infinity()));
                    }
                }
            }
        }
        bias.upload();
        biasQ.upload();
    }
    const auto biasAt = [&](int64_t b, int64_t h, int64_t s, int64_t j) {
        return biasInQueryType ? biasQ.at(b, h, s, j) : bias.at(b, h, s, j);
    };

    // The attention sink, one logit per query head in the query's dtype, read as [1, H, 1, 1].
    DeviceTensor<T> sink({1, spec.heads, 1, 1}, {0, 1, 0, 0}, 7U, true);
    if(spec.sinkValue)
    {
        sink.fillWith(T(*spec.sinkValue));
    }
    if(spec.sinkNegativeInfinity)
    {
        sink.fillWith(T(-std::numeric_limits<float>::infinity()));
    }

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
    if(spec.withSink)
    {
        buffers.push_back({FLYDSL_SDPA_RDNA_SINK_UID, sink.device()});
    }
    if(spec.scaleKind == SdpaScale::RUNTIME_TENSOR)
    {
        buffers.push_back({FLYDSL_SDPA_RDNA_SCALE_UID, &runtimeScale});
    }

    // A decode launch with more than one KV split writes its partials to workspace: exactly
    // the size the engine reports, so a launch that writes past it fails the case.
    const auto workspaceBytes = handler.workspaceBytes(fixture.context(), *bound, *kernel);
    const GuardedWorkspace workspace(workspaceBytes);
    const Handle handle;
    handler.launch(
        handle, *prepared, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace.get());
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
    EXPECT_EQ(workspace.overwrittenGuardBytes(), 0U)
        << "the launch wrote past the " << workspaceBytes << "-byte workspace it reported";
    o.readBack();
    stats.readBack();

    const double scale
        = spec.scaleKind == SdpaScale::ABSENT ? 1.0 : static_cast<double>(spec.scale);
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
                // A NaN score (NaN in Q, or in a key the row sees) keeps O and the LSE NaN,
                // as PyTorch's softmax and logsumexp do.
                bool nanRow = false;
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
                    nanRow = nanRow || std::isnan(scores[static_cast<size_t>(j)]);
                    maxScore = std::max(maxScore, scores[static_cast<size_t>(j)]);
                }
                // A row no key reaches: O = 0 and LSE = -inf, as CpuFpReferenceSdpa defines
                // it. The mask leaves it no key, or a bias masks every key it has with -inf
                // (its max score is -inf).
                const bool hasKeys = nanRow || maxScore > -std::numeric_limits<double>::infinity();
                double sum = 0.0;
                for(int64_t j = 0; j < spec.seqLenKv; ++j)
                {
                    scores[static_cast<size_t>(j)]
                        = isMasked(s, j) || !hasKeys
                              ? 0.0
                              : std::exp(scores[static_cast<size_t>(j)] - maxScore);
                    sum += scores[static_cast<size_t>(j)];
                }
                // The sink's logit joins the denominator of a row that reaches keys, as a key
                // with no value; a row that reaches none has O = 0 and LSE = the sink.
                const double sinkLogit = spec.withSink ? sink.at(0, h, 0, 0) : 0.0;
                if(spec.withSink && hasKeys)
                {
                    sum += std::exp(sinkLogit - maxScore);
                }
                if(wantStats)
                {
                    double expectedLse = -std::numeric_limits<double>::infinity();
                    if(hasKeys)
                    {
                        expectedLse = maxScore + std::log(sum);
                    }
                    else if(spec.withSink)
                    {
                        expectedLse = sinkLogit;
                    }
                    const double actualLse = stats.at(b, h, s, 0);
                    // A keyless row without a sink must be exactly -inf; any other value is a
                    // full miss.
                    const bool negativeInfinity = std::isinf(actualLse) && actualLse < 0.0;
                    double lseError = std::numeric_limits<double>::max();
                    if(std::isinf(expectedLse) && expectedLse < 0.0)
                    {
                        // A keyless row without a sink, or with a sink of -inf.
                        lseError = negativeInfinity ? 0.0 : lseError;
                    }
                    else if(std::isnan(expectedLse))
                    {
                        // A NaN row: NaN, not the keyless -inf or a finite value.
                        lseError = std::isnan(actualLse) ? 0.0 : lseError;
                    }
                    else if(std::isfinite(actualLse))
                    {
                        lseError = std::abs(actualLse - expectedLse);
                    }
                    if(lseError > worstLse)
                    {
                        worstLse = lseError;
                        whereLse = "b=" + std::to_string(b) + " h=" + std::to_string(h) + " s="
                                   + std::to_string(s) + ": got " + std::to_string(actualLse)
                                   + ", expected " + std::to_string(expectedLse);
                    }
                }
                for(int64_t e = 0; e < dv; ++e)
                {
                    double expected = 0.0;
                    for(int64_t j = 0; j < spec.seqLenKv; ++j)
                    {
                        expected += scores[static_cast<size_t>(j)] * vAt(b, vHead, j, e);
                    }
                    expected = hasKeys ? expected / sum : 0.0;
                    // NaN or inf (an unwritten element reads as NaN) is a full miss.
                    const double actual = o.at(b, h, s, e);
                    double error = std::isfinite(actual) ? std::abs(actual - expected)
                                                         : std::numeric_limits<double>::max();
                    if(std::isnan(expected))
                    {
                        // A NaN row must read NaN (the fill is 7 when NaN is expected).
                        error = std::isnan(actual) ? 0.0 : std::numeric_limits<double>::max();
                    }
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

TEST(TestGpuFlydslSdpaRdnaDispatch, ComputesTheDeprecatedCausalBooleanAtABottomRightAlignment)
{
    // causal_mask with diagonal_alignment BOTTOM_RIGHT is bottom-right causal, so with fewer
    // queries than keys each row sees Skv - Sq more keys than top-left would give it.
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.seqLenQ = 64;
    spec.seqLenKv = 300;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.deprecatedCausalTopLeft = true;
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

TEST(TestGpuFlydslSdpaRdnaDispatch, AppliesAWideBandWithMoreQueriesThanKeys)
{
    SKIP_IF_NO_DEVICES();
    // Bottom-right, Sq > Skv, a right bound past Skv: the band still reaches back to the
    // first rows, which a bound cut to Skv would leave keyless.
    SdpaGraphSpec spec;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.rightBound = 250;
    spec.seqLenQ = 300;
    spec.seqLenKv = 64;
    spec.stats = SdpaStats::LSE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, ServesAValueHeadDimNarrowerThanTheQueryOne)
{
    SKIP_IF_NO_DEVICES();
    // Multi-head latent attention, un-absorbed: Q and K 192 wide, V and O 128.
    SdpaGraphSpec spec;
    spec.headDim = 192;
    spec.valueHeadDim = 128;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.heads = 4;
    spec.kvHeads = 4;
    spec.seqLenQ = 300;
    spec.seqLenKv = 300;
    spec.stats = SdpaStats::LSE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);

    // Non-causal cross-attention with GQA, the last tile partial, in f16.
    spec.dataType = DataType::HALF;
    spec.mask = SdpaMask::NONE;
    spec.kvHeads = 2;
    spec.seqLenQ = 150;
    spec.seqLenKv = 333;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);

    // Bottom-right causal over a longer cache, BHSD.
    spec.dataType = DataType::BFLOAT16;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.layout = SdpaLayout::BHSD;
    spec.seqLenQ = 64;
    spec.seqLenKv = 257;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, DecodesAbsorbedMlaWithOneKvHead)
{
    SKIP_IF_NO_DEVICES();
    // Absorbed multi-head latent attention: every query head reads one KV head whose
    // keys are 576 wide (a 512 latent and a 64 rotary part) and whose values are the
    // 512-wide latent, over a cache long enough to split.
    SdpaGraphSpec spec;
    spec.headDim = 576;
    spec.valueHeadDim = 512;
    spec.heads = 16;
    spec.kvHeads = 1;
    spec.seqLenQ = 1;
    spec.seqLenKv = 3000;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);

    // A two-token speculative step across row tiles, in f16.
    spec.dataType = DataType::HALF;
    spec.heads = 32;
    spec.seqLenQ = 2;
    spec.seqLenKv = 777;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE, true);

    // A chunk of prefill (16 heads x 40 positions): no prefill object exists at this width,
    // so the decode engine serves it.
    spec.dataType = DataType::BFLOAT16;
    spec.heads = 16;
    spec.seqLenQ = 40;
    spec.seqLenKv = 300;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, DecodesHeadDim512WithOneKvHead)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.headDim = 512;
    spec.heads = 8;
    spec.kvHeads = 1;
    spec.seqLenQ = 1;
    spec.seqLenKv = 1500;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
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

TEST(TestGpuFlydslSdpaRdnaDispatch, ZeroesRowsATopLeftWindowStartsPastTheKeys)
{
    SKIP_IF_NO_DEVICES();
    // Top-left with Sq > Skv + left: rows past Skv + left have a window that starts after
    // the last key, so they see none (O = 0, LSE = -inf), causal and left-only, prefill and
    // decode objects alike.
    SdpaGraphSpec spec;
    spec.mask = SdpaMask::SLIDING_WINDOW;
    spec.leftBound = 40;
    spec.seqLenQ = 300;
    spec.seqLenKv = 128;
    spec.stats = SdpaStats::LSE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
    spec.rightBound = -1;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, AddsABiasThatMasksWholeRowsAtTheLowestFiniteValue)
{
    SKIP_IF_NO_DEVICES();
    // A padding mask's fully masked rows: every key's score is about -FLT_MAX, so the row's
    // softmax is uniform rather than NaN. f32 and query-dtype biases, prefill and decode.
    SdpaGraphSpec spec;
    spec.withBias = true;
    spec.biasMasksRows = true;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.seqLenQ = 77;
    spec.seqLenKv = 99;
    spec.stats = SdpaStats::LSE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
    spec.biasDataType = DataType::BFLOAT16;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
    // Decode: a few rows per head, the first of them masked, over a cache.
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 2;
    spec.seqLenKv = 300;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
    spec.biasDataType = DataType::FLOAT;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, WeighsDecodeSplitsByKeyCountUnderALowestFiniteBias)
{
    SKIP_IF_NO_DEVICES();
    // Every key's score rounds to the bias, -FLT_MAX, so the softmax is uniform and O is the
    // mean of V. 2049 keys are 65 tiles, two splits (decodeSplits: 65 / 32 by work): the
    // first split holds 33 tiles, keys [0, 1056), the second 993 keys. With V = 1 on the
    // first split's keys and 0 on the second's, O = 1056 / 2049 = 0.515. A merge weighing the
    // splits by exp(m + ln l - M) loses ln l to rounding at m ~ -3.4e38 and gives 0.5.
    SdpaGraphSpec spec;
    spec.batch = 1;
    spec.heads = 1;
    spec.kvHeads = 1;
    spec.seqLenQ = 1;
    spec.seqLenKv = 2049;
    spec.withBias = true;
    spec.biasMasksRows = true; // row 0, the only one
    spec.stats = SdpaStats::LSE;
    spec.valueOnesBelowKey = 1056;
    // V is 0 or 1 and P uniform, so only O's bf16 rounding (< 2e-3 at 0.5) is left; the
    // equal-weight error is 0.0154.
    constexpr double TOLERANCE = 5e-3;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, TOLERANCE, true);
    spec.biasDataType = DataType::BFLOAT16;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, TOLERANCE, true);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, AddsABiasThatMasksWholeRowsWithMinusInfinity)
{
    SKIP_IF_NO_DEVICES();
    // Every third row's bias is -inf on every key, so the row reaches no key: O = 0 and
    // LSE = -inf, not NaN, as hipDNN's CPU reference and the CDNA kernels give (COVERAGE.md
    // 5.7). f32 and query-dtype biases, prefill and decode.
    SdpaGraphSpec spec;
    spec.withBias = true;
    spec.biasNegativeInfinityRows = true;
    spec.headDim = 64;
    spec.heads = 4;
    spec.kvHeads = 4;
    spec.seqLenQ = 200;
    spec.seqLenKv = 300;
    spec.stats = SdpaStats::LSE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
    spec.biasDataType = DataType::BFLOAT16;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
    // Decode, one split: rows 0 and 3 of each head's four masked.
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 4;
    spec.biasDataType = DataType::FLOAT;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
    // Decode over 2085 keys, two splits: every split of a masked row sees only -inf.
    spec.seqLenKv = 2085;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
    spec.biasDataType = DataType::BFLOAT16;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, AddsABiasThatMasksLeadingKeysWithMinusInfinity)
{
    SKIP_IF_NO_DEVICES();
    // A boolean key-padding mask cast to the bias: the first 40 keys of every row -inf, part
    // of a KV tile, and the softmax is over the rest (COVERAGE.md 5.7). Prefill, plain and
    // bottom-right causal, and decode.
    SdpaGraphSpec spec;
    spec.withBias = true;
    spec.biasNegativeInfinityLeadingKeys = 40;
    spec.headDim = 64;
    spec.heads = 4;
    spec.kvHeads = 4;
    spec.seqLenQ = 200;
    spec.seqLenKv = 300;
    spec.stats = SdpaStats::LSE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.biasDataType = DataType::BFLOAT16;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
    // Decode over 2085 keys, two splits (the first holds keys [0, 1056)).
    spec.mask = SdpaMask::NONE;
    spec.biasDataType = DataType::FLOAT;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 4;
    spec.seqLenKv = 2085;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
    // The whole first split -inf: its partial reaches no key, and the merge must give it no
    // weight rather than a NaN.
    spec.biasNegativeInfinityLeadingKeys = 1100;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, AddsASinkFarAboveTheScores)
{
    SKIP_IF_NO_DEVICES();
    // A sink 200 above scores of at most a few units: exp(sink - max) alone overflows f32
    // (past ~88.7), so the sink must join the row's max. O ~ 0 and LSE = the sink, as the
    // CDNA kernels give (COVERAGE.md 5.7).
    SdpaGraphSpec spec;
    spec.headDim = 64;
    spec.heads = 16;
    spec.kvHeads = 2;
    spec.withSink = true;
    spec.sinkValue = 200.0F;
    spec.stats = SdpaStats::LSE;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.seqLenQ = 300;
    spec.seqLenKv = 300;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
    // Non-causal, f16.
    spec.mask = SdpaMask::NONE;
    spec.dataType = DataType::HALF;
    spec.seqLenQ = 129;
    spec.seqLenKv = 333;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
    // Rows no key reaches next to rows that reach keys: LSE = the sink for both.
    spec.dataType = DataType::BFLOAT16;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.seqLenQ = 300;
    spec.seqLenKv = 64;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
    // Decode over 2085 keys, two splits: the merge adds the sink against the max.
    spec.mask = SdpaMask::NONE;
    spec.heads = 8;
    spec.seqLenQ = 4;
    spec.seqLenKv = 2085;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, GivesANanRowANanLse)
{
    SKIP_IF_NO_DEVICES();
    // PyTorch: a row whose softmax sum is NaN has O = NaN and logsumexp NaN, not the -inf of
    // a row no key reaches. Prefill, with and without a sink: NaN in one query row.
    SdpaGraphSpec spec;
    spec.stats = SdpaStats::LSE;
    spec.queryNanPosition = 5;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
    spec.withSink = true;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
    // Decode with one split: NaN in one of the GQA group's rows.
    spec = SdpaGraphSpec{};
    spec.stats = SdpaStats::LSE;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 2;
    spec.seqLenKv = 300;
    spec.queryNanPosition = 1;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
    // Decode with two splits (2049 keys, as above): NaN in a key of the first split makes
    // every row NaN; the merge must not report the other split's finite LSE.
    spec = SdpaGraphSpec{};
    spec.stats = SdpaStats::LSE;
    spec.batch = 1;
    spec.heads = 2;
    spec.kvHeads = 1;
    spec.seqLenQ = 1;
    spec.seqLenKv = 2049;
    spec.keyNanPosition = 5;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
    spec.withSink = true;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
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

// The window spans several splits. The LSE is checked too: one key more or less at the
// window's left edge moves O by less than bf16's budget over 128 keys, but moves the LSE
// past its 1e-3 one, so an edge the split path gets wrong cannot pass.
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
    spec.stats = SdpaStats::LSE;
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

// gpt-oss: head_dim 64, GQA, a sink logit per query head, causal with and without a
// sliding window; the LSE includes the sink.
TEST(TestGpuFlydslSdpaRdnaDispatch, AddsAnAttentionSink)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 150;
    spec.seqLenKv = 150;
    spec.stats = SdpaStats::LSE;
    spec.withSink = true;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
    spec.mask = SdpaMask::SLIDING_WINDOW;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
}

// A row no key reaches keeps only the sink: O = 0 and LSE = the sink's logit.
TEST(TestGpuFlydslSdpaRdnaDispatch, AddsASinkToRowsNoKeyReaches)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.headDim = 128;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.heads = 4;
    spec.kvHeads = 4;
    spec.seqLenQ = 80;
    spec.seqLenKv = 48;
    spec.stats = SdpaStats::LSE;
    spec.withSink = true;
    spec.sinkRank4 = true;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
}

// With one KV split the decode kernel adds the sink itself; with several the partials leave
// it out and the merge adds it.
TEST(TestGpuFlydslSdpaRdnaDispatch, DecodesWithASinkInOneSplitAndAcrossSplits)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 1;
    spec.seqLenKv = 60;
    spec.stats = SdpaStats::LSE;
    spec.withSink = true;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
    spec.seqLenKv = 2100;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
    spec.dataType = DataType::HALF;
    spec.headDim = 128;
    spec.seqLenQ = 4;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE, true);
}

// V and the LSE in layouts of their own.
TEST(TestGpuFlydslSdpaRdnaDispatch, ReadsVAndWritesTheLseThroughTheirOwnLayouts)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 150;
    spec.seqLenKv = 200;
    spec.valueLayout = SdpaLayout::BHSD;
    spec.stats = SdpaStats::LSE;
    spec.statsSequenceMajor = true;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
    spec.seqLenQ = 1;
    spec.seqLenKv = 2100;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
}

// The runtime scale is read from host memory at launch; one the kernels cannot take (not
// positive, not finite, or past the shared 2^20 bound) is refused there. Every operand is
// bound to real device memory, so the scale is the only thing a launch can refuse, and a
// servable scale with the same buffers launches.
TEST(TestGpuFlydslSdpaRdnaDispatch, RefusesANonPositiveRuntimeScaleAtLaunch)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.scaleKind = SdpaScale::RUNTIME_TENSOR;
    const auto kernel = findStagedKernel("bf16", spec.headDim, 0, 0);
    if(!kernel.has_value())
    {
        reportUnstaged(shippedOpFor(FLYDSL_SDPA_RDNA), "nothing staged for bf16 non-causal");
        return;
    }
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec), packedArchDeviceProperties());
    const auto bound = matchesGraph(FLYDSL_SDPA_RDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());
    ASSERT_TRUE(matchesKernel(FLYDSL_SDPA_RDNA, fixture.context(), *kernel, *bound));
    const auto& handler = dispatchHandler(FLYDSL_SDPA_RDNA);
    const auto prepared = handler.prepare(fixture.context(), *bound, *kernel);
    ASSERT_NE(prepared, nullptr);

    using Bf16 = hipdnn_data_sdk::types::bfloat16;
    const std::vector<int64_t> qDims{spec.batch, spec.heads, spec.seqLenQ, spec.headDim};
    const std::vector<int64_t> kvDims{spec.batch, spec.kvHeads, spec.seqLenKv, spec.headDim};
    const auto qStrides = sdpaStrides(spec.layout, spec.heads, spec.seqLenQ, spec.headDim);
    const auto kvStrides = sdpaStrides(spec.layout, spec.kvHeads, spec.seqLenKv, spec.headDim);
    const DeviceTensor<Bf16> q(qDims, qStrides, 1U, true);
    const DeviceTensor<Bf16> k(kvDims, kvStrides, 2U, true);
    const DeviceTensor<Bf16> v(kvDims, kvStrides, 3U, true);
    const DeviceTensor<Bf16> o(qDims, qStrides, 0U, false);
    const auto buffersWith = [&](float& scale) {
        return std::vector<hipdnnPluginDeviceBuffer_t>{{FLYDSL_SDPA_RDNA_SCALE_UID, &scale},
                                                       {FLYDSL_SDPA_RDNA_Q_UID, q.device()},
                                                       {FLYDSL_SDPA_RDNA_K_UID, k.device()},
                                                       {FLYDSL_SDPA_RDNA_V_UID, v.device()},
                                                       {FLYDSL_SDPA_RDNA_O_UID, o.device()}};
    };

    const Handle handle;
    for(float scale : {0.0F,
                       -0.125F,
                       std::numeric_limits<float>::quiet_NaN(),
                       std::numeric_limits<float>::infinity(),
                       2097152.0F /* 2^21, past the 2^20 bound */})
    {
        const auto buffers = buffersWith(scale);
        EXPECT_THROW(
            handler.launch(
                handle, *prepared, buffers.data(), static_cast<uint32_t>(buffers.size()), nullptr),
            hipdnn_plugin_sdk::HipdnnPluginException)
            << scale;
    }

    // The control: with the same buffers, a servable scale launches.
    float servable = 0.125F;
    const auto buffers = buffersWith(servable);
    EXPECT_NO_THROW(handler.launch(
        handle, *prepared, buffers.data(), static_cast<uint32_t>(buffers.size()), nullptr));
    EXPECT_EQ(hipSuccess, hipDeviceSynchronize());
}

TEST(TestGpuFlydslSdpaRdnaDispatch, AppliesTheDefaultScaleWhenTheGraphGivesNone)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.scaleKind = SdpaScale::ABSENT;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
}

// ---------------------------------------------------------------------------
// hipkernel:flydsl_sdpa_aiw: AOTriton's gfx1201 kernel, its own engine
// ---------------------------------------------------------------------------

/// An aiw object carrying exactly the metadata its KMD declares.
KernelDefinition makeAiwKernel(const std::string& dtype,
                               int64_t headDim,
                               int64_t masked,
                               std::optional<int64_t> headDimMax = std::nullopt,
                               int64_t hasBias = 0,
                               int64_t biasF32 = 0,
                               std::optional<int64_t> headDimV = std::nullopt,
                               int64_t hasSink = 0)
{
    KernelDefinition kernel;
    kernel.name = "sdpa_aiw_" + dtype + "_d" + std::to_string(headDim);
    kernel.metadata = {{FLYDSL_SDPA_RDNA_DTYPE_FIELD, dtype},
                       {FLYDSL_SDPA_RDNA_HEAD_DIM_FIELD, headDim},
                       {FLYDSL_SDPA_RDNA_CAUSAL_FIELD, masked},
                       {FLYDSL_SDPA_RDNA_HEAD_DIM_MAX_FIELD, headDimMax.value_or(headDim)},
                       {FLYDSL_SDPA_RDNA_HEAD_DIM_V_FIELD, headDimV.value_or(headDim)},
                       {FLYDSL_SDPA_RDNA_HAS_BIAS_FIELD, hasBias},
                       {FLYDSL_SDPA_AIW_BIAS_F32_FIELD, biasF32},
                       {FLYDSL_SDPA_AIW_BIAS_KSTRIDED_FIELD, int64_t{0}},
                       {FLYDSL_SDPA_RDNA_HAS_SINK_FIELD, hasSink},
                       {FLYDSL_SDPA_RDNA_BLOCK_M_FIELD, int64_t{128}},
                       {FLYDSL_SDPA_AIW_BLOCK_THREADS_FIELD, int64_t{256}}};
    kernel.priority = 100;
    return kernel;
}

bool aiwMatches(const SdpaGraphSpec& spec, const KernelDefinition& kernel)
{
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    const auto bound = matchesGraph(FLYDSL_SDPA_AIW, fixture.context());
    return bound.has_value() && matchesKernel(FLYDSL_SDPA_AIW, fixture.context(), kernel, *bound);
}

TEST(TestFlydslSdpaAiwKernelMatch, AdmitsTheObjectBuiltForTheGraphAndDeclinesOtherAxes)
{
    const SdpaGraphSpec spec;
    EXPECT_TRUE(aiwMatches(spec, makeAiwKernel("bf16", 64, 0)));
    EXPECT_FALSE(aiwMatches(spec, makeAiwKernel("f16", 64, 0)));
    EXPECT_FALSE(aiwMatches(spec, makeAiwKernel("bf16", 128, 0)));
    EXPECT_FALSE(aiwMatches(spec, makeAiwKernel("bf16", 64, 1)));
    EXPECT_FALSE(aiwMatches(spec, makeAiwKernel("bf16", 64, 0, 64, 0, 0, int64_t{64}, 1)));
}

TEST(TestFlydslSdpaAiwKernelMatch, RunsTheMaskedObjectForAnyBound)
{
    // The non-masked object applies no window, so every bound -- causal, a band, a left-only
    // window -- runs on the masked one.
    SdpaGraphSpec spec;
    for(const auto mask :
        {SdpaMask::TOP_LEFT_CAUSAL, SdpaMask::BOTTOM_RIGHT_CAUSAL, SdpaMask::SLIDING_WINDOW})
    {
        spec.mask = mask;
        EXPECT_TRUE(aiwMatches(spec, makeAiwKernel("bf16", 64, 1)));
        EXPECT_FALSE(aiwMatches(spec, makeAiwKernel("bf16", 64, 0)));
    }
    spec.mask = SdpaMask::NONE;
    spec.leftBound = 16;
    EXPECT_TRUE(aiwMatches(spec, makeAiwKernel("bf16", 64, 1))) << "a left-only window";
    EXPECT_FALSE(aiwMatches(spec, makeAiwKernel("bf16", 64, 0))) << "a left-only window";
}

TEST(TestFlydslSdpaAiwKernelMatch, APaddedTierServesMultiplesOfEightUpToItsTile)
{
    SdpaGraphSpec spec;
    const auto tier = makeAiwKernel("bf16", 0, 0, int64_t{96}, 0, 0, int64_t{0});
    for(const int64_t headDim : {16, 72, 80, 96})
    {
        spec.headDim = headDim;
        EXPECT_TRUE(aiwMatches(spec, tier)) << headDim;
    }
    for(const int64_t headDim : {100, 104, 128})
    {
        spec.headDim = headDim;
        EXPECT_FALSE(aiwMatches(spec, tier)) << headDim;
    }
}

TEST(TestFlydslSdpaAiwKernelMatch, PairsTheMaskObjectWithTheMasksDtype)
{
    SdpaGraphSpec spec;
    spec.withBias = true;
    spec.biasDataType = DataType::FLOAT;
    spec.alignment[FLYDSL_SDPA_RDNA_BIAS_UID] = 32;
    EXPECT_TRUE(aiwMatches(spec, makeAiwKernel("bf16", 64, 0, 64, 1, 1)));
    EXPECT_FALSE(aiwMatches(spec, makeAiwKernel("bf16", 64, 0, 64, 1, 0)));
    EXPECT_FALSE(aiwMatches(spec, makeAiwKernel("bf16", 64, 0)));
    spec.biasDataType = DataType::BFLOAT16;
    EXPECT_TRUE(aiwMatches(spec, makeAiwKernel("bf16", 64, 0, 64, 1, 0)));
    EXPECT_FALSE(aiwMatches(spec, makeAiwKernel("bf16", 64, 0, 64, 1, 1)));
}

TEST(TestFlydslSdpaAiwKernelMatch, DeclinesRowsItsVectorLoadsCannotReachAligned)
{
    // The kernel moves rows as 16-byte vectors and promises LLVM that alignment: a row
    // pitch that is not a multiple of 8 elements would be read misaligned.
    SdpaGraphSpec spec;
    spec.valueSequenceStride = spec.kvHeads * spec.headDim + 8;
    EXPECT_TRUE(aiwMatches(spec, makeAiwKernel("bf16", 64, 0))) << "a padded, aligned pitch";
    spec.valueSequenceStride = spec.kvHeads * spec.headDim + 4;
    EXPECT_FALSE(aiwMatches(spec, makeAiwKernel("bf16", 64, 0))) << "a pitch of 8n + 4 elements";
}

TEST(TestFlydslSdpaAiwKernelMatch, DeclinesAMaskWhoseRowsItsVectorLoadsCannotReachAligned)
{
    // The mask is read 8 keys at a time at the vector's alignment: a packed mask's row
    // pitch is Skv, so 77 keys would start rows misaligned; 80, or 77 padded to 80, do not.
    SdpaGraphSpec spec;
    spec.withBias = true;
    spec.alignment[FLYDSL_SDPA_RDNA_BIAS_UID] = 32;
    spec.seqLenKv = 80;
    EXPECT_TRUE(aiwMatches(spec, makeAiwKernel("bf16", 64, 0, 64, 1, 1))) << "Skv 80";
    spec.seqLenKv = 77;
    EXPECT_FALSE(aiwMatches(spec, makeAiwKernel("bf16", 64, 0, 64, 1, 1))) << "Skv 77";
    spec.biasDims = {1, 1, spec.seqLenQ, spec.seqLenKv};
    spec.biasStrides = {spec.seqLenQ * 80, spec.seqLenQ * 80, 80, 1};
    EXPECT_TRUE(aiwMatches(spec, makeAiwKernel("bf16", 64, 0, 64, 1, 1))) << "Skv 77, pitch 80";
    // A mask that broadcasts over the queries steps no row at all.
    spec.biasDims = {spec.seqLenKv};
    spec.biasStrides = {1};
    EXPECT_TRUE(aiwMatches(spec, makeAiwKernel("bf16", 64, 0, 64, 1, 1))) << "[Skv] mask";
}

TEST(TestFlydslSdpaAiwKernelMatch, DeclinesAnF32MaskWhoseBaseIsNotAlignedTo32Bytes)
{
    // An f32 mask is loaded 8 x f32 at a time at that type's natural alignment, 32 bytes:
    // the schema's default 16 is not enough. A query-dtype mask needs only 16.
    SdpaGraphSpec spec;
    spec.withBias = true;
    spec.biasDataType = DataType::FLOAT;
    EXPECT_FALSE(aiwMatches(spec, makeAiwKernel("bf16", 64, 0, 64, 1, 1))) << "f32, 16 bytes";
    spec.alignment[FLYDSL_SDPA_RDNA_BIAS_UID] = 32;
    EXPECT_TRUE(aiwMatches(spec, makeAiwKernel("bf16", 64, 0, 64, 1, 1))) << "f32, 32 bytes";
    spec.alignment.clear();
    spec.biasDataType = DataType::BFLOAT16;
    EXPECT_TRUE(aiwMatches(spec, makeAiwKernel("bf16", 64, 0, 64, 1, 0))) << "bf16, 16 bytes";
}

TEST(TestFlydslSdpaAiwKernelMatch, DeclinesAMaskWhoseKeyAxisIsNotContiguous)
{
    // A mask stored key-major: the object reads eight adjacent keys per load.
    SdpaGraphSpec spec;
    spec.withBias = true;
    spec.biasDims = {1, 1, spec.seqLenQ, spec.seqLenKv};
    spec.biasStrides
        = {spec.seqLenQ * spec.seqLenKv, spec.seqLenQ * spec.seqLenKv, 1, spec.seqLenQ};
    EXPECT_FALSE(aiwMatches(spec, makeAiwKernel("bf16", 64, 0, 64, 1, 1)));
}

TEST(TestFlydslSdpaAiwDispatch, ResolvesFromTheDispatchRegistry)
{
    EXPECT_NO_THROW(static_cast<void>(dispatchHandler(FLYDSL_SDPA_AIW)));
}

TEST(TestFlydslSdpaAiwDispatch, RefusesToPrepareWithoutTheMatcherSBindings)
{
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph());
    EXPECT_THROW(dispatchHandler(FLYDSL_SDPA_AIW)
                     .prepare(fixture.context(), BoundTokens{}, makeAiwKernel("bf16", 64, 0)),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

// On device: the staged aiw objects (gfx12 only; elsewhere every case skips) against the
// double-precision reference, across the features its vendoring added.
TEST(TestGpuFlydslSdpaAiwDispatch, ComputesCausalGqaWithTheLse)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.headDim = 128;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 300;
    spec.seqLenKv = 300;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
    spec.layout = SdpaLayout::BHSD;
    spec.mask = SdpaMask::NONE;
    spec.dataType = DataType::HALF;
    spec.headDim = 64;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE, RdnaFamily::AIW);
}

TEST(TestGpuFlydslSdpaAiwDispatch, AddsAMaskWithACausalMaskOrAWindow)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.headDim = 128;
    spec.seqLenQ = 200;
    spec.seqLenKv = 200;
    spec.withBias = true;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.biasDataType = DataType::FLOAT;
    // hipMalloc'd: the 32-byte base the f32 mask's loads need.
    spec.alignment[FLYDSL_SDPA_RDNA_BIAS_UID] = 32;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
    spec.headDim = 64;
    spec.biasDataType = DataType::BFLOAT16;
    spec.mask = SdpaMask::SLIDING_WINDOW;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
    // The two cases above broadcast over batch and heads (the default {1, 1, Sq, Skv}, both
    // strides 0). A mask per (batch, head), then one per head shared across the batch: the
    // kernel must step by the batch and head strides the graph gives.
    ASSERT_GT(spec.batch, 1);
    ASSERT_GT(spec.heads, 1);
    spec.biasDims = {spec.batch, spec.heads, spec.seqLenQ, spec.seqLenKv};
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
    spec.biasDims = {1, spec.heads, spec.seqLenQ, spec.seqLenKv};
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
}

TEST(TestGpuFlydslSdpaAiwDispatch, AddsAMaskThatMasksWholeRowsAtTheLowestFiniteValue)
{
    SKIP_IF_NO_DEVICES();
    // A padding mask's fully masked rows (HF's convention): every third row's mask is the
    // mask dtype's lowest finite value on every key, so its scores are all about -FLT_MAX
    // and its softmax is uniform, O = mean(V), with an LSE at the mask value, as PyTorch
    // and the §5 kernels give. Upstream added mask * log2e to base-2 scores, which overflows
    // to -inf below -2.36e38 (O = 0, LSE = -inf); README modification 9. The f32 mask's
    // lowest is -FLT_MAX itself, which the kernel's m floor must not sit above.
    SdpaGraphSpec spec;
    spec.withBias = true;
    spec.biasMasksRows = true;
    spec.seqLenQ = 77;
    // A multiple of 8: the mask's rows must stay aligned for the object's vector loads.
    spec.seqLenKv = 104;
    spec.stats = SdpaStats::LSE;
    spec.biasDataType = DataType::FLOAT;
    spec.alignment[FLYDSL_SDPA_RDNA_BIAS_UID] = 32;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
    spec.alignment.clear();
    spec.biasDataType = DataType::BFLOAT16;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
    // Partly masked rows: the causal bound removes some of a masked row's keys (-inf) and
    // the mask's lowest value covers the rest, so O is the mean of V over the keys the
    // bound keeps. Top-left row 0 keeps one key; bottom-right rows keep Skv - Sq more.
    spec.headDim = 128;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
    spec.dataType = DataType::HALF;
    spec.headDim = 64;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.biasDataType = DataType::FLOAT;
    spec.alignment[FLYDSL_SDPA_RDNA_BIAS_UID] = 32;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE, RdnaFamily::AIW);
}

/// The aiw cases' mask: Skv a multiple of 8 (the mask's rows stay aligned for the object's
/// vector loads), an f32 mask declared 32-byte aligned.
SdpaGraphSpec aiwMaskSpec()
{
    SdpaGraphSpec spec;
    spec.withBias = true;
    spec.seqLenQ = 77;
    spec.seqLenKv = 104;
    spec.stats = SdpaStats::LSE;
    spec.biasDataType = DataType::FLOAT;
    spec.alignment[FLYDSL_SDPA_RDNA_BIAS_UID] = 32;
    return spec;
}

TEST(TestGpuFlydslSdpaAiwDispatch, AddsABiasThatMasksWholeRowsWithMinusInfinity)
{
    SKIP_IF_NO_DEVICES();
    // Every third row's mask is -inf on every key, so the row reaches no key: O = 0 and
    // LSE = -inf, not NaN, as the RDNA and CDNA kernels give (COVERAGE.md 5.7). The kernel
    // was built without fast math's no-infinities flag so that such a mask works (README
    // modification 9). f32 and query-dtype masks, plain and bottom-right causal.
    auto spec = aiwMaskSpec();
    spec.biasNegativeInfinityRows = true;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
    spec.alignment.clear();
    spec.biasDataType = DataType::BFLOAT16;
    spec.headDim = 128;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
    spec.dataType = DataType::HALF;
    spec.headDim = 64;
    spec.mask = SdpaMask::NONE;
    spec.biasDataType = DataType::HALF;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE, RdnaFamily::AIW);
}

TEST(TestGpuFlydslSdpaAiwDispatch, AddsABiasThatMasksLeadingKeysWithMinusInfinity)
{
    SKIP_IF_NO_DEVICES();
    // A boolean key-padding mask cast to the mask: the first 40 keys of every row -inf, part
    // of a KV tile, and the softmax is over the rest (COVERAGE.md 5.7). f32 and query-dtype
    // masks, plain and bottom-right causal.
    auto spec = aiwMaskSpec();
    spec.biasNegativeInfinityLeadingKeys = 40;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
    spec.alignment.clear();
    spec.biasDataType = DataType::BFLOAT16;
    spec.headDim = 128;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
}

TEST(TestGpuFlydslSdpaAiwDispatch, AddsASinkFarAboveTheScores)
{
    SKIP_IF_NO_DEVICES();
    // A sink 200 above scores of at most a few units: exp(sink - max) alone overflows f32,
    // so the sink must join the row's max. O ~ 0 and LSE = the sink, as the RDNA and CDNA
    // kernels give (COVERAGE.md 5.7).
    SdpaGraphSpec spec;
    spec.headDim = 64;
    spec.heads = 16;
    spec.kvHeads = 2;
    spec.withSink = true;
    spec.sinkValue = 200.0F;
    spec.stats = SdpaStats::LSE;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.seqLenQ = 300;
    spec.seqLenKv = 300;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
    // Non-causal, f16, at head_dim 128.
    spec.mask = SdpaMask::NONE;
    spec.dataType = DataType::HALF;
    spec.headDim = 128;
    spec.seqLenQ = 129;
    spec.seqLenKv = 333;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE, RdnaFamily::AIW);
    // Rows no key reaches next to rows that reach keys: LSE = the sink for both.
    spec.dataType = DataType::BFLOAT16;
    spec.headDim = 64;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.seqLenQ = 300;
    spec.seqLenKv = 64;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
}

TEST(TestGpuFlydslSdpaAiwDispatch, AppliesBottomRightWindowsAndBands)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.headDim = 64;
    spec.seqLenQ = 150;
    spec.seqLenKv = 400;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
    spec.leftBound = 40;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.leftBound.reset();
    spec.rightBound = 24;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
    // A left-only window, no causal bound.
    spec.mask = SdpaMask::NONE;
    spec.rightBound.reset();
    spec.leftBound = 32;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
}

TEST(TestGpuFlydslSdpaAiwDispatch, ZeroesRowsNoKeyReaches)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.headDim = 128;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.seqLenQ = 300;
    spec.seqLenKv = 64;
    spec.stats = SdpaStats::LSE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
    spec.dataType = DataType::HALF;
    spec.withSink = true;
    spec.headDim = 64;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE, RdnaFamily::AIW);
}

TEST(TestGpuFlydslSdpaAiwDispatch, AddsAnAttentionSink)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.headDim = 64;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 256;
    spec.seqLenKv = 256;
    spec.withSink = true;
    spec.mask = SdpaMask::SLIDING_WINDOW;
    spec.stats = SdpaStats::LSE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
}

/// A sink logit of -inf contributes nothing: rows with keys are plain softmax, and a row no
/// key reaches (bottom-right, Sq > Skv) gets O = 0 and LSE = -inf, not NaN.
SdpaGraphSpec sinkOfMinusInfinitySpec()
{
    SdpaGraphSpec spec;
    spec.headDim = 64;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 300;
    spec.seqLenKv = 64;
    spec.withSink = true;
    spec.sinkNegativeInfinity = true;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    return spec;
}

TEST(TestGpuFlydslSdpaAiwDispatch, AddsASinkOfMinusInfinityToRowsNoKeyReaches)
{
    SKIP_IF_NO_DEVICES();
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        sinkOfMinusInfinitySpec(), BF16_TOLERANCE, RdnaFamily::AIW);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, AddsASinkOfMinusInfinityToRowsNoKeyReaches)
{
    SKIP_IF_NO_DEVICES();
    auto spec = sinkOfMinusInfinitySpec();
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
}

TEST(TestGpuFlydslSdpaAiwDispatch, ServesMlaAPaddedHeadDimAndVHeadsOfTheirOwn)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.headDim = 192;
    spec.valueHeadDim = 128;
    spec.seqLenQ = 200;
    spec.seqLenKv = 200;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
    spec.valueHeadDim.reset();
    spec.headDim = 80;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
    spec.headDim = 136;
    spec.mask = SdpaMask::NONE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
    spec.headDim = 64;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.valueHeads = 4;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
}

TEST(TestGpuFlydslSdpaAiwDispatch, WritesTheLseThroughItsOwnLayoutAndReadsARuntimeScale)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.headDim = 256;
    spec.seqLenQ = 150;
    spec.seqLenKv = 150;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    spec.statsSequenceMajor = true;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
    spec.headDim = 64;
    spec.scaleKind = SdpaScale::RUNTIME_TENSOR;
    spec.scale = 0.07F;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
}

} // namespace

// The harness above, reached with a named object (FlydslSdpaObjectTestSupport.hpp).
void testing::expectRdnaObjectMatchesReference(const SdpaGraphSpec& spec,
                                               RdnaObjectEngine engine,
                                               const KernelDefinition& kernel)
{
    auto family = RdnaFamily::PREFILL;
    if(engine == RdnaObjectEngine::AIW)
    {
        family = RdnaFamily::AIW;
    }
    else if(engine == RdnaObjectEngine::DECODE)
    {
        family = RdnaFamily::DECODE;
    }
    if(spec.dataType == DataType::HALF)
    {
        expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(
            spec, F16_TOLERANCE, family, &kernel);
    }
    else
    {
        expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
            spec, BF16_TOLERANCE, family, &kernel);
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
