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
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaCpuReferenceTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaDropout.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaObjectTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaPhiloxOracle.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaRdnaTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaRngDump.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaRngDumpTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaShared.hpp"

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
        // V wider than Q and K: the aiw padded tiers take both widths at runtime; the RDNA
        // objects decline it by their baked head_dim_v.
        AcceptedCase{"ValueHeadDimWider", accepted([](SdpaGraphSpec& s) { s.valueHeadDim = 128; })},
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
        // A FLOAT sink, cuDNN's and hipDNN's reference's dtype (kernel modifications 31 and
        // 11), rank 1 and rank 4, and strided across heads.
        AcceptedCase{"FloatSink", accepted([](SdpaGraphSpec& s) {
                         s.withSink = true;
                         s.sinkDataType = DataType::FLOAT;
                     })},
        AcceptedCase{"FloatSinkRank4", accepted([](SdpaGraphSpec& s) {
                         s.withSink = true;
                         s.sinkRank4 = true;
                         s.sinkDataType = DataType::FLOAT;
                     })},
        AcceptedCase{"FloatSinkStridedAcrossHeads", accepted([](SdpaGraphSpec& s) {
                         s.withSink = true;
                         s.sinkDataType = DataType::FLOAT;
                         s.sinkHeadStride = 3;
                     })},
        // The int32 offset bound counts 16-bit halves: a query-dtype sink reaches 2^25 per head
        // over 64 heads, where a FLOAT one declines (the shared decline table).
        AcceptedCase{"SinkHeadOffsetsAtTheInt32Limit", accepted([](SdpaGraphSpec& s) {
                         s.withSink = true;
                         s.heads = 64;
                         s.kvHeads = 8;
                         s.sinkHeadStride = int64_t{1} << 25;
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

TEST(TestFlydslSdpaRdnaBinding, ALeftBoundCountsTheDiagonalKey)
{
    // cuDNN's left_bound L keeps L keys up to and including the diagonal; the kernels take
    // the keys before it, L - 1: 1 is the diagonal alone (0 before it), 2 one key more.
    SdpaGraphSpec spec;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.leftBound = 1;
    EXPECT_EQ(boundInt(bindingsFor(spec), FLYDSL_SDPA_RDNA_LEFT_BOUND_TOKEN), 0);
    spec.leftBound = 2;
    EXPECT_EQ(boundInt(bindingsFor(spec), FLYDSL_SDPA_RDNA_LEFT_BOUND_TOKEN), 1);
    spec.leftBound.reset();
    EXPECT_EQ(boundInt(bindingsFor(spec), FLYDSL_SDPA_RDNA_LEFT_BOUND_TOKEN), -1);
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

    // N2: an MLA graph with a bias runs on the MLA bias objects, of the bias's dtype; the
    // plain MLA object declines it, and a bias MLA object a graph without one.
    auto biased = mla;
    biased.withBias = true;
    const GraphFixture withBias(buildFlydslSdpaRdnaGraph(biased));
    const auto biasBound = bindingsFor(biased);
    const auto mlaBias
        = makeFlydslSdpaRdnaKernel("bf16", 192, 0, 64, 32, 1, 1, 192, 0, 1, 0, int64_t{128});
    const auto mlaBiasQ
        = makeFlydslSdpaRdnaKernel("bf16", 192, 0, 64, 32, 1, 1, 192, 0, 1, 1, int64_t{128});
    EXPECT_TRUE(matchesKernel(FLYDSL_SDPA_RDNA, withBias.context(), mlaBias, biasBound));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, withBias.context(), mlaBiasQ, biasBound));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, withBias.context(), mlaKernel, biasBound));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, narrowV.context(), mlaBias, narrowBound));
}

TEST(TestFlydslSdpaRdnaKernelMatch, DeclinesAValueHeadDimWiderThanTheQueryOne)
{
    // The graph match admits V wider than Q/K (the aiw padded tiers serve it); every RDNA
    // prefill and decode object declines it: a generic one's V is its runtime head_dim, a
    // specialized or MLA one's its baked head_dim_v.
    SdpaGraphSpec wide;
    wide.headDim = 96;
    wide.valueHeadDim = 128;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(wide));
    const auto bound = matchesGraph(FLYDSL_SDPA_RDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());
    const auto generic
        = makeFlydslSdpaRdnaKernel("bf16", 0, 0, 64, 32, 1, 0, int64_t{128}, 0, 1, 0, int64_t{0});
    const auto genericDecode
        = makeFlydslSdpaRdnaKernel("bf16", 0, 0, 16, 32, 1, 0, int64_t{128}, 1, 0, 0, int64_t{0});
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, fixture.context(), generic, *bound));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA_DECODE, fixture.context(), genericDecode, *bound));
    const auto specialized
        = makeFlydslSdpaRdnaKernel("bf16", 96, 0, 64, 32, 1, 0, int64_t{96}, 0, 1, 0, int64_t{96});
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, fixture.context(), specialized, *bound));

    // The square graph at the same tier still runs (the control).
    SdpaGraphSpec square;
    square.headDim = 96;
    const GraphFixture squareFixture(buildFlydslSdpaRdnaGraph(square));
    EXPECT_TRUE(
        matchesKernel(FLYDSL_SDPA_RDNA, squareFixture.context(), generic, bindingsFor(square)));
    EXPECT_TRUE(matchesKernel(
        FLYDSL_SDPA_RDNA_DECODE, squareFixture.context(), genericDecode, bindingsFor(square)));
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

TEST(TestFlydslSdpaRdnaKernelMatch, AWideGenericDecodeTierServesMultiplesOfEightUpTo512)
{
    // H4 (kernel modification 30): the generic decode tier up to 512, its columns split four
    // ways, serves every multiple of 8 past 256 at any query length; no bias, V as wide as
    // Q/K.
    const auto wide
        = makeFlydslSdpaRdnaKernel("bf16", 0, 1, 16, 32, 4, 0, int64_t{512}, 1, 0, 0, int64_t{0});
    const auto matches = [&wide](int64_t headDim, int64_t seqLenQ) {
        SdpaGraphSpec spec;
        spec.headDim = headDim;
        spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
        spec.heads = 8;
        spec.kvHeads = 2;
        spec.seqLenQ = seqLenQ;
        spec.seqLenKv = 600;
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
        return matchesKernel(FLYDSL_SDPA_RDNA_DECODE, fixture.context(), wide, bindingsFor(spec));
    };
    for(const int64_t headDim : {264, 320, 384, 448, 504, 512})
    {
        EXPECT_TRUE(matches(headDim, 1)) << headDim;
        EXPECT_TRUE(matches(headDim, 37)) << headDim << " at 37 query rows";
    }
    EXPECT_TRUE(matches(64, 1)) << "below 256 too, ranked under the narrower tiers";
    EXPECT_FALSE(matches(520, 1)) << "past the tier";
    EXPECT_FALSE(matches(260, 1)) << "not a multiple of 8";

    SdpaGraphSpec biased;
    biased.headDim = 384;
    biased.seqLenQ = 1;
    biased.withBias = true;
    const GraphFixture biasFixture(buildFlydslSdpaRdnaGraph(biased));
    EXPECT_FALSE(
        matchesKernel(FLYDSL_SDPA_RDNA_DECODE, biasFixture.context(), wide, bindingsFor(biased)))
        << "no bias variant";
    SdpaGraphSpec mla;
    mla.headDim = 384;
    mla.valueHeadDim = 256;
    mla.seqLenQ = 1;
    const GraphFixture mlaFixture(buildFlydslSdpaRdnaGraph(mla));
    EXPECT_FALSE(
        matchesKernel(FLYDSL_SDPA_RDNA_DECODE, mlaFixture.context(), wide, bindingsFor(mla)))
        << "V is the runtime head_dim";
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

TEST(TestFlydslSdpaRdnaKernelMatch, PairsAGenericTiersBiasVariantWithAGraphThatHasABias)
{
    // head_dim 80 has no specialized object: the generic tier up to 96 serves it, and a bias
    // runs on that tier's bias object in the bias's dtype, as at a specialized head_dim.
    SdpaGraphSpec biased;
    biased.headDim = 80;
    biased.withBias = true;
    const GraphFixture withBias(buildFlydslSdpaRdnaGraph(biased));
    const auto biasBound = bindingsFor(biased);
    SdpaGraphSpec plainSpec;
    plainSpec.headDim = 80;
    const GraphFixture without(buildFlydslSdpaRdnaGraph(plainSpec));
    const auto plainBound = bindingsFor(plainSpec);

    const auto plain = makeFlydslSdpaRdnaKernel("bf16", 0, 0, 128, 32, 1, 0, int64_t{96});
    const auto bias = makeFlydslSdpaRdnaKernel("bf16", 0, 0, 128, 32, 1, 1, int64_t{96});
    const auto biasQ = makeFlydslSdpaRdnaKernel("bf16", 0, 0, 128, 32, 1, 1, int64_t{96}, 0, 0, 1);
    EXPECT_TRUE(matchesKernel(FLYDSL_SDPA_RDNA, withBias.context(), bias, biasBound));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, withBias.context(), plain, biasBound));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, withBias.context(), biasQ, biasBound));
    EXPECT_TRUE(matchesKernel(FLYDSL_SDPA_RDNA, without.context(), plain, plainBound));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, without.context(), bias, plainBound));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, without.context(), biasQ, plainBound));

    biased.biasDataType = biased.dataType;
    const GraphFixture withBiasQ(buildFlydslSdpaRdnaGraph(biased));
    const auto biasQBound = bindingsFor(biased);
    EXPECT_TRUE(matchesKernel(FLYDSL_SDPA_RDNA, withBiasQ.context(), biasQ, biasQBound));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, withBiasQ.context(), bias, biasQBound));

    // The tier's head-dim range holds with a bias as without: past its largest declines.
    biased.headDim = 104;
    const GraphFixture wider(buildFlydslSdpaRdnaGraph(biased));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, wider.context(), biasQ, bindingsFor(biased)));
}

TEST(TestFlydslSdpaRdnaKernelMatch, AGenericDecodeBiasObjectServesOneTileAtAnyHeadItCovers)
{
    const auto plain = makeFlydslSdpaRdnaKernel("bf16", 0, 1, 16, 32, 1, 0, int64_t{96}, 1);
    const auto bias = makeFlydslSdpaRdnaKernel("bf16", 0, 1, 16, 32, 1, 1, int64_t{96}, 1);
    SdpaGraphSpec spec;
    spec.headDim = 80;
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

    // One 16-row tile, as every decode bias object: an MQA group of 32 declines.
    SdpaGraphSpec mqa = spec;
    mqa.heads = 32;
    mqa.kvHeads = 1;
    const GraphFixture wide(buildFlydslSdpaRdnaGraph(mqa));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA_DECODE, wide.context(), bias, bindingsFor(mqa)));

    SdpaGraphSpec plainSpec = spec;
    plainSpec.withBias = false;
    const GraphFixture without(buildFlydslSdpaRdnaGraph(plainSpec));
    EXPECT_FALSE(
        matchesKernel(FLYDSL_SDPA_RDNA_DECODE, without.context(), bias, bindingsFor(plainSpec)));
    EXPECT_TRUE(
        matchesKernel(FLYDSL_SDPA_RDNA_DECODE, without.context(), plain, bindingsFor(plainSpec)));
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

TEST(TestFlydslSdpaRdnaKernelMatch, ServesAFloatSinkOnTheSinkObjects)
{
    // A FLOAT sink (kernel modifications 31 and 11) binds its dtype and its head stride in f32
    // elements, and runs on the sink objects, prefill and decode, which read either dtype; a
    // query-dtype sink binds 0. The kernels take the dtype in bit 32 of the stride argument.
    SdpaGraphSpec sunk;
    sunk.withSink = true;
    sunk.sinkDataType = DataType::FLOAT;
    sunk.sinkHeadStride = 3;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(sunk));
    const auto bound = bindingsFor(sunk);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_RDNA_SINK_FLOAT_TOKEN), 1);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_RDNA_SINK_STRIDE_TOKEN), 3);
    const auto sink
        = makeFlydslSdpaRdnaKernel("bf16", 64, 0, 128, 32, 1, 0, 64, 0, 0, 0, std::nullopt, 1);
    EXPECT_TRUE(matchesKernel(FLYDSL_SDPA_RDNA, fixture.context(), sink, bound));
    EXPECT_FALSE(matchesKernel(
        FLYDSL_SDPA_RDNA, fixture.context(), makeFlydslSdpaRdnaKernel("bf16", 64, 0), bound));
    sunk.sinkDataType.reset();
    EXPECT_EQ(boundInt(bindingsFor(sunk), FLYDSL_SDPA_RDNA_SINK_FLOAT_TOKEN), 0);

    EXPECT_EQ(flydsl_sdpa::rdnaSinkStrideArgument(3, false), 3);
    EXPECT_EQ(flydsl_sdpa::rdnaSinkStrideArgument(3, true), (int64_t{1} << 32) | 3);
    EXPECT_EQ(flydsl_sdpa::rdnaSinkStrideArgument(0, true), int64_t{1} << 32);
}

TEST(TestFlydslSdpaRdnaKernelMatch, PairsABiasWithASinkWithTheObjectBuiltForBoth)
{
    // N1: a bias and a sink together run on the objects built with both, prefill and decode;
    // an object with only one of them declines the graph.
    SdpaGraphSpec both;
    both.withBias = true;
    both.withSink = true;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(both));
    const auto bound = bindingsFor(both);
    const auto biasAndSink
        = makeFlydslSdpaRdnaKernel("bf16", 64, 0, 128, 32, 1, 1, 64, 0, 0, 0, std::nullopt, 1);
    const auto sinkOnly
        = makeFlydslSdpaRdnaKernel("bf16", 64, 0, 128, 32, 1, 0, 64, 0, 0, 0, std::nullopt, 1);
    const auto biasOnly = makeFlydslSdpaRdnaKernel("bf16", 64, 0, 128, 32, 1, 1);
    EXPECT_TRUE(matchesKernel(FLYDSL_SDPA_RDNA, fixture.context(), biasAndSink, bound));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, fixture.context(), sinkOnly, bound));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, fixture.context(), biasOnly, bound));

    auto decode = both;
    decode.seqLenQ = 1;
    const GraphFixture decodeFixture(buildFlydslSdpaRdnaGraph(decode));
    const auto decodeBound = bindingsFor(decode);
    const auto decodeBoth
        = makeFlydslSdpaRdnaKernel("bf16", 64, 0, 16, 32, 1, 1, 64, 1, 0, 0, std::nullopt, 1);
    const auto decodeSinkOnly
        = makeFlydslSdpaRdnaKernel("bf16", 64, 0, 16, 32, 1, 0, 64, 1, 0, 0, std::nullopt, 1);
    EXPECT_TRUE(
        matchesKernel(FLYDSL_SDPA_RDNA_DECODE, decodeFixture.context(), decodeBoth, decodeBound));
    EXPECT_FALSE(matchesKernel(
        FLYDSL_SDPA_RDNA_DECODE, decodeFixture.context(), decodeSinkOnly, decodeBound));
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
                                                 int64_t hasSink = 0,
                                                 int64_t hasDropout = 0)
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
            // Absent is a plain object: no decode object drops.
            const auto* dropout
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SDPA_AIW_HAS_DROPOUT_FIELD);
            if((dropout == nullptr ? 0 : *dropout) != hasDropout)
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
                                                    int64_t hasSink,
                                                    int64_t hasDropout = 0)
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
               || field(FLYDSL_SDPA_RDNA_HAS_SINK_FIELD) != hasSink
               || std::max<int64_t>(field(FLYDSL_SDPA_AIW_HAS_DROPOUT_FIELD), 0) != hasDropout)
            {
                continue;
            }
            const auto widthV = headDimV.value_or(headDim);
            const bool exact = dim == headDim && dimV == widthV;
            // A padded tier takes both widths at runtime (H1).
            const bool padded = dim == 0 && headDim % 8 == 0 && widthV % 8 == 0
                                && std::max(headDim, widthV) <= dimMax;
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
    // Under padding each batch has its own lengths (TEST-LOCAL ORACLE for padding and for
    // packed sequences, until hipDNN's references serve them): rows past a batch's query length
    // are padding, keys past its key length are masked, and a bottom-right diagonal sits at its
    // own Skv_b - Sq_b. Packed (THD), sequence b is the same problem stored end to end.
    const bool packed = spec.packed();
    // Both lengths are padding with or without padding_mask (hipDNN's SEM-008).
    const bool padded = !spec.seqLensQ.empty() && !spec.seqLensKv.empty();
    // Each batch's slot: the rows it owns in its operands, the sequence extent unless packed.
    // A padded length outside [0, the slot] is clamped to it, as hipDNN's GPU reference clamps
    // one to [0, S]; packed with lengths (cuDNN's form), each slot's rows past its length are
    // padded rows.
    const auto slotQ = [&](int64_t b) {
        return packed ? static_cast<int64_t>(spec.packedSeqLensQ[static_cast<size_t>(b)])
                      : spec.seqLenQ;
    };
    const auto slotKv = [&](int64_t b) {
        return packed ? static_cast<int64_t>(spec.packedSeqLensKv[static_cast<size_t>(b)])
                      : spec.seqLenKv;
    };
    const auto lengthQ = [&](int64_t b) {
        return padded ? std::clamp<int64_t>(spec.seqLensQ[static_cast<size_t>(b)], 0, slotQ(b))
                      : slotQ(b);
    };
    const auto lengthKv = [&](int64_t b) {
        return padded ? std::clamp<int64_t>(spec.seqLensKv[static_cast<size_t>(b)], 0, slotKv(b))
                      : slotKv(b);
    };

    // A left_bound L keeps L keys up to and including the diagonal (cuDNN, SEM-001).
    const auto isMasked = [&](int64_t b, int64_t row, int64_t key) {
        const int64_t align
            = spec.mask == SdpaMask::BOTTOM_RIGHT_CAUSAL ? lengthKv(b) - lengthQ(b) : 0;
        return key >= lengthKv(b) || (right >= 0 && key >= row + 1 + align + right)
               || (left > 0 && key <= row + align - left);
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
                                    spec.withSink ? 1 : 0,
                                    spec.dropoutProbability.value_or(0.0F) > 0.0F ? 1 : 0)
              : findStagedKernel(dtype,
                                 spec.headDim,
                                 causal ? 1 : 0,
                                 spec.withBias ? 1 : 0,
                                 decode ? 1 : 0,
                                 spec.withBias && spec.biasDataType != DataType::FLOAT ? 1 : 0,
                                 spec.valueHeadDim,
                                 spec.withSink ? 1 : 0,
                                 spec.dropoutProbability.value_or(0.0F) > 0.0F ? 1 : 0);
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
    // Device-value overrides: a step in V, NaN in Q or K, or the dropout mask exposed (Q zero,
    // V the identity on its first keys) (separate tensors only).
    if(spec.valueOnesBelowKey || spec.queryNanPosition || spec.keyNanPosition
       || spec.exposeDropoutMask)
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
            for(int64_t h = 0; spec.exposeDropoutMask && h < valueHeads; ++h)
            {
                for(int64_t j = 0; j < spec.seqLenKv; ++j)
                {
                    for(int64_t e = 0; e < dv; ++e)
                    {
                        vTensor->set(b, h, j, e, T(j == e ? 1.0F : 0.0F));
                    }
                }
            }
            for(int64_t h = 0; spec.exposeDropoutMask && h < spec.heads; ++h)
            {
                for(int64_t i = 0; i < spec.seqLenQ; ++i)
                {
                    for(int64_t e = 0; e < d; ++e)
                    {
                        qTensor->set(b, h, i, e, T(0.0F));
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
    const bool expectsNan = spec.queryNanPosition.has_value() || spec.keyNanPosition.has_value()
                            || (spec.withBias && spec.biasNanRows);
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

    // Packed (THD): each operand's sequences end to end, token-major ([T, H, D]), with the
    // values the dense tensors hold, and the cumulative token tables Q/O and K/V go through.
    // O and the LSE come back the same way and are unpacked into the dense tensors, whose
    // rows past a sequence's length then keep the fill.
    std::vector<int32_t> offsetsQ{0};
    std::vector<int32_t> offsetsKv{0};
    for(int64_t b = 0; packed && b < spec.batch; ++b)
    {
        offsetsQ.push_back(offsetsQ.back() + static_cast<int32_t>(slotQ(b)));
        offsetsKv.push_back(offsetsKv.back() + static_cast<int32_t>(slotKv(b)));
    }
    const int64_t tokensQ = std::max<int64_t>(offsetsQ.back(), 1);
    const int64_t tokensKv = std::max<int64_t>(offsetsKv.back(), 1);
    const auto tokenMajor = [](int64_t tokens, int64_t heads, int64_t width) {
        return std::vector<int64_t>{tokens * heads * width, width, heads * width, 1};
    };
    std::optional<DeviceTensor<T>> packedQ;
    std::optional<DeviceTensor<T>> packedK;
    std::optional<DeviceTensor<T>> packedV;
    std::optional<DeviceTensor<T>> packedO;
    std::optional<DeviceTensor<float>> packedStats;
    std::optional<DeviceTensor<int32_t>> tableQ;
    std::optional<DeviceTensor<int32_t>> tableKv;
    if(packed)
    {
        ASSERT_FALSE(spec.packedQkv);
        ASSERT_FALSE(spec.dropoutProbability.has_value());
        packedQ.emplace(std::vector<int64_t>{1, spec.heads, tokensQ, d},
                        tokenMajor(tokensQ, spec.heads, d),
                        0U,
                        false);
        packedK.emplace(std::vector<int64_t>{1, spec.kvHeads, tokensKv, d},
                        tokenMajor(tokensKv, spec.kvHeads, d),
                        0U,
                        false);
        packedV.emplace(std::vector<int64_t>{1, valueHeads, tokensKv, dv},
                        tokenMajor(tokensKv, valueHeads, dv),
                        0U,
                        false);
        packedO.emplace(std::vector<int64_t>{1, spec.heads, tokensQ, dv},
                        tokenMajor(tokensQ, spec.heads, dv),
                        0U,
                        false);
        packedO->fillWith(T(unwritten));
        packedStats.emplace(std::vector<int64_t>{1, spec.heads, tokensQ, 1},
                            std::vector<int64_t>{tokensQ * spec.heads, 1, spec.heads, 1},
                            0U,
                            false);
        packedStats->fillWith(unwritten);
        for(int64_t b = 0; b < spec.batch; ++b)
        {
            const auto rowQ = static_cast<int64_t>(offsetsQ[static_cast<size_t>(b)]);
            const auto rowKv = static_cast<int64_t>(offsetsKv[static_cast<size_t>(b)]);
            for(int64_t h = 0; h < spec.heads; ++h)
            {
                for(int64_t s = 0; s < slotQ(b); ++s)
                {
                    for(int64_t e = 0; e < d; ++e)
                    {
                        packedQ->set(0, h, rowQ + s, e, T(static_cast<float>(qAt(b, h, s, e))));
                    }
                }
            }
            for(int64_t j = 0; j < slotKv(b); ++j)
            {
                for(int64_t h = 0; h < spec.kvHeads; ++h)
                {
                    for(int64_t e = 0; e < d; ++e)
                    {
                        packedK->set(0, h, rowKv + j, e, T(static_cast<float>(kAt(b, h, j, e))));
                    }
                }
                for(int64_t h = 0; h < valueHeads; ++h)
                {
                    for(int64_t e = 0; e < dv; ++e)
                    {
                        packedV->set(0, h, rowKv + j, e, T(static_cast<float>(vAt(b, h, j, e))));
                    }
                }
            }
        }
        packedQ->upload();
        packedK->upload();
        packedV->upload();
        tableQ.emplace(std::vector<int64_t>{spec.batch + 1, 1, 1, 1},
                       std::vector<int64_t>{1, 1, 1, 1},
                       0U,
                       false);
        tableKv.emplace(std::vector<int64_t>{spec.batch + 1, 1, 1, 1},
                        std::vector<int64_t>{1, 1, 1, 1},
                        0U,
                        false);
        for(int64_t b = 0; b <= spec.batch; ++b)
        {
            tableQ->set(b, 0, 0, 0, offsetsQ[static_cast<size_t>(b)]);
            tableKv->set(b, 0, 0, 0, offsetsKv[static_cast<size_t>(b)]);
        }
        tableQ->upload();
        tableKv->upload();
        qPtr = packedQ->device();
        kPtr = packedK->device();
        vPtr = packedV->device();
    }

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
        // Every third query row masked on every key with the dtype's lowest finite value,
        // as a padding mask writes it: the row's softmax is uniform over its keys. Then -inf:
        // on the leading keys of every row (whole KV tiles when it spans them), as a boolean
        // key-padding mask cast to the bias writes it, or on every key of every third row, a
        // row no key reaches.
        for(int64_t b = 0; b < biasDims4[0]; ++b)
        {
            for(int64_t h = 0; h < biasDims4[1]; ++h)
            {
                for(int64_t s = 0; s < biasDims4[2]; ++s)
                {
                    if(spec.biasMasksRows && s % 3 == 0)
                    {
                        for(int64_t j = 0; j < biasDims4[3]; ++j)
                        {
                            bias.set(b, h, s, j, std::numeric_limits<float>::lowest());
                            biasQ.set(b, h, s, j, std::numeric_limits<T>::lowest());
                        }
                    }
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
    if(spec.withBias && (spec.biasNanRows || spec.biasNanPastKeys))
    {
        // NaN on every key of every third row from the second, a row whose O and LSE are NaN;
        // and under padding NaN at and past each batch's key length, which the kernels must
        // mask whatever the bias holds there. That needs a bias of its own per batch.
        ASSERT_TRUE(!spec.biasNanPastKeys
                    || (!spec.seqLensKv.empty() && biasDims4[0] == spec.batch));
        const float nan = std::numeric_limits<float>::quiet_NaN();
        for(int64_t b = 0; b < biasDims4[0]; ++b)
        {
            const int64_t firstPast
                = spec.biasNanPastKeys
                      ? std::clamp<int64_t>(spec.seqLensKv[static_cast<size_t>(b)], 0, biasDims4[3])
                      : biasDims4[3];
            for(int64_t h = 0; h < biasDims4[1]; ++h)
            {
                for(int64_t s = 0; s < biasDims4[2]; ++s)
                {
                    const bool nanRow = spec.biasNanRows && s % 3 == 1;
                    for(int64_t j = nanRow ? 0 : firstPast; j < biasDims4[3]; ++j)
                    {
                        bias.set(b, h, s, j, nan);
                        biasQ.set(b, h, s, j, T(nan));
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

    // The attention sink, one logit per query head, read as [1, H, 1, 1]: in the query's
    // dtype, or in f32 for a FLOAT sink (kernel modifications 31 and 11), whose random values
    // carry low bits no 16-bit dtype holds, so a sink read as 16-bit is caught.
    const bool floatSink = spec.sinkDataType == DataType::FLOAT;
    DeviceTensor<T> sink({1, spec.heads, 1, 1}, {0, 1, 0, 0}, 7U, true);
    DeviceTensor<float> sinkFloat({1, spec.heads, 1, 1}, {0, 1, 0, 0}, 7U, true);
    for(int64_t h = 0; floatSink && h < spec.heads; ++h)
    {
        sinkFloat.set(0,
                      h,
                      0,
                      0,
                      static_cast<float>(sinkFloat.at(0, h, 0, 0))
                          + static_cast<float>((h % 7) + 1) * 1.0e-3F);
    }
    sinkFloat.upload();
    if(spec.sinkNegativeInfinity)
    {
        sink.fillWith(T(-std::numeric_limits<float>::infinity()));
        sinkFloat.fillWith(-std::numeric_limits<float>::infinity());
    }
    else if(spec.sinkValue.has_value())
    {
        sink.fillWith(T(*spec.sinkValue));
        sinkFloat.fillWith(*spec.sinkValue);
    }
    // The logit the kernels read, exactly widened to f32 from the query's dtype.
    const auto sinkAt
        = [&](int64_t h) { return floatSink ? sinkFloat.at(0, h, 0, 0) : sink.at(0, h, 0, 0); };

    float runtimeScale = spec.scale;
    std::vector<hipdnnPluginDeviceBuffer_t> buffers{
        {FLYDSL_SDPA_RDNA_Q_UID, qPtr},
        {FLYDSL_SDPA_RDNA_K_UID, kPtr},
        {FLYDSL_SDPA_RDNA_V_UID, vPtr},
        {FLYDSL_SDPA_RDNA_O_UID, packed ? packedO->device() : o.device()}};
    if(wantStats)
    {
        buffers.push_back(
            {FLYDSL_SDPA_RDNA_STATS_UID, packed ? packedStats->device() : stats.device()});
    }
    if(packed)
    {
        buffers.push_back({FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID, tableQ->device()});
        buffers.push_back({FLYDSL_SDPA_RDNA_RAGGED_OFFSET_KV_UID, tableKv->device()});
    }
    if(spec.withBias)
    {
        buffers.push_back(
            {FLYDSL_SDPA_RDNA_BIAS_UID, biasInQueryType ? biasQ.device() : bias.device()});
    }
    if(spec.withSink)
    {
        buffers.push_back(
            {FLYDSL_SDPA_RDNA_SINK_UID, floatSink ? sinkFloat.device() : sink.device()});
    }
    if(spec.scaleKind == SdpaScale::RUNTIME_TENSOR)
    {
        buffers.push_back({FLYDSL_SDPA_RDNA_SCALE_UID, &runtimeScale});
    }
    // The RNG dump, FLOAT [B, Hq, Sq, Skv] packed, NaN until written: an element the writer
    // skips fails the 0/1 check.
    const RngDumpExtents dumpExtents{spec.batch, spec.heads, spec.seqLenQ, spec.seqLenKv};
    std::optional<DeviceTensor<float>> rngDump;
    if(spec.withRngDump)
    {
        const std::vector<int64_t> dims{spec.batch, spec.heads, spec.seqLenQ, spec.seqLenKv};
        rngDump.emplace(dims, rowMajorStrides(dims), 0U, false);
        rngDump->fillWith(std::numeric_limits<float>::quiet_NaN());
        buffers.push_back({FLYDSL_SDPA_RDNA_RNG_DUMP_UID, rngDump->device()});
    }
    // The per-batch lengths, as INT32 [B, 1, 1, 1] device arrays.
    DeviceTensor<int32_t> seqLensQ({spec.batch, 1, 1, 1}, {1, 1, 1, 1}, 0U, false);
    DeviceTensor<int32_t> seqLensKv({spec.batch, 1, 1, 1}, {1, 1, 1, 1}, 0U, false);
    for(int64_t b = 0; b < spec.batch && !spec.seqLensQ.empty(); ++b)
    {
        seqLensQ.set(b, 0, 0, 0, spec.seqLensQ[static_cast<size_t>(b)]);
    }
    for(int64_t b = 0; b < spec.batch && !spec.seqLensKv.empty(); ++b)
    {
        seqLensKv.set(b, 0, 0, 0, spec.seqLensKv[static_cast<size_t>(b)]);
    }
    seqLensQ.upload();
    seqLensKv.upload();
    // The Philox seed and offset: INT64 device values, or host values given at execute.
    DeviceTensor<int64_t> dropoutSeed({1, 1, 1, 1}, {1, 1, 1, 1}, 0U, false);
    DeviceTensor<int64_t> dropoutOffset({1, 1, 1, 1}, {1, 1, 1, 1}, 0U, false);
    dropoutSeed.set(0, 0, 0, 0, static_cast<int64_t>(spec.dropoutSeed));
    dropoutOffset.set(0, 0, 0, 0, static_cast<int64_t>(spec.dropoutOffset));
    dropoutSeed.upload();
    dropoutOffset.upload();
    auto hostSeed64 = static_cast<int64_t>(spec.dropoutSeed);
    auto hostOffset64 = static_cast<int64_t>(spec.dropoutOffset);
    auto hostSeed32 = static_cast<int32_t>(spec.dropoutSeed);
    auto hostOffset32 = static_cast<int32_t>(spec.dropoutOffset);
    for(const auto& [present, uid, device, dataType, host64, host32] :
        {std::tuple{spec.withDropoutSeed,
                    FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID,
                    dropoutSeed.device(),
                    spec.dropoutSeedDataType.value_or(DataType::INT64),
                    &hostSeed64,
                    &hostSeed32},
         std::tuple{spec.withDropoutOffset,
                    FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_UID,
                    dropoutOffset.device(),
                    spec.dropoutOffsetDataType.value_or(DataType::INT64),
                    &hostOffset64,
                    &hostOffset32}})
    {
        if(!present)
        {
            continue;
        }
        void* host
            = dataType == DataType::INT32 ? static_cast<void*>(host32) : static_cast<void*>(host64);
        buffers.push_back({uid, spec.passByValueTensors.count(uid) != 0 ? host : device});
    }
    if(!spec.seqLensQ.empty())
    {
        buffers.push_back({FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID, seqLensQ.device()});
    }
    if(!spec.seqLensKv.empty())
    {
        buffers.push_back({FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID, seqLensKv.device()});
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
    if(packed)
    {
        packedO->readBack();
        packedStats->readBack();
        for(int64_t b = 0; b < spec.batch; ++b)
        {
            const auto rowQ = static_cast<int64_t>(offsetsQ[static_cast<size_t>(b)]);
            for(int64_t h = 0; h < spec.heads; ++h)
            {
                for(int64_t s = 0; s < slotQ(b); ++s)
                {
                    for(int64_t e = 0; e < dv; ++e)
                    {
                        o.set(b, h, s, e, T(static_cast<float>(packedO->at(0, h, rowQ + s, e))));
                    }
                    stats.set(b, h, s, 0, static_cast<float>(packedStats->at(0, h, rowQ + s, 0)));
                }
            }
        }
    }

    const double scale
        = spec.scaleKind == SdpaScale::ABSENT ? 1.0 : static_cast<double>(spec.scale);
    // The RNG dump: the oracle's mask bit for bit with hipDNN's keep-rate check, and O as
    // hipDNN's CPU reference computes it with the dump as its explicit mask.
    if(rngDump.has_value())
    {
        ASSERT_FALSE(packed || spec.withBias || spec.withSink || padded);
        rngDump->readBack();
        std::vector<float> dumped(dumpExtents.elements());
        for(int64_t b = 0; b < spec.batch; ++b)
        {
            for(int64_t h = 0; h < spec.heads; ++h)
            {
                for(int64_t i = 0; i < spec.seqLenQ; ++i)
                {
                    for(int64_t j = 0; j < spec.seqLenKv; ++j)
                    {
                        dumped[dumpExtents.index(b, h, i, j)]
                            = static_cast<float>(rngDump->at(b, h, i, j));
                    }
                }
            }
        }
        const float p = spec.dropoutProbability.value_or(0.0F);
        expectRngDumpIsTheOracleMask(dumped, dumpExtents, p, spec.dropoutSeed, spec.dropoutOffset);
        RngDumpReferenceProblem reference;
        reference.extents = dumpExtents;
        reference.kvHeads = spec.kvHeads;
        reference.headDim = d;
        reference.headDimV = dv;
        reference.scale = static_cast<float>(scale);
        reference.p = p;
        reference.leftBound = left;
        reference.rightBound = right;
        reference.topLeft = spec.mask != SdpaMask::BOTTOM_RIGHT_CAUSAL;
        expectOutputIsTheReferenceUnderTheDump(
            reference,
            dumped,
            qAt,
            kAt,
            vAt,
            [&o](int64_t b, int64_t h, int64_t s, int64_t e) { return o.at(b, h, s, e); },
            absoluteTolerance);
    }
    const int64_t group = spec.heads / spec.kvHeads;
    // Dropout (TEST-LOCAL ORACLE, FlydslSdpaPhiloxOracle.hpp, until hipDNN's reference serves
    // it): a key's probability is kept with its Philox bit and scaled by 1 / (1 - p) in O;
    // the softmax's normaliser and the LSE stay undropped.
    const float dropoutP = spec.dropoutProbability.value_or(0.0F);
    const bool drops = dropoutP > 0.0F;
    const auto dropThreshold = flydsl_sdpa::dropoutThreshold(dropoutP);
    const double dropScale = drops ? static_cast<double>(flydsl_sdpa::dropoutScale(dropoutP)) : 1.0;
    const auto keeps = [&](int64_t b, int64_t h, int64_t i, int64_t j) {
        return !drops
               || philox_oracle::keeps(spec.dropoutSeed,
                                       spec.dropoutOffset,
                                       philox_oracle::MaskCoordinates{
                                           b, h, i, j, spec.heads, spec.seqLenQ, spec.seqLenKv},
                                       dropThreshold);
    };

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
                if(s >= lengthQ(b))
                {
                    // A padded row: O = 0 and LSE = -inf exactly, sink objects included,
                    // hipDNN's reference contract (RDNA kernel modification 28, aiw 10). A
                    // packed graph's rows past a sequence's slot are not outputs; packed with
                    // lengths, the slot's rows past the length are padded rows (modification 29).
                    if(s >= slotQ(b))
                    {
                        continue;
                    }
                    for(int64_t e = 0; e < dv; ++e)
                    {
                        const double actual = o.at(b, h, s, e);
                        if(actual != 0.0)
                        {
                            worst = std::numeric_limits<double>::max();
                            where = "padded row b=" + std::to_string(b) + " h=" + std::to_string(h)
                                    + " s=" + std::to_string(s)
                                    + " is not 0: " + std::to_string(actual);
                        }
                    }
                    const double actualLse = stats.at(b, h, s, 0);
                    const bool negativeInfinity = std::isinf(actualLse) && actualLse < 0.0;
                    if(wantStats && !negativeInfinity)
                    {
                        worstLse = std::numeric_limits<double>::max();
                        whereLse = "padded row b=" + std::to_string(b) + " h=" + std::to_string(h)
                                   + " s=" + std::to_string(s)
                                   + " LSE is not -inf: " + std::to_string(actualLse);
                    }
                    continue;
                }
                double maxScore = -std::numeric_limits<double>::infinity();
                for(int64_t j = 0; j < spec.seqLenKv; ++j)
                {
                    if(isMasked(b, s, j))
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
                // A row whose every reachable key a bias masks with -inf has a max of -inf; its
                // scores are then exp(-inf) = 0, not exp(-inf - -inf), so it is keyless.
                const double shift = std::isinf(maxScore) && maxScore < 0.0 ? 0.0 : maxScore;
                double sum = 0.0;
                for(int64_t j = 0; j < spec.seqLenKv; ++j)
                {
                    scores[static_cast<size_t>(j)]
                        = isMasked(b, s, j) ? 0.0
                                            : std::exp(scores[static_cast<size_t>(j)] - shift);
                    sum += scores[static_cast<size_t>(j)];
                }
                // A row no key reaches: O = 0 and LSE = -inf, as the reference defines it.
                // Only a sum of exactly 0: a NaN sum (NaN in Q, or in a key the row sees)
                // keeps O and the LSE NaN, as PyTorch's softmax and logsumexp do.
                const bool hasKeys = !(sum == 0.0);
                // The sink's logit joins the denominator of a row that reaches keys, as a key
                // with no value; a row that reaches none has O = 0 and LSE = the sink.
                const double sinkLogit = spec.withSink ? sinkAt(h) : 0.0;
                // The terms are taken against the row's max, or against a sink above it, so a
                // sink far above a row at the mask's lowest finite value does not overflow.
                double lseShift = maxScore;
                if(spec.withSink && hasKeys)
                {
                    if(sinkLogit > maxScore)
                    {
                        const double rescale = std::exp(maxScore - sinkLogit);
                        for(auto& term : scores)
                        {
                            term *= rescale;
                        }
                        sum *= rescale;
                        lseShift = sinkLogit;
                    }
                    sum += std::exp(sinkLogit - lseShift);
                }
                if(wantStats)
                {
                    double expectedLse = -std::numeric_limits<double>::infinity();
                    if(hasKeys)
                    {
                        expectedLse = lseShift + std::log(sum);
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
                    // A keyless row's LSE is the sink itself: an f32 sink's bits exactly, so
                    // one read as 16-bit is a full miss.
                    if(floatSink && !hasKeys && std::isfinite(expectedLse)
                       && static_cast<float>(actualLse) != static_cast<float>(expectedLse))
                    {
                        lseError = std::numeric_limits<double>::max();
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
                        if(keeps(b, h, s, j))
                        {
                            expected += scores[static_cast<size_t>(j)] * vAt(b, vHead, j, e);
                        }
                    }
                    expected = hasKeys ? expected * dropScale / sum : 0.0;
                    // NaN or inf (an unwritten element reads as NaN) is a full miss.
                    const double actual = o.at(b, h, s, e);
                    double error = std::isfinite(actual) ? std::abs(actual - expected)
                                                         : std::numeric_limits<double>::max();
                    if(std::isnan(expected))
                    {
                        // A NaN row must read NaN (the fill is 7 when NaN is expected).
                        error = std::isnan(actual) ? 0.0 : std::numeric_limits<double>::max();
                    }
                    else if(spec.exposeDropoutMask && (actual != 0.0) != (expected != 0.0))
                    {
                        // The mask read off O: a key kept or dropped against the oracle's bit
                        // is a full miss, however small the probability.
                        error = std::numeric_limits<double>::max();
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

    // A sink, against hipDNN's CPU reference as well (FLOAT [1, Hq, 1, 1], SEM-022): with the
    // bias, the band and padding the graph has. Not packed, dropped or NaN: the reference's
    // packed form, dropout and NaN rows are checked above.
    if(spec.withSink && !packed && !drops && !expectsNan)
    {
        SdpaCpuReferenceProblem reference;
        reference.batch = spec.batch;
        reference.heads = spec.heads;
        reference.kvHeads = spec.kvHeads;
        reference.valueHeads = valueHeads;
        reference.seqLenQ = spec.seqLenQ;
        reference.seqLenKv = spec.seqLenKv;
        reference.headDim = d;
        reference.headDimV = dv;
        reference.scale = static_cast<float>(scale);
        reference.leftBound = left;
        reference.rightBound = right;
        reference.topLeft = spec.mask != SdpaMask::BOTTOM_RIGHT_CAUSAL;
        if(padded)
        {
            for(int64_t b = 0; b < spec.batch; ++b)
            {
                reference.seqLensQ.push_back(static_cast<int32_t>(lengthQ(b)));
                reference.seqLensKv.push_back(static_cast<int32_t>(lengthKv(b)));
            }
        }
        for(int64_t h = 0; h < spec.heads; ++h)
        {
            reference.sink.push_back(static_cast<float>(sinkAt(h)));
        }
        reference.withBias = spec.withBias;
        reference.withLse = wantStats;
        expectSdpaIsTheCpuReference(
            reference,
            qAt,
            kAt,
            vAt,
            biasAt,
            [&o](int64_t b, int64_t h, int64_t s, int64_t e) { return o.at(b, h, s, e); },
            [&stats](int64_t b, int64_t h, int64_t s) { return stats.at(b, h, s, 0); },
            absoluteTolerance,
            1e-3);
    }
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

TEST(TestGpuFlydslSdpaRdnaDispatch, KeepsLeftBoundKeysUpToAndIncludingTheDiagonal)
{
    SKIP_IF_NO_DEVICES();
    // cuDNN's left_bound counts the diagonal key: 1 keeps the diagonal alone, so each row's
    // O is its own V row; 2 adds the key before it. Both corners, more keys than queries.
    for(const int64_t left : {int64_t{1}, int64_t{2}})
    {
        SCOPED_TRACE("left_bound " + std::to_string(left));
        SdpaGraphSpec spec;
        spec.leftBound = left;
        spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
        spec.seqLenQ = 40;
        spec.seqLenKv = 40;
        spec.stats = SdpaStats::LSE;
        expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
        spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
        spec.seqLenKv = 72;
        expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
    }
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

    // N2: with an additive bias, f32 and in the query's dtype, rows the bias masks whole
    // with -inf (keyless: O = 0, LSE = -inf).
    spec.withBias = true;
    spec.biasNegativeInfinityRows = true;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
    spec.biasNegativeInfinityRows = false;
    spec.biasDataType = DataType::BFLOAT16;
    spec.mask = SdpaMask::NONE;
    spec.layout = SdpaLayout::BSHD;
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
    // softmax is uniform rather than NaN. f32 and query-dtype biases, prefill and decode, on a
    // specialized head_dim (64) and a generic tier (80, the tiers up to 96).
    for(const int64_t headDim : {64, 80})
    {
        SCOPED_TRACE("head_dim " + std::to_string(headDim));
        SdpaGraphSpec spec;
        spec.headDim = headDim;
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
}

TEST(TestGpuFlydslSdpaRdnaDispatch, AddsABiasThatMasksWholeRowsAndTilesWithNegativeInfinity)
{
    SKIP_IF_NO_DEVICES();
    // COVERAGE §5.7, as the CDNA suite checks it: every third row -inf on every key (O = 0,
    // LSE = -inf), and the other rows -inf on their first 64 keys, two whole 32-key tiles.
    // Bottom-right with 22 more keys than queries, so rows below 42 see only -inf keys and are
    // keyless too, while the rest take their softmax over the keys past 64. f32 and
    // query-dtype biases, prefill and decode, on a specialized head_dim (64) and a generic
    // tier (80).
    for(const int64_t headDim : {64, 80})
    {
        SCOPED_TRACE("head_dim " + std::to_string(headDim));
        SdpaGraphSpec spec;
        spec.headDim = headDim;
        spec.withBias = true;
        spec.biasNegativeInfinityRows = true;
        spec.biasNegativeInfinityLeadingKeys = 64;
        spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
        spec.seqLenQ = 77;
        spec.seqLenKv = 99;
        spec.stats = SdpaStats::LSE;
        expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
        spec.biasDataType = DataType::BFLOAT16;
        expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
        spec.heads = 8;
        spec.kvHeads = 2;
        spec.seqLenQ = 2;
        spec.seqLenKv = 300;
        expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
        spec.biasDataType = DataType::FLOAT;
        expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
    }
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
    // equal-weight error is 0.0154. A specialized head_dim (64) and a generic tier (80).
    constexpr double TOLERANCE = 5e-3;
    for(const int64_t headDim : {64, 80})
    {
        SCOPED_TRACE("head_dim " + std::to_string(headDim));
        spec.headDim = headDim;
        spec.biasDataType = DataType::FLOAT;
        expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, TOLERANCE, true);
        spec.biasDataType = DataType::BFLOAT16;
        expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, TOLERANCE, true);
    }
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

// A bias on a head_dim no specialized object serves runs on the narrowest generic tier's bias
// object: 40 up to 64, 80 up to 96, 112 up to 128, 152 up to 160, 200 up to 224, 240 up to
// 256. Each tier with an f32 bias (bf16 operands) and with one in the query's dtype (f16),
// a per-head bias, both causal variants across the tiers, and the LSE.
TEST(TestGpuFlydslSdpaRdnaDispatch, AddsABiasAtEveryGenericHeadDimTier)
{
    SKIP_IF_NO_DEVICES();
    struct TierCase
    {
        int64_t headDim;
        SdpaMask mask;
    };
    for(const auto& [headDim, mask] : {TierCase{40, SdpaMask::TOP_LEFT_CAUSAL},
                                       TierCase{80, SdpaMask::NONE},
                                       TierCase{112, SdpaMask::BOTTOM_RIGHT_CAUSAL},
                                       TierCase{152, SdpaMask::NONE},
                                       TierCase{200, SdpaMask::TOP_LEFT_CAUSAL},
                                       TierCase{240, SdpaMask::BOTTOM_RIGHT_CAUSAL}})
    {
        SCOPED_TRACE("head_dim " + std::to_string(headDim));
        SdpaGraphSpec spec;
        spec.headDim = headDim;
        spec.mask = mask;
        spec.heads = 4;
        spec.kvHeads = 2;
        spec.seqLenQ = 77;
        spec.seqLenKv = 99;
        spec.stats = SdpaStats::LSE;
        spec.withBias = true;
        spec.biasDims = {1, spec.heads, spec.seqLenQ, spec.seqLenKv};
        expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
        spec.dataType = DataType::HALF;
        spec.biasDataType = DataType::HALF;
        expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
    }
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

// H4 (kernel modification 30): the generic decode tier up to 512, for head dims no RDNA
// prefill object serves: across splits and column tiles, at a short prefill chunk too.
TEST(TestGpuFlydslSdpaRdnaDispatch, DecodesGenericHeadDimsPast256)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.headDim = 264;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 1;
    spec.seqLenKv = 700;
    spec.stats = SdpaStats::LSE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
    spec.headDim = 384;
    spec.seqLenQ = 3;
    spec.dataType = DataType::HALF;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE, true);
    spec.headDim = 504;
    spec.mask = SdpaMask::NONE;
    spec.seqLenQ = 20;
    spec.seqLenKv = 300;
    spec.dataType = DataType::BFLOAT16;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
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

// A bias on a head_dim no specialized decode object serves: the generic decode tier up to 96
// (80) and the one up to 256 with its column split (200), an f32 bias and one in the query's
// dtype, in one split and across several.
TEST(TestGpuFlydslSdpaRdnaDispatch, DecodesWithABiasAtAGenericHeadDim)
{
    SKIP_IF_NO_DEVICES();
    for(const int64_t headDim : {80, 200})
    {
        SCOPED_TRACE("head_dim " + std::to_string(headDim));
        SdpaGraphSpec spec;
        spec.headDim = headDim;
        spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
        spec.heads = 8;
        spec.kvHeads = 2;
        spec.seqLenQ = 2;
        spec.seqLenKv = 300;
        spec.stats = SdpaStats::LSE;
        spec.withBias = true;
        spec.biasDims = {1, spec.heads, spec.seqLenQ, spec.seqLenKv};
        expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
        spec.seqLenQ = 1;
        spec.seqLenKv = 2100;
        spec.biasDataType = DataType::BFLOAT16;
        spec.biasDims = {1, spec.heads, spec.seqLenQ, spec.seqLenKv};
        expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
    }
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

// N1: a bias and a sink together, each in the natural-log domain: a row every key of which the
// bias masks with -inf gets O = 0 and LSE = the sink; a row the bias holds at the lowest finite
// value with a sink far above it takes the whole softmax from the sink; f32 and query-dtype
// biases, prefill and decode (one split and several).
TEST(TestGpuFlydslSdpaRdnaDispatch, AddsABiasAndASinkTogether)
{
    SKIP_IF_NO_DEVICES();
    for(const int64_t headDim : {64, 128})
    {
        SdpaGraphSpec spec;
        spec.headDim = headDim;
        spec.heads = 4;
        spec.kvHeads = 2;
        spec.seqLenQ = 96;
        spec.seqLenKv = 112;
        spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
        spec.stats = SdpaStats::LSE;
        spec.withBias = true;
        spec.withSink = true;
        spec.biasNegativeInfinityRows = true;
        expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
        spec.biasNegativeInfinityRows = false;
        spec.biasMasksRows = true;
        spec.sinkValue = 1.0e30F;
        spec.biasDataType = DataType::BFLOAT16;
        expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
        spec.sinkValue.reset();
        spec.biasMasksRows = false;
        spec.mask = SdpaMask::NONE;
        spec.dataType = DataType::HALF;
        spec.biasDataType = DataType::FLOAT;
        expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
        // Decode: one split, then a long cache split several ways (the merge adds the sink).
        spec.seqLenQ = 1;
        spec.seqLenKv = 64;
        spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
        expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE, true);
        spec.seqLenKv = 4096;
        expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE, true);
    }
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

/// A FLOAT sink, cuDNN's and hipDNN's reference's dtype (kernel modifications 31 and 11), on
/// @p family's sink objects, against the harness's reference and hipDNN's CPU reference: rank
/// 1 and rank 4, rows no key reaches (LSE = the sink's f32 bits), a sink far above the scores,
/// a sink of -inf, a bias and a sink together (f32 and query-dtype biases, a row the bias holds
/// at -inf or its lowest value), and padding. Decode runs one split and several.
void expectFloatSinksMatchTheReference(RdnaFamily family)
{
    using hipdnn_data_sdk::types::bfloat16;
    using hipdnn_data_sdk::types::half;
    const bool decode = family == RdnaFamily::DECODE;
    const auto base = [decode]() {
        SdpaGraphSpec spec;
        spec.headDim = 64;
        spec.heads = 8;
        spec.kvHeads = 2;
        spec.seqLenQ = decode ? 1 : 150;
        spec.seqLenKv = decode ? 2100 : 150;
        spec.mask = decode ? SdpaMask::BOTTOM_RIGHT_CAUSAL : SdpaMask::TOP_LEFT_CAUSAL;
        spec.stats = SdpaStats::LSE;
        spec.withSink = true;
        spec.sinkDataType = DataType::FLOAT;
        return spec;
    };
    {
        SCOPED_TRACE("rank 1");
        auto spec = base();
        expectSdpaMatchesReference<bfloat16>(spec, BF16_TOLERANCE, family);
        if(decode)
        {
            // One split: the main kernel adds the sink; above, the merge did.
            spec.seqLenKv = 60;
            expectSdpaMatchesReference<bfloat16>(spec, BF16_TOLERANCE, family);
        }
    }
    {
        SCOPED_TRACE("rank 4, rows no key reaches, then a sink of -inf");
        auto spec = base();
        spec.sinkRank4 = true;
        spec.dataType = DataType::HALF;
        spec.headDim = 128;
        spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
        spec.seqLenQ = decode ? 4 : 80;
        spec.seqLenKv = decode ? 2 : 48;
        expectSdpaMatchesReference<half>(spec, F16_TOLERANCE, family);
        spec.sinkNegativeInfinity = true;
        expectSdpaMatchesReference<half>(spec, F16_TOLERANCE, family);
    }
    {
        // 200 + 2^-7: bf16 and f16 both round it to 200, past the LSE's tolerance.
        SCOPED_TRACE("a sink far above the scores");
        auto spec = base();
        spec.sinkValue = 200.0078125F;
        expectSdpaMatchesReference<bfloat16>(spec, BF16_TOLERANCE, family);
    }
    for(const int64_t headDim : {64, 128})
    {
        SCOPED_TRACE("a bias and a sink, head_dim " + std::to_string(headDim));
        auto spec = base();
        spec.headDim = headDim;
        spec.withBias = true;
        spec.biasNegativeInfinityRows = true;
        if(!decode)
        {
            // aiw reads the mask's rows in 8-key vectors, an f32 one 32-byte aligned.
            spec.seqLenQ = 160;
            spec.seqLenKv = 160;
            spec.alignment[FLYDSL_SDPA_RDNA_BIAS_UID] = 32;
        }
        expectSdpaMatchesReference<bfloat16>(spec, BF16_TOLERANCE, family);
        spec.biasNegativeInfinityRows = false;
        spec.biasMasksRows = true;
        spec.sinkValue = 1.0e30F;
        spec.biasDataType = DataType::BFLOAT16;
        expectSdpaMatchesReference<bfloat16>(spec, BF16_TOLERANCE, family);
    }
    {
        SCOPED_TRACE("padding");
        auto spec = base();
        spec.batch = 3;
        if(decode)
        {
            spec.seqLenKv = 600;
            setPadding(spec, {1, 1, 0}, {600, 0, 300});
        }
        else
        {
            spec.seqLenQ = 80;
            spec.seqLenKv = 96;
            setPadding(spec, {80, 33, 5}, {64, 96, 17});
        }
        expectSdpaMatchesReference<bfloat16>(spec, BF16_TOLERANCE, family);
    }
}

TEST(TestGpuFlydslSdpaRdnaDispatch, AddsAFloatSink)
{
    SKIP_IF_NO_DEVICES();
    expectFloatSinksMatchTheReference(RdnaFamily::PREFILL);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, DecodesWithAFloatSink)
{
    SKIP_IF_NO_DEVICES();
    expectFloatSinksMatchTheReference(RdnaFamily::DECODE);
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
    kernel.metadata
        = {{FLYDSL_SDPA_RDNA_DTYPE_FIELD, dtype},
           {FLYDSL_SDPA_RDNA_HEAD_DIM_FIELD, headDim},
           {FLYDSL_SDPA_RDNA_CAUSAL_FIELD, masked},
           {FLYDSL_SDPA_RDNA_HEAD_DIM_MAX_FIELD, headDimMax.value_or(headDim)},
           {FLYDSL_SDPA_RDNA_HEAD_DIM_V_FIELD, headDimV.value_or(headDim)},
           {FLYDSL_SDPA_RDNA_HAS_BIAS_FIELD, hasBias},
           {FLYDSL_SDPA_AIW_BIAS_F32_FIELD, biasF32},
           {FLYDSL_SDPA_AIW_BIAS_KSTRIDED_FIELD, int64_t{0}},
           {FLYDSL_SDPA_RDNA_HAS_SINK_FIELD, hasSink},
           {FLYDSL_SDPA_RDNA_BLOCK_M_FIELD, int64_t{128}},
           {FLYDSL_SDPA_AIW_BLOCK_THREADS_FIELD, int64_t{256}},
           {FLYDSL_SDPA_AIW_FILL_SYMBOL_FIELD, std::string("flash_attn_aiw_padfill_kernel_1")}};
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

TEST(TestFlydslSdpaAiwKernelMatch, APaddedTierServesAnyHeadDimPairUpToItsTile)
{
    // H1: hdim_qk and hdim_vo are two runtime widths, so a padded tier serves V narrower or
    // wider than Q/K, each a multiple of 8 and the wider one within the tile.
    const auto tier
        = [](int64_t tile) { return makeAiwKernel("bf16", 0, 0, tile, 0, 0, int64_t{0}); };
    const auto pair = [](int64_t headDim, int64_t headDimV) {
        SdpaGraphSpec spec;
        spec.headDim = headDim;
        spec.valueHeadDim = headDimV;
        return spec;
    };
    EXPECT_TRUE(aiwMatches(pair(128, 64), tier(128)));
    EXPECT_TRUE(aiwMatches(pair(96, 128), tier(128))) << "V wider than Q/K";
    EXPECT_TRUE(aiwMatches(pair(200, 128), tier(224)));
    EXPECT_TRUE(aiwMatches(pair(256, 64), tier(256)));
    EXPECT_TRUE(aiwMatches(pair(64, 256), tier(256)));
    EXPECT_FALSE(aiwMatches(pair(96, 136), tier(128))) << "V past the tile";
    EXPECT_FALSE(aiwMatches(pair(136, 96), tier(128))) << "Q/K past the tile";
    EXPECT_FALSE(aiwMatches(pair(84, 64), tier(128))) << "Q/K not a multiple of 8";

    // V's width not a multiple of 8, on rows the vector loads still reach aligned (one batch,
    // one head, a padded row pitch), so only the width gate can decline it.
    auto oddV = pair(64, 60);
    oddV.batch = 1;
    oddV.heads = 1;
    oddV.kvHeads = 1;
    oddV.valueSequenceStride = 64;
    oddV.outputSequenceStride = 64;
    EXPECT_FALSE(aiwMatches(oddV, tier(128))) << "V not a multiple of 8";
    auto evenV = oddV;
    evenV.valueHeadDim = 56;
    EXPECT_TRUE(aiwMatches(evenV, tier(128))) << "the control";

    // An exact object serves only its own pair (MLA 192/128 is one).
    EXPECT_FALSE(aiwMatches(pair(96, 128), makeAiwKernel("bf16", 96, 0)));
    EXPECT_FALSE(aiwMatches(pair(192, 96), makeAiwKernel("bf16", 192, 0, 192, 0, 0, 128)));
    EXPECT_TRUE(aiwMatches(pair(192, 128), makeAiwKernel("bf16", 192, 0, 192, 0, 0, 128)));
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

TEST(TestFlydslSdpaAiwKernelMatch, PairsAPaddedTiersMaskVariantWithTheMasksDtype)
{
    // head_dim 80 runs on the padded tier up to 96, a mask on that tier's mask object in the
    // mask's dtype, as at an exact head_dim.
    const auto plain = makeAiwKernel("bf16", 0, 0, int64_t{96}, 0, 0, int64_t{0});
    const auto mask = makeAiwKernel("bf16", 0, 0, int64_t{96}, 1, 0, int64_t{0});
    const auto maskF32 = makeAiwKernel("bf16", 0, 0, int64_t{96}, 1, 1, int64_t{0});
    SdpaGraphSpec spec;
    spec.headDim = 80;
    EXPECT_TRUE(aiwMatches(spec, plain));
    EXPECT_FALSE(aiwMatches(spec, mask));
    EXPECT_FALSE(aiwMatches(spec, maskF32));
    spec.withBias = true;
    spec.biasDataType = DataType::FLOAT;
    spec.alignment[FLYDSL_SDPA_RDNA_BIAS_UID] = 32;
    EXPECT_TRUE(aiwMatches(spec, maskF32));
    EXPECT_FALSE(aiwMatches(spec, mask));
    EXPECT_FALSE(aiwMatches(spec, plain));
    spec.biasDataType = DataType::BFLOAT16;
    EXPECT_TRUE(aiwMatches(spec, mask));
    EXPECT_FALSE(aiwMatches(spec, maskF32));
    // The tier's head-dim range holds with a mask as without.
    spec.headDim = 104;
    EXPECT_FALSE(aiwMatches(spec, mask));
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

// ---------------------------------------------------------------------------
// Padding (per-batch lengths under padding_mask): the graph matcher binds it, only the aiw
// objects serve it
// ---------------------------------------------------------------------------

/// A padded graph every gate accepts: two batches of their own lengths.
SdpaGraphSpec paddedSpec()
{
    SdpaGraphSpec spec;
    setPadding(spec, {128, 37}, {100, 128});
    return spec;
}

TEST(TestFlydslSdpaVarlenGraph, BindsTheLengthTensorsUnderThePaddingMask)
{
    const GraphFixture padded(buildFlydslSdpaRdnaGraph(paddedSpec()));
    const auto bound = matchesGraph(FLYDSL_SDPA_RDNA, padded.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(boundInt(*bound, FLYDSL_SDPA_RDNA_SEQ_LEN_Q_TOKEN), FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID);
    EXPECT_EQ(boundInt(*bound, FLYDSL_SDPA_RDNA_SEQ_LEN_KV_TOKEN), FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID);

    const GraphFixture dense(buildFlydslSdpaRdnaGraph());
    const auto denseBound = matchesGraph(FLYDSL_SDPA_RDNA, dense.context());
    ASSERT_TRUE(denseBound.has_value());
    EXPECT_EQ(boundInt(*denseBound, FLYDSL_SDPA_RDNA_SEQ_LEN_Q_TOKEN), -1);
    EXPECT_EQ(boundInt(*denseBound, FLYDSL_SDPA_RDNA_SEQ_LEN_KV_TOKEN), -1);
}

TEST(TestFlydslSdpaVarlenGraph, ReadsBothLengthsWithoutThePaddingMaskAsPadding)
{
    // hipDNN's references read both length tensors as padding with padding_mask false (their
    // SEM-008); cuDNN would refuse the graph. Bound as padding, on every engine.
    auto spec = paddedSpec();
    spec.paddingMask = false;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    for(const auto* engine : {&FLYDSL_SDPA_RDNA, &FLYDSL_SDPA_RDNA_DECODE, &FLYDSL_SDPA_AIW})
    {
        const auto bound = matchesGraph(*engine, fixture.context());
        ASSERT_TRUE(bound.has_value()) << engine->engineName;
        EXPECT_EQ(boundInt(*bound, FLYDSL_SDPA_RDNA_SEQ_LEN_Q_TOKEN),
                  FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID);
        EXPECT_EQ(boundInt(*bound, FLYDSL_SDPA_RDNA_SEQ_LEN_KV_TOKEN),
                  FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID);
    }
}

TEST(TestFlydslSdpaVarlenGraph, BindsLengthsOfOneBatchWhateverTheirStride)
{
    SdpaGraphSpec spec;
    spec.batch = 1;
    setPadding(spec, {100}, {50});
    spec.seqLenBatchStride = 7;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    EXPECT_TRUE(matchesGraph(FLYDSL_SDPA_RDNA, fixture.context()).has_value());
}

/// Each gate on the padding fields, applied to a padded graph the matcher accepts.
std::vector<SdpaFieldDecline> paddingDeclines()
{
    std::vector<SdpaFieldDecline> cases;
    const auto add = [&cases](std::string name, std::function<void(SdpaGraphSpec&)> edit) {
        cases.push_back({std::move(name),
                         [edit = std::move(edit)](SdpaGraphSpec& s) {
                             s = paddedSpec();
                             edit(s);
                         },
                         [](SdpaGraphSpec& s) { s = paddedSpec(); }});
    };
    add("PaddingMaskWithoutQueryLengths", [](SdpaGraphSpec& s) { s.seqLensQ.clear(); });
    add("PaddingMaskWithoutKeyLengths", [](SdpaGraphSpec& s) { s.seqLensKv.clear(); });
    add("QueryLengthsAlone", [](SdpaGraphSpec& s) {
        s.paddingMask = false;
        s.seqLensKv.clear();
    });
    add("KeyLengthsAlone", [](SdpaGraphSpec& s) {
        s.paddingMask = false;
        s.seqLensQ.clear();
    });
    // Without padding_mask, the lengths' own gates still apply (SEM-008 reads them as
    // padding).
    add("Int64LengthsWithoutThePaddingMask", [](SdpaGraphSpec& s) {
        s.paddingMask = false;
        s.seqLenDataType = DataType::INT64;
    });
    add("VirtualQueryLengthsWithoutThePaddingMask", [](SdpaGraphSpec& s) {
        s.paddingMask = false;
        s.virtualTensors.insert(FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID);
    });
    add("VirtualKeyLengthsWithoutThePaddingMask", [](SdpaGraphSpec& s) {
        s.paddingMask = false;
        s.virtualTensors.insert(FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID);
    });
    add("RaggedQueryWithLengthsWithoutThePaddingMask", [](SdpaGraphSpec& s) {
        s.paddingMask = false;
        s.raggedTensors.insert(FLYDSL_SDPA_RDNA_Q_UID);
    });
    add("Int64Lengths", [](SdpaGraphSpec& s) { s.seqLenDataType = DataType::INT64; });
    add("FloatLengths", [](SdpaGraphSpec& s) { s.seqLenDataType = DataType::FLOAT; });
    add("LengthsOfTwoPerBatch", [](SdpaGraphSpec& s) { s.seqLenDims = {s.batch, 2, 1, 1}; });
    add("LengthsOfRankOne", [](SdpaGraphSpec& s) { s.seqLenDims = {s.batch}; });
    add("LengthsOfRankFive", [](SdpaGraphSpec& s) { s.seqLenDims = {s.batch, 1, 1, 1, 1}; });
    add("LengthsForAnotherBatch", [](SdpaGraphSpec& s) { s.seqLenDims = {s.batch + 1, 1, 1, 1}; });
    add("LengthsAtAStrideOfTwo", [](SdpaGraphSpec& s) { s.seqLenBatchStride = 2; });
    add("QueryLengthsPassedByValue",
        [](SdpaGraphSpec& s) { s.passByValueTensors.insert(FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID); });
    add("KeyLengthsPassedByValue",
        [](SdpaGraphSpec& s) { s.passByValueTensors.insert(FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID); });
    add("VirtualQueryLengths",
        [](SdpaGraphSpec& s) { s.virtualTensors.insert(FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID); });
    add("VirtualKeyLengths",
        [](SdpaGraphSpec& s) { s.virtualTensors.insert(FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID); });
    add("RaggedQueryLengths",
        [](SdpaGraphSpec& s) { s.raggedTensors.insert(FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID); });
    add("RaggedKeyLengths",
        [](SdpaGraphSpec& s) { s.raggedTensors.insert(FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID); });
    add("QueryLengthsAlignedTo2",
        [](SdpaGraphSpec& s) { s.alignment[FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID] = 2; });
    add("KeyLengthsAlignedTo2",
        [](SdpaGraphSpec& s) { s.alignment[FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID] = 2; });
    add("PaddedRaggedQuery",
        [](SdpaGraphSpec& s) { s.raggedTensors.insert(FLYDSL_SDPA_RDNA_Q_UID); });
    return cases;
}

INSTANTIATE_TEST_SUITE_P(Padding,
                         TestFlydslSdpaRdnaFieldDeclines,
                         ::testing::ValuesIn(paddingDeclines()),
                         [](const auto& info) { return info.param.name; });

TEST(TestFlydslSdpaVarlenKernelMatch, AdmitsAPaddedGraphOnEveryRdnaEngine)
{
    // Every object reads the per-batch lengths at runtime (RDNA kernel modification 26; the
    // aiw kernel's varlen path), so the objects that serve a graph dense serve it padded.
    struct Case
    {
        const PackSymbols* engine;
        KernelDefinition kernel;
        int64_t seqLenQ;
    };
    for(const auto& c :
        {Case{&FLYDSL_SDPA_RDNA, makeFlydslSdpaRdnaKernel("bf16", 64, 0), 128},
         Case{&FLYDSL_SDPA_RDNA_DECODE,
              makeFlydslSdpaRdnaKernel("bf16", 64, 0, 16, 32, 1, 0, std::nullopt, 1),
              1},
         Case{&FLYDSL_SDPA_AIW, makeAiwKernel("bf16", 64, 0), 128}})
    {
        SdpaGraphSpec padded;
        padded.seqLenQ = c.seqLenQ;
        setPadding(padded, {static_cast<int32_t>(c.seqLenQ), 1}, {100, 128});
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(padded));
        const auto bound = matchesGraph(*c.engine, fixture.context());
        ASSERT_TRUE(bound.has_value());
        EXPECT_TRUE(matchesKernel(*c.engine, fixture.context(), c.kernel, *bound))
            << c.engine->engineName;
    }
}

TEST(TestFlydslSdpaVarlenKernelMatch, DeclinesAPaddedGraphOnAnAiwObjectWithoutAFillKernel)
{
    // A padded graph's padded rows are written by the object's fill kernel (aiw modification
    // 10): an object that names none declines it, and still serves the graph dense.
    auto kernel = makeAiwKernel("bf16", 64, 0);
    kernel.metadata.erase(FLYDSL_SDPA_AIW_FILL_SYMBOL_FIELD);
    EXPECT_FALSE(aiwMatches(paddedSpec(), kernel));
    EXPECT_TRUE(aiwMatches(SdpaGraphSpec{}, kernel));
    EXPECT_TRUE(aiwMatches(paddedSpec(), makeAiwKernel("bf16", 64, 0)));

    // A fill grid of more than 65,535 tiles of 16 rows (its y extent) declines too; the main
    // kernel's tiles of 128 rows still fit, so the graph is served dense.
    auto longPadded = paddedSpec();
    longPadded.seqLenQ = 1100000;
    EXPECT_FALSE(aiwMatches(longPadded, makeAiwKernel("bf16", 64, 0)));
    longPadded.paddingMask = false;
    longPadded.seqLensQ.clear();
    longPadded.seqLensKv.clear();
    EXPECT_TRUE(aiwMatches(longPadded, makeAiwKernel("bf16", 64, 0)));
}

TEST(TestFlydslSdpaVarlenKernelMatch, AdmitsABottomRightWindowUnderPaddingOnRdna)
{
    // The RDNA kernels compute each batch's diagonal from its own lengths for every bound;
    // only the aiw kernel takes one literal window per launch.
    auto window = paddedSpec();
    window.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    window.leftBound = 16;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(window));
    const auto bound = matchesGraph(FLYDSL_SDPA_RDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_TRUE(matchesKernel(
        FLYDSL_SDPA_RDNA, fixture.context(), makeFlydslSdpaRdnaKernel("bf16", 64, 1), *bound));
    EXPECT_FALSE(aiwMatches(window, makeAiwKernel("bf16", 64, 1)));
}

TEST(TestFlydslSdpaVarlenKernelMatch, AdmitsEveryMaskTheAiwKernelResolvesPerSequence)
{
    for(const auto mask :
        {SdpaMask::TOP_LEFT_CAUSAL, SdpaMask::BOTTOM_RIGHT_CAUSAL, SdpaMask::SLIDING_WINDOW})
    {
        auto spec = paddedSpec();
        spec.mask = mask;
        EXPECT_TRUE(aiwMatches(spec, makeAiwKernel("bf16", 64, 1)));
    }
    // A top-left band: one literal bound is right for every sequence.
    auto band = paddedSpec();
    band.rightBound = 5;
    EXPECT_TRUE(aiwMatches(band, makeAiwKernel("bf16", 64, 1)));
}

TEST(TestFlydslSdpaVarlenKernelMatch, DeclinesABottomRightWindowOrBandUnderPadding)
{
    // A bottom-right diagonal at each batch's own Skv_b - Sq_b: the kernel resolves only
    // bottom-right causal per sequence, so any other bottom-right bound declines padded and
    // is served dense.
    auto window = paddedSpec();
    window.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    window.leftBound = 16;
    auto band = paddedSpec();
    band.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    band.rightBound = 3;
    auto leftOnly = paddedSpec();
    leftOnly.diagonalAlignment
        = hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    leftOnly.leftBound = 16;
    for(auto* spec : {&window, &band, &leftOnly})
    {
        EXPECT_FALSE(aiwMatches(*spec, makeAiwKernel("bf16", 64, 1)));
        spec->paddingMask = false;
        spec->seqLensQ.clear();
        spec->seqLensKv.clear();
        EXPECT_TRUE(aiwMatches(*spec, makeAiwKernel("bf16", 64, 1)));
    }
}

// ---------------------------------------------------------------------------
// Packed (THD) sequences, compact: Q/O and K/V ragged through cumulative token tables. The
// graph matcher binds the tables; the aiw objects read them, the RDNA objects decline.
// ---------------------------------------------------------------------------

/// Two sequences packed end to end, the LSE packed alike.
SdpaGraphSpec packedSpec()
{
    SdpaGraphSpec spec;
    spec.stats = SdpaStats::LSE;
    setPacked(spec, {128, 37}, {100, 128});
    return spec;
}

TEST(TestFlydslSdpaPackedGraph, BindsTheTablesOfPackedOperands)
{
    const GraphFixture packed(buildFlydslSdpaRdnaGraph(packedSpec()));
    const auto bound = matchesGraph(FLYDSL_SDPA_AIW, packed.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(boundInt(*bound, FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_TOKEN),
              FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID);
    EXPECT_EQ(boundInt(*bound, FLYDSL_SDPA_RDNA_RAGGED_OFFSET_KV_TOKEN),
              FLYDSL_SDPA_RDNA_RAGGED_OFFSET_KV_UID);
    EXPECT_EQ(boundInt(*bound, FLYDSL_SDPA_RDNA_SEQ_LEN_Q_TOKEN), -1);
    EXPECT_EQ(boundInt(*bound, FLYDSL_SDPA_RDNA_STATS_TOKEN), FLYDSL_SDPA_RDNA_STATS_UID);

    const GraphFixture dense(buildFlydslSdpaRdnaGraph());
    const auto denseBound = matchesGraph(FLYDSL_SDPA_AIW, dense.context());
    ASSERT_TRUE(denseBound.has_value());
    EXPECT_EQ(boundInt(*denseBound, FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_TOKEN), -1);
    EXPECT_EQ(boundInt(*denseBound, FLYDSL_SDPA_RDNA_RAGGED_OFFSET_KV_TOKEN), -1);
}

TEST(TestFlydslSdpaPackedGraph, AcceptsOneTableForQueriesAndKeysAndNoLse)
{
    // Self-attention: one table for every operand.
    auto shared = packedSpec();
    setPacked(shared, {128, 37}, {128, 37});
    shared.raggedOffsetOf[FLYDSL_SDPA_RDNA_K_UID] = FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID;
    shared.raggedOffsetOf[FLYDSL_SDPA_RDNA_V_UID] = FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID;
    const GraphFixture sharedFixture(buildFlydslSdpaRdnaGraph(shared));
    const auto bound = matchesGraph(FLYDSL_SDPA_AIW, sharedFixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(boundInt(*bound, FLYDSL_SDPA_RDNA_RAGGED_OFFSET_KV_TOKEN),
              FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID);

    auto noLse = packedSpec();
    noLse.stats = SdpaStats::NONE;
    const GraphFixture noLseFixture(buildFlydslSdpaRdnaGraph(noLse));
    EXPECT_TRUE(matchesGraph(FLYDSL_SDPA_AIW, noLseFixture.context()).has_value());
}

TEST(TestFlydslSdpaPackedGraph, BindsTheLengthsOfPackedSequencesWithOrWithoutThePaddingMask)
{
    // P3, cuDNN's packed form: each table entry a slot, the lengths its rows and keys.
    for(const bool paddingMask : {true, false})
    {
        auto spec = packedSpec();
        setPadding(spec, {100, 37}, {90, 64});
        spec.paddingMask = paddingMask;
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
        const auto bound = matchesGraph(FLYDSL_SDPA_RDNA, fixture.context());
        ASSERT_TRUE(bound.has_value()) << paddingMask;
        EXPECT_EQ(boundInt(*bound, FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_TOKEN),
                  FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID);
        EXPECT_EQ(boundInt(*bound, FLYDSL_SDPA_RDNA_RAGGED_OFFSET_KV_TOKEN),
                  FLYDSL_SDPA_RDNA_RAGGED_OFFSET_KV_UID);
        EXPECT_EQ(boundInt(*bound, FLYDSL_SDPA_RDNA_SEQ_LEN_Q_TOKEN),
                  FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID);
        EXPECT_EQ(boundInt(*bound, FLYDSL_SDPA_RDNA_SEQ_LEN_KV_TOKEN),
                  FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID);
    }
}

TEST(TestFlydslSdpaPackedKernelMatch, RunsPackedSequencesWithLengthsOnTheRdnaPrefillObjectsOnly)
{
    // The RDNA prefill kernel reads both (modification 29); the decode objects decline packed
    // graphs, and the aiw objects decline this form (their fill kernel addresses padded rows
    // by batch).
    auto spec = packedSpec();
    setPadding(spec, {100, 37}, {90, 64});
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    for(const auto* engine : {&FLYDSL_SDPA_RDNA, &FLYDSL_SDPA_RDNA_DECODE, &FLYDSL_SDPA_AIW})
    {
        const auto bound = matchesGraph(*engine, fixture.context());
        ASSERT_TRUE(bound.has_value()) << engine->engineName;
        const auto kernel
            = engine == &FLYDSL_SDPA_AIW
                  ? makeAiwKernel("bf16", 64, 0)
                  : makeFlydslSdpaRdnaKernel("bf16",
                                             64,
                                             0,
                                             engine == &FLYDSL_SDPA_RDNA ? 64 : 16,
                                             32,
                                             1,
                                             0,
                                             std::nullopt,
                                             engine == &FLYDSL_SDPA_RDNA_DECODE ? 1 : 0);
        EXPECT_EQ(matchesKernel(*engine, fixture.context(), kernel, *bound),
                  engine == &FLYDSL_SDPA_RDNA)
            << engine->engineName;
    }
    // The aiw object serves either half alone (the controls).
    EXPECT_TRUE(aiwMatches(packedSpec(), makeAiwKernel("bf16", 64, 0)));
    auto padded = paddedSpec();
    EXPECT_TRUE(aiwMatches(padded, makeAiwKernel("bf16", 64, 0)));
}

/// Each gate on packed sequences, applied to a packed graph the matcher accepts.
std::vector<SdpaFieldDecline> packedDeclines()
{
    std::vector<SdpaFieldDecline> cases;
    const auto add = [&cases](std::string name, std::function<void(SdpaGraphSpec&)> edit) {
        cases.push_back({std::move(name),
                         [edit = std::move(edit)](SdpaGraphSpec& s) {
                             s = packedSpec();
                             edit(s);
                         },
                         [](SdpaGraphSpec& s) { s = packedSpec(); }});
    };
    constexpr int64_t Q_TABLE = FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID;
    constexpr int64_t KV_TABLE = FLYDSL_SDPA_RDNA_RAGGED_OFFSET_KV_UID;
    // An operand left dense beside packed ones, or through the other table.
    for(const auto& [uid, name] : {std::pair{FLYDSL_SDPA_RDNA_Q_UID, "Query"},
                                   std::pair{FLYDSL_SDPA_RDNA_K_UID, "Key"},
                                   std::pair{FLYDSL_SDPA_RDNA_V_UID, "Value"},
                                   std::pair{FLYDSL_SDPA_RDNA_O_UID, "Output"}})
    {
        const int64_t operand = uid;
        const bool querySide = uid == FLYDSL_SDPA_RDNA_Q_UID || uid == FLYDSL_SDPA_RDNA_O_UID;
        add(std::string("Dense") + name,
            [operand](SdpaGraphSpec& s) { s.unpackedTensors.insert(operand); });
        add(std::string(name) + "CountingElements",
            [operand](SdpaGraphSpec& s) { s.raggedMultiplierOf[operand] = 1; });
        add(std::string(name) + "ThroughTheOtherTable", [operand, querySide](SdpaGraphSpec& s) {
            s.raggedOffsetOf[operand] = querySide ? KV_TABLE : Q_TABLE;
        });
    }
    add("KeyThroughATableTheGraphLacks", [](SdpaGraphSpec& s) {
        s.raggedOffsetOf[FLYDSL_SDPA_RDNA_K_UID] = FLYDSL_SDPA_RDNA_RAGGED_OFFSET_UID;
        s.raggedOffsetOf[FLYDSL_SDPA_RDNA_V_UID] = FLYDSL_SDPA_RDNA_RAGGED_OFFSET_UID;
    });
    add("HeadMajorPacking", [](SdpaGraphSpec& s) { s.layout = SdpaLayout::BHSD; });
    // The LSE: dense, or packed other than Q is.
    add("DenseLse", [](SdpaGraphSpec& s) { s.packedStats = false; });
    add("LseThroughTheKeyTable",
        [](SdpaGraphSpec& s) { s.raggedOffsetOf[FLYDSL_SDPA_RDNA_STATS_UID] = KV_TABLE; });
    add("LseCountingElements",
        [](SdpaGraphSpec& s) { s.raggedMultiplierOf[FLYDSL_SDPA_RDNA_STATS_UID] = 1; });
    add("HeadMajorLse", [](SdpaGraphSpec& s) { s.statsSequenceMajor = false; });
    // The tables.
    add("Int64Offsets", [](SdpaGraphSpec& s) { s.raggedOffsetDataType = DataType::INT64; });
    add("OffsetsForAnotherBatch",
        [](SdpaGraphSpec& s) { s.raggedOffsetDims = {s.batch, 1, 1, 1}; });
    add("OffsetsOfRankOne", [](SdpaGraphSpec& s) { s.raggedOffsetDims = {s.batch + 1}; });
    add("OffsetsOfTwoPerBatch",
        [](SdpaGraphSpec& s) { s.raggedOffsetDims = {s.batch + 1, 2, 1, 1}; });
    add("OffsetsWithASequenceExtent",
        [](SdpaGraphSpec& s) { s.raggedOffsetDims = {s.batch + 1, 1, 2, 1}; });
    add("OffsetsWithAHeadDimExtent",
        [](SdpaGraphSpec& s) { s.raggedOffsetDims = {s.batch + 1, 1, 1, 2}; });
    add("OffsetsAtAStrideOfTwo", [](SdpaGraphSpec& s) { s.raggedOffsetBatchStride = 2; });
    for(const auto& [uid, name] :
        {std::pair{Q_TABLE, "QueryOffsets"}, std::pair{KV_TABLE, "KeyOffsets"}})
    {
        const int64_t table = uid;
        add(std::string("Virtual") + name,
            [table](SdpaGraphSpec& s) { s.virtualTensors.insert(table); });
        add(std::string(name) + "PassedByValue",
            [table](SdpaGraphSpec& s) { s.passByValueTensors.insert(table); });
        add(std::string("Ragged") + name,
            [table](SdpaGraphSpec& s) { s.raggedTensors.insert(table); });
        add(std::string(name) + "AlignedTo2",
            [table](SdpaGraphSpec& s) { s.alignment[table] = 2; });
    }
    // What packed sequences are not served with. Both lengths are (P3, cuDNN's packed form);
    // one alone, the switch alone, or a length tensor the kernels cannot read is not.
    add("PackedUnderThePaddingMaskAlone", [](SdpaGraphSpec& s) { s.paddingMask = true; });
    add("PackedWithQueryLengths", [](SdpaGraphSpec& s) { s.seqLensQ = {128, 37}; });
    add("PackedWithKeyLengths", [](SdpaGraphSpec& s) { s.seqLensKv = {100, 128}; });
    add("PackedWithQueryLengthsUnderThePaddingMask", [](SdpaGraphSpec& s) {
        s.paddingMask = true;
        s.seqLensQ = {128, 37};
    });
    add("PackedWithInt64Lengths", [](SdpaGraphSpec& s) {
        setPadding(s, {128, 37}, {100, 128});
        s.seqLenDataType = DataType::INT64;
    });
    add("PackedWithLengthsOfAnotherBatch", [](SdpaGraphSpec& s) {
        setPadding(s, {128, 37}, {100, 128});
        s.seqLenDims = {s.batch + 1, 1, 1, 1};
    });
    // Each length tensor is checked on its own.
    for(const auto& [uid, name] : {std::pair{FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID, "Query"},
                                   std::pair{FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID, "Key"}})
    {
        const int64_t length = uid;
        add(std::string("PackedWithAVirtual") + name + "Length", [length](SdpaGraphSpec& s) {
            setPadding(s, {128, 37}, {100, 128});
            s.virtualTensors.insert(length);
        });
    }
    // With both lengths, the packing is held to the same rules as without.
    for(const auto& [uid, name] : {std::pair{FLYDSL_SDPA_RDNA_Q_UID, "Query"},
                                   std::pair{FLYDSL_SDPA_RDNA_K_UID, "Key"},
                                   std::pair{FLYDSL_SDPA_RDNA_V_UID, "Value"},
                                   std::pair{FLYDSL_SDPA_RDNA_O_UID, "Output"}})
    {
        const int64_t operand = uid;
        add(std::string("Dense") + name + "WithLengths", [operand](SdpaGraphSpec& s) {
            setPadding(s, {128, 37}, {100, 128});
            s.unpackedTensors.insert(operand);
        });
    }
    add("Int64OffsetsWithLengths", [](SdpaGraphSpec& s) {
        setPadding(s, {128, 37}, {100, 128});
        s.raggedOffsetDataType = DataType::INT64;
    });
    add("PackedWithABias", [](SdpaGraphSpec& s) { s.withBias = true; });
    add("PackedAndDropping", [](SdpaGraphSpec& s) { setDropout(s, 0.1F); });
    return cases;
}

INSTANTIATE_TEST_SUITE_P(Packed,
                         TestFlydslSdpaRdnaFieldDeclines,
                         ::testing::ValuesIn(packedDeclines()),
                         [](const auto& info) { return info.param.name; });

TEST(TestFlydslSdpaPackedKernelMatch, RunsOnTheAiwAndRdnaPrefillObjectsOnly)
{
    // The aiw kernel reads the tables (VarlenBits STACKED, CUMULATIVE, REUSE), and so does the
    // RDNA prefill kernel (modification 29: bit 3 of its flags); the decode kernel and its
    // merge index every operand by batch, so a decode object declines a packed graph it
    // serves dense.
    struct Case
    {
        const PackSymbols* engine;
        KernelDefinition kernel;
        int64_t seqLenQ;
        bool servesPacked;
    };
    for(const auto& c :
        {Case{&FLYDSL_SDPA_AIW, makeAiwKernel("bf16", 64, 0), 128, true},
         Case{&FLYDSL_SDPA_RDNA, makeFlydslSdpaRdnaKernel("bf16", 64, 0), 128, true},
         Case{&FLYDSL_SDPA_RDNA,
              makeFlydslSdpaRdnaKernel(
                  "bf16", 0, 0, 64, 32, 1, 0, int64_t{128}, 0, 1, 0, int64_t{0}),
              128,
              true},
         Case{&FLYDSL_SDPA_RDNA_DECODE,
              makeFlydslSdpaRdnaKernel("bf16", 64, 0, 16, 32, 1, 0, std::nullopt, 1),
              1,
              false}})
    {
        SdpaGraphSpec dense;
        dense.seqLenQ = c.seqLenQ;
        auto packed = dense;
        setPacked(packed, {static_cast<int32_t>(c.seqLenQ), 1}, {100, 128});
        for(const auto& [spec, served] :
            {std::pair{&dense, true}, std::pair{&packed, c.servesPacked}})
        {
            const GraphFixture fixture(buildFlydslSdpaRdnaGraph(*spec));
            const auto bound = matchesGraph(*c.engine, fixture.context());
            ASSERT_TRUE(bound.has_value()) << c.engine->engineName;
            EXPECT_EQ(matchesKernel(*c.engine, fixture.context(), c.kernel, *bound), served)
                << c.engine->engineName << (spec == &packed ? " packed" : " dense");
        }
    }
}

TEST(TestFlydslSdpaPackedKernelMatch, ResolvesOnlyBottomRightCausalPerSequence)
{
    // As under padding: each packed sequence has its own lengths, and the aiw kernel resolves
    // only bottom-right causal per sequence; top-left bounds are one literal for all.
    for(const auto mask :
        {SdpaMask::TOP_LEFT_CAUSAL, SdpaMask::BOTTOM_RIGHT_CAUSAL, SdpaMask::SLIDING_WINDOW})
    {
        auto spec = packedSpec();
        spec.mask = mask;
        EXPECT_TRUE(aiwMatches(spec, makeAiwKernel("bf16", 64, 1)));
    }
    auto window = packedSpec();
    window.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    window.leftBound = 16;
    EXPECT_FALSE(aiwMatches(window, makeAiwKernel("bf16", 64, 1)));
    window.packedSeqLensQ.clear();
    window.packedSeqLensKv.clear();
    window.packedStats = false;
    EXPECT_TRUE(aiwMatches(window, makeAiwKernel("bf16", 64, 1)));
}

TEST(TestFlydslSdpaPackedKernelMatch, ServesEveryMaskPackedOnTheRdnaPrefillObjects)
{
    // The RDNA prefill kernel moves every bottom-right bound with each sequence's own lengths
    // (modification 26), packed as padded, so a bottom-right window or band runs packed too.
    const auto causalKernel = makeFlydslSdpaRdnaKernel("bf16", 64, 1);
    for(const auto mask :
        {SdpaMask::TOP_LEFT_CAUSAL, SdpaMask::BOTTOM_RIGHT_CAUSAL, SdpaMask::SLIDING_WINDOW})
    {
        auto spec = packedSpec();
        spec.mask = mask;
        spec.leftBound = 16;
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
        const auto bound = matchesGraph(FLYDSL_SDPA_RDNA, fixture.context());
        ASSERT_TRUE(bound.has_value());
        EXPECT_TRUE(matchesKernel(FLYDSL_SDPA_RDNA, fixture.context(), causalKernel, *bound));
    }
}

// ---------------------------------------------------------------------------
// Dropout: the graph matcher binds a probability with its seed and offset, only the aiw
// dropout objects apply it
// ---------------------------------------------------------------------------

/// A graph that drops with probability 0.1, from a device INT64 seed and offset.
SdpaGraphSpec droppingSpec()
{
    SdpaGraphSpec spec;
    setDropout(spec, 0.1F);
    return spec;
}

/// An aiw object built with dropout.
KernelDefinition makeAiwDropoutKernel(const std::string& dtype, int64_t headDim, int64_t masked)
{
    auto kernel = makeAiwKernel(dtype, headDim, masked);
    kernel.metadata[FLYDSL_SDPA_AIW_HAS_DROPOUT_FIELD] = int64_t{1};
    return kernel;
}

TEST(TestFlydslSdpaDropoutGraph, BindsTheProbabilitySeedAndOffset)
{
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(droppingSpec()));
    const auto bound = matchesGraph(FLYDSL_SDPA_RDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());
    const auto bits = static_cast<int32_t>(boundInt(*bound, FLYDSL_SDPA_RDNA_DROPOUT_P_TOKEN));
    float probability = 0.0F;
    std::memcpy(&probability, &bits, sizeof(probability));
    EXPECT_EQ(probability, 0.1F);
    EXPECT_EQ(boundInt(*bound, FLYDSL_SDPA_RDNA_DROPOUT_SEED_TOKEN),
              FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID);
    EXPECT_EQ(boundInt(*bound, FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_TOKEN),
              FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_UID);

    const GraphFixture dense(buildFlydslSdpaRdnaGraph());
    const auto denseBound = matchesGraph(FLYDSL_SDPA_RDNA, dense.context());
    ASSERT_TRUE(denseBound.has_value());
    EXPECT_EQ(boundInt(*denseBound, FLYDSL_SDPA_RDNA_DROPOUT_P_TOKEN), 0);
    EXPECT_EQ(boundInt(*denseBound, FLYDSL_SDPA_RDNA_DROPOUT_SEED_TOKEN), -1);
    EXPECT_EQ(boundInt(*denseBound, FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_TOKEN), -1);
}

/// The seed and offset in every form the matcher takes.
std::vector<AcceptedCase> dropoutAccepts()
{
    const auto with = [](const std::function<void(SdpaGraphSpec&)>& edit) {
        auto spec = droppingSpec();
        edit(spec);
        return spec;
    };
    return {
        AcceptedCase{"DeviceInt64SeedAndOffset", droppingSpec()},
        AcceptedCase{"SeedAndOffsetPassedByValue", with([](SdpaGraphSpec& s) {
                         s.passByValueTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID);
                         s.passByValueTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_UID);
                     })},
        AcceptedCase{"Int32SeedAndOffsetPassedByValue", with([](SdpaGraphSpec& s) {
                         s.passByValueTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID);
                         s.passByValueTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_UID);
                         s.dropoutSeedDataType = DataType::INT32;
                         s.dropoutOffsetDataType = DataType::INT32;
                     })},
        AcceptedCase{"ProbabilityZeroAlone", with([](SdpaGraphSpec& s) {
                         s.withDropoutSeed = false;
                         s.withDropoutOffset = false;
                         s.dropoutProbability = 0.0F;
                     })},
        AcceptedCase{"ProbabilityZeroWithSeedAndOffset",
                     with([](SdpaGraphSpec& s) { s.dropoutProbability = 0.0F; })},
        AcceptedCase{"ProbabilityZeroUnderPadding", with([](SdpaGraphSpec& s) {
                         s.dropoutProbability = 0.0F;
                         setPadding(s, {128, 3}, {128, 40});
                     })},
        AcceptedCase{"ProbabilityJustBelowOne",
                     with([](SdpaGraphSpec& s) { s.dropoutProbability = 0.99F; })},
    };
}

INSTANTIATE_TEST_SUITE_P(Dropout,
                         TestFlydslSdpaRdnaGraphAccepts,
                         ::testing::ValuesIn(dropoutAccepts()),
                         [](const auto& info) { return info.param.name; });

/// Each gate on the dropout fields, applied to a dropping graph the matcher accepts.
std::vector<SdpaFieldDecline> dropoutDeclines()
{
    using Builder = hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesBuilder;
    std::vector<SdpaFieldDecline> cases;
    const auto add = [&cases](std::string name, std::function<void(SdpaGraphSpec&)> edit) {
        cases.push_back({std::move(name),
                         [edit = std::move(edit)](SdpaGraphSpec& s) {
                             s = droppingSpec();
                             edit(s);
                         },
                         [](SdpaGraphSpec& s) { s = droppingSpec(); }});
    };
    const auto field = [&add](std::string name, std::function<void(Builder&)> write) {
        add(std::move(name),
            [write = std::move(write)](SdpaGraphSpec& s) { s.extraAttributes = write; });
    };
    constexpr int64_t AUX = FLYDSL_SDPA_RDNA_AUX_UID;
    field("WithADropoutMask", [](Builder& b) { b.add_dropout_mask_tensor_uid(AUX); });
    field("WithADropoutScale", [](Builder& b) { b.add_dropout_scale_tensor_uid(AUX); });
    // A one-element dump no writer fills; a well-formed one is served (sdpaRngDumpDeclines).
    field("WithAnRngDump", [](Builder& b) { b.add_rng_dump_tensor_uid(AUX); });
    for(const auto& [name, p] :
        {std::pair{"ProbabilityOne", 1.0F},
         std::pair{"ProbabilityAboveOne", 1.5F},
         std::pair{"ProbabilityNegative", -0.1F},
         std::pair{"ProbabilityNaN", std::numeric_limits<float>::quiet_NaN()},
         std::pair{"ProbabilityInfinite", std::numeric_limits<float>::infinity()}})
    {
        const float probability = p;
        add(name, [probability](SdpaGraphSpec& s) { s.dropoutProbability = probability; });
    }
    add("WithoutASeed", [](SdpaGraphSpec& s) { s.withDropoutSeed = false; });
    add("WithoutAnOffset", [](SdpaGraphSpec& s) { s.withDropoutOffset = false; });
    add("SeedAndOffsetWithoutAProbability", [](SdpaGraphSpec& s) { s.dropoutProbability.reset(); });
    add("Int32SeedOnTheDevice", [](SdpaGraphSpec& s) { s.dropoutSeedDataType = DataType::INT32; });
    add("Int32OffsetOnTheDevice",
        [](SdpaGraphSpec& s) { s.dropoutOffsetDataType = DataType::INT32; });
    add("FloatSeedPassedByValue", [](SdpaGraphSpec& s) {
        s.dropoutSeedDataType = DataType::FLOAT;
        s.passByValueTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID);
    });
    add("SeedOfTwoValues", [](SdpaGraphSpec& s) { s.dropoutSeedDims = {1, 1, 1, 2}; });
    add("SeedOfRankOne", [](SdpaGraphSpec& s) { s.dropoutSeedDims = {1}; });
    add("SeedOfRankFive", [](SdpaGraphSpec& s) { s.dropoutSeedDims = {1, 1, 1, 1, 1}; });
    add("VirtualSeed",
        [](SdpaGraphSpec& s) { s.virtualTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID); });
    add("VirtualOffset",
        [](SdpaGraphSpec& s) { s.virtualTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_UID); });
    add("RaggedSeed",
        [](SdpaGraphSpec& s) { s.raggedTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID); });
    add("RaggedOffset",
        [](SdpaGraphSpec& s) { s.raggedTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_UID); });
    add("SeedAlignedTo4",
        [](SdpaGraphSpec& s) { s.alignment[FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID] = 4; });
    add("OffsetAlignedTo4",
        [](SdpaGraphSpec& s) { s.alignment[FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_UID] = 4; });
    // A bad seed declines even where nothing is drawn: the matcher gates what it reads.
    add("Int32DeviceSeedAtProbabilityZero", [](SdpaGraphSpec& s) {
        s.dropoutProbability = 0.0F;
        s.dropoutSeedDataType = DataType::INT32;
    });
    // Each family serves dropout and padding apart first.
    add("UnderPadding", [](SdpaGraphSpec& s) { setPadding(s, {128, 3}, {128, 40}); });
    return cases;
}

INSTANTIATE_TEST_SUITE_P(Dropout,
                         TestFlydslSdpaRdnaFieldDeclines,
                         ::testing::ValuesIn(dropoutDeclines()),
                         [](const auto& info) { return info.param.name; });

TEST(TestFlydslSdpaDropoutKernelMatch, LeavesADroppingGraphToTheAiwDropoutObjects)
{
    // Prefill and decode: the plain objects serve probability 0 and decline a drop.
    struct Case
    {
        const PackSymbols* engine;
        KernelDefinition kernel;
        int64_t seqLenQ;
    };
    for(const auto& c :
        {Case{&FLYDSL_SDPA_RDNA, makeFlydslSdpaRdnaKernel("bf16", 64, 0), 128},
         Case{&FLYDSL_SDPA_RDNA_DECODE,
              makeFlydslSdpaRdnaKernel("bf16", 64, 0, 16, 32, 1, 0, std::nullopt, 1),
              1}})
    {
        auto zero = droppingSpec();
        zero.seqLenQ = c.seqLenQ;
        zero.dropoutProbability = 0.0F;
        auto dropping = zero;
        dropping.dropoutProbability = 0.25F;
        const GraphFixture zeroFixture(buildFlydslSdpaRdnaGraph(zero));
        const GraphFixture droppingFixture(buildFlydslSdpaRdnaGraph(dropping));
        const auto zeroBound = matchesGraph(*c.engine, zeroFixture.context());
        const auto droppingBound = matchesGraph(*c.engine, droppingFixture.context());
        ASSERT_TRUE(zeroBound.has_value());
        ASSERT_TRUE(droppingBound.has_value());
        EXPECT_TRUE(matchesKernel(*c.engine, zeroFixture.context(), c.kernel, *zeroBound))
            << c.engine->engineName;
        EXPECT_FALSE(matchesKernel(*c.engine, droppingFixture.context(), c.kernel, *droppingBound))
            << c.engine->engineName;
    }

    // RDNA prefill (kernel modification 27): a dropping graph runs on a dropout object only.
    auto prefillDropout = makeFlydslSdpaRdnaKernel("bf16", 64, 0);
    prefillDropout.metadata[FLYDSL_SDPA_AIW_HAS_DROPOUT_FIELD] = int64_t{1};
    {
        auto zeroSpec = droppingSpec();
        zeroSpec.dropoutProbability = 0.0F;
        const GraphFixture droppingFixture(buildFlydslSdpaRdnaGraph(droppingSpec()));
        const GraphFixture zeroFixture(buildFlydslSdpaRdnaGraph(zeroSpec));
        const auto droppingBound = matchesGraph(FLYDSL_SDPA_RDNA, droppingFixture.context());
        const auto zeroBound = matchesGraph(FLYDSL_SDPA_RDNA, zeroFixture.context());
        ASSERT_TRUE(droppingBound.has_value());
        ASSERT_TRUE(zeroBound.has_value());
        EXPECT_TRUE(matchesKernel(
            FLYDSL_SDPA_RDNA, droppingFixture.context(), prefillDropout, *droppingBound));
        EXPECT_FALSE(
            matchesKernel(FLYDSL_SDPA_RDNA, zeroFixture.context(), prefillDropout, *zeroBound));
    }

    // aiw: a dropping graph runs on a dropout object only; probability 0 and no dropout on a
    // plain one only.
    auto zero = droppingSpec();
    zero.dropoutProbability = 0.0F;
    EXPECT_TRUE(aiwMatches(droppingSpec(), makeAiwDropoutKernel("bf16", 64, 0)));
    EXPECT_FALSE(aiwMatches(droppingSpec(), makeAiwKernel("bf16", 64, 0)));
    EXPECT_TRUE(aiwMatches(zero, makeAiwKernel("bf16", 64, 0)));
    EXPECT_FALSE(aiwMatches(zero, makeAiwDropoutKernel("bf16", 64, 0)));
    EXPECT_TRUE(aiwMatches(SdpaGraphSpec{}, makeAiwKernel("bf16", 64, 0)));
    EXPECT_FALSE(aiwMatches(SdpaGraphSpec{}, makeAiwDropoutKernel("bf16", 64, 0)));
}

TEST(TestFlydslSdpaDropoutKernelMatch, DropsThroughABiasOnlyOnTheTwinsBuiltForBoth)
{
    // R1 and R2: a dropping graph with a bias runs on the dropout twins built with a bias; the
    // dropout twin without one and the bias object without dropout decline it.
    auto spec = droppingSpec();
    spec.withBias = true;
    spec.biasDataType = spec.dataType;
    const auto withDropout = [](KernelDefinition kernel, int64_t hasDropout) {
        kernel.metadata[FLYDSL_SDPA_AIW_HAS_DROPOUT_FIELD] = hasDropout;
        return kernel;
    };
    // aiw: the mask in the query's dtype (bias_f32 0).
    const auto aiwBoth = withDropout(makeAiwKernel("bf16", 64, 0, std::nullopt, 1, 0), 1);
    const auto aiwTwin = withDropout(makeAiwKernel("bf16", 64, 0), 1);
    const auto aiwMask = withDropout(makeAiwKernel("bf16", 64, 0, std::nullopt, 1, 0), 0);
    EXPECT_TRUE(aiwMatches(spec, aiwBoth));
    EXPECT_FALSE(aiwMatches(spec, aiwTwin));
    EXPECT_FALSE(aiwMatches(spec, aiwMask));
    // RDNA: an f32 bias (bias_q 0), so only the bias and dropout gates tell the objects apart.
    auto f32Bias = spec;
    f32Bias.biasDataType = DataType::FLOAT;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(f32Bias));
    const auto bound = bindingsFor(f32Bias);
    const auto rdnaBoth
        = withDropout(makeFlydslSdpaRdnaKernel("bf16", 64, 0, 128, 32, 1, 1, 64), 1);
    const auto rdnaTwin = withDropout(makeFlydslSdpaRdnaKernel("bf16", 64, 0), 1);
    const auto rdnaBias
        = withDropout(makeFlydslSdpaRdnaKernel("bf16", 64, 0, 128, 32, 1, 1, 64), 0);
    EXPECT_TRUE(matchesKernel(FLYDSL_SDPA_RDNA, fixture.context(), rdnaBoth, bound));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, fixture.context(), rdnaTwin, bound));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_RDNA, fixture.context(), rdnaBias, bound));
}

TEST(TestFlydslSdpaDropoutKernelMatch, ServesOnlyTheShippedDropoutClasses)
{
    // COVERAGE §5.10: exact head dims 64, 128, 256, with or without a mask (R1), no sink. A
    // drop with a sink, at a padded head dim or MLA finds no staged object.
    const auto* staged = findLoadedSet(FLYDSL_SDPA_AIW.engineName);
    if(staged == nullptr)
    {
        GTEST_SKIP() << "this build staged no aiw objects";
    }
    const auto served = [staged](const SdpaGraphSpec& spec) {
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec), packedArchDeviceProperties());
        const auto bound = matchesGraph(FLYDSL_SDPA_AIW, fixture.context());
        if(!bound.has_value())
        {
            return false;
        }
        for(const auto& pack : staged->packs)
        {
            for(const auto& kernel : pack.kernels)
            {
                const KernelDefinition definition{kernel.id,
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
                if(matchesKernel(FLYDSL_SDPA_AIW, fixture.context(), definition, *bound))
                {
                    return true;
                }
            }
        }
        return false;
    };
    for(const int64_t headDim : {64, 128, 256})
    {
        for(const auto mask : {SdpaMask::NONE, SdpaMask::TOP_LEFT_CAUSAL})
        {
            auto spec = droppingSpec();
            spec.headDim = headDim;
            spec.mask = mask;
            EXPECT_TRUE(served(spec)) << "head_dim " << headDim;
        }
    }
    auto bias = droppingSpec();
    bias.withBias = true;
    bias.biasDataType = bias.dataType;
    auto sink = droppingSpec();
    sink.withSink = true;
    auto padded = droppingSpec();
    padded.headDim = 80;
    auto mla = droppingSpec();
    mla.headDim = 192;
    mla.valueHeadDim = 128;
    // A mask in the query's dtype and an f32 one, both causal variants (R1).
    for(const auto biasType : {bias.dataType, DataType::FLOAT})
    {
        for(const auto mask : {SdpaMask::NONE, SdpaMask::BOTTOM_RIGHT_CAUSAL})
        {
            auto masked = bias;
            masked.biasDataType = biasType;
            masked.mask = mask;
            // An f32 mask is loaded 32 bytes at a time (the aiw f32-mask objects' rule).
            masked.alignment[FLYDSL_SDPA_RDNA_BIAS_UID] = 32;
            EXPECT_TRUE(served(masked)) << "a drop with a mask";
        }
    }
    for(const auto* spec : {&sink, &padded, &mla})
    {
        EXPECT_FALSE(served(*spec));
        auto plain = *spec;
        plain.dropoutProbability = 0.0F;
        EXPECT_TRUE(served(plain)) << "the same graph undropped is served";
    }
}

TEST(TestFlydslSdpaDropoutDispatch, NeedsWorkspaceOnlyForASeedGivenByValueOnRdnaToo)
{
    const auto& handler = dispatchHandler(FLYDSL_SDPA_RDNA);
    auto kernel = makeFlydslSdpaRdnaKernel("bf16", 64, 0);
    kernel.metadata[FLYDSL_SDPA_AIW_HAS_DROPOUT_FIELD] = int64_t{1};
    const auto workspace = [&handler, &kernel](const SdpaGraphSpec& spec) {
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
        const auto bound = matchesGraph(FLYDSL_SDPA_RDNA, fixture.context());
        EXPECT_TRUE(bound.has_value());
        return bound.has_value() ? handler.workspaceBytes(fixture.context(), *bound, kernel)
                                 : size_t{0};
    };
    EXPECT_EQ(workspace(droppingSpec()), 0U);
    auto seedByValue = droppingSpec();
    seedByValue.passByValueTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID);
    EXPECT_EQ(workspace(seedByValue), 8U);
}

TEST(TestFlydslSdpaDropoutKernelMatch, ServesTheShippedRdnaDropoutClassesOnGfx11Only)
{
    // Kernel modification 27's twins: the specialized head dims, no bias or sink, on the
    // gfx11-generic shard; on gfx12 a dropping graph is the aiw objects' (decision Q3).
    const auto* staged = findLoadedSet(FLYDSL_SDPA_RDNA.engineName);
    if(staged == nullptr)
    {
        GTEST_SKIP() << "this build staged no RDNA prefill objects";
    }
    // A member of each family this build staged: a build packs only its GPU_TARGETS.
    std::string gfx11;
    std::string gfx12;
    for(const auto& pack : staged->packs)
    {
        for(const auto& arch : pack.arch)
        {
            if(gfx11.empty() && arch.rfind("gfx11", 0) == 0)
            {
                gfx11 = arch;
            }
            if(gfx12.empty() && arch.rfind("gfx12", 0) == 0)
            {
                gfx12 = arch;
            }
        }
    }
    if(gfx11.empty())
    {
        GTEST_SKIP() << "this build staged no gfx11 RDNA shard";
    }
    const auto servedOn = [staged](const SdpaGraphSpec& spec, const std::string& arch) {
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
        const auto bound = matchesGraph(FLYDSL_SDPA_RDNA, fixture.context());
        if(!bound.has_value())
        {
            return false;
        }
        for(const auto& pack : staged->packs)
        {
            for(const auto& kernel : pack.kernels)
            {
                const auto& arches = kernel.arch.empty() ? pack.arch : kernel.arch;
                if(std::find(arches.begin(), arches.end(), arch) == arches.end())
                {
                    continue;
                }
                const KernelDefinition definition{kernel.id,
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
                if(matchesKernel(FLYDSL_SDPA_RDNA, fixture.context(), definition, *bound))
                {
                    return true;
                }
            }
        }
        return false;
    };
    for(const int64_t headDim : {64, 96, 128, 256})
    {
        for(const auto mask : {SdpaMask::NONE, SdpaMask::TOP_LEFT_CAUSAL})
        {
            auto spec = droppingSpec();
            spec.headDim = headDim;
            spec.mask = mask;
            EXPECT_TRUE(servedOn(spec, gfx11)) << "head_dim " << headDim;
            if(!gfx12.empty())
            {
                EXPECT_FALSE(servedOn(spec, gfx12)) << "head_dim " << headDim;
            }
        }
    }
    auto bias = droppingSpec();
    bias.withBias = true;
    bias.biasDataType = bias.dataType;
    auto sink = droppingSpec();
    sink.withSink = true;
    auto generic = droppingSpec();
    generic.headDim = 80;
    // A bias in the query's dtype and an f32 one, at every specialized head dim, both causal
    // variants (R2).
    for(const int64_t headDim : {64, 96, 128, 256})
    {
        for(const auto biasType : {bias.dataType, DataType::FLOAT})
        {
            for(const auto mask : {SdpaMask::NONE, SdpaMask::BOTTOM_RIGHT_CAUSAL})
            {
                auto biased = bias;
                biased.headDim = headDim;
                biased.biasDataType = biasType;
                biased.mask = mask;
                EXPECT_TRUE(servedOn(biased, gfx11)) << "a drop with a bias, head_dim " << headDim;
            }
        }
    }
    for(const auto* spec : {&sink, &generic})
    {
        EXPECT_FALSE(servedOn(*spec, gfx11));
        auto plain = *spec;
        plain.dropoutProbability = 0.0F;
        EXPECT_TRUE(servedOn(plain, gfx11)) << "the same graph undropped is served";
    }
}

TEST(TestFlydslSdpaDropoutDispatch, NeedsWorkspaceOnlyForASeedGivenByValue)
{
    const auto& handler = dispatchHandler(FLYDSL_SDPA_AIW);
    const auto workspace = [&handler](const SdpaGraphSpec& spec) {
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
        const auto bound = matchesGraph(FLYDSL_SDPA_AIW, fixture.context());
        EXPECT_TRUE(bound.has_value());
        return bound.has_value() ? handler.workspaceBytes(fixture.context(),
                                                          *bound,
                                                          makeAiwDropoutKernel("bf16", 64, 0))
                                 : size_t{0};
    };
    EXPECT_EQ(workspace(droppingSpec()), 0U);
    auto seedByValue = droppingSpec();
    seedByValue.passByValueTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID);
    EXPECT_EQ(workspace(seedByValue), 8U);
    auto offsetByValue = droppingSpec();
    offsetByValue.passByValueTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_UID);
    EXPECT_EQ(workspace(offsetByValue), 0U);
    auto zero = seedByValue;
    zero.dropoutProbability = 0.0F;
    EXPECT_EQ(workspace(zero), 0U) << "nothing is drawn at probability 0";
}

// ---------------------------------------------------------------------------
// RNG dump: the keep mask a dropping launch applied, written out by the mask writer
// (FlydslSdpaRngDump.hpp)
// ---------------------------------------------------------------------------

INSTANTIATE_TEST_SUITE_P(RngDump,
                         TestFlydslSdpaRdnaFieldDeclines,
                         ::testing::ValuesIn(sdpaRngDumpDeclines()),
                         [](const auto& info) { return info.param.name; });

/// The writer's own engine, which serves no graph.
const PackSymbols FLYDSL_SDPA_DROPOUT_MASK{"hipkernel:flydsl_sdpa_dropout_mask",
                                           "hipkernel.flydsl_sdpa_dropout_mask.graph_match",
                                           "",
                                           "hipkernel.flydsl_sdpa_dropout_mask.kernel_match",
                                           "hipkernel.flydsl_sdpa_dropout_mask.score",
                                           "hipkernel.flydsl_sdpa_dropout_mask.dispatch",
                                           "",
                                           "",
                                           ""};

/// Every staged object of @p engine built with dropout.
std::vector<KernelDefinition> stagedDropoutKernels(const PackSymbols& engine)
{
    std::vector<KernelDefinition> kernels;
    const auto* staged = findLoadedSet(engine.engineName);
    for(const auto& pack : staged == nullptr ? decltype(staged->packs){} : staged->packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto* dropout
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SDPA_AIW_HAS_DROPOUT_FIELD);
            if(dropout == nullptr || *dropout != 1)
            {
                continue;
            }
            kernels.push_back(KernelDefinition{kernel.id,
                                               pack.id,
                                               pack.dispatchId,
                                               kernel.source,
                                               kernel.metadata,
                                               kernel.priority,
                                               kernel.arch.empty() ? pack.arch : kernel.arch,
                                               kernel.originDirectory,
                                               kernel.name,
                                               kernel.treeRoot,
                                               kernel.sources});
        }
    }
    return kernels;
}

TEST(TestFlydslSdpaRngDumpGraph, BindsTheDumpOrNone)
{
    SdpaGraphSpec withDump;
    setDropoutWithRngDump(withDump, 0.2F);
    const GraphFixture dumping(buildFlydslSdpaRdnaGraph(withDump));
    const GraphFixture dropping(buildFlydslSdpaRdnaGraph(droppingSpec()));
    const GraphFixture dense(buildFlydslSdpaRdnaGraph());
    for(const auto* engine : {&FLYDSL_SDPA_RDNA, &FLYDSL_SDPA_RDNA_DECODE, &FLYDSL_SDPA_AIW})
    {
        const auto bound = matchesGraph(*engine, dumping.context());
        ASSERT_TRUE(bound.has_value()) << engine->engineName;
        EXPECT_EQ(boundInt(*bound, FLYDSL_SDPA_RDNA_RNG_DUMP_TOKEN), FLYDSL_SDPA_RDNA_RNG_DUMP_UID);
        for(const auto* fixture : {&dropping, &dense})
        {
            const auto none = matchesGraph(*engine, fixture->context());
            ASSERT_TRUE(none.has_value()) << engine->engineName;
            EXPECT_EQ(boundInt(*none, FLYDSL_SDPA_RDNA_RNG_DUMP_TOKEN), -1);
        }
    }
}

// An axis of one element: no index multiplies its stride, so any stride is the packed layout.
TEST(TestFlydslSdpaRngDumpGraph, TakesAnyStrideOnAUnitAxis)
{
    SdpaGraphSpec spec;
    setDropoutWithRngDump(spec, 0.2F);
    spec.batch = 1;
    spec.heads = 1;
    spec.kvHeads = 1;
    spec.rngDumpStrides = {7, 3, spec.seqLenKv, 1};
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    const auto bound = matchesGraph(FLYDSL_SDPA_RDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(boundInt(*bound, FLYDSL_SDPA_RDNA_RNG_DUMP_TOKEN), FLYDSL_SDPA_RDNA_RNG_DUMP_UID);
}

// No family serves packed dropout, so no graph reaches servableRngDump's packed gate: it is
// tested on the function, with a dense and a packed binding for the same dropping graph.
TEST(TestFlydslSdpaRngDump, DeclinesADumpOverPackedOperands)
{
    SdpaGraphSpec spec;
    setDropoutWithRngDump(spec, 0.2F);
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    const auto* attributes = flydsl_sdpa::sdpaNode(fixture.context());
    ASSERT_NE(attributes, nullptr);
    const flydsl_sdpa::SdpaDropout dropout{
        0.2F, FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID, FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_UID};
    flydsl_sdpa::SdpaVarlen packed;
    packed.raggedOffsetQ = FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID;
    packed.raggedOffsetKv = FLYDSL_SDPA_RDNA_RAGGED_OFFSET_KV_UID;
    const auto serve = [&](const flydsl_sdpa::SdpaVarlen& varlen) {
        return flydsl_sdpa::servableRngDump(fixture.context(),
                                            *attributes,
                                            dropout,
                                            varlen,
                                            spec.batch,
                                            spec.heads,
                                            spec.seqLenQ,
                                            spec.seqLenKv);
    };
    EXPECT_EQ(serve(flydsl_sdpa::SdpaVarlen{}),
              std::optional<int64_t>(FLYDSL_SDPA_RDNA_RNG_DUMP_UID));
    EXPECT_FALSE(serve(packed).has_value());
}

// A dump is written by the writer beside the dropout object: an object whose shard does not
// ship it (its directory moved) declines the dump but still serves the dropping graph.
TEST(TestFlydslSdpaRngDumpKernelMatch, NeedsTheWriterInTheObjectsShard)
{
    SdpaGraphSpec withDump;
    setDropoutWithRngDump(withDump, 0.2F);
    const GraphFixture dumping(buildFlydslSdpaRdnaGraph(withDump));
    const GraphFixture dropping(buildFlydslSdpaRdnaGraph(droppingSpec()));
    size_t checked = 0;
    for(const auto* engine : {&FLYDSL_SDPA_RDNA, &FLYDSL_SDPA_AIW})
    {
        const auto dumpBound = matchesGraph(*engine, dumping.context());
        const auto dropBound = matchesGraph(*engine, dropping.context());
        ASSERT_TRUE(dumpBound.has_value() && dropBound.has_value());
        for(const auto& kernel : stagedDropoutKernels(*engine))
        {
            if(!matchesKernel(*engine, dropping.context(), kernel, *dropBound))
            {
                continue;
            }
            EXPECT_TRUE(matchesKernel(*engine, dumping.context(), kernel, *dumpBound))
                << kernel.name;
            auto moved = kernel;
            moved.originDirectory = std::filesystem::path("/nonexistent") / "FlyDSL"
                                    / kernel.originDirectory.parent_path().filename()
                                    / kernel.originDirectory.filename();
            EXPECT_FALSE(matchesKernel(*engine, dumping.context(), moved, *dumpBound))
                << kernel.name;
            EXPECT_TRUE(matchesKernel(*engine, dropping.context(), moved, *dropBound))
                << kernel.name;
            ++checked;
        }
    }
    if(!flydslShippedArches("sdpa_rdna").empty() || !flydslShippedArches("sdpa_aiw").empty())
    {
        EXPECT_GT(checked, 0U) << "no staged dropout object serves the bf16 d64 dropping graph";
    }
}

// Every staged dropout object's shard ships the writer, as the packer wrote it: a kpack object
// with the argument list the natives launch and its workgroup size.
TEST(TestFlydslSdpaRngDumpKernel, EveryDroppingShardShipsTheWriter)
{
    size_t checked = 0;
    for(const auto* engine : {&FLYDSL_SDPA_RDNA, &FLYDSL_SDPA_AIW})
    {
        for(const auto& kernel : stagedDropoutKernels(*engine))
        {
            ASSERT_TRUE(flydsl_sdpa::hasDropoutMaskWriter(kernel)) << kernel.name;
            for(const auto& arch : kernel.arch)
            {
                const auto mask = flydsl_sdpa::dropoutMaskKernel(kernel, arch);
                EXPECT_EQ(mask.source.kind, KernelSourceKind::KPACK) << arch;
                EXPECT_EQ(mask.source.symbol, "sdpa_dropout_mask_kernel_0") << arch;
                EXPECT_EQ(mask.getIntMetadata("block_threads"), 256) << arch;
                const auto& expected = flydsl_sdpa::dropoutMaskSignature();
                ASSERT_EQ(mask.source.signature.size(), expected.size()) << arch;
                for(size_t i = 0; i < expected.size(); ++i)
                {
                    EXPECT_EQ(mask.source.signature[i].kind, expected[i].kind) << arch << " " << i;
                    EXPECT_EQ(mask.source.signature[i].size, expected[i].size) << arch << " " << i;
                    EXPECT_EQ(mask.source.signature[i].offset, expected[i].offset)
                        << arch << " " << i;
                }
                ++checked;
            }
        }
    }
    if(!flydslShippedArches("sdpa_rdna").empty() || !flydslShippedArches("sdpa_aiw").empty())
    {
        EXPECT_GT(checked, 0U);
    }
}

// The writer's engine serves no graph: no graph asks for a keep mask alone.
TEST(TestFlydslSdpaDropoutMaskEngine, ServesNoGraph)
{
    SdpaGraphSpec spec;
    setDropoutWithRngDump(spec, 0.2F);
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    EXPECT_FALSE(matchesGraph(FLYDSL_SDPA_DROPOUT_MASK, fixture.context()).has_value());
    const auto* staged = findLoadedSet(FLYDSL_SDPA_DROPOUT_MASK.engineName);
    if(staged == nullptr)
    {
        EXPECT_TRUE(flydslShippedArches("sdpa_dropout_mask").empty());
        return;
    }
    for(const auto& pack : staged->packs)
    {
        ASSERT_EQ(pack.kernels.size(), 1U);
        const auto& kernel = pack.kernels.front();
        const KernelDefinition definition{kernel.id,
                                          pack.id,
                                          pack.dispatchId,
                                          kernel.source,
                                          kernel.metadata,
                                          kernel.priority,
                                          pack.arch,
                                          kernel.originDirectory,
                                          kernel.name,
                                          kernel.treeRoot,
                                          kernel.sources};
        EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_DROPOUT_MASK, fixture.context(), definition, {}));
        EXPECT_THROW(static_cast<void>(dispatchHandler(FLYDSL_SDPA_DROPOUT_MASK)
                                           .prepare(fixture.context(), {}, definition)),
                     hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

TEST(TestGpuFlydslSdpaRdnaDispatch, WritesTheRngDumpItApplied)
{
    SKIP_IF_NO_DEVICES();
    // The RDNA dropout objects are gfx11-generic; gfx12 drops on the aiw objects.
    if(packedArchDeviceProperties().gcnArchName.rfind("gfx11", 0) != 0)
    {
        GTEST_SKIP() << "the RDNA dropout objects ship for gfx11 only";
    }
    struct Case
    {
        int64_t headDim;
        SdpaMask mask;
        int64_t seqLenKv;
        float p;
    };
    // Keys not a multiple of 4 (a draw past the last key), masked keys (causal both ways),
    // GQA, and a seed past 32 bits with an offset that carries into the counter's high word.
    for(const auto& c : {Case{64, SdpaMask::NONE, 70, 0.25F},
                         Case{128, SdpaMask::TOP_LEFT_CAUSAL, 96, 0.1F},
                         Case{64, SdpaMask::BOTTOM_RIGHT_CAUSAL, 53, 0.5F}})
    {
        SdpaGraphSpec spec;
        spec.dataType = DataType::HALF;
        spec.batch = 2;
        spec.heads = 4;
        spec.kvHeads = 2;
        spec.headDim = c.headDim;
        spec.seqLenQ = 40;
        spec.seqLenKv = c.seqLenKv;
        spec.mask = c.mask;
        spec.stats = SdpaStats::LSE;
        setDropoutWithRngDump(spec, c.p, 0x0123456789ABCDEFULL, (uint64_t{1} << 32) - 7);
        expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
    }
    // The seed and the offset given by value: the seed through the engine's workspace, the
    // offset as offset2 with no offset1 pointer.
    SdpaGraphSpec byValue;
    byValue.seqLenQ = 33;
    byValue.seqLenKv = 77;
    byValue.stats = SdpaStats::LSE;
    setDropoutWithRngDump(byValue, 0.3F, 42U, 5U);
    byValue.passByValueTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID);
    byValue.passByValueTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_UID);
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(byValue, BF16_TOLERANCE);
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

// A mask on a head_dim no exact object serves runs on the narrowest padded tier's mask object:
// 40 up to 64, 80 up to 96, 112 up to 128, 152 up to 160, 200 up to 224, 240 up to 256. Each
// tier with an f32 mask (bf16 operands) and with one in the query's dtype (f16), per head,
// with and without a bound across the tiers, and the LSE.
TEST(TestGpuFlydslSdpaAiwDispatch, AddsAMaskOnEveryPaddedTier)
{
    SKIP_IF_NO_DEVICES();
    struct TierCase
    {
        int64_t headDim;
        SdpaMask mask;
    };
    for(const auto& [headDim, mask] : {TierCase{40, SdpaMask::TOP_LEFT_CAUSAL},
                                       TierCase{80, SdpaMask::NONE},
                                       TierCase{112, SdpaMask::BOTTOM_RIGHT_CAUSAL},
                                       TierCase{152, SdpaMask::NONE},
                                       TierCase{200, SdpaMask::SLIDING_WINDOW},
                                       TierCase{240, SdpaMask::BOTTOM_RIGHT_CAUSAL}})
    {
        SCOPED_TRACE("head_dim " + std::to_string(headDim));
        SdpaGraphSpec spec;
        spec.headDim = headDim;
        spec.mask = mask;
        spec.heads = 4;
        spec.kvHeads = 2;
        spec.seqLenQ = 77;
        // A multiple of 8: the mask's rows must stay aligned for the object's vector loads.
        spec.seqLenKv = 104;
        spec.stats = SdpaStats::LSE;
        spec.withBias = true;
        spec.biasDims = {1, spec.heads, spec.seqLenQ, spec.seqLenKv};
        spec.biasDataType = DataType::FLOAT;
        spec.alignment[FLYDSL_SDPA_RDNA_BIAS_UID] = 32;
        expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
            spec, BF16_TOLERANCE, RdnaFamily::AIW);
        spec.alignment.clear();
        spec.dataType = DataType::HALF;
        spec.biasDataType = DataType::HALF;
        expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(
            spec, F16_TOLERANCE, RdnaFamily::AIW);
    }
}

TEST(TestGpuFlydslSdpaAiwDispatch, AddsAMaskWithTheEdgeCasesOnAPaddedTier)
{
    SKIP_IF_NO_DEVICES();
    // COVERAGE §5.7 on the padded tier up to 96 (head_dim 80), as §5's kernels give it: rows
    // at the mask dtype's lowest finite value (O = mean(V)); then -inf on every key of every
    // third row and on the first 64 keys of the others (bottom-right, so rows below 37 see
    // only -inf keys: O = 0, LSE = -inf). An f32 mask and one in the query's dtype.
    SdpaGraphSpec spec;
    spec.headDim = 80;
    spec.withBias = true;
    spec.biasMasksRows = true;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.seqLenQ = 77;
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
    spec.biasMasksRows = false;
    spec.biasNegativeInfinityRows = true;
    spec.biasNegativeInfinityLeadingKeys = 64;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
    spec.biasDataType = DataType::FLOAT;
    spec.alignment[FLYDSL_SDPA_RDNA_BIAS_UID] = 32;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
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

TEST(TestGpuFlydslSdpaAiwDispatch, AddsAFloatSink)
{
    SKIP_IF_NO_DEVICES();
    expectFloatSinksMatchTheReference(RdnaFamily::AIW);
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

TEST(TestGpuFlydslSdpaAiwDispatch, ServesAnyHeadDimPairOnAPaddedTier)
{
    // H1: V narrower or wider than Q/K on the padded tiers, the LSE included. (200, 128) runs
    // on the 224 tier, below the exact MLA 192/128 object; the last two on the 384 tier.
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.seqLenQ = 150;
    spec.seqLenKv = 170;
    spec.stats = SdpaStats::LSE;
    const std::array<std::array<int64_t, 2>, 6> pairs{
        {{128, 64}, {96, 128}, {200, 128}, {256, 64}, {320, 128}, {256, 384}}};
    for(const auto& [headDim, headDimV] : pairs)
    {
        spec.headDim = headDim;
        spec.valueHeadDim = headDimV;
        spec.mask = headDim < headDimV ? SdpaMask::BOTTOM_RIGHT_CAUSAL : SdpaMask::NONE;
        SCOPED_TRACE(std::to_string(headDim) + "/" + std::to_string(headDimV));
        expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
            spec, BF16_TOLERANCE, RdnaFamily::AIW);
    }
    spec.headDim = 64;
    spec.valueHeadDim = 128;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.dataType = DataType::HALF;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE, RdnaFamily::AIW);
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
// Dropout on the RDNA prefill objects (kernel modification 27; gfx11 objects only): the mask
// read off O bit for bit against the oracle, and the output and undropped LSE against the
// reference with the oracle's mask. The same oracle checks the aiw objects, so a graph gets one
// mask on either family.
TEST(TestGpuFlydslSdpaRdnaDispatch, DropsThroughABias)
{
    SKIP_IF_NO_DEVICES();
    if(packedArchDeviceProperties().gcnArchName.rfind("gfx11", 0) != 0)
    {
        GTEST_SKIP() << "the RDNA dropout objects ship for gfx11 only";
    }
    // R2: the mask read off O bit for bit through an additive bias (Q zero and V the
    // identity, so each O element is one key's kept, scaled probability), with a full bias
    // and a key-broadcast one [B, 1, 1, Skv], f32 and in the query's dtype, both causal
    // variants; rows the bias masks whole with -inf stay O = 0 and LSE = -inf under dropout.
    for(const int64_t headDim : {64, 96, 128, 256})
    {
        for(const auto mask : {SdpaMask::NONE, SdpaMask::BOTTOM_RIGHT_CAUSAL})
        {
            SdpaGraphSpec spec;
            spec.dataType = DataType::HALF;
            spec.batch = 2;
            spec.heads = 3;
            spec.kvHeads = 3;
            spec.headDim = headDim;
            spec.seqLenQ = 80;
            spec.seqLenKv = headDim;
            spec.mask = mask;
            spec.stats = SdpaStats::LSE;
            spec.withBias = true;
            setDropout(spec, 0.2F, 0x0123456789ABCDEFULL, 99);
            spec.exposeDropoutMask = true;
            expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
            spec.biasDims = {spec.batch, 1, 1, spec.seqLenKv};
            spec.biasDataType = DataType::HALF;
            expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
            spec.biasDims.clear();
            spec.biasDataType = DataType::FLOAT;
            spec.exposeDropoutMask = false;
            spec.biasNegativeInfinityRows = true;
            expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
        }
    }
}

TEST(TestGpuFlydslSdpaRdnaDispatch, DropsTheOraclesMaskBitForBit)
{
    SKIP_IF_NO_DEVICES();
    // The RDNA dropout twins are gfx11-generic objects; gfx12 drops on the aiw objects
    // (decision Q3), so a gfx12 device has nothing here to run.
    if(packedArchDeviceProperties().gcnArchName.rfind("gfx11", 0) != 0)
    {
        GTEST_SKIP() << "the RDNA dropout objects ship for gfx11 only";
    }
    struct Case
    {
        int64_t headDim;
        SdpaMask mask;
        float p;
    };
    for(const auto& c : {Case{64, SdpaMask::NONE, 0.25F},
                         Case{96, SdpaMask::TOP_LEFT_CAUSAL, 0.1F},
                         Case{128, SdpaMask::NONE, 0.3F},
                         Case{256, SdpaMask::TOP_LEFT_CAUSAL, 0.5F}})
    {
        SdpaGraphSpec spec;
        spec.dataType = DataType::HALF;
        spec.batch = 2;
        spec.heads = 3;
        spec.kvHeads = 3;
        spec.headDim = c.headDim;
        spec.seqLenQ = 80;
        spec.seqLenKv = c.headDim;
        spec.mask = c.mask;
        spec.stats = SdpaStats::LSE;
        // A seed past 32 bits, and an offset that carries into the counter's high word.
        setDropout(spec, c.p, 0x0123456789ABCDEFULL, (uint64_t{1} << 32) - 7);
        spec.exposeDropoutMask = true;
        expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
    }
}

TEST(TestGpuFlydslSdpaRdnaDispatch, DropsWithTheUndroppedLse)
{
    SKIP_IF_NO_DEVICES();
    // The RDNA dropout twins are gfx11-generic objects; gfx12 drops on the aiw objects
    // (decision Q3), so a gfx12 device has nothing here to run.
    if(packedArchDeviceProperties().gcnArchName.rfind("gfx11", 0) != 0)
    {
        GTEST_SKIP() << "the RDNA dropout objects ship for gfx11 only";
    }
    SdpaGraphSpec spec;
    spec.headDim = 128;
    spec.heads = 4;
    spec.kvHeads = 2;
    spec.seqLenQ = 150;
    spec.seqLenKv = 130;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    setDropout(spec, 0.2F, 42U, 5U);
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
    spec.mask = SdpaMask::NONE;
    spec.headDim = 64;
    spec.layout = SdpaLayout::BHSD;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, DropsWithASeedAndOffsetGivenByValue)
{
    SKIP_IF_NO_DEVICES();
    // The RDNA dropout twins are gfx11-generic objects; gfx12 drops on the aiw objects
    // (decision Q3), so a gfx12 device has nothing here to run.
    if(packedArchDeviceProperties().gcnArchName.rfind("gfx11", 0) != 0)
    {
        GTEST_SKIP() << "the RDNA dropout objects ship for gfx11 only";
    }
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.headDim = 64;
    spec.seqLenQ = 40;
    spec.seqLenKv = 64;
    spec.stats = SdpaStats::LSE;
    setDropout(spec, 0.3F, 0xFEDCBA9876543210ULL, 1000U);
    spec.exposeDropoutMask = true;
    spec.passByValueTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID);
    spec.passByValueTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_UID);
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
}

// Padding on device, RDNA prefill and decode (kernel modifications 26 and 28): each batch's
// own lengths, padded rows written as O = 0 and LSE = -inf (sink objects too), keys past a
// batch's key length masked, every mask at each batch's own corner.
TEST(TestGpuFlydslSdpaRdnaDispatch, PadsEachBatchToItsOwnLengths)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.batch = 5;
    spec.headDim = 64;
    spec.seqLenQ = 130;
    spec.seqLenKv = 130;
    spec.stats = SdpaStats::LSE;
    // Full; one query row; no query row; odd lengths; no key, so every row is keyless.
    setPadding(spec, {130, 1, 0, 77, 30}, {130, 100, 33, 129, 0});
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.dataType = DataType::HALF;
    spec.layout = SdpaLayout::BHSD;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, PutsEachBatchsBottomRightMaskAtItsOwnCorner)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.batch = 4;
    spec.headDim = 64;
    spec.heads = 4;
    spec.kvHeads = 2;
    spec.seqLenQ = 96;
    spec.seqLenKv = 96;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    // More queries than keys (leading keyless rows), fewer, equal, one of each.
    setPadding(spec, {70, 20, 96, 1}, {37, 90, 96, 1});
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
    // A bottom-right window, each batch's own.
    spec.leftBound = 8;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
}

namespace
{

/// Padded graphs with a bias at every key length mod 8 (the aiw objects read the bias 8 keys at
/// a time), more queries than keys, a bias of its own per batch that is NaN past each batch's
/// keys (masked whatever it holds there), rows at the lowest finite value, -inf and NaN, and an
/// f32 bias. The CDNA vectorized objects read past a batch's bias view here until vendored-kernel
/// modification 37; the RDNA objects' bias view spans the launch's extents and they read one key
/// at a time, and the aiw objects read whole vectors only inside a batch's keys (its tail tile
/// one key at a time), so neither family has that hazard: this pins it.
void expectPaddedBiasAtEveryKeyLength(RdnaFamily family)
{
    using hipdnn_data_sdk::types::bfloat16;
    SdpaGraphSpec spec;
    spec.batch = 4;
    spec.heads = 4;
    spec.kvHeads = 2;
    spec.seqLenQ = 160;
    spec.seqLenKv = 296;
    spec.stats = SdpaStats::LSE;
    spec.withBias = true;
    spec.biasDataType = DataType::BFLOAT16;
    spec.biasDims = {4, 4, 160, 296};
    spec.biasNanPastKeys = true;
    struct Lengths
    {
        std::vector<int32_t> q;
        std::vector<int32_t> kv;
    };
    std::array<Lengths, 3> lengthSets{Lengths{{129, 70, 160, 3}, {57, 58, 59, 60}},
                                      Lengths{{129, 70, 160, 3}, {61, 62, 63, 64}},
                                      Lengths{{160, 1, 100, 159}, {293, 294, 295, 133}}};
    if(family == RdnaFamily::DECODE)
    {
        // A speculative step of up to 3 rows.
        spec.seqLenQ = 3;
        spec.biasDims = {4, 4, 3, 296};
        for(auto& lengths : lengthSets)
        {
            lengths.q = {3, 1, 2, 3};
        }
    }
    for(const auto& lengths : lengthSets)
    {
        setPadding(spec, lengths.q, lengths.kv);
        for(const int64_t headDim : {int64_t{64}, int64_t{128}})
        {
            for(const auto mask : {SdpaMask::NONE, SdpaMask::BOTTOM_RIGHT_CAUSAL})
            {
                SCOPED_TRACE("keys " + std::to_string(lengths.kv.front()) + ", head_dim "
                             + std::to_string(headDim)
                             + (mask == SdpaMask::NONE ? "" : ", bottom-right"));
                spec.headDim = headDim;
                spec.mask = mask;
                expectSdpaMatchesReference<bfloat16>(spec, BF16_TOLERANCE, family);
            }
        }
    }
    auto edge = spec;
    edge.headDim = 128;
    edge.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    setPadding(edge, lengthSets[1].q, lengthSets[1].kv);
    {
        SCOPED_TRACE("rows at the lowest finite value, -inf leading keys");
        auto rows = edge;
        rows.biasMasksRows = true;
        rows.biasNegativeInfinityLeadingKeys = 40;
        expectSdpaMatchesReference<bfloat16>(rows, BF16_TOLERANCE, family);
    }
    {
        SCOPED_TRACE("-inf rows");
        auto rows = edge;
        rows.biasNegativeInfinityRows = true;
        expectSdpaMatchesReference<bfloat16>(rows, BF16_TOLERANCE, family);
    }
    {
        SCOPED_TRACE("NaN rows");
        auto rows = edge;
        rows.biasNanRows = true;
        expectSdpaMatchesReference<bfloat16>(rows, BF16_TOLERANCE, family);
    }
    {
        SCOPED_TRACE("an f32 bias");
        auto f32 = edge;
        f32.biasDataType = DataType::FLOAT;
        f32.alignment[FLYDSL_SDPA_RDNA_BIAS_UID] = 32;
        expectSdpaMatchesReference<bfloat16>(f32, BF16_TOLERANCE, family);
    }
}

} // namespace

TEST(TestGpuFlydslSdpaRdnaDispatch, PadsWithABiasAtEveryKeyLength)
{
    SKIP_IF_NO_DEVICES();
    expectPaddedBiasAtEveryKeyLength(RdnaFamily::PREFILL);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, DecodesPaddedBatchesWithABiasAtEveryKeyLength)
{
    SKIP_IF_NO_DEVICES();
    expectPaddedBiasAtEveryKeyLength(RdnaFamily::DECODE);
}

TEST(TestGpuFlydslSdpaAiwDispatch, PadsWithABiasAtEveryKeyLength)
{
    SKIP_IF_NO_DEVICES();
    expectPaddedBiasAtEveryKeyLength(RdnaFamily::AIW);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, PadsWithABiasASinkAndAGenericHeadDim)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.batch = 3;
    spec.headDim = 64;
    spec.seqLenQ = 80;
    spec.seqLenKv = 96;
    spec.stats = SdpaStats::LSE;
    setPadding(spec, {80, 33, 5}, {64, 96, 17});
    spec.withBias = true;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
    spec.withBias = false;
    spec.withSink = true;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
    spec.withSink = false;
    spec.headDim = 80;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, DecodesEachBatchAgainstItsOwnCache)
{
    SKIP_IF_NO_DEVICES();
    // Batched decode with different context lengths: lengths 0, 1, short and full, splits
    // sized for the longest left empty for the shorter.
    SdpaGraphSpec spec;
    spec.batch = 4;
    spec.headDim = 64;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 1;
    spec.seqLenKv = 600;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    setPadding(spec, {1, 1, 1, 0}, {600, 1, 0, 300});
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
    // A speculative step of 3 rows, one batch with 2 of them.
    spec.seqLenQ = 3;
    setPadding(spec, {3, 2, 3, 1}, {600, 40, 3, 300});
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
    spec.withBias = true;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
    // A long cache split several ways, so the merge kernel runs: it fills the padded rows
    // (the second batch's last two).
    spec.withBias = false;
    spec.batch = 2;
    spec.seqLenKv = 4096;
    setPadding(spec, {3, 1}, {4096, 100});
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, PadsWithLengthsAndNoPaddingMask)
{
    SKIP_IF_NO_DEVICES();
    // SEM-008: the lengths without padding_mask are padding, as hipDNN's references read them.
    SdpaGraphSpec spec;
    spec.batch = 3;
    spec.headDim = 64;
    spec.seqLenQ = 96;
    spec.seqLenKv = 96;
    spec.stats = SdpaStats::LSE;
    setPadding(spec, {96, 40, 0}, {50, 96, 7});
    spec.paddingMask = false;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
    spec.seqLenQ = 1;
    setPadding(spec, {1, 0, 1}, {96, 40, 0});
    spec.paddingMask = false;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, ClampsLengthsOutsideTheSequence)
{
    SKIP_IF_NO_DEVICES();
    // A length below 0 reads as 0 and one past the sequence extent as the extent, as hipDNN's
    // GPU reference clamps them (modification 28): no row or key outside the batch's slice.
    SdpaGraphSpec spec;
    spec.batch = 3;
    spec.headDim = 64;
    spec.seqLenQ = 80;
    spec.seqLenKv = 96;
    spec.stats = SdpaStats::LSE;
    setPadding(spec, {-5, 500, 40}, {500, 30, -1});
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
    spec.seqLenQ = 2;
    spec.seqLenKv = 600;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    setPadding(spec, {7, -2, 1}, {9000, 300, -40});
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE, true);
}

// Padding on device: each batch's own lengths, rows past a batch's query length written as
// O = 0 and LSE = -inf (aiw modification 10; the harness fills them with NaN beforehand), keys
// past its key length masked, a bottom-right diagonal at each batch's Skv_b - Sq_b.
TEST(TestGpuFlydslSdpaAiwDispatch, PadsWithLengthsAndNoPaddingMask)
{
    SKIP_IF_NO_DEVICES();
    // SEM-008, as on RDNA.
    SdpaGraphSpec spec;
    spec.batch = 3;
    spec.headDim = 128;
    spec.seqLenQ = 96;
    spec.seqLenKv = 96;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    setPadding(spec, {96, 40, 0}, {50, 96, 7});
    spec.paddingMask = false;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
}

TEST(TestGpuFlydslSdpaAiwDispatch, ClampsLengthsOutsideTheSequence)
{
    SKIP_IF_NO_DEVICES();
    // As on RDNA: below 0 reads as 0, past the extent as the extent (aiw modification 10).
    SdpaGraphSpec spec;
    spec.batch = 3;
    spec.headDim = 64;
    spec.seqLenQ = 80;
    spec.seqLenKv = 96;
    spec.stats = SdpaStats::LSE;
    setPadding(spec, {-5, 500, 40}, {500, 30, -1});
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
}

TEST(TestGpuFlydslSdpaAiwDispatch, PadsEachBatchToItsOwnLengths)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.batch = 5;
    spec.headDim = 128;
    spec.seqLenQ = 200;
    spec.seqLenKv = 200;
    spec.stats = SdpaStats::LSE;
    // Full; one query row; no query row; an odd length; no key, so every row is keyless
    // (O = 0, LSE = -inf).
    setPadding(spec, {200, 1, 0, 77, 30}, {200, 150, 33, 129, 0});
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.dataType = DataType::HALF;
    spec.layout = SdpaLayout::BHSD;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE, RdnaFamily::AIW);
}

TEST(TestGpuFlydslSdpaAiwDispatch, PutsEachBatchsBottomRightDiagonalAtItsOwnCorner)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.batch = 4;
    spec.headDim = 64;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 160;
    spec.seqLenKv = 160;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    // More queries than keys (leading keyless rows), fewer, equal, and one of each.
    setPadding(spec, {100, 40, 160, 1}, {37, 150, 160, 1});
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
}

TEST(TestGpuFlydslSdpaAiwDispatch, PadsUnderATopLeftWindowAMaskAndASink)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.batch = 3;
    spec.headDim = 64;
    spec.seqLenQ = 192;
    spec.seqLenKv = 192;
    spec.mask = SdpaMask::SLIDING_WINDOW;
    spec.stats = SdpaStats::LSE;
    setPadding(spec, {192, 70, 5}, {150, 192, 64});
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);

    spec.mask = SdpaMask::NONE;
    spec.withBias = true;
    spec.biasDataType = spec.dataType;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);

    spec.withBias = false;
    spec.withSink = true;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
}

TEST(TestGpuFlydslSdpaAiwDispatch, PadsAPaddedHeadDimTier)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.batch = 2;
    spec.headDim = 80;
    spec.seqLenQ = 130;
    spec.seqLenKv = 170;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    setPadding(spec, {129, 17}, {170, 18});
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
}

// Packed (THD) on device: each sequence stored end to end through the cumulative token tables,
// against the same per-sequence reference as padding (the harness packs the dense values and
// unpacks O and the [T, H] LSE).
TEST(TestGpuFlydslSdpaAiwDispatch, PacksSequencesEndToEnd)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.batch = 5;
    spec.headDim = 128;
    spec.seqLenQ = 200;
    spec.seqLenKv = 200;
    spec.stats = SdpaStats::LSE;
    // The longest; one query token; none; an odd length; no key, so every row is keyless
    // (O = 0, LSE = -inf).
    setPacked(spec, {200, 1, 0, 77, 30}, {200, 150, 33, 129, 0});
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.dataType = DataType::HALF;
    spec.stats = SdpaStats::NONE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE, RdnaFamily::AIW);
}

TEST(TestGpuFlydslSdpaAiwDispatch, PutsEachPackedSequencesBottomRightDiagonalAtItsOwnCorner)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.batch = 4;
    spec.headDim = 64;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 160;
    spec.seqLenKv = 160;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    // More queries than keys (leading keyless rows), fewer, equal, and one of each.
    setPacked(spec, {100, 40, 160, 1}, {37, 150, 160, 1});
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
}

TEST(TestGpuFlydslSdpaAiwDispatch, PacksUnderATopLeftWindowAndWithASink)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.batch = 3;
    spec.headDim = 64;
    spec.seqLenQ = 192;
    spec.seqLenKv = 192;
    spec.mask = SdpaMask::SLIDING_WINDOW;
    spec.stats = SdpaStats::LSE;
    setPacked(spec, {192, 70, 5}, {150, 192, 64});
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);

    spec.mask = SdpaMask::NONE;
    spec.withSink = true;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
}

TEST(TestGpuFlydslSdpaAiwDispatch, PacksAPaddedHeadDimTierThroughOneTable)
{
    SKIP_IF_NO_DEVICES();
    // Self-attention, every operand through one table.
    SdpaGraphSpec spec;
    spec.batch = 3;
    spec.headDim = 80;
    spec.seqLenQ = 170;
    spec.seqLenKv = 170;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    setPacked(spec, {170, 18, 99}, {170, 18, 99});
    spec.raggedOffsetOf[FLYDSL_SDPA_RDNA_K_UID] = FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID;
    spec.raggedOffsetOf[FLYDSL_SDPA_RDNA_V_UID] = FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
}

// Packed (THD) sequences on the RDNA prefill objects (kernel modification 29), through the
// same test-local per-sequence oracle as the aiw cases. The packed buffers hold exactly the
// sequences' tokens, so a load past the last sequence would leave the allocation.
TEST(TestGpuFlydslSdpaRdnaDispatch, PacksSequencesEndToEnd)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.batch = 5;
    spec.headDim = 128;
    spec.seqLenQ = 200;
    spec.seqLenKv = 200;
    spec.stats = SdpaStats::LSE;
    // The longest; one query token; none; an odd length; no key, so every row is keyless
    // (O = 0, LSE = -inf).
    setPacked(spec, {200, 1, 0, 77, 30}, {200, 150, 33, 129, 0});
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::PREFILL);
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.dataType = DataType::HALF;
    spec.stats = SdpaStats::NONE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(
        spec, F16_TOLERANCE, RdnaFamily::PREFILL);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, PutsEachPackedSequencesBottomRightBoundsAtItsOwnCorner)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.batch = 4;
    spec.headDim = 64;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 160;
    spec.seqLenKv = 160;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    // More queries than keys (leading keyless rows), fewer, equal, and one of each.
    setPacked(spec, {100, 40, 160, 1}, {37, 150, 160, 1});
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::PREFILL);
    // A bottom-right window, which the aiw kernel cannot move per sequence.
    spec.leftBound = 24;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::PREFILL);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, PacksAGenericHeadDimWithASinkThroughOneTable)
{
    SKIP_IF_NO_DEVICES();
    // Self-attention, every operand through one table, on a generic tier.
    SdpaGraphSpec spec;
    spec.batch = 3;
    spec.headDim = 80;
    spec.seqLenQ = 170;
    spec.seqLenKv = 170;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    setPacked(spec, {170, 18, 99}, {170, 18, 99});
    spec.raggedOffsetOf[FLYDSL_SDPA_RDNA_K_UID] = FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID;
    spec.raggedOffsetOf[FLYDSL_SDPA_RDNA_V_UID] = FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::PREFILL);
    // A sink, at a head dim with sink objects.
    spec.headDim = 64;
    spec.mask = SdpaMask::NONE;
    spec.withSink = true;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::PREFILL);
}

TEST(TestGpuFlydslSdpaRdnaDispatch, PacksSequencesWithTrailingPaddingPerBatch)
{
    // P3, cuDNN's packed form: each table entry a slot, the per-batch lengths its rows and
    // keys. The slot's rows past the length are padded rows (O = 0, LSE = -inf), its keys past
    // the length are never read (the packed buffer holds real values there), and a length past
    // the slot is clamped to it.
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.batch = 4;
    spec.headDim = 64;
    spec.heads = 4;
    spec.kvHeads = 2;
    spec.seqLenQ = 160;
    spec.seqLenKv = 160;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    setPacked(spec, {160, 50, 1, 90}, {160, 70, 40, 33});
    setPadding(spec, {150, 50, 0, 200}, {100, 3, 40, 0});
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::PREFILL);
    // Without padding_mask (hipDNN's SEM-008), top-left, a generic tier, f16.
    spec.paddingMask = false;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.headDim = 80;
    spec.dataType = DataType::HALF;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(
        spec, F16_TOLERANCE, RdnaFamily::PREFILL);
    // A sink: a padded row is not a keyless row, so it is -inf, not the sink.
    spec.headDim = 64;
    spec.mask = SdpaMask::NONE;
    spec.dataType = DataType::BFLOAT16;
    spec.withSink = true;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::PREFILL);
}

// Dropout on device: the mask read off O bit for bit against the oracle (Q zero, V the
// identity, so each O element is one key's kept-and-scaled probability), and the full
// output and the undropped LSE against the reference with the oracle's mask.
TEST(TestGpuFlydslSdpaAiwDispatch, DropsThroughAMask)
{
    SKIP_IF_NO_DEVICES();
    // R1: the mask read off O bit for bit through an additive bias (Q zero and V the
    // identity, so each O element is one key's kept, scaled probability), with a full bias
    // and a key-broadcast one [B, 1, 1, Skv], f32 and in the query's dtype, both causal
    // variants; rows the bias masks whole with -inf stay O = 0 and LSE = -inf under dropout.
    for(const int64_t headDim : {64, 128, 256})
    {
        for(const auto mask : {SdpaMask::NONE, SdpaMask::BOTTOM_RIGHT_CAUSAL})
        {
            SdpaGraphSpec spec;
            spec.dataType = DataType::HALF;
            spec.batch = 2;
            spec.heads = 3;
            spec.kvHeads = 3;
            spec.headDim = headDim;
            spec.seqLenQ = 80;
            spec.seqLenKv = headDim;
            spec.mask = mask;
            spec.stats = SdpaStats::LSE;
            spec.withBias = true;
            // The aiw f32-mask objects load the mask 32 bytes at a time.
            spec.alignment[FLYDSL_SDPA_RDNA_BIAS_UID] = 32;
            setDropout(spec, 0.2F, 0x0123456789ABCDEFULL, 99);
            spec.exposeDropoutMask = true;
            expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(
                spec, F16_TOLERANCE, RdnaFamily::AIW);
            spec.biasDims = {spec.batch, 1, 1, spec.seqLenKv};
            spec.biasDataType = DataType::HALF;
            expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(
                spec, F16_TOLERANCE, RdnaFamily::AIW);
            spec.biasDims.clear();
            spec.biasDataType = DataType::FLOAT;
            spec.exposeDropoutMask = false;
            spec.biasNegativeInfinityRows = true;
            expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(
                spec, F16_TOLERANCE, RdnaFamily::AIW);
        }
    }
}

TEST(TestGpuFlydslSdpaAiwDispatch, DropsTheOraclesMaskBitForBit)
{
    SKIP_IF_NO_DEVICES();
    struct Case
    {
        int64_t headDim;
        SdpaMask mask;
        float p;
    };
    for(const auto& c : {Case{64, SdpaMask::NONE, 0.25F},
                         Case{128, SdpaMask::TOP_LEFT_CAUSAL, 0.1F},
                         Case{256, SdpaMask::NONE, 0.5F}})
    {
        SdpaGraphSpec spec;
        spec.dataType = DataType::HALF;
        spec.batch = 2;
        spec.heads = 3;
        spec.kvHeads = 3;
        spec.headDim = c.headDim;
        spec.seqLenQ = 150;
        spec.seqLenKv = c.headDim;
        spec.mask = c.mask;
        spec.stats = SdpaStats::LSE;
        // A seed past 32 bits, and an offset that carries into the counter's high word.
        setDropout(spec, c.p, 0x0123456789ABCDEFULL, (uint64_t{1} << 32) - 7);
        spec.exposeDropoutMask = true;
        expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(
            spec, F16_TOLERANCE, RdnaFamily::AIW);
    }
}

TEST(TestGpuFlydslSdpaAiwDispatch, WritesTheRngDumpItApplied)
{
    SKIP_IF_NO_DEVICES();
    struct Case
    {
        int64_t headDim;
        SdpaMask mask;
        int64_t seqLenKv;
        float p;
    };
    // Keys not a multiple of 4, masked keys, GQA, a seed past 32 bits and an offset that
    // carries into the counter's high word.
    for(const auto& c :
        {Case{64, SdpaMask::NONE, 70, 0.25F}, Case{128, SdpaMask::TOP_LEFT_CAUSAL, 96, 0.1F}})
    {
        SdpaGraphSpec spec;
        spec.dataType = DataType::HALF;
        spec.batch = 2;
        spec.heads = 4;
        spec.kvHeads = 2;
        spec.headDim = c.headDim;
        spec.seqLenQ = 40;
        spec.seqLenKv = c.seqLenKv;
        spec.mask = c.mask;
        spec.stats = SdpaStats::LSE;
        setDropoutWithRngDump(spec, c.p, 0x0123456789ABCDEFULL, (uint64_t{1} << 32) - 7);
        expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(
            spec, F16_TOLERANCE, RdnaFamily::AIW);
    }
    // The seed and the offset given by value.
    SdpaGraphSpec byValue;
    byValue.seqLenQ = 33;
    byValue.seqLenKv = 77;
    byValue.stats = SdpaStats::LSE;
    setDropoutWithRngDump(byValue, 0.3F, 42U, 5U);
    byValue.passByValueTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID);
    byValue.passByValueTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_UID);
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        byValue, BF16_TOLERANCE, RdnaFamily::AIW);
}

TEST(TestGpuFlydslSdpaAiwDispatch, DropsWithTheUndroppedLse)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.headDim = 128;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 300;
    spec.seqLenKv = 270;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    setDropout(spec, 0.2F, 42U, 5U);
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
    spec.mask = SdpaMask::NONE;
    spec.headDim = 64;
    spec.layout = SdpaLayout::BHSD;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
}

TEST(TestGpuFlydslSdpaAiwDispatch, DropsWithASeedAndOffsetGivenByValue)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.headDim = 64;
    spec.seqLenQ = 70;
    spec.seqLenKv = 64;
    spec.stats = SdpaStats::LSE;
    setDropout(spec, 0.3F, 0xFEDCBA9876543210ULL, 1000U);
    spec.exposeDropoutMask = true;
    spec.passByValueTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID);
    spec.passByValueTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_UID);
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE, RdnaFamily::AIW);
    // INT32 by value: widened on the host to the same counter.
    spec.dropoutSeed = 77U;
    spec.dropoutSeedDataType = DataType::INT32;
    spec.dropoutOffsetDataType = DataType::INT32;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE, RdnaFamily::AIW);
}

TEST(TestGpuFlydslSdpaAiwDispatch, ComputesTheUndroppedResultAtProbabilityZero)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.headDim = 128;
    spec.seqLenQ = 200;
    spec.seqLenKv = 200;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    setDropout(spec, 0.0F);
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        spec, BF16_TOLERANCE, RdnaFamily::AIW);
}

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
