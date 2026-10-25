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
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>

#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Handle.hpp"
// packedArchDeviceProperties() and reportUnstaged(): a device case that finds nothing staged
// fails on an arch this build ships the op for, and skips elsewhere.
#include "engines/kernel_ingestor_engine/packs/FlydslNormDeviceTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaCpuReferenceTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaDropout.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaObjectTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaPhiloxOracle.hpp"
// The SDPA graph builder: plain hipDNN graph construction, shared with the RDNA suites.
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaRdnaTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaRngDump.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaRngDumpTestSupport.hpp"

/**
 * @file TestFlydslSdpaCdnaEngine.cpp
 * @brief The hipkernel:flydsl_sdpa_cdna pack's seams: what its graph matcher accepts and
 *        declines, what it binds, which candidate a graph selects, and real launches of
 *        the staged gfx90a/gfx942/gfx950 objects checked against a host reference.
 *
 * The CDNA objects bake the dtype, head_dim (or a head-dim bucket), the causal flag and,
 * on the dual-wave kernel, the diagonal; head counts, lengths and strides are kernel
 * arguments, each within the 32-bit bounds the decline cases pin one by one. A GPU case that finds no object staged fails on a device
 * this build ships the CDNA ops for (HIPKERNELPROVIDER_FLYDSL_ARCHS_sdpa_cdna*), and skips
 * on any other.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hipdnn_flatbuffers_sdk::data_objects::DataType;

// Restated rather than shared with the pack, so a rename on either side fails here.
constexpr PackSymbols FLYDSL_SDPA_CDNA{"hipkernel:flydsl_sdpa_cdna",
                                       "hipkernel.flydsl_sdpa_cdna.graph_match",
                                       "",
                                       "hipkernel.flydsl_sdpa_cdna.kernel_match",
                                       "hipkernel.flydsl_sdpa_cdna.score",
                                       "hipkernel.flydsl_sdpa_cdna.dispatch",
                                       "flydsl_sdpa_cdna.q.uid",
                                       "flydsl_sdpa_cdna.k.uid",
                                       "flydsl_sdpa_cdna.o.uid"};
/// The other two CDNA kernel families' engines, which share the generic engine's graph
/// matcher and bound tokens.
constexpr PackSymbols FLYDSL_SDPA_CDNA_DUALWAVE{"hipkernel:flydsl_sdpa_cdna_dualwave",
                                                "hipkernel.flydsl_sdpa_cdna_dualwave.graph_match",
                                                "",
                                                "hipkernel.flydsl_sdpa_cdna_dualwave.kernel_match",
                                                "hipkernel.flydsl_sdpa_cdna_dualwave.score",
                                                "hipkernel.flydsl_sdpa_cdna_dualwave.dispatch",
                                                "flydsl_sdpa_cdna.q.uid",
                                                "flydsl_sdpa_cdna.k.uid",
                                                "flydsl_sdpa_cdna.o.uid"};
constexpr PackSymbols FLYDSL_SDPA_CDNA_DECODE{"hipkernel:flydsl_sdpa_cdna_decode",
                                              "hipkernel.flydsl_sdpa_cdna_decode.graph_match",
                                              "",
                                              "hipkernel.flydsl_sdpa_cdna_decode.kernel_match",
                                              "hipkernel.flydsl_sdpa_cdna_decode.score",
                                              "hipkernel.flydsl_sdpa_cdna_decode.dispatch",
                                              "flydsl_sdpa_cdna.q.uid",
                                              "flydsl_sdpa_cdna.k.uid",
                                              "flydsl_sdpa_cdna.o.uid"};

/// The engine a kernel belongs to, read from its instance name -- `sdpa_cdna_dualwave_...`,
/// `sdpa_cdna_decode_...` or `sdpa_cdna_...` -- as the staged objects' names and the ones
/// makeCdnaKernel gives both spell it.
const PackSymbols& cdnaEngineFor(const KernelDefinition& kernel)
{
    if(kernel.name.rfind("sdpa_cdna_dualwave_", 0) == 0)
    {
        return FLYDSL_SDPA_CDNA_DUALWAVE;
    }
    if(kernel.name.rfind("sdpa_cdna_decode_", 0) == 0)
    {
        return FLYDSL_SDPA_CDNA_DECODE;
    }
    return FLYDSL_SDPA_CDNA;
}

/// A metadata field of type @p T, or nullptr when absent or of another type.
template <typename T>
const T* tryGetMetadataField(const MetadataValues& values, const char* field)
{
    const auto it = values.find(std::string(field));
    return it == values.end() ? nullptr : std::get_if<T>(&it->second);
}

constexpr const char* CDNA_CAUSAL_TOKEN = "flydsl_sdpa_cdna.causal";
constexpr const char* CDNA_SCALE_SOURCE_TOKEN = "flydsl_sdpa_cdna.scale_source";
constexpr const char* CDNA_SCALE_TOKEN = "flydsl_sdpa_cdna.scale";
constexpr const char* CDNA_STATS_TOKEN = "flydsl_sdpa_cdna.stats.uid";
constexpr const char* CDNA_DIAG_OFFSET_TOKEN = "flydsl_sdpa_cdna.diag_offset";
constexpr const char* CDNA_WINDOW_TOKEN = "flydsl_sdpa_cdna.window_lo";
constexpr const char* CDNA_SINK_STRIDE_TOKEN = "flydsl_sdpa_cdna.sink.stride";
constexpr const char* CDNA_SINK_FLOAT_TOKEN = "flydsl_sdpa_cdna.sink.float";
constexpr int64_t CDNA_NO_WINDOW = -(int64_t{1} << 30);

constexpr const char* CDNA_DTYPE_FIELD = "dtype";
constexpr const char* CDNA_HEAD_DIM_FIELD = "head_dim";
constexpr const char* CDNA_CAUSAL_FIELD = "causal";
constexpr const char* CDNA_BLOCK_M_FIELD = "block_m";
constexpr const char* CDNA_KV_ALIGN_FIELD = "kv_align";
constexpr const char* CDNA_ANY_DIAGONAL_FIELD = "any_diagonal";
constexpr const char* CDNA_HAS_BIAS_FIELD = "has_bias";
constexpr const char* CDNA_HEAD_DIM_V_FIELD = "head_dim_v";
constexpr const char* CDNA_QK_MASK_FIELD = "qk_mask";
constexpr const char* CDNA_MERGE_SYMBOL_FIELD = "merge_symbol";
constexpr const char* CDNA_PREFER_FEATURE_FIELD = "prefer_feature";
constexpr const char* CDNA_PREFER_MIN_FIELD = "prefer_min";
constexpr const char* CDNA_PREFER_MAX_FIELD = "prefer_max";
constexpr const char* CDNA_BIAS_TOKEN = "flydsl_sdpa_cdna.bias.uid";
constexpr const char* CDNA_BIAS_KSTRIDED_FIELD = "bias_kstrided";
constexpr const char* CDNA_HAS_DROPOUT_FIELD = "has_dropout";

/// The model shape the first CDNA objects are built for.
SdpaGraphSpec cdnaSpec()
{
    SdpaGraphSpec spec;
    spec.batch = 1;
    spec.heads = 32;
    spec.kvHeads = 8;
    spec.seqLenQ = 256;
    spec.seqLenKv = 256;
    spec.headDim = 128;
    spec.scaleKind = SdpaScale::ABSENT;
    return spec;
}

/// The model shape padded, cuDNN's way: two batches, each with its own query and key
/// lengths (vendored-kernel modification 32).
SdpaGraphSpec cdnaPaddedSpec()
{
    auto spec = cdnaSpec();
    spec.batch = 2;
    setPadding(spec, {256, 100}, {256, 77});
    return spec;
}

/// A kernel carrying exactly the metadata its family's KMD declares: "dualwave", "generic"
/// (prefill) or "decode" (the generic kernel's split-KV objects). Each family's knobs that
/// hold one value across it are not in its KMD, so not here.
KernelDefinition makeCdnaKernel(const std::string& family = "dualwave",
                                int64_t headDim = 128,
                                int64_t causal = 0,
                                const std::string& dtype = "bf16",
                                int64_t blockM = 256,
                                int64_t kvAlign = 0,
                                int64_t anyDiagonal = 1,
                                int64_t hasBias = 0,
                                std::optional<int64_t> headDimV = std::nullopt,
                                int64_t qkMask = 0)
{
    KernelDefinition kernel;
    kernel.name
        = (family == "generic" ? std::string("sdpa_cdna_") : "sdpa_cdna_" + family + "_") + dtype;
    kernel.metadata = {{CDNA_DTYPE_FIELD, dtype},
                       {CDNA_HEAD_DIM_FIELD, headDim},
                       {CDNA_CAUSAL_FIELD, causal},
                       {CDNA_BLOCK_M_FIELD, blockM},
                       {CDNA_HAS_BIAS_FIELD, hasBias},
                       {CDNA_HEAD_DIM_V_FIELD, headDimV.value_or(headDim)}};
    if(family == "dualwave")
    {
        kernel.metadata[CDNA_ANY_DIAGONAL_FIELD] = anyDiagonal;
    }
    else
    {
        kernel.metadata[CDNA_QK_MASK_FIELD] = qkMask;
        kernel.metadata["dv_split"] = int64_t{1};
        kernel.metadata["has_sink"] = int64_t{0};
    }
    if(family == "generic")
    {
        kernel.metadata[CDNA_KV_ALIGN_FIELD] = kvAlign;
        kernel.metadata[CDNA_BIAS_KSTRIDED_FIELD] = int64_t{0};
        kernel.metadata[CDNA_HAS_DROPOUT_FIELD] = int64_t{0};
    }
    if(family == "decode")
    {
        kernel.metadata[CDNA_MERGE_SYMBOL_FIELD] = std::string("flash_attn_generic_merge_kernel_1");
    }
    kernel.priority = 100;
    return kernel;
}

/// The kernel matcher and score of @p kernel's own engine.
bool matchesCdnaKernel(const MatchContext& context,
                       const KernelDefinition& kernel,
                       const BoundTokens& bound)
{
    return matchesKernel(cdnaEngineFor(kernel), context, kernel, bound);
}

double scoreCdnaKernel(const MatchContext& context, const KernelDefinition& kernel)
{
    return scoreKernel(cdnaEngineFor(kernel), context, kernel);
}

std::optional<BoundTokens> cdnaGraphMatch(const SdpaGraphSpec& spec)
{
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    return matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
}

// ---------------------------------------------------------------------------
// graph_match
// ---------------------------------------------------------------------------

struct CdnaCase
{
    std::string name;
    SdpaGraphSpec spec;
};

class TestFlydslSdpaCdnaGraphAccepts : public ::testing::TestWithParam<CdnaCase>
{
};

TEST_P(TestFlydslSdpaCdnaGraphAccepts, BindsTheGraph)
{
    EXPECT_TRUE(cdnaGraphMatch(GetParam().spec).has_value());
}

std::vector<CdnaCase> acceptedCases()
{
    std::vector<CdnaCase> cases;
    cases.push_back({"NonCausal", cdnaSpec()});
    // An attention sink per query head, rank 1 and rank 4.
    auto sink = cdnaSpec();
    sink.withSink = true;
    cases.push_back({"AttentionSink", sink});
    sink.sinkRank4 = true;
    cases.push_back({"AttentionSinkRank4", sink});
    // A FLOAT sink, cuDNN's and hipDNN's reference's dtype (vendored-kernel modification
    // 36), rank 4 and rank 1, strided across heads.
    sink.sinkDataType = DataType::FLOAT;
    cases.push_back({"FloatSinkRank4", sink});
    sink.sinkRank4 = false;
    sink.sinkHeadStride = 3;
    cases.push_back({"FloatSinkStridedAcrossHeads", sink});
    // The int32 offset bound counts 16-bit halves: a query-dtype sink reaches 2^25 per head
    // over 64 heads, where a FLOAT one declines.
    auto limitSink = cdnaSpec();
    limitSink.withSink = true;
    limitSink.headDim = 64;
    limitSink.heads = 64;
    limitSink.kvHeads = 8;
    limitSink.sinkHeadStride = int64_t{1} << 25;
    cases.push_back({"SinkHeadOffsetsAtTheInt32Limit", limitSink});
    // Multi-head latent attention, un-absorbed: V narrower than Q and K.
    auto mla = cdnaSpec();
    mla.headDim = 192;
    mla.valueHeadDim = 128;
    cases.push_back({"ValueHeadDimNarrower", mla});
    auto topLeft = cdnaSpec();
    topLeft.mask = SdpaMask::TOP_LEFT_CAUSAL;
    cases.push_back({"TopLeftCausal", topLeft});
    auto bottomRight = cdnaSpec();
    bottomRight.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    cases.push_back({"BottomRightCausal", bottomRight});
    auto scaleValue = cdnaSpec();
    scaleValue.scaleKind = SdpaScale::VALUE;
    scaleValue.scale = 0.125F;
    cases.push_back({"ScaleValue", scaleValue});
    auto constantScale = cdnaSpec();
    constantScale.scaleKind = SdpaScale::CONSTANT_TENSOR;
    cases.push_back({"ConstantScaleTensor", constantScale});
    auto runtimeScale = cdnaSpec();
    runtimeScale.scaleKind = SdpaScale::RUNTIME_TENSOR;
    cases.push_back({"RuntimeScaleTensor", runtimeScale});
    // Every value a field can take that the kernels do handle, beside the declines.
    for(const auto computeType : {DataType::UNSET, DataType::HALF, DataType::BFLOAT16})
    {
        auto compute = cdnaSpec();
        compute.computeDataType = computeType;
        cases.push_back({std::string("ComputeType") + EnumNameDataType(computeType), compute});
    }
    auto wideAlignment = cdnaSpec();
    for(const auto uid : {FLYDSL_SDPA_RDNA_Q_UID,
                          FLYDSL_SDPA_RDNA_K_UID,
                          FLYDSL_SDPA_RDNA_V_UID,
                          FLYDSL_SDPA_RDNA_O_UID})
    {
        wideAlignment.alignment[uid] = 256;
    }
    cases.push_back({"OperandsAlignedTo256", wideAlignment});
    auto statsUnset = cdnaSpec();
    statsUnset.stats = SdpaStats::LSE;
    statsUnset.generateStats = std::nullopt;
    cases.push_back({"StatsTensorWithGenerateStatsUnset", statsUnset});
    auto rank4Scale = cdnaSpec();
    rank4Scale.scaleKind = SdpaScale::RUNTIME_TENSOR;
    rank4Scale.scaleDims = {1, 1, 1, 1};
    cases.push_back({"ScaleTensorOfRankFour", rank4Scale});
    auto batched = cdnaSpec();
    batched.batch = 3;
    batched.seqLenQ = 333;
    batched.seqLenKv = 333;
    cases.push_back({"BatchedWithARaggedTail", batched});
    auto mha = cdnaSpec();
    mha.kvHeads = 32;
    cases.push_back({"Mha", mha});
    auto stats = cdnaSpec();
    stats.stats = SdpaStats::LSE;
    cases.push_back({"SoftmaxStats", stats});
    auto bhsd = cdnaSpec();
    bhsd.layout = SdpaLayout::BHSD;
    cases.push_back({"BhsdLayout", bhsd});
    // V with its own layout, and an LSE laid out [B, Sq, H].
    auto vOwn = cdnaSpec();
    vOwn.valueLayout = SdpaLayout::BHSD;
    cases.push_back({"ValueInItsOwnLayout", vOwn});
    auto statsStrided = cdnaSpec();
    statsStrided.stats = SdpaStats::LSE;
    statsStrided.statsSequenceMajor = true;
    cases.push_back({"SequenceMajorStats", statsStrided});
    auto bias = cdnaSpec();
    bias.withBias = true;
    bias.biasDataType = DataType::BFLOAT16;
    cases.push_back({"BiasInTheQueryDtype", bias});
    auto fullBias = bias;
    fullBias.biasDims = {1, 32, 256, 256};
    cases.push_back({"BiasPerHead", fullBias});
    auto paddingMask = bias;
    paddingMask.batch = 2;
    paddingMask.biasDims = {2, 1, 1, 256};
    cases.push_back({"BiasBroadcastOverHeadsAndQueries", paddingMask});
    // Biases only the strided bias objects read (vendored-kernel modification 31).
    auto f32Bias = cdnaSpec();
    f32Bias.withBias = true;
    f32Bias.biasDataType = DataType::FLOAT;
    cases.push_back({"BiasInF32", f32Bias});
    auto keyBroadcast = bias;
    keyBroadcast.biasDims = {1, 1, 256, 1};
    cases.push_back({"BiasBroadcastAlongTheKeys", keyBroadcast});
    auto expanded = bias;
    expanded.batch = 2;
    expanded.biasDims = {2, 32, 256, 256};
    expanded.biasStrides = {256, 0, 0, 1};
    cases.push_back({"BiasExpandedWithStrideZero", expanded});
    auto unaligned = bias;
    unaligned.seqLenQ = 333;
    unaligned.seqLenKv = 333;
    cases.push_back({"BiasRowsNotSixteenByteAligned", unaligned});
    auto biasTail = bias;
    biasTail.seqLenKv = 333;
    biasTail.biasDims = {1, 1, 256, 333};
    biasTail.biasStrides = {86016, 86016, 336, 1};
    cases.push_back({"BiasOverAKeyLengthNotAMultipleOfFour", biasTail});
    auto overlapping = bias;
    overlapping.biasDims = {1, 1, 256, 256};
    overlapping.biasStrides = {2048, 2048, 8, 1};
    cases.push_back({"BiasRowsOverlapping", overlapping});
    // A bias at a head dim only the bucket objects serve, and a bias with a sink.
    auto bucketBias = bias;
    bucketBias.headDim = 96;
    cases.push_back({"BiasAtABucketHeadDim", bucketBias});
    auto biasSink = bias;
    biasSink.withSink = true;
    cases.push_back({"BiasWithASink", biasSink});
    auto packed = cdnaSpec();
    packed.kvHeads = 32;
    packed.packedQkv = true;
    cases.push_back({"PackedQkv", packed});
    auto cross = cdnaSpec();
    cross.seqLenKv = 512;
    cases.push_back({"UnequalSequenceLengths", cross});
    auto chunked = cdnaSpec();
    chunked.mask = SdpaMask::TOP_LEFT_CAUSAL;
    chunked.seqLenQ = 256;
    chunked.seqLenKv = 1024;
    cases.push_back({"TopLeftCausalWithFewerQueriesThanKeys", chunked});
    auto crossCausal = cdnaSpec();
    crossCausal.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    crossCausal.seqLenQ = 1024;
    crossCausal.seqLenKv = 256;
    cases.push_back({"BottomRightCausalWithMoreQueriesThanKeys", crossCausal});
    auto band = cdnaSpec();
    band.mask = SdpaMask::TOP_LEFT_CAUSAL;
    band.seqLenQ = 256;
    band.seqLenKv = 1024;
    band.rightBound = 100;
    cases.push_back({"TopLeftBandInsideTheKeys", band});
    // Diagonals that pass the keys (vendored-kernel modification 30): the last rows see
    // every key.
    auto topLeftOverflow = cdnaSpec();
    topLeftOverflow.mask = SdpaMask::TOP_LEFT_CAUSAL;
    topLeftOverflow.seqLenQ = 512;
    topLeftOverflow.seqLenKv = 256;
    cases.push_back({"TopLeftCausalWithMoreQueriesThanKeys", topLeftOverflow});
    auto bottomRightBand = cdnaSpec();
    bottomRightBand.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    bottomRightBand.rightBound = 8;
    cases.push_back({"BottomRightBandPastTheKeys", bottomRightBand});
    auto wideBand = cdnaSpec();
    wideBand.mask = SdpaMask::TOP_LEFT_CAUSAL;
    wideBand.rightBound = 16;
    cases.push_back({"TopLeftBandPastTheKeys", wideBand});
    // A right bound at the largest value: stored as Skv - 1, without overflowing.
    auto hugeRight = cdnaSpec();
    hugeRight.mask = SdpaMask::TOP_LEFT_CAUSAL;
    hugeRight.rightBound = std::numeric_limits<int64_t>::max();
    cases.push_back({"LargestRightBound", hugeRight});
    auto window = cdnaSpec();
    window.mask = SdpaMask::SLIDING_WINDOW;
    cases.push_back({"SlidingWindow", window});
    auto bottomRightWindow = cdnaSpec();
    bottomRightWindow.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    bottomRightWindow.seqLenQ = 64;
    bottomRightWindow.seqLenKv = 1024;
    bottomRightWindow.leftBound = 128;
    cases.push_back({"BottomRightSlidingWindow", bottomRightWindow});
    // The deprecated booleans alone, or with the right bound they imply.
    auto deprecated = cdnaSpec();
    deprecated.deprecatedCausalTopLeft = true;
    cases.push_back({"DeprecatedTopLeftCausal", deprecated});
    auto deprecatedBottomRight = cdnaSpec();
    deprecatedBottomRight.deprecatedCausalBottomRight = true;
    deprecatedBottomRight.rightBound = 0;
    deprecatedBottomRight.seqLenQ = 64;
    cases.push_back({"DeprecatedBottomRightCausalWithItsOwnRightBound", deprecatedBottomRight});
    // A deprecated boolean with a left_bound keeps the window (gpt-oss SWA graphs).
    auto deprecatedWindow = cdnaSpec();
    deprecatedWindow.deprecatedCausalBottomRight = true;
    deprecatedWindow.leftBound = 32;
    cases.push_back({"DeprecatedCausalWithAWindow", deprecatedWindow});
    // A window wider than any row reaches hides nothing; the bound is compared, never
    // subtracted, so even the largest one is no overflow.
    auto wideWindow = cdnaSpec();
    wideWindow.mask = SdpaMask::TOP_LEFT_CAUSAL;
    wideWindow.leftBound = std::numeric_limits<int64_t>::max();
    cases.push_back({"LargestLeftBound", wideWindow});
    // A dropout probability of 0 is the undropped result every object computes, with or
    // without its seed and offset, as on the RDNA engines (FlydslSdpaDropout.hpp).
    auto noDrop = cdnaSpec();
    noDrop.dropoutProbability = 0.0F;
    cases.push_back({"DropoutProbabilityZero", noDrop});
    auto noDropSeeded = cdnaSpec();
    setDropout(noDropSeeded, 0.0F);
    cases.push_back({"DropoutProbabilityZeroWithASeedAndAnOffset", noDropSeeded});
    // Padding: each batch's own query and key lengths (vendored-kernel modification 32),
    // under every mask, with a bias, a sink and the LSE, and without padding_mask as hipDNN's
    // references read them (their SEM-008).
    cases.push_back({"PaddedSequences", cdnaPaddedSpec()});
    auto paddedNoSwitch = cdnaPaddedSpec();
    paddedNoSwitch.paddingMask = false;
    cases.push_back({"PaddedSequencesWithoutThePaddingMask", paddedNoSwitch});
    for(const auto mask : {SdpaMask::TOP_LEFT_CAUSAL, SdpaMask::BOTTOM_RIGHT_CAUSAL})
    {
        auto paddedCausal = cdnaPaddedSpec();
        paddedCausal.mask = mask;
        cases.push_back(
            {mask == SdpaMask::TOP_LEFT_CAUSAL ? "PaddedTopLeftCausal" : "PaddedBottomRightCausal",
             paddedCausal});
    }
    auto paddedWindow = cdnaPaddedSpec();
    paddedWindow.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    paddedWindow.leftBound = 32;
    cases.push_back({"PaddedBottomRightWindow", paddedWindow});
    auto paddedEverything = cdnaPaddedSpec();
    paddedEverything.withBias = true;
    paddedEverything.biasDataType = paddedEverything.dataType;
    paddedEverything.withSink = true;
    paddedEverything.stats = SdpaStats::LSE;
    cases.push_back({"PaddedWithABiasASinkAndTheLse", paddedEverything});
    return cases;
}

INSTANTIATE_TEST_SUITE_P(,
                         TestFlydslSdpaCdnaGraphAccepts,
                         ::testing::ValuesIn(acceptedCases()),
                         [](const auto& info) { return info.param.name; });

// Every field, each value no kernel handles declined by all three engines sharing the matcher.
class TestFlydslSdpaCdnaFieldDeclines : public ::testing::TestWithParam<SdpaFieldDecline>
{
};

TEST_P(TestFlydslSdpaCdnaFieldDeclines, DeclinesTheField)
{
    auto spec = cdnaSpec();
    ASSERT_TRUE(cdnaGraphMatch(spec).has_value())
        << "the spec the case edits must be accepted on its own";
    if(GetParam().control)
    {
        auto control = spec;
        GetParam().control(control);
        ASSERT_TRUE(cdnaGraphMatch(control).has_value())
            << "the case's feature without its defect must be accepted";
    }
    GetParam().apply(spec);
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    for(const auto* engine :
        {&FLYDSL_SDPA_CDNA, &FLYDSL_SDPA_CDNA_DUALWAVE, &FLYDSL_SDPA_CDNA_DECODE})
    {
        EXPECT_FALSE(matchesGraph(*engine, fixture.context()).has_value()) << engine->engineName;
    }
}

/// The shared table without the f32 case: the generic CDNA objects serve f32 graphs
/// (vendored-kernel modification 35; TestFlydslSdpaCdnaF32Graph), which every other family
/// declines.
std::vector<SdpaFieldDecline> cdnaFieldDeclines()
{
    auto cases = sdpaFieldDeclines();
    cases.erase(std::remove_if(cases.begin(),
                               cases.end(),
                               [](const SdpaFieldDecline& c) { return c.name == "Fp32"; }),
                cases.end());
    return cases;
}

INSTANTIATE_TEST_SUITE_P(,
                         TestFlydslSdpaCdnaFieldDeclines,
                         ::testing::ValuesIn(cdnaFieldDeclines()),
                         [](const auto& info) { return info.param.name; });

// Q, K, V, O, the bias, the LSE and the sink, each a host scalar where the kernel reads a
// device pointer.
INSTANTIATE_TEST_SUITE_P(PassByValue,
                         TestFlydslSdpaCdnaFieldDeclines,
                         ::testing::ValuesIn(sdpaPassByValueDeclines()),
                         [](const auto& info) { return info.param.name; });

/// 2^31: one past the largest 32-bit kernel argument.
constexpr int64_t TWO_POW_31 = int64_t{1} << 31;

/// The CDNA matcher's own bounds on strides, extents and the bias and LSE layouts, one case
/// per bound, each beside a control just inside it. These graphs need only dims and
/// strides: nothing is allocated.
std::vector<SdpaFieldDecline> cdnaBoundDeclines()
{
    std::vector<SdpaFieldDecline> cases;
    const auto edit = [&cases](std::string name,
                               std::function<void(SdpaGraphSpec&)> apply,
                               std::function<void(SdpaGraphSpec&)> control) {
        cases.push_back({std::move(name), std::move(apply), std::move(control)});
    };
    // Operand strides: each a 32-bit kernel argument and a multiple of the 8-element vector,
    // and each (batch, head) slice one buffer view within 2^31 bytes. A batch axis is never
    // stepped inside a slice, so only its own bound holds it.
    edit(
        "OutputBatchStridePastInt32",
        [](SdpaGraphSpec& s) {
            s.batch = 2;
            s.outputBatchStride = TWO_POW_31;
        },
        [](SdpaGraphSpec& s) {
            s.batch = 2;
            s.outputBatchStride = TWO_POW_31 - 8;
        });
    edit(
        "ValueSequenceStrideNotAMultipleOfEight",
        [](SdpaGraphSpec& s) { s.valueSequenceStride = s.kvHeads * s.headDim + 4; },
        [](SdpaGraphSpec& s) { s.valueSequenceStride = s.kvHeads * s.headDim + 8; });
    // (Skv - 1) * 2^24 + D elements is past 2^31 bytes; at 2^22 it is just inside.
    edit(
        "ValueSlicePastTwoGiB",
        [](SdpaGraphSpec& s) { s.valueSequenceStride = int64_t{1} << 24; },
        [](SdpaGraphSpec& s) { s.valueSequenceStride = int64_t{1} << 22; });
    // The batch is a 32-bit kernel argument. (A query or key length past int32 cannot reach
    // that bound: its slice is past 2^31 bytes first.)
    edit(
        "BatchPastInt32",
        [](SdpaGraphSpec& s) { s.batch = TWO_POW_31; },
        [](SdpaGraphSpec& s) { s.batch = TWO_POW_31 - 1; });

    // The bias: read in the query's dtype, keys contiguous, rank at most 4, each stride a
    // 32-bit multiple of 8, and one (batch, head) slice within 2^31 bytes.
    const auto withBias = [](SdpaGraphSpec& s) {
        s.withBias = true;
        s.biasDataType = s.dataType;
    };
    const auto perBatchBias = [withBias](SdpaGraphSpec& s, int64_t batchStride) {
        withBias(s);
        s.batch = 2;
        s.biasDims = {2, 1, s.seqLenQ, s.seqLenKv};
        s.biasStrides = {batchStride, s.seqLenQ * s.seqLenKv, s.seqLenKv, 1};
    };
    edit(
        "BiasBatchStridePastInt32",
        [perBatchBias](SdpaGraphSpec& s) { perBatchBias(s, TWO_POW_31); },
        [perBatchBias](SdpaGraphSpec& s) { perBatchBias(s, TWO_POW_31 - 8); });
    const auto biasRowPitch = [withBias](SdpaGraphSpec& s, int64_t rowStride, int64_t keyStride) {
        withBias(s);
        s.biasDims = {1, 1, s.seqLenQ, s.seqLenKv};
        s.biasStrides = {s.seqLenQ * rowStride, s.seqLenQ * rowStride, rowStride, keyStride};
    };
    edit(
        "BiasSlicePastTwoGiB",
        [biasRowPitch](SdpaGraphSpec& s) { biasRowPitch(s, int64_t{1} << 24, 1); },
        [biasRowPitch](SdpaGraphSpec& s) { biasRowPitch(s, int64_t{1} << 22, 1); });
    // An f32 bias's slice in bytes: twice the elements' reach of the same pitch in bf16.
    edit(
        "F32BiasSlicePastTwoGiB",
        [biasRowPitch](SdpaGraphSpec& s) {
            biasRowPitch(s, int64_t{1} << 22, 1);
            s.biasDataType = DataType::FLOAT;
        },
        [biasRowPitch](SdpaGraphSpec& s) {
            biasRowPitch(s, int64_t{1} << 21, 1);
            s.biasDataType = DataType::FLOAT;
        });
    // The strided objects read keys at their own stride, so the keys' reach counts too.
    edit(
        "BiasKeyReachPastTwoGiB",
        [biasRowPitch](SdpaGraphSpec& s) { biasRowPitch(s, s.seqLenKv, int64_t{1} << 23); },
        [biasRowPitch](SdpaGraphSpec& s) { biasRowPitch(s, s.seqLenKv, int64_t{1} << 21); });
    edit(
        "BiasOfRankFive",
        [withBias](SdpaGraphSpec& s) {
            withBias(s);
            s.biasDims = {1, 1, 1, s.seqLenQ, s.seqLenKv};
        },
        [withBias](SdpaGraphSpec& s) {
            withBias(s);
            s.biasDims = {1, 1, s.seqLenQ, s.seqLenKv};
        });

    // The LSE: every element within 32-bit byte offsets (the merge kernel addresses the
    // whole tensor from one base), and each stepped axis at a positive stride.
    const auto batchedStats = [](SdpaGraphSpec& s) {
        s.stats = SdpaStats::LSE;
        s.batch = 2;
    };
    edit(
        "StatsPastThirtyTwoBitByteOffsets",
        [batchedStats](SdpaGraphSpec& s) {
            batchedStats(s);
            s.statsBatchStride = int64_t{1} << 29;
        },
        [batchedStats](SdpaGraphSpec& s) {
            batchedStats(s);
            // The last element's offset one below 2^29, its byte offset just inside 2^31.
            s.statsBatchStride = (int64_t{1} << 29) - s.heads * s.seqLenQ - 1;
        });
    edit(
        "StatsBatchStrideZero",
        [batchedStats](SdpaGraphSpec& s) {
            batchedStats(s);
            s.statsBatchStride = 0;
        },
        batchedStats);
    edit(
        "StatsBatchStrideNegative",
        [batchedStats](SdpaGraphSpec& s) {
            batchedStats(s);
            s.statsBatchStride = -s.heads * s.seqLenQ;
        },
        batchedStats);
    return cases;
}

INSTANTIATE_TEST_SUITE_P(Bounds,
                         TestFlydslSdpaCdnaFieldDeclines,
                         ::testing::ValuesIn(cdnaBoundDeclines()),
                         [](const auto& info) { return info.param.name; });

/// Each gate on the padding fields (FlydslSdpaVarlen.hpp, and servableVarlenAsReferenceReads's
/// lengths without padding_mask), applied to a padded graph the CDNA matcher accepts.
std::vector<SdpaFieldDecline> cdnaPaddingDeclines()
{
    std::vector<SdpaFieldDecline> cases;
    const auto add = [&cases](std::string name, std::function<void(SdpaGraphSpec&)> edit) {
        cases.push_back({std::move(name),
                         [edit = std::move(edit)](SdpaGraphSpec& s) {
                             s = cdnaPaddedSpec();
                             edit(s);
                         },
                         [](SdpaGraphSpec& s) { s = cdnaPaddedSpec(); }});
    };
    const auto noSwitch = [](SdpaGraphSpec& s) { s.paddingMask = false; };
    add("PaddingMaskWithoutQueryLengths", [](SdpaGraphSpec& s) { s.seqLensQ.clear(); });
    add("PaddingMaskWithoutKeyLengths", [](SdpaGraphSpec& s) { s.seqLensKv.clear(); });
    add("QueryLengthsAlone", [noSwitch](SdpaGraphSpec& s) {
        noSwitch(s);
        s.seqLensKv.clear();
    });
    add("KeyLengthsAlone", [noSwitch](SdpaGraphSpec& s) {
        noSwitch(s);
        s.seqLensQ.clear();
    });
    add("Int64Lengths", [](SdpaGraphSpec& s) { s.seqLenDataType = DataType::INT64; });
    add("Int64LengthsWithoutThePaddingMask", [noSwitch](SdpaGraphSpec& s) {
        noSwitch(s);
        s.seqLenDataType = DataType::INT64;
    });
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
    add("VirtualQueryLengthsWithoutThePaddingMask", [noSwitch](SdpaGraphSpec& s) {
        noSwitch(s);
        s.virtualTensors.insert(FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID);
    });
    add("VirtualKeyLengthsWithoutThePaddingMask", [noSwitch](SdpaGraphSpec& s) {
        noSwitch(s);
        s.virtualTensors.insert(FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID);
    });
    add("RaggedQueryLengths",
        [](SdpaGraphSpec& s) { s.raggedTensors.insert(FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID); });
    add("RaggedKeyLengths",
        [](SdpaGraphSpec& s) { s.raggedTensors.insert(FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID); });
    add("QueryLengthsAlignedTo2",
        [](SdpaGraphSpec& s) { s.alignment[FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID] = 2; });
    add("KeyLengthsAlignedTo2WithoutThePaddingMask", [noSwitch](SdpaGraphSpec& s) {
        noSwitch(s);
        s.alignment[FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID] = 2;
    });
    add("PaddedRaggedQuery",
        [](SdpaGraphSpec& s) { s.raggedTensors.insert(FLYDSL_SDPA_RDNA_Q_UID); });
    add("RaggedQueryWithLengthsWithoutThePaddingMask", [noSwitch](SdpaGraphSpec& s) {
        noSwitch(s);
        s.raggedTensors.insert(FLYDSL_SDPA_RDNA_Q_UID);
    });
    return cases;
}

INSTANTIATE_TEST_SUITE_P(Padding,
                         TestFlydslSdpaCdnaFieldDeclines,
                         ::testing::ValuesIn(cdnaPaddingDeclines()),
                         [](const auto& info) { return info.param.name; });

// ---------------------------------------------------------------------------
// Packed (THD) sequences, compact (vendored-kernel modification 34): Q/O and K/V ragged
// through cumulative token tables, the LSE packed through Q's. The generic prefill objects
// read the tables; the dual-wave and decode objects decline.
// ---------------------------------------------------------------------------

/// The model shape's two sequences packed end to end, the LSE packed alike.
SdpaGraphSpec cdnaPackedSpec()
{
    auto spec = cdnaSpec();
    spec.batch = 2;
    spec.stats = SdpaStats::LSE;
    setPacked(spec, {256, 100}, {200, 256});
    return spec;
}

constexpr const char* CDNA_RAGGED_OFFSET_Q_TOKEN = "flydsl_sdpa_cdna.ragged_offset_q.uid";
constexpr const char* CDNA_RAGGED_OFFSET_KV_TOKEN = "flydsl_sdpa_cdna.ragged_offset_kv.uid";

TEST(TestFlydslSdpaCdnaPackedGraph, BindsTheTablesOfPackedOperands)
{
    using Bound = std::optional<int64_t>;
    const auto bound = cdnaGraphMatch(cdnaPackedSpec());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, CDNA_RAGGED_OFFSET_Q_TOKEN),
              Bound(FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID));
    EXPECT_EQ(tryGetBoundInt(*bound, CDNA_RAGGED_OFFSET_KV_TOKEN),
              Bound(FLYDSL_SDPA_RDNA_RAGGED_OFFSET_KV_UID));
    EXPECT_EQ(tryGetBoundInt(*bound, "flydsl_sdpa_cdna.seq_len_q.uid"), Bound(-1));
    EXPECT_EQ(tryGetBoundInt(*bound, CDNA_STATS_TOKEN), Bound(FLYDSL_SDPA_RDNA_STATS_UID));

    const auto dense = cdnaGraphMatch(cdnaSpec());
    ASSERT_TRUE(dense.has_value());
    EXPECT_EQ(tryGetBoundInt(*dense, CDNA_RAGGED_OFFSET_Q_TOKEN), Bound(-1));
    EXPECT_EQ(tryGetBoundInt(*dense, CDNA_RAGGED_OFFSET_KV_TOKEN), Bound(-1));
}

TEST(TestFlydslSdpaCdnaPackedGraph, AcceptsOneTableASinkEveryMaskAndNoLse)
{
    // Self-attention: one table for every operand.
    auto shared = cdnaPackedSpec();
    setPacked(shared, {256, 100}, {256, 100});
    shared.raggedOffsetOf[FLYDSL_SDPA_RDNA_K_UID] = FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID;
    shared.raggedOffsetOf[FLYDSL_SDPA_RDNA_V_UID] = FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID;
    const auto bound = cdnaGraphMatch(shared);
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, CDNA_RAGGED_OFFSET_KV_TOKEN),
              std::optional<int64_t>(FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID));

    auto noLse = cdnaPackedSpec();
    noLse.stats = SdpaStats::NONE;
    EXPECT_TRUE(cdnaGraphMatch(noLse).has_value());
    auto sink = cdnaPackedSpec();
    sink.withSink = true;
    EXPECT_TRUE(cdnaGraphMatch(sink).has_value());
    for(const auto mask :
        {SdpaMask::TOP_LEFT_CAUSAL, SdpaMask::BOTTOM_RIGHT_CAUSAL, SdpaMask::SLIDING_WINDOW})
    {
        auto masked = cdnaPackedSpec();
        masked.mask = mask;
        EXPECT_TRUE(cdnaGraphMatch(masked).has_value()) << static_cast<int>(mask);
    }
}

/// Each gate on packed sequences, applied to a packed graph the CDNA matcher accepts.
std::vector<SdpaFieldDecline> cdnaPackedDeclines()
{
    std::vector<SdpaFieldDecline> cases;
    const auto add = [&cases](std::string name, std::function<void(SdpaGraphSpec&)> edit) {
        cases.push_back({std::move(name),
                         [edit = std::move(edit)](SdpaGraphSpec& s) {
                             s = cdnaPackedSpec();
                             edit(s);
                         },
                         [](SdpaGraphSpec& s) { s = cdnaPackedSpec(); }});
    };
    constexpr int64_t Q_TABLE = FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID;
    constexpr int64_t KV_TABLE = FLYDSL_SDPA_RDNA_RAGGED_OFFSET_KV_UID;
    // An operand left dense beside packed ones, counting elements, or through the other table.
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
    add("HeadMajorPacking", [](SdpaGraphSpec& s) { s.layout = SdpaLayout::BHSD; });
    // The LSE: dense beside packed operands, or packed other than Q is.
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
    add("OffsetsAtAStrideOfTwo", [](SdpaGraphSpec& s) { s.raggedOffsetBatchStride = 2; });
    add("VirtualQueryOffsets",
        [](SdpaGraphSpec& s) { s.virtualTensors.insert(FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID); });
    add("KeyOffsetsPassedByValue", [](SdpaGraphSpec& s) {
        s.passByValueTensors.insert(FLYDSL_SDPA_RDNA_RAGGED_OFFSET_KV_UID);
    });
    // Per-batch lengths with packing (P3): both or none, each one the kernels read.
    add("PackedWithQueryLengths", [](SdpaGraphSpec& s) { s.seqLensQ = {256, 100}; });
    add("PackedWithKeyLengths", [](SdpaGraphSpec& s) { s.seqLensKv = {200, 256}; });
    add("PackedUnderThePaddingMaskAlone", [](SdpaGraphSpec& s) { s.paddingMask = true; });
    add("PackedAndPaddedWithInt64Lengths", [](SdpaGraphSpec& s) {
        setPadding(s, {256, 100}, {200, 256});
        s.seqLenDataType = DataType::INT64;
    });
    add("PackedAndPaddedWithVirtualKeyLengths", [](SdpaGraphSpec& s) {
        setPadding(s, {256, 100}, {200, 256});
        s.virtualTensors.insert(FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID);
    });
    add("PaddedWithADenseKey", [](SdpaGraphSpec& s) {
        setPadding(s, {256, 100}, {200, 256});
        s.unpackedTensors.insert(FLYDSL_SDPA_RDNA_K_UID);
    });
    add("PaddedWithOffsetsCountingElements", [](SdpaGraphSpec& s) {
        setPadding(s, {256, 100}, {200, 256});
        s.raggedMultiplierOf[FLYDSL_SDPA_RDNA_O_UID] = 1;
    });
    add("PaddedWithKeysCountingElements", [](SdpaGraphSpec& s) {
        setPadding(s, {256, 100}, {200, 256});
        s.raggedMultiplierOf[FLYDSL_SDPA_RDNA_K_UID] = 1;
    });
    add("PaddedWithVirtualQueryOffsets", [](SdpaGraphSpec& s) {
        setPadding(s, {256, 100}, {200, 256});
        s.virtualTensors.insert(FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID);
    });
    add("PaddedWithInt64Offsets", [](SdpaGraphSpec& s) {
        setPadding(s, {256, 100}, {200, 256});
        s.raggedOffsetDataType = DataType::INT64;
    });
    add("PaddedWithADenseLse", [](SdpaGraphSpec& s) {
        setPadding(s, {256, 100}, {200, 256});
        s.packedStats = false;
    });
    add("PackedPaddedAndDropping", [](SdpaGraphSpec& s) {
        setPadding(s, {256, 100}, {200, 256});
        setDropout(s, 0.1F);
    });
    add("PackedPaddedWithABias", [](SdpaGraphSpec& s) {
        setPadding(s, {256, 100}, {200, 256});
        s.withBias = true;
        s.biasDataType = s.dataType;
    });
    // What packed sequences are not served with.
    add("PackedWithABias", [](SdpaGraphSpec& s) {
        s.withBias = true;
        s.biasDataType = s.dataType;
    });
    add("PackedAndDropping", [](SdpaGraphSpec& s) { setDropout(s, 0.1F); });
    return cases;
}

INSTANTIATE_TEST_SUITE_P(Packed,
                         TestFlydslSdpaCdnaFieldDeclines,
                         ::testing::ValuesIn(cdnaPackedDeclines()),
                         [](const auto& info) { return info.param.name; });

TEST(TestFlydslSdpaCdnaPackedGraph, BindsPackedSequencesWithPerBatchLengths)
{
    // P3: packed slots and the real lengths within them, with or without padding_mask; a
    // sink, every mask and no LSE too.
    using Bound = std::optional<int64_t>;
    auto spec = cdnaPackedSpec();
    setPadding(spec, {200, 100}, {150, 256});
    for(const bool paddingMask : {true, false})
    {
        SCOPED_TRACE(paddingMask ? "padding_mask" : "lengths alone");
        spec.paddingMask = paddingMask;
        const auto bound = cdnaGraphMatch(spec);
        ASSERT_TRUE(bound.has_value());
        EXPECT_EQ(tryGetBoundInt(*bound, "flydsl_sdpa_cdna.seq_len_q.uid"),
                  Bound(FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID));
        EXPECT_EQ(tryGetBoundInt(*bound, "flydsl_sdpa_cdna.seq_len_kv.uid"),
                  Bound(FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID));
        EXPECT_EQ(tryGetBoundInt(*bound, CDNA_RAGGED_OFFSET_Q_TOKEN),
                  Bound(FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID));
        EXPECT_EQ(tryGetBoundInt(*bound, CDNA_RAGGED_OFFSET_KV_TOKEN),
                  Bound(FLYDSL_SDPA_RDNA_RAGGED_OFFSET_KV_UID));
    }
    auto variant = spec;
    variant.withSink = true;
    variant.stats = SdpaStats::NONE;
    variant.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    variant.leftBound = 32;
    EXPECT_TRUE(cdnaGraphMatch(variant).has_value());
}

TEST(TestFlydslSdpaCdnaPackedGraph, DeclinesARaggedLseUnderDenseOperands)
{
    // A ragged LSE beside dense operands: no table to pack it through.
    auto spec = cdnaSpec();
    spec.stats = SdpaStats::LSE;
    spec.statsSequenceMajor = true;
    ASSERT_TRUE(cdnaGraphMatch(spec).has_value());
    spec.raggedTensors.insert(FLYDSL_SDPA_RDNA_STATS_UID);
    EXPECT_FALSE(cdnaGraphMatch(spec).has_value());
}

class TestFlydslSdpaCdnaGraphDeclines : public ::testing::TestWithParam<CdnaCase>
{
};

TEST_P(TestFlydslSdpaCdnaGraphDeclines, DeclinesTheGraph)
{
    EXPECT_FALSE(cdnaGraphMatch(GetParam().spec).has_value());
}

std::vector<CdnaCase> declinedCases()
{
    std::vector<CdnaCase> cases;
    // A sink in neither the query's dtype nor FLOAT.
    auto halfSink = cdnaSpec();
    halfSink.withSink = true;
    halfSink.sinkDataType = DataType::HALF;
    cases.push_back({"SinkInTheOther16BitDtype", halfSink});
    // A head's sink offset h * stride past int32: the kernels compute it in 16-bit halves in
    // 32 bits, so heads 32 and up would read another head's logit; a FLOAT sink's element is
    // two halves, so 2^25 already reaches past it.
    auto wideSinkStride = cdnaSpec();
    wideSinkStride.withSink = true;
    wideSinkStride.headDim = 64;
    wideSinkStride.heads = 64;
    wideSinkStride.kvHeads = 8;
    wideSinkStride.sinkHeadStride = int64_t{1} << 26;
    cases.push_back({"SinkHeadOffsetsPastInt32", wideSinkStride});
    wideSinkStride.sinkHeadStride = int64_t{1} << 25;
    wideSinkStride.sinkDataType = DataType::FLOAT;
    cases.push_back({"FloatSinkHeadOffsetsPastInt32", wideSinkStride});
    auto wideV = cdnaSpec();
    wideV.valueHeadDim = 256;
    cases.push_back({"ValueHeadDimWider", wideV});
    auto leftOnly = cdnaSpec();
    leftOnly.leftBound = 64;
    cases.push_back({"SlidingWindowWithNoRightBound", leftOnly});
    // A bias in neither the query's dtype nor f32.
    auto f16Bias = cdnaSpec();
    f16Bias.withBias = true;
    f16Bias.biasDataType = DataType::HALF;
    cases.push_back({"BiasInAThirdDtype", f16Bias});
    // A negative stride on a stepped bias axis.
    auto backwardsBias = cdnaSpec();
    backwardsBias.withBias = true;
    backwardsBias.biasDataType = DataType::BFLOAT16;
    backwardsBias.biasDims = {1, 1, 256, 256};
    backwardsBias.biasStrides = {65536, 65536, -256, 1};
    cases.push_back({"BiasWithANegativeQueryStride", backwardsBias});
    // A key axis neither the key length nor 1: no broadcast reads it.
    auto wrongKeys = cdnaSpec();
    wrongKeys.withBias = true;
    wrongKeys.biasDataType = DataType::BFLOAT16;
    wrongKeys.biasDims = {1, 1, 256, 128};
    cases.push_back({"BiasOfTheWrongKeyLength", wrongKeys});
    auto bf16Stats = cdnaSpec();
    bf16Stats.stats = SdpaStats::LSE_BF16;
    cases.push_back({"Bf16Stats", bf16Stats});
    auto wideStats = cdnaSpec();
    wideStats.stats = SdpaStats::LSE_WRONG_SHAPE;
    cases.push_back({"StatsOfTheWrongShape", wideStats});
    auto statsNowhere = cdnaSpec();
    statsNowhere.stats = SdpaStats::REQUESTED_WITHOUT_TENSOR;
    cases.push_back({"StatsRequestedWithoutATensor", statsNowhere});
    auto deviceScale = cdnaSpec();
    deviceScale.scaleKind = SdpaScale::DEVICE_TENSOR;
    cases.push_back({"DeviceScaleTensor", deviceScale});
    auto negativeScale = cdnaSpec();
    negativeScale.scaleKind = SdpaScale::VALUE;
    negativeScale.scale = -0.125F;
    cases.push_back({"NegativeScale", negativeScale});
    // Past the shared bound every family serves (2^20).
    auto hugeScale = cdnaSpec();
    hugeScale.scaleKind = SdpaScale::VALUE;
    hugeScale.scale = 2097152.0F;
    cases.push_back({"ScaleAboveTheBound", hugeScale});
    auto dropout = cdnaSpec();
    dropout.withDropout = true;
    cases.push_back({"Dropout", dropout});
    auto ragged = cdnaSpec();
    ragged.raggedQuery = true;
    cases.push_back({"RaggedQuery", ragged});
    auto group = cdnaSpec();
    group.heads = 30;
    group.kvHeads = 8;
    cases.push_back({"HeadsNotAMultipleOfKvHeads", group});
    auto bothDeprecated = cdnaSpec();
    bothDeprecated.deprecatedCausalTopLeft = true;
    bothDeprecated.deprecatedCausalBottomRight = true;
    cases.push_back({"BothDeprecatedBooleans", bothDeprecated});
    auto unknownAlignment = cdnaSpec();
    unknownAlignment.mask = SdpaMask::TOP_LEFT_CAUSAL;
    unknownAlignment.diagonalAlignment
        = static_cast<hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment>(7);
    cases.push_back({"UnknownDiagonalAlignment", unknownAlignment});
    auto unknownAlignmentDeprecated = cdnaSpec();
    unknownAlignmentDeprecated.deprecatedCausalTopLeft = true;
    unknownAlignmentDeprecated.diagonalAlignment
        = static_cast<hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment>(7);
    cases.push_back(
        {"UnknownDiagonalAlignmentUnderTheDeprecatedBoolean", unknownAlignmentDeprecated});
    // Each query head reads V's head through K's.
    auto valueHeads = cdnaSpec();
    valueHeads.valueHeads = 4;
    cases.push_back({"ValueHeadsOfTheirOwn", valueHeads});
    return cases;
}

INSTANTIATE_TEST_SUITE_P(,
                         TestFlydslSdpaCdnaGraphDeclines,
                         ::testing::ValuesIn(declinedCases()),
                         [](const auto& info) { return info.param.name; });

TEST(TestFlydslSdpaCdnaBinding, BindsTheCausalFlag)
{
    auto causal = cdnaSpec();
    causal.mask = SdpaMask::TOP_LEFT_CAUSAL;
    const auto causalBound = cdnaGraphMatch(causal);
    const auto plainBound = cdnaGraphMatch(cdnaSpec());
    ASSERT_TRUE(causalBound.has_value());
    ASSERT_TRUE(plainBound.has_value());
    EXPECT_EQ(tryGetBoundInt(*causalBound, CDNA_CAUSAL_TOKEN), 1);
    EXPECT_EQ(tryGetBoundInt(*plainBound, CDNA_CAUSAL_TOKEN), 0);
}

TEST(TestFlydslSdpaCdnaBinding, BindsTheWindowsLowerEdgeOrNone)
{
    const auto windowOf = [](SdpaMask mask, int64_t sq, int64_t skv, std::optional<int64_t> left) {
        auto spec = cdnaSpec();
        spec.mask = mask;
        spec.seqLenQ = sq;
        spec.seqLenKv = skv;
        spec.leftBound = left;
        const auto bound = cdnaGraphMatch(spec);
        return bound.has_value() ? tryGetBoundInt(*bound, CDNA_WINDOW_TOKEN) : std::nullopt;
    };
    EXPECT_EQ(windowOf(SdpaMask::TOP_LEFT_CAUSAL, 256, 256, std::nullopt), CDNA_NO_WINDOW);
    // left_bound = L keeps the L keys ending at the diagonal (cuDNN), so the lower edge is
    // align - (L - 1).
    EXPECT_EQ(windowOf(SdpaMask::TOP_LEFT_CAUSAL, 256, 256, 64), -63);
    EXPECT_EQ(windowOf(SdpaMask::TOP_LEFT_CAUSAL, 256, 256, 1), 0);
    EXPECT_EQ(windowOf(SdpaMask::TOP_LEFT_CAUSAL, 256, 256, 2), -1);
    EXPECT_EQ(windowOf(SdpaMask::BOTTOM_RIGHT_CAUSAL, 256, 1024, 64), 705);
    // A window no row reaches is none, so the dual-wave objects still serve it.
    EXPECT_EQ(windowOf(SdpaMask::TOP_LEFT_CAUSAL, 256, 256, 256), CDNA_NO_WINDOW);
    EXPECT_EQ(windowOf(SdpaMask::TOP_LEFT_CAUSAL, 256, 256, 255), -254);
    EXPECT_EQ(windowOf(SdpaMask::TOP_LEFT_CAUSAL, 256, 256, 254), -253);
}

TEST(TestFlydslSdpaCdnaBinding, BindsTheDiagonalOffset)
{
    const auto offsetOf = [](SdpaMask mask, int64_t sq, int64_t skv, std::optional<int64_t> right) {
        auto spec = cdnaSpec();
        spec.mask = mask;
        spec.seqLenQ = sq;
        spec.seqLenKv = skv;
        spec.rightBound = right;
        const auto bound = cdnaGraphMatch(spec);
        return bound.has_value() ? tryGetBoundInt(*bound, CDNA_DIAG_OFFSET_TOKEN) : std::nullopt;
    };
    EXPECT_EQ(offsetOf(SdpaMask::TOP_LEFT_CAUSAL, 256, 1024, std::nullopt), 0);
    EXPECT_EQ(offsetOf(SdpaMask::BOTTOM_RIGHT_CAUSAL, 256, 1024, std::nullopt), 768);
    EXPECT_EQ(offsetOf(SdpaMask::BOTTOM_RIGHT_CAUSAL, 1024, 256, std::nullopt), -768);
    EXPECT_EQ(offsetOf(SdpaMask::TOP_LEFT_CAUSAL, 256, 1024, 100), 100);
    // Past the keys (modification 30): as the graph states it, below Skv - 1.
    EXPECT_EQ(offsetOf(SdpaMask::TOP_LEFT_CAUSAL, 1024, 256, std::nullopt), 0);
    EXPECT_EQ(offsetOf(SdpaMask::BOTTOM_RIGHT_CAUSAL, 1024, 256, 900), 132);
    EXPECT_EQ(offsetOf(SdpaMask::TOP_LEFT_CAUSAL, 256, 256, 254), 254);
}

TEST(TestFlydslSdpaCdnaBinding, BindsADiagonalThatShowsEveryKeyAsNoMask)
{
    // A diagonal at Skv - 1 or past it, with no window, shows every row every key: it binds as
    // no mask, so the dual-wave and split decode objects serve it too. At one query row a
    // bottom-right diagonal sits at Skv - 1 inside the keys and stays causal; so does a
    // diagonal one short of it, and one with a window.
    const auto bind = [](SdpaMask mask,
                         int64_t sq,
                         int64_t skv,
                         std::optional<int64_t> right,
                         std::optional<int64_t> left = std::nullopt) {
        auto spec = cdnaSpec();
        spec.mask = mask;
        spec.seqLenQ = sq;
        spec.seqLenKv = skv;
        spec.rightBound = right;
        spec.leftBound = left;
        const auto bound = cdnaGraphMatch(spec);
        return bound.has_value() ? std::pair(tryGetBoundInt(*bound, CDNA_CAUSAL_TOKEN),
                                             tryGetBoundInt(*bound, CDNA_DIAG_OFFSET_TOKEN))
                                 : std::pair(std::optional<int64_t>(), std::optional<int64_t>());
    };
    using Bound = std::optional<int64_t>;
    const auto noMask = std::pair(Bound(0), Bound(0));
    EXPECT_EQ(bind(SdpaMask::TOP_LEFT_CAUSAL, 256, 256, 255), noMask);
    EXPECT_EQ(bind(SdpaMask::TOP_LEFT_CAUSAL, 256, 256, 256), noMask);
    EXPECT_EQ(bind(SdpaMask::BOTTOM_RIGHT_CAUSAL, 1024, 256, 2000), noMask);
    EXPECT_EQ(bind(SdpaMask::TOP_LEFT_CAUSAL, 256, 256, std::numeric_limits<int64_t>::max()),
              noMask);
    EXPECT_EQ(bind(SdpaMask::TOP_LEFT_CAUSAL, 2, 256, 255), noMask);
    // Kept causal.
    EXPECT_EQ(bind(SdpaMask::TOP_LEFT_CAUSAL, 256, 256, 254), std::pair(Bound(1), Bound(254)));
    EXPECT_EQ(bind(SdpaMask::BOTTOM_RIGHT_CAUSAL, 1, 256, std::nullopt),
              std::pair(Bound(1), Bound(255)));
    EXPECT_EQ(bind(SdpaMask::TOP_LEFT_CAUSAL, 1, 256, 255), std::pair(Bound(1), Bound(255)));
    EXPECT_EQ(bind(SdpaMask::TOP_LEFT_CAUSAL, 256, 256, 255, 16), std::pair(Bound(1), Bound(255)));
}

TEST(TestFlydslSdpaCdnaBinding, BindsTheDeprecatedBooleansAtTheirAlignment)
{
    // causal_mask is right 0 at the alignment the graph stores, so diagonal_alignment
    // refines it; causal_mask_bottom_right is always bottom-right. The reference resolves
    // both this way.
    using hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment;
    const auto bind = [](bool topLeft, bool bottomRight, DiagonalAlignment alignment) {
        auto spec = cdnaSpec();
        spec.deprecatedCausalTopLeft = topLeft;
        spec.deprecatedCausalBottomRight = bottomRight;
        spec.diagonalAlignment = alignment;
        spec.seqLenQ = 256;
        spec.seqLenKv = 1024;
        return cdnaGraphMatch(spec);
    };
    const auto topLeft = bind(true, false, DiagonalAlignment::TOP_LEFT);
    const auto refined = bind(true, false, DiagonalAlignment::BOTTOM_RIGHT);
    const auto bottomRight = bind(false, true, DiagonalAlignment::TOP_LEFT);
    ASSERT_TRUE(topLeft.has_value());
    ASSERT_TRUE(refined.has_value());
    ASSERT_TRUE(bottomRight.has_value());
    EXPECT_EQ(tryGetBoundInt(*topLeft, CDNA_CAUSAL_TOKEN), 1);
    EXPECT_EQ(tryGetBoundInt(*topLeft, CDNA_DIAG_OFFSET_TOKEN), 0);
    EXPECT_EQ(tryGetBoundInt(*refined, CDNA_CAUSAL_TOKEN), 1);
    EXPECT_EQ(tryGetBoundInt(*refined, CDNA_DIAG_OFFSET_TOKEN), 768);
    EXPECT_EQ(tryGetBoundInt(*bottomRight, CDNA_CAUSAL_TOKEN), 1);
    EXPECT_EQ(tryGetBoundInt(*bottomRight, CDNA_DIAG_OFFSET_TOKEN), 768);

    // A right_bound the graph sets wins over the boolean's 0: a band 16 keys to the right
    // of the top-left diagonal, not plain causal.
    auto band = cdnaSpec();
    band.deprecatedCausalTopLeft = true;
    band.diagonalAlignment = DiagonalAlignment::TOP_LEFT;
    band.seqLenQ = 256;
    band.seqLenKv = 1024;
    band.rightBound = 16;
    const auto banded = cdnaGraphMatch(band);
    ASSERT_TRUE(banded.has_value());
    EXPECT_EQ(tryGetBoundInt(*banded, CDNA_CAUSAL_TOKEN), 1);
    EXPECT_EQ(tryGetBoundInt(*banded, CDNA_DIAG_OFFSET_TOKEN), 16);
}

TEST(TestFlydslSdpaCdnaBinding, BindsTheBiasOrNone)
{
    auto withBias = cdnaSpec();
    withBias.withBias = true;
    withBias.biasDataType = DataType::BFLOAT16;
    const auto biasBound = cdnaGraphMatch(withBias);
    const auto plainBound = cdnaGraphMatch(cdnaSpec());
    ASSERT_TRUE(biasBound.has_value());
    ASSERT_TRUE(plainBound.has_value());
    EXPECT_EQ(tryGetBoundInt(*biasBound, CDNA_BIAS_TOKEN), FLYDSL_SDPA_RDNA_BIAS_UID);
    EXPECT_EQ(tryGetBoundInt(*plainBound, CDNA_BIAS_TOKEN), -1);
}

TEST(TestFlydslSdpaCdnaKernelMatch, RunsABiasGraphOnlyOnAnObjectBuiltToAddIt)
{
    auto withBias = cdnaSpec();
    withBias.withBias = true;
    withBias.biasDataType = DataType::BFLOAT16;
    for(const auto& [spec, graphHasBias] :
        {std::pair<SdpaGraphSpec, bool>{withBias, true}, {cdnaSpec(), false}})
    {
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
        const auto bound = matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
        ASSERT_TRUE(bound.has_value());
        for(const auto& [family, blockM] :
            {std::pair<std::string, int64_t>{"dualwave", 256}, {"generic", 128}})
        {
            const auto plain = makeCdnaKernel(family, 128, 0, "bf16", blockM, 0, 1, 0);
            const auto biased = makeCdnaKernel(family, 128, 0, "bf16", blockM, 0, 1, 1);
            EXPECT_EQ(matchesCdnaKernel(fixture.context(), plain, *bound), !graphHasBias) << family;
            EXPECT_EQ(matchesCdnaKernel(fixture.context(), biased, *bound), graphHasBias) << family;
        }
    }
}

TEST(TestFlydslSdpaCdnaBinding, BindsTheStatsTensorOrNone)
{
    auto withStats = cdnaSpec();
    withStats.stats = SdpaStats::LSE;
    const auto statsBound = cdnaGraphMatch(withStats);
    const auto plainBound = cdnaGraphMatch(cdnaSpec());
    ASSERT_TRUE(statsBound.has_value());
    ASSERT_TRUE(plainBound.has_value());
    EXPECT_EQ(tryGetBoundInt(*statsBound, CDNA_STATS_TOKEN), FLYDSL_SDPA_RDNA_STATS_UID);
    EXPECT_EQ(tryGetBoundInt(*plainBound, CDNA_STATS_TOKEN), -1);
}

TEST(TestFlydslSdpaCdnaBinding, BindsTheScaleAsAValueOrATensor)
{
    // No scale on the graph binds the default, 1.0 (cuDNN's), as a value.
    const auto defaulted = cdnaGraphMatch(cdnaSpec());
    ASSERT_TRUE(defaulted.has_value());
    const float expected = 1.0F;
    int32_t expectedBits = 0;
    std::memcpy(&expectedBits, &expected, sizeof(expected));
    EXPECT_EQ(tryGetBoundInt(*defaulted, CDNA_SCALE_SOURCE_TOKEN), 0);
    EXPECT_EQ(tryGetBoundInt(*defaulted, CDNA_SCALE_TOKEN), expectedBits);

    auto runtime = cdnaSpec();
    runtime.scaleKind = SdpaScale::RUNTIME_TENSOR;
    const auto tensor = cdnaGraphMatch(runtime);
    ASSERT_TRUE(tensor.has_value());
    EXPECT_EQ(tryGetBoundInt(*tensor, CDNA_SCALE_SOURCE_TOKEN), 1);
    EXPECT_EQ(tryGetBoundInt(*tensor, CDNA_SCALE_TOKEN), FLYDSL_SDPA_RDNA_SCALE_UID);
}

// ---------------------------------------------------------------------------
// kernel_match
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaCdnaKernelMatch, AdmitsTheCandidateBuiltForTheGraphsShape)
{
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(cdnaSpec()));
    const auto bound = matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_TRUE(matchesCdnaKernel(fixture.context(), makeCdnaKernel(), *bound));
    EXPECT_TRUE(matchesCdnaKernel(
        fixture.context(), makeCdnaKernel("generic", 128, 0, "bf16", 128), *bound));
}

TEST(TestFlydslSdpaCdnaKernelMatch, AnMlaObjectServesItsVHeadDimAndQkHeadDimsUpToItsBucket)
{
    // The qk_mask object: Q/K head dims up to its 256 bucket in steps of 16, V exactly 128.
    const auto mla = makeCdnaKernel("generic", 256, 0, "bf16", 128, 0, 1, 0, int64_t{128}, 1);
    const auto plain256 = makeCdnaKernel("generic", 256, 0, "bf16", 128);
    const auto matches = [](int64_t headDim, int64_t headDimV, const KernelDefinition& kernel) {
        auto spec = cdnaSpec();
        spec.headDim = headDim;
        spec.valueHeadDim = headDimV;
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
        const auto bound = matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
        return bound.has_value() && matchesCdnaKernel(fixture.context(), kernel, *bound);
    };
    EXPECT_TRUE(matches(192, 128, mla));
    EXPECT_TRUE(matches(256, 128, mla));
    EXPECT_TRUE(matches(144, 128, mla));
    // Not a multiple of the masks' 8 columns, past the bucket, or another V width.
    EXPECT_FALSE(matches(196, 128, mla));
    EXPECT_TRUE(matches(200, 128, mla));
    EXPECT_FALSE(matches(192, 192, mla));
    EXPECT_FALSE(matches(192, 64, mla));
    // An object without masks serves its own head dims only.
    EXPECT_FALSE(matches(192, 128, plain256));
    EXPECT_TRUE(matches(256, 256, plain256));
    EXPECT_FALSE(matches(256, 256, mla));
}

TEST(TestFlydslSdpaCdnaKernelMatch, ABucketObjectServesEveryHeadDimItsMasksCover)
{
    // A qk_mask object whose V is as wide as its Q/K bucket: V follows Q/K's runtime width.
    const auto bucket = makeCdnaKernel("generic", 128, 0, "bf16", 64, 0, 1, 0, int64_t{128}, 1);
    const auto matches = [](int64_t headDim, int64_t headDimV, const KernelDefinition& kernel) {
        auto spec = cdnaSpec();
        spec.headDim = headDim;
        spec.valueHeadDim = headDimV;
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
        const auto bound = matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
        return bound.has_value() && matchesCdnaKernel(fixture.context(), kernel, *bound);
    };
    EXPECT_TRUE(matches(80, 80, bucket));
    EXPECT_TRUE(matches(112, 112, bucket));
    EXPECT_TRUE(matches(128, 128, bucket));
    EXPECT_FALSE(matches(80, 64, bucket)) << "V follows Q/K";
    EXPECT_TRUE(matches(88, 88, bucket));
    EXPECT_FALSE(matches(84, 84, bucket)) << "not a multiple of the masks' 8 columns";
    EXPECT_FALSE(matches(144, 144, bucket)) << "past the bucket";
}

TEST(TestFlydslSdpaCdnaKernelMatch, AdmitsAnyHeadCountBecauseTheyAreKernelArguments)
{
    for(const auto& [heads, kvHeads] : {std::pair<int64_t, int64_t>{8, 8},
                                        std::pair<int64_t, int64_t>{16, 1},
                                        std::pair<int64_t, int64_t>{40, 5},
                                        std::pair<int64_t, int64_t>{64, 8}})
    {
        auto spec = cdnaSpec();
        spec.heads = heads;
        spec.kvHeads = kvHeads;
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
        const auto bound = matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
        ASSERT_TRUE(bound.has_value()) << heads << "/" << kvHeads;
        EXPECT_TRUE(matchesCdnaKernel(fixture.context(), makeCdnaKernel(), *bound))
            << heads << "/" << kvHeads;
    }
}

TEST(TestFlydslSdpaCdnaKernelMatch, DeclinesEachBakedFieldThatDisagrees)
{
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(cdnaSpec()));
    const auto bound = matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());
    const auto declines = [&](const KernelDefinition& kernel) {
        return !matchesCdnaKernel(fixture.context(), kernel, *bound);
    };
    EXPECT_TRUE(declines(makeCdnaKernel("dualwave", 64)));
    EXPECT_TRUE(declines(makeCdnaKernel("dualwave", 128, 1)));
    EXPECT_TRUE(declines(makeCdnaKernel("dualwave", 128, 0, "f16")));
}

TEST(TestFlydslSdpaCdnaKernelMatch, RunsTheOffsetZeroObjectOnlyAtOffsetZero)
{
    const auto admits = [](SdpaMask mask, int64_t sq, int64_t skv, const KernelDefinition& kernel) {
        auto spec = cdnaSpec();
        spec.mask = mask;
        spec.seqLenQ = sq;
        spec.seqLenKv = skv;
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
        const auto bound = matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
        return bound.has_value() && matchesCdnaKernel(fixture.context(), kernel, *bound);
    };
    const auto offsetZero = makeCdnaKernel("dualwave", 128, 1, "bf16", 256, 0, 0);
    const auto anyOffset = makeCdnaKernel("dualwave", 128, 1, "bf16", 256, 0, 1);
    EXPECT_TRUE(admits(SdpaMask::BOTTOM_RIGHT_CAUSAL, 256, 256, offsetZero));
    EXPECT_TRUE(admits(SdpaMask::TOP_LEFT_CAUSAL, 256, 1024, offsetZero));
    EXPECT_FALSE(admits(SdpaMask::BOTTOM_RIGHT_CAUSAL, 256, 1024, offsetZero));
    EXPECT_TRUE(admits(SdpaMask::BOTTOM_RIGHT_CAUSAL, 256, 1024, anyOffset));
}

TEST(TestFlydslSdpaCdnaKernelMatch, DeclinesATileWhosePaddingRowsPassThirtyTwoBits)
{
    // BSHD with 131072 query heads: the sequence stride is 2^24 elements. Two query rows fit,
    // but a tile's padding rows are addressed past them: at 128 or 256 rows the last one's
    // byte offset passes 2^31 and would wrap onto a real row; at 64 rows it does not.
    auto spec = cdnaSpec();
    spec.heads = 131072;
    spec.kvHeads = 1;
    spec.seqLenQ = 2;
    spec.seqLenKv = 256;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    const auto bound = matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_TRUE(matchesCdnaKernel(
        fixture.context(), makeCdnaKernel("generic", 128, 1, "bf16", 64), *bound));
    EXPECT_FALSE(matchesCdnaKernel(
        fixture.context(), makeCdnaKernel("generic", 128, 1, "bf16", 128), *bound));
    EXPECT_FALSE(matchesCdnaKernel(fixture.context(), makeCdnaKernel("dualwave", 128, 1), *bound));
}

/// Whether the CDNA graph matcher binds @p spec and @p kernel's own engine admits it, on a
/// device with @p computeUnits compute units (0: the test default, which reports none).
bool cdnaAdmits(const SdpaGraphSpec& spec, const KernelDefinition& kernel, int computeUnits = 0)
{
    auto properties = testDeviceProperties();
    properties.multiProcessorCount = computeUnits;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec), properties);
    const auto bound = matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
    return bound.has_value() && matchesCdnaKernel(fixture.context(), kernel, *bound);
}

TEST(TestFlydslSdpaCdnaKernelMatch, RunsABiasTheVectorizedObjectsCannotReadOnTheStridedOnes)
{
    // Each bias the vectorized objects (the dual-wave kernel's and the generic kernel's
    // other bias objects) cannot read, one gate each, beside the bf16 bias they read: the
    // strided objects (vendored-kernel modification 31) read every one.
    auto base = cdnaSpec();
    base.withBias = true;
    base.biasDataType = DataType::BFLOAT16;
    base.biasDims = {1, 1, 256, 256};
    base.biasStrides = {65536, 65536, 256, 1};
    const auto edited = [&base](const std::function<void(SdpaGraphSpec&)>& edit) {
        auto spec = base;
        edit(spec);
        return spec;
    };
    const std::vector<std::pair<std::string, SdpaGraphSpec>> unread{
        {"f32", edited([](SdpaGraphSpec& s) { s.biasDataType = DataType::FLOAT; })},
        {"broadcast along the keys", edited([](SdpaGraphSpec& s) {
             s.biasDims = {1, 1, 256, 1};
             s.biasStrides = {256, 256, 1, 1};
         })},
        {"keys not contiguous",
         edited([](SdpaGraphSpec& s) { s.biasStrides = {131072, 131072, 512, 2}; })},
        {"a row pitch not a multiple of 8",
         edited([](SdpaGraphSpec& s) { s.biasStrides = {66560, 66560, 260, 1}; })},
        {"a head pitch not a multiple of 8", edited([](SdpaGraphSpec& s) {
             s.biasDims = {1, 32, 256, 256};
             s.biasStrides = {2097280, 65540, 256, 1};
         })},
        {"rows that overlap",
         edited([](SdpaGraphSpec& s) { s.biasStrides = {65536, 65536, 248, 1}; })},
        {"a key length not a multiple of 4", edited([](SdpaGraphSpec& s) {
             s.seqLenKv = 334;
             s.biasDims = {1, 1, 256, 334};
             s.biasStrides = {86016, 86016, 336, 1};
         })},
    };
    const auto kstrided = [] {
        auto kernel = makeCdnaKernel("generic", 128, 0, "bf16", 128, 0, 1, 1, int64_t{128}, 1);
        kernel.metadata[CDNA_BIAS_KSTRIDED_FIELD] = int64_t{1};
        return kernel;
    }();
    const auto vectorBucket
        = makeCdnaKernel("generic", 128, 0, "bf16", 128, 0, 1, 1, int64_t{128}, 1);
    const auto vectorExact = makeCdnaKernel("generic", 128, 0, "bf16", 128, 0, 1, 1);
    const auto dualwave = makeCdnaKernel("dualwave", 128, 0, "bf16", 256, 0, 1, 1);
    // The control: the vectorized objects read the base bias, and so do the strided ones.
    EXPECT_TRUE(cdnaAdmits(base, vectorExact));
    EXPECT_TRUE(cdnaAdmits(base, vectorBucket));
    EXPECT_TRUE(cdnaAdmits(base, dualwave));
    EXPECT_TRUE(cdnaAdmits(base, kstrided));
    for(const auto& [name, spec] : unread)
    {
        SCOPED_TRACE(name);
        EXPECT_FALSE(cdnaAdmits(spec, vectorExact));
        EXPECT_FALSE(cdnaAdmits(spec, vectorBucket));
        EXPECT_FALSE(cdnaAdmits(spec, dualwave));
        EXPECT_TRUE(cdnaAdmits(spec, kstrided));
    }
    // A strided object is a bias object: it serves no graph without a bias.
    EXPECT_FALSE(cdnaAdmits(cdnaSpec(), kstrided));
}

TEST(TestFlydslSdpaCdnaKernelMatch, AddsABiasAtABucketHeadDim)
{
    // A bias at head dim 96 on a bucket bias object; the plain bucket object declines it.
    auto bucket = cdnaSpec();
    bucket.headDim = 96;
    bucket.withBias = true;
    bucket.biasDataType = DataType::BFLOAT16;
    EXPECT_TRUE(cdnaAdmits(
        bucket, makeCdnaKernel("generic", 128, 0, "bf16", 128, 0, 1, 1, int64_t{128}, 1)));
    EXPECT_FALSE(cdnaAdmits(
        bucket, makeCdnaKernel("generic", 128, 0, "bf16", 128, 0, 1, 0, int64_t{128}, 1)));
}

TEST(TestFlydslSdpaCdnaKernelMatch, AddsABiasWithASink)
{
    // A bias with a sink on a bias and sink object; the sink object without a bias and the
    // bias object without a sink decline it.
    auto sink = cdnaSpec();
    sink.withBias = true;
    sink.biasDataType = DataType::BFLOAT16;
    sink.withSink = true;
    const auto withSink = [](int64_t hasBias) {
        auto kernel = makeCdnaKernel("generic", 128, 0, "bf16", 128, 0, 1, hasBias);
        kernel.metadata["has_sink"] = int64_t{1};
        return kernel;
    };
    EXPECT_TRUE(cdnaAdmits(sink, withSink(1)));
    EXPECT_FALSE(cdnaAdmits(sink, withSink(0)));
    EXPECT_FALSE(cdnaAdmits(sink, makeCdnaKernel("generic", 128, 0, "bf16", 128, 0, 1, 1)));
    // A bias the vectorized objects do not read -- f32, or broadcast along the keys -- runs
    // with a sink on the strided sink-and-bias objects alone (COVERAGE.md section 7.5 item
    // 16), which read the vectorized ones' bias too.
    auto stridedWithSink = withSink(1);
    stridedWithSink.metadata["bias_kstrided"] = int64_t{1};
    auto f32Sink = sink;
    f32Sink.biasDataType = DataType::FLOAT;
    auto keyBroadcastSink = sink;
    keyBroadcastSink.biasDims = {1, 32, 256, 1};
    for(const auto& spec : {f32Sink, keyBroadcastSink})
    {
        EXPECT_FALSE(cdnaAdmits(spec, withSink(1)));
        EXPECT_TRUE(cdnaAdmits(spec, stridedWithSink));
    }
    EXPECT_TRUE(cdnaAdmits(sink, stridedWithSink));
    // Without a sink, the strided sink object declines; without a bias, it declines too.
    auto f32NoSink = f32Sink;
    f32NoSink.withSink = false;
    EXPECT_FALSE(cdnaAdmits(f32NoSink, stridedWithSink));
}

TEST(TestFlydslSdpaCdnaKernelMatch, DeclinesATileWhoseQueryOrOutputPaddingRowsPassThirtyTwoBits)
{
    // Q's and O's padding rows are bounded each on its own. Q alone: Q/K 192 wide and V/O
    // 128 (an MLA object), 65536 query heads, so Q's sequence stride is half again O's.
    // At 128-row tiles Q's last padding row passes 2^31 bytes and O's does not; at 64 both
    // fit.
    auto queryOnly = cdnaSpec();
    queryOnly.heads = 65536;
    queryOnly.kvHeads = 1;
    queryOnly.headDim = 192;
    queryOnly.valueHeadDim = 128;
    queryOnly.seqLenQ = 2;
    queryOnly.mask = SdpaMask::TOP_LEFT_CAUSAL;
    const auto mlaTile = [](int64_t blockM) {
        return makeCdnaKernel("generic", 256, 1, "bf16", blockM, 0, 1, 0, int64_t{128}, 1);
    };
    EXPECT_TRUE(cdnaAdmits(queryOnly, mlaTile(64)));
    EXPECT_FALSE(cdnaAdmits(queryOnly, mlaTile(128)));

    // O alone: a padded O row pitch of 2^24 elements, Q packed.
    auto outputOnly = cdnaSpec();
    outputOnly.seqLenQ = 2;
    outputOnly.mask = SdpaMask::TOP_LEFT_CAUSAL;
    outputOnly.outputSequenceStride = int64_t{1} << 24;
    EXPECT_TRUE(cdnaAdmits(outputOnly, makeCdnaKernel("generic", 128, 1, "bf16", 64)));
    EXPECT_FALSE(cdnaAdmits(outputOnly, makeCdnaKernel("generic", 128, 1, "bf16", 128)));
}

TEST(TestFlydslSdpaCdnaKernelMatch, DeclinesASplitObjectWhoseOutputPassesThirtyTwoBits)
{
    // The merge kernel stores O at one 32-bit byte offset over the whole tensor: four batches
    // of 65536 queries x 32 heads x 128 reach 2^31 bytes, though each (batch, head) slice
    // is far inside; two batches fit.
    auto spec = cdnaSpec();
    spec.batch = 4;
    spec.seqLenQ = 65536;
    const auto split = makeCdnaKernel("decode", 128, 0, "bf16", 64);
    EXPECT_FALSE(cdnaAdmits(spec, split));
    spec.batch = 2;
    EXPECT_TRUE(cdnaAdmits(spec, split));
}

TEST(TestFlydslSdpaCdnaKernelMatch, DeclinesAPackedObjectWhoseQueryGroupPassesThirtyTwoBits)
{
    // A packed object views one KV head's group of query heads as one buffer. Absorbed MLA,
    // BSHD, 16 query heads over one KV head: at 116509 queries each Q slice is just inside
    // 2^31 bytes, but the group reaches 15 heads (8640 elements) further, past it; O, 512
    // wide, fits whole. At 116000 queries the group fits too.
    auto spec = cdnaSpec();
    spec.headDim = 576;
    spec.valueHeadDim = 512;
    spec.heads = 16;
    spec.kvHeads = 1;
    spec.seqLenQ = 116509;
    const auto packed = makeCdnaKernel("decode", 576, 0, "bf16", 64, 0, 1, 0, int64_t{512});
    EXPECT_FALSE(cdnaAdmits(spec, packed));
    spec.seqLenQ = 116000;
    EXPECT_TRUE(cdnaAdmits(spec, packed));
}

TEST(TestFlydslSdpaCdnaKernelMatch, DeclinesALaunchGridPastInt32)
{
    // The generic kernel flattens (batch, query tile, query head) onto x: 2^24 batches of
    // four 64-row tiles over 32 heads is 2^31 workgroups, one past int32; 2^24 - 1 fit.
    auto generic = cdnaSpec();
    generic.batch = int64_t{1} << 24;
    const auto tile = makeCdnaKernel("generic", 128, 0, "bf16", 64);
    EXPECT_FALSE(cdnaAdmits(generic, tile));
    generic.batch = (int64_t{1} << 24) - 1;
    EXPECT_TRUE(cdnaAdmits(generic, tile));

    // The dual-wave kernel's grid takes the query heads as one dimension: 2^31 heads (BHSD,
    // one batch, so no batch stride steps past them) do not fit; 2^30 do.
    auto dualwave = cdnaSpec();
    dualwave.layout = SdpaLayout::BHSD;
    dualwave.kvHeads = 1;
    dualwave.heads = int64_t{1} << 31;
    EXPECT_FALSE(cdnaAdmits(dualwave, makeCdnaKernel()));
    dualwave.heads = int64_t{1} << 30;
    EXPECT_TRUE(cdnaAdmits(dualwave, makeCdnaKernel()));
}

TEST(TestFlydslSdpaCdnaKernelMatch, DeclinesASplitWorkspacePastThirtyTwoBits)
{
    // A device with a very large compute-unit count splits the keys 64 ways: the partials
    // of 32 heads x 2048 queries x 129 floats per split then pass 2^31 bytes. With the test
    // default (no compute units reported) the same graph runs unsplit.
    auto spec = cdnaSpec();
    spec.seqLenQ = 2048;
    spec.seqLenKv = 16384;
    const auto split = makeCdnaKernel("decode", 128, 0, "bf16", 64);
    EXPECT_FALSE(cdnaAdmits(spec, split, 1 << 20));
    EXPECT_TRUE(cdnaAdmits(spec, split));
}

TEST(TestFlydslSdpaCdnaKernelMatch, SizesTheSplitWorkspaceForEachPartialAndItsLse)
{
    // The launch lays the split workspace out as the f32 partial O, Dv per (batch, query
    // head, query position, split), then each partial's LSE: B * H * Sq * splits * (Dv + 1)
    // f32s, and the caller allocates what workspaceBytes reports. The split count is the
    // engine's own choice, so the size must be a whole number of splits of that row, between
    // 2 and 64. One stat short per row is not: Dv and Dv + 1 are coprime and Dv + 1 > 64.
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
    for(const auto& shape : {Shape{1, 32, 8, 1, 8192, 128, 0},
                             Shape{2, 16, 2, 1, 20000, 64, 0},
                             Shape{1, 8, 8, 3, 4096, 128, 304},
                             Shape{3, 4, 1, 2, 70000, 64, 80}})
    {
        SCOPED_TRACE("B " + std::to_string(shape.batch) + ", H " + std::to_string(shape.heads)
                     + ", Sq " + std::to_string(shape.seqLenQ) + ", Skv "
                     + std::to_string(shape.seqLenKv) + ", D " + std::to_string(shape.headDim));
        auto spec = cdnaSpec();
        spec.batch = shape.batch;
        spec.heads = shape.heads;
        spec.kvHeads = shape.kvHeads;
        spec.seqLenQ = shape.seqLenQ;
        spec.seqLenKv = shape.seqLenKv;
        spec.headDim = shape.headDim;
        auto properties = testDeviceProperties();
        properties.multiProcessorCount = shape.computeUnits;
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec), properties);
        const auto bound = matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
        ASSERT_TRUE(bound.has_value());
        const auto kernel = makeCdnaKernel("decode", shape.headDim, 0, "bf16", 64);
        ASSERT_TRUE(matchesCdnaKernel(fixture.context(), kernel, *bound));
        const auto bytes = dispatchHandler(FLYDSL_SDPA_CDNA_DECODE)
                               .workspaceBytes(fixture.context(), *bound, kernel);
        const auto perSplit
            = static_cast<size_t>(shape.batch * shape.heads * shape.seqLenQ * (shape.headDim + 1))
              * sizeof(float);
        EXPECT_EQ(bytes % perSplit, 0U) << bytes << " bytes";
        EXPECT_GE(bytes / perSplit, 2U) << bytes << " bytes";
        EXPECT_LE(bytes / perSplit, 64U) << bytes << " bytes";
    }
}

TEST(TestFlydslSdpaCdnaKernelMatch, RunsAnyDiagonalOnTheGenericAndDecodeObjects)
{
    // Only the dual-wave kernel bakes the diagonal; the generic and decode objects take any
    // offset (here 768 and 8191) as an argument.
    auto prefill = cdnaSpec();
    prefill.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    prefill.seqLenKv = 1024;
    EXPECT_TRUE(cdnaAdmits(prefill, makeCdnaKernel("generic", 128, 1, "bf16", 128)));
    auto decode = cdnaSpec();
    decode.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    decode.seqLenQ = 1;
    decode.seqLenKv = 8192;
    EXPECT_TRUE(cdnaAdmits(decode, makeCdnaKernel("decode", 128, 1, "bf16", 64)));
}

TEST(TestFlydslSdpaCdnaKernelMatch, RunsADiagonalPastTheKeysOnlyOnTheGenericPrefillObjects)
{
    // The generic prefill objects clamp each row's diagonal to the last key (modification
    // 30); the dual-wave and split decode objects mask their last KV tile's padding with the
    // diagonal alone, so they decline a diagonal that passes the keys, the dual-wave
    // offset-0 object included (top-left with more queries than keys is offset 0).
    auto topLeft = cdnaSpec();
    topLeft.mask = SdpaMask::TOP_LEFT_CAUSAL;
    topLeft.seqLenQ = 512;
    topLeft.seqLenKv = 256;
    auto band = cdnaSpec();
    band.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    band.rightBound = 1;
    auto shortQuery = cdnaSpec();
    shortQuery.mask = SdpaMask::TOP_LEFT_CAUSAL;
    shortQuery.seqLenQ = 40;
    shortQuery.seqLenKv = 20;
    for(const auto& [name, spec] : {std::pair{"top-left, Sq > Skv", topLeft},
                                    std::pair{"band one past the keys", band},
                                    std::pair{"40 queries over 20 keys", shortQuery}})
    {
        SCOPED_TRACE(name);
        EXPECT_TRUE(cdnaAdmits(spec, makeCdnaKernel("generic", 128, 1, "bf16", 128)));
        EXPECT_FALSE(cdnaAdmits(spec, makeCdnaKernel("decode", 128, 1, "bf16", 64)));
        EXPECT_FALSE(cdnaAdmits(spec, makeCdnaKernel("dualwave", 128, 1)));
        EXPECT_FALSE(cdnaAdmits(spec, makeCdnaKernel("dualwave", 128, 1, "bf16", 256, 0, 0)));
    }
    // The control: the same diagonals inside the keys stay on the dual-wave and decode
    // objects, and a non-causal graph with more queries than keys is no diagonal at all.
    auto inside = band;
    inside.rightBound = 0;
    EXPECT_TRUE(cdnaAdmits(inside, makeCdnaKernel("dualwave", 128, 1)));
    EXPECT_TRUE(cdnaAdmits(inside, makeCdnaKernel("decode", 128, 1, "bf16", 64)));
    auto nonCausal = topLeft;
    nonCausal.mask = SdpaMask::NONE;
    EXPECT_TRUE(cdnaAdmits(nonCausal, makeCdnaKernel("dualwave", 128, 0)));
    EXPECT_TRUE(cdnaAdmits(nonCausal, makeCdnaKernel("decode", 128, 0, "bf16", 64)));
}

TEST(TestFlydslSdpaCdnaKernelMatch, RunsAnExact512PrefillGraphOnTheExactObject)
{
    // The exact 512/512 prefill objects serve 512/512 only: a narrower Q/K, a narrower V, or a
    // bias (no 512 bias object) declines on them.
    auto spec = cdnaSpec();
    spec.heads = 1;
    spec.kvHeads = 1;
    spec.headDim = 512;
    auto wide = makeCdnaKernel("generic", 512, 0, "bf16", 128);
    wide.metadata["dv_split"] = int64_t{4};
    EXPECT_TRUE(cdnaAdmits(spec, wide));
    auto narrower = spec;
    narrower.headDim = 504;
    EXPECT_FALSE(cdnaAdmits(narrower, wide));
    auto mla = spec;
    mla.valueHeadDim = 256;
    EXPECT_FALSE(cdnaAdmits(mla, wide));
    auto bias = spec;
    bias.withBias = true;
    bias.biasDataType = bias.dataType;
    EXPECT_FALSE(cdnaAdmits(bias, wide));
}

TEST(TestFlydslSdpaCdnaKernelMatch, RunsADiagonalThatShowsEveryKeyOnEveryNonCausalObject)
{
    // A band wide enough that every row sees every key binds as no mask, so the dual-wave and
    // split decode objects' non-causal builds serve it beside the generic one, and no causal
    // object does. One key short of that, it stays on the generic causal objects alone.
    auto everyKey = cdnaSpec();
    everyKey.mask = SdpaMask::TOP_LEFT_CAUSAL;
    everyKey.seqLenQ = 512;
    everyKey.seqLenKv = 256;
    everyKey.rightBound = 255;
    for(const auto& kernel : {makeCdnaKernel("generic", 128, 0, "bf16", 128),
                              makeCdnaKernel("decode", 128, 0, "bf16", 64),
                              makeCdnaKernel("dualwave", 128, 0)})
    {
        SCOPED_TRACE(kernel.name);
        EXPECT_TRUE(cdnaAdmits(everyKey, kernel));
    }
    EXPECT_FALSE(cdnaAdmits(everyKey, makeCdnaKernel("generic", 128, 1, "bf16", 128)));
    auto oneShort = everyKey;
    oneShort.rightBound = 254;
    EXPECT_TRUE(cdnaAdmits(oneShort, makeCdnaKernel("generic", 128, 1, "bf16", 128)));
    EXPECT_FALSE(cdnaAdmits(oneShort, makeCdnaKernel("decode", 128, 1, "bf16", 64)));
    EXPECT_FALSE(cdnaAdmits(oneShort, makeCdnaKernel("dualwave", 128, 0)));
}

TEST(TestFlydslSdpaCdnaKernelMatch, RunsASlidingWindowOnlyOnTheGenericKernel)
{
    auto spec = cdnaSpec();
    spec.mask = SdpaMask::SLIDING_WINDOW;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    const auto bound = matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_FALSE(matchesCdnaKernel(fixture.context(), makeCdnaKernel("dualwave", 128, 1), *bound));
    EXPECT_TRUE(matchesCdnaKernel(
        fixture.context(), makeCdnaKernel("generic", 128, 1, "bf16", 128), *bound));
}

TEST(TestFlydslSdpaCdnaKernelMatch, AddsASinkOnlyOnTheGenericKernel)
{
    auto spec = cdnaSpec();
    spec.withSink = true;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    const auto bound = matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());
    auto withSink = makeCdnaKernel("generic", 128, 0, "bf16", 64);
    withSink.metadata["has_sink"] = int64_t{1};
    EXPECT_TRUE(matchesCdnaKernel(fixture.context(), withSink, *bound));
    // A plain object, or the dual-wave kernel, which builds none.
    EXPECT_FALSE(matchesCdnaKernel(
        fixture.context(), makeCdnaKernel("generic", 128, 0, "bf16", 64), *bound));
    EXPECT_FALSE(matchesCdnaKernel(fixture.context(), makeCdnaKernel(), *bound));
    // A sink object does not serve a graph without one.
    const GraphFixture plain(buildFlydslSdpaRdnaGraph(cdnaSpec()));
    const auto plainBound = matchesGraph(FLYDSL_SDPA_CDNA, plain.context());
    ASSERT_TRUE(plainBound.has_value());
    EXPECT_FALSE(matchesCdnaKernel(plain.context(), withSink, *plainBound));
}

TEST(TestFlydslSdpaCdnaKernelMatch, ServesAFloatSinkOnTheSinkObjects)
{
    // A FLOAT sink (vendored-kernel modification 36) binds its dtype and its head stride in
    // f32 elements, and runs on the sink objects, which read either dtype; a query-dtype sink
    // binds 0.
    auto spec = cdnaSpec();
    spec.withSink = true;
    spec.sinkDataType = DataType::FLOAT;
    spec.sinkHeadStride = 3;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    const auto bound = matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, CDNA_SINK_FLOAT_TOKEN), std::optional<int64_t>(1));
    EXPECT_EQ(tryGetBoundInt(*bound, CDNA_SINK_STRIDE_TOKEN), std::optional<int64_t>(3));
    auto withSink = makeCdnaKernel("generic", 128, 0, "bf16", 64);
    withSink.metadata["has_sink"] = int64_t{1};
    EXPECT_TRUE(matchesCdnaKernel(fixture.context(), withSink, *bound));
    EXPECT_FALSE(matchesCdnaKernel(
        fixture.context(), makeCdnaKernel("generic", 128, 0, "bf16", 64), *bound));
    spec.sinkDataType.reset();
    const GraphFixture query(buildFlydslSdpaRdnaGraph(spec));
    const auto queryBound = matchesGraph(FLYDSL_SDPA_CDNA, query.context());
    ASSERT_TRUE(queryBound.has_value());
    EXPECT_EQ(tryGetBoundInt(*queryBound, CDNA_SINK_FLOAT_TOKEN), std::optional<int64_t>(0));
    EXPECT_TRUE(matchesCdnaKernel(query.context(), withSink, *queryBound));
}

TEST(TestFlydslSdpaCdnaKernelMatch, ReadsVThroughItsOwnStridesOnlyOnTheMaskedPath)
{
    auto spec = cdnaSpec();
    spec.valueLayout = SdpaLayout::BHSD;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    const auto bound = matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());
    // A Q/K-masked object reads V through V's strides; an exact one, or the dual-wave
    // kernel, reads it through K's.
    EXPECT_TRUE(matchesCdnaKernel(fixture.context(),
                                  makeCdnaKernel("generic", 128, 0, "bf16", 128, 0, 1, 0, 128, 1),
                                  *bound));
    EXPECT_FALSE(matchesCdnaKernel(
        fixture.context(), makeCdnaKernel("generic", 128, 0, "bf16", 128), *bound));
    EXPECT_FALSE(matchesCdnaKernel(fixture.context(), makeCdnaKernel(), *bound));
    // With V laid out as K, every object serves.
    const GraphFixture same(buildFlydslSdpaRdnaGraph(cdnaSpec()));
    const auto sameBound = matchesGraph(FLYDSL_SDPA_CDNA, same.context());
    ASSERT_TRUE(sameBound.has_value());
    EXPECT_TRUE(matchesCdnaKernel(same.context(), makeCdnaKernel(), *sameBound));
}

TEST(TestFlydslSdpaCdnaKernelMatch, WritesAStridedLseOnlyOnTheGenericKernel)
{
    auto spec = cdnaSpec();
    spec.stats = SdpaStats::LSE;
    spec.statsSequenceMajor = true;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    const auto bound = matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_TRUE(matchesCdnaKernel(
        fixture.context(), makeCdnaKernel("generic", 128, 0, "bf16", 128), *bound));
    EXPECT_FALSE(matchesCdnaKernel(fixture.context(), makeCdnaKernel(), *bound));
}

TEST(TestFlydslSdpaCdnaKernelMatch, RunsASplitObjectOnlyWithItsMergeKernel)
{
    auto spec = cdnaSpec();
    spec.seqLenQ = 1;
    spec.seqLenKv = 8192;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    const auto bound = matchesGraph(FLYDSL_SDPA_CDNA_DECODE, fixture.context());
    ASSERT_TRUE(bound.has_value());
    auto split = makeCdnaKernel("decode", 128, 0, "bf16", 64);
    EXPECT_TRUE(matchesCdnaKernel(fixture.context(), split, *bound));
    // Without the merge kernel's name its splits could not be combined.
    split.metadata[CDNA_MERGE_SYMBOL_FIELD] = std::string();
    EXPECT_FALSE(matchesCdnaKernel(fixture.context(), split, *bound));
    // The decode engine serves every query length its grid and workspace hold, not only
    // decode steps: which engine runs a graph several serve is the heuristic's to say.
    spec.seqLenQ = 512;
    spec.seqLenKv = 512;
    const GraphFixture prefill(buildFlydslSdpaRdnaGraph(spec));
    const auto prefillBound = matchesGraph(FLYDSL_SDPA_CDNA_DECODE, prefill.context());
    ASSERT_TRUE(prefillBound.has_value());
    EXPECT_TRUE(matchesCdnaKernel(
        prefill.context(), makeCdnaKernel("decode", 128, 0, "bf16", 64), *prefillBound));
}

TEST(TestFlydslSdpaCdnaKernelMatch, HoldsTheKeyLengthToTheCandidatesAlignment)
{
    const auto admits = [](int64_t seqLen, const KernelDefinition& kernel) {
        auto spec = cdnaSpec();
        spec.seqLenQ = seqLen;
        spec.seqLenKv = seqLen;
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
        const auto bound = matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
        return bound.has_value() && matchesCdnaKernel(fixture.context(), kernel, *bound);
    };
    const auto aligned = makeCdnaKernel("generic", 128, 0, "bf16", 128, 64);
    const auto anyLength = makeCdnaKernel("generic", 128, 0, "bf16", 128, 0);
    EXPECT_TRUE(admits(256, aligned));
    EXPECT_FALSE(admits(333, aligned)) << "the object skips the KV tail mask";
    EXPECT_TRUE(admits(256, anyLength));
    EXPECT_TRUE(admits(333, anyLength));

    // An object that does not say which it is may be either, so it is not run.
    auto unstated = anyLength;
    unstated.metadata.erase(std::string(CDNA_KV_ALIGN_FIELD));
    EXPECT_FALSE(admits(256, unstated));
}

TEST(TestFlydslSdpaCdnaKernelMatch, ScoresACompanionAboveTheDefaultWhereItsRulePrefersIt)
{
    const auto scoreAt = [](int64_t batch, int64_t seqLen, const KernelDefinition& kernel) {
        auto spec = cdnaSpec();
        spec.batch = batch;
        spec.seqLenQ = seqLen;
        spec.seqLenKv = seqLen;
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
        return scoreCdnaKernel(fixture.context(), kernel);
    };
    const auto withRule
        = [](KernelDefinition kernel, const std::string& feature, int64_t low, int64_t high) {
              kernel.priority = 80;
              kernel.metadata[CDNA_PREFER_FEATURE_FIELD] = feature;
              kernel.metadata[CDNA_PREFER_MIN_FIELD] = low;
              kernel.metadata[CDNA_PREFER_MAX_FIELD] = high;
              return kernel;
          };

    // The default carries no rule: it scores its priority at every size. The generic kernel's
    // tiles rank against each other this way, the engine's interim native ranker until a
    // trained UHD ranker replaces it.
    auto defaultTier = withRule(makeCdnaKernel("generic", 128, 0, "bf16", 128), "none", 0, 0);
    defaultTier.priority = 100;
    EXPECT_EQ(scoreAt(1, 128, defaultTier), 100.0);
    EXPECT_EQ(scoreAt(8, 8192, defaultTier), 100.0);

    // A short-query companion: above the default below 256 query rows, below it after.
    const auto shortQuery
        = withRule(makeCdnaKernel("generic", 128, 0, "bf16", 64), "seq_q", 0, 256);
    EXPECT_GT(scoreAt(1, 128, shortQuery), scoreAt(1, 128, defaultTier));
    EXPECT_LT(scoreAt(1, 256, shortQuery), scoreAt(1, 256, defaultTier));

    // A large-problem companion, by batch x heads x query rows (32 heads here), from
    // its lower bound with no upper one.
    const auto large = withRule(makeCdnaKernel("generic", 128, 0, "bf16", 256), "rows", 49152, 0);
    EXPECT_LT(scoreAt(1, 1024, large), scoreAt(1, 1024, defaultTier));
    EXPECT_GT(scoreAt(2, 1024, large), scoreAt(2, 1024, defaultTier));
    EXPECT_GT(scoreAt(64, 8192, large), scoreAt(64, 8192, defaultTier));

    // An object that states no rule ranks by its priority alone.
    EXPECT_EQ(scoreAt(1, 128, makeCdnaKernel()), 100.0);
}

TEST(TestFlydslSdpaCdnaBinding, BindsThePerBatchLengthsAndTheDiagonalsAlignment)
{
    const auto lengths = [](const BoundTokens& bound) {
        return std::pair{tryGetBoundInt(bound, "flydsl_sdpa_cdna.seq_len_q.uid"),
                         tryGetBoundInt(bound, "flydsl_sdpa_cdna.seq_len_kv.uid")};
    };
    using Bound = std::optional<int64_t>;
    auto padded = cdnaPaddedSpec();
    padded.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    const auto bound = cdnaGraphMatch(padded);
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(
        lengths(*bound),
        std::pair(Bound(FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID), Bound(FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID)));
    EXPECT_EQ(tryGetBoundInt(*bound, "flydsl_sdpa_cdna.diag_bottom_right"), 1);
    // Lengths without padding_mask bind the same (hipDNN's references' SEM-008).
    padded.paddingMask = false;
    const auto noSwitch = cdnaGraphMatch(padded);
    ASSERT_TRUE(noSwitch.has_value());
    EXPECT_EQ(lengths(*noSwitch), lengths(*bound));

    auto topLeft = cdnaSpec();
    topLeft.mask = SdpaMask::TOP_LEFT_CAUSAL;
    const auto dense = cdnaGraphMatch(topLeft);
    ASSERT_TRUE(dense.has_value());
    EXPECT_EQ(lengths(*dense), std::pair(Bound(-1), Bound(-1)));
    EXPECT_EQ(tryGetBoundInt(*dense, "flydsl_sdpa_cdna.diag_bottom_right"), 0);
}

TEST(TestFlydslSdpaCdnaKernelMatch, RunsPaddingOnlyOnObjectsThatReadEachBatchsLengths)
{
    // Each object, admitted for the dense graph (the control) and then asked the padded one.
    struct Case
    {
        std::string name;
        KernelDefinition kernel;
        SdpaMask mask;
        bool servesPadding;
    };
    const auto generic = [](int64_t causal, int64_t kvAlign = 0) {
        return makeCdnaKernel("generic", 128, causal, "bf16", 128, kvAlign);
    };
    const auto decode
        = [](int64_t causal) { return makeCdnaKernel("decode", 128, causal, "bf16", 64); };
    const std::vector<Case> cases{
        {"GenericNonCausal", generic(0), SdpaMask::NONE, true},
        {"GenericTopLeft", generic(1), SdpaMask::TOP_LEFT_CAUSAL, true},
        {"GenericBottomRight", generic(1), SdpaMask::BOTTOM_RIGHT_CAUSAL, true},
        // Built without the KV tail mask: a batch's keys past its length would be read.
        {"GenericKvAligned", generic(0, 64), SdpaMask::NONE, false},
        {"DecodeNonCausal", decode(0), SdpaMask::NONE, true},
        {"DecodeBottomRight", decode(1), SdpaMask::BOTTOM_RIGHT_CAUSAL, true},
        // The plain causal compare: a top-left diagonal passes a batch's keys when it has
        // more queries than keys.
        {"DecodeTopLeft", decode(1), SdpaMask::TOP_LEFT_CAUSAL, false},
        // The dual-wave kernel reads no per-batch lengths.
        {"DualwaveNonCausal", makeCdnaKernel("dualwave", 128, 0), SdpaMask::NONE, false},
        {"DualwaveBottomRight",
         makeCdnaKernel("dualwave", 128, 1),
         SdpaMask::BOTTOM_RIGHT_CAUSAL,
         false},
    };
    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.name);
        auto dense = cdnaSpec();
        dense.batch = 2;
        dense.mask = c.mask;
        EXPECT_TRUE(cdnaAdmits(dense, c.kernel));
        auto padded = cdnaPaddedSpec();
        padded.mask = c.mask;
        EXPECT_EQ(cdnaAdmits(padded, c.kernel), c.servesPadding);
    }
}

TEST(TestFlydslSdpaCdnaKernelMatch, RunsPackedSequencesOnlyOnTheGenericPrefillObjects)
{
    // The generic prefill objects read the row-offset tables (vendored-kernel modification
    // 34); the dual-wave and the split, packed-GQA decode objects index every operand by
    // batch; an object built without the KV tail mask would read a sequence's keys past its
    // length. Each admitted for the dense graph (the control), then asked the packed one.
    struct Case
    {
        std::string name;
        KernelDefinition kernel;
        SdpaMask mask;
        bool servesPacked;
    };
    const std::vector<Case> cases{
        {"GenericNonCausal", makeCdnaKernel("generic", 128, 0, "bf16", 128), SdpaMask::NONE, true},
        {"GenericBottomRight",
         makeCdnaKernel("generic", 128, 1, "bf16", 128),
         SdpaMask::BOTTOM_RIGHT_CAUSAL,
         true},
        {"GenericKvAligned",
         makeCdnaKernel("generic", 128, 0, "bf16", 128, 64),
         SdpaMask::NONE,
         false},
        {"DecodeNonCausal", makeCdnaKernel("decode", 128, 0, "bf16", 64), SdpaMask::NONE, false},
        {"DecodeBottomRight",
         makeCdnaKernel("decode", 128, 1, "bf16", 64),
         SdpaMask::BOTTOM_RIGHT_CAUSAL,
         false},
        {"DualwaveNonCausal", makeCdnaKernel("dualwave", 128, 0), SdpaMask::NONE, false},
    };
    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.name);
        auto dense = cdnaSpec();
        dense.batch = 2;
        dense.mask = c.mask;
        EXPECT_TRUE(cdnaAdmits(dense, c.kernel));
        auto packed = cdnaPackedSpec();
        packed.stats = SdpaStats::NONE;
        packed.mask = c.mask;
        EXPECT_EQ(cdnaAdmits(packed, c.kernel), c.servesPacked);
        // With per-batch lengths in the slots (P3), the same objects.
        setPadding(packed, {200, 100}, {150, 256});
        EXPECT_EQ(cdnaAdmits(packed, c.kernel), c.servesPacked);
    }
}

TEST(TestFlydslSdpaCdnaKernelMatch, RanksAPaddedGraphOnTheMaskedDefaultTier)
{
    // A key length the KV tile divides: dense, the object built without the tail mask ranks
    // first; padded, it declines and the masked object of the same tier serves.
    auto aligned = makeCdnaKernel("generic", 128, 0, "bf16", 128, 64);
    aligned.priority = 110;
    auto masked = makeCdnaKernel("generic", 128, 0, "bf16", 128, 0);
    masked.priority = 100;
    const auto best = [&](const SdpaGraphSpec& spec) {
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
        const auto bound = matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
        std::optional<int64_t> top;
        for(const auto* kernel : {&aligned, &masked})
        {
            if(bound.has_value() && matchesCdnaKernel(fixture.context(), *kernel, *bound)
               && (!top.has_value() || kernel->priority > *top))
            {
                top = static_cast<int64_t>(kernel->priority);
            }
        }
        return top;
    };
    auto dense = cdnaSpec();
    dense.batch = 2;
    EXPECT_EQ(best(dense), 110);
    EXPECT_EQ(best(cdnaPaddedSpec()), 100);
}

// ---------------------------------------------------------------------------
// Dropout (vendored-kernel modification 33): the graph matcher binds a probability with its
// seed and offset; only the generic dropout objects apply it
// ---------------------------------------------------------------------------

/// The model shape dropping with probability 0.1, from a device INT64 seed and offset.
SdpaGraphSpec cdnaDroppingSpec()
{
    auto spec = cdnaSpec();
    setDropout(spec, 0.1F);
    return spec;
}

/// A generic prefill object built with dropout.
KernelDefinition makeCdnaDropoutKernel(int64_t headDim = 128, int64_t causal = 0)
{
    auto kernel = makeCdnaKernel("generic", headDim, causal, "bf16", 128);
    kernel.metadata[CDNA_HAS_DROPOUT_FIELD] = int64_t{1};
    return kernel;
}

std::vector<CdnaCase> cdnaDropoutAccepts()
{
    const auto with = [](const std::function<void(SdpaGraphSpec&)>& edit) {
        auto spec = cdnaDroppingSpec();
        edit(spec);
        return spec;
    };
    return {
        {"DeviceInt64SeedAndOffset", cdnaDroppingSpec()},
        {"SeedAndOffsetPassedByValue", with([](SdpaGraphSpec& s) {
             s.passByValueTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID);
             s.passByValueTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_UID);
         })},
        {"Int32SeedAndOffsetPassedByValue", with([](SdpaGraphSpec& s) {
             s.passByValueTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID);
             s.passByValueTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_UID);
             s.dropoutSeedDataType = DataType::INT32;
             s.dropoutOffsetDataType = DataType::INT32;
         })},
        {"ProbabilityJustBelowOne", with([](SdpaGraphSpec& s) { s.dropoutProbability = 0.99F; })},
        {"BottomRightCausal",
         with([](SdpaGraphSpec& s) { s.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL; })},
        {"SlidingWindow", with([](SdpaGraphSpec& s) { s.mask = SdpaMask::SLIDING_WINDOW; })},
        {"WithTheLse", with([](SdpaGraphSpec& s) { s.stats = SdpaStats::LSE; })},
        // Under padding (D6): each batch's mask at the graph's declared extents.
        {"UnderPadding", with([](SdpaGraphSpec& s) {
             s.batch = 2;
             setPadding(s, {256, 3}, {256, 40});
         })},
        {"UnderPaddingWithoutThePaddingMask", with([](SdpaGraphSpec& s) {
             s.batch = 2;
             setPadding(s, {256, 3}, {256, 40});
             s.paddingMask = false;
         })},
        // With a bias (the dropout objects' bias builds), in the query's dtype or f32 (the
        // latter finds no dropout object at kernel match).
        {"WithABias", with([](SdpaGraphSpec& s) {
             s.withBias = true;
             s.biasDataType = s.dataType;
         })},
        {"WithAnF32Bias", with([](SdpaGraphSpec& s) {
             s.withBias = true;
             s.biasDataType = DataType::FLOAT;
         })},
        // And under padding, where the bias builds read the bias as the vectorized objects
        // do (vendored-kernel modification 37), at any key length per batch.
        {"WithABiasUnderPadding", with([](SdpaGraphSpec& s) {
             s.batch = 2;
             setPadding(s, {256, 3}, {256, 41});
             s.withBias = true;
             s.biasDataType = s.dataType;
         })},
    };
}

INSTANTIATE_TEST_SUITE_P(Dropout,
                         TestFlydslSdpaCdnaGraphAccepts,
                         ::testing::ValuesIn(cdnaDropoutAccepts()),
                         [](const auto& info) { return info.param.name; });

/// Each gate on the dropout fields (FlydslSdpaDropout.hpp, and the CDNA matcher's
/// combinations), applied to a dropping graph the CDNA matcher accepts.
std::vector<SdpaFieldDecline> cdnaDropoutDeclines()
{
    using Builder = hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesBuilder;
    std::vector<SdpaFieldDecline> cases;
    const auto add = [&cases](std::string name, std::function<void(SdpaGraphSpec&)> edit) {
        cases.push_back({std::move(name),
                         [edit = std::move(edit)](SdpaGraphSpec& s) {
                             s = cdnaDroppingSpec();
                             edit(s);
                         },
                         [](SdpaGraphSpec& s) { s = cdnaDroppingSpec(); }});
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
    add("VirtualSeed",
        [](SdpaGraphSpec& s) { s.virtualTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID); });
    add("VirtualOffset",
        [](SdpaGraphSpec& s) { s.virtualTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_UID); });
    add("RaggedSeed",
        [](SdpaGraphSpec& s) { s.raggedTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID); });
    add("SeedAlignedTo4",
        [](SdpaGraphSpec& s) { s.alignment[FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID] = 4; });
    add("OffsetAlignedTo4",
        [](SdpaGraphSpec& s) { s.alignment[FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_UID] = 4; });
    // A bad seed declines even where nothing is drawn: the matcher gates what it reads.
    add("Int32DeviceSeedAtProbabilityZero", [](SdpaGraphSpec& s) {
        s.dropoutProbability = 0.0F;
        s.dropoutSeedDataType = DataType::INT32;
    });
    // The dropout objects add no sink yet.
    add("WithASink", [](SdpaGraphSpec& s) { s.withSink = true; });
    add("WithABiasAndASink", [](SdpaGraphSpec& s) {
        s.withBias = true;
        s.biasDataType = s.dataType;
        s.withSink = true;
    });
    return cases;
}

INSTANTIATE_TEST_SUITE_P(Dropout,
                         TestFlydslSdpaCdnaFieldDeclines,
                         ::testing::ValuesIn(cdnaDropoutDeclines()),
                         [](const auto& info) { return info.param.name; });

TEST(TestFlydslSdpaCdnaBinding, BindsTheDropoutProbabilitySeedAndOffset)
{
    const auto bound = cdnaGraphMatch(cdnaDroppingSpec());
    ASSERT_TRUE(bound.has_value());
    const auto bits = tryGetBoundInt(*bound, "flydsl_sdpa_cdna.dropout_p");
    ASSERT_TRUE(bits.has_value());
    float probability = 0.0F;
    const auto bits32 = static_cast<int32_t>(*bits);
    std::memcpy(&probability, &bits32, sizeof(probability));
    EXPECT_EQ(probability, 0.1F);
    EXPECT_EQ(tryGetBoundInt(*bound, "flydsl_sdpa_cdna.dropout_seed.uid"),
              FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, "flydsl_sdpa_cdna.dropout_offset.uid"),
              FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_UID);

    const auto dense = cdnaGraphMatch(cdnaSpec());
    ASSERT_TRUE(dense.has_value());
    EXPECT_EQ(tryGetBoundInt(*dense, "flydsl_sdpa_cdna.dropout_p"), 0);
    EXPECT_EQ(tryGetBoundInt(*dense, "flydsl_sdpa_cdna.dropout_seed.uid"), -1);
    EXPECT_EQ(tryGetBoundInt(*dense, "flydsl_sdpa_cdna.dropout_offset.uid"), -1);
}

// RNG dump (FlydslSdpaRngDump.hpp): the shared decline table, the binding, and the writer
// beside every staged dropout object.
INSTANTIATE_TEST_SUITE_P(RngDump,
                         TestFlydslSdpaCdnaFieldDeclines,
                         ::testing::ValuesIn(sdpaRngDumpDeclines()),
                         [](const auto& info) { return info.param.name; });

TEST(TestFlydslSdpaCdnaBinding, BindsTheRngDumpOrNone)
{
    auto withDump = cdnaDroppingSpec();
    withDump.withRngDump = true;
    const auto bound = cdnaGraphMatch(withDump);
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, "flydsl_sdpa_cdna.rng_dump.uid"),
              FLYDSL_SDPA_RDNA_RNG_DUMP_UID);
    for(const auto& spec : {cdnaDroppingSpec(), cdnaSpec()})
    {
        const auto none = cdnaGraphMatch(spec);
        ASSERT_TRUE(none.has_value());
        EXPECT_EQ(tryGetBoundInt(*none, "flydsl_sdpa_cdna.rng_dump.uid"), -1);
    }
    // Under padding the CDNA objects drop too, so the dump is served there as well.
    auto padded = withDump;
    setPadding(padded, {128, 3}, {100, 128});
    const auto paddedBound = cdnaGraphMatch(padded);
    ASSERT_TRUE(paddedBound.has_value());
    EXPECT_EQ(tryGetBoundInt(*paddedBound, "flydsl_sdpa_cdna.rng_dump.uid"),
              FLYDSL_SDPA_RDNA_RNG_DUMP_UID);
}

/// Every staged CDNA object built with dropout.
std::vector<KernelDefinition> stagedCdnaDropoutKernels()
{
    std::vector<KernelDefinition> kernels;
    const auto* staged = findLoadedSet(FLYDSL_SDPA_CDNA.engineName);
    for(const auto& pack : staged == nullptr ? decltype(staged->packs){} : staged->packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto* dropout = tryGetMetadataField<int64_t>(kernel.metadata, "has_dropout");
            if(dropout != nullptr && *dropout == 1)
            {
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
    }
    return kernels;
}

// Every staged dropout object's shard ships the writer with the argument list the native
// launches, and an object whose shard does not (its directory moved) declines a dump while
// still serving the dropping graph.
TEST(TestFlydslSdpaCdnaKernelMatch, RunsADumpOnlyBesideTheWriter)
{
    auto withDump = cdnaDroppingSpec();
    withDump.withRngDump = true;
    const GraphFixture dumping(buildFlydslSdpaRdnaGraph(withDump));
    const GraphFixture dropping(buildFlydslSdpaRdnaGraph(cdnaDroppingSpec()));
    const auto dumpBound = matchesGraph(FLYDSL_SDPA_CDNA, dumping.context());
    const auto dropBound = matchesGraph(FLYDSL_SDPA_CDNA, dropping.context());
    ASSERT_TRUE(dumpBound.has_value() && dropBound.has_value());
    size_t writers = 0;
    size_t matched = 0;
    for(const auto& kernel : stagedCdnaDropoutKernels())
    {
        ASSERT_TRUE(flydsl_sdpa::hasDropoutMaskWriter(kernel)) << kernel.name;
        for(const auto& arch : kernel.arch)
        {
            const auto mask = flydsl_sdpa::dropoutMaskKernel(kernel, arch);
            EXPECT_EQ(mask.source.kind, KernelSourceKind::KPACK) << arch;
            EXPECT_EQ(mask.getIntMetadata("block_threads"), 256) << arch;
            const auto& expected = flydsl_sdpa::dropoutMaskSignature();
            ASSERT_EQ(mask.source.signature.size(), expected.size()) << arch;
            for(size_t i = 0; i < expected.size(); ++i)
            {
                EXPECT_EQ(mask.source.signature[i].kind, expected[i].kind) << arch << " " << i;
                EXPECT_EQ(mask.source.signature[i].size, expected[i].size) << arch << " " << i;
                EXPECT_EQ(mask.source.signature[i].offset, expected[i].offset) << arch << " " << i;
            }
            ++writers;
        }
        if(!matchesKernel(FLYDSL_SDPA_CDNA, dropping.context(), kernel, *dropBound))
        {
            continue;
        }
        EXPECT_TRUE(matchesKernel(FLYDSL_SDPA_CDNA, dumping.context(), kernel, *dumpBound))
            << kernel.name;
        auto moved = kernel;
        moved.originDirectory = std::filesystem::path("/nonexistent") / "FlyDSL" / "sdpa_cdna"
                                / kernel.originDirectory.filename();
        EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_CDNA, dumping.context(), moved, *dumpBound))
            << kernel.name;
        EXPECT_TRUE(matchesKernel(FLYDSL_SDPA_CDNA, dropping.context(), moved, *dropBound))
            << kernel.name;
        ++matched;
    }
    if(!flydslShippedArches("sdpa_cdna").empty())
    {
        EXPECT_GT(writers, 0U);
        EXPECT_GT(matched, 0U) << "no staged dropout object serves the dropping graph";
    }
}

TEST(TestFlydslSdpaCdnaKernelMatch, RunsADroppingGraphOnlyOnTheDropoutObjects)
{
    const auto dropping = cdnaDroppingSpec();
    auto noDrop = cdnaDroppingSpec();
    noDrop.dropoutProbability = 0.0F;
    // The dropout object: a dropping graph, never a plain one (probability 0 included).
    EXPECT_TRUE(cdnaAdmits(dropping, makeCdnaDropoutKernel()));
    EXPECT_FALSE(cdnaAdmits(cdnaSpec(), makeCdnaDropoutKernel()));
    EXPECT_FALSE(cdnaAdmits(noDrop, makeCdnaDropoutKernel()));
    // The plain objects: the undropped result at probability 0, never a drop.
    for(const auto& kernel : {makeCdnaKernel("generic", 128, 0, "bf16", 128),
                              makeCdnaKernel("dualwave", 128, 0),
                              makeCdnaKernel("decode", 128, 0, "bf16", 64)})
    {
        SCOPED_TRACE(kernel.name);
        EXPECT_TRUE(cdnaAdmits(noDrop, kernel));
        EXPECT_FALSE(cdnaAdmits(dropping, kernel));
    }
    // An object whose metadata says nothing about dropout serves nothing.
    auto unknown = makeCdnaKernel("generic", 128, 0, "bf16", 128);
    unknown.metadata.erase(CDNA_HAS_DROPOUT_FIELD);
    EXPECT_FALSE(cdnaAdmits(cdnaSpec(), unknown));
}

TEST(TestFlydslSdpaCdnaKernelMatch, RunsAKeyAlignedDroppingGraphOnTheAlignedDropoutObjects)
{
    // The non-causal dropout objects' builds without the KV tail mask (gfx942, gfx950): a
    // dropping graph whose key length the KV tile divides, never another one, never a padded
    // one, and never a graph that does not drop.
    auto aligned = makeCdnaDropoutKernel();
    aligned.metadata[CDNA_KV_ALIGN_FIELD] = int64_t{64};
    const auto masked = makeCdnaDropoutKernel();
    const auto dropping = cdnaDroppingSpec();
    ASSERT_EQ(dropping.seqLenKv % 64, 0);
    EXPECT_TRUE(cdnaAdmits(dropping, aligned));
    EXPECT_TRUE(cdnaAdmits(dropping, masked));
    auto ragged = dropping;
    ragged.seqLenQ = 333;
    ragged.seqLenKv = 333;
    EXPECT_FALSE(cdnaAdmits(ragged, aligned)) << "the object skips the KV tail mask";
    EXPECT_TRUE(cdnaAdmits(ragged, masked));
    auto padded = cdnaPaddedSpec();
    setDropout(padded, 0.1F);
    EXPECT_FALSE(cdnaAdmits(padded, aligned));
    EXPECT_TRUE(cdnaAdmits(padded, masked));
    auto noDrop = dropping;
    noDrop.dropoutProbability = 0.0F;
    EXPECT_FALSE(cdnaAdmits(noDrop, aligned));
}

TEST(TestFlydslSdpaCdnaKernelMatch, RunsADroppingGraphWithABiasOnTheDropoutBiasObjects)
{
    // Dropout with a bias (COVERAGE.md section 7.9's second wave): the dropout objects' bias
    // builds, which read the bias the vectorized objects read. Neither the plain dropout
    // object nor a bias object without dropout serves it; an f32 bias, which only the strided
    // objects read, has no dropout object.
    auto dropping = cdnaDroppingSpec();
    dropping.withBias = true;
    dropping.biasDataType = dropping.dataType;
    ASSERT_TRUE(cdnaGraphMatch(dropping).has_value());
    auto dropoutBias = makeCdnaDropoutKernel();
    dropoutBias.metadata[CDNA_HAS_BIAS_FIELD] = int64_t{1};
    EXPECT_TRUE(cdnaAdmits(dropping, dropoutBias));
    EXPECT_FALSE(cdnaAdmits(dropping, makeCdnaDropoutKernel()));
    EXPECT_FALSE(cdnaAdmits(dropping, makeCdnaKernel("generic", 128, 0, "bf16", 128, 0, 1, 1)));
    EXPECT_FALSE(cdnaAdmits(cdnaDroppingSpec(), dropoutBias));
    auto f32Bias = dropping;
    f32Bias.biasDataType = DataType::FLOAT;
    EXPECT_FALSE(cdnaAdmits(f32Bias, dropoutBias));
    // Under padding too, at a key length no multiple of the 4-key runs (vendored-kernel
    // modification 37), and not on the plain dropout object.
    auto padded = dropping;
    setPadding(padded, {256, 100}, {256, 77});
    padded.batch = 2;
    ASSERT_TRUE(cdnaGraphMatch(padded).has_value());
    EXPECT_TRUE(cdnaAdmits(padded, dropoutBias));
    EXPECT_FALSE(cdnaAdmits(padded, makeCdnaDropoutKernel()));
}

TEST(TestFlydslSdpaCdnaKernelMatch, RunsAPaddedGraphsBiasOnTheVectorizedObjects)
{
    // Under padding a batch's key length need not be a multiple of the vectorized objects'
    // 4-key runs: their last real row reads the bias up to the launch's key length
    // (vendored-kernel modification 37), so they serve it as they do dense, sink included,
    // and the strided objects still do. A bias the vectorized objects cannot read (f32)
    // stays on the strided ones.
    auto padded = cdnaPaddedSpec();
    padded.withBias = true;
    padded.biasDataType = padded.dataType;
    auto dense = padded;
    dense.seqLensQ.clear();
    dense.seqLensKv.clear();
    dense.paddingMask = false;
    const auto vectorized = makeCdnaKernel("generic", 128, 0, "bf16", 128, 0, 1, 1);
    auto strided = vectorized;
    strided.metadata["bias_kstrided"] = int64_t{1};
    EXPECT_TRUE(cdnaAdmits(dense, vectorized));
    EXPECT_TRUE(cdnaAdmits(padded, vectorized));
    EXPECT_TRUE(cdnaAdmits(padded, strided));
    auto f32 = padded;
    f32.biasDataType = DataType::FLOAT;
    EXPECT_FALSE(cdnaAdmits(f32, vectorized));
    EXPECT_TRUE(cdnaAdmits(f32, strided));
    padded.withSink = true;
    auto vectorizedSink = vectorized;
    vectorizedSink.metadata["has_sink"] = int64_t{1};
    strided.metadata["has_sink"] = int64_t{1};
    EXPECT_TRUE(cdnaAdmits(padded, vectorizedSink));
    EXPECT_TRUE(cdnaAdmits(padded, strided));
}

TEST(TestFlydslSdpaCdnaKernelMatch, RunsAPaddedDroppingGraphOnTheDropoutObjects)
{
    // Dropout under padding (D6): the dropout objects read each batch's lengths too.
    auto padded = cdnaPaddedSpec();
    setDropout(padded, 0.1F);
    for(const auto mask : {SdpaMask::NONE, SdpaMask::BOTTOM_RIGHT_CAUSAL})
    {
        padded.mask = mask;
        EXPECT_TRUE(cdnaAdmits(padded, makeCdnaDropoutKernel(128, mask == SdpaMask::NONE ? 0 : 1)));
        EXPECT_FALSE(cdnaAdmits(
            padded, makeCdnaKernel("decode", 128, mask == SdpaMask::NONE ? 0 : 1, "bf16", 64)));
    }
}

// ---------------------------------------------------------------------------
// GPU: the staged objects against a host reference
// ---------------------------------------------------------------------------

/// The op whose shipped-arch list (HIPKERNELPROVIDER_FLYDSL_ARCHS_<op>) holds @p engine's
/// objects, for reportUnstaged().
std::string_view shippedOpFor(const PackSymbols& engine)
{
    if(&engine == &FLYDSL_SDPA_CDNA_DUALWAVE)
    {
        return "sdpa_cdna_dualwave";
    }
    return &engine == &FLYDSL_SDPA_CDNA_DECODE ? "sdpa_cdna_decode" : "sdpa_cdna";
}

/// Every staged object for the context's arch that the kernel matcher admits for the graph.
/// Each must compute the same answer, so the GPU cases run them all: at an aligned length
/// that is both a masked object and one that skips the KV tail mask.
std::vector<KernelDefinition> stagedCdnaCandidates(const MatchContext& context,
                                                   const BoundTokens& bound)
{
    std::vector<KernelDefinition> candidates;
    const auto& device = context.deviceProperties.gcnArchName;
    for(const auto* engine :
        {&FLYDSL_SDPA_CDNA, &FLYDSL_SDPA_CDNA_DUALWAVE, &FLYDSL_SDPA_CDNA_DECODE})
    {
        // A build for RDNA targets only stages no CDNA set at all, and gfx942 no dual-wave
        // one. A set missing on a device this build ships its op for is a packaging fault,
        // not a family the device lacks.
        const auto* staged = findLoadedSet(engine->engineName);
        if(staged == nullptr)
        {
            if(flydslShipsForThisDevice(shippedOpFor(*engine)))
            {
                ADD_FAILURE() << "this build ships " << shippedOpFor(*engine)
                              << " for this device, but no " << engine->engineName
                              << " set is loaded";
            }
            continue;
        }
        for(const auto& pack : staged->packs)
        {
            for(const auto& kernel : pack.kernels)
            {
                const auto& arches = kernel.arch.empty() ? pack.arch : kernel.arch;
                if(std::find(arches.begin(), arches.end(), device) == arches.end())
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
                if(matchesCdnaKernel(context, definition, bound))
                {
                    candidates.push_back(std::move(definition));
                }
            }
        }
    }
    return candidates;
}

using Bf16 = hipdnn_data_sdk::types::bfloat16;

/// Deterministic k/32 values for |k| <= 32: exact in bf16, so the reference reads exactly
/// what the device does.
template <typename T>
std::vector<T> fillValues(size_t count, uint32_t seed)
{
    std::vector<T> values(count);
    uint32_t state = seed;
    for(auto& value : values)
    {
        state = state * 1664525U + 1013904223U;
        const auto k = static_cast<int32_t>((state >> 16) % 65U) - 32;
        value = T(static_cast<float>(k) / 32.0F);
    }
    return values;
}

/// The LSE is computed in f32 from exact bf16 inputs; its magnitude is about ln(Skv). Tight
/// enough that an error the size vendored-kernel modification 21 fixed (1.2e-3) sits near
/// the bound.
constexpr double LSE_TOLERANCE = 2e-3;

/// The harness's host copies of Q, K, V and O and their logical (B, H, S, D) strides, K and V
/// starting @p kStart and @p vStart elements in (inside one packed buffer).
template <typename T>
struct CdnaHostOperands
{
    const std::vector<T>& q;
    const std::vector<T>& k;
    const std::vector<T>& v;
    const std::vector<T>& o;
    const std::vector<int64_t>& qStrides;
    const std::vector<int64_t>& kvStrides;
    const std::vector<int64_t>& vStrides;
    const std::vector<int64_t>& oStrides;
    int64_t kStart = 0;
    int64_t vStart = 0;

    static double at(const std::vector<T>& t,
                     const std::vector<int64_t>& strides,
                     int64_t start,
                     int64_t b,
                     int64_t h,
                     int64_t s,
                     int64_t d)
    {
        const int64_t index
            = start + b * strides[0] + h * strides[1] + s * strides[2] + d * strides[3];
        return static_cast<double>(static_cast<float>(t[static_cast<size_t>(index)]));
    }
};

/// A CDNA launch's RNG dump (FlydslSdpaRngDumpTestSupport.hpp): the oracle's mask bit for bit at
/// every element, padded rows and keys past a batch's length included, with hipDNN's keep-rate
/// check; and O as hipDNN's CPU reference computes it with the dump as its explicit mask. With
/// a bias the dump alone: that reference adds none, and the harness checks O against the
/// oracle's mask and the bias.
template <typename T>
void expectCdnaRngDump(const RngDumpReferenceProblem& reference,
                       const std::vector<float>& dumped,
                       const SdpaGraphSpec& spec,
                       const CdnaHostOperands<T>& host,
                       double tolerance)
{
    using Host = CdnaHostOperands<T>;
    expectRngDumpIsTheOracleMask(
        dumped, reference.extents, reference.p, spec.dropoutSeed, spec.dropoutOffset);
    if(spec.withBias)
    {
        return;
    }
    expectOutputIsTheReferenceUnderTheDump(
        reference,
        dumped,
        [&host](int64_t b, int64_t h, int64_t s, int64_t e) {
            return Host::at(host.q, host.qStrides, 0, b, h, s, e);
        },
        [&host](int64_t b, int64_t h, int64_t j, int64_t e) {
            return Host::at(host.k, host.kvStrides, host.kStart, b, h, j, e);
        },
        [&host](int64_t b, int64_t h, int64_t j, int64_t e) {
            return Host::at(host.v, host.vStrides, host.vStart, b, h, j, e);
        },
        [&host](int64_t b, int64_t h, int64_t s, int64_t e) {
            return Host::at(host.o, host.oStrides, 0, b, h, s, e);
        },
        tolerance);
}

/// NaN over the harness's bias values at @p biasOffset (b, h, s, j), within @p extents [B, H, Sq,
/// Skv]: on every key of every third row from the second (biasNanRows), a row whose O and LSE are
/// NaN; and under padding at and past each batch's key length (biasNanPastKeys), which the
/// kernels must mask whatever the bias holds there; that needs a bias of its own per batch
/// (@p biasDims as the graph declares it).
template <typename BiasOffset>
void writeCdnaBiasNans(const SdpaGraphSpec& spec,
                       const std::vector<int64_t>& biasDims,
                       const std::array<int64_t, 4>& extents,
                       BiasOffset biasOffset,
                       std::vector<float>& biasValues)
{
    const auto [b, h, s, skv] = extents;
    ASSERT_TRUE(!spec.biasNanPastKeys
                || (!spec.seqLensKv.empty() && biasDims.size() == 4 && biasDims[0] == b));
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for(int64_t bi = 0; bi < b; ++bi)
    {
        const int64_t firstPast
            = spec.biasNanPastKeys
                  ? std::clamp<int64_t>(spec.seqLensKv[static_cast<size_t>(bi)], 0, skv)
                  : skv;
        for(int64_t hi = 0; hi < h; ++hi)
        {
            for(int64_t si = 0; si < s; ++si)
            {
                const bool nanRow = spec.biasNanRows && si % 3 == 1;
                for(int64_t j = nanRow ? 0 : firstPast; j < skv; ++j)
                {
                    biasValues[biasOffset(bi, hi, si, j)] = nan;
                }
            }
        }
    }
}

/// The sink logits a CDNA case binds, one per query head (none without a sink): random values
/// in the query's dtype, held in @p sinkValues for a query-dtype sink; for a FLOAT sink (vendored-
/// kernel modification 36) the same plus low bits no 16-bit dtype holds, so a sink read as 16-bit
/// is caught; or every one the spec's value, or -inf.
template <typename T>
std::vector<float> cdnaSinkLogits(const SdpaGraphSpec& spec, std::vector<T>& sinkValues)
{
    const bool floatSink = spec.sinkDataType == DataType::FLOAT;
    sinkValues
        = spec.withSink ? fillValues<T>(static_cast<size_t>(spec.heads), 5U) : std::vector<T>{};
    std::vector<float> logits(sinkValues.size());
    for(size_t hi = 0; hi < sinkValues.size(); ++hi)
    {
        logits[hi] = static_cast<float>(sinkValues[hi])
                     + (floatSink ? static_cast<float>((hi % 7) + 1) * 1.0e-3F : 0.0F);
    }
    if(spec.sinkValue)
    {
        std::fill(sinkValues.begin(), sinkValues.end(), T(*spec.sinkValue));
        std::fill(logits.begin(),
                  logits.end(),
                  floatSink ? *spec.sinkValue : static_cast<float>(T(*spec.sinkValue)));
    }
    if(spec.sinkNegativeInfinity)
    {
        std::fill(sinkValues.begin(), sinkValues.end(), T(-std::numeric_limits<float>::infinity()));
        std::fill(logits.begin(), logits.end(), -std::numeric_limits<float>::infinity());
    }
    return logits;
}

/// A CDNA sink case's O and LSE against hipDNN's CPU reference (FLOAT [1, Hq, 1, 1], SEM-022;
/// FlydslSdpaCpuReferenceTestSupport.hpp), with the bias, band and padding @p problem holds.
template <typename T, typename BiasAt>
void expectCdnaSinkIsTheCpuReference(const SdpaCpuReferenceProblem& problem,
                                     const CdnaHostOperands<T>& host,
                                     BiasAt biasAt,
                                     const std::vector<float>& stats,
                                     bool statsSequenceMajor,
                                     double tolerance,
                                     double lseTolerance)
{
    using Host = CdnaHostOperands<T>;
    expectSdpaIsTheCpuReference(
        problem,
        [&host](int64_t b, int64_t h, int64_t s, int64_t e) {
            return Host::at(host.q, host.qStrides, 0, b, h, s, e);
        },
        [&host](int64_t b, int64_t h, int64_t j, int64_t e) {
            return Host::at(host.k, host.kvStrides, host.kStart, b, h, j, e);
        },
        [&host](int64_t b, int64_t h, int64_t j, int64_t e) {
            return Host::at(host.v, host.vStrides, host.vStart, b, h, j, e);
        },
        biasAt,
        [&host](int64_t b, int64_t h, int64_t s, int64_t e) {
            return Host::at(host.o, host.oStrides, 0, b, h, s, e);
        },
        [&](int64_t b, int64_t h, int64_t s) {
            const auto index = statsSequenceMajor ? (b * problem.seqLenQ + s) * problem.heads + h
                                                  : (b * problem.heads + h) * problem.seqLenQ + s;
            return static_cast<double>(stats[static_cast<size_t>(index)]);
        },
        tolerance,
        lseTolerance);
}

/// Runs each of @p kernels for @p spec through prepare() and launch() and checks every
/// output element against a double-precision reference, in the memory layout @p spec names:
/// BSHD, BHSD, or Q, K and V as views into one packed buffer. The inputs and the reference
/// are computed once and shared by every kernel: the reference is the costly part, and the
/// kernels a shape admits are often a dozen tile variants.
template <typename T>
void expectCandidatesMatchReference(const MatchContext& context,
                                    const BoundTokens& bound,
                                    const std::vector<KernelDefinition>& kernels,
                                    const SdpaGraphSpec& spec,
                                    double absoluteTolerance)
{
    // The deprecated booleans resolved as the reference does: either one alone is causal;
    // causal_mask keeps the stored alignment, and causal_mask_bottom_right is bottom-right.
    using hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment;
    const bool causal = spec.mask != SdpaMask::NONE || spec.deprecatedCausalTopLeft
                        || spec.deprecatedCausalBottomRight;
    const bool alignBottomRight
        = spec.deprecatedCausalBottomRight
          || spec.diagonalAlignment.value_or(spec.mask == SdpaMask::BOTTOM_RIGHT_CAUSAL
                                                 ? DiagonalAlignment::BOTTOM_RIGHT
                                                 : DiagonalAlignment::TOP_LEFT)
                 == DiagonalAlignment::BOTTOM_RIGHT;
    const int64_t b = spec.batch;
    const int64_t s = spec.seqLenQ;
    const int64_t skv = spec.seqLenKv;
    const int64_t h = spec.heads;
    const int64_t hk = spec.kvHeads;
    const int64_t d = spec.headDim;
    // V's and O's head dim: Q/K's, or narrower (MLA).
    const int64_t dv = spec.valueHeadDim.value_or(d);
    // Logical (B, H, S, D) strides of each operand, as the graph states them. In a packed
    // buffer K and V start one and two heads' worth of a token's row after Q.
    const auto qStrides
        = spec.packedQkv ? sdpaPackedQkvStrides(spec) : sdpaStrides(spec.layout, h, s, d);
    const auto kvStrides
        = spec.packedQkv ? sdpaPackedQkvStrides(spec) : sdpaStrides(spec.layout, hk, skv, d);
    const auto vStrides = spec.packedQkv
                              ? sdpaPackedQkvStrides(spec)
                              : sdpaStrides(spec.valueLayout.value_or(spec.layout), hk, skv, dv);
    const auto oStrides = sdpaStrides(spec.layout, h, s, dv);
    const int64_t kStart = spec.packedQkv ? h * d : 0;
    const int64_t vStart = spec.packedQkv ? 2 * h * d : 0;
    const auto qCount = static_cast<size_t>(spec.packedQkv ? b * s * 3 * h * d : b * s * h * d);
    const auto kvCount = static_cast<size_t>(b * skv * hk * d);
    const auto vCount = static_cast<size_t>(b * skv * hk * dv);
    const auto oCount = static_cast<size_t>(b * s * h * dv);
    auto q = fillValues<T>(qCount, 1U);
    auto k = spec.packedQkv ? q : fillValues<T>(kvCount, 2U);
    auto v = spec.packedQkv ? q : fillValues<T>(vCount, 3U);
    if(spec.valueScale.has_value() && !spec.packedQkv)
    {
        for(auto& value : v)
        {
            value = T(static_cast<float>(value) * *spec.valueScale);
        }
    }
    // The dropout mask exposed (separate tensors only): Q zero, so every visible key has the
    // same probability, and V the identity on its first head_dim keys, so O[i][e] is nonzero
    // exactly when key e of row i is kept.
    if(spec.exposeDropoutMask)
    {
        ASSERT_FALSE(spec.packedQkv);
        std::fill(q.begin(), q.end(), T(0.0F));
        const auto vExposeStrides
            = sdpaStrides(spec.valueLayout.value_or(spec.layout), hk, skv, dv);
        for(int64_t bi = 0; bi < b; ++bi)
        {
            for(int64_t hi = 0; hi < hk; ++hi)
            {
                for(int64_t j = 0; j < skv; ++j)
                {
                    for(int64_t e = 0; e < dv; ++e)
                    {
                        v[static_cast<size_t>(bi * vExposeStrides[0] + hi * vExposeStrides[1]
                                              + j * vExposeStrides[2] + e * vExposeStrides[3])]
                            = T(j == e ? 1.0F : 0.0F);
                    }
                }
            }
        }
    }
    // NaN in Q's first column at one query position (every batch and head), or in K's at one
    // key (every batch and KV head): every row that sees it must come out NaN, O and LSE, as
    // PyTorch's does. Separate Q and K buffers only.
    if(!spec.packedQkv && (spec.queryNanPosition || spec.keyNanPosition))
    {
        const T nan(std::numeric_limits<float>::quiet_NaN());
        const auto qNanStrides = sdpaStrides(spec.layout, h, s, d);
        const auto kNanStrides = sdpaStrides(spec.layout, hk, skv, d);
        for(int64_t bi = 0; bi < b; ++bi)
        {
            for(int64_t hi = 0; spec.queryNanPosition && hi < h; ++hi)
            {
                q[static_cast<size_t>(bi * qNanStrides[0] + hi * qNanStrides[1]
                                      + *spec.queryNanPosition * qNanStrides[2])]
                    = nan;
            }
            for(int64_t hi = 0; spec.keyNanPosition && hi < hk; ++hi)
            {
                k[static_cast<size_t>(bi * kNanStrides[0] + hi * kNanStrides[1]
                                      + *spec.keyNanPosition * kNanStrides[2])]
                    = nan;
            }
        }
    }
    std::vector<T> o(oCount, T(0.0F));
    // Freed on every exit, including a failed ASSERT part-way through a kernel.
    std::vector<void*> allocations;
    struct FreeAll
    {
        std::vector<void*>& pointers;
        FreeAll(const FreeAll&) = delete;
        FreeAll& operator=(const FreeAll&) = delete;
        FreeAll(FreeAll&&) = delete;
        FreeAll& operator=(FreeAll&&) = delete;
        ~FreeAll()
        {
            for(auto* pointer : pointers)
            {
                static_cast<void>(hipFree(pointer));
            }
        }
    } const freeAll{allocations};

    void* qDevice = nullptr;
    void* kDevice = nullptr;
    void* vDevice = nullptr;
    void* oDevice = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&qDevice, qCount * sizeof(T)));
    allocations.push_back(qDevice);
    ASSERT_EQ(hipSuccess, hipMalloc(&oDevice, oCount * sizeof(T)));
    allocations.push_back(oDevice);
    ASSERT_EQ(hipSuccess, hipMemcpy(qDevice, q.data(), qCount * sizeof(T), hipMemcpyHostToDevice));
    if(spec.packedQkv)
    {
        kDevice = static_cast<T*>(qDevice) + kStart;
        vDevice = static_cast<T*>(qDevice) + vStart;
    }
    else
    {
        ASSERT_EQ(hipSuccess, hipMalloc(&kDevice, kvCount * sizeof(T)));
        allocations.push_back(kDevice);
        ASSERT_EQ(hipSuccess, hipMalloc(&vDevice, vCount * sizeof(T)));
        allocations.push_back(vDevice);
        ASSERT_EQ(hipSuccess,
                  hipMemcpy(kDevice, k.data(), kvCount * sizeof(T), hipMemcpyHostToDevice));
        ASSERT_EQ(hipSuccess,
                  hipMemcpy(vDevice, v.data(), vCount * sizeof(T), hipMemcpyHostToDevice));
    }
    // The LSE output, [B, H, Sq] f32, when the graph asks for it.
    const bool withStats = spec.stats == SdpaStats::LSE;
    const auto statsCount = static_cast<size_t>(b * h * s);
    std::vector<float> stats(statsCount, 0.0F);
    void* statsDevice = nullptr;
    if(withStats)
    {
        ASSERT_EQ(hipSuccess, hipMalloc(&statsDevice, statsCount * sizeof(float)));
        allocations.push_back(statsDevice);
    }

    std::vector<hipdnnPluginDeviceBuffer_t> buffers{{FLYDSL_SDPA_RDNA_Q_UID, qDevice},
                                                    {FLYDSL_SDPA_RDNA_K_UID, kDevice},
                                                    {FLYDSL_SDPA_RDNA_V_UID, vDevice},
                                                    {FLYDSL_SDPA_RDNA_O_UID, oDevice}};
    if(withStats)
    {
        buffers.push_back({FLYDSL_SDPA_RDNA_STATS_UID, statsDevice});
    }
    // Packed (THD; vendored-kernel modification 34): each operand's sequences end to end,
    // token-major ([T, H, D], the strides setPacked's graph states), holding the dense
    // tensors' values, and the cumulative token tables Q/O and K/V go through. O and the LSE
    // come back the same way and are unpacked into the dense ones, so the oracle below is
    // unchanged. A guard of S rows past the last sequence must stay unwritten.
    const bool packed = spec.packed();
    std::vector<int32_t> offsetsQ{0};
    std::vector<int32_t> offsetsKv{0};
    for(int64_t bi = 0; packed && bi < b; ++bi)
    {
        offsetsQ.push_back(offsetsQ.back() + spec.packedSeqLensQ[static_cast<size_t>(bi)]);
        offsetsKv.push_back(offsetsKv.back() + spec.packedSeqLensKv[static_cast<size_t>(bi)]);
    }
    const int64_t tokensQ = offsetsQ.back() + s;
    const int64_t tokensKv = offsetsKv.back() + skv;
    const auto packedOCount = static_cast<size_t>(tokensQ * h * dv);
    const auto packedStatsCount = static_cast<size_t>(tokensQ * h);
    void* packedQDevice = nullptr;
    void* packedKDevice = nullptr;
    void* packedVDevice = nullptr;
    void* packedODevice = nullptr;
    void* packedStatsDevice = nullptr;
    void* tableQDevice = nullptr;
    void* tableKvDevice = nullptr;
    if(packed)
    {
        ASSERT_FALSE(spec.packedQkv);
        // Sequence bi's token si, head hi, element e of a dense operand, packed at its token.
        const auto pack = [](const std::vector<T>& dense,
                             const std::vector<int64_t>& strides,
                             const std::vector<int32_t>& offsets,
                             const std::vector<int32_t>& lengths,
                             int64_t tokens,
                             int64_t heads,
                             int64_t width) {
            std::vector<T> out(static_cast<size_t>(tokens * heads * width), T(0.0F));
            for(size_t bi = 0; bi < lengths.size(); ++bi)
            {
                for(int64_t si = 0; si < lengths[bi]; ++si)
                {
                    for(int64_t hi = 0; hi < heads; ++hi)
                    {
                        for(int64_t e = 0; e < width; ++e)
                        {
                            out[static_cast<size_t>(((offsets[bi] + si) * heads + hi) * width + e)]
                                = dense[static_cast<size_t>(static_cast<int64_t>(bi) * strides[0]
                                                            + hi * strides[1] + si * strides[2]
                                                            + e * strides[3])];
                        }
                    }
                }
            }
            return out;
        };
        const auto upload = [&allocations](void** device, const auto& host) {
            using Element = typename std::decay_t<decltype(host)>::value_type;
            ASSERT_EQ(hipSuccess, hipMalloc(device, host.size() * sizeof(Element)));
            allocations.push_back(*device);
            ASSERT_EQ(
                hipSuccess,
                hipMemcpy(
                    *device, host.data(), host.size() * sizeof(Element), hipMemcpyHostToDevice));
        };
        upload(&packedQDevice, pack(q, qStrides, offsetsQ, spec.packedSeqLensQ, tokensQ, h, d));
        upload(&packedKDevice,
               pack(k, kvStrides, offsetsKv, spec.packedSeqLensKv, tokensKv, hk, d));
        upload(&packedVDevice,
               pack(v, vStrides, offsetsKv, spec.packedSeqLensKv, tokensKv, hk, dv));
        upload(&tableQDevice, offsetsQ);
        upload(&tableKvDevice, offsetsKv);
        // Poisoned per kernel, below.
        ASSERT_EQ(hipSuccess, hipMalloc(&packedODevice, packedOCount * sizeof(T)));
        allocations.push_back(packedODevice);
        ASSERT_EQ(hipSuccess, hipMalloc(&packedStatsDevice, packedStatsCount * sizeof(float)));
        allocations.push_back(packedStatsDevice);
        for(auto& buffer : buffers)
        {
            if(buffer.uid == FLYDSL_SDPA_RDNA_Q_UID)
            {
                buffer.ptr = packedQDevice;
            }
            else if(buffer.uid == FLYDSL_SDPA_RDNA_K_UID)
            {
                buffer.ptr = packedKDevice;
            }
            else if(buffer.uid == FLYDSL_SDPA_RDNA_V_UID)
            {
                buffer.ptr = packedVDevice;
            }
            else if(buffer.uid == FLYDSL_SDPA_RDNA_O_UID)
            {
                buffer.ptr = packedODevice;
            }
            else if(buffer.uid == FLYDSL_SDPA_RDNA_STATS_UID)
            {
                buffer.ptr = packedStatsDevice;
            }
        }
        buffers.push_back({FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID, tableQDevice});
        buffers.push_back({FLYDSL_SDPA_RDNA_RAGGED_OFFSET_KV_UID, tableKvDevice});
    }
    // The bias, in the query's dtype, right-aligned to [B, H, Sq, Skv] as the graph states it.
    const auto biasLayout = sdpaBiasLayout(spec);
    const auto& biasDims = biasLayout.first;
    const auto& biasStrideList = biasLayout.second;
    size_t biasCount = 1;
    for(size_t axis = 0; axis < biasDims.size(); ++axis)
    {
        biasCount += static_cast<size_t>((biasDims[axis] - 1) * biasStrideList[axis]);
    }
    // The bias element offset for (batch, head, query, key): broadcast axes are absent or of
    // extent 1, and right-alignment maps the trailing axes.
    const auto biasOffset = [&](int64_t bi, int64_t hi, int64_t si, int64_t j) {
        const std::array<int64_t, 4> index{bi, hi, si, j};
        int64_t offset = 0;
        const auto rank = static_cast<int64_t>(biasDims.size());
        for(int64_t axis = 0; axis < rank; ++axis)
        {
            const auto full = static_cast<size_t>(4 - rank + axis);
            const auto declared = static_cast<size_t>(axis);
            if(biasDims[declared] > 1)
            {
                offset += index[full] * biasStrideList[declared];
            }
        }
        return static_cast<size_t>(offset);
    };
    // The bias in the query's dtype, or f32 (read by the strided bias objects, vendored-kernel
    // modification 31); held as f32 here either way, each value exact in its dtype.
    const bool biasF32 = spec.biasDataType == DataType::FLOAT;
    std::vector<float> biasValues;
    if(spec.withBias)
    {
        for(const auto value : fillValues<T>(biasCount, 4U))
        {
            biasValues.push_back(static_cast<float>(value));
        }
    }
    const float biasLowest = biasF32 ? std::numeric_limits<float>::lowest()
                                     : static_cast<float>(std::numeric_limits<T>::lowest());
    if(spec.withBias
       && (spec.biasMasksRows || spec.biasNegativeInfinityLeadingKeys > 0
           || spec.biasNegativeInfinityRows))
    {
        const float negativeInfinity = -std::numeric_limits<float>::infinity();
        for(int64_t bi = 0; bi < b; ++bi)
        {
            for(int64_t hi = 0; hi < h; ++hi)
            {
                for(int64_t si = 0; si < s; ++si)
                {
                    // Every third row at the dtype's lowest finite value, as a padding mask
                    // writes it: the row's scores stay finite, so its softmax is uniform over
                    // its keys rather than keyless.
                    if(spec.biasMasksRows && si % 3 == 0)
                    {
                        for(int64_t j = 0; j < skv; ++j)
                        {
                            biasValues[biasOffset(bi, hi, si, j)] = biasLowest;
                        }
                    }
                    const int64_t masked
                        = spec.biasNegativeInfinityRows && si % 3 == 0
                              ? skv
                              : std::min(spec.biasNegativeInfinityLeadingKeys, skv);
                    for(int64_t j = 0; j < masked; ++j)
                    {
                        biasValues[biasOffset(bi, hi, si, j)] = negativeInfinity;
                    }
                }
            }
        }
    }
    if(spec.withBias && (spec.biasNanRows || spec.biasNanPastKeys))
    {
        writeCdnaBiasNans(spec, biasDims, {b, h, s, skv}, biasOffset, biasValues);
    }
    void* biasDevice = nullptr;
    if(spec.withBias)
    {
        if(biasF32)
        {
            ASSERT_EQ(hipSuccess, hipMalloc(&biasDevice, biasCount * sizeof(float)));
            allocations.push_back(biasDevice);
            ASSERT_EQ(hipSuccess,
                      hipMemcpy(biasDevice,
                                biasValues.data(),
                                biasCount * sizeof(float),
                                hipMemcpyHostToDevice));
        }
        else
        {
            std::vector<T> narrowed;
            narrowed.reserve(biasCount);
            for(const auto value : biasValues)
            {
                narrowed.push_back(T(value));
            }
            ASSERT_EQ(hipSuccess, hipMalloc(&biasDevice, biasCount * sizeof(T)));
            allocations.push_back(biasDevice);
            ASSERT_EQ(
                hipSuccess,
                hipMemcpy(
                    biasDevice, narrowed.data(), biasCount * sizeof(T), hipMemcpyHostToDevice));
        }
        buffers.push_back({FLYDSL_SDPA_RDNA_BIAS_UID, biasDevice});
    }
    const auto biasAt = [&](int64_t bi, int64_t hi, int64_t si, int64_t j) {
        return static_cast<double>(biasValues[biasOffset(bi, hi, si, j)]);
    };
    // The attention sink: one logit per query head, in the query's dtype or, for a FLOAT sink
    // (vendored-kernel modification 36), in f32 with low bits no 16-bit dtype holds, so a sink
    // read as 16-bit is caught; or every one the spec's value.
    const bool floatSink = spec.sinkDataType == DataType::FLOAT;
    std::vector<T> sinkValues;
    const auto sinkLogits = cdnaSinkLogits(spec, sinkValues);
    void* sinkDevice = nullptr;
    if(spec.withSink)
    {
        const size_t sinkBytes = static_cast<size_t>(h) * (floatSink ? sizeof(float) : sizeof(T));
        ASSERT_EQ(hipSuccess, hipMalloc(&sinkDevice, sinkBytes));
        allocations.push_back(sinkDevice);
        ASSERT_EQ(hipSuccess,
                  hipMemcpy(sinkDevice,
                            floatSink ? static_cast<const void*>(sinkLogits.data())
                                      : static_cast<const void*>(sinkValues.data()),
                            sinkBytes,
                            hipMemcpyHostToDevice));
        buffers.push_back({FLYDSL_SDPA_RDNA_SINK_UID, sinkDevice});
    }
    // A runtime scale arrives with the buffers, as a host scalar.
    float runtimeScale = spec.scale;
    if(spec.scaleKind == SdpaScale::RUNTIME_TENSOR)
    {
        buffers.push_back({FLYDSL_SDPA_RDNA_SCALE_UID, &runtimeScale});
    }
    // The per-batch query and key lengths of a padded graph, INT32 [B, 1, 1, 1] on the device.
    void* seqLensQDevice = nullptr;
    void* seqLensKvDevice = nullptr;
    for(const auto& [values, device, uid] :
        {std::tuple{&spec.seqLensQ, &seqLensQDevice, FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID},
         std::tuple{&spec.seqLensKv, &seqLensKvDevice, FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID}})
    {
        if(values->empty())
        {
            continue;
        }
        ASSERT_EQ(hipSuccess, hipMalloc(device, values->size() * sizeof(int32_t)));
        allocations.push_back(*device);
        ASSERT_EQ(
            hipSuccess,
            hipMemcpy(
                *device, values->data(), values->size() * sizeof(int32_t), hipMemcpyHostToDevice));
        buffers.push_back({uid, *device});
    }
    // The Philox seed and offset: INT64 device values, or host values given at execute.
    auto hostSeed64 = static_cast<int64_t>(spec.dropoutSeed);
    auto hostOffset64 = static_cast<int64_t>(spec.dropoutOffset);
    auto hostSeed32 = static_cast<int32_t>(spec.dropoutSeed);
    auto hostOffset32 = static_cast<int32_t>(spec.dropoutOffset);
    void* seedDevice = nullptr;
    void* offsetDevice = nullptr;
    for(const auto& [present, uid, device, dataType, host64, host32] :
        {std::tuple{spec.withDropoutSeed,
                    FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID,
                    &seedDevice,
                    spec.dropoutSeedDataType.value_or(DataType::INT64),
                    &hostSeed64,
                    &hostSeed32},
         std::tuple{spec.withDropoutOffset,
                    FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_UID,
                    &offsetDevice,
                    spec.dropoutOffsetDataType.value_or(DataType::INT64),
                    &hostOffset64,
                    &hostOffset32}})
    {
        if(!present)
        {
            continue;
        }
        if(spec.passByValueTensors.count(uid) != 0)
        {
            buffers.push_back({uid,
                               dataType == DataType::INT32 ? static_cast<void*>(host32)
                                                           : static_cast<void*>(host64)});
            continue;
        }
        ASSERT_EQ(hipSuccess, hipMalloc(device, sizeof(int64_t)));
        allocations.push_back(*device);
        ASSERT_EQ(hipSuccess, hipMemcpy(*device, host64, sizeof(int64_t), hipMemcpyHostToDevice));
        buffers.push_back({uid, *device});
    }
    // The RNG dump, FLOAT [B, Hq, Sq, Skv] packed: a fresh NaN-filled buffer per kernel, below.
    const RngDumpExtents dumpExtents{b, h, s, skv};
    if(spec.withRngDump)
    {
        ASSERT_FALSE(packed || spec.withSink || spec.packedQkv);
        buffers.push_back({FLYDSL_SDPA_RDNA_RNG_DUMP_UID, nullptr});
    }

    // Element (batch, head, token, d) of a buffer laid out by logical (B, H, S, D) strides.
    const auto at = [](const std::vector<T>& t,
                       const std::vector<int64_t>& strides,
                       int64_t start,
                       int64_t bi,
                       int64_t hi,
                       int64_t si,
                       int64_t di) {
        const int64_t index
            = start + bi * strides[0] + hi * strides[1] + si * strides[2] + di * strides[3];
        return static_cast<double>(static_cast<float>(t[static_cast<size_t>(index)]));
    };
    const double scale
        = spec.scaleKind == SdpaScale::ABSENT ? 1.0 : static_cast<double>(spec.scale);
    const int64_t group = h / hk;
    // Under padding each batch has its own lengths (TEST-LOCAL ORACLE for padding, as hipDNN's
    // references define it): each clamped to [0, S], as the GPU reference clamps them; rows
    // past a batch's query length are padding, O = 0 and LSE = -inf; keys past its key length
    // are never read; and a bottom-right diagonal sits at its own Skv_b - Sq_b. Lengths given
    // without padding_mask are padding too (the references' SEM-008).
    const bool padded = !spec.seqLensQ.empty() && !spec.seqLensKv.empty();
    const auto clampedLength
        = [](int32_t value, int64_t full) { return std::clamp<int64_t>(value, 0, full); };
    // Packed (TEST-LOCAL ORACLE until hipDNN's ragged reference runs here), sequence bi is the
    // same problem at its own token counts; its rows past them do not exist, so go unchecked.
    // With per-batch lengths too (P3), a sequence's length is clamped to its slot, and the
    // slot's rows past it are padded rows.
    const auto slotQ = [&](int64_t bi) {
        return packed ? static_cast<int64_t>(spec.packedSeqLensQ[static_cast<size_t>(bi)]) : s;
    };
    const auto slotKv = [&](int64_t bi) {
        return packed ? static_cast<int64_t>(spec.packedSeqLensKv[static_cast<size_t>(bi)]) : skv;
    };
    const auto lengthQ = [&](int64_t bi) {
        return padded ? clampedLength(spec.seqLensQ[static_cast<size_t>(bi)], slotQ(bi))
                      : slotQ(bi);
    };
    const auto lengthKv = [&](int64_t bi) {
        return padded ? clampedLength(spec.seqLensKv[static_cast<size_t>(bi)], slotKv(bi))
                      : slotKv(bi);
    };
    // The reference's rule: query row i sees key j <= i + align + right, align 0 top-left
    // and Skv - Sq bottom-right (per batch under padding).
    const auto alignOf
        = [&](int64_t bi) { return alignBottomRight ? lengthKv(bi) - lengthQ(bi) : 0; };
    // Dropout (TEST-LOCAL ORACLE, FlydslSdpaPhiloxOracle.hpp, until hipDNN's reference serves
    // it): a key's probability is kept with its Philox bit and scaled by 1 / (1 - p) in O; the
    // softmax's normaliser and the LSE stay undropped. The mask's coordinates are the graph's
    // (padded) extents, so a padded batch draws the dense one's mask.
    const float dropoutP = spec.dropoutProbability.value_or(0.0F);
    const bool drops = dropoutP > 0.0F;
    const auto dropThreshold = flydsl_sdpa::dropoutThreshold(dropoutP);
    const double dropScale = drops ? static_cast<double>(flydsl_sdpa::dropoutScale(dropoutP)) : 1.0;
    const auto keeps = [&](int64_t bi, int64_t hi, int64_t si, int64_t j) {
        return !drops
               || philox_oracle::keeps(spec.dropoutSeed,
                                       spec.dropoutOffset,
                                       philox_oracle::MaskCoordinates{bi, hi, si, j, h, s, skv},
                                       dropThreshold);
    };
    // And with a left bound L, no key j <= i + align - L (a sliding window of L keys up to and
    // including the diagonal, cuDNN's convention).
    std::optional<int64_t> left = spec.leftBound;
    if(spec.mask == SdpaMask::SLIDING_WINDOW && !left.has_value())
    {
        left = 64;
    }
    // The reference, once: each row's kind and LSE, and O. A row is NaN (NaN in a score it
    // sees), keyless (no key reaches it), ordinary, padded (past its batch's query length:
    // O = 0 and LSE = -inf exactly) or absent (past a packed sequence's tokens: unchecked).
    enum class Row : uint8_t
    {
        ORDINARY,
        NAN_ROW,
        KEYLESS,
        PADDED,
        ABSENT,
    };
    const auto rowCount = static_cast<size_t>(b * h * s);
    std::vector<Row> rowKind(rowCount, Row::ORDINARY);
    std::vector<double> expectedLse(rowCount, 0.0);
    std::vector<double> expectedO(rowCount * static_cast<size_t>(dv), 0.0);
    std::vector<double> scores(static_cast<size_t>(skv));
    for(int64_t bi = 0; bi < b; ++bi)
    {
        const int64_t align = alignOf(bi);
        const int64_t offset = align + spec.rightBound.value_or(0);
        const int64_t keys = lengthKv(bi);
        for(int64_t hi = 0; hi < h; ++hi)
        {
            for(int64_t si = 0; si < s; ++si)
            {
                if(si >= slotQ(bi))
                {
                    rowKind[static_cast<size_t>((bi * h + hi) * s + si)] = Row::ABSENT;
                    continue;
                }
                if(si >= lengthQ(bi))
                {
                    // A padded row: O = 0 and LSE = -inf, exactly, on a sink object too (it is
                    // padding, not a row no key reached).
                    rowKind[static_cast<size_t>((bi * h + hi) * s + si)] = Row::PADDED;
                    continue;
                }
                // Negative for a row no key reaches: O is 0 and the LSE -inf.
                const int64_t lastKey = causal ? std::min(si + offset, keys - 1) : keys - 1;
                const int64_t firstKey
                    = left.has_value() ? std::max<int64_t>(0, si + align - *left + 1) : 0;
                double maxScore = -std::numeric_limits<double>::infinity();
                // A NaN score (NaN in Q, or in K at a visible key) makes the row's O and LSE
                // NaN, as PyTorch's softmax does.
                bool nanRow = false;
                for(int64_t j = firstKey; j <= lastKey; ++j)
                {
                    double dot = 0.0;
                    for(int64_t e = 0; e < d; ++e)
                    {
                        dot += at(q, qStrides, 0, bi, hi, si, e)
                               * at(k, kvStrides, kStart, bi, hi / group, j, e);
                    }
                    scores[static_cast<size_t>(j)]
                        = dot * scale + (spec.withBias ? biasAt(bi, hi, si, j) : 0.0);
                    nanRow = nanRow || std::isnan(scores[static_cast<size_t>(j)]);
                    maxScore = std::max(maxScore, scores[static_cast<size_t>(j)]);
                }
                // A row whose every key the bias masks with -inf reaches no key either, as
                // CpuFpReferenceSdpa defines it.
                const bool keyless = !nanRow
                                     && (lastKey < firstKey
                                         || !(maxScore > -std::numeric_limits<double>::infinity()));
                // The sink's logit joins the denominator of a row that reaches keys, and its
                // max, so a row at a bias's lowest value beside a finite sink does not
                // overflow; a row that reaches none has O = 0 and LSE = the sink.
                const double sinkLogit
                    = spec.withSink ? static_cast<double>(sinkLogits[static_cast<size_t>(hi)])
                                    : -std::numeric_limits<double>::infinity();
                if(spec.withSink && !keyless && !nanRow)
                {
                    maxScore = std::max(maxScore, sinkLogit);
                }
                double sum = 0.0;
                for(int64_t j = firstKey; j <= lastKey; ++j)
                {
                    scores[static_cast<size_t>(j)]
                        = keyless ? 0.0 : std::exp(scores[static_cast<size_t>(j)] - maxScore);
                    sum += scores[static_cast<size_t>(j)];
                }
                if(spec.withSink && !keyless)
                {
                    sum += std::exp(sinkLogit - maxScore);
                }
                const auto row = static_cast<size_t>((bi * h + hi) * s + si);
                if(nanRow)
                {
                    rowKind[row] = Row::NAN_ROW;
                }
                else if(keyless)
                {
                    rowKind[row] = Row::KEYLESS;
                }
                // Natural log of the row's sum of exp(scaled score). A row no key reaches has
                // LSE -inf, or the sink's logit when it has a finite one.
                expectedLse[row] = maxScore + std::log(sum);
                if(keyless)
                {
                    expectedLse[row] = spec.withSink && std::isfinite(sinkLogit)
                                           ? sinkLogit
                                           : -std::numeric_limits<double>::infinity();
                }
                for(int64_t e = 0; e < dv && !keyless && !nanRow; ++e)
                {
                    double expected = 0.0;
                    for(int64_t j = firstKey; j <= lastKey; ++j)
                    {
                        if(keeps(bi, hi, si, j))
                        {
                            expected += scores[static_cast<size_t>(j)]
                                        * at(v, vStrides, vStart, bi, hi / group, j, e);
                        }
                    }
                    expectedO[row * static_cast<size_t>(dv) + static_cast<size_t>(e)]
                        = expected * dropScale / sum;
                }
            }
        }
    }

    for(const auto& kernel : kernels)
    {
        SCOPED_TRACE(kernel.name);
        const auto& handler = dispatchHandler(cdnaEngineFor(kernel));
        const auto prepared = handler.prepare(context, bound, kernel);
        ASSERT_NE(prepared, nullptr);
        // All-ones bytes are NaN in bf16 and f16 (and f32): an element the kernel never
        // writes fails rather than reading as a zero, or as the previous kernel's answer.
        ASSERT_EQ(hipSuccess, hipMemset(oDevice, 0xFF, oCount * sizeof(T)));
        if(withStats)
        {
            ASSERT_EQ(hipSuccess, hipMemset(statsDevice, 0xFF, statsCount * sizeof(float)));
        }
        if(packed)
        {
            ASSERT_EQ(hipSuccess, hipMemset(packedODevice, 0xFF, packedOCount * sizeof(T)));
            ASSERT_EQ(hipSuccess,
                      hipMemset(packedStatsDevice, 0xFF, packedStatsCount * sizeof(float)));
        }
        DeviceRngDump rngDump(dumpExtents, spec.withRngDump);
        for(auto& buffer : buffers)
        {
            if(buffer.uid == FLYDSL_SDPA_RDNA_RNG_DUMP_UID)
            {
                buffer.ptr = rngDump.device();
            }
        }
        // A split object launched with more than one split writes its partials here: exactly the
        // size the engine reports, so a launch that writes past it fails the case.
        const auto workspaceBytes = handler.workspaceBytes(context, bound, kernel);
        const GuardedWorkspace workspace(workspaceBytes);
        const Handle handle;
        handler.launch(handle,
                       *prepared,
                       buffers.data(),
                       static_cast<uint32_t>(buffers.size()),
                       workspace.get());
        ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
        EXPECT_EQ(workspace.overwrittenGuardBytes(), 0U)
            << "the launch wrote past the " << workspaceBytes << "-byte workspace it reported";
        ASSERT_EQ(hipSuccess,
                  hipMemcpy(o.data(), oDevice, oCount * sizeof(T), hipMemcpyDeviceToHost));
        if(withStats)
        {
            ASSERT_EQ(
                hipSuccess,
                hipMemcpy(
                    stats.data(), statsDevice, statsCount * sizeof(float), hipMemcpyDeviceToHost));
        }
        // Packed: O and the LSE unpacked into the dense tensors, and the guard rows checked.
        int64_t guardWrites = 0;
        if(packed)
        {
            std::vector<T> packedO(packedOCount);
            std::vector<float> packedStats(packedStatsCount);
            ASSERT_EQ(hipSuccess,
                      hipMemcpy(packedO.data(),
                                packedODevice,
                                packedOCount * sizeof(T),
                                hipMemcpyDeviceToHost));
            ASSERT_EQ(hipSuccess,
                      hipMemcpy(packedStats.data(),
                                packedStatsDevice,
                                packedStatsCount * sizeof(float),
                                hipMemcpyDeviceToHost));
            for(int64_t bi = 0; bi < b; ++bi)
            {
                const int64_t first = offsetsQ[static_cast<size_t>(bi)];
                for(int64_t si = 0; si < spec.packedSeqLensQ[static_cast<size_t>(bi)]; ++si)
                {
                    for(int64_t hi = 0; hi < h; ++hi)
                    {
                        for(int64_t e = 0; e < dv; ++e)
                        {
                            o[static_cast<size_t>(bi * oStrides[0] + hi * oStrides[1]
                                                  + si * oStrides[2] + e * oStrides[3])]
                                = packedO[static_cast<size_t>(((first + si) * h + hi) * dv + e)];
                        }
                        if(withStats)
                        {
                            const auto statsIndex = spec.statsSequenceMajor
                                                        ? (bi * s + si) * h + hi
                                                        : (bi * h + hi) * s + si;
                            stats[static_cast<size_t>(statsIndex)]
                                = packedStats[static_cast<size_t>((first + si) * h + hi)];
                        }
                    }
                }
            }
            // The guard: every element still the fill's NaN.
            for(int64_t t = offsetsQ.back(); t < tokensQ; ++t)
            {
                for(int64_t hi = 0; hi < h; ++hi)
                {
                    for(int64_t e = 0; e < dv; ++e)
                    {
                        guardWrites += std::isnan(static_cast<float>(
                                           packedO[static_cast<size_t>((t * h + hi) * dv + e)]))
                                           ? 0
                                           : 1;
                    }
                    guardWrites
                        += std::isnan(packedStats[static_cast<size_t>(t * h + hi)]) ? 0 : 1;
                }
            }
        }
        EXPECT_EQ(guardWrites, 0) << "packed O or LSE written past the last sequence";
        // The RNG dump: the oracle's mask bit for bit at every element, padded rows and keys
        // past a batch's length included, with hipDNN's keep-rate check; and O as hipDNN's CPU
        // reference computes it with the dump as its explicit mask.
        if(spec.withRngDump)
        {
            expectCdnaRngDump(
                RngDumpReferenceProblem{dumpExtents,
                                        hk,
                                        d,
                                        dv,
                                        static_cast<float>(scale),
                                        dropoutP,
                                        left.value_or(-1),
                                        causal ? spec.rightBound.value_or(0) : -1,
                                        !alignBottomRight,
                                        padded ? spec.seqLensQ : std::vector<int32_t>{},
                                        padded ? spec.seqLensKv : std::vector<int32_t>{}},
                rngDump.readBack(),
                spec,
                CdnaHostOperands<T>{
                    q, k, v, o, qStrides, kvStrides, vStrides, oStrides, kStart, vStart},
                absoluteTolerance);
        }
        double worst = 0.0;
        int64_t nonFinite = 0;
        int64_t firstNonFiniteRow = std::numeric_limits<int64_t>::max();
        int64_t lastNonFiniteRow = -1;
        std::string where;
        double worstLse = 0.0;
        std::string whereLse;
        int64_t paddedRowMisses = 0;
        std::string wherePadded;
        for(int64_t bi = 0; bi < b; ++bi)
        {
            for(int64_t hi = 0; hi < h; ++hi)
            {
                for(int64_t si = 0; si < s; ++si)
                {
                    const auto row = static_cast<size_t>((bi * h + hi) * s + si);
                    if(rowKind[row] == Row::ABSENT)
                    {
                        continue;
                    }
                    if(rowKind[row] == Row::PADDED)
                    {
                        bool exact = true;
                        for(int64_t e = 0; e < dv; ++e)
                        {
                            exact = exact && at(o, oStrides, 0, bi, hi, si, e) == 0.0;
                        }
                        if(withStats)
                        {
                            const auto statsIndex = spec.statsSequenceMajor
                                                        ? (bi * s + si) * h + hi
                                                        : (bi * h + hi) * s + si;
                            const auto lse = stats[static_cast<size_t>(statsIndex)];
                            exact = exact && std::isinf(lse) && lse < 0.0F;
                        }
                        if(!exact)
                        {
                            ++paddedRowMisses;
                            wherePadded = "b=" + std::to_string(bi) + " h=" + std::to_string(hi)
                                          + " s=" + std::to_string(si);
                        }
                        continue;
                    }
                    if(withStats)
                    {
                        // [B, H, Sq] packed, or [B, Sq, H] when the graph lays it out so.
                        const auto statsIndex = spec.statsSequenceMajor ? (bi * s + si) * h + hi
                                                                        : (bi * h + hi) * s + si;
                        const auto lse
                            = static_cast<double>(stats[static_cast<size_t>(statsIndex)]);
                        double lseError = std::abs(lse - expectedLse[row]);
                        if(rowKind[row] == Row::NAN_ROW)
                        {
                            lseError = std::isnan(lse) ? 0.0 : 1e30;
                        }
                        else if(rowKind[row] == Row::KEYLESS && std::isinf(expectedLse[row]))
                        {
                            // No sink, or a sink of -inf: the LSE is exactly -inf.
                            lseError = std::isinf(lse) && lse < 0.0 ? 0.0 : 1e30;
                        }
                        else if(rowKind[row] == Row::KEYLESS && floatSink
                                && static_cast<float>(lse)
                                       != static_cast<float>(expectedLse[row]))
                        {
                            // The sink itself: an f32 sink's bits exactly, so one read as
                            // 16-bit is a full miss.
                            lseError = 1e30;
                        }
                        if(!(lseError <= LSE_TOLERANCE))
                        {
                            worstLse = std::max(worstLse, std::isnan(lseError) ? 1e30 : lseError);
                            whereLse = "b=" + std::to_string(bi) + " h=" + std::to_string(hi)
                                       + " s=" + std::to_string(si);
                        }
                    }
                    for(int64_t e = 0; e < dv; ++e)
                    {
                        const double expected
                            = expectedO[row * static_cast<size_t>(dv) + static_cast<size_t>(e)];
                        // NaN or inf (an unwritten element reads as NaN) is a full miss, but
                        // in a NaN row, where anything but NaN is.
                        const double actual = at(o, oStrides, 0, bi, hi, si, e);
                        double error = std::isfinite(actual) ? std::abs(actual - expected)
                                                             : std::numeric_limits<double>::max();
                        if(rowKind[row] == Row::NAN_ROW)
                        {
                            error = std::isnan(actual) ? 0.0 : std::numeric_limits<double>::max();
                        }
                        else if(spec.exposeDropoutMask && (actual != 0.0) != (expected != 0.0))
                        {
                            // The mask read off O: a key kept or dropped against the oracle's
                            // bit is a full miss, however small the probability.
                            error = std::numeric_limits<double>::max();
                        }
                        else if(!std::isfinite(actual))
                        {
                            ++nonFinite;
                            firstNonFiniteRow = std::min(firstNonFiniteRow, si);
                            lastNonFiniteRow = std::max(lastNonFiniteRow, si);
                        }
                        if(error > worst)
                        {
                            worst = error;
                            where = "b=" + std::to_string(bi) + " h=" + std::to_string(hi)
                                    + " s=" + std::to_string(si) + " d=" + std::to_string(e)
                                    + ": got " + std::to_string(actual) + ", expected "
                                    + std::to_string(expected);
                        }
                    }
                }
            }
        }
        EXPECT_LE(worst, absoluteTolerance)
            << "worst error at " << where << "; " << nonFinite << " non-finite element(s)"
            << (nonFinite > 0 ? " in query rows " + std::to_string(firstNonFiniteRow) + "-"
                                    + std::to_string(lastNonFiniteRow)
                              : std::string())
            << "; split workspace " << workspaceBytes << " bytes";
        EXPECT_EQ(worstLse, 0.0) << "an LSE outside tolerance, worst at " << whereLse;
        EXPECT_EQ(paddedRowMisses, 0)
            << "padded rows not O = 0 and LSE = -inf, last at " << wherePadded;

        // A sink, against hipDNN's CPU reference as well (FLOAT [1, Hq, 1, 1], SEM-022): with
        // the bias, the band and padding the graph has; and a bias under padding
        // (vendored-kernel modification 37), sink or not. Not packed, dropped or NaN: the
        // reference's packed form, dropout and NaN rows are checked above.
        if((spec.withSink || (padded && spec.withBias)) && !packed && !drops
           && !spec.queryNanPosition && !spec.keyNanPosition && !spec.biasNanRows)
        {
            SdpaCpuReferenceProblem reference{b,
                                              h,
                                              hk,
                                              hk,
                                              s,
                                              skv,
                                              d,
                                              dv,
                                              static_cast<float>(scale),
                                              left.value_or(-1),
                                              causal ? spec.rightBound.value_or(0) : -1,
                                              !alignBottomRight,
                                              {},
                                              {},
                                              sinkLogits,
                                              spec.withBias,
                                              withStats};
            for(int64_t bi = 0; padded && bi < b; ++bi)
            {
                reference.seqLensQ.push_back(static_cast<int32_t>(lengthQ(bi)));
                reference.seqLensKv.push_back(static_cast<int32_t>(lengthKv(bi)));
            }
            expectCdnaSinkIsTheCpuReference(
                reference,
                CdnaHostOperands<T>{
                    q, k, v, o, qStrides, kvStrides, vStrides, oStrides, kStart, vStart},
                biasAt,
                stats,
                spec.statsSequenceMajor,
                absoluteTolerance,
                LSE_TOLERANCE);
        }
    }
}

/// One kernel: the per-object suite's entry point.
template <typename T>
void expectCandidateMatchesReference(const MatchContext& context,
                                     const BoundTokens& bound,
                                     const KernelDefinition& kernel,
                                     const SdpaGraphSpec& spec,
                                     double absoluteTolerance)
{
    expectCandidatesMatchReference<T>(context, bound, {kernel}, spec, absoluteTolerance);
}

/// Every staged candidate the matcher admits for @p spec must match the reference.
///
/// No candidate at all fails on a device this build ships @p family's op for (and skips
/// elsewhere): a mis-pointed descriptor root, a dropped object or a matcher that declines
/// every object would otherwise read as a device FlyDSL does not serve. A case aimed at
/// one engine names it as @p family, and on a device that engine ships for, one of its
/// objects must be among those run.
void expectCdnaMatchesReference(const SdpaGraphSpec& spec,
                                double absoluteTolerance,
                                const PackSymbols& family = FLYDSL_SDPA_CDNA)
{
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec), packedArchDeviceProperties());
    const auto bound = matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());
    const auto candidates = stagedCdnaCandidates(fixture.context(), *bound);
    if(candidates.empty())
    {
        reportUnstaged(shippedOpFor(family),
                       "no CDNA object staged for this device ("
                           + packedArchDeviceProperties().gcnArchName + ") and shape");
        return;
    }
    if(&family != &FLYDSL_SDPA_CDNA && flydslShipsForThisDevice(shippedOpFor(family))
       && std::none_of(candidates.begin(), candidates.end(), [&family](const auto& kernel) {
              return &cdnaEngineFor(kernel) == &family;
          }))
    {
        ADD_FAILURE() << "this build ships " << shippedOpFor(family)
                      << " for this device, but none of its objects serves the shape";
    }
    if(spec.dataType == DataType::HALF)
    {
        // P is rounded to the input dtype before the second product: f16's 11-bit
        // mantissa allows about an eighth of bf16's budget.
        expectCandidatesMatchReference<hipdnn_data_sdk::types::half>(
            fixture.context(), *bound, candidates, spec, std::min(absoluteTolerance, 4e-3));
    }
    else if(spec.dataType == DataType::FLOAT)
    {
        // Exact f32 products and an f32 P (vendored-kernel modification 35): the error is
        // the accumulation's and exp2's, about 1e-6 on these values.
        expectCandidatesMatchReference<float>(
            fixture.context(), *bound, candidates, spec, std::min(absoluteTolerance, 1e-5));
    }
    else
    {
        expectCandidatesMatchReference<Bf16>(
            fixture.context(), *bound, candidates, spec, absoluteTolerance);
    }
}

/// bf16 output rounding is ~2^-8 relative on values of magnitude <= 1.
constexpr double BF16_TOLERANCE = 2e-2;
/// Exact f32 (vendored-kernel modification 35): f32 products and accumulation, P in f32.
constexpr double F32_TOLERANCE = 1e-5;

// The model shape, where the dual-wave kernel ships (gfx950) one of its objects runs too.
TEST(TestGpuFlydslSdpaCdnaDispatch, NonCausalMatchesTheReference)
{
    expectCdnaMatchesReference(cdnaSpec(), BF16_TOLERANCE, FLYDSL_SDPA_CDNA_DUALWAVE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, CausalMatchesTheReference)
{
    auto spec = cdnaSpec();
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE, FLYDSL_SDPA_CDNA_DUALWAVE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, BatchedRaggedTailMatchesTheReference)
{
    auto spec = cdnaSpec();
    spec.batch = 2;
    spec.seqLenQ = 333;
    spec.seqLenKv = 333;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, NonCausalRaggedTailMatchesTheReference)
{
    // A key length that is not a multiple of the KV tile: only an object that masks the
    // last tile may run it.
    auto spec = cdnaSpec();
    spec.batch = 2;
    spec.seqLenQ = 333;
    spec.seqLenKv = 333;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, AppliesAScaleValue)
{
    auto spec = cdnaSpec();
    spec.scaleKind = SdpaScale::VALUE;
    spec.scale = 0.07F;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ReadsARuntimeScaleAtExecute)
{
    auto spec = cdnaSpec();
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.scaleKind = SdpaScale::RUNTIME_TENSOR;
    spec.scale = 0.2F;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ServesMultiHeadAttention)
{
    auto spec = cdnaSpec();
    spec.heads = 8;
    spec.kvHeads = 8;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ServesMultiQueryAttention)
{
    auto spec = cdnaSpec();
    spec.heads = 16;
    spec.kvHeads = 1;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ServesAnOddGroupSizeAtARaggedLength)
{
    // Five query heads per KV head, and a length that is not a multiple of any tile.
    auto spec = cdnaSpec();
    spec.heads = 40;
    spec.kvHeads = 8;
    spec.seqLenQ = 200;
    spec.seqLenKv = 200;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, WritesTheLogSumExpOfEachRow)
{
    auto spec = cdnaSpec();
    spec.stats = SdpaStats::LSE;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, WritesTheLogSumExpOfACausalRaggedBatch)
{
    auto spec = cdnaSpec();
    spec.batch = 2;
    spec.seqLenQ = 333;
    spec.seqLenKv = 333;
    spec.heads = 16;
    spec.kvHeads = 4;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ServesCrossAttentionWithBottomRightCausal)
{
    // A diagonal that falls mid-tile, where the gfx950 pipeline needs cross_seqlen.
    auto spec = cdnaSpec();
    spec.batch = 2;
    spec.seqLenQ = 333;
    spec.seqLenKv = 700;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ServesTheDeprecatedCausalBooleanAtABottomRightAlignment)
{
    // causal_mask with diagonal_alignment BOTTOM_RIGHT is bottom-right causal: with fewer
    // queries than keys each row sees Skv - Sq more keys than top-left would give it.
    auto spec = cdnaSpec();
    spec.seqLenQ = 256;
    spec.seqLenKv = 1024;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.deprecatedCausalTopLeft = true;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ServesTheDeprecatedCausalBooleansAlone)
{
    // Each boolean as the only mask on the graph: no right_bound, so ignoring the boolean
    // runs the graph non-causal, and reading the wrong corner moves the diagonal by Skv - Sq.
    using hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment;
    auto spec = cdnaSpec();
    spec.seqLenQ = 200;
    spec.seqLenKv = 333;
    spec.stats = SdpaStats::LSE;
    {
        SCOPED_TRACE("causal_mask, top-left");
        auto topLeft = spec;
        topLeft.deprecatedCausalTopLeft = true;
        expectCdnaMatchesReference(topLeft, BF16_TOLERANCE);
    }
    {
        SCOPED_TRACE("causal_mask, refined to bottom-right by diagonal_alignment");
        auto refined = spec;
        refined.deprecatedCausalTopLeft = true;
        refined.diagonalAlignment = DiagonalAlignment::BOTTOM_RIGHT;
        expectCdnaMatchesReference(refined, BF16_TOLERANCE);
    }
    {
        SCOPED_TRACE("causal_mask_bottom_right over a top-left alignment");
        auto bottomRight = spec;
        bottomRight.deprecatedCausalBottomRight = true;
        expectCdnaMatchesReference(bottomRight, BF16_TOLERANCE);
    }
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ServesMoreQueriesThanKeysWithRowsNoKeyReaches)
{
    auto spec = cdnaSpec();
    spec.seqLenQ = 700;
    spec.seqLenKv = 333;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

// The full tier only: sweeps and large shapes whose reference costs tens of seconds on a CDNA
// runner, beside the quick tier's 300 s budget for the whole binary. Instantiated once under
// the `Full` prefix (below), which HIP_MLOPS_ENGINE_test_categories.yaml excludes from the
// quick, standard and comprehensive tiers; the parameter only carries that prefix.
class TestGpuFlydslSdpaCdnaFullTier : public ::testing::TestWithParam<int>
{
};

TEST_P(TestGpuFlydslSdpaCdnaFullTier, ServesTheCombinationsNoOtherCaseReaches)
{
    // f16 objects built with kv_align (they run only at a key length that is a multiple of
    // 64): non-causal at each exact head dim.
    for(const int64_t headDim : {int64_t{64}, int64_t{128}, int64_t{256}})
    {
        SCOPED_TRACE("f16 aligned, head_dim " + std::to_string(headDim));
        auto spec = cdnaSpec();
        spec.dataType = DataType::HALF;
        spec.headDim = headDim;
        spec.seqLenQ = 200;
        spec.seqLenKv = 1024;
        expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    }
    // A split merge where no split reaches the first rows (bottom-right, Sq > Skv), with the
    // LSE, with and without a sink: the merge must give O = 0 and LSE -inf (or the sink).
    {
        SCOPED_TRACE("merge, rows no split reaches");
        auto spec = cdnaSpec();
        spec.headDim = 64;
        spec.seqLenQ = 600;
        spec.seqLenKv = 520;
        spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
        spec.stats = SdpaStats::LSE;
        expectCdnaMatchesReference(spec, BF16_TOLERANCE);
        spec.withSink = true;
        expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    }
    // A window with more queries than keys, bottom-right: the window's lower edge clamps.
    {
        SCOPED_TRACE("bottom-right window, Sq > Skv");
        auto spec = cdnaSpec();
        spec.seqLenQ = 500;
        spec.seqLenKv = 300;
        spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
        spec.leftBound = 48;
        spec.stats = SdpaStats::LSE;
        expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    }
    // One key, and one query over one key.
    for(const int64_t seqLenQ : {int64_t{200}, int64_t{1}})
    {
        SCOPED_TRACE("Skv 1, Sq " + std::to_string(seqLenQ));
        auto spec = cdnaSpec();
        spec.seqLenQ = seqLenQ;
        spec.seqLenKv = 1;
        spec.stats = SdpaStats::LSE;
        expectCdnaMatchesReference(spec, BF16_TOLERANCE);
        spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
        expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    }
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ServesAVaeMidBlockAtHeadDim512)
{
    // The Stable Diffusion / Flux VAE mid-block attention: one head x 512 channels, non-causal,
    // on the generic prefill engine's exact 512/512 objects (O split four ways across
    // workgroups; K chunked on gfx942 and gfx90a). A ragged tail, and causal with GQA.
    auto spec = cdnaSpec();
    spec.batch = 2;
    spec.heads = 1;
    spec.kvHeads = 1;
    spec.headDim = 512;
    spec.seqLenQ = 1024;
    spec.seqLenKv = 1024;
    spec.stats = SdpaStats::LSE;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE, FLYDSL_SDPA_CDNA);
    spec.seqLenQ = 1000;
    spec.seqLenKv = 777;
    spec.dataType = DataType::HALF;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE, FLYDSL_SDPA_CDNA);
    spec.dataType = DataType::BFLOAT16;
    spec.heads = 4;
    spec.batch = 1;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE, FLYDSL_SDPA_CDNA);
}

/// DeepSeek-V4's attention: Q/K and V 512 wide, one KV head, causal, with or without a 128-key
/// sliding window. The decode engine's wide objects serve it (any query length), so it must be
/// right here too.
void expectDeepSeekV4Prefill(int64_t seqLen, bool window)
{
    SCOPED_TRACE("Sq = Skv = " + std::to_string(seqLen) + (window ? ", window 128" : ""));
    auto spec = cdnaSpec();
    spec.headDim = 512;
    spec.heads = 8;
    spec.kvHeads = 1;
    spec.seqLenQ = seqLen;
    spec.seqLenKv = seqLen;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    if(window)
    {
        spec.leftBound = 128;
    }
    spec.stats = SdpaStats::LSE;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE, FLYDSL_SDPA_CDNA_DECODE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ServesDeepSeekV4PrefillAtHeadDim512)
{
    // Long enough for the window to bind; the prefill-sized lengths are in
    // Full/TestGpuFlydslSdpaCdnaFullTier.
    expectDeepSeekV4Prefill(256, true);
}

TEST_P(TestGpuFlydslSdpaCdnaFullTier, ServesDeepSeekV4PrefillAtHeadDim512)
{
    for(const int64_t seqLen : {int64_t{1024}, int64_t{2048}})
    {
        for(const bool window : {true, false})
        {
            expectDeepSeekV4Prefill(seqLen, window);
        }
    }
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ServesChunkedPrefillWithTopLeftCausal)
{
    auto spec = cdnaSpec();
    spec.seqLenQ = 200;
    spec.seqLenKv = 600;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ServesATopLeftBand)
{
    auto spec = cdnaSpec();
    spec.seqLenQ = 128;
    spec.seqLenKv = 512;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.rightBound = 64;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ServesDiagonalsThatPassTheKeys)
{
    // Rows whose diagonal passes the last key see every key, and the last KV tile's padding
    // stays out (modification 30): top-left with more queries than keys, and bands wider
    // than Skv - Sq, at ragged key lengths, each head dim's generic object, f16, a bias, a
    // sink, a window, and decode-sized queries.
    auto spec = cdnaSpec();
    spec.stats = SdpaStats::LSE;
    {
        SCOPED_TRACE("top-left, Sq 700 > Skv 333");
        auto topLeft = spec;
        topLeft.mask = SdpaMask::TOP_LEFT_CAUSAL;
        topLeft.seqLenQ = 700;
        topLeft.seqLenKv = 333;
        expectCdnaMatchesReference(topLeft, BF16_TOLERANCE);
        for(const int64_t headDim : {int64_t{64}, int64_t{256}, int64_t{96}})
        {
            SCOPED_TRACE("head_dim " + std::to_string(headDim));
            auto other = topLeft;
            other.headDim = headDim;
            expectCdnaMatchesReference(other, BF16_TOLERANCE);
        }
        auto half = topLeft;
        half.dataType = DataType::HALF;
        expectCdnaMatchesReference(half, BF16_TOLERANCE);
    }
    {
        SCOPED_TRACE("top-left band 100, Sq = Skv = 300");
        auto band = spec;
        band.mask = SdpaMask::TOP_LEFT_CAUSAL;
        band.seqLenQ = 300;
        band.seqLenKv = 300;
        band.rightBound = 100;
        expectCdnaMatchesReference(band, BF16_TOLERANCE);
        SCOPED_TRACE("with a window of 64");
        band.leftBound = 64;
        expectCdnaMatchesReference(band, BF16_TOLERANCE);
    }
    {
        SCOPED_TRACE("bottom-right band 250, Sq 500 > Skv 300: rows from 250 see every key");
        auto band = spec;
        band.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
        band.seqLenQ = 500;
        band.seqLenKv = 300;
        band.rightBound = 250;
        expectCdnaMatchesReference(band, BF16_TOLERANCE);
        SCOPED_TRACE("with a sink");
        band.headDim = 64;
        band.withSink = true;
        expectCdnaMatchesReference(band, BF16_TOLERANCE);
    }
    {
        SCOPED_TRACE("top-left with a per-head bias, Sq 400 > Skv 264");
        auto bias = spec;
        bias.mask = SdpaMask::TOP_LEFT_CAUSAL;
        bias.seqLenQ = 400;
        bias.seqLenKv = 264;
        bias.withBias = true;
        bias.biasDataType = DataType::BFLOAT16;
        bias.biasDims = {1, 32, 400, 264};
        expectCdnaMatchesReference(bias, BF16_TOLERANCE);
    }
    {
        // Decode-sized queries: the prefill objects serve them, the decode objects declining.
        SCOPED_TRACE("top-left, Sq 40 > Skv 20, and a band past 20000 keys at Sq 3");
        auto shortQuery = spec;
        shortQuery.mask = SdpaMask::TOP_LEFT_CAUSAL;
        shortQuery.seqLenQ = 40;
        shortQuery.seqLenKv = 20;
        expectCdnaMatchesReference(shortQuery, BF16_TOLERANCE);
        shortQuery.seqLenQ = 3;
        shortQuery.seqLenKv = 20000;
        shortQuery.rightBound = 19999;
        expectCdnaMatchesReference(shortQuery, BF16_TOLERANCE);
    }
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ServesSlidingWindows)
{
    // The diagonal alone (left_bound 1) and one key more, narrower than a KV tile, across a
    // tile boundary, and wider than the keys; top-left and bottom-right; ragged lengths.
    for(const int64_t left :
        {int64_t{1}, int64_t{2}, int64_t{37}, int64_t{64}, int64_t{200}, int64_t{4096}})
    {
        SCOPED_TRACE("left_bound " + std::to_string(left));
        auto spec = cdnaSpec();
        spec.seqLenQ = 333;
        spec.seqLenKv = 333;
        spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
        spec.leftBound = left;
        expectCdnaMatchesReference(spec, BF16_TOLERANCE);
        spec.seqLenQ = 96;
        spec.seqLenKv = 700;
        spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
        spec.dataType = DataType::HALF;
        expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    }
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ServesSlidingWindowsAtHeadDim64And256)
{
    for(const int64_t headDim : {int64_t{64}, int64_t{256}})
    {
        SCOPED_TRACE("head_dim " + std::to_string(headDim));
        auto spec = cdnaSpec();
        spec.headDim = headDim;
        spec.seqLenQ = 300;
        spec.seqLenKv = 300;
        spec.mask = SdpaMask::SLIDING_WINDOW;
        expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    }
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ServesNonCausalCrossAttentionAndDecode)
{
    auto cross = cdnaSpec();
    cross.seqLenQ = 300;
    cross.seqLenKv = 100;
    expectCdnaMatchesReference(cross, BF16_TOLERANCE);
    auto decode = cdnaSpec();
    decode.seqLenQ = 1;
    decode.seqLenKv = 1000;
    decode.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    expectCdnaMatchesReference(decode, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ServesHeadDim64InBothDtypes)
{
    auto spec = cdnaSpec();
    spec.headDim = 64;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    spec.dataType = DataType::HALF;
    spec.mask = SdpaMask::NONE;
    spec.seqLenQ = 300;
    spec.seqLenKv = 300;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ServesAValueHeadDimNarrowerThanTheQueryOne)
{
    // Multi-head latent attention, un-absorbed: Q and K 192 wide, V and O 128.
    auto spec = cdnaSpec();
    spec.headDim = 192;
    spec.valueHeadDim = 128;
    spec.heads = 8;
    spec.kvHeads = 8;
    spec.seqLenQ = 300;
    spec.seqLenKv = 300;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);

    // Non-causal cross-attention with GQA and a ragged last tile, in f16.
    spec.dataType = DataType::HALF;
    spec.mask = SdpaMask::NONE;
    spec.kvHeads = 2;
    spec.seqLenQ = 150;
    spec.seqLenKv = 333;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);

    // Bottom-right causal over a longer cache, BHSD, and a Q/K head dim off the bucket's
    // power of two that is not 192.
    spec.dataType = DataType::BFLOAT16;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.layout = SdpaLayout::BHSD;
    spec.headDim = 160;
    spec.seqLenQ = 64;
    spec.seqLenKv = 257;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, DecodesWithSplitKeysAndPackedGroups)
{
    // Decode-shaped graphs, which the split, packed objects serve: a single step over a
    // long cache (many splits), a multi-token causal step, MQA, MHA, BHSD and a window.
    auto spec = cdnaSpec();
    spec.seqLenQ = 1;
    spec.seqLenKv = 20000;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE, FLYDSL_SDPA_CDNA_DECODE);

    spec.batch = 2;
    spec.seqLenQ = 7;
    spec.seqLenKv = 5000;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE, FLYDSL_SDPA_CDNA_DECODE);

    spec.batch = 1;
    spec.heads = 16;
    spec.kvHeads = 1;
    spec.seqLenQ = 1;
    spec.seqLenKv = 3000;
    spec.mask = SdpaMask::NONE;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE, FLYDSL_SDPA_CDNA_DECODE);

    spec.heads = 8;
    spec.kvHeads = 8;
    spec.seqLenQ = 3;
    spec.layout = SdpaLayout::BHSD;
    spec.dataType = DataType::HALF;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE, FLYDSL_SDPA_CDNA_DECODE);

    spec.layout = SdpaLayout::BSHD;
    spec.dataType = DataType::BFLOAT16;
    spec.heads = 32;
    spec.kvHeads = 8;
    spec.seqLenQ = 4;
    spec.mask = SdpaMask::SLIDING_WINDOW;
    spec.leftBound = 700;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE, FLYDSL_SDPA_CDNA_DECODE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, DecodesWithSplitKeysAtHeadDim64And256)
{
    for(const int64_t headDim : {int64_t{64}, int64_t{256}})
    {
        SCOPED_TRACE("head_dim " + std::to_string(headDim));
        auto spec = cdnaSpec();
        spec.headDim = headDim;
        spec.seqLenQ = 2;
        spec.seqLenKv = 9000;
        spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
        spec.stats = SdpaStats::LSE;
        expectCdnaMatchesReference(spec, BF16_TOLERANCE, FLYDSL_SDPA_CDNA_DECODE);
    }
}

TEST(TestGpuFlydslSdpaCdnaDispatch, DecodesAbsorbedMlaWithOneKvHead)
{
    // Absorbed multi-head latent attention: every query head reads one KV head whose keys
    // are 576 wide and whose values are its 512-wide latent; and DeepSeek-V4's 512/512.
    // Only the decode engine's objects are that wide.
    auto spec = cdnaSpec();
    spec.headDim = 576;
    spec.valueHeadDim = 512;
    spec.heads = 16;
    spec.kvHeads = 1;
    spec.seqLenQ = 1;
    spec.seqLenKv = 3000;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE, FLYDSL_SDPA_CDNA_DECODE);

    // A two-token step in f16, non-causal.
    spec.dataType = DataType::HALF;
    spec.seqLenQ = 2;
    spec.seqLenKv = 777;
    spec.mask = SdpaMask::NONE;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE, FLYDSL_SDPA_CDNA_DECODE);

    spec.dataType = DataType::BFLOAT16;
    spec.headDim = 512;
    spec.heads = 8;
    spec.seqLenQ = 1;
    spec.seqLenKv = 1500;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE, FLYDSL_SDPA_CDNA_DECODE);
}

/// The 96, 160, 192 and 224 buckets (vendored-kernel modification 26), at @p headDim, every
/// candidate the matcher admits (wider buckets too): causal bf16 with the LSE, then non-causal
/// f16 BHSD at a ragged length.
void expectBucketWidth(int64_t headDim)
{
    SCOPED_TRACE("head_dim " + std::to_string(headDim));
    auto spec = cdnaSpec();
    spec.headDim = headDim;
    spec.seqLenQ = 200;
    spec.seqLenKv = 333;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    spec.dataType = DataType::HALF;
    spec.layout = SdpaLayout::BHSD;
    spec.mask = SdpaMask::NONE;
    spec.seqLenQ = 129;
    spec.seqLenKv = 777;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ServesAWidthInsideABucketThatIsNotAPowerOfTwo)
{
    // Inside the 160 bucket; every width of every bucket is in Full/TestGpuFlydslSdpaCdnaFullTier.
    expectBucketWidth(152);
}

TEST_P(TestGpuFlydslSdpaCdnaFullTier, ServesEveryWidthInsideTheBucketsThatAreNotPowersOfTwo)
{
    for(const int64_t headDim : {int64_t{72},
                                 int64_t{88},
                                 int64_t{96},
                                 int64_t{136},
                                 int64_t{152},
                                 int64_t{160},
                                 int64_t{168},
                                 int64_t{184},
                                 int64_t{192},
                                 int64_t{200},
                                 int64_t{216},
                                 int64_t{224}})
    {
        expectBucketWidth(headDim);
    }
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ServesHeadDimsBetweenTheBuiltOnes)
{
    // Q, K and V all the graph's head dim, served by the 64, 128 and 256 bucket objects.
    for(const int64_t headDim : {int64_t{16},
                                 int64_t{32},
                                 int64_t{40},
                                 int64_t{72},
                                 int64_t{80},
                                 int64_t{88},
                                 int64_t{96},
                                 int64_t{160},
                                 int64_t{192},
                                 int64_t{200},
                                 int64_t{240},
                                 int64_t{248}})
    {
        SCOPED_TRACE("head_dim " + std::to_string(headDim));
        auto spec = cdnaSpec();
        spec.headDim = headDim;
        spec.seqLenQ = 200;
        spec.seqLenKv = 333;
        spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
        spec.stats = SdpaStats::LSE;
        expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    }
    // Non-causal, f16, BHSD, a sliding window.
    auto spec = cdnaSpec();
    spec.headDim = 80;
    spec.dataType = DataType::HALF;
    spec.layout = SdpaLayout::BHSD;
    spec.seqLenQ = 129;
    spec.seqLenKv = 129;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    spec.dataType = DataType::BFLOAT16;
    spec.layout = SdpaLayout::BSHD;
    spec.headDim = 112;
    spec.mask = SdpaMask::SLIDING_WINDOW;
    spec.leftBound = 37;
    spec.seqLenQ = 300;
    spec.seqLenKv = 300;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, AddsAnAttentionSink)
{
    // GPT-OSS: head_dim 64, GQA, a sink logit per query head, causal with and without a
    // sliding window; the LSE includes the sink.
    auto spec = cdnaSpec();
    spec.headDim = 64;
    spec.heads = 16;
    spec.kvHeads = 2;
    spec.withSink = true;
    spec.stats = SdpaStats::LSE;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.seqLenQ = 300;
    spec.seqLenKv = 300;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    spec.mask = SdpaMask::SLIDING_WINDOW;
    spec.leftBound = 128;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);

    // Non-causal, f16, the rank-4 form.
    spec.mask = SdpaMask::NONE;
    spec.leftBound.reset();
    spec.dataType = DataType::HALF;
    spec.sinkRank4 = true;
    spec.seqLenQ = 129;
    spec.seqLenKv = 333;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);

    // Rows no key reaches (bottom-right, more queries than keys): O = 0, LSE = the sink.
    spec.dataType = DataType::BFLOAT16;
    spec.sinkRank4 = false;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.seqLenQ = 300;
    spec.seqLenKv = 64;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);

    // A decode step over a long cache, split across workgroups: the merge adds the sink.
    spec.seqLenQ = 1;
    spec.seqLenKv = 20000;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, AddsASinkFarAboveTheScores)
{
    // A sink 200 above scores of at most 8: exp(sink - max) alone overflows f32 (past ~88.7),
    // so the sink must join the row's max. PyTorch and the RDNA kernels give O ~ 0 and
    // LSE = the sink (vendored-kernel modification 29).
    auto spec = cdnaSpec();
    spec.headDim = 64;
    spec.heads = 16;
    spec.kvHeads = 2;
    spec.withSink = true;
    spec.sinkValue = 200.0F;
    spec.stats = SdpaStats::LSE;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.seqLenQ = 300;
    spec.seqLenKv = 300;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    // Non-causal, f16.
    spec.mask = SdpaMask::NONE;
    spec.dataType = DataType::HALF;
    spec.seqLenQ = 129;
    spec.seqLenKv = 333;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    // Rows no key reaches next to rows that reach keys: LSE = the sink for both.
    spec.dataType = DataType::BFLOAT16;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.seqLenQ = 300;
    spec.seqLenKv = 64;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    // A decode step split across workgroups: the merge adds the sink against the max.
    spec.seqLenQ = 1;
    spec.seqLenKv = 20000;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, AddsASinkOfMinusInfinityToRowsNoKeyReaches)
{
    // A sink logit of -inf contributes nothing: rows with keys are plain softmax, and a row
    // no key reaches (bottom-right, Sq > Skv) gets O = 0 and LSE = -inf, not NaN -- as the
    // RDNA and aiw kernels give for the same graph.
    auto spec = cdnaSpec();
    spec.headDim = 64;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 300;
    spec.seqLenKv = 64;
    spec.withSink = true;
    spec.sinkNegativeInfinity = true;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    // Non-causal: every row reaches keys.
    spec.mask = SdpaMask::NONE;
    spec.seqLenQ = 129;
    spec.seqLenKv = 333;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    // A decode step split across workgroups: the merge adds the sink against the max.
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.seqLenQ = 1;
    spec.seqLenKv = 20000;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, KeepsANanRowNan)
{
    // NaN in K at one key of one split of a decode over a long cache: every row that sees it
    // has O = NaN and LSE = NaN, as PyTorch's do. A split whose sum is NaN must not be stored
    // as an empty one, which the merge would skip, leaving a finite answer from the other
    // splits (vendored-kernel modification 29).
    auto spec = cdnaSpec();
    spec.headDim = 64;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.stats = SdpaStats::LSE;
    spec.seqLenQ = 1;
    spec.seqLenKv = 20000;
    spec.keyNanPosition = 5;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    // With a sink, which the merge adds.
    spec.withSink = true;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    spec.withSink = false;
    // Bottom-right with four queries and the NaN key near the end: only the last two rows see
    // it, and the first two stay finite.
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.seqLenQ = 4;
    spec.keyNanPosition = spec.seqLenKv - 2;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);

    // NaN in one query row of a prefill: that row's O and LSE are NaN, every other row finite.
    spec.keyNanPosition.reset();
    spec.queryNanPosition = 37;
    spec.seqLenQ = 300;
    spec.seqLenKv = 300;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    spec.mask = SdpaMask::NONE;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ServesHalfAtHeadDim128)
{
    auto spec = cdnaSpec();
    spec.dataType = DataType::HALF;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ServesHeadDim256WhereShipped)
{
    auto spec = cdnaSpec();
    spec.headDim = 256;
    spec.seqLenQ = 200;
    spec.seqLenKv = 200;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.dataType = DataType::HALF;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

// K in BSHD and V in BHSD: the masked objects read V through its own strides.
TEST(TestGpuFlydslSdpaCdnaDispatch, ReadsVThroughItsOwnLayout)
{
    auto spec = cdnaSpec();
    spec.valueLayout = SdpaLayout::BHSD;
    spec.seqLenQ = 200;
    spec.seqLenKv = 333;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    spec.headDim = 64;
    spec.mask = SdpaMask::NONE;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

// An LSE laid out [B, Sq, H], from the generic kernel and from the split decode's merge.
TEST(TestGpuFlydslSdpaCdnaDispatch, WritesTheLseThroughItsOwnStrides)
{
    auto spec = cdnaSpec();
    spec.batch = 2;
    spec.seqLenQ = 200;
    spec.seqLenKv = 333;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    spec.statsSequenceMajor = true;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    spec.headDim = 64;
    spec.seqLenQ = 1;
    spec.seqLenKv = 4096;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

// A bias in a sliding window at head dims where gfx950's default is the dual-wave kernel,
// which takes no window: the generic kernel's bias objects serve it.
TEST(TestGpuFlydslSdpaCdnaDispatch, AddsABiasInsideASlidingWindow)
{
    auto spec = cdnaSpec();
    // A packed bias's rows move as 16-byte vectors: a key length that is a multiple of 8.
    spec.seqLenQ = 320;
    spec.seqLenKv = 320;
    spec.mask = SdpaMask::SLIDING_WINDOW;
    spec.stats = SdpaStats::LSE;
    spec.withBias = true;
    spec.biasDataType = spec.dataType;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    spec.headDim = 64;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

// The runtime scale is read from host memory at launch; one the kernels cannot take (not
// positive, not finite, or past the shared 2^20 bound) is refused there. Every operand is
// bound to real device memory, so the scale is the only thing a launch can refuse, and a
// servable scale with the same buffers launches.
TEST(TestGpuFlydslSdpaCdnaDispatch, RefusesANonPositiveRuntimeScaleAtLaunch)
{
    auto spec = cdnaSpec();
    spec.scaleKind = SdpaScale::RUNTIME_TENSOR;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec), packedArchDeviceProperties());
    const auto bound = matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());
    const auto candidates = stagedCdnaCandidates(fixture.context(), *bound);
    if(candidates.empty())
    {
        reportUnstaged("sdpa_cdna", "no CDNA object staged for bf16 non-causal at head_dim 128");
        return;
    }
    const auto& kernel = candidates.front();
    const auto& handler = dispatchHandler(cdnaEngineFor(kernel));
    const auto prepared = handler.prepare(fixture.context(), *bound, kernel);
    ASSERT_NE(prepared, nullptr);

    // Q, K, V and O as the spec lays them out (BSHD), and a split object's workspace.
    const auto bytesOf = [&spec](int64_t heads, int64_t sequence) {
        return static_cast<size_t>(spec.batch * heads * sequence * spec.headDim) * sizeof(uint16_t);
    };
    const std::vector<std::pair<int64_t, size_t>> operands{
        {FLYDSL_SDPA_RDNA_Q_UID, bytesOf(spec.heads, spec.seqLenQ)},
        {FLYDSL_SDPA_RDNA_K_UID, bytesOf(spec.kvHeads, spec.seqLenKv)},
        {FLYDSL_SDPA_RDNA_V_UID, bytesOf(spec.kvHeads, spec.seqLenKv)},
        {FLYDSL_SDPA_RDNA_O_UID, bytesOf(spec.heads, spec.seqLenQ)}};
    std::vector<void*> allocations;
    std::vector<hipdnnPluginDeviceBuffer_t> operandBuffers;
    for(const auto& [uid, bytes] : operands)
    {
        void* device = nullptr;
        ASSERT_EQ(hipSuccess, hipMalloc(&device, bytes));
        allocations.push_back(device);
        // Zeros: finite in bf16, so the control launch reads well-formed values.
        ASSERT_EQ(hipSuccess, hipMemset(device, 0, bytes));
        operandBuffers.push_back({uid, device});
    }
    const auto workspaceBytes = handler.workspaceBytes(fixture.context(), *bound, kernel);
    void* workspace = nullptr;
    if(workspaceBytes > 0)
    {
        ASSERT_EQ(hipSuccess, hipMalloc(&workspace, workspaceBytes));
        allocations.push_back(workspace);
    }
    const auto buffersWith = [&operandBuffers](float& scale) {
        auto buffers = operandBuffers;
        buffers.push_back({FLYDSL_SDPA_RDNA_SCALE_UID, &scale});
        return buffers;
    };

    const Handle handle;
    for(float scale : {0.0F,
                       -0.125F,
                       std::numeric_limits<float>::quiet_NaN(),
                       std::numeric_limits<float>::infinity(),
                       2097152.0F /* 2^21, past the 2^20 bound */})
    {
        const auto buffers = buffersWith(scale);
        EXPECT_THROW(handler.launch(handle,
                                    *prepared,
                                    buffers.data(),
                                    static_cast<uint32_t>(buffers.size()),
                                    workspace),
                     hipdnn_plugin_sdk::HipdnnPluginException)
            << scale;
    }

    // The control: with the same buffers, a servable scale launches.
    float servable = 0.125F;
    const auto buffers = buffersWith(servable);
    EXPECT_NO_THROW(handler.launch(
        handle, *prepared, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace));
    EXPECT_EQ(hipSuccess, hipDeviceSynchronize());
    for(void* allocation : allocations)
    {
        static_cast<void>(hipFree(allocation));
    }
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ServesBhsdLayout)
{
    auto spec = cdnaSpec();
    spec.layout = SdpaLayout::BHSD;
    spec.seqLenQ = 200;
    spec.seqLenKv = 333;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    spec.headDim = 64;
    spec.dataType = DataType::HALF;
    spec.mask = SdpaMask::NONE;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ServesPackedQkvViews)
{
    auto spec = cdnaSpec();
    spec.kvHeads = 32;
    spec.packedQkv = true;
    spec.seqLenQ = 300;
    spec.seqLenKv = 300;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, AddsABiasPerHead)
{
    auto spec = cdnaSpec();
    spec.withBias = true;
    spec.biasDataType = DataType::BFLOAT16;
    spec.batch = 2;
    spec.biasDims = {2, 32, 256, 256};
    spec.stats = SdpaStats::LSE;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, AddsABiasAtEveryHeadDim)
{
    // The generic kernel's bias (gfx942, and gfx950 at 256), and the dual-wave one's.
    for(const int64_t headDim : {int64_t{64}, int64_t{128}, int64_t{256}})
    {
        SCOPED_TRACE("head_dim " + std::to_string(headDim));
        auto spec = cdnaSpec();
        spec.headDim = headDim;
        spec.withBias = true;
        spec.biasDataType = DataType::BFLOAT16;
        spec.seqLenQ = 200;
        spec.seqLenKv = 264;
        spec.biasDims = {1, 32, 200, 264};
        expectCdnaMatchesReference(spec, BF16_TOLERANCE);
        spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
        spec.dataType = DataType::HALF;
        spec.biasDataType = DataType::HALF;
        spec.biasDims = {1, 1, 1, 264};
        expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    }
}

TEST(TestGpuFlydslSdpaCdnaDispatch, AddsABiasThatMasksWholeTilesWithNegativeInfinity)
{
    // A boolean key-padding mask cast to the bias: the first 128 keys of every row are -inf,
    // a whole KV tile at every block size, and every third row is masked on every key, so
    // it reaches no key (O 0, LSE -inf). Non-causal, where nothing else floors the running
    // max (vendored-kernel modification 28); at the default scale and at 1/sqrt(d). Skv 264
    // ends in a part tile, so the kernel's own key-padding mask meets the bias's -inf: under
    // fast math a -inf mask value was folded out of the bias build's exponent.
    for(const int64_t headDim : {int64_t{64}, int64_t{128}})
    {
        for(const int64_t seqLenKv : {int64_t{384}, int64_t{264}})
        {
            for(const bool scaled : {false, true})
            {
                SCOPED_TRACE("head_dim " + std::to_string(headDim) + ", Skv "
                             + std::to_string(seqLenKv) + (scaled ? ", scaled" : ""));
                auto spec = cdnaSpec();
                spec.headDim = headDim;
                spec.withBias = true;
                spec.biasDataType = spec.dataType;
                spec.seqLenQ = 256;
                spec.seqLenKv = seqLenKv;
                spec.stats = SdpaStats::LSE;
                spec.biasNegativeInfinityLeadingKeys = 128;
                spec.biasNegativeInfinityRows = true;
                if(scaled)
                {
                    spec.scaleKind = SdpaScale::VALUE;
                    spec.scale = 1.0F / std::sqrt(static_cast<float>(headDim));
                }
                expectCdnaMatchesReference(spec, BF16_TOLERANCE);
            }
        }
    }
}

/// A padding mask's fully masked rows, as Hugging Face writes them: every third row's bias is
/// the query dtype's lowest finite value on every key, so its scores are finite and equal and
/// its softmax is uniform over the keys the causal mask leaves (O the mean of V, LSE that score
/// plus log(count)), as PyTorch and CpuFpReferenceSdpa give (vendored-kernel modification 28).
/// With the scale, the first 128 keys of every row are also -inf, leaving such a row the rest,
/// and in a causal row before them no key at all. The dual-wave kernel's bias (gfx950 at 64
/// and 128) and the generic one's. Skv 264 ends in a part tile, so the key-padding mask's keys
/// sit next to the lowest rows' keys.
void expectLowestFiniteBiasRows(int64_t headDim, int64_t seqLenKv, bool causal, bool scaled)
{
    SCOPED_TRACE("head_dim " + std::to_string(headDim) + ", Skv " + std::to_string(seqLenKv)
                 + (causal ? ", causal" : "") + (scaled ? ", scaled" : ""));
    auto spec = cdnaSpec();
    spec.headDim = headDim;
    spec.withBias = true;
    spec.biasDataType = spec.dataType;
    spec.biasMasksRows = true;
    spec.seqLenQ = 256;
    spec.seqLenKv = seqLenKv;
    spec.stats = SdpaStats::LSE;
    if(causal)
    {
        spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    }
    if(scaled)
    {
        spec.scaleKind = SdpaScale::VALUE;
        spec.scale = 1.0F / std::sqrt(static_cast<float>(headDim));
        spec.biasNegativeInfinityLeadingKeys = 128;
    }
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, AddsABiasThatMasksWholeRowsAtTheLowestFiniteValue)
{
    // The configuration that reaches every behavior at once: a part tile, the causal rows
    // that see no key, and the -inf keys beside the lowest rows. Every combination is in
    // Full/TestGpuFlydslSdpaCdnaFullTier.
    expectLowestFiniteBiasRows(128, 264, true, true);
}

TEST_P(TestGpuFlydslSdpaCdnaFullTier, AddsABiasThatMasksWholeRowsAtTheLowestFiniteValue)
{
    for(const int64_t headDim : {int64_t{64}, int64_t{128}, int64_t{256}})
    {
        for(const int64_t seqLenKv : {int64_t{384}, int64_t{264}})
        {
            for(const bool causal : {false, true})
            {
                for(const bool scaled : {false, true})
                {
                    expectLowestFiniteBiasRows(headDim, seqLenKv, causal, scaled);
                }
            }
        }
    }
}

TEST(TestGpuFlydslSdpaCdnaDispatch, AddsAPaddingMaskWithBottomRightCausalCrossAttention)
{
    auto spec = cdnaSpec();
    spec.withBias = true;
    spec.batch = 2;
    spec.seqLenQ = 200;
    spec.seqLenKv = 328;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.headDim = 64;
    spec.dataType = DataType::HALF;
    spec.biasDataType = DataType::HALF;
    spec.biasDims = {2, 1, 1, 328};
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    // The mask itself: the first 40 keys of every row -inf, part of a KV tile and part of
    // each row the causal mask leaves live.
    spec.biasNegativeInfinityLeadingKeys = 40;
    spec.stats = SdpaStats::LSE;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, AddsABiasTheVectorizedObjectsCannotRead)
{
    // The strided bias objects (vendored-kernel modification 31): an f32 bias, a bias
    // broadcast along the keys, rows that are not 16-byte aligned, a key length that is not a
    // multiple of 4 under a padded pitch, rows that overlap, and a bias expanded with stride
    // 0. Each at the bucket widths' exact head dims, non-causal and causal, with the LSE.
    for(const int64_t headDim : {int64_t{64}, int64_t{128}, int64_t{256}})
    {
        for(const bool causal : {false, true})
        {
            auto spec = cdnaSpec();
            spec.headDim = headDim;
            spec.withBias = true;
            spec.biasDataType = DataType::BFLOAT16;
            spec.seqLenQ = 200;
            spec.seqLenKv = 264;
            spec.stats = SdpaStats::LSE;
            if(causal)
            {
                spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
            }
            const std::string where
                = "head_dim " + std::to_string(headDim) + (causal ? ", causal" : "");
            {
                SCOPED_TRACE(where + ", f32 per head");
                auto f32 = spec;
                f32.biasDataType = DataType::FLOAT;
                f32.biasDims = {1, 32, 200, 264};
                expectCdnaMatchesReference(f32, BF16_TOLERANCE);
            }
            {
                SCOPED_TRACE(where + ", broadcast along the keys, f32 and bf16");
                auto keys = spec;
                keys.batch = 2;
                keys.biasDims = {2, 1, 200, 1};
                expectCdnaMatchesReference(keys, BF16_TOLERANCE);
                keys.biasDataType = DataType::FLOAT;
                expectCdnaMatchesReference(keys, BF16_TOLERANCE);
            }
            {
                SCOPED_TRACE(where + ", rows of 333 keys, packed");
                auto packed = spec;
                packed.seqLenKv = 333;
                packed.biasDims = {1, 1, 200, 333};
                expectCdnaMatchesReference(packed, BF16_TOLERANCE);
            }
            {
                SCOPED_TRACE(where + ", 334 keys at a pitch of 336, and rows that overlap");
                auto padded = spec;
                padded.seqLenKv = 334;
                padded.biasDims = {1, 1, 200, 334};
                padded.biasStrides = {67200, 67200, 336, 1};
                expectCdnaMatchesReference(padded, BF16_TOLERANCE);
                padded.biasStrides = {67200, 67200, 104, 1};
                expectCdnaMatchesReference(padded, BF16_TOLERANCE);
            }
            {
                SCOPED_TRACE(where + ", expanded over batch, heads and rows with stride 0");
                auto expanded = spec;
                expanded.batch = 2;
                expanded.biasDims = {2, 32, 200, 264};
                expanded.biasStrides = {264, 0, 0, 1};
                expectCdnaMatchesReference(expanded, BF16_TOLERANCE);
            }
        }
    }
}

TEST(TestGpuFlydslSdpaCdnaDispatch, AddsAnF32BiasThatMasksWholeRowsAtItsLowestFiniteValue)
{
    // An f32 padding mask: every third row's bias is f32's lowest finite value (-FLT_MAX) on
    // every key, and the first 128 keys of every row -inf (every third row then reaches no key
    // in the causal case). Its scores sit at -FLT_MAX, the masked-score sentinel's value; the
    // strided objects keep their scores halved so it stays a score (modification 31): O the
    // mean of V over the live keys, LSE -FLT_MAX, as PyTorch and the RDNA kernels give.
    for(const int64_t headDim : {int64_t{64}, int64_t{128}, int64_t{256}})
    {
        for(const bool causal : {false, true})
        {
            SCOPED_TRACE("head_dim " + std::to_string(headDim) + (causal ? ", causal" : ""));
            auto spec = cdnaSpec();
            spec.headDim = headDim;
            spec.withBias = true;
            spec.biasDataType = DataType::FLOAT;
            spec.biasMasksRows = true;
            spec.seqLenQ = 256;
            spec.seqLenKv = 264;
            spec.stats = SdpaStats::LSE;
            spec.scaleKind = SdpaScale::VALUE;
            spec.scale = 1.0F / std::sqrt(static_cast<float>(headDim));
            if(causal)
            {
                spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
            }
            expectCdnaMatchesReference(spec, BF16_TOLERANCE);
            spec.biasNegativeInfinityLeadingKeys = 128;
            expectCdnaMatchesReference(spec, BF16_TOLERANCE);
        }
    }
}

TEST(TestGpuFlydslSdpaCdnaDispatch, AddsABiasAtTheBucketHeadDims)
{
    // Head dims no exact object serves, on the bucket bias objects: on gfx90a 96 and 160 have
    // buckets of their own, elsewhere they take the next power of two's.
    for(const int64_t headDim : {int64_t{72}, int64_t{96}, int64_t{160}, int64_t{200}})
    {
        for(const bool causal : {false, true})
        {
            SCOPED_TRACE("head_dim " + std::to_string(headDim) + (causal ? ", causal" : ""));
            auto spec = cdnaSpec();
            spec.headDim = headDim;
            spec.withBias = true;
            spec.biasDataType = DataType::BFLOAT16;
            spec.seqLenQ = 200;
            spec.seqLenKv = 264;
            spec.biasDims = {1, 32, 200, 264};
            spec.stats = SdpaStats::LSE;
            spec.biasNegativeInfinityRows = true;
            if(causal)
            {
                spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
            }
            expectCdnaMatchesReference(spec, BF16_TOLERANCE);
            spec.dataType = DataType::HALF;
            spec.biasDataType = DataType::HALF;
            spec.biasDims = {1, 1, 1, 264};
            spec.biasNegativeInfinityRows = false;
            expectCdnaMatchesReference(spec, BF16_TOLERANCE);
        }
    }
}

TEST(TestGpuFlydslSdpaCdnaDispatch, AddsABiasAndASink)
{
    // A bias and a sink together (the sink objects' bias builds): a per-head bias, a padding
    // mask whose rows are at the lowest finite value, -inf leading keys that leave causal
    // rows keyless (O 0, LSE the sink), and a sink far above the scores (LSE the sink).
    for(const int64_t headDim : {int64_t{64}, int64_t{128}})
    {
        for(const bool causal : {false, true})
        {
            SCOPED_TRACE("head_dim " + std::to_string(headDim) + (causal ? ", causal" : ""));
            auto spec = cdnaSpec();
            spec.headDim = headDim;
            spec.withBias = true;
            spec.biasDataType = DataType::BFLOAT16;
            spec.withSink = true;
            spec.seqLenQ = 200;
            spec.seqLenKv = 264;
            spec.biasDims = {1, 32, 200, 264};
            spec.stats = SdpaStats::LSE;
            if(causal)
            {
                spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
            }
            expectCdnaMatchesReference(spec, BF16_TOLERANCE);
            spec.biasMasksRows = true;
            spec.biasNegativeInfinityLeadingKeys = 128;
            expectCdnaMatchesReference(spec, BF16_TOLERANCE);
            spec.biasMasksRows = false;
            spec.biasNegativeInfinityLeadingKeys = 0;
            spec.sinkValue = 200.0F;
            expectCdnaMatchesReference(spec, BF16_TOLERANCE);
        }
    }
}

TEST(TestGpuFlydslSdpaCdnaDispatch, AddsABiasTheVectorizedObjectsCannotReadWithASink)
{
    // f32 and key-broadcast biases with a sink, on the strided sink-and-bias objects
    // (COVERAGE.md section 7.5 item 16): rows at f32's lowest finite value average V over
    // their keys with the sink in the denominator; -inf leading keys leave causal rows keyless
    // (O 0, LSE the sink); and a sink far above the scores (LSE the sink).
    for(const int64_t headDim : {int64_t{64}, int64_t{128}})
    {
        for(const bool causal : {false, true})
        {
            SCOPED_TRACE("head_dim " + std::to_string(headDim) + (causal ? ", causal" : ""));
            auto spec = cdnaSpec();
            spec.headDim = headDim;
            spec.withBias = true;
            spec.biasDataType = DataType::FLOAT;
            spec.withSink = true;
            spec.seqLenQ = 200;
            spec.seqLenKv = 261;
            spec.biasDims = {1, 32, 200, 261};
            spec.stats = SdpaStats::LSE;
            if(causal)
            {
                spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
            }
            expectCdnaMatchesReference(spec, BF16_TOLERANCE);
            spec.biasMasksRows = true;
            spec.biasNegativeInfinityLeadingKeys = 128;
            expectCdnaMatchesReference(spec, BF16_TOLERANCE);
            spec.biasMasksRows = false;
            spec.biasNegativeInfinityLeadingKeys = 0;
            spec.sinkValue = 200.0F;
            expectCdnaMatchesReference(spec, BF16_TOLERANCE);
            // Broadcast along the keys, in the query's dtype.
            spec.sinkValue.reset();
            spec.biasDataType = DataType::BFLOAT16;
            spec.biasDims = {1, 32, 200, 1};
            expectCdnaMatchesReference(spec, BF16_TOLERANCE);
        }
    }
}

TEST_P(TestGpuFlydslSdpaCdnaFullTier, LongerSequenceMatchesTheReference)
{
    auto spec = cdnaSpec();
    spec.seqLenQ = 1024;
    spec.seqLenKv = 1024;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

// Padding (vendored-kernel modification 32): each batch's own lengths, rows past a batch's
// query length O = 0 and LSE = -inf, keys past its key length never read. Every staged object
// the matchers admit runs, the generic prefill and split decode objects alike.

/// A padded GQA graph over three batches: whole, cut inside a query tile, and empty.
SdpaGraphSpec paddedGpuSpec()
{
    auto spec = cdnaSpec();
    spec.batch = 3;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 300;
    spec.seqLenKv = 300;
    spec.stats = SdpaStats::LSE;
    setPadding(spec, {300, 129, 0}, {300, 63, 200});
    return spec;
}

TEST(TestGpuFlydslSdpaCdnaDispatch, PadsEachBatchToItsOwnLengths)
{
    expectCdnaMatchesReference(paddedGpuSpec(), BF16_TOLERANCE);
    // In f16, and BHSD.
    auto spec = paddedGpuSpec();
    spec.dataType = DataType::HALF;
    spec.layout = SdpaLayout::BHSD;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, GivesABatchWithNoKeysRowsNoKeyReaches)
{
    // With no mask and no bias a row's sum is never 0 dense; a padded batch with no keys
    // makes it 0, and its rows must come out O = 0, LSE = -inf, not 0 * inf.
    auto spec = paddedGpuSpec();
    spec.seqLensQ = {300, 17, 300};
    spec.seqLensKv = {0, 300, 1};
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, PadsEveryMaskAtEachBatchsOwnDiagonal)
{
    // A batch with more queries than keys and one with fewer, at both corners; a bottom-right
    // diagonal sits at each batch's own Skv_b - Sq_b, and so does its window.
    auto spec = paddedGpuSpec();
    spec.seqLensQ = {200, 64, 300};
    spec.seqLensKv = {100, 256, 300};
    for(const auto mask : {SdpaMask::TOP_LEFT_CAUSAL, SdpaMask::BOTTOM_RIGHT_CAUSAL})
    {
        SCOPED_TRACE(mask == SdpaMask::TOP_LEFT_CAUSAL ? "top-left" : "bottom-right");
        spec.mask = mask;
        spec.leftBound.reset();
        expectCdnaMatchesReference(spec, BF16_TOLERANCE);
        spec.leftBound = 40;
        expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    }
    // A band right of the diagonal.
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.leftBound.reset();
    spec.rightBound = 50;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, PadsWithABiasAndASink)
{
    // A padded row's LSE is -inf even beside a sink; a row of a batch with no keys is keyless,
    // so its LSE is the sink.
    auto spec = paddedGpuSpec();
    spec.seqLensKv = {300, 0, 200};
    spec.seqLensQ = {300, 129, 100};
    spec.withSink = true;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    spec.withSink = false;
    spec.withBias = true;
    spec.biasDataType = spec.dataType;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    // An f32 bias, which the objects that read it one key at a time serve (modification 31).
    spec.biasDataType = DataType::FLOAT;
    spec.withSink = false;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    // A bias the vectorized objects read (a row pitch of 296), over batches whose key lengths
    // are no multiple of 4 and that have more queries than keys: the last real row's final keys
    // keep their bias on the vectorized objects (vendored-kernel modification 37) and on the
    // strided ones. With a sink too.
    spec.seqLenKv = 296;
    spec.biasDataType = spec.dataType;
    spec.seqLensQ = {300, 129, 7};
    spec.seqLensKv = {63, 61, 296};
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    spec.headDim = 64;
    spec.withSink = true;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, PadsWithABiasAtEveryKeyLengthModuloTheVectorRuns)
{
    // Under padding the vectorized bias objects read the last real row's bias up to the
    // launch's key length (vendored-kernel modification 37). Before it that row's final 4-key
    // run straddled the batch's bias view and read as zero whenever the batch's key length was
    // no multiple of 4. Here: every key length mod 4 and mod 8, more queries than keys, keys
    // across a KV tile, a bias of its own per batch that is NaN past each batch's keys (masked
    // whatever it holds there), and rows at the lowest finite value, -inf and NaN. Every staged
    // object the matcher admits runs (vectorized and strided), against the harness's reference
    // and, but for NaN rows, hipDNN's CPU reference.
    auto spec = cdnaSpec();
    spec.batch = 4;
    spec.heads = 4;
    spec.kvHeads = 2;
    spec.seqLenQ = 160;
    spec.seqLenKv = 296;
    spec.stats = SdpaStats::LSE;
    spec.withBias = true;
    spec.biasDataType = spec.dataType;
    spec.biasDims = {4, 4, 160, 296};
    spec.biasNanPastKeys = true;
    struct Lengths
    {
        std::vector<int32_t> q;
        std::vector<int32_t> kv;
    };
    const std::array<Lengths, 3> lengthSets{Lengths{{129, 70, 160, 3}, {57, 58, 59, 60}},
                                            Lengths{{129, 70, 160, 3}, {61, 62, 63, 64}},
                                            Lengths{{160, 1, 100, 159}, {293, 294, 295, 133}}};
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
                expectCdnaMatchesReference(spec, BF16_TOLERANCE);
            }
        }
    }
    auto edge = spec;
    edge.headDim = 128;
    setPadding(edge, lengthSets[1].q, lengthSets[1].kv);
    for(const auto mask : {SdpaMask::TOP_LEFT_CAUSAL, SdpaMask::BOTTOM_RIGHT_CAUSAL})
    {
        edge.mask = mask;
        {
            SCOPED_TRACE("rows at the lowest finite value, -inf leading keys");
            auto rows = edge;
            rows.biasMasksRows = true;
            rows.biasNegativeInfinityLeadingKeys = 40;
            expectCdnaMatchesReference(rows, BF16_TOLERANCE);
        }
        {
            SCOPED_TRACE("-inf rows");
            auto rows = edge;
            rows.biasNegativeInfinityRows = true;
            expectCdnaMatchesReference(rows, BF16_TOLERANCE);
        }
        {
            SCOPED_TRACE("NaN rows");
            auto rows = edge;
            rows.biasNanRows = true;
            expectCdnaMatchesReference(rows, BF16_TOLERANCE);
        }
    }
    {
        // The bucket objects, f16, and a sink (the vectorized sink-and-bias objects).
        SCOPED_TRACE("head_dim 80, f16");
        auto bucket = edge;
        bucket.headDim = 80;
        bucket.dataType = DataType::HALF;
        bucket.biasDataType = DataType::HALF;
        expectCdnaMatchesReference(bucket, BF16_TOLERANCE);
        SCOPED_TRACE("a sink");
        auto sink = edge;
        sink.headDim = 64;
        sink.withSink = true;
        sink.biasMasksRows = true;
        expectCdnaMatchesReference(sink, BF16_TOLERANCE);
        SCOPED_TRACE("head_dim 256");
        auto wide = edge;
        wide.headDim = 256;
        expectCdnaMatchesReference(wide, BF16_TOLERANCE);
    }
}

TEST(TestGpuFlydslSdpaCdnaDispatch, AddsAFloatSink)
{
    // A FLOAT sink, cuDNN's and hipDNN's reference's dtype (vendored-kernel modification 36),
    // against the harness's reference and hipDNN's CPU reference: rank 1 and rank 4, rows no
    // key reaches (LSE = the sink's f32 bits), a sink far above the scores, a sink of -inf,
    // decode split across workgroups (the merge reads it), a bias the vectorized objects read
    // and an f32 one only the strided objects do, and padding.
    auto spec = cdnaSpec();
    spec.headDim = 64;
    spec.heads = 16;
    spec.kvHeads = 2;
    spec.withSink = true;
    spec.sinkDataType = DataType::FLOAT;
    spec.stats = SdpaStats::LSE;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.seqLenQ = 300;
    spec.seqLenKv = 300;
    {
        SCOPED_TRACE("rank 1");
        expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    }
    {
        SCOPED_TRACE("rank 4, f16, head_dim 128, rows no key reaches");
        auto rank4 = spec;
        rank4.sinkRank4 = true;
        rank4.dataType = DataType::HALF;
        rank4.headDim = 128;
        rank4.seqLenKv = 64;
        expectCdnaMatchesReference(rank4, BF16_TOLERANCE);
        SCOPED_TRACE("a sink of -inf");
        rank4.sinkNegativeInfinity = true;
        expectCdnaMatchesReference(rank4, BF16_TOLERANCE);
    }
    {
        // 200 + 2^-7: bf16 and f16 both round it to 200, past the LSE's tolerance.
        SCOPED_TRACE("a sink far above the scores");
        auto far = spec;
        far.sinkValue = 200.0078125F;
        expectCdnaMatchesReference(far, BF16_TOLERANCE);
    }
    {
        SCOPED_TRACE("decode split across workgroups");
        auto step = spec;
        step.seqLenQ = 1;
        step.seqLenKv = 20000;
        expectCdnaMatchesReference(step, BF16_TOLERANCE);
    }
    for(const int64_t headDim : {int64_t{64}, int64_t{128}})
    {
        SCOPED_TRACE("a bias and a sink, head_dim " + std::to_string(headDim));
        auto biased = cdnaSpec();
        biased.headDim = headDim;
        biased.withBias = true;
        biased.biasDataType = DataType::BFLOAT16;
        biased.withSink = true;
        biased.sinkDataType = DataType::FLOAT;
        biased.seqLenQ = 200;
        biased.seqLenKv = 264;
        biased.biasDims = {1, 32, 200, 264};
        biased.stats = SdpaStats::LSE;
        biased.mask = SdpaMask::TOP_LEFT_CAUSAL;
        biased.biasMasksRows = true;
        biased.biasNegativeInfinityLeadingKeys = 128;
        expectCdnaMatchesReference(biased, BF16_TOLERANCE);
        // An f32 bias: the strided sink-and-bias objects (modification 31).
        biased.biasDataType = DataType::FLOAT;
        biased.biasMasksRows = false;
        biased.biasNegativeInfinityLeadingKeys = 0;
        biased.sinkValue = 1.0e30F;
        expectCdnaMatchesReference(biased, BF16_TOLERANCE);
    }
    {
        // A padded row's LSE is -inf beside a sink; a batch with no keys is keyless, LSE the sink.
        SCOPED_TRACE("padding");
        auto padded = paddedGpuSpec();
        padded.seqLensKv = {300, 0, 200};
        padded.seqLensQ = {300, 129, 100};
        padded.withSink = true;
        padded.sinkDataType = DataType::FLOAT;
        expectCdnaMatchesReference(padded, BF16_TOLERANCE);
    }
}

TEST(TestGpuFlydslSdpaCdnaDispatch, PadsAtABucketHeadDim)
{
    auto spec = paddedGpuSpec();
    spec.headDim = 80;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, DecodesPaddedBatchesWithSplitKeys)
{
    // Batched decode over caches of different lengths, the main inference use of padding: a
    // long one (many splits, so the merge runs), a one-key one, an empty one, and a batch
    // with no query.
    auto spec = cdnaSpec();
    spec.batch = 4;
    spec.seqLenQ = 1;
    spec.seqLenKv = 4096;
    spec.stats = SdpaStats::LSE;
    setPadding(spec, {1, 1, 0, 1}, {4096, 1, 300, 0});
    for(const auto mask : {SdpaMask::NONE, SdpaMask::BOTTOM_RIGHT_CAUSAL})
    {
        spec.mask = mask;
        expectCdnaMatchesReference(spec, BF16_TOLERANCE, FLYDSL_SDPA_CDNA_DECODE);
    }
    // A multi-token step, and a sink.
    spec.seqLenQ = 5;
    spec.seqLensQ = {5, 2, 0, 5};
    spec.withSink = true;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE, FLYDSL_SDPA_CDNA_DECODE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ClampsLengthsOutsideTheSequence)
{
    // As hipDNN's GPU reference clamps them: to [0, S].
    auto spec = paddedGpuSpec();
    spec.seqLensQ = {1000, -5, 150};
    spec.seqLensKv = {-1, 999, 150};
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, KeepsANanRowNanBesidePaddedRows)
{
    // A NaN row in a query tile that also holds padded rows: the NaN row's O and LSE stay
    // NaN and the padded rows are O = 0, LSE = -inf (the padded rows' stores must not fold
    // into the no-keys test, which fast math would make NaN-blind), at both corners and
    // with no mask, and in a decode whose merge fills a padded batch.
    auto spec = paddedGpuSpec();
    spec.seqLensQ = {300, 140, 0};
    spec.queryNanPosition = 130;
    for(const auto mask :
        {SdpaMask::NONE, SdpaMask::TOP_LEFT_CAUSAL, SdpaMask::BOTTOM_RIGHT_CAUSAL})
    {
        spec.mask = mask;
        expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    }
    spec.withSink = true;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);

    auto decode = cdnaSpec();
    decode.batch = 3;
    decode.seqLenQ = 1;
    decode.seqLenKv = 4096;
    decode.stats = SdpaStats::LSE;
    decode.keyNanPosition = 5;
    setPadding(decode, {1, 0, 1}, {4096, 4096, 3});
    expectCdnaMatchesReference(decode, BF16_TOLERANCE, FLYDSL_SDPA_CDNA_DECODE);
}

// Dropout (vendored-kernel modification 33): the mask AOTriton (>= 0.15) and the aiw objects
// draw, bit for bit, checked against the test-local oracle (FlydslSdpaPhiloxOracle.hpp).

TEST(TestGpuFlydslSdpaCdnaDispatch, DropsTheOraclesMaskBitForBit)
{
    // The aiw objects' own cases (TestGpuFlydslSdpaAiwDispatch.DropsTheOraclesMaskBitForBit),
    // the same graphs, seed and offset: both families must draw this one mask.
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
        SCOPED_TRACE("head_dim " + std::to_string(c.headDim));
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
        expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    }
}

TEST(TestGpuFlydslSdpaCdnaDispatch, DropsWithABias)
{
    // The dropout objects' bias builds: the bias added to the scores, the mask drawn after the
    // softmax's sum (the LSE undropped, the bias in it). A per-head bias, a padding mask whose
    // rows are at the lowest finite value, -inf leading keys that leave causal rows keyless;
    // and the mask bit for bit through O (Q zero, V the identity).
    for(const int64_t headDim : {int64_t{64}, int64_t{128}, int64_t{256}})
    {
        for(const bool causal : {false, true})
        {
            SCOPED_TRACE("head_dim " + std::to_string(headDim) + (causal ? ", causal" : ""));
            auto spec = cdnaSpec();
            spec.headDim = headDim;
            spec.heads = 8;
            spec.kvHeads = 2;
            spec.seqLenQ = 200;
            spec.seqLenKv = 264;
            spec.withBias = true;
            spec.biasDataType = DataType::BFLOAT16;
            spec.biasDims = {1, 8, 200, 264};
            spec.stats = SdpaStats::LSE;
            spec.mask = causal ? SdpaMask::TOP_LEFT_CAUSAL : SdpaMask::NONE;
            setDropout(spec, 0.1F, 1234U, 77U);
            expectCdnaMatchesReference(spec, BF16_TOLERANCE);
            spec.biasMasksRows = true;
            spec.biasNegativeInfinityLeadingKeys = 128;
            expectCdnaMatchesReference(spec, BF16_TOLERANCE);
        }
    }
    auto mask = cdnaSpec();
    mask.heads = 8;
    mask.kvHeads = 2;
    mask.seqLenQ = 64;
    mask.seqLenKv = 128;
    mask.withBias = true;
    mask.biasDataType = DataType::BFLOAT16;
    mask.biasDims = {1, 1, 64, 128};
    mask.exposeDropoutMask = true;
    setDropout(mask, 0.3F, 99U, 7U);
    {
        SCOPED_TRACE("the mask through O");
        expectCdnaMatchesReference(mask, BF16_TOLERANCE);
    }
    // Under padding (vendored-kernel modification 37): key lengths no multiple of 4, a batch
    // with more queries than keys, one with no query, a bias of its own per batch that is NaN
    // past each batch's keys; the full output, then the mask bit for bit through O and the RNG
    // dump.
    auto padded = cdnaSpec();
    padded.batch = 3;
    padded.heads = 8;
    padded.kvHeads = 2;
    padded.seqLenQ = 300;
    padded.seqLenKv = 296;
    padded.withBias = true;
    padded.biasDataType = DataType::BFLOAT16;
    padded.biasDims = {3, 8, 300, 296};
    padded.biasNanPastKeys = true;
    padded.stats = SdpaStats::LSE;
    padded.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    setPadding(padded, {300, 129, 0}, {295, 63, 200});
    setDropout(padded, 0.1F, 1234U, 77U);
    {
        SCOPED_TRACE("under padding");
        expectCdnaMatchesReference(padded, BF16_TOLERANCE);
        padded.mask = SdpaMask::NONE;
        padded.headDim = 64;
        padded.biasMasksRows = true;
        expectCdnaMatchesReference(padded, BF16_TOLERANCE);
    }
    auto paddedMask = mask;
    paddedMask.batch = 3;
    paddedMask.biasDims = {3, 8, 64, 128};
    paddedMask.biasNanPastKeys = true;
    paddedMask.withRngDump = true;
    setPadding(paddedMask, {64, 33, 0}, {127, 61, 128});
    {
        SCOPED_TRACE("under padding, the mask through O and the RNG dump");
        expectCdnaMatchesReference(paddedMask, BF16_TOLERANCE);
    }
}

TEST(TestGpuFlydslSdpaCdnaDispatch, DropsWithTheUndroppedLse)
{
    // GQA, a key length that is no multiple of 4 or of the KV tile, more queries than one
    // tile, both dtypes; the LSE is the undropped one.
    auto spec = cdnaSpec();
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 300;
    spec.seqLenKv = 270;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    setDropout(spec, 0.2F, 42U, 5U);
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    spec.mask = SdpaMask::NONE;
    spec.seqLenKv = 301;
    spec.headDim = 64;
    spec.layout = SdpaLayout::BHSD;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    spec.dataType = DataType::HALF;
    spec.headDim = 256;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, DropsWithASeedAndOffsetGivenByValue)
{
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
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    // INT32 by value: widened on the host to the same counter.
    spec.dropoutSeed = 77U;
    spec.dropoutSeedDataType = DataType::INT32;
    spec.dropoutOffsetDataType = DataType::INT32;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    // One by value and one on the device.
    spec.dropoutSeedDataType = DataType::INT64;
    spec.dropoutOffsetDataType = DataType::INT64;
    spec.passByValueTensors.erase(FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID);
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, WritesTheRngDumpItApplied)
{
    // The aiw and RDNA cases' graphs, seed and offset (TestGpuFlydslSdpa*Dispatch.
    // WritesTheRngDumpItApplied): every family must write this one mask. Keys not a multiple
    // of 4, masked keys, GQA, a seed past 32 bits and an offset into the counter's high word.
    struct Case
    {
        int64_t headDim;
        SdpaMask mask;
        int64_t seqLenKv;
        float p;
    };
    for(const auto& c :
        {Case{64, SdpaMask::NONE, 70, 0.25F}, Case{128, SdpaMask::TOP_LEFT_CAUSAL, 96, 0.1F}})
    {
        SCOPED_TRACE("head_dim " + std::to_string(c.headDim));
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
        expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    }
    // Under padding: the raw bit at padded rows and at keys past a batch's length too, the
    // dense graph's mask; the seed and offset given by value.
    SdpaGraphSpec padded;
    padded.dataType = DataType::HALF;
    padded.batch = 3;
    padded.heads = 3;
    padded.kvHeads = 3;
    padded.headDim = 128;
    padded.seqLenQ = 50;
    padded.seqLenKv = 64;
    padded.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    padded.stats = SdpaStats::LSE;
    setDropoutWithRngDump(padded, 0.3F, 42U, 5U);
    setPadding(padded, {50, 17, 0}, {64, 30, 64});
    padded.passByValueTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID);
    padded.passByValueTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_UID);
    expectCdnaMatchesReference(padded, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, DropsTheDenseMaskUnderPadding)
{
    // D6: a padded batch draws its mask at the graph's declared extents, so its real rows'
    // bits are the dense graph's (read off O bit for bit, Q zero and V the identity), and
    // its padded rows are O = 0, LSE = -inf.
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.batch = 3;
    spec.heads = 3;
    spec.kvHeads = 3;
    spec.headDim = 128;
    spec.seqLenQ = 150;
    spec.seqLenKv = 128;
    spec.stats = SdpaStats::LSE;
    setDropout(spec, 0.25F, 0x0123456789ABCDEFULL, (uint64_t{1} << 32) - 7);
    setPadding(spec, {150, 77, 0}, {128, 50, 128});
    spec.exposeDropoutMask = true;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, DropsUnderPaddingWithTheUndroppedLse)
{
    // D6: the full output and the undropped LSE of a padded, dropping GQA graph, a batch
    // with no keys and one with no queries among them, and lengths without padding_mask.
    auto spec = paddedGpuSpec();
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    setDropout(spec, 0.2F, 42U, 5U);
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    spec.seqLensKv = {0, 300, 199};
    spec.mask = SdpaMask::NONE;
    spec.headDim = 64;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    spec.paddingMask = false;
    spec.dataType = DataType::HALF;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ComputesTheUndroppedResultAtProbabilityZero)
{
    auto spec = cdnaSpec();
    spec.seqLenQ = 200;
    spec.seqLenKv = 200;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    setDropout(spec, 0.0F);
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, PadsWithoutThePaddingMask)
{
    // Lengths without padding_mask are padding too (hipDNN's references' SEM-008).
    auto spec = paddedGpuSpec();
    spec.paddingMask = false;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

/// Packed (THD; vendored-kernel modification 34): sequences of lengths full, odd and 0 end
/// to end, GQA, the LSE packed alike; the harness packs the dense values token-major and
/// checks no store reaches past the last sequence.
SdpaGraphSpec packedGpuSpec()
{
    auto spec = cdnaSpec();
    spec.batch = 3;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 300;
    spec.seqLenKv = 300;
    spec.stats = SdpaStats::LSE;
    setPacked(spec, {300, 129, 0}, {300, 63, 200});
    return spec;
}

TEST(TestGpuFlydslSdpaCdnaDispatch, PacksSequencesEndToEnd)
{
    expectCdnaMatchesReference(packedGpuSpec(), BF16_TOLERANCE);
    // In f16, without the LSE, and with one table for queries and keys.
    auto spec = packedGpuSpec();
    spec.dataType = DataType::HALF;
    spec.stats = SdpaStats::NONE;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    setPacked(spec, {300, 1, 77}, {300, 1, 77});
    spec.stats = SdpaStats::LSE;
    spec.raggedOffsetOf[FLYDSL_SDPA_RDNA_K_UID] = FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID;
    spec.raggedOffsetOf[FLYDSL_SDPA_RDNA_V_UID] = FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, PacksSequencesWithNoKeysAndASink)
{
    // A sequence with no keys gives rows no key reaches (O = 0, LSE = -inf, or the sink's
    // logit beside a sink); one with no queries writes nothing.
    auto spec = packedGpuSpec();
    setPacked(spec, {300, 17, 0}, {0, 300, 1});
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    spec.withSink = true;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, PacksEveryMaskAtEachSequencesOwnDiagonal)
{
    // More queries than keys and fewer, at both corners, a window and a band: a bottom-right
    // diagonal sits at each sequence's own Skv_b - Sq_b, as under padding.
    auto spec = packedGpuSpec();
    setPacked(spec, {200, 64, 300}, {100, 256, 300});
    for(const auto mask : {SdpaMask::TOP_LEFT_CAUSAL, SdpaMask::BOTTOM_RIGHT_CAUSAL})
    {
        SCOPED_TRACE(mask == SdpaMask::TOP_LEFT_CAUSAL ? "top-left" : "bottom-right");
        spec.mask = mask;
        spec.leftBound.reset();
        expectCdnaMatchesReference(spec, BF16_TOLERANCE);
        spec.leftBound = 40;
        expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    }
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.leftBound.reset();
    spec.rightBound = 50;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, PacksAtABucketHeadDimAndASingleTokenExtent)
{
    // A bucket head dim (the qk_mask objects), and decode-shaped sequences whose sequence
    // extent is 1, where the token stride still steps the packed tokens.
    auto spec = packedGpuSpec();
    spec.headDim = 80;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    auto decode = packedGpuSpec();
    decode.batch = 4;
    decode.seqLenQ = 1;
    decode.seqLenKv = 512;
    setPacked(decode, {1, 0, 1, 1}, {512, 3, 0, 77});
    decode.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    expectCdnaMatchesReference(decode, BF16_TOLERANCE);
}

// ---------------------------------------------------------------------------
// Exact f32 (vendored-kernel modification 35): the generic prefill objects only, no bias,
// sink or dropout yet
// ---------------------------------------------------------------------------

SdpaGraphSpec cdnaF32Spec()
{
    auto spec = cdnaSpec();
    spec.dataType = DataType::FLOAT;
    return spec;
}

TEST(TestFlydslSdpaCdnaF32Graph, AcceptsWhatTheF32ObjectsServe)
{
    std::vector<CdnaCase> cases;
    cases.push_back({"NonCausal", cdnaF32Spec()});
    for(const auto mask :
        {SdpaMask::TOP_LEFT_CAUSAL, SdpaMask::BOTTOM_RIGHT_CAUSAL, SdpaMask::SLIDING_WINDOW})
    {
        auto masked = cdnaF32Spec();
        masked.mask = mask;
        cases.push_back({"Mask" + std::to_string(static_cast<int>(mask)), masked});
    }
    auto mqa = cdnaF32Spec();
    mqa.kvHeads = 1;
    cases.push_back({"Mqa", mqa});
    auto lse = cdnaF32Spec();
    lse.stats = SdpaStats::LSE;
    cases.push_back({"Lse", lse});
    auto bhsd = cdnaF32Spec();
    bhsd.layout = SdpaLayout::BHSD;
    cases.push_back({"Bhsd", bhsd});
    auto packedQkv = cdnaF32Spec();
    packedQkv.kvHeads = packedQkv.heads;
    packedQkv.packedQkv = true;
    cases.push_back({"PackedQkv", packedQkv});
    auto floatMma = cdnaF32Spec();
    floatMma.mmaCoreMode = DataType::FLOAT;
    cases.push_back({"MmaCoreModeFloat", floatMma});
    auto padded = cdnaPaddedSpec();
    padded.dataType = DataType::FLOAT;
    cases.push_back({"Padded", padded});
    auto packed = cdnaPackedSpec();
    packed.dataType = DataType::FLOAT;
    cases.push_back({"Packed", packed});
    auto packedSlots = packed;
    setPadding(packedSlots, {200, 100}, {150, 256});
    cases.push_back({"PackedWithPerBatchLengths", packedSlots});
    for(const int64_t headDim : {64, 72, 200, 256})
    {
        auto width = cdnaF32Spec();
        width.headDim = headDim;
        cases.push_back({"HeadDim" + std::to_string(headDim), width});
    }
    for(const auto& [name, spec] : cases)
    {
        EXPECT_TRUE(cdnaGraphMatch(spec).has_value()) << name;
    }
}

TEST(TestFlydslSdpaCdnaF32Graph, DeclinesWhatTheF32ObjectsDoNotServe)
{
    std::vector<CdnaCase> cases;
    auto mixed = cdnaF32Spec();
    mixed.keyDataType = DataType::BFLOAT16;
    cases.push_back({"MixedDtypes", mixed});
    for(const auto biasType : {DataType::FLOAT, DataType::BFLOAT16})
    {
        auto bias = cdnaF32Spec();
        bias.withBias = true;
        bias.biasDataType = biasType;
        cases.push_back({"Bias" + std::to_string(static_cast<int>(biasType)), bias});
    }
    auto sink = cdnaF32Spec();
    sink.withSink = true;
    sink.sinkDataType = DataType::FLOAT;
    cases.push_back({"Sink", sink});
    auto dropout = cdnaF32Spec();
    setDropout(dropout, 0.2F);
    cases.push_back({"Dropout", dropout});
    for(const auto mma : {DataType::HALF, DataType::BFLOAT16})
    {
        auto reduced = cdnaF32Spec();
        reduced.mmaCoreMode = mma;
        cases.push_back({"MmaCoreMode" + std::to_string(static_cast<int>(mma)), reduced});
    }
    // And the other way: an f32 MMA on a 16-bit graph.
    auto floatOnBf16 = cdnaSpec();
    floatOnBf16.mmaCoreMode = DataType::FLOAT;
    cases.push_back({"FloatMmaOnBf16", floatOnBf16});
    for(const auto& [name, spec] : cases)
    {
        EXPECT_FALSE(cdnaGraphMatch(spec).has_value()) << name;
    }
    // Each case's control: the same graph in bf16 (or without the f32 MMA) is served.
    auto control = cdnaF32Spec();
    setDropout(control, 0.0F);
    EXPECT_TRUE(cdnaGraphMatch(control).has_value()) << "probability 0";
}

TEST(TestFlydslSdpaCdnaF32Graph, HoldsAnF32OperandsBytesToThirtyTwoBits)
{
    // Every operand view is addressed by 32-bit byte offsets: an f32 sequence of 5M tokens at
    // head_dim 128 spans 2.56 GB, past them, where the same graph in bf16 (1.28 GB) fits.
    auto spec = cdnaSpec();
    spec.heads = 1;
    spec.kvHeads = 1;
    spec.seqLenQ = 5000000;
    spec.seqLenKv = 5000000;
    EXPECT_TRUE(cdnaGraphMatch(spec).has_value()) << "bf16";
    spec.dataType = DataType::FLOAT;
    EXPECT_FALSE(cdnaGraphMatch(spec).has_value()) << "f32";
}

TEST(TestFlydslSdpaCdnaF32KernelMatch, RunsF32OnlyOnF32GenericPrefillObjects)
{
    const auto f32 = cdnaF32Spec();
    const auto bf16 = cdnaSpec();
    EXPECT_TRUE(cdnaAdmits(f32, makeCdnaKernel("generic", 128, 0, "f32", 128)));
    EXPECT_FALSE(cdnaAdmits(f32, makeCdnaKernel("generic", 128, 0, "bf16", 128)));
    EXPECT_FALSE(cdnaAdmits(bf16, makeCdnaKernel("generic", 128, 0, "f32", 128)));
    // A bucket serves f32 head dims up to it, as in 16 bits.
    auto narrow = cdnaF32Spec();
    narrow.headDim = 72;
    EXPECT_TRUE(cdnaAdmits(
        narrow, makeCdnaKernel("generic", 128, 0, "f32", 128, 0, 1, 0, std::nullopt, int64_t{1})));
    // The dual-wave and split decode kernels take no f32, whatever their records say; the
    // same graphs in bf16 run on the same records.
    EXPECT_FALSE(cdnaAdmits(f32, makeCdnaKernel("dualwave", 128, 0, "f32", 256)));
    EXPECT_TRUE(cdnaAdmits(bf16, makeCdnaKernel("dualwave", 128, 0, "bf16", 256)));
    auto decode = cdnaF32Spec();
    decode.seqLenQ = 1;
    decode.seqLenKv = 4096;
    EXPECT_FALSE(cdnaAdmits(decode, makeCdnaKernel("decode", 128, 0, "f32", 64), 304));
    auto decodeBf16 = decode;
    decodeBf16.dataType = DataType::BFLOAT16;
    EXPECT_TRUE(cdnaAdmits(decodeBf16, makeCdnaKernel("decode", 128, 0, "bf16", 64), 304));
}

/// Exact f32: f32 products and accumulation, P in f32.
SdpaGraphSpec cdnaF32GpuSpec()
{
    auto spec = cdnaF32Spec();
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 300;
    spec.seqLenKv = 333;
    spec.stats = SdpaStats::LSE;
    return spec;
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ComputesF32AtEveryHeadDimItServes)
{
    // The buckets up to 64 and 128 (their own widths and narrower ones) and 256 (a bucket on
    // gfx950, exact elsewhere), with and without a bottom-right diagonal, GQA, ragged last
    // tiles and the LSE; gfx950's 256 bucket at a width inside it.
    std::vector<int64_t> headDims{64, 40, 128, 72, 256};
    if(packedArchDeviceProperties().gcnArchName == "gfx950")
    {
        headDims.push_back(200);
    }
    for(const int64_t headDim : headDims)
    {
        for(const auto mask : {SdpaMask::NONE, SdpaMask::BOTTOM_RIGHT_CAUSAL})
        {
            SCOPED_TRACE("head_dim " + std::to_string(headDim)
                         + (mask == SdpaMask::NONE ? "" : ", causal"));
            auto spec = cdnaF32GpuSpec();
            spec.headDim = headDim;
            spec.mask = mask;
            expectCdnaMatchesReference(spec, F32_TOLERANCE);
        }
    }
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ComputesF32WithEveryMaskAndLayout)
{
    auto spec = cdnaF32GpuSpec();
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    expectCdnaMatchesReference(spec, F32_TOLERANCE);
    spec.mask = SdpaMask::SLIDING_WINDOW;
    expectCdnaMatchesReference(spec, F32_TOLERANCE);
    spec.mask = SdpaMask::NONE;
    spec.layout = SdpaLayout::BHSD;
    spec.kvHeads = 1;
    expectCdnaMatchesReference(spec, F32_TOLERANCE);
    auto packedQkv = cdnaF32GpuSpec();
    packedQkv.kvHeads = packedQkv.heads;
    packedQkv.seqLenKv = packedQkv.seqLenQ;
    packedQkv.packedQkv = true;
    packedQkv.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    expectCdnaMatchesReference(packedQkv, F32_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ComputesF32EdgeCases)
{
    // COVERAGE.md section 7's rows reachable without a bias: rows no key reaches (more
    // queries than keys, bottom-right), a NaN row beside finite ones, padded batches (one
    // with no keys) and packed sequences.
    auto keyless = cdnaF32GpuSpec();
    keyless.seqLenQ = 333;
    keyless.seqLenKv = 64;
    keyless.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    expectCdnaMatchesReference(keyless, F32_TOLERANCE);
    auto nanRow = cdnaF32GpuSpec();
    nanRow.queryNanPosition = 130;
    expectCdnaMatchesReference(nanRow, F32_TOLERANCE);
    auto padded = paddedGpuSpec();
    padded.dataType = DataType::FLOAT;
    padded.seqLensKv = {0, 300, 1};
    padded.queryNanPosition = 100;
    expectCdnaMatchesReference(padded, F32_TOLERANCE);
    auto packed = packedGpuSpec();
    packed.dataType = DataType::FLOAT;
    packed.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    expectCdnaMatchesReference(packed, F32_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, KeepsF32DenormalInputs)
{
    // V in f32's denormal range (k/32 times 2^-130): O is a convex mix of V, so it is
    // denormal too, and must match to a few denormal ulps. With denormals flushed (the
    // 16-bit objects' DAZ) every O element would read 0, about 1e-40 off.
    for(const int64_t headDim : {64, 128, 256})
    {
        SCOPED_TRACE("head_dim " + std::to_string(headDim));
        auto spec = cdnaF32GpuSpec();
        spec.headDim = headDim;
        spec.stats = SdpaStats::NONE;
        spec.valueScale = std::ldexp(1.0F, -130);
        expectCdnaMatchesReference(spec, 5e-42);
    }
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ServesADiagonalThatShowsEveryKeyOnTheNonCausalObjects)
{
    // Bound as no mask: every staged object that admits it -- the dual-wave kernel's on
    // gfx950, the split decode objects' at a short query -- matches the reference's banded
    // graph.
    auto prefill = cdnaSpec();
    prefill.seqLenQ = 512;
    prefill.seqLenKv = 256;
    prefill.mask = SdpaMask::TOP_LEFT_CAUSAL;
    prefill.rightBound = 300;
    prefill.stats = SdpaStats::LSE;
    expectCdnaMatchesReference(prefill, BF16_TOLERANCE, FLYDSL_SDPA_CDNA_DUALWAVE);
    auto decode = cdnaSpec();
    decode.seqLenQ = 4;
    decode.seqLenKv = 2048;
    decode.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    decode.rightBound = 3;
    decode.stats = SdpaStats::LSE;
    expectCdnaMatchesReference(decode, BF16_TOLERANCE, FLYDSL_SDPA_CDNA_DECODE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, PacksPaddedSequencesInTheirSlots)
{
    // P3: each packed slot holds a shorter real sequence (cuDNN's ragged operands with
    // seq_len). The slot's rows past it are padded rows, O = 0 and LSE = -inf; its keys past it
    // are never read; nothing reaches the next slot. Lengths 0, short, full, and past the slot
    // (clamped to it), with and without padding_mask, at both corners.
    auto spec = packedGpuSpec();
    setPacked(spec, {300, 129, 40}, {300, 63, 200});
    setPadding(spec, {250, 129, 0}, {300, 30, 999});
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    spec.paddingMask = false;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.withSink = true;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, ComputesF32InPaddedSlotsAndAtAnEveryKeyDiagonal)
{
    // The f32 objects (modification 35) under the host-side rules that arrived beside them:
    // packed slots with per-batch lengths (P3), and a diagonal at or past the last key bound
    // as no mask, which runs on the non-causal f32 objects.
    auto slots = packedGpuSpec();
    slots.dataType = DataType::FLOAT;
    setPacked(slots, {300, 129, 40}, {300, 63, 200});
    setPadding(slots, {250, 129, 0}, {300, 30, 999});
    expectCdnaMatchesReference(slots, F32_TOLERANCE);
    slots.paddingMask = false;
    slots.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    expectCdnaMatchesReference(slots, F32_TOLERANCE);
    auto everyKey = cdnaF32GpuSpec();
    everyKey.seqLenQ = 512;
    everyKey.seqLenKv = 256;
    everyKey.mask = SdpaMask::TOP_LEFT_CAUSAL;
    everyKey.rightBound = 300;
    everyKey.stats = SdpaStats::LSE;
    expectCdnaMatchesReference(everyKey, F32_TOLERANCE);
}

INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuFlydslSdpaCdnaFullTier,
                         ::testing::Values(0),
                         [](const ::testing::TestParamInfo<int>&) { return std::string("Sweep"); });

} // namespace

// The harness above, reached with a named object (FlydslSdpaObjectTestSupport.hpp).
void testing::expectCdnaObjectMatchesReference(const SdpaGraphSpec& spec,
                                               const KernelDefinition& kernel)
{
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec), packedArchDeviceProperties());
    const auto bound = matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
    ASSERT_TRUE(bound.has_value()) << "the graph " << kernel.name << " is meant for is declined";
    ASSERT_TRUE(matchesCdnaKernel(fixture.context(), kernel, *bound))
        << kernel.name << " does not admit the graph built for it";
    if(spec.dataType == DataType::HALF)
    {
        expectCandidateMatchesReference<hipdnn_data_sdk::types::half>(
            fixture.context(), *bound, kernel, spec, 4e-3);
    }
    else if(spec.dataType == DataType::FLOAT)
    {
        expectCandidateMatchesReference<float>(
            fixture.context(), *bound, kernel, spec, F32_TOLERANCE);
    }
    else
    {
        expectCandidateMatchesReference<Bf16>(
            fixture.context(), *bound, kernel, spec, BF16_TOLERANCE);
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
