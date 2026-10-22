// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose: a future edit that puts this file back on
// an ungated source list then yields an empty translation unit rather than a suite that
// reads a staged shard nothing produced.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
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
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaObjectTestSupport.hpp"
// The SDPA graph builder: plain hipDNN graph construction, shared with the RDNA suites.
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaRdnaTestGraphs.hpp"

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

INSTANTIATE_TEST_SUITE_P(,
                         TestFlydslSdpaCdnaFieldDeclines,
                         ::testing::ValuesIn(sdpaFieldDeclines()),
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
    edit(
        "BiasKeysNotContiguous",
        [biasRowPitch](SdpaGraphSpec& s) { biasRowPitch(s, 2 * s.seqLenKv, 2); },
        [biasRowPitch](SdpaGraphSpec& s) { biasRowPitch(s, 2 * s.seqLenKv, 1); });
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
    auto topLeftOverflow = cdnaSpec();
    topLeftOverflow.mask = SdpaMask::TOP_LEFT_CAUSAL;
    topLeftOverflow.seqLenQ = 512;
    topLeftOverflow.seqLenKv = 256;
    cases.push_back({"TopLeftCausalWithMoreQueriesThanKeys", topLeftOverflow});
    // A sink in another dtype than the query's.
    auto f32Sink = cdnaSpec();
    f32Sink.withSink = true;
    f32Sink.sinkDataType = DataType::FLOAT;
    cases.push_back({"SinkInAnotherDtype", f32Sink});
    // A head's sink offset h * stride past int32: the kernels compute it in 32 bits, so heads
    // 32 and up would read another head's logit.
    auto wideSinkStride = cdnaSpec();
    wideSinkStride.withSink = true;
    wideSinkStride.headDim = 64;
    wideSinkStride.heads = 64;
    wideSinkStride.kvHeads = 8;
    wideSinkStride.sinkHeadStride = int64_t{1} << 26;
    cases.push_back({"SinkHeadOffsetsPastInt32", wideSinkStride});
    auto wideV = cdnaSpec();
    wideV.valueHeadDim = 256;
    cases.push_back({"ValueHeadDimWider", wideV});
    auto bottomRightBand = cdnaSpec();
    bottomRightBand.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    bottomRightBand.rightBound = 8;
    cases.push_back({"BottomRightBand", bottomRightBand});
    auto leftOnly = cdnaSpec();
    leftOnly.leftBound = 64;
    cases.push_back({"SlidingWindowWithNoRightBound", leftOnly});
    auto band = cdnaSpec();
    band.mask = SdpaMask::TOP_LEFT_CAUSAL;
    band.rightBound = 16;
    cases.push_back({"RightBoundBand", band});
    auto f32Bias = cdnaSpec();
    f32Bias.withBias = true;
    cases.push_back({"BiasInAnotherDtype", f32Bias});
    auto keyBroadcast = cdnaSpec();
    keyBroadcast.withBias = true;
    keyBroadcast.biasDataType = DataType::BFLOAT16;
    keyBroadcast.biasDims = {1, 1, 256, 1};
    cases.push_back({"BiasBroadcastAlongTheKeys", keyBroadcast});
    auto unaligned = cdnaSpec();
    unaligned.withBias = true;
    unaligned.biasDataType = DataType::BFLOAT16;
    unaligned.seqLenQ = 333;
    unaligned.seqLenKv = 333;
    cases.push_back({"BiasRowsNotSixteenByteAligned", unaligned});
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
    auto fp32 = cdnaSpec();
    fp32.dataType = DataType::FLOAT;
    cases.push_back({"Fp32", fp32});
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
    // A right bound past the keys' diagonal, at the largest value: declined without
    // overflowing.
    auto hugeRight = cdnaSpec();
    hugeRight.mask = SdpaMask::TOP_LEFT_CAUSAL;
    hugeRight.rightBound = std::numeric_limits<int64_t>::max();
    cases.push_back({"LargestRightBound", hugeRight});
    // The bias is read as runs of 4 keys and its view ends at the last key, so a run that
    // straddles that end could read as zero: a key length not a multiple of 4 declines.
    auto biasTail = cdnaSpec();
    biasTail.withBias = true;
    biasTail.biasDataType = DataType::BFLOAT16;
    biasTail.seqLenKv = 333;
    biasTail.biasDims = {1, 1, 1, 333};
    cases.push_back({"BiasOverAKeyLengthNotAMultipleOfFour", biasTail});
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
    EXPECT_EQ(windowOf(SdpaMask::TOP_LEFT_CAUSAL, 256, 256, 64), -64);
    EXPECT_EQ(windowOf(SdpaMask::BOTTOM_RIGHT_CAUSAL, 256, 1024, 64), 704);
    // A window no row reaches is none, so the dual-wave objects still serve it.
    EXPECT_EQ(windowOf(SdpaMask::TOP_LEFT_CAUSAL, 256, 256, 255), CDNA_NO_WINDOW);
    EXPECT_EQ(windowOf(SdpaMask::TOP_LEFT_CAUSAL, 256, 256, 254), -254);
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

/// Runs @p kernel for @p spec through prepare() and launch() and checks every output element
/// against a double-precision reference, in the memory layout @p spec names: BSHD, BHSD, or
/// Q, K and V as views into one packed buffer.
template <typename T>
void expectCandidateMatchesReference(const MatchContext& context,
                                     const BoundTokens& bound,
                                     const KernelDefinition& kernel,
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
    const auto& handler = dispatchHandler(cdnaEngineFor(kernel));
    const auto prepared = handler.prepare(context, bound, kernel);
    ASSERT_NE(prepared, nullptr);

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
    const auto v = spec.packedQkv ? q : fillValues<T>(vCount, 3U);
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

    void* qDevice = nullptr;
    void* kDevice = nullptr;
    void* vDevice = nullptr;
    void* oDevice = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&qDevice, qCount * sizeof(T)));
    ASSERT_EQ(hipSuccess, hipMalloc(&oDevice, oCount * sizeof(T)));
    ASSERT_EQ(hipSuccess, hipMemcpy(qDevice, q.data(), qCount * sizeof(T), hipMemcpyHostToDevice));
    if(spec.packedQkv)
    {
        kDevice = static_cast<T*>(qDevice) + kStart;
        vDevice = static_cast<T*>(qDevice) + vStart;
    }
    else
    {
        ASSERT_EQ(hipSuccess, hipMalloc(&kDevice, kvCount * sizeof(T)));
        ASSERT_EQ(hipSuccess, hipMalloc(&vDevice, vCount * sizeof(T)));
        ASSERT_EQ(hipSuccess,
                  hipMemcpy(kDevice, k.data(), kvCount * sizeof(T), hipMemcpyHostToDevice));
        ASSERT_EQ(hipSuccess,
                  hipMemcpy(vDevice, v.data(), vCount * sizeof(T), hipMemcpyHostToDevice));
    }
    // All-ones bytes are NaN in bf16 and f16: an element the kernel never writes fails rather
    // than reading as a zero.
    ASSERT_EQ(hipSuccess, hipMemset(oDevice, 0xFF, oCount * sizeof(T)));
    // The LSE output, [B, H, Sq] f32, when the graph asks for it.
    const bool withStats = spec.stats == SdpaStats::LSE;
    const auto statsCount = static_cast<size_t>(b * h * s);
    std::vector<float> stats(statsCount, 0.0F);
    void* statsDevice = nullptr;
    if(withStats)
    {
        ASSERT_EQ(hipSuccess, hipMalloc(&statsDevice, statsCount * sizeof(float)));
        ASSERT_EQ(hipSuccess, hipMemset(statsDevice, 0xFF, statsCount * sizeof(float)));
    }

    std::vector<hipdnnPluginDeviceBuffer_t> buffers{{FLYDSL_SDPA_RDNA_Q_UID, qDevice},
                                                    {FLYDSL_SDPA_RDNA_K_UID, kDevice},
                                                    {FLYDSL_SDPA_RDNA_V_UID, vDevice},
                                                    {FLYDSL_SDPA_RDNA_O_UID, oDevice}};
    if(withStats)
    {
        buffers.push_back({FLYDSL_SDPA_RDNA_STATS_UID, statsDevice});
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
    auto biasValues = spec.withBias ? fillValues<T>(biasCount, 4U) : std::vector<T>{};
    if(spec.withBias
       && (spec.biasMasksRows || spec.biasNegativeInfinityLeadingKeys > 0
           || spec.biasNegativeInfinityRows))
    {
        const T negativeInfinity(-std::numeric_limits<float>::infinity());
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
                            biasValues[biasOffset(bi, hi, si, j)]
                                = std::numeric_limits<T>::lowest();
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
    void* biasDevice = nullptr;
    if(spec.withBias)
    {
        ASSERT_EQ(hipSuccess, hipMalloc(&biasDevice, biasCount * sizeof(T)));
        ASSERT_EQ(
            hipSuccess,
            hipMemcpy(biasDevice, biasValues.data(), biasCount * sizeof(T), hipMemcpyHostToDevice));
        buffers.push_back({FLYDSL_SDPA_RDNA_BIAS_UID, biasDevice});
    }
    const auto biasAt = [&](int64_t bi, int64_t hi, int64_t si, int64_t j) {
        return static_cast<double>(static_cast<float>(biasValues[biasOffset(bi, hi, si, j)]));
    };
    // The attention sink: one logit per query head, in the query's dtype, or every one the
    // spec's value.
    auto sinkValues = spec.withSink ? fillValues<T>(static_cast<size_t>(h), 5U) : std::vector<T>{};
    if(spec.sinkValue)
    {
        std::fill(sinkValues.begin(), sinkValues.end(), T(*spec.sinkValue));
    }
    if(spec.sinkNegativeInfinity)
    {
        std::fill(sinkValues.begin(), sinkValues.end(), T(-std::numeric_limits<float>::infinity()));
    }
    void* sinkDevice = nullptr;
    if(spec.withSink)
    {
        ASSERT_EQ(hipSuccess, hipMalloc(&sinkDevice, static_cast<size_t>(h) * sizeof(T)));
        ASSERT_EQ(hipSuccess,
                  hipMemcpy(sinkDevice,
                            sinkValues.data(),
                            static_cast<size_t>(h) * sizeof(T),
                            hipMemcpyHostToDevice));
        buffers.push_back({FLYDSL_SDPA_RDNA_SINK_UID, sinkDevice});
    }
    // A runtime scale arrives with the buffers, as a host scalar.
    float runtimeScale = spec.scale;
    if(spec.scaleKind == SdpaScale::RUNTIME_TENSOR)
    {
        buffers.push_back({FLYDSL_SDPA_RDNA_SCALE_UID, &runtimeScale});
    }
    // A split object launched with more than one split writes its partials here.
    const auto workspaceBytes = handler.workspaceBytes(context, bound, kernel);
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
    ASSERT_EQ(hipSuccess, hipMemcpy(o.data(), oDevice, oCount * sizeof(T), hipMemcpyDeviceToHost));
    static_cast<void>(hipFree(qDevice));
    if(!spec.packedQkv)
    {
        static_cast<void>(hipFree(kDevice));
        static_cast<void>(hipFree(vDevice));
    }
    static_cast<void>(hipFree(oDevice));
    static_cast<void>(hipFree(sinkDevice));
    if(spec.withBias)
    {
        static_cast<void>(hipFree(biasDevice));
    }
    if(withStats)
    {
        ASSERT_EQ(
            hipSuccess,
            hipMemcpy(
                stats.data(), statsDevice, statsCount * sizeof(float), hipMemcpyDeviceToHost));
        static_cast<void>(hipFree(statsDevice));
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
    // The reference's rule: query row i sees key j <= i + align + right, align 0 top-left
    // and Skv - Sq bottom-right.
    const int64_t align = alignBottomRight ? skv - s : 0;
    const int64_t offset = align + spec.rightBound.value_or(0);
    // And with a left bound, no key j < i + align - left (a sliding window).
    std::optional<int64_t> left = spec.leftBound;
    if(spec.mask == SdpaMask::SLIDING_WINDOW && !left.has_value())
    {
        left = 64;
    }
    double worst = 0.0;
    int64_t nonFinite = 0;
    int64_t firstNonFiniteRow = std::numeric_limits<int64_t>::max();
    int64_t lastNonFiniteRow = -1;
    std::string where;
    double worstLse = 0.0;
    std::string whereLse;
    std::vector<double> scores(static_cast<size_t>(skv));
    for(int64_t bi = 0; bi < b; ++bi)
    {
        for(int64_t hi = 0; hi < h; ++hi)
        {
            for(int64_t si = 0; si < s; ++si)
            {
                // Negative for a row no key reaches: O is 0 and the LSE -inf.
                const int64_t lastKey = causal ? std::min(si + offset, skv - 1) : skv - 1;
                const int64_t firstKey
                    = left.has_value() ? std::max<int64_t>(0, si + align - *left) : 0;
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
                double sum = 0.0;
                for(int64_t j = firstKey; j <= lastKey; ++j)
                {
                    scores[static_cast<size_t>(j)]
                        = keyless ? 0.0 : std::exp(scores[static_cast<size_t>(j)] - maxScore);
                    sum += scores[static_cast<size_t>(j)];
                }
                // The sink's logit joins the denominator of a row that reaches keys; a row
                // that reaches none has O = 0 and LSE = the sink.
                const double sinkLogit = spec.withSink ? static_cast<double>(static_cast<float>(
                                                             sinkValues[static_cast<size_t>(hi)]))
                                                       : -std::numeric_limits<double>::infinity();
                if(spec.withSink && !keyless)
                {
                    sum += std::exp(sinkLogit - maxScore);
                }
                if(withStats)
                {
                    // Natural log of the row's sum of exp(scaled score).
                    // [B, H, Sq] packed, or [B, Sq, H] when the graph lays it out so.
                    const auto statsIndex
                        = spec.statsSequenceMajor ? (bi * s + si) * h + hi : (bi * h + hi) * s + si;
                    const auto lse = static_cast<double>(stats[static_cast<size_t>(statsIndex)]);
                    // A row no key reaches has LSE -inf.
                    const bool negativeInfinity = std::isinf(lse) && lse < 0.0;
                    double lseError = std::abs(lse - (maxScore + std::log(sum)));
                    if(nanRow)
                    {
                        lseError = std::isnan(lse) ? 0.0 : 1e30;
                    }
                    else if(keyless && spec.withSink && std::isfinite(sinkLogit))
                    {
                        lseError = std::abs(lse - sinkLogit);
                    }
                    else if(keyless)
                    {
                        // No sink, or a sink of -inf: the LSE is exactly -inf.
                        lseError = negativeInfinity ? 0.0 : 1e30;
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
                    double expected = 0.0;
                    for(int64_t j = firstKey; j <= lastKey; ++j)
                    {
                        expected += scores[static_cast<size_t>(j)]
                                    * at(v, vStrides, vStart, bi, hi / group, j, e);
                    }
                    expected = keyless ? 0.0 : expected / sum;
                    // NaN or inf (an unwritten element reads as NaN) is a full miss, but in a
                    // NaN row, where anything but NaN is.
                    const double actual = at(o, oStrides, 0, bi, hi, si, e);
                    double error = std::isfinite(actual) ? std::abs(actual - expected)
                                                         : std::numeric_limits<double>::max();
                    if(nanRow)
                    {
                        error = std::isnan(actual) ? 0.0 : std::numeric_limits<double>::max();
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
                                + " s=" + std::to_string(si) + " d=" + std::to_string(e) + ": got "
                                + std::to_string(actual) + ", expected " + std::to_string(expected);
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
    for(const auto& kernel : candidates)
    {
        SCOPED_TRACE(kernel.name);
        if(spec.dataType == DataType::HALF)
        {
            // P is rounded to the input dtype before the second product: f16's 11-bit
            // mantissa allows about an eighth of bf16's budget.
            expectCandidateMatchesReference<hipdnn_data_sdk::types::half>(
                fixture.context(), *bound, kernel, spec, std::min(absoluteTolerance, 4e-3));
        }
        else
        {
            expectCandidateMatchesReference<Bf16>(
                fixture.context(), *bound, kernel, spec, absoluteTolerance);
        }
    }
}

/// bf16 output rounding is ~2^-8 relative on values of magnitude <= 1.
constexpr double BF16_TOLERANCE = 2e-2;

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

TEST(TestGpuFlydslSdpaCdnaDispatch, ServesTheCombinationsNoOtherCaseReaches)
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

TEST(TestGpuFlydslSdpaCdnaDispatch, ServesDeepSeekV4PrefillAtHeadDim512)
{
    // DeepSeek-V4's attention: Q/K and V 512 wide, one KV head, causal with a 128-key
    // sliding window, prefill-sized. The decode engine's wide objects serve it (any query
    // length), so it must be right here too: with the window, and plain causal.
    for(const int64_t seqLen : {int64_t{1024}, int64_t{2048}})
    {
        for(const bool window : {true, false})
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

TEST(TestGpuFlydslSdpaCdnaDispatch, ServesSlidingWindows)
{
    // Narrower than a KV tile, across a tile boundary, and wider than the keys; top-left
    // and bottom-right; ragged lengths.
    for(const int64_t left : {int64_t{0}, int64_t{37}, int64_t{64}, int64_t{200}, int64_t{4096}})
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

TEST(TestGpuFlydslSdpaCdnaDispatch, ServesEveryWidthInsideTheBucketsThatAreNotPowersOfTwo)
{
    // The 96, 160, 192 and 224 buckets (vendored-kernel modification 26), at each one's
    // own width and inside it, every candidate the matcher admits (wider buckets too):
    // causal bf16 with the LSE, then non-causal f16 BHSD at a ragged length.
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

TEST(TestGpuFlydslSdpaCdnaDispatch, AddsABiasThatMasksWholeRowsAtTheLowestFiniteValue)
{
    // A padding mask's fully masked rows, as Hugging Face writes them: every third row's bias
    // is the query dtype's lowest finite value on every key, so its scores are finite and
    // equal and its softmax is uniform over the keys the causal mask leaves (O the mean of V,
    // LSE that score plus log(count)), as PyTorch and CpuFpReferenceSdpa give
    // (vendored-kernel modification 28). With the scale, the first 128 keys of every row are
    // also -inf, leaving such a row the rest, and in a causal row before them no key at all.
    // The dual-wave kernel's bias (gfx950 at 64 and 128) and the generic one's. Skv 264 ends
    // in a part tile, so the key-padding mask's keys sit next to the lowest rows' keys.
    for(const int64_t headDim : {int64_t{64}, int64_t{128}, int64_t{256}})
    {
        for(const int64_t seqLenKv : {int64_t{384}, int64_t{264}})
        {
            for(const bool causal : {false, true})
            {
                for(const bool scaled : {false, true})
                {
                    SCOPED_TRACE("head_dim " + std::to_string(headDim) + ", Skv "
                                 + std::to_string(seqLenKv) + (causal ? ", causal" : "")
                                 + (scaled ? ", scaled" : ""));
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

TEST(TestGpuFlydslSdpaCdnaDispatch, LongerSequenceMatchesTheReference)
{
    auto spec = cdnaSpec();
    spec.seqLenQ = 1024;
    spec.seqLenKv = 1024;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

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
    else
    {
        expectCandidateMatchesReference<Bf16>(
            fixture.context(), *bound, kernel, spec, BF16_TOLERANCE);
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
