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
#include <limits>
#include <optional>
#include <string>
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
// The SDPA graph builder: plain hipDNN graph construction, shared with the RDNA suites.
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaRdnaTestGraphs.hpp"

/**
 * @file TestFlydslSdpaCdnaEngine.cpp
 * @brief The hipkernel:flydsl_sdpa_cdna pack's seams: what its graph matcher accepts and
 *        declines, what it binds, which candidate a graph selects, and real launches of
 *        the staged gfx942/gfx950 objects checked against a host reference.
 *
 * The CDNA kernels bake the head counts, head_dim and contiguous BSHD, and assume equal
 * query and key lengths, so the accepted envelope is narrow on purpose; the
 * decline cases pin its edges. The GPU cases skip on any device this build packed no CDNA
 * object for.
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
constexpr const char* CDNA_KERNEL_FIELD = "kernel";
constexpr const char* CDNA_HEAD_DIM_FIELD = "head_dim";
constexpr const char* CDNA_CAUSAL_FIELD = "causal";
constexpr const char* CDNA_BLOCK_M_FIELD = "block_m";
constexpr const char* CDNA_KV_ALIGN_FIELD = "kv_align";
constexpr const char* CDNA_ANY_DIAGONAL_FIELD = "any_diagonal";
constexpr const char* CDNA_HAS_BIAS_FIELD = "has_bias";
constexpr const char* CDNA_HEAD_DIM_V_FIELD = "head_dim_v";
constexpr const char* CDNA_QK_MASK_FIELD = "qk_mask";
constexpr const char* CDNA_SPLIT_KV_FIELD = "split_kv";
constexpr const char* CDNA_PACK_GQA_FIELD = "pack_gqa";
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

/// A kernel carrying exactly the metadata this pack's KMD declares.
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
    kernel.name = "sdpa_cdna_" + dtype + "_" + family;
    kernel.metadata = {{CDNA_DTYPE_FIELD, dtype},
                       {CDNA_KERNEL_FIELD, family},
                       {CDNA_HEAD_DIM_FIELD, headDim},
                       {CDNA_CAUSAL_FIELD, causal},
                       {CDNA_BLOCK_M_FIELD, blockM},
                       {CDNA_KV_ALIGN_FIELD, kvAlign},
                       {CDNA_ANY_DIAGONAL_FIELD, anyDiagonal},
                       {CDNA_HAS_BIAS_FIELD, hasBias},
                       {CDNA_HEAD_DIM_V_FIELD, headDimV.value_or(headDim)},
                       {CDNA_QK_MASK_FIELD, qkMask},
                       {CDNA_SPLIT_KV_FIELD, int64_t{0}},
                       {CDNA_PACK_GQA_FIELD, int64_t{0}},
                       {"dv_split", int64_t{1}},
                       {CDNA_MERGE_SYMBOL_FIELD, std::string()}};
    kernel.priority = 100;
    return kernel;
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
    return cases;
}

INSTANTIATE_TEST_SUITE_P(,
                         TestFlydslSdpaCdnaGraphAccepts,
                         ::testing::ValuesIn(acceptedCases()),
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
            EXPECT_EQ(matchesKernel(FLYDSL_SDPA_CDNA, fixture.context(), plain, *bound),
                      !graphHasBias)
                << family;
            EXPECT_EQ(matchesKernel(FLYDSL_SDPA_CDNA, fixture.context(), biased, *bound),
                      graphHasBias)
                << family;
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
    // No scale on the graph binds the default, 1/sqrt(head_dim), as a value.
    const auto defaulted = cdnaGraphMatch(cdnaSpec());
    ASSERT_TRUE(defaulted.has_value());
    const float expected = 1.0F / std::sqrt(128.0F);
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
    EXPECT_TRUE(matchesKernel(FLYDSL_SDPA_CDNA, fixture.context(), makeCdnaKernel(), *bound));
    EXPECT_TRUE(matchesKernel(FLYDSL_SDPA_CDNA,
                              fixture.context(),
                              makeCdnaKernel("generic", 128, 0, "bf16", 128),
                              *bound));
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
        return bound.has_value()
               && matchesKernel(FLYDSL_SDPA_CDNA, fixture.context(), kernel, *bound);
    };
    EXPECT_TRUE(matches(192, 128, mla));
    EXPECT_TRUE(matches(256, 128, mla));
    EXPECT_TRUE(matches(144, 128, mla));
    // Not a multiple of the masks' 16 columns, past the bucket, or another V width.
    EXPECT_FALSE(matches(200, 128, mla));
    EXPECT_FALSE(matches(192, 192, mla));
    EXPECT_FALSE(matches(192, 64, mla));
    // An object without masks serves its own head dims only.
    EXPECT_FALSE(matches(192, 128, plain256));
    EXPECT_TRUE(matches(256, 256, plain256));
    EXPECT_FALSE(matches(256, 256, mla));
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
        EXPECT_TRUE(matchesKernel(FLYDSL_SDPA_CDNA, fixture.context(), makeCdnaKernel(), *bound))
            << heads << "/" << kvHeads;
    }
}

TEST(TestFlydslSdpaCdnaKernelMatch, DeclinesEachBakedFieldThatDisagrees)
{
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(cdnaSpec()));
    const auto bound = matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());
    const auto declines = [&](const KernelDefinition& kernel) {
        return !matchesKernel(FLYDSL_SDPA_CDNA, fixture.context(), kernel, *bound);
    };
    EXPECT_TRUE(declines(makeCdnaKernel("dualwave", 64)));
    EXPECT_TRUE(declines(makeCdnaKernel("dualwave", 128, 1)));
    EXPECT_TRUE(declines(makeCdnaKernel("dualwave", 128, 0, "f16")));
    EXPECT_TRUE(declines(makeCdnaKernel("wmma")));
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
        return bound.has_value()
               && matchesKernel(FLYDSL_SDPA_CDNA, fixture.context(), kernel, *bound);
    };
    const auto offsetZero = makeCdnaKernel("dualwave", 128, 1, "bf16", 256, 0, 0);
    const auto anyOffset = makeCdnaKernel("dualwave", 128, 1, "bf16", 256, 0, 1);
    EXPECT_TRUE(admits(SdpaMask::BOTTOM_RIGHT_CAUSAL, 256, 256, offsetZero));
    EXPECT_TRUE(admits(SdpaMask::TOP_LEFT_CAUSAL, 256, 1024, offsetZero));
    EXPECT_FALSE(admits(SdpaMask::BOTTOM_RIGHT_CAUSAL, 256, 1024, offsetZero));
    EXPECT_TRUE(admits(SdpaMask::BOTTOM_RIGHT_CAUSAL, 256, 1024, anyOffset));
}

TEST(TestFlydslSdpaCdnaKernelMatch, RunsASlidingWindowOnlyOnTheGenericKernel)
{
    auto spec = cdnaSpec();
    spec.mask = SdpaMask::SLIDING_WINDOW;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    const auto bound = matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_FALSE(matchesKernel(
        FLYDSL_SDPA_CDNA, fixture.context(), makeCdnaKernel("dualwave", 128, 1), *bound));
    EXPECT_TRUE(matchesKernel(FLYDSL_SDPA_CDNA,
                              fixture.context(),
                              makeCdnaKernel("generic", 128, 1, "bf16", 128),
                              *bound));
}

TEST(TestFlydslSdpaCdnaKernelMatch, RunsASplitObjectOnlyWithItsMergeKernel)
{
    auto spec = cdnaSpec();
    spec.seqLenQ = 1;
    spec.seqLenKv = 8192;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    const auto bound = matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());
    auto split = makeCdnaKernel("generic", 128, 0, "bf16", 64);
    split.metadata[CDNA_SPLIT_KV_FIELD] = int64_t{1};
    split.metadata[CDNA_PACK_GQA_FIELD] = int64_t{1};
    split.metadata[CDNA_MERGE_SYMBOL_FIELD] = std::string("flash_attn_generic_merge_kernel_1");
    EXPECT_TRUE(matchesKernel(FLYDSL_SDPA_CDNA, fixture.context(), split, *bound));
    // Without the merge kernel's name its splits could not be combined.
    split.metadata[CDNA_MERGE_SYMBOL_FIELD] = std::string();
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_CDNA, fixture.context(), split, *bound));
    // A generic object that does not say whether it splits or packs is not run.
    auto unstated = makeCdnaKernel("generic", 128, 0, "bf16", 64);
    unstated.metadata.erase(std::string(CDNA_PACK_GQA_FIELD));
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA_CDNA, fixture.context(), unstated, *bound));
}

TEST(TestFlydslSdpaCdnaKernelMatch, HoldsTheKeyLengthToTheCandidatesAlignment)
{
    const auto admits = [](int64_t seqLen, const KernelDefinition& kernel) {
        auto spec = cdnaSpec();
        spec.seqLenQ = seqLen;
        spec.seqLenKv = seqLen;
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
        const auto bound = matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
        return bound.has_value()
               && matchesKernel(FLYDSL_SDPA_CDNA, fixture.context(), kernel, *bound);
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
        return scoreKernel(FLYDSL_SDPA_CDNA, fixture.context(), kernel);
    };
    const auto withRule
        = [](KernelDefinition kernel, const std::string& feature, int64_t low, int64_t high) {
              kernel.priority = 80;
              kernel.metadata[CDNA_PREFER_FEATURE_FIELD] = feature;
              kernel.metadata[CDNA_PREFER_MIN_FIELD] = low;
              kernel.metadata[CDNA_PREFER_MAX_FIELD] = high;
              return kernel;
          };

    // The default carries no rule: it scores its priority at every size.
    auto defaultTier = withRule(makeCdnaKernel(), "none", 0, 0);
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

/// Every staged object for this device's arch that the kernel matcher admits for the graph.
/// Each must compute the same answer, so the GPU cases run them all: at an aligned length
/// that is both a masked object and one that skips the KV tail mask.
std::vector<KernelDefinition> stagedCdnaCandidates(const MatchContext& context,
                                                   const BoundTokens& bound)
{
    std::vector<KernelDefinition> candidates;
    // A build for RDNA targets only stages no CDNA set at all.
    const auto* staged = findLoadedSet(FLYDSL_SDPA_CDNA.engineName);
    if(staged == nullptr)
    {
        return candidates;
    }
    const auto device = packedArchDeviceProperties().gcnArchName;
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
                                        kernel.treeRoot};
            if(matchesKernel(FLYDSL_SDPA_CDNA, context, definition, bound))
            {
                candidates.push_back(std::move(definition));
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

/// The LSE is computed in f32 from exact bf16 inputs; its magnitude is about ln(Skv).
constexpr double LSE_TOLERANCE = 1e-2;

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
    const bool causal = spec.mask != SdpaMask::NONE;
    const auto& handler = dispatchHandler(FLYDSL_SDPA_CDNA);
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
    const auto vStrides
        = spec.packedQkv ? sdpaPackedQkvStrides(spec) : sdpaStrides(spec.layout, hk, skv, dv);
    const auto oStrides = sdpaStrides(spec.layout, h, s, dv);
    const int64_t kStart = spec.packedQkv ? h * d : 0;
    const int64_t vStart = spec.packedQkv ? 2 * h * d : 0;
    const auto qCount = static_cast<size_t>(spec.packedQkv ? b * s * 3 * h * d : b * s * h * d);
    const auto kvCount = static_cast<size_t>(b * skv * hk * d);
    const auto vCount = static_cast<size_t>(b * skv * hk * dv);
    const auto oCount = static_cast<size_t>(b * s * h * dv);
    const auto q = fillValues<T>(qCount, 1U);
    const auto k = spec.packedQkv ? q : fillValues<T>(kvCount, 2U);
    const auto v = spec.packedQkv ? q : fillValues<T>(vCount, 3U);
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
    ASSERT_EQ(hipSuccess, hipMemset(oDevice, 0, oCount * sizeof(T)));
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
    const auto biasValues = spec.withBias ? fillValues<T>(biasCount, 4U) : std::vector<T>{};
    void* biasDevice = nullptr;
    if(spec.withBias)
    {
        ASSERT_EQ(hipSuccess, hipMalloc(&biasDevice, biasCount * sizeof(T)));
        ASSERT_EQ(
            hipSuccess,
            hipMemcpy(biasDevice, biasValues.data(), biasCount * sizeof(T), hipMemcpyHostToDevice));
        buffers.push_back({FLYDSL_SDPA_RDNA_BIAS_UID, biasDevice});
    }
    // The bias element for (batch, head, query, key): broadcast axes are absent or of
    // extent 1, and right-alignment maps the trailing axes.
    const auto biasAt = [&](int64_t bi, int64_t hi, int64_t si, int64_t j) {
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
        return static_cast<double>(static_cast<float>(biasValues[static_cast<size_t>(offset)]));
    };
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
    const double scale = spec.scaleKind == SdpaScale::ABSENT
                             ? 1.0 / std::sqrt(static_cast<double>(d))
                             : static_cast<double>(spec.scale);
    const int64_t group = h / hk;
    // The reference's rule: query row i sees key j <= i + align + right, align 0 top-left
    // and Skv - Sq bottom-right.
    const int64_t align = spec.mask == SdpaMask::BOTTOM_RIGHT_CAUSAL ? skv - s : 0;
    const int64_t offset = align + spec.rightBound.value_or(0);
    // And with a left bound, no key j < i + align - left (a sliding window).
    std::optional<int64_t> left = spec.leftBound;
    if(spec.mask == SdpaMask::SLIDING_WINDOW && !left.has_value())
    {
        left = 64;
    }
    double worst = 0.0;
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
                    maxScore = std::max(maxScore, scores[static_cast<size_t>(j)]);
                }
                double sum = 0.0;
                for(int64_t j = firstKey; j <= lastKey; ++j)
                {
                    scores[static_cast<size_t>(j)]
                        = std::exp(scores[static_cast<size_t>(j)] - maxScore);
                    sum += scores[static_cast<size_t>(j)];
                }
                if(withStats)
                {
                    // Natural log of the row's sum of exp(scaled score).
                    const auto lse
                        = static_cast<double>(stats[static_cast<size_t>((bi * h + hi) * s + si)]);
                    // A row no key reaches has LSE -inf.
                    const bool keyless = lastKey < firstKey;
                    const bool negativeInfinity = std::isinf(lse) && lse < 0.0;
                    double lseError = std::abs(lse - (maxScore + std::log(sum)));
                    if(keyless)
                    {
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
                    expected = lastKey < firstKey ? 0.0 : expected / sum;
                    const double error = std::abs(at(o, oStrides, 0, bi, hi, si, e) - expected);
                    if(error > worst)
                    {
                        worst = error;
                        where = "b=" + std::to_string(bi) + " h=" + std::to_string(hi)
                                + " s=" + std::to_string(si) + " d=" + std::to_string(e);
                    }
                }
            }
        }
    }
    EXPECT_LE(worst, absoluteTolerance) << "worst error at " << where;
    EXPECT_EQ(worstLse, 0.0) << "an LSE outside tolerance, worst at " << whereLse;
}

/// Every staged candidate the matcher admits for @p spec must match the reference.
void expectCdnaMatchesReference(const SdpaGraphSpec& spec, double absoluteTolerance)
{
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec), packedArchDeviceProperties());
    const auto bound = matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());
    const auto candidates = stagedCdnaCandidates(fixture.context(), *bound);
    if(candidates.empty())
    {
        GTEST_SKIP() << "no CDNA object packed for this device ("
                     << packedArchDeviceProperties().gcnArchName << ") and shape";
    }
    for(const auto& kernel : candidates)
    {
        SCOPED_TRACE(kernel.name);
        if(spec.dataType == DataType::HALF)
        {
            expectCandidateMatchesReference<hipdnn_data_sdk::types::half>(
                fixture.context(), *bound, kernel, spec, absoluteTolerance);
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

TEST(TestGpuFlydslSdpaCdnaDispatch, NonCausalMatchesTheReference)
{
    expectCdnaMatchesReference(cdnaSpec(), BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, CausalMatchesTheReference)
{
    auto spec = cdnaSpec();
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
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

TEST(TestGpuFlydslSdpaCdnaDispatch, ServesMoreQueriesThanKeysWithRowsNoKeyReaches)
{
    auto spec = cdnaSpec();
    spec.seqLenQ = 700;
    spec.seqLenKv = 333;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
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
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);

    spec.batch = 2;
    spec.seqLenQ = 7;
    spec.seqLenKv = 5000;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);

    spec.batch = 1;
    spec.heads = 16;
    spec.kvHeads = 1;
    spec.seqLenQ = 1;
    spec.seqLenKv = 3000;
    spec.mask = SdpaMask::NONE;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);

    spec.heads = 8;
    spec.kvHeads = 8;
    spec.seqLenQ = 3;
    spec.layout = SdpaLayout::BHSD;
    spec.dataType = DataType::HALF;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);

    spec.layout = SdpaLayout::BSHD;
    spec.dataType = DataType::BFLOAT16;
    spec.heads = 32;
    spec.kvHeads = 8;
    spec.seqLenQ = 4;
    spec.mask = SdpaMask::SLIDING_WINDOW;
    spec.leftBound = 700;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
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
        expectCdnaMatchesReference(spec, BF16_TOLERANCE);
    }
}

TEST(TestGpuFlydslSdpaCdnaDispatch, DecodesAbsorbedMlaWithOneKvHead)
{
    // Absorbed multi-head latent attention: every query head reads one KV head whose keys
    // are 576 wide and whose values are its 512-wide latent; and DeepSeek-V4's 512/512.
    auto spec = cdnaSpec();
    spec.headDim = 576;
    spec.valueHeadDim = 512;
    spec.heads = 16;
    spec.kvHeads = 1;
    spec.seqLenQ = 1;
    spec.seqLenKv = 3000;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);

    // A two-token step in f16, non-causal.
    spec.dataType = DataType::HALF;
    spec.seqLenQ = 2;
    spec.seqLenKv = 777;
    spec.mask = SdpaMask::NONE;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);

    spec.dataType = DataType::BFLOAT16;
    spec.headDim = 512;
    spec.heads = 8;
    spec.seqLenQ = 1;
    spec.seqLenKv = 1500;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
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
}

TEST(TestGpuFlydslSdpaCdnaDispatch, LongerSequenceMatchesTheReference)
{
    auto spec = cdnaSpec();
    spec.seqLenQ = 1024;
    spec.seqLenKv = 1024;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
