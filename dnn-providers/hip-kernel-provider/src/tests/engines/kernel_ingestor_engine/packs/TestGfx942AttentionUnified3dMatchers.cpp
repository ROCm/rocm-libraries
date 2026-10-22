// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "Gfx942AttentionUnified3dTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx942AttentionUnified3dGeometry.hpp"

/**
 * @file TestGfx942AttentionUnified3dMatchers.cpp
 * @brief Matcher acceptance/refusal for hipkernel:Gfx942AttentionUnified3d beyond what the
 *        admission table can express: tensor layouts (Q/O contiguity, the strided K/V
 *        contract of `kernels/common/attention_kv_cache.py::KvTensorLayout`), every v1
 *        refusal, and the kernel match's field comparisons.
 *
 * CPU-only: no device, no compile, no launch.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing::gfx942_unified_3d
{
namespace
{

using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;

constexpr std::string_view SCALE_BITS_TOKEN = "gfx942_attention_unified3d.scale_bits";

GraphSpec base()
{
    return GraphSpec{};
}

GraphSpec withExtra(std::function<void(data_objects::SdpaAttributesBuilder&)> extra)
{
    GraphSpec spec;
    spec.extra = std::move(extra);
    return spec;
}

// =============================================================================
// Accepted graphs
// =============================================================================

TEST(TestGfx942AttentionUnified3dMatchers, AcceptsTheBaseDecodeGraph)
{
    EXPECT_TRUE(graphMatches(base()));
}

TEST(TestGfx942AttentionUnified3dMatchers, AcceptsNoMaskAndBottomRightCausal)
{
    // Identical at Sq = 1: the single query row is the last row and sees every key.
    auto spec = base();
    spec.mask = Mask::NO_MASK;
    EXPECT_TRUE(graphMatches(spec));
    spec.mask = Mask::BOTTOM_RIGHT_CAUSAL;
    EXPECT_TRUE(graphMatches(spec));
}

TEST(TestGfx942AttentionUnified3dMatchers, AcceptsBf16AndEveryHeadSizeAndGqaRatio)
{
    for(const int64_t headSize : {64, 128, 256})
    {
        for(const int64_t ratio : {1, 2, 4, 8, 16})
        {
            auto spec = base();
            spec.dataType = data_objects::DataType::BFLOAT16;
            spec.headSize = headSize;
            spec.numKvHeads = 2;
            spec.numQueryHeads = 2 * ratio;
            EXPECT_TRUE(graphMatches(spec)) << "D" << headSize << " ratio " << ratio;
        }
    }
}

TEST(TestGfx942AttentionUnified3dMatchers, AcceptsAnyBatchAndKvLength)
{
    for(const int64_t batch : {1, 3, 64, 1024})
    {
        for(const int64_t seqLenKv : {1, 17, 8192, 131072})
        {
            auto spec = base();
            spec.batch = batch;
            spec.seqLenKv = seqLenKv;
            EXPECT_TRUE(graphMatches(spec)) << "B" << batch << " Skv" << seqLenKv;
        }
    }
}

TEST(TestGfx942AttentionUnified3dMatchers, AcceptsBshdBhsdMixedAndPaddedKvViews)
{
    auto spec = base();
    spec.kStrides = bshdStrides(spec.numKvHeads, spec.seqLenKv, spec.headSize);
    spec.vStrides = bhsdStrides(spec.numKvHeads, spec.seqLenKv, spec.headSize);
    EXPECT_TRUE(graphMatches(spec)) << "K BSHD, V BHSD";

    // A KV cache allocated for 1024 tokens with 512 valid: the batch and head strides
    // span the capacity, the logical Skv is 512.
    constexpr int64_t CAPACITY = 1024;
    spec.kStrides = bhsdStrides(spec.numKvHeads, CAPACITY, spec.headSize);
    spec.vStrides = spec.kStrides;
    EXPECT_TRUE(graphMatches(spec)) << "capacity-padded BHSD";

    // Token rows padded by 8 elements (16 bytes) in BSHD.
    const int64_t row = spec.numKvHeads * spec.headSize + 8;
    spec.kStrides = std::vector<int64_t>{spec.seqLenKv * row, spec.headSize, row, 1};
    spec.vStrides = spec.kStrides;
    EXPECT_TRUE(graphMatches(spec)) << "row-padded BSHD";
}

TEST(TestGfx942AttentionUnified3dMatchers, AcceptsQoUnitAxesWithAnyStride)
{
    // B = 1 and Sq = 1: those axes' strides are never multiplied by a non-zero index.
    auto spec = base();
    spec.batch = 1;
    spec.qStrides = std::vector<int64_t>{12345, spec.headSize, 777, 1};
    spec.oStrides = spec.qStrides;
    EXPECT_TRUE(graphMatches(spec));
}

TEST(TestGfx942AttentionUnified3dMatchers, BindsTheScaleAndDefaultsAnAbsentOneToOne)
{
    auto spec = base();
    spec.attnScaleValue = 0.125F;
    {
        const auto graph = fixture(spec);
        const auto bound = matchesGraph(PACK, graph.context());
        ASSERT_TRUE(bound.has_value());
        const auto bits = tryGetBoundInt(*bound, SCALE_BITS_TOKEN);
        ASSERT_TRUE(bits.has_value());
        float scale = 0.0F;
        const auto raw = static_cast<int32_t>(*bits);
        std::memcpy(&scale, &raw, sizeof(scale));
        EXPECT_EQ(scale, 0.125F);
    }
    spec.attnScaleValue.reset();
    {
        const auto graph = fixture(spec);
        const auto bound = matchesGraph(PACK, graph.context());
        ASSERT_TRUE(bound.has_value());
        float scale = 0.0F;
        const auto raw = static_cast<int32_t>(*tryGetBoundInt(*bound, SCALE_BITS_TOKEN));
        std::memcpy(&scale, &raw, sizeof(scale));
        EXPECT_EQ(scale, 1.0F) << "an absent attn_scale_value is cuDNN's 1.0, not 1/sqrt(D)";
    }
}

// =============================================================================
// Q / O layout: contiguous [B, Hq, 1, D]
// =============================================================================

TEST(TestGfx942AttentionUnified3dMatchers, DeclinesANonContiguousQuery)
{
    auto spec = base();
    // Head-padded rows: the kernel indexes (token * Hq + head) * D with no strides.
    spec.qStrides
        = std::vector<int64_t>{spec.numQueryHeads * (spec.headSize + 8), spec.headSize + 8, 1, 1};
    EXPECT_FALSE(graphMatches(spec));

    spec = base();
    // Batch stride past Hq * D.
    spec.qStrides = std::vector<int64_t>{
        2 * spec.numQueryHeads * spec.headSize, spec.headSize, spec.headSize, 1};
    EXPECT_FALSE(graphMatches(spec));
}

TEST(TestGfx942AttentionUnified3dMatchers, DeclinesANonContiguousOutput)
{
    auto spec = base();
    // Head-major [B, D, Hq] storage of O.
    spec.oStrides = std::vector<int64_t>{spec.numQueryHeads * spec.headSize,
                                         1,
                                         spec.numQueryHeads * spec.headSize,
                                         spec.numQueryHeads};
    EXPECT_FALSE(graphMatches(spec));
}

// =============================================================================
// K / V: KvTensorLayout's contract
// =============================================================================

TEST(TestGfx942AttentionUnified3dMatchers, DeclinesANonUnitHeadSizeStride)
{
    auto spec = base();
    spec.kStrides = std::vector<int64_t>{spec.numKvHeads * spec.seqLenKv * spec.headSize,
                                         spec.seqLenKv * spec.headSize,
                                         1,
                                         spec.seqLenKv};
    EXPECT_FALSE(graphMatches(spec));
}

TEST(TestGfx942AttentionUnified3dMatchers, DeclinesNonPositiveStrides)
{
    for(const int64_t bad : {0, -1})
    {
        auto spec = base();
        auto strides = bhsdStrides(spec.numKvHeads, spec.seqLenKv, spec.headSize);
        strides[1] = bad;
        spec.vStrides = strides;
        EXPECT_FALSE(graphMatches(spec)) << bad;
    }
}

TEST(TestGfx942AttentionUnified3dMatchers, DeclinesAMisalignedOuterStride)
{
    // A token stride of D + 4 elements is 8 bytes off a 16-byte row boundary.
    auto spec = base();
    const int64_t token = spec.headSize + 4;
    spec.kStrides = std::vector<int64_t>{
        spec.numKvHeads * spec.seqLenKv * token, spec.seqLenKv * token, token, 1};
    EXPECT_FALSE(graphMatches(spec));
}

TEST(TestGfx942AttentionUnified3dMatchers, DeclinesAnOverlappingView)
{
    // Every head aliases the first: head stride 0 is refused as non-positive, so alias
    // through a head stride shorter than one row.
    auto spec = base();
    spec.kStrides = std::vector<int64_t>{
        spec.numKvHeads * spec.seqLenKv * spec.headSize, 8, spec.headSize, 1};
    EXPECT_FALSE(graphMatches(spec));
}

TEST(TestGfx942AttentionUnified3dMatchers, DeclinesAPerHeadSpanPastTheBufferOffsetLimit)
{
    // BHSD at Skv = 2^23, D = 128: per-head span 2 * 2^23 * 128 = 2^31 bytes > 0x7FFF0000.
    auto spec = base();
    spec.batch = 1;
    spec.numKvHeads = 1;
    spec.numQueryHeads = 1;
    spec.seqLenKv = int64_t{1} << 23;
    EXPECT_FALSE(graphMatches(spec));

    // One token fewer than the limit allows is accepted: 0x7FFF0000 / 256 = 8388352 tokens.
    spec.seqLenKv = 0x7FFF0000 / (2 * spec.headSize);
    EXPECT_TRUE(graphMatches(spec));
    spec.seqLenKv += 1;
    EXPECT_FALSE(graphMatches(spec));
}

TEST(TestGfx942AttentionUnified3dMatchers, DeclinesATokenStridePastTheBufferOffsetLimit)
{
    // Skv = 1 keeps the span tiny; the token stride alone is past 0x7FFF0000 bytes.
    auto spec = base();
    spec.seqLenKv = 1;
    const int64_t token = 0x40000000; // 2^31 bytes
    spec.kStrides = std::vector<int64_t>{spec.numKvHeads * token, token, token, 1};
    EXPECT_FALSE(graphMatches(spec));
}

// =============================================================================
// Shape and capability
// =============================================================================

TEST(TestGfx942AttentionUnified3dMatchers, DeclinesMoreThanOneQueryToken)
{
    auto spec = base();
    spec.seqLenQ = 2;
    spec.qStrides = std::vector<int64_t>{2 * spec.numQueryHeads * spec.headSize,
                                         spec.headSize,
                                         spec.numQueryHeads * spec.headSize,
                                         1};
    spec.oStrides = spec.qStrides;
    EXPECT_FALSE(graphMatches(spec));
}

TEST(TestGfx942AttentionUnified3dMatchers, DeclinesUnsupportedHeadSizesAndGqaRatios)
{
    for(const int64_t headSize : {32, 96, 192, 512})
    {
        auto spec = base();
        spec.headSize = headSize;
        EXPECT_FALSE(graphMatches(spec)) << "D" << headSize;
    }
    for(const auto& [hq, hkv] :
        std::vector<std::pair<int64_t, int64_t>>{{12, 4}, {32, 1}, {64, 2}, {6, 4}, {8, 16}})
    {
        auto spec = base();
        spec.numQueryHeads = hq;
        spec.numKvHeads = hkv;
        EXPECT_FALSE(graphMatches(spec)) << hq << "/" << hkv;
    }
}

TEST(TestGfx942AttentionUnified3dMatchers, DeclinesUnsupportedOrMixedDataTypes)
{
    auto spec = base();
    spec.dataType = data_objects::DataType::FLOAT;
    EXPECT_FALSE(graphMatches(spec));
    spec = base();
    spec.oDataType = data_objects::DataType::BFLOAT16;
    EXPECT_FALSE(graphMatches(spec));
}

TEST(TestGfx942AttentionUnified3dMatchers, DeclinesInconsistentOperandShapes)
{
    auto spec = base();
    spec.vDims
        = std::vector<int64_t>{spec.batch, spec.numKvHeads, spec.seqLenKv + 1, spec.headSize};
    EXPECT_FALSE(graphMatches(spec)) << "V's Skv differs from K's";
    spec = base();
    spec.oDims = std::vector<int64_t>{spec.batch, spec.numQueryHeads, 1, spec.headSize / 2};
    EXPECT_FALSE(graphMatches(spec)) << "O's D differs from Q's";
    spec = base();
    spec.twoNodes = true;
    EXPECT_FALSE(graphMatches(spec)) << "two nodes";
}

// =============================================================================
// Masks and every v1 refusal
// =============================================================================

TEST(TestGfx942AttentionUnified3dMatchers, DeclinesTopLeftCausal)
{
    auto spec = base();
    spec.mask = Mask::TOP_LEFT_CAUSAL;
    EXPECT_FALSE(graphMatches(spec));
    // Even where top-left and bottom-right coincide (Skv = Sq = 1): rocKE refuses it.
    spec.seqLenKv = 1;
    EXPECT_FALSE(graphMatches(spec));
}

TEST(TestGfx942AttentionUnified3dMatchers, DeclinesEverySlidingWindowInV1)
{
    // rocKE admits these; the engine refuses them (see the admission test's exception).
    auto spec = base();
    spec.mask = Mask::WINDOW_128;
    EXPECT_FALSE(graphMatches(spec));
    for(const auto& [left, right] :
        std::vector<std::pair<int64_t, int64_t>>{{0, 0}, {127, 0}, {-1, 5}, {64, -1}})
    {
        spec = base();
        spec.bounds = std::pair{left, right};
        EXPECT_FALSE(graphMatches(spec)) << left << "," << right;
    }
}

TEST(TestGfx942AttentionUnified3dMatchers, DeclinesEveryUnsupportedOptionalFeature)
{
    using Builder = data_objects::SdpaAttributesBuilder;
    const std::vector<std::pair<const char*, std::function<void(Builder&)>>> cases = {
        {"attn_mask", [](Builder& b) { b.add_attn_mask_tensor_uid(EXTRA_UID); }},
        {"scale tensor", [](Builder& b) { b.add_scale_tensor_uid(EXTRA_UID); }},
        {"dropout probability", [](Builder& b) { b.add_dropout_probability(0.1F); }},
        {"dropout seed", [](Builder& b) { b.add_seed_tensor_uid(EXTRA_UID); }},
        {"dropout mask", [](Builder& b) { b.add_dropout_mask_tensor_uid(EXTRA_UID); }},
        {"stats", [](Builder& b) { b.add_stats_tensor_uid(EXTRA_UID); }},
        {"generate_stats", [](Builder& b) { b.add_generate_stats(true); }},
        {"max", [](Builder& b) { b.add_max_tensor_uid(EXTRA_UID); }},
        {"sum_exp", [](Builder& b) { b.add_sum_exp_tensor_uid(EXTRA_UID); }},
        {"sinks", [](Builder& b) { b.add_sink_token_tensor_uid(EXTRA_UID); }},
        {"page table k", [](Builder& b) { b.add_page_table_k_tensor_uid(EXTRA_UID); }},
        {"page table v", [](Builder& b) { b.add_page_table_v_tensor_uid(EXTRA_UID); }},
        {"max_seq_len_kv", [](Builder& b) { b.add_max_seq_len_kv(512); }},
        {"ragged seq_len_q", [](Builder& b) { b.add_seq_len_q_tensor_uid(EXTRA_UID); }},
        {"ragged seq_len_kv", [](Builder& b) { b.add_seq_len_kv_tensor_uid(EXTRA_UID); }},
        {"block mask", [](Builder& b) { b.add_block_mask_tensor_uid(EXTRA_UID); }},
        {"fp8 descale q", [](Builder& b) { b.add_descale_q_tensor_uid(EXTRA_UID); }},
        {"fp8 descale k", [](Builder& b) { b.add_descale_k_tensor_uid(EXTRA_UID); }},
        {"fp8 scale o", [](Builder& b) { b.add_scale_o_tensor_uid(EXTRA_UID); }},
        {"fp8 amax o", [](Builder& b) { b.add_amax_o_tensor_uid(EXTRA_UID); }},
    };
    // hipDNN's SdpaAttributes has no softcap field, so there is no softcap graph to refuse.
    for(const auto& [what, extra] : cases)
    {
        EXPECT_FALSE(graphMatches(withExtra(extra))) << what;
    }

    auto spec = base();
    spec.alibiMask = true;
    EXPECT_FALSE(graphMatches(spec)) << "alibi";
    spec = base();
    spec.paddingMask = true;
    EXPECT_FALSE(graphMatches(spec)) << "padding mask";
    spec = base();
    spec.mmaCoreMode = data_objects::DataType::FLOAT;
    EXPECT_FALSE(graphMatches(spec)) << "mma_core_mode";
}

TEST(TestGfx942AttentionUnified3dMatchers, DeclinesANonFiniteScale)
{
    for(const float scale :
        {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
    {
        auto spec = base();
        spec.attnScaleValue = scale;
        EXPECT_FALSE(graphMatches(spec)) << scale;
    }
}

// =============================================================================
// Kernel match
// =============================================================================

TEST(TestGfx942AttentionUnified3dMatchers, KernelMatchAcceptsTheGraphsCellAtAnyKnob)
{
    for(const int64_t block : {16, 32, 64})
    {
        for(const int64_t segments : {2, 7, 16, 57, 128})
        {
            KernelSpec kernel;
            kernel.blockSize = block;
            kernel.numSegments = segments;
            EXPECT_TRUE(kernelMatchesGraph(base(), kernel)) << block << "/" << segments;
        }
    }
}

TEST(TestGfx942AttentionUnified3dMatchers, KernelMatchComparesEveryFeatureField)
{
    const auto refused = [](auto mutate, const char* what) {
        KernelSpec kernel;
        mutate(kernel);
        EXPECT_FALSE(kernelMatchesGraph(base(), kernel)) << what;
    };
    refused([](KernelSpec& k) { k.dtype = "BF16"; }, "dtype");
    refused([](KernelSpec& k) { k.headSize = 64; }, "head_size");
    refused([](KernelSpec& k) { k.numQueryHeads = 16; }, "num_query_heads");
    refused([](KernelSpec& k) { k.numKvHeads = 4; }, "num_kv_heads");
    refused([](KernelSpec& k) { k.slidingWindow = 128; }, "sliding_window");
}

TEST(TestGfx942AttentionUnified3dMatchers, KernelMatchNeverComparesBatchOrKvLength)
{
    // The canonical build records num_seqs 32; batch and Skv are runtime.
    for(const int64_t batch : {1, 32, 333})
    {
        for(const int64_t seqLenKv : {1, 4096, 100000})
        {
            auto spec = base();
            spec.batch = batch;
            spec.seqLenKv = seqLenKv;
            EXPECT_TRUE(kernelMatchesGraph(spec, KernelSpec{})) << batch << "/" << seqLenKv;
        }
    }
}

TEST(TestGfx942AttentionUnified3dMatchers, KernelMatchDeclinesMalformedRecords)
{
    KernelSpec kernel;
    kernel.blockSize = 128;
    EXPECT_FALSE(kernelMatchesGraph(base(), kernel)) << "block_size outside {16, 32, 64}";
    kernel = KernelSpec{};
    kernel.blockSize = 32;
    kernel.tileSizeOverride = 32; // D128 block 32 builds with tile 16
    EXPECT_FALSE(kernelMatchesGraph(base(), kernel)) << "tile disagrees with rocKE's rule";
    kernel = KernelSpec{};
    kernel.numSegments = 0;
    EXPECT_FALSE(kernelMatchesGraph(base(), kernel)) << "num_segments 0";

    auto definition = makeKernel(KernelSpec{});
    definition.metadata.erase("num_segments");
    const auto graph = fixture(base());
    const auto bound = matchesGraph(PACK, graph.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_FALSE(matchesKernel(PACK, graph.context(), definition, *bound)) << "missing field";
}

// =============================================================================
// Dispatch handler surface that needs no device
// =============================================================================

TEST(TestGfx942AttentionUnified3dMatchers, DeclaresTheSegmentAndReduceLaunches)
{
    EXPECT_EQ(dispatchHandler(PACK).launchNames(), (std::set<std::string>{"segment", "reduce"}));
}

TEST(TestGfx942AttentionUnified3dMatchers, WorkspaceIsTheGeometryHeadersLayoutForTheCandidate)
{
    auto spec = base();
    spec.batch = 3;
    const auto graph = fixture(spec);
    const auto bound = matchesGraph(PACK, graph.context());
    ASSERT_TRUE(bound.has_value());
    for(const int64_t segments : {2, 16, 128})
    {
        KernelSpec kernel;
        kernel.numSegments = segments;
        const auto bytes
            = dispatchHandler(PACK).workspaceBytes(graph.context(), *bound, makeKernel(kernel));
        EXPECT_EQ(static_cast<int64_t>(bytes),
                  gfx942AttentionUnified3dWorkspace(3, spec.numQueryHeads, segments, spec.headSize)
                      .totalBytes)
            << segments;
    }
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::testing::gfx942_unified_3d

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
