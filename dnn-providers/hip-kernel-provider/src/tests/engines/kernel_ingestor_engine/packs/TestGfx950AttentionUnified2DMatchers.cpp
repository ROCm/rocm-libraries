// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

// Local copies of the symbol strings so this test does not depend on the pack's internals.
constexpr std::string_view GRAPH_MATCHER_SYMBOL
    = "hipkernel.gfx950_attention_unified2_d.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL
    = "hipkernel.gfx950_attention_unified2_d.kernel_match";

constexpr int64_t Q_UID = 1, K_UID = 2, V_UID = 3, O_UID = 4;
constexpr int64_t BATCH = 2, HEADS_Q = 32, HEADS_KV = 8, SEQ = 256, HEAD_SIZE = 128;
constexpr float SCALE = 0.08838834764831843F;
constexpr int64_t BLOCK_SIZE = 16;

std::vector<int64_t> bshdStrides(int64_t h, int64_t s, int64_t d)
{
    // B=BATCH dimension stride is h*s*d; each dim is B/h/s/d.
    return {s * h * d, d, h * d, 1};
}

// A minimal SdpaAttributes builder.
struct GraphSpec
{
    int64_t batch = BATCH;
    int64_t headsQ = HEADS_Q;
    int64_t headsKv = HEADS_KV;
    int64_t seqQ = SEQ;
    int64_t seqKv = SEQ;
    int64_t headSize = HEAD_SIZE;
    data_objects::DataType dtype = data_objects::DataType::BFLOAT16;

    // Mask controls
    bool causalMaskDeprecated = false;
    bool bottomRightDeprecated = false;
    int64_t leftBound = -1; // -1 = unset
    int64_t rightBound = -2; // -2 = use alignment default (right=0, BOTTOM_RIGHT)
    data_objects::DiagonalAlignment alignment = data_objects::DiagonalAlignment::BOTTOM_RIGHT;

    // Override strides for specific operands.
    std::vector<int64_t> qStrideOverride;
    std::vector<int64_t> kStrideOverride;
    std::vector<int64_t> vStrideOverride;
    std::vector<int64_t> oStrideOverride;

    // Optional attribute uids for decline testing.
    bool useAttnMask = false;
    bool useScaleTensor = false;
    bool useVarlen = false;
    bool useSinkTensor = false;
    bool useAlibi = false;
    bool usePaddingMask = false;
    bool useFp8 = false;
    bool twoNodes = false;
    bool pointwiseOnly = false;
    float scale = SCALE;
    data_objects::DataType mmaCoreMode = data_objects::DataType::UNSET;
    data_objects::AttentionImplementation impl
        = data_objects::AttentionImplementation::AUTO;
};

flatbuffers::FlatBufferBuilder buildGraph(const GraphSpec& spec)
{
    flatbuffers::FlatBufferBuilder b(8192);

    auto qDims = std::vector<int64_t>{spec.batch, spec.headsQ, spec.seqQ, spec.headSize};
    auto kvDims = std::vector<int64_t>{spec.batch, spec.headsKv, spec.seqKv, spec.headSize};

    auto dflt_q = bshdStrides(spec.headsQ, spec.seqQ, spec.headSize);
    auto dflt_k = bshdStrides(spec.headsKv, spec.seqKv, spec.headSize);
    const auto& qStrides = spec.qStrideOverride.empty() ? dflt_q : spec.qStrideOverride;
    const auto& kStrides = spec.kStrideOverride.empty() ? dflt_k : spec.kStrideOverride;
    const auto& vStrides = spec.vStrideOverride.empty() ? dflt_k : spec.vStrideOverride;
    const auto& oStrides = spec.oStrideOverride.empty() ? dflt_q : spec.oStrideOverride;

    auto qT = data_objects::CreateTensorAttributesDirect(
        b, Q_UID, nullptr, spec.dtype, &qStrides, &qDims, false);
    auto kT = data_objects::CreateTensorAttributesDirect(
        b, K_UID, nullptr, spec.dtype, &kStrides, &kvDims, false);
    auto vT = data_objects::CreateTensorAttributesDirect(
        b, V_UID, nullptr, spec.dtype, &vStrides, &kvDims, false);
    auto oT = data_objects::CreateTensorAttributesDirect(
        b, O_UID, nullptr, spec.dtype, &oStrides, &qDims, false);

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors{qT, kT, vT, oT};

    data_objects::SdpaAttributesBuilder sdpa(b);
    sdpa.add_q_tensor_uid(Q_UID);
    sdpa.add_k_tensor_uid(K_UID);
    sdpa.add_v_tensor_uid(V_UID);
    sdpa.add_o_tensor_uid(O_UID);
    sdpa.add_causal_mask(spec.causalMaskDeprecated);
    sdpa.add_causal_mask_bottom_right(spec.bottomRightDeprecated);
    if(spec.leftBound != -1)
    {
        sdpa.add_left_bound(spec.leftBound);
    }
    if(spec.rightBound == -2)
    {
        sdpa.add_right_bound(0);
    }
    else if(spec.rightBound != -1)
    {
        sdpa.add_right_bound(spec.rightBound);
    }
    sdpa.add_diagonal_alignment(spec.alignment);
    sdpa.add_attn_scale_value(spec.scale);
    if(spec.useAttnMask)
    {
        sdpa.add_attn_mask_tensor_uid(99);
    }
    if(spec.useScaleTensor)
    {
        sdpa.add_scale_tensor_uid(99);
    }
    if(spec.useVarlen)
    {
        sdpa.add_seq_len_q_tensor_uid(99);
    }
    if(spec.useSinkTensor)
    {
        sdpa.add_sink_token_tensor_uid(99);
    }
    sdpa.add_alibi_mask(spec.useAlibi);
    sdpa.add_padding_mask(spec.usePaddingMask);
    sdpa.add_mma_core_mode(spec.mmaCoreMode);
    sdpa.add_implementation(spec.impl);
    auto sdpaAttr = sdpa.Finish();

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    if(!spec.pointwiseOnly)
    {
        auto node = data_objects::CreateNodeDirect(
            b,
            "sdpa",
            data_objects::DataType::FLOAT,
            data_objects::NodeAttributes::SdpaAttributes,
            sdpaAttr.Union());
        nodes.push_back(node);
    }
    if(spec.twoNodes || spec.pointwiseOnly)
    {
        // Add a non-SDPA node to trigger the "not a single SDPA" decline.
        data_objects::PointwiseAttributesBuilder pw(b);
        auto pwAttr = pw.Finish();
        auto node2 = data_objects::CreateNodeDirect(
            b,
            "relu",
            data_objects::DataType::FLOAT,
            data_objects::NodeAttributes::PointwiseAttributes,
            pwAttr.Union());
        nodes.push_back(node2);
    }

    auto graph = data_objects::CreateGraphDirect(
        b, "gfx950_attention_unified_2d_test", &tensors, &nodes);
    b.Finish(graph);
    return b;
}

std::optional<BoundTokens> matchGraph(const GraphSpec& spec)
{
    auto builder = buildGraph(spec);
    hipdnn_flatbuffers_sdk::GraphWrapper graph(builder.GetBufferPointer(),
                                               builder.GetSize());
    hipdnn_flatbuffers_sdk::DeviceProperties props;
    props.gcnArchName = "gfx950";
    props.warpSize = 64;
    MatchContext context{graph, 0, props};

    auto matcher = GraphMatchRegistry::resolve(std::string(GRAPH_MATCHER_SYMBOL));
    return (*matcher)(context);
}

// Canonical accepted graph spec (bottom-right causal, 32/8 GQA, d128).
GraphSpec canonical()
{
    return GraphSpec{};
}

KernelDefinition makeKernel(const std::string& dtype = "BF16",
                             int64_t headSize = HEAD_SIZE,
                             int64_t numQHeads = HEADS_Q,
                             int64_t numKvHeads = HEADS_KV,
                             int64_t blockSize = BLOCK_SIZE,
                             int64_t slidingWindow = 0,
                             int64_t useSinks = 0,
                             int64_t numSeqs = 256)
{
    KernelDefinition k;
    k.kernelId.engineId = "00000000-0000-4000-8000-0000000000a1";
    k.kernelId.packId = "00000000-0000-4000-8000-0000000000a2";
    k.kernelId.kernelId = "00000000-0000-4000-8000-0000000000a3";
    k.name = "attention_unified_2d.bf16_d128_hq32_kv8_bs16_t16_nw1_mw16.gfx950";
    k.source.kind = KernelSourceKind::KPACK;
    k.source.library = "fake.kpack";
    k.source.tocKey = "toc#0";
    k.source.symbol = "attention_unified_2d";
    k.originDirectory = "/nonexistent";
    k.treeRoot = "/nonexistent";
    k.metadata["dtype"] = std::string(dtype);
    k.metadata["head_size"] = int64_t{headSize};
    k.metadata["num_query_heads"] = int64_t{numQHeads};
    k.metadata["num_kv_heads"] = int64_t{numKvHeads};
    k.metadata["block_size"] = int64_t{blockSize};
    k.metadata["sliding_window"] = int64_t{slidingWindow};
    k.metadata["use_sinks"] = int64_t{useSinks};
    k.metadata["num_seqs"] = int64_t{numSeqs};
    k.metadata["tile_size"] = int64_t{blockSize};
    k.metadata["num_warps"] = int64_t{1};
    k.metadata["block_m_per_warp"] = int64_t{16};
    k.metadata["use_mfma_32x32"] = int64_t{0};
    k.metadata["use_transposed_qk_32x32"] = int64_t{0};
    k.metadata["use_register_pv"] = int64_t{0};
    k.metadata["use_k_single_buffer"] = int64_t{0};
    return k;
}

bool matchesKernel(const GraphSpec& gs, const KernelDefinition& k)
{
    auto builder = buildGraph(gs);
    hipdnn_flatbuffers_sdk::GraphWrapper graph(builder.GetBufferPointer(),
                                               builder.GetSize());
    hipdnn_flatbuffers_sdk::DeviceProperties props;
    props.gcnArchName = "gfx950";
    props.warpSize = 64;
    MatchContext context{graph, 0, props};

    auto matcher = GraphMatchRegistry::resolve(std::string(GRAPH_MATCHER_SYMBOL));
    auto bound = (*matcher)(context);
    if(!bound.has_value())
    {
        return false;
    }
    auto km = KernelMatchRegistry::resolve(std::string(KERNEL_MATCHER_SYMBOL));
    return (*km)(context, *bound, k);
}

} // namespace

// ---------------------------------------------------------------------------
// TestGfx950AttentionUnified2DGraphMatch
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionUnified2DGraphMatch, AcceptsBottomRightCausalGraph)
{
    registerNativeIngestorSymbols();
    EXPECT_TRUE(matchGraph(canonical()).has_value());
}

TEST(TestGfx950AttentionUnified2DGraphMatch, AcceptsTopLeftCausalWhenSeqLensEqual)
{
    registerNativeIngestorSymbols();
    GraphSpec s = canonical();
    s.rightBound = -1; // no right_bound => top-left path
    s.alignment = data_objects::DiagonalAlignment::TOP_LEFT;
    // seqQ == seqKv, so top-left == bottom-right numerically
    EXPECT_TRUE(matchGraph(s).has_value());
}

TEST(TestGfx950AttentionUnified2DGraphMatch, AcceptsFp16)
{
    registerNativeIngestorSymbols();
    GraphSpec s = canonical();
    s.dtype = data_objects::DataType::HALF;
    EXPECT_TRUE(matchGraph(s).has_value());
}

TEST(TestGfx950AttentionUnified2DGraphMatch, AcceptsHeadSize64)
{
    registerNativeIngestorSymbols();
    GraphSpec s = canonical();
    s.headSize = 64;
    EXPECT_TRUE(matchGraph(s).has_value());
}

TEST(TestGfx950AttentionUnified2DGraphMatch, AcceptsHeadSize256)
{
    registerNativeIngestorSymbols();
    GraphSpec s = canonical();
    s.headSize = 256;
    EXPECT_TRUE(matchGraph(s).has_value());
}

TEST(TestGfx950AttentionUnified2DGraphMatch, AcceptsGroupedQueryAttention)
{
    registerNativeIngestorSymbols();
    GraphSpec s = canonical();
    s.headsQ = 32;
    s.headsKv = 4;
    EXPECT_TRUE(matchGraph(s).has_value());
}

TEST(TestGfx950AttentionUnified2DGraphMatch, DeclinesNoMask)
{
    registerNativeIngestorSymbols();
    GraphSpec s = canonical();
    s.rightBound = -1;  // no bound at all => NO_MASK
    s.alignment = data_objects::DiagonalAlignment::TOP_LEFT;
    s.seqQ = 512; s.seqKv = 1024; // seqQ != seqKv so top-left != bottom-right
    // Actually easier: use NO_MASK by leaving right_bound unset
    GraphSpec s2 = canonical();
    // Force NO_MASK path: no right_bound, no deprecated bool, aligned TOP_LEFT
    s2.rightBound = -1;
    s2.alignment = data_objects::DiagonalAlignment::TOP_LEFT;
    s2.seqQ = 512; s2.seqKv = 512;
    // When seqQ==seqKv top-left==bottom-right, so use different seqLens to force NO_MASK
    // Actually the cleanest: don't set right_bound (UNBOUNDED) and alignment TOP_LEFT
    // => maskTypeFor returns NO_MASK
    GraphSpec nm;
    nm.rightBound = -1; // no right bound => UNBOUNDED => NO_MASK
    nm.alignment = data_objects::DiagonalAlignment::TOP_LEFT;
    EXPECT_FALSE(matchGraph(nm).has_value());
}

TEST(TestGfx950AttentionUnified2DGraphMatch, DeclinesTopLeftCausalWhenSeqLensDiffer)
{
    registerNativeIngestorSymbols();
    GraphSpec s = canonical();
    s.rightBound = -1;
    s.alignment = data_objects::DiagonalAlignment::TOP_LEFT;
    s.seqQ = 128;
    s.seqKv = 256;
    // Resize O to match Q
    s.qStrideOverride = bshdStrides(s.headsQ, s.seqQ, s.headSize);
    s.oStrideOverride = bshdStrides(s.headsQ, s.seqQ, s.headSize);
    EXPECT_FALSE(matchGraph(s).has_value());
}

TEST(TestGfx950AttentionUnified2DGraphMatch, DeclinesUnsupportedHeadSize)
{
    registerNativeIngestorSymbols();
    GraphSpec s = canonical();
    s.headSize = 192;
    EXPECT_FALSE(matchGraph(s).has_value());
}

TEST(TestGfx950AttentionUnified2DGraphMatch, DeclinesFloat32Dtype)
{
    registerNativeIngestorSymbols();
    GraphSpec s = canonical();
    s.dtype = data_objects::DataType::FLOAT;
    EXPECT_FALSE(matchGraph(s).has_value());
}

TEST(TestGfx950AttentionUnified2DGraphMatch, DeclinesNonDivisibleGqa)
{
    registerNativeIngestorSymbols();
    GraphSpec s = canonical();
    s.headsQ = 7;
    s.headsKv = 4;
    EXPECT_FALSE(matchGraph(s).has_value());
}

TEST(TestGfx950AttentionUnified2DGraphMatch, DeclinesBhsdQ)
{
    registerNativeIngestorSymbols();
    GraphSpec s = canonical();
    // BHSD: B*H*S*D, H*S*D, S*D, D
    s.qStrideOverride
        = {HEADS_Q * SEQ * HEAD_SIZE, SEQ * HEAD_SIZE, HEAD_SIZE, 1};
    EXPECT_FALSE(matchGraph(s).has_value());
}

TEST(TestGfx950AttentionUnified2DGraphMatch, DeclinesBhsdK)
{
    registerNativeIngestorSymbols();
    GraphSpec s = canonical();
    s.kStrideOverride
        = {HEADS_KV * SEQ * HEAD_SIZE, SEQ * HEAD_SIZE, HEAD_SIZE, 1};
    EXPECT_FALSE(matchGraph(s).has_value());
}

TEST(TestGfx950AttentionUnified2DGraphMatch, DeclinesAttnMaskTensor)
{
    registerNativeIngestorSymbols();
    GraphSpec s = canonical();
    s.useAttnMask = true;
    EXPECT_FALSE(matchGraph(s).has_value());
}

TEST(TestGfx950AttentionUnified2DGraphMatch, DeclinesScaleTensor)
{
    registerNativeIngestorSymbols();
    GraphSpec s = canonical();
    s.useScaleTensor = true;
    EXPECT_FALSE(matchGraph(s).has_value());
}

TEST(TestGfx950AttentionUnified2DGraphMatch, DeclinesVarlen)
{
    registerNativeIngestorSymbols();
    GraphSpec s = canonical();
    s.useVarlen = true;
    EXPECT_FALSE(matchGraph(s).has_value());
}

TEST(TestGfx950AttentionUnified2DGraphMatch, DeclinesSinkToken)
{
    registerNativeIngestorSymbols();
    GraphSpec s = canonical();
    s.useSinkTensor = true;
    EXPECT_FALSE(matchGraph(s).has_value());
}

TEST(TestGfx950AttentionUnified2DGraphMatch, DeclinesAlibi)
{
    registerNativeIngestorSymbols();
    GraphSpec s = canonical();
    s.useAlibi = true;
    EXPECT_FALSE(matchGraph(s).has_value());
}

TEST(TestGfx950AttentionUnified2DGraphMatch, DeclinesTwoNodes)
{
    registerNativeIngestorSymbols();
    GraphSpec s = canonical();
    s.twoNodes = true;
    EXPECT_FALSE(matchGraph(s).has_value());
}

TEST(TestGfx950AttentionUnified2DGraphMatch, DeclinesPointwiseOnly)
{
    registerNativeIngestorSymbols();
    GraphSpec s = canonical();
    s.pointwiseOnly = true;
    EXPECT_FALSE(matchGraph(s).has_value());
}

// ---------------------------------------------------------------------------
// TestGfx950AttentionUnified2DKernelMatch
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionUnified2DKernelMatch, AcceptsMatchingKernel)
{
    registerNativeIngestorSymbols();
    EXPECT_TRUE(matchesKernel(canonical(), makeKernel()));
}

TEST(TestGfx950AttentionUnified2DKernelMatch, AcceptsFp16Kernel)
{
    registerNativeIngestorSymbols();
    GraphSpec s = canonical();
    s.dtype = data_objects::DataType::HALF;
    EXPECT_TRUE(matchesKernel(s, makeKernel("FP16")));
}

TEST(TestGfx950AttentionUnified2DKernelMatch, RejectsDtypeMismatch)
{
    registerNativeIngestorSymbols();
    // Graph is BF16, kernel says FP16
    EXPECT_FALSE(matchesKernel(canonical(), makeKernel("FP16")));
}

TEST(TestGfx950AttentionUnified2DKernelMatch, RejectsHeadSizeMismatch)
{
    registerNativeIngestorSymbols();
    EXPECT_FALSE(matchesKernel(canonical(), makeKernel("BF16", 64)));
}

TEST(TestGfx950AttentionUnified2DKernelMatch, RejectsSeqLenKvNotDivisibleByBlockSize)
{
    registerNativeIngestorSymbols();
    // seqKv=256 / blockSize=32 = 8 (ok), seqKv=256 / blockSize=17 = not integer
    EXPECT_FALSE(matchesKernel(canonical(), makeKernel("BF16", HEAD_SIZE, HEADS_Q, HEADS_KV, 17)));
}

TEST(TestGfx950AttentionUnified2DKernelMatch, RejectsBatchExceedsCapacity)
{
    registerNativeIngestorSymbols();
    // numSeqs=1 -> capacity = 2^ceil(log2(2)) - 1 = 1, batch=2 > 1
    EXPECT_FALSE(matchesKernel(canonical(),
                               makeKernel("BF16", HEAD_SIZE, HEADS_Q, HEADS_KV, BLOCK_SIZE, 0, 0, 1)));
}

TEST(TestGfx950AttentionUnified2DKernelMatch, AcceptsWhenBatchEqualsCapacity)
{
    registerNativeIngestorSymbols();
    // numSeqs=3 -> iterations=ceil(log2(4))=2 -> capacity=3; batch=2 <= 3
    EXPECT_TRUE(matchesKernel(canonical(),
                              makeKernel("BF16", HEAD_SIZE, HEADS_Q, HEADS_KV, BLOCK_SIZE, 0, 0, 3)));
}

TEST(TestGfx950AttentionUnified2DKernelMatch, RejectsBakedSlidingWindow)
{
    registerNativeIngestorSymbols();
    // Kernel bakes a sliding window (!=0), graph didn't ask for one
    EXPECT_FALSE(matchesKernel(canonical(),
                               makeKernel("BF16", HEAD_SIZE, HEADS_Q, HEADS_KV, BLOCK_SIZE, 256)));
}

TEST(TestGfx950AttentionUnified2DKernelMatch, RejectsBakedSinks)
{
    registerNativeIngestorSymbols();
    EXPECT_FALSE(matchesKernel(canonical(),
                               makeKernel("BF16", HEAD_SIZE, HEADS_Q, HEADS_KV, BLOCK_SIZE, 0, 1)));
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
