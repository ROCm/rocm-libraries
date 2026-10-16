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
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
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
constexpr int64_t RAGGED_OFFSET_UID = 5;
// A uid no tensor in the graph carries, for the optional attributes the matcher declines
// on presence alone.
constexpr int64_t ABSENT_UID = 99;
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
    // Operand uid that carries a ragged (THD) offset table, if any.
    std::optional<int64_t> raggedOperand;
    bool twoNodes = false;
    bool pointwiseOnly = false;
    float scale = SCALE;
    data_objects::DataType mmaCoreMode = data_objects::DataType::UNSET;
    data_objects::AttentionImplementation impl = data_objects::AttentionImplementation::AUTO;
};

flatbuffers::FlatBufferBuilder buildGraph(const GraphSpec& spec)
{
    flatbuffers::FlatBufferBuilder b(8192);

    auto qDims = std::vector<int64_t>{spec.batch, spec.headsQ, spec.seqQ, spec.headSize};
    auto kvDims = std::vector<int64_t>{spec.batch, spec.headsKv, spec.seqKv, spec.headSize};

    auto defaultQStrides = bshdStrides(spec.headsQ, spec.seqQ, spec.headSize);
    auto defaultKStrides = bshdStrides(spec.headsKv, spec.seqKv, spec.headSize);
    const auto& qStrides = spec.qStrideOverride.empty() ? defaultQStrides : spec.qStrideOverride;
    const auto& kStrides = spec.kStrideOverride.empty() ? defaultKStrides : spec.kStrideOverride;
    const auto& vStrides = spec.vStrideOverride.empty() ? defaultKStrides : spec.vStrideOverride;
    const auto& oStrides = spec.oStrideOverride.empty() ? defaultQStrides : spec.oStrideOverride;

    const auto tensorFor = [&](int64_t uid,
                               data_objects::DataType dtype,
                               const std::vector<int64_t>& strides,
                               const std::vector<int64_t>& dims) {
        const ::flatbuffers::Optional<int64_t> ragged
            = spec.raggedOperand == uid ? ::flatbuffers::Optional<int64_t>(RAGGED_OFFSET_UID)
                                        : ::flatbuffers::nullopt;
        return data_objects::CreateTensorAttributesDirect(b,
                                                          uid,
                                                          nullptr,
                                                          dtype,
                                                          &strides,
                                                          &dims,
                                                          false,
                                                          data_objects::TensorValue::NONE,
                                                          0,
                                                          false,
                                                          ragged);
    };

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors{
        tensorFor(Q_UID, spec.dtype, qStrides, qDims),
        tensorFor(K_UID, spec.dtype, kStrides, kvDims),
        tensorFor(V_UID, spec.dtype, vStrides, kvDims),
        tensorFor(O_UID, spec.dtype, oStrides, qDims),
    };
    if(spec.raggedOperand.has_value())
    {
        const std::vector<int64_t> offsetDims{spec.batch + 1, 1, 1, 1};
        const std::vector<int64_t> offsetStrides{1, 1, 1, 1};
        tensors.push_back(
            tensorFor(RAGGED_OFFSET_UID, data_objects::DataType::INT32, offsetStrides, offsetDims));
    }

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
        sdpa.add_attn_mask_tensor_uid(ABSENT_UID);
    }
    if(spec.useScaleTensor)
    {
        sdpa.add_scale_tensor_uid(ABSENT_UID);
    }
    if(spec.useVarlen)
    {
        sdpa.add_seq_len_q_tensor_uid(ABSENT_UID);
    }
    if(spec.useSinkTensor)
    {
        sdpa.add_sink_token_tensor_uid(ABSENT_UID);
    }
    if(spec.useFp8)
    {
        sdpa.add_descale_q_tensor_uid(ABSENT_UID);
    }
    sdpa.add_alibi_mask(spec.useAlibi);
    sdpa.add_padding_mask(spec.usePaddingMask);
    sdpa.add_mma_core_mode(spec.mmaCoreMode);
    sdpa.add_implementation(spec.impl);
    auto sdpaAttr = sdpa.Finish();

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    if(!spec.pointwiseOnly)
    {
        auto node = data_objects::CreateNodeDirect(b,
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
        auto node2
            = data_objects::CreateNodeDirect(b,
                                             "relu",
                                             data_objects::DataType::FLOAT,
                                             data_objects::NodeAttributes::PointwiseAttributes,
                                             pwAttr.Union());
        nodes.push_back(node2);
    }

    auto graph = data_objects::CreateGraphDirect(b,
                                                 "gfx950_attention_unified_2d_test",
                                                 data_objects::DataType::UNSET,
                                                 data_objects::DataType::UNSET,
                                                 data_objects::DataType::UNSET,
                                                 &tensors,
                                                 &nodes);
    b.Finish(graph);
    return b;
}

std::optional<BoundTokens> matchGraph(const GraphSpec& spec)
{
    auto builder = buildGraph(spec);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    DeviceProperties props;
    props.gcnArchName = "gfx950";
    props.warpSize = 64;
    const MatchContext context{graph, 0, props};

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
    k.dispatchId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-0000000000a1");
    k.packId = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-0000000000a2");
    k.kernelId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-0000000000a3");
    k.name = "attention_unified_2d.bf16_d128_hq32_kv8_bs16_t16_nw1_mw16.gfx950";
    k.source.kind = KernelSourceKind::KPACK;
    k.source.library = "fake.kpack";
    k.source.tocKey = "toc#0";
    k.source.symbol = "attention_unified_2d";
    k.originDirectory = "/nonexistent";
    k.treeRoot = "/nonexistent";
    k.metadata["dtype"] = std::string(dtype);
    k.metadata["head_size"] = headSize;
    k.metadata["num_query_heads"] = numQHeads;
    k.metadata["num_kv_heads"] = numKvHeads;
    k.metadata["block_size"] = blockSize;
    k.metadata["sliding_window"] = slidingWindow;
    k.metadata["use_sinks"] = useSinks;
    k.metadata["num_seqs"] = numSeqs;
    k.metadata["tile_size"] = blockSize;
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
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    DeviceProperties props;
    props.gcnArchName = "gfx950";
    props.warpSize = 64;
    const MatchContext context{graph, 0, props};

    auto matcher = GraphMatchRegistry::resolve(std::string(GRAPH_MATCHER_SYMBOL));
    auto bound = (*matcher)(context);
    if(!bound.has_value())
    {
        return false;
    }
    auto km = KernelMatcherRegistry::resolve(std::string(KERNEL_MATCHER_SYMBOL));
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
    // right_bound 0 (the default) with TOP_LEFT alignment is hipDNN's top-left causal mask.
    s.alignment = data_objects::DiagonalAlignment::TOP_LEFT;
    // seqQ == seqKv, so top-left == bottom-right numerically
    EXPECT_TRUE(matchGraph(s).has_value());
}

TEST(TestGfx950AttentionUnified2DGraphMatch, AcceptsHalfPrecision)
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
    // No right_bound at all: an unmasked graph, whatever the alignment says.
    s.rightBound = -1;
    EXPECT_FALSE(matchGraph(s).has_value());
}

TEST(TestGfx950AttentionUnified2DGraphMatch, DeclinesTopLeftCausalWhenSeqLensDiffer)
{
    registerNativeIngestorSymbols();
    GraphSpec s = canonical();
    // right_bound 0 (the default) with TOP_LEFT alignment is hipDNN's top-left causal mask,
    // which the bottom-right kernel reproduces only when seqlen_q == seqlen_kv.
    s.alignment = data_objects::DiagonalAlignment::TOP_LEFT;
    s.seqQ = 128;
    s.seqKv = 256;
    EXPECT_FALSE(matchGraph(s).has_value());
}

TEST(TestGfx950AttentionUnified2DGraphMatch, AcceptsBottomRightCausalWhenQueryIsShorterThanKv)
{
    registerNativeIngestorSymbols();
    GraphSpec s = canonical();
    s.seqQ = 128;
    s.seqKv = 256;
    EXPECT_TRUE(matchGraph(s).has_value());
}

TEST(TestGfx950AttentionUnified2DGraphMatch, DeclinesBottomRightCausalWhenQueryIsLongerThanKv)
{
    registerNativeIngestorSymbols();
    GraphSpec s = canonical();
    s.seqQ = 256;
    s.seqKv = 128;
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
    s.qStrideOverride = {HEADS_Q * SEQ * HEAD_SIZE, SEQ * HEAD_SIZE, HEAD_SIZE, 1};
    EXPECT_FALSE(matchGraph(s).has_value());
}

TEST(TestGfx950AttentionUnified2DGraphMatch, DeclinesBhsdK)
{
    registerNativeIngestorSymbols();
    GraphSpec s = canonical();
    s.kStrideOverride = {HEADS_KV * SEQ * HEAD_SIZE, SEQ * HEAD_SIZE, HEAD_SIZE, 1};
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

TEST(TestGfx950AttentionUnified2DGraphMatch, DeclinesEachUnsupportedGraphFeature)
{
    registerNativeIngestorSymbols();

    struct Case
    {
        const char* feature;
        void (*apply)(GraphSpec&);
    };
    const std::vector<Case> cases = {
        {"left_bound (sliding window)", [](GraphSpec& s) { s.leftBound = 128; }},
        {"bidirectional right_bound", [](GraphSpec& s) { s.rightBound = 4; }},
        {"both deprecated causal booleans",
         [](GraphSpec& s) {
             s.causalMaskDeprecated = true;
             s.bottomRightDeprecated = true;
         }},
        {"deprecated top-left causal with seqlen_q != seqlen_kv",
         [](GraphSpec& s) {
             s.causalMaskDeprecated = true;
             s.seqQ = 128;
         }},
        {"padding_mask", [](GraphSpec& s) { s.usePaddingMask = true; }},
        {"fp8 scaling tensor", [](GraphSpec& s) { s.useFp8 = true; }},
        {"mma_core_mode FLOAT",
         [](GraphSpec& s) { s.mmaCoreMode = data_objects::DataType::FLOAT; }},
        {"implementation COMPOSITE",
         [](GraphSpec& s) { s.impl = data_objects::AttentionImplementation::COMPOSITE; }},
        {"implementation UNIFIED",
         [](GraphSpec& s) { s.impl = data_objects::AttentionImplementation::UNIFIED; }},
    };

    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.feature);
        GraphSpec s = canonical();
        c.apply(s);
        EXPECT_FALSE(matchGraph(s).has_value());
    }
}

TEST(TestGfx950AttentionUnified2DGraphMatch, AcceptsAnMmaCoreModeTheKernelRuns)
{
    registerNativeIngestorSymbols();
    for(const auto mode : {data_objects::DataType::HALF, data_objects::DataType::BFLOAT16})
    {
        GraphSpec s = canonical();
        s.mmaCoreMode = mode;
        EXPECT_TRUE(matchGraph(s).has_value()) << "mma_core_mode " << static_cast<int>(mode);
    }
}

// A ragged operand addresses each batch from a device offset table; the engine builds a
// uniform dense layout and never reads it.
TEST(TestGfx950AttentionUnified2DGraphMatch, DeclinesRaggedOperands)
{
    registerNativeIngestorSymbols();
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        GraphSpec s = canonical();
        s.raggedOperand = uid;
        EXPECT_FALSE(matchGraph(s).has_value()) << "ragged operand uid " << uid;
    }
}

// ---------------------------------------------------------------------------
// TestGfx950AttentionUnified2DKernelMatch
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionUnified2DKernelMatch, AcceptsMatchingKernel)
{
    registerNativeIngestorSymbols();
    EXPECT_TRUE(matchesKernel(canonical(), makeKernel()));
}

TEST(TestGfx950AttentionUnified2DKernelMatch, AcceptsHalfPrecisionKernel)
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

// A kernel baked at num_seqs=3 runs 2 search iterations, so its capacity is 2^2 - 1 = 3.
TEST(TestGfx950AttentionUnified2DKernelMatch, AcceptsWhenBatchEqualsCapacity)
{
    registerNativeIngestorSymbols();
    GraphSpec s = canonical();
    s.batch = 3;
    EXPECT_TRUE(matchesKernel(
        s, makeKernel("BF16", HEAD_SIZE, HEADS_Q, HEADS_KV, BLOCK_SIZE, 0, 0, /* numSeqs= */ 3)));
}

TEST(TestGfx950AttentionUnified2DKernelMatch, RejectsBatchOneAboveCapacity)
{
    registerNativeIngestorSymbols();
    GraphSpec s = canonical();
    s.batch = 4;
    EXPECT_FALSE(matchesKernel(
        s, makeKernel("BF16", HEAD_SIZE, HEADS_Q, HEADS_KV, BLOCK_SIZE, 0, 0, /* numSeqs= */ 3)));
}

TEST(TestGfx950AttentionUnified2DKernelMatch, RejectsNumQueryHeadsMismatch)
{
    registerNativeIngestorSymbols();
    EXPECT_FALSE(matchesKernel(canonical(), makeKernel("BF16", HEAD_SIZE, HEADS_Q / 2, HEADS_KV)));
}

TEST(TestGfx950AttentionUnified2DKernelMatch, RejectsNumKvHeadsMismatch)
{
    registerNativeIngestorSymbols();
    EXPECT_FALSE(matchesKernel(canonical(), makeKernel("BF16", HEAD_SIZE, HEADS_Q, HEADS_KV / 2)));
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
