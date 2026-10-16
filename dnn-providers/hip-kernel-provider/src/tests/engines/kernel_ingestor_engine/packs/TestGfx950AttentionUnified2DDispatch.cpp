// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionUnified2DGeometry.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

constexpr std::string_view GRAPH_MATCHER_SYMBOL
    = "hipkernel.gfx950_attention_unified2_d.graph_match";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.gfx950_attention_unified2_d.dispatch";

constexpr int64_t Q_UID = 11, K_UID = 22, V_UID = 33, O_UID = 44, BHSD_UID = 55;
constexpr int64_t BATCH = 2, HEADS_Q = 32, HEADS_KV = 8, SEQ = 256, HEAD_SIZE = 128;
constexpr int64_t BLOCK_SIZE = 16;
constexpr float SCALE = 0.08838834764831843F;

constexpr std::string_view O_TOKEN = "gfx950_attention_unified_2d.o.uid";

constexpr const char* MISSING_TOKEN_MARKER = "missing bound token";
constexpr const char* OUTPUT_LAYOUT_MARKER = "not dense BSHD";

std::vector<int64_t> bshdStrides(int64_t h, int64_t s, int64_t d)
{
    return {s * h * d, d, h * d, 1};
}

flatbuffers::FlatBufferBuilder buildGraph(float scale = SCALE,
                                          data_objects::DiagonalAlignment alignment
                                          = data_objects::DiagonalAlignment::BOTTOM_RIGHT,
                                          bool withBhsdTensor = false)
{
    flatbuffers::FlatBufferBuilder b(4096);

    auto bshd = bshdStrides(HEADS_Q, SEQ, HEAD_SIZE);
    auto bshdKv = bshdStrides(HEADS_KV, SEQ, HEAD_SIZE);
    const std::vector<int64_t> qDims{BATCH, HEADS_Q, SEQ, HEAD_SIZE};
    const std::vector<int64_t> kvDims{BATCH, HEADS_KV, SEQ, HEAD_SIZE};

    // An extra BHSD tensor no node references, so a bound O token can be made to name a
    // real but wrongly-strided tensor (graph_match itself declines a BHSD O).
    const std::vector<int64_t> bhsd{
        BATCH * HEADS_Q * SEQ * HEAD_SIZE, SEQ * HEAD_SIZE, HEAD_SIZE, 1};

    auto qT = data_objects::CreateTensorAttributesDirect(
        b, Q_UID, nullptr, data_objects::DataType::BFLOAT16, &bshd, &qDims, false);
    auto kT = data_objects::CreateTensorAttributesDirect(
        b, K_UID, nullptr, data_objects::DataType::BFLOAT16, &bshdKv, &kvDims, false);
    auto vT = data_objects::CreateTensorAttributesDirect(
        b, V_UID, nullptr, data_objects::DataType::BFLOAT16, &bshdKv, &kvDims, false);
    auto oT = data_objects::CreateTensorAttributesDirect(
        b, O_UID, nullptr, data_objects::DataType::BFLOAT16, &bshd, &qDims, false);
    auto bhsdT = data_objects::CreateTensorAttributesDirect(
        b, BHSD_UID, nullptr, data_objects::DataType::BFLOAT16, &bhsd, &qDims, false);

    data_objects::SdpaAttributesBuilder sdpa(b);
    sdpa.add_q_tensor_uid(Q_UID);
    sdpa.add_k_tensor_uid(K_UID);
    sdpa.add_v_tensor_uid(V_UID);
    sdpa.add_o_tensor_uid(O_UID);
    sdpa.add_right_bound(0);
    sdpa.add_diagonal_alignment(alignment);
    sdpa.add_attn_scale_value(scale);
    sdpa.add_alibi_mask(false);
    sdpa.add_padding_mask(false);
    sdpa.add_mma_core_mode(data_objects::DataType::UNSET);
    sdpa.add_implementation(data_objects::AttentionImplementation::AUTO);
    auto sdpaAttr = sdpa.Finish();
    auto node = data_objects::CreateNodeDirect(b,
                                               "sdpa",
                                               data_objects::DataType::FLOAT,
                                               data_objects::NodeAttributes::SdpaAttributes,
                                               sdpaAttr.Union());

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors{qT, kT, vT, oT};
    if(withBhsdTensor)
    {
        tensors.push_back(bhsdT);
    }
    const std::vector<flatbuffers::Offset<data_objects::Node>> nodes{node};
    auto graph = data_objects::CreateGraphDirect(b,
                                                 "gfx950_attention_unified_2d_dispatch",
                                                 data_objects::DataType::UNSET,
                                                 data_objects::DataType::UNSET,
                                                 data_objects::DataType::UNSET,
                                                 &tensors,
                                                 &nodes);
    b.Finish(graph);
    return b;
}

struct AttentionGraph
{
    flatbuffers::FlatBufferBuilder builder;
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph;
    DeviceProperties properties;

    explicit AttentionGraph(float scale = SCALE,
                            data_objects::DiagonalAlignment alignment
                            = data_objects::DiagonalAlignment::BOTTOM_RIGHT,
                            bool withBhsdTensor = false)
        : builder(buildGraph(scale, alignment, withBhsdTensor))
        , graph(builder.GetBufferPointer(), builder.GetSize())
        , properties([]() {
            DeviceProperties p;
            p.gcnArchName = "gfx950";
            p.warpSize = 64;
            return p;
        }())
    {
    }

    MatchContext context() const
    {
        return MatchContext{graph, 0, properties};
    }
};

const IKernelDispatchHandler<Handle>* dispatchHandler()
{
    return DispatchRegistry<Handle>::resolve(std::string(DISPATCH_SYMBOL));
}

std::optional<BoundTokens> bindingsFor(const AttentionGraph& ag)
{
    auto matcher = GraphMatchRegistry::resolve(std::string(GRAPH_MATCHER_SYMBOL));
    return (*matcher)(ag.context());
}

// A minimal KPACK KernelDefinition with correct metadata.
KernelDefinition
    makeKernel(int64_t blockMPerWarp = 16, int64_t numWarps = 1, int64_t blockSize = BLOCK_SIZE)
{
    KernelDefinition kernel;
    kernel.dispatchId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-0000000000b1");
    kernel.packId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-0000000000b2");
    kernel.kernelId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-0000000000b3");
    kernel.name = "attention_unified_2d.bf16_d128_hq32_kv8_bs16_t16_nw1_mw16.gfx950";
    kernel.source.kind = KernelSourceKind::KPACK;
    kernel.source.library = "there-is-no-archive-here.kpack";
    kernel.source.tocKey = "toc#0";
    kernel.source.symbol = "attention_unified_2d";
    {
        std::vector<KernelArgument> sig;
        sig.reserve(18);
        for(int i = 0; i < 10; ++i)
        {
            sig.push_back({"global_buffer", 8U, static_cast<uint32_t>(i * 8), ""});
        }
        for(int i = 0; i < 5; ++i)
        {
            sig.push_back({"by_value", 4U, static_cast<uint32_t>(80 + i * 4), ""});
        }
        for(int i = 0; i < 3; ++i)
        {
            sig.push_back({"by_value", 4U, static_cast<uint32_t>(100 + i * 4), ""});
        }
        kernel.source.signature = sig;
    }
    kernel.originDirectory = "/nonexistent";
    kernel.treeRoot = "/nonexistent";
    kernel.metadata["dtype"] = std::string("BF16");
    kernel.metadata["head_size"] = HEAD_SIZE;
    kernel.metadata["num_query_heads"] = HEADS_Q;
    kernel.metadata["num_kv_heads"] = HEADS_KV;
    kernel.metadata["block_size"] = blockSize;
    kernel.metadata["sliding_window"] = int64_t{0};
    kernel.metadata["use_sinks"] = int64_t{0};
    kernel.metadata["num_seqs"] = int64_t{256};
    kernel.metadata["tile_size"] = blockSize;
    kernel.metadata["num_warps"] = numWarps;
    kernel.metadata["block_m_per_warp"] = blockMPerWarp;
    kernel.metadata["use_mfma_32x32"] = int64_t{0};
    kernel.metadata["use_transposed_qk_32x32"] = int64_t{0};
    kernel.metadata["use_register_pv"] = int64_t{0};
    kernel.metadata["use_k_single_buffer"] = int64_t{0};
    return kernel;
}

std::string prepareFailure(const AttentionGraph& ag,
                           const BoundTokens& bound,
                           const KernelDefinition& kernel)
{
    auto handler = dispatchHandler();
    try
    {
        handler->prepare(ag.context(), bound, kernel);
    }
    catch(const std::exception& e)
    {
        return e.what();
    }
    return "(no exception)";
}

} // namespace

// ---------------------------------------------------------------------------
// TestGfx950AttentionUnified2DDispatch
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionUnified2DDispatch, DispatchSymbolResolves)
{
    registerNativeIngestorSymbols();
    EXPECT_NE(dispatchHandler(), nullptr);
}

TEST(TestGfx950AttentionUnified2DDispatch, WorkspaceBytesIsNonZero)
{
    registerNativeIngestorSymbols();
    const AttentionGraph ag;
    auto bound = bindingsFor(ag);
    ASSERT_TRUE(bound.has_value());
    auto kernel = makeKernel();
    const size_t ws = dispatchHandler()->workspaceBytes(ag.context(), *bound, kernel);
    EXPECT_GT(ws, 0U);
}

TEST(TestGfx950AttentionUnified2DDispatch, WorkspaceSizeMatchesGeometryComputation)
{
    registerNativeIngestorSymbols();
    const AttentionGraph ag;
    auto bound = bindingsFor(ag);
    ASSERT_TRUE(bound.has_value());
    auto kernel = makeKernel();
    const size_t ws = dispatchHandler()->workspaceBytes(ag.context(), *bound, kernel);
    // Expected: layout from gfx950AttentionUnified2DWorkspace(BATCH, SEQ, SEQ, BLOCK_SIZE)
    const auto layout = gfx950AttentionUnified2DWorkspace(BATCH, SEQ, SEQ, BLOCK_SIZE, "test");
    EXPECT_EQ(ws, static_cast<size_t>(layout.totalBytes));
}

TEST(TestGfx950AttentionUnified2DDispatch, WorkspaceIsEmptyWhenThePageSizeDoesNotTileSeqLenKv)
{
    registerNativeIngestorSymbols();
    const AttentionGraph ag;
    auto bound = bindingsFor(ag);
    ASSERT_TRUE(bound.has_value());
    // SEQ is 256, which a 48-token page does not divide.
    const auto kernel = makeKernel(16, 1, 48);
    EXPECT_EQ(dispatchHandler()->workspaceBytes(ag.context(), *bound, kernel), 0U);
}

// The identity block table maps sequence b, page j to physical page
// b * blocksPerSequence + j, which is what a dense contiguous K/V already is. A wrong
// table addresses the wrong memory with no fault. The case has batch > 1 and
// seqlen_q != seqlen_kv so that a table, length or query offset built from the wrong
// extent or the wrong sequence cannot coincide with the right one.
TEST(TestGfx950AttentionUnified2DDispatch, StagedWorkspaceHoldsTheIdentityTableAndOffsets)
{
    constexpr int64_t STAGED_BATCH = 3;
    constexpr int64_t STAGED_SEQ_Q = 64;
    constexpr int64_t STAGED_SEQ_KV = 128;
    constexpr int64_t BLOCKS_PER_SEQUENCE = STAGED_SEQ_KV / BLOCK_SIZE; // 8
    constexpr int64_t INT32_BYTES = 4;

    const auto layout = gfx950AttentionUnified2DWorkspace(
        STAGED_BATCH, STAGED_SEQ_Q, STAGED_SEQ_KV, BLOCK_SIZE, "test");
    const auto staged
        = gfx950AttentionUnified2DStageWorkspace(STAGED_BATCH, STAGED_SEQ_Q, STAGED_SEQ_KV, layout);

    // Everything outside the three sections, padding included, stays zero.
    std::vector<int32_t> expected(static_cast<size_t>(layout.totalBytes / INT32_BYTES), 0);
    const auto sectionAt
        = [&expected](int64_t byteOffset) { return expected.begin() + byteOffset / INT32_BYTES; };
    // block_tables: [0..7 | 8..15 | 16..23]
    std::iota(sectionAt(layout.blockTablesOffset),
              sectionAt(layout.blockTablesOffset) + STAGED_BATCH * BLOCKS_PER_SEQUENCE,
              0);
    // seq_lens: every sequence is seqlen_kv long.
    std::fill_n(sectionAt(layout.seqLensOffset), STAGED_BATCH, static_cast<int32_t>(STAGED_SEQ_KV));
    // query_start_len: b * seqlen_q, closed by batch * seqlen_q.
    const std::vector<int32_t> queryStartLen{0, 64, 128, 192};
    std::copy(queryStartLen.begin(), queryStartLen.end(), sectionAt(layout.queryStartLenOffset));

    EXPECT_EQ(staged, expected);
}

TEST(TestGfx950AttentionUnified2DDispatch, RefusesNonBshdOutput)
{
    registerNativeIngestorSymbols();
    const AttentionGraph ag(
        SCALE, data_objects::DiagonalAlignment::BOTTOM_RIGHT, /* withBhsdTensor= */ true);
    auto bound = bindingsFor(ag);
    ASSERT_TRUE(bound.has_value());
    (*bound)[std::string(O_TOKEN)] = BHSD_UID;
    const auto msg = prepareFailure(ag, *bound, makeKernel());
    EXPECT_NE(msg.find(OUTPUT_LAYOUT_MARKER), std::string::npos)
        << "Expected output-layout error, got: " << msg;
}

TEST(TestGfx950AttentionUnified2DDispatch, ScaleIsRoundTrippedThroughBitPattern)
{
    registerNativeIngestorSymbols();
    // Round-trip: graph_match stores scale as int64 bits, prepare() recovers them.
    // The test exercises bindingsFor() which calls graph_match; a different scale
    // produces different bit pattern in the bound token.
    constexpr float SCALE_ALT = 0.125F;
    int32_t bitsExpected = 0;
    std::memcpy(&bitsExpected, &SCALE_ALT, sizeof(SCALE_ALT));

    const AttentionGraph ag(SCALE_ALT);
    auto bound = bindingsFor(ag);
    ASSERT_TRUE(bound.has_value());
    const auto bits = tryGetBoundInt(*bound, "gfx950_attention_unified_2d.scale_bits");
    ASSERT_TRUE(bits.has_value());
    EXPECT_EQ(*bits, static_cast<int64_t>(bitsExpected));
}

TEST(TestGfx950AttentionUnified2DDispatch, RefusesToPrepareWithoutMatcherBindings)
{
    registerNativeIngestorSymbols();
    const AttentionGraph ag;
    const BoundTokens emptyBound;
    auto kernel = makeKernel();
    const auto msg = prepareFailure(ag, emptyBound, kernel);
    EXPECT_NE(msg.find(MISSING_TOKEN_MARKER), std::string::npos)
        << "Expected missing-token error, got: " << msg;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
