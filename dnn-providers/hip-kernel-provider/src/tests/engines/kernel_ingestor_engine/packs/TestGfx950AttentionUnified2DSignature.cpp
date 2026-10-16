// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
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

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

namespace
{

using namespace hipdnn_plugin_sdk::ingestor;

// Symbol names match the installed descriptor files exactly.
constexpr std::string_view GRAPH_MATCHER_SYMBOL
    = "hipkernel.gfx950_attention_unified2_d.graph_match";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.gfx950_attention_unified2_d.dispatch";

// Argument kinds.
constexpr const char* BUFFER_KIND = "global_buffer";
constexpr const char* BY_VALUE_KIND = "by_value";
constexpr uint32_t POINTER_BYTES = 8;
constexpr uint32_t SCALAR_BYTES_F32 = 4;
constexpr uint32_t SCALAR_BYTES_I32 = 4;

// Canonical 18-argument ABI (Python: candidate.signature(tuning_spec)).
// Offsets are for diagnostic display only — the loader compares kind, size, order.
std::vector<KernelArgument> pythonAbi()
{
    return {KernelArgument{BUFFER_KIND, POINTER_BYTES, 0, "output_ptr"},
            KernelArgument{BUFFER_KIND, POINTER_BYTES, 8, "query_ptr"},
            KernelArgument{BUFFER_KIND, POINTER_BYTES, 16, "key_cache_ptr"},
            KernelArgument{BUFFER_KIND, POINTER_BYTES, 24, "value_cache_ptr"},
            KernelArgument{BUFFER_KIND, POINTER_BYTES, 32, "sink_ptr"},
            KernelArgument{BUFFER_KIND, POINTER_BYTES, 40, "block_tables_ptr"},
            KernelArgument{BUFFER_KIND, POINTER_BYTES, 48, "seq_lens_ptr"},
            KernelArgument{BUFFER_KIND, POINTER_BYTES, 56, "alibi_slopes_ptr"},
            KernelArgument{BUFFER_KIND, POINTER_BYTES, 64, "qq_bias_ptr"},
            KernelArgument{BUFFER_KIND, POINTER_BYTES, 72, "query_start_len_ptr"},
            KernelArgument{BY_VALUE_KIND, SCALAR_BYTES_F32, 80, "scale"},
            KernelArgument{BY_VALUE_KIND, SCALAR_BYTES_F32, 84, "k_scale"},
            KernelArgument{BY_VALUE_KIND, SCALAR_BYTES_F32, 88, "v_scale"},
            KernelArgument{BY_VALUE_KIND, SCALAR_BYTES_F32, 92, "out_scale"},
            KernelArgument{BY_VALUE_KIND, SCALAR_BYTES_F32, 96, "softcap"},
            KernelArgument{BY_VALUE_KIND, SCALAR_BYTES_I32, 100, "num_seqs"},
            KernelArgument{BY_VALUE_KIND, SCALAR_BYTES_I32, 104, "block_table_stride"},
            KernelArgument{BY_VALUE_KIND, SCALAR_BYTES_I32, 108, "qq_bias_stride_0"}};
}

// A canonical dense BSHD causal graph that graph_match accepts.
// Built once to prove the dispatch handler can load and proceed to its signature check.
flatbuffers::FlatBufferBuilder buildAcceptedSdpaGraph()
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    flatbuffers::FlatBufferBuilder b(4096);

    constexpr int64_t Q_UID = 1;
    constexpr int64_t K_UID = 2;
    constexpr int64_t V_UID = 3;
    constexpr int64_t O_UID = 4;
    constexpr int64_t BATCH = 2;
    constexpr int64_t HEADS_Q = 4;
    constexpr int64_t HEADS_KV = 4;
    constexpr int64_t SEQ = 256;
    constexpr int64_t HEAD_SIZE = 128;

    const std::vector<int64_t> dims{BATCH, HEADS_Q, SEQ, HEAD_SIZE};
    const std::vector<int64_t> strides{
        SEQ * HEADS_Q * HEAD_SIZE, HEAD_SIZE, HEADS_Q * HEAD_SIZE, 1};
    const std::vector<int64_t> kdims{BATCH, HEADS_KV, SEQ, HEAD_SIZE};
    const std::vector<int64_t> kstrides{
        SEQ * HEADS_KV * HEAD_SIZE, HEAD_SIZE, HEADS_KV * HEAD_SIZE, 1};

    auto qT = CreateTensorAttributesDirect(
        b, Q_UID, nullptr, DataType::BFLOAT16, &strides, &dims, false);
    auto kT = CreateTensorAttributesDirect(
        b, K_UID, nullptr, DataType::BFLOAT16, &kstrides, &kdims, false);
    auto vT = CreateTensorAttributesDirect(
        b, V_UID, nullptr, DataType::BFLOAT16, &kstrides, &kdims, false);
    auto oT = CreateTensorAttributesDirect(
        b, O_UID, nullptr, DataType::BFLOAT16, &strides, &dims, false);

    SdpaAttributesBuilder sdpa(b);
    sdpa.add_q_tensor_uid(Q_UID);
    sdpa.add_k_tensor_uid(K_UID);
    sdpa.add_v_tensor_uid(V_UID);
    sdpa.add_o_tensor_uid(O_UID);
    // bottom-right causal: right_bound = 0, diagonal_alignment = BOTTOM_RIGHT
    sdpa.add_right_bound(0);
    sdpa.add_diagonal_alignment(DiagonalAlignment::BOTTOM_RIGHT);
    sdpa.add_attn_scale_value(0.08838834764831843F);
    sdpa.add_alibi_mask(false);
    sdpa.add_padding_mask(false);
    sdpa.add_mma_core_mode(DataType::UNSET);
    sdpa.add_implementation(AttentionImplementation::AUTO);
    auto sdpaAttr = sdpa.Finish();

    auto node = CreateNodeDirect(
        b, "sdpa", DataType::FLOAT, NodeAttributes::SdpaAttributes, sdpaAttr.Union());

    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors{qT, kT, vT, oT};
    const std::vector<flatbuffers::Offset<Node>> nodes{node};
    auto graph = CreateGraphDirect(b,
                                   "gfx950_attention_unified_2d_signature",
                                   DataType::UNSET,
                                   DataType::UNSET,
                                   DataType::UNSET,
                                   &tensors,
                                   &nodes);
    b.Finish(graph);
    return b;
}

// A minimal KPACK KernelDefinition that graph_match will accept.
KernelDefinition makeKernel(const std::vector<KernelArgument>& recorded)
{
    KernelDefinition kernel;
    kernel.dispatchId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-000000000051");
    kernel.packId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-000000000052");
    kernel.kernelId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-000000000053");
    kernel.name = "attention_unified_2d.bf16_d128_hq4_kv4_bs16_t16_nw1_mw16.gfx950";
    kernel.source.kind = KernelSourceKind::KPACK;
    kernel.source.library = "there-is-no-archive-here.kpack";
    kernel.source.tocKey = "toc#0";
    kernel.source.symbol = "attention_unified_2d";
    kernel.source.signature = recorded;
    kernel.originDirectory = "/nonexistent";
    kernel.treeRoot = "/nonexistent";
    kernel.metadata["dtype"] = std::string("BF16");
    kernel.metadata["head_size"] = int64_t{128};
    kernel.metadata["num_query_heads"] = int64_t{4};
    kernel.metadata["num_kv_heads"] = int64_t{4};
    kernel.metadata["block_size"] = int64_t{16};
    kernel.metadata["sliding_window"] = int64_t{0};
    kernel.metadata["use_sinks"] = int64_t{0};
    kernel.metadata["num_seqs"] = int64_t{256};
    kernel.metadata["tile_size"] = int64_t{16};
    kernel.metadata["num_warps"] = int64_t{1};
    kernel.metadata["block_m_per_warp"] = int64_t{16};
    kernel.metadata["use_mfma_32x32"] = int64_t{0};
    kernel.metadata["use_transposed_qk_32x32"] = int64_t{0};
    kernel.metadata["use_register_pv"] = int64_t{0};
    kernel.metadata["use_k_single_buffer"] = int64_t{0};
    return kernel;
}

constexpr const char* MISMATCH_MARKER = "is packaged with arguments";

std::string prepareFailure(const std::vector<KernelArgument>& recorded)
{
    registerNativeIngestorSymbols();

    auto builder = buildAcceptedSdpaGraph();
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
        return "(graph_match returned nullopt — graph was declined)";
    }

    auto handler = DispatchRegistry<Handle>::resolve(std::string(DISPATCH_SYMBOL));
    auto kernel = makeKernel(recorded);
    try
    {
        handler->prepare(context, *bound, kernel);
    }
    catch(const std::exception& e)
    {
        return e.what();
    }
    return "(no exception thrown)";
}

void expectRefused(const std::vector<KernelArgument>& recorded)
{
    const auto msg = prepareFailure(recorded);
    EXPECT_NE(msg.find(MISMATCH_MARKER), std::string::npos)
        << "Expected mismatch message containing '" << MISMATCH_MARKER << "', got: " << msg;
}

std::vector<KernelArgument> withArg(std::size_t idx, KernelArgument arg)
{
    auto abi = pythonAbi();
    abi[idx] = std::move(arg);
    return abi;
}

} // namespace

// ---------------------------------------------------------------------------
// TestGfx950AttentionUnified2DSignature
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionUnified2DSignature, AcceptsTheAbiThePythonDeclares)
{
    const auto msg = prepareFailure(pythonAbi());
    EXPECT_EQ(msg.find(MISMATCH_MARKER), std::string::npos)
        << "The canonical ABI should not trigger a mismatch: " << msg;
}

TEST(TestGfx950AttentionUnified2DSignature, RefusesADroppedTrailingArgument)
{
    auto abi = pythonAbi();
    abi.pop_back();
    expectRefused(abi);
}

TEST(TestGfx950AttentionUnified2DSignature, RefusesAnExtraTrailingArgument)
{
    auto abi = pythonAbi();
    abi.push_back(KernelArgument{"by_value", 4U, 112U, "extra"});
    expectRefused(abi);
}

TEST(TestGfx950AttentionUnified2DSignature, RefusesAScalarInAPointerSlot)
{
    // slot 0 is output_ptr (global_buffer/8); swap with a by_value f32
    expectRefused(withArg(0, KernelArgument{"by_value", SCALAR_BYTES_F32, 0U, "output_ptr"}));
}

TEST(TestGfx950AttentionUnified2DSignature, RefusesAPointerInAScalarSlot)
{
    // slot 10 is scale (by_value/f32); swap with a pointer
    expectRefused(withArg(10, KernelArgument{"global_buffer", POINTER_BYTES, 80U, "scale"}));
}

TEST(TestGfx950AttentionUnified2DSignature, RefusesANarrowedPointerSlot)
{
    // pointer narrowed to 4 bytes
    expectRefused(withArg(1, KernelArgument{"global_buffer", 4U, 8U, "query_ptr"}));
}

TEST(TestGfx950AttentionUnified2DSignature, RefusesAWidenedScalarSlot)
{
    // scale widened to 8 bytes
    expectRefused(withArg(10, KernelArgument{"by_value", 8U, 80U, "scale"}));
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
