// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"

/**
 * @file TestAttentionPagedUnifiedMatchers.cpp
 * @brief Graph and kernel matching for hipkernel:AttentionPagedUnified: the paged
 *        uniform-decode (dense Q) and ragged-Q graphs it serves, and a logged decline,
 *        with its cause, for everything else. Matcher-only: no device, no launch.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

namespace d = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::DeviceProperties;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::MatchContext;
using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;

constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.attention_paged_unified.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.attention_paged_unified.kernel_match";

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;
constexpr int64_t PAGE_TABLE_UID = 5;
constexpr int64_t SEQ_LEN_KV_UID = 6;
constexpr int64_t SEQ_LEN_Q_UID = 7;
constexpr int64_t RAGGED_OFFSET_UID = 8;
constexpr int64_t EXTRA_UID = 9;

constexpr int64_t BATCH = 4;
constexpr int64_t QUERY_HEADS = 32;
constexpr int64_t KV_HEADS = 8;
constexpr int64_t HEAD_SIZE = 128;
constexpr int64_t BLOCK_SIZE = 16;
constexpr int64_t NUM_BLOCKS = 1000;
constexpr int64_t MAX_BLOCKS = 64;

std::unique_ptr<d::TensorAttributesT>
    tensor(int64_t uid, d::DataType type, std::vector<int64_t> dims, std::vector<int64_t> strides)
{
    auto t = std::make_unique<d::TensorAttributesT>();
    t->uid = uid;
    t->data_type = type;
    t->dims = std::move(dims);
    t->strides = std::move(strides);
    return t;
}

std::vector<int64_t> packed(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(size_t i = dims.size(); i > 1; --i)
    {
        strides[i - 2] = strides[i - 1] * dims[i - 1];
    }
    return strides;
}

/**
 * The uniform-decode graph the vLLM plugin builds: Q/O `[B, Hq, 1, D]` token-major,
 * K/V the interleaved views of one `[num_blocks, Hkv, block_size, 2*D]` cache, one page
 * table `[B, 1, M, 1]` (vLLM's block_table view) for both K and V, INT32 seq_len_kv, and
 * bottom-right causal masking.
 */
d::GraphT decodeGraph(d::DataType type = d::DataType::BFLOAT16)
{
    d::GraphT g;
    const int64_t tokenStride = QUERY_HEADS * HEAD_SIZE;
    g.tensors.push_back(tensor(
        Q_UID, type, {BATCH, QUERY_HEADS, 1, HEAD_SIZE}, {tokenStride, HEAD_SIZE, tokenStride, 1}));
    const std::vector<int64_t> kvStrides
        = {KV_HEADS * BLOCK_SIZE * 2 * HEAD_SIZE, BLOCK_SIZE * 2 * HEAD_SIZE, 2 * HEAD_SIZE, 1};
    g.tensors.push_back(
        tensor(K_UID, type, {NUM_BLOCKS, KV_HEADS, BLOCK_SIZE, HEAD_SIZE}, kvStrides));
    g.tensors.push_back(
        tensor(V_UID, type, {NUM_BLOCKS, KV_HEADS, BLOCK_SIZE, HEAD_SIZE}, kvStrides));
    g.tensors.push_back(tensor(
        O_UID, type, {BATCH, QUERY_HEADS, 1, HEAD_SIZE}, {tokenStride, HEAD_SIZE, tokenStride, 1}));
    g.tensors.push_back(tensor(PAGE_TABLE_UID,
                               d::DataType::INT32,
                               {BATCH, 1, MAX_BLOCKS, 1},
                               {MAX_BLOCKS, MAX_BLOCKS, 1, 1}));
    g.tensors.push_back(tensor(SEQ_LEN_KV_UID, d::DataType::INT32, {BATCH, 1, 1, 1}, {1, 1, 1, 1}));

    d::SdpaAttributesT a;
    a.q_tensor_uid = Q_UID;
    a.k_tensor_uid = K_UID;
    a.v_tensor_uid = V_UID;
    a.o_tensor_uid = O_UID;
    a.page_table_k_tensor_uid = PAGE_TABLE_UID;
    a.page_table_v_tensor_uid = PAGE_TABLE_UID;
    a.seq_len_kv_tensor_uid = SEQ_LEN_KV_UID;
    a.max_seq_len_kv = static_cast<int32_t>(MAX_BLOCKS * BLOCK_SIZE);
    a.padding_mask = true;
    a.right_bound = 0;
    a.diagonal_alignment = d::DiagonalAlignment::BOTTOM_RIGHT;
    a.attn_scale_value = 0.08838835F;
    auto n = std::make_unique<d::NodeT>();
    n->compute_data_type = d::DataType::FLOAT;
    n->attributes.Set(a);
    g.nodes.push_back(std::move(n));
    return g;
}

/// The mixed-batch graph: Q/O ragged at the sequence axis through one token-unit
/// offset table (multiplier = token stride), plus seq_len_q.
d::GraphT raggedGraph()
{
    auto g = decodeGraph();
    const int64_t maxTokens = 2048;
    // Q is a view of a fused QKV projection: token stride (Hq + 2 Hkv) * D.
    const int64_t qTokenStride = (QUERY_HEADS + 2 * KV_HEADS) * HEAD_SIZE;
    auto& q = *g.tensors[0];
    q.dims = {BATCH, QUERY_HEADS, maxTokens, HEAD_SIZE};
    q.strides = {maxTokens * qTokenStride, HEAD_SIZE, qTokenStride, 1};
    q.ragged_offset_tensor_uid = RAGGED_OFFSET_UID;
    q.ragged_offset_multiplier = qTokenStride;
    auto& o = *g.tensors[3];
    const int64_t oTokenStride = QUERY_HEADS * HEAD_SIZE;
    o.dims = {BATCH, QUERY_HEADS, maxTokens, HEAD_SIZE};
    o.strides = {maxTokens * oTokenStride, HEAD_SIZE, oTokenStride, 1};
    o.ragged_offset_tensor_uid = RAGGED_OFFSET_UID;
    o.ragged_offset_multiplier = oTokenStride;
    g.tensors.push_back(tensor(SEQ_LEN_Q_UID, d::DataType::INT32, {BATCH, 1, 1, 1}, {1, 1, 1, 1}));
    g.tensors.push_back(
        tensor(RAGGED_OFFSET_UID, d::DataType::INT32, {BATCH + 1, 1, 1, 1}, {1, 1, 1, 1}));
    g.nodes[0]->attributes.AsSdpaAttributes()->seq_len_q_tensor_uid = SEQ_LEN_Q_UID;
    return g;
}

DeviceProperties deviceProperties()
{
    DeviceProperties properties;
    properties.gcnArchName = "gfx942";
    properties.warpSize = 64;
    return properties;
}

d::SdpaAttributesT& attrs(d::GraphT& g)
{
    return *g.nodes[0]->attributes.AsSdpaAttributes();
}

d::TensorAttributesT& find(d::GraphT& g, int64_t uid)
{
    for(auto& t : g.tensors)
    {
        if(t->uid == uid)
        {
            return *t;
        }
    }
    throw std::runtime_error("no tensor " + std::to_string(uid));
}

using Mutation = std::function<void(d::GraphT&)>;

std::optional<BoundTokens> match(d::GraphT g, const Mutation& change = {})
{
    registerNativeIngestorSymbols();
    if(change)
    {
        change(g);
    }
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(d::Graph::Pack(builder, &g));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper(
        builder.GetBufferPointer(), builder.GetSize());
    const auto properties = deviceProperties();
    const MatchContext context{wrapper, 0, properties};
    return hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
        std::string(GRAPH_MATCHER_SYMBOL))(context);
}

struct LoggedDecline
{
    std::string cause;
    std::string detail;
};

/// The decline graph_match logged at INFO, or an empty cause.
LoggedDecline loggedDecline(d::GraphT g, const Mutation& change)
{
    const std::string marker
        = std::string(ATTENTION_PAGED_UNIFIED_ENGINE_NAME) + " declined the graph [";
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    EXPECT_FALSE(match(std::move(g), change).has_value());
    for(const auto& log : recorder.getRecordedLogs())
    {
        const auto at = log.message.find(marker);
        const auto end = at == std::string::npos ? at : log.message.find(']', at + marker.size());
        if(log.severity == HIPDNN_SEV_INFO && end != std::string::npos)
        {
            return {log.message.substr(at + marker.size(), end - at - marker.size()),
                    log.message.substr(end + 1)};
        }
    }
    return {};
}

KernelDefinition kernel(const std::string& dtype,
                        int64_t headSize,
                        int64_t blockSize,
                        int64_t queriesPerKv,
                        int64_t numWarps = 1,
                        int64_t kvHeads = 1)
{
    KernelDefinition k;
    k.kernelId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-0000000000a1");
    k.metadata = {
        {std::string("dtype"), dtype},
        {std::string("head_size"), headSize},
        {std::string("block_size"), blockSize},
        {std::string("num_query_heads"), queriesPerKv},
        {std::string("num_kv_heads"), kvHeads},
        {std::string("num_warps"), numWarps},
        {std::string("block_m_per_warp"), int64_t{16}},
    };
    return k;
}

bool kernelMatches(const d::GraphT& g, const KernelDefinition& candidate)
{
    registerNativeIngestorSymbols();
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(d::Graph::Pack(builder, &g));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper(
        builder.GetBufferPointer(), builder.GetSize());
    const auto properties = deviceProperties();
    const MatchContext context{wrapper, 0, properties};
    const auto bound = hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
        std::string(GRAPH_MATCHER_SYMBOL))(context);
    EXPECT_TRUE(bound.has_value());
    if(!bound.has_value())
    {
        return false;
    }
    return hipdnn_plugin_sdk::ingestor::KernelMatcherRegistry::resolve(
        std::string(KERNEL_MATCHER_SYMBOL))(context, *bound, candidate);
}

TEST(TestAttentionPagedUnifiedMatchers, ServesPagedUniformDecode)
{
    const auto bound = match(decodeGraph());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, "attention_paged_unified.queries_per_kv"), 4);
    EXPECT_EQ(tryGetBoundInt(*bound, "attention_paged_unified.block_size"), BLOCK_SIZE);
    EXPECT_EQ(tryGetBoundInt(*bound, "attention_paged_unified.ragged_offset.uid"), -1);
    EXPECT_EQ(tryGetBoundInt(*bound, "attention_paged_unified.seq_len_q.uid"), -1);
    EXPECT_EQ(tryGetBoundInt(*bound, "attention_paged_unified.window"), 0x3FFFFFFF);
}

TEST(TestAttentionPagedUnifiedMatchers, ServesSpeculativeDecodeRowsWithSeqLenQ)
{
    EXPECT_TRUE(match(decodeGraph(), [](d::GraphT& g) {
                    for(const auto uid : {Q_UID, O_UID})
                    {
                        auto& t = find(g, uid);
                        t.dims[2] = 3;
                        t.strides
                            = {3 * QUERY_HEADS * HEAD_SIZE, HEAD_SIZE, QUERY_HEADS * HEAD_SIZE, 1};
                    }
                    g.tensors.push_back(
                        tensor(SEQ_LEN_Q_UID, d::DataType::INT32, {BATCH, 1, 1, 1}, {1, 1, 1, 1}));
                    attrs(g).seq_len_q_tensor_uid = SEQ_LEN_Q_UID;
                }).has_value());
}

TEST(TestAttentionPagedUnifiedMatchers, ServesRaggedQueryOverPagedKv)
{
    const auto bound = match(raggedGraph());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, "attention_paged_unified.ragged_offset.uid"),
              RAGGED_OFFSET_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, "attention_paged_unified.seq_len_q.uid"), SEQ_LEN_Q_UID);
    // seq_len_q is optional for a ragged Q: the offsets then give each length.
    EXPECT_TRUE(match(raggedGraph(), [](d::GraphT& g) {
                    attrs(g).seq_len_q_tensor_uid = flatbuffers::nullopt;
                }).has_value());
}

TEST(TestAttentionPagedUnifiedMatchers, LimitsOnlyARaggedBatchToTheSequenceSearch)
{
    // A dense Q indexes its sequence from the grid, so it has no 65536-sequence limit.
    EXPECT_TRUE(match(decodeGraph(), [](d::GraphT& g) {
                    for(const auto uid : {Q_UID, O_UID, PAGE_TABLE_UID, SEQ_LEN_KV_UID})
                    {
                        find(g, uid).dims[0] = 65537;
                    }
                }).has_value());
    EXPECT_TRUE(match(raggedGraph(), [](d::GraphT& g) {
                    for(const auto uid :
                        {Q_UID, O_UID, PAGE_TABLE_UID, SEQ_LEN_KV_UID, SEQ_LEN_Q_UID})
                    {
                        find(g, uid).dims[0] = 65536;
                    }
                    find(g, RAGGED_OFFSET_UID).dims[0] = 65537;
                }).has_value());
}

TEST(TestAttentionPagedUnifiedMatchers, ServesSlidingWindowsAndDeprecatedBottomRightFlag)
{
    const auto bound = match(decodeGraph(), [](d::GraphT& g) { attrs(g).left_bound = 4095; });
    ASSERT_TRUE(bound.has_value());
    // hipDNN keeps k >= d - left_bound; the kernel keeps d - k < window.
    EXPECT_EQ(tryGetBoundInt(*bound, "attention_paged_unified.window"), 4096);
    EXPECT_EQ(tryGetBoundInt(*match(decodeGraph(), [](d::GraphT& g) { attrs(g).left_bound = 0; }),
                             "attention_paged_unified.window"),
              1);
    EXPECT_TRUE(match(decodeGraph(), [](d::GraphT& g) {
                    attrs(g).right_bound = flatbuffers::nullopt;
                    attrs(g).diagonal_alignment = d::DiagonalAlignment::TOP_LEFT;
                    attrs(g).causal_mask_bottom_right = true;
                }).has_value());
}

TEST(TestAttentionPagedUnifiedMatchers, ServesOtherContainerLayoutsAndFp16)
{
    // Plain NHD caches: [num_blocks, block_size, Hkv, D] contiguous.
    EXPECT_TRUE(match(decodeGraph(), [](d::GraphT& g) {
                    for(const auto uid : {K_UID, V_UID})
                    {
                        find(g, uid).strides = {
                            BLOCK_SIZE * KV_HEADS * HEAD_SIZE, HEAD_SIZE, KV_HEADS * HEAD_SIZE, 1};
                    }
                }).has_value());
    // HND: packed [num_blocks, Hkv, block_size, D].
    EXPECT_TRUE(match(decodeGraph(), [](d::GraphT& g) {
                    for(const auto uid : {K_UID, V_UID})
                    {
                        find(g, uid).strides
                            = packed({NUM_BLOCKS, KV_HEADS, BLOCK_SIZE, HEAD_SIZE});
                    }
                }).has_value());
    EXPECT_TRUE(match(decodeGraph(d::DataType::HALF)).has_value());
}

TEST(TestAttentionPagedUnifiedMatchers, IgnoresStridesOfUnitExtentAxes)
{
    EXPECT_TRUE(match(decodeGraph(), [](d::GraphT& g) {
                    find(g, Q_UID).strides[2] = 7; // Sq == 1
                    find(g, PAGE_TABLE_UID).strides[1] = 3;
                }).has_value());
}

TEST(TestAttentionPagedUnifiedMatchers, MatchesOnlyTheCandidateBuiltForTheGraph)
{
    EXPECT_TRUE(kernelMatches(decodeGraph(), kernel("BF16", HEAD_SIZE, BLOCK_SIZE, 4)));
    EXPECT_FALSE(kernelMatches(decodeGraph(), kernel("FP16", HEAD_SIZE, BLOCK_SIZE, 4)));
    EXPECT_FALSE(kernelMatches(decodeGraph(), kernel("BF16", 64, BLOCK_SIZE, 4)));
    EXPECT_FALSE(kernelMatches(decodeGraph(), kernel("BF16", HEAD_SIZE, 32, 4)));
    EXPECT_FALSE(kernelMatches(decodeGraph(), kernel("BF16", HEAD_SIZE, BLOCK_SIZE, 8)));
    // A candidate not built for runtime head counts (num_kv_heads != 1) never matches.
    EXPECT_FALSE(kernelMatches(decodeGraph(), kernel("BF16", HEAD_SIZE, BLOCK_SIZE, 4, 1, 2)));
    // Nor one whose geometry has no query rows per block.
    EXPECT_FALSE(kernelMatches(decodeGraph(), kernel("BF16", HEAD_SIZE, BLOCK_SIZE, 4, 0)));
    KernelDefinition incomplete = kernel("BF16", HEAD_SIZE, BLOCK_SIZE, 4);
    incomplete.metadata.erase("num_warps");
    EXPECT_FALSE(kernelMatches(decodeGraph(), incomplete));
}

struct DeclineCase
{
    const char* name;
    std::function<d::GraphT()> graph;
    Mutation change;
    const char* cause;
};

TEST(TestAttentionPagedUnifiedMatchers, DeclinesWithALoggedCause)
{
    const auto decode = [] { return decodeGraph(); };
    const auto ragged = [] { return raggedGraph(); };
    const std::vector<DeclineCase> cases = {
        {"no page tables",
         decode,
         [](d::GraphT& g) {
             attrs(g).page_table_k_tensor_uid = flatbuffers::nullopt;
             attrs(g).page_table_v_tensor_uid = flatbuffers::nullopt;
         },
         "not_paged"},
        {"two page tables",
         decode,
         [](d::GraphT& g) {
             g.tensors.push_back(tensor(EXTRA_UID,
                                        d::DataType::INT32,
                                        {BATCH, 1, MAX_BLOCKS, 1},
                                        {MAX_BLOCKS, MAX_BLOCKS, 1, 1}));
             attrs(g).page_table_v_tensor_uid = EXTRA_UID;
         },
         "page_table"},
        {"int64 page table",
         decode,
         [](d::GraphT& g) { find(g, PAGE_TABLE_UID).data_type = d::DataType::INT64; },
         "page_table"},
        {"strided page-table entries",
         decode,
         [](d::GraphT& g) { find(g, PAGE_TABLE_UID).strides = {2 * MAX_BLOCKS, 1, 2, 1}; },
         "page_table"},
        {"fp32",
         decode,
         [](d::GraphT& g) {
             for(auto& t : g.tensors)
             {
                 if(t->data_type == d::DataType::BFLOAT16)
                 {
                     t->data_type = d::DataType::FLOAT;
                 }
             }
         },
         "data_type"},
        {"mixed dtypes",
         decode,
         [](d::GraphT& g) { find(g, O_UID).data_type = d::DataType::HALF; },
         "data_type"},
        {"half compute",
         decode,
         [](d::GraphT& g) { g.nodes[0]->compute_data_type = d::DataType::HALF; },
         "compute_data_type"},
        {"head size 96",
         decode,
         [](d::GraphT& g) {
             for(auto& t : g.tensors)
             {
                 if(t->uid <= O_UID)
                 {
                     t->dims[3] = 96;
                 }
             }
         },
         "head_size"},
        {"block size 8",
         decode,
         [](d::GraphT& g) {
             find(g, K_UID).dims[2] = 8;
             find(g, V_UID).dims[2] = 8;
         },
         "block_size"},
        {"gqa 3",
         decode,
         [](d::GraphT& g) {
             find(g, K_UID).dims[1] = 4;
             find(g, V_UID).dims[1] = 4;
             find(g, Q_UID).dims[1] = 12;
             find(g, O_UID).dims[1] = 12;
         },
         "gqa"},
        {"unequal K/V strides",
         decode,
         [](d::GraphT& g) {
             find(g, V_UID).strides = packed({NUM_BLOCKS, KV_HEADS, BLOCK_SIZE, HEAD_SIZE});
         },
         "kv_layout"},
        {"misaligned K token stride",
         decode,
         [](d::GraphT& g) {
             find(g, K_UID).strides[2] = 2 * HEAD_SIZE + 4;
             find(g, V_UID).strides[2] = 2 * HEAD_SIZE + 4;
         },
         "kv_layout"},
        {"misaligned Q head stride",
         decode,
         [](d::GraphT& g) { find(g, Q_UID).strides[1] = HEAD_SIZE + 4; },
         "layout"},
        {"no seq_len_kv",
         decode,
         [](d::GraphT& g) { attrs(g).seq_len_kv_tensor_uid = flatbuffers::nullopt; },
         "seq_len_kv"},
        {"seq_len_kv shape",
         decode,
         [](d::GraphT& g) { find(g, SEQ_LEN_KV_UID).dims = {1, BATCH, 1, 1}; },
         "seq_len_kv"},
        {"ragged Q only",
         ragged,
         [](d::GraphT& g) { find(g, O_UID).ragged_offset_tensor_uid = flatbuffers::nullopt; },
         "ragged"},
        {"int64 offsets",
         ragged,
         [](d::GraphT& g) { find(g, RAGGED_OFFSET_UID).data_type = d::DataType::INT64; },
         "ragged"},
        {"misaligned multiplier",
         ragged,
         [](d::GraphT& g) { find(g, Q_UID).ragged_offset_multiplier = 4100; },
         "layout"},
        {"zero multiplier",
         ragged,
         [](d::GraphT& g) {
             find(g, Q_UID).ragged_offset_multiplier = 0;
             find(g, O_UID).ragged_offset_multiplier = 0;
         },
         "layout"},
        {"ragged batch past the sequence search",
         ragged,
         [](d::GraphT& g) {
             constexpr int64_t SEQUENCES = 65537;
             for(const auto uid : {Q_UID, O_UID, PAGE_TABLE_UID, SEQ_LEN_KV_UID, SEQ_LEN_Q_UID})
             {
                 find(g, uid).dims[0] = SEQUENCES;
             }
             find(g, RAGGED_OFFSET_UID).dims[0] = SEQUENCES + 1;
         },
         "size_limit"},
        {"page table past its buffer range",
         decode,
         [](d::GraphT& g) {
             // (B - 1) * stride + M entries: within int32, past 0x7FFF0000 bytes.
             find(g, PAGE_TABLE_UID).strides[0] = 0x0B000000;
         },
         "size_limit"},
        {"top-left causal",
         decode,
         [](d::GraphT& g) { attrs(g).diagonal_alignment = d::DiagonalAlignment::TOP_LEFT; },
         "mask"},
        {"no mask",
         decode,
         [](d::GraphT& g) { attrs(g).right_bound = flatbuffers::nullopt; },
         "mask"},
        {"right bound 2", decode, [](d::GraphT& g) { attrs(g).right_bound = 2; }, "mask"},
        {"sinks",
         decode,
         [](d::GraphT& g) { attrs(g).sink_token_tensor_uid = EXTRA_UID; },
         "sinks"},
        {"alibi", decode, [](d::GraphT& g) { attrs(g).alibi_mask = true; }, "alibi_mask"},
        {"fp8 descale",
         decode,
         [](d::GraphT& g) { attrs(g).descale_k_tensor_uid = EXTRA_UID; },
         "fp8"},
        {"stats", decode, [](d::GraphT& g) { attrs(g).generate_stats = true; }, "stats"},
        {"dropout", decode, [](d::GraphT& g) { attrs(g).dropout_probability = 0.1F; }, "dropout"},
        {"bias",
         decode,
         [](d::GraphT& g) { attrs(g).attn_mask_tensor_uid = EXTRA_UID; },
         "attn_mask"},
        {"scale tensor",
         decode,
         [](d::GraphT& g) { attrs(g).scale_tensor_uid = EXTRA_UID; },
         "scale_tensor"},
        {"block mask",
         decode,
         [](d::GraphT& g) { attrs(g).block_mask_tensor_uid = EXTRA_UID; },
         "block_mask"},
        {"implementation",
         decode,
         [](d::GraphT& g) { attrs(g).implementation = d::AttentionImplementation::COMPOSITE; },
         "implementation"},
        {"override shapes",
         decode,
         [](d::GraphT& g) { g.is_override_shape_enabled = true; },
         "node"},
        {"virtual K", decode, [](d::GraphT& g) { find(g, K_UID).virtual_ = true; }, "operand"},
    };
    for(const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.name);
        const auto decline = loggedDecline(testCase.graph(), testCase.change);
        EXPECT_EQ(decline.cause, testCase.cause) << decline.detail;
        EXPECT_FALSE(decline.detail.empty());
    }
}

TEST(TestAttentionPagedUnifiedMatchers, LogsNoDeclineForAGraphItServes)
{
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    EXPECT_TRUE(match(decodeGraph()).has_value());
    EXPECT_TRUE(match(raggedGraph()).has_value());
    EXPECT_FALSE(recorder.hasLogContaining(std::string(ATTENTION_PAGED_UNIFIED_ENGINE_NAME)
                                           + " declined the graph"));
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
