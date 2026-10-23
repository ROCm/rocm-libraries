// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <exception>
#include <filesystem>
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
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"

/**
 * @file TestGfx1151WmmaAttentionPaged.cpp
 * @brief hipkernel:Gfx1151WmmaAttention on the cuDNN paged-K/V SDPA graph: which graphs
 *        the native matcher binds (dense or ragged Q, page tables, sequence lengths,
 *        arbitrary container strides incl. vLLM's interleaved K|V cache, attention sinks),
 *        what it binds, which catalog object serves each, and the kernel ABI prepare()
 *        marshals. Paged graphs are the prefill engine's alone:
 *        hipkernel:Gfx1151WmmaAttentionDecode, whose objects are dense, declines every
 *        one of them.
 *
 * No device: the GPU half (numerics against a gather-then-dense reference, HIP-graph
 * capture and replay) is IntegrationGpuGfx1151WmmaAttentionPaged.cpp.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::DeviceProperties;
using hipdnn_plugin_sdk::ingestor::KernelArgument;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::KernelSourceKind;
using hipdnn_plugin_sdk::ingestor::MatchContext;
using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;

constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.gfx1151_wmma_attention.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.gfx1151_wmma_attention.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx1151_wmma_attention.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.gfx1151_wmma_attention.dispatch";
constexpr std::string_view DECODE_GRAPH_MATCHER_SYMBOL
    = "hipkernel.gfx1151_wmma_attention_decode.graph_match";
constexpr std::string_view DECODE_KERNEL_MATCHER_SYMBOL
    = "hipkernel.gfx1151_wmma_attention_decode.kernel_match";

// Restated rather than shared with the pack: these strings are the contract between the
// matcher that writes a token and the dispatch that reads it.
constexpr std::string_view LAYOUT_TOKEN = "gfx1151_wmma_attention.layout";
constexpr std::string_view PAGE_SIZE_TOKEN = "gfx1151_wmma_attention.page_size";
constexpr std::string_view PAGE_TABLE_UID_TOKEN = "gfx1151_wmma_attention.page_table.uid";
constexpr std::string_view PAGE_TABLE_STRIDE_TOKEN = "gfx1151_wmma_attention.page_table.stride";
constexpr std::string_view SEQ_LEN_KV_UID_TOKEN = "gfx1151_wmma_attention.seq_len_kv.uid";
constexpr std::string_view SEQ_LEN_Q_UID_TOKEN = "gfx1151_wmma_attention.seq_len_q.uid";
constexpr std::string_view Q_OFFSET_UID_TOKEN = "gfx1151_wmma_attention.q_offset.uid";
constexpr std::string_view MAX_SEQ_LEN_KV_TOKEN = "gfx1151_wmma_attention.max_seq_len_kv";
constexpr std::string_view SINK_UID_TOKEN = "gfx1151_wmma_attention.sink.uid";

// GraphLayout in the pack.
constexpr int64_t LAYOUT_PAGED_DENSE_Q = 1;
constexpr int64_t LAYOUT_PAGED_DENSE_Q_SEQUSED = 2;
constexpr int64_t LAYOUT_PAGED_RAGGED_Q = 3;

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;
constexpr int64_t LSE_UID = 5;
constexpr int64_t PAGE_TABLE_UID = 6;
constexpr int64_t PAGE_TABLE_V_UID = 7;
constexpr int64_t SEQ_LEN_KV_UID = 8;
constexpr int64_t SEQ_LEN_Q_UID = 9;
constexpr int64_t Q_OFFSET_UID = 10;
constexpr int64_t O_OFFSET_UID = 11;
constexpr int64_t SINK_UID = 12;
constexpr int64_t BIAS_UID = 13;

/// One paged SDPA graph. The defaults are vLLM's uniform decode over the TRITON_ATTN
/// cache: dense Q [B, Hq, 1, D] (BSHD), K and V the two halves of one interleaved cache
/// [num_blocks, Hkv, page, 2D] viewed as [num_blocks, Hkv, page, D], one block table.
struct PagedSpec
{
    int64_t batch = 4;
    int64_t queryHeads = 8;
    int64_t kvHeads = 2;
    int64_t queryLength = 1;
    int64_t headSize = 128;
    int64_t pageSize = 16;
    int64_t numBlocks = 64;
    int64_t maxBlocksPerSeq = 8;
    std::optional<int32_t> maxSeqLenKv = 128;
    data_objects::DataType dtype = data_objects::DataType::BFLOAT16;
    bool interleavedCache = true;
    std::optional<int64_t> cacheRow; // the cache's token stride (2D or D) when absent
    bool raggedQ = false;
    bool seqLenQ = false;
    bool seqLenKv = true;
    bool returnLse = false;
    // Attention sinks: sink_token [1, sinkHeads, 1, 1] (Hq when absent) in sinkDtype (Q's
    // when absent) with head stride sinkHeadStride.
    bool sinks = false;
    std::optional<data_objects::DataType> sinkDtype;
    std::optional<int64_t> sinkHeads;
    int64_t sinkHeadStride = 1;
    // Defects, each of which the engine must decline.
    bool separateVTable = false;
    int64_t tableEntryStride = 1;
    data_objects::DataType tableDtype = data_objects::DataType::INT32;
    std::optional<int64_t> qMultiplier; // the token stride when absent
    bool oOwnOffsets = false;
    bool alibi = false;
    bool bias = false;
    int64_t leftBound = -1;
    std::optional<int64_t> vNumBlocks; // numBlocks when absent
};

std::vector<int64_t> packedStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(size_t i = dims.size(); i > 1; --i)
    {
        strides[i - 2] = strides[i - 1] * dims[i - 1];
    }
    return strides;
}

flatbuffers::FlatBufferBuilder buildPagedGraph(const PagedSpec& spec)
{
    flatbuffers::FlatBufferBuilder builder;
    const int64_t b = spec.batch;
    const int64_t d = spec.headSize;
    const int64_t token = spec.queryHeads * d;
    const std::vector<int64_t> qDims{b, spec.queryHeads, spec.queryLength, d};
    // BSHD; a ragged Q addresses sequences through its offsets (batch stride unused).
    const std::vector<int64_t> qStrides{spec.queryLength * token, d, token, 1};
    const std::vector<int64_t> cacheDims{spec.numBlocks, spec.kvHeads, spec.pageSize, d};
    const std::vector<int64_t> vCacheDims{
        spec.vNumBlocks.value_or(spec.numBlocks), spec.kvHeads, spec.pageSize, d};
    const int64_t row = spec.cacheRow.value_or(spec.interleavedCache ? 2 * d : d);
    const std::vector<int64_t> cacheStrides{
        spec.kvHeads * spec.pageSize * row, spec.pageSize * row, row, 1};
    const std::vector<int64_t> tableDims{b, 1, spec.maxBlocksPerSeq, 1};
    const std::vector<int64_t> tableStrides{
        spec.maxBlocksPerSeq * spec.tableEntryStride, 1, spec.tableEntryStride, 1};
    const std::vector<int64_t> vectorDims{b, 1, 1, 1};
    const std::vector<int64_t> vectorStrides{1, 1, 1, 1};
    const std::vector<int64_t> offsetDims{b + 1, 1, 1, 1};
    const std::vector<int64_t> lseDims{b, spec.queryHeads, spec.queryLength, 1};
    const auto lseStrides = packedStrides(lseDims);
    const std::vector<int64_t> sinkDims{1, spec.sinkHeads.value_or(spec.queryHeads), 1, 1};
    const std::vector<int64_t> sinkStrides{
        sinkDims[1] * spec.sinkHeadStride, spec.sinkHeadStride, 1, 1};
    const std::vector<int64_t> biasDims{b, spec.queryHeads, spec.queryLength, 128};
    const auto biasStrides = packedStrides(biasDims);

    const auto tensor = [&](int64_t uid,
                            data_objects::DataType dtype,
                            const std::vector<int64_t>& strides,
                            const std::vector<int64_t>& dims,
                            ::flatbuffers::Optional<int64_t> ragged = ::flatbuffers::nullopt,
                            int64_t multiplier = 1) {
        return data_objects::CreateTensorAttributesDirect(builder,
                                                          uid,
                                                          nullptr,
                                                          dtype,
                                                          &strides,
                                                          &dims,
                                                          false,
                                                          data_objects::TensorValue::NONE,
                                                          0,
                                                          false,
                                                          ragged,
                                                          16,
                                                          multiplier);
    };
    const auto qRagged
        = spec.raggedQ ? ::flatbuffers::Optional<int64_t>(Q_OFFSET_UID) : ::flatbuffers::nullopt;
    const auto oRagged
        = spec.raggedQ
              ? ::flatbuffers::Optional<int64_t>(spec.oOwnOffsets ? O_OFFSET_UID : Q_OFFSET_UID)
              : ::flatbuffers::nullopt;
    const int64_t multiplier = spec.raggedQ ? spec.qMultiplier.value_or(token) : 1;

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors{
        tensor(Q_UID, spec.dtype, qStrides, qDims, qRagged, multiplier),
        tensor(K_UID, spec.dtype, cacheStrides, cacheDims),
        tensor(V_UID, spec.dtype, cacheStrides, vCacheDims),
        tensor(O_UID, spec.dtype, qStrides, qDims, oRagged, spec.raggedQ ? token : 1),
        tensor(PAGE_TABLE_UID, spec.tableDtype, tableStrides, tableDims),
        tensor(SEQ_LEN_KV_UID, data_objects::DataType::INT32, vectorStrides, vectorDims),
        tensor(SEQ_LEN_Q_UID, data_objects::DataType::INT32, vectorStrides, vectorDims),
        tensor(Q_OFFSET_UID, data_objects::DataType::INT32, vectorStrides, offsetDims),
        tensor(O_OFFSET_UID, data_objects::DataType::INT32, vectorStrides, offsetDims),
        tensor(PAGE_TABLE_V_UID, spec.tableDtype, tableStrides, tableDims),
        tensor(SINK_UID, spec.sinkDtype.value_or(spec.dtype), sinkStrides, sinkDims),
        tensor(BIAS_UID, data_objects::DataType::FLOAT, biasStrides, biasDims),
    };
    if(spec.returnLse)
    {
        tensors.push_back(tensor(LSE_UID, data_objects::DataType::FLOAT, lseStrides, lseDims));
    }

    data_objects::SdpaAttributesBuilder attributes(builder);
    attributes.add_q_tensor_uid(Q_UID);
    attributes.add_k_tensor_uid(K_UID);
    attributes.add_v_tensor_uid(V_UID);
    attributes.add_o_tensor_uid(O_UID);
    if(spec.returnLse)
    {
        attributes.add_stats_tensor_uid(LSE_UID);
        attributes.add_generate_stats(true);
    }
    attributes.add_page_table_k_tensor_uid(PAGE_TABLE_UID);
    attributes.add_page_table_v_tensor_uid(spec.separateVTable ? PAGE_TABLE_V_UID : PAGE_TABLE_UID);
    if(spec.seqLenKv)
    {
        attributes.add_seq_len_kv_tensor_uid(SEQ_LEN_KV_UID);
    }
    if(spec.seqLenQ)
    {
        attributes.add_seq_len_q_tensor_uid(SEQ_LEN_Q_UID);
    }
    if(spec.maxSeqLenKv.has_value())
    {
        attributes.add_max_seq_len_kv(*spec.maxSeqLenKv);
    }
    if(spec.sinks)
    {
        attributes.add_sink_token_tensor_uid(SINK_UID);
    }
    if(spec.bias)
    {
        attributes.add_attn_mask_tensor_uid(BIAS_UID);
    }
    attributes.add_alibi_mask(spec.alibi);
    attributes.add_padding_mask(true);
    attributes.add_left_bound(spec.leftBound);
    attributes.add_right_bound(0);
    attributes.add_diagonal_alignment(data_objects::DiagonalAlignment::BOTTOM_RIGHT);
    attributes.add_attn_scale_value(0.125F);
    attributes.add_implementation(data_objects::AttentionImplementation::AUTO);
    attributes.add_mma_core_mode(data_objects::DataType::FLOAT);

    const std::vector<flatbuffers::Offset<data_objects::Node>> nodes{
        data_objects::CreateNodeDirect(builder,
                                       "sdpa",
                                       data_objects::DataType::FLOAT,
                                       data_objects::NodeAttributes::SdpaAttributes,
                                       attributes.Finish().Union())};
    const auto name = builder.CreateString("gfx1151_wmma_attention_paged_test");
    const auto tensorVector = builder.CreateVector(tensors);
    const auto nodeVector = builder.CreateVector(nodes);
    data_objects::GraphBuilder graph(builder);
    graph.add_name(name);
    graph.add_tensors(tensorVector);
    graph.add_nodes(nodeVector);
    builder.Finish(graph.Finish());
    return builder;
}

class Fixture
{
public:
    explicit Fixture(const PagedSpec& spec)
        : _builder(buildPagedGraph(spec))
        , _graph(_builder.GetBufferPointer(), _builder.GetSize())
    {
        _properties.gcnArchName = "gfx1100";
        _properties.warpSize = 32;
        _properties.multiProcessorCount = 48;
        registerNativeIngestorSymbols();
    }

    MatchContext context() const
    {
        return MatchContext{_graph, 0, _properties};
    }

    std::optional<BoundTokens> bound() const
    {
        return hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
            std::string(GRAPH_MATCHER_SYMBOL))(context());
    }

    /// What the decode engine's graph matcher binds.
    std::optional<BoundTokens> decodeBound() const
    {
        return hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
            std::string(DECODE_GRAPH_MATCHER_SYMBOL))(context());
    }

private:
    flatbuffers::FlatBufferBuilder _builder;
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper _graph;
    DeviceProperties _properties;
};

bool matches(const PagedSpec& spec)
{
    return Fixture(spec).bound().has_value();
}

int64_t boundInt(const BoundTokens& bound, std::string_view token)
{
    const auto value = tryGetBoundInt(bound, token);
    EXPECT_TRUE(value.has_value()) << "no bound token " << token;
    return value.value_or(-1);
}

/// One catalog object's metadata (configs/gfx1151_wmma_attention.yaml).
struct ObjectSpec
{
    std::string kind = "prefill";
    std::string dtype = "BF16";
    int64_t headSize = 128;
    int64_t blockM = 128;
    std::string layout = "paged";
    int64_t pageBlockSize = 16;
    int64_t denseQ = 1;
    int64_t seqUsedQ = 0;
    int64_t sinks = 0;
};

ObjectSpec pagedObject(int64_t headSize, int64_t page, int64_t denseQ, int64_t seqUsedQ)
{
    ObjectSpec spec;
    spec.headSize = headSize;
    spec.blockM = headSize > 128 ? 64 : 128;
    spec.pageBlockSize = page;
    spec.denseQ = denseQ;
    spec.seqUsedQ = seqUsedQ;
    return spec;
}

ObjectSpec denseObject(int64_t headSize)
{
    ObjectSpec spec = pagedObject(headSize, 0, 0, 0);
    spec.layout = "dense";
    return spec;
}

KernelDefinition makeKernel(const ObjectSpec& spec)
{
    KernelDefinition kernel;
    kernel.kernelId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-000000000011");
    kernel.packId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-000000000012");
    kernel.dispatchId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-000000000013");
    const bool decode = spec.kind == "decode";
    kernel.metadata = {{"dtype", spec.dtype},
                       {"kind", spec.kind},
                       {"head_size", spec.headSize},
                       {"head_size_v", spec.headSize},
                       {"mask_mode", std::string(decode ? "runtime" : "causal")},
                       {"bias", std::string("none")},
                       {"block_size", int64_t{decode ? 128 : 256}},
                       {"block_m", spec.blockM},
                       {"dv_split", int64_t{1}},
                       {"layout", spec.layout},
                       {"page_block_size", spec.pageBlockSize},
                       {"dense_q", spec.denseQ},
                       {"seqused_q", spec.seqUsedQ},
                       {"sinks", spec.sinks}};
    return kernel;
}

bool serves(const PagedSpec& graph, const ObjectSpec& object)
{
    const Fixture fixture(graph);
    const auto bound = fixture.bound();
    return bound.has_value()
           && hipdnn_plugin_sdk::ingestor::KernelMatcherRegistry::resolve(
               std::string(KERNEL_MATCHER_SYMBOL))(fixture.context(), *bound, makeKernel(object));
}

/// Whether the decode engine's kernel matcher admits @p object for @p graph, given the
/// tokens the prefill engine's graph matcher bound (the decode engine binds none).
bool decodeEngineServes(const PagedSpec& graph, const ObjectSpec& object)
{
    const Fixture fixture(graph);
    const auto bound = fixture.bound();
    return bound.has_value()
           && hipdnn_plugin_sdk::ingestor::KernelMatcherRegistry::resolve(std::string(
               DECODE_KERNEL_MATCHER_SYMBOL))(fixture.context(), *bound, makeKernel(object));
}

/// The shipped decode object of @p headSize: four waves, runtime masks, dense.
ObjectSpec decodeObject(int64_t headSize)
{
    ObjectSpec spec = denseObject(headSize);
    spec.kind = "decode";
    spec.blockM = 16;
    return spec;
}

PagedSpec raggedGraph()
{
    PagedSpec spec;
    spec.raggedQ = true;
    spec.seqLenQ = true;
    spec.queryLength = 300; // the longest sequence's query rows
    return spec;
}

// -----------------------------------------------------------------------------
// Graph match
// -----------------------------------------------------------------------------

TEST(TestGfx1151WmmaAttentionPaged, BindsUniformDecodeOverTheInterleavedCache)
{
    const Fixture fixture(PagedSpec{});
    const auto bound = fixture.bound();
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(boundInt(*bound, LAYOUT_TOKEN), LAYOUT_PAGED_DENSE_Q);
    EXPECT_EQ(boundInt(*bound, PAGE_SIZE_TOKEN), 16);
    EXPECT_EQ(boundInt(*bound, PAGE_TABLE_UID_TOKEN), PAGE_TABLE_UID);
    EXPECT_EQ(boundInt(*bound, PAGE_TABLE_STRIDE_TOKEN), 8);
    EXPECT_EQ(boundInt(*bound, SEQ_LEN_KV_UID_TOKEN), SEQ_LEN_KV_UID);
    EXPECT_EQ(boundInt(*bound, MAX_SEQ_LEN_KV_TOKEN), 128);
    EXPECT_FALSE(tryGetBoundInt(*bound, SEQ_LEN_Q_UID_TOKEN).has_value());
    EXPECT_FALSE(tryGetBoundInt(*bound, Q_OFFSET_UID_TOKEN).has_value());
}

TEST(TestGfx1151WmmaAttentionPaged, BindsSeqLenQOfADenseQuery)
{
    PagedSpec spec;
    spec.queryLength = 3; // speculative decode, k = 2
    spec.seqLenQ = true;
    spec.returnLse = true;
    const Fixture fixture(spec);
    const auto bound = fixture.bound();
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(boundInt(*bound, LAYOUT_TOKEN), LAYOUT_PAGED_DENSE_Q_SEQUSED);
    EXPECT_EQ(boundInt(*bound, SEQ_LEN_Q_UID_TOKEN), SEQ_LEN_Q_UID);
}

TEST(TestGfx1151WmmaAttentionPaged, BindsARaggedQueryAndItsOffsets)
{
    const Fixture fixture(raggedGraph());
    const auto bound = fixture.bound();
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(boundInt(*bound, LAYOUT_TOKEN), LAYOUT_PAGED_RAGGED_Q);
    EXPECT_EQ(boundInt(*bound, SEQ_LEN_Q_UID_TOKEN), SEQ_LEN_Q_UID);
    EXPECT_EQ(boundInt(*bound, Q_OFFSET_UID_TOKEN), Q_OFFSET_UID);
}

TEST(TestGfx1151WmmaAttentionPaged, AcceptsSeparateCachePlanesSlidingWindowsAndGqa)
{
    PagedSpec spec;
    spec.interleavedCache = false;
    spec.leftBound = 63;
    spec.queryHeads = 32;
    spec.kvHeads = 1;
    spec.headSize = 64;
    spec.dtype = data_objects::DataType::HALF;
    EXPECT_TRUE(matches(spec));
}

TEST(TestGfx1151WmmaAttentionPaged, AcceptsACachePastFourGiB)
{
    // The kernel addresses a head and a page in 64 bits: only one page of one head
    // must be I32-addressable, not the cache.
    PagedSpec spec;
    spec.headSize = 256;
    spec.kvHeads = 4;
    spec.numBlocks = 100000; // [100000, 4, 16, 512] BF16 elements: past 4 GiB
    EXPECT_TRUE(matches(spec));
    spec.dtype = data_objects::DataType::HALF;
    spec.interleavedCache = false;
    spec.numBlocks = 300000;
    EXPECT_TRUE(matches(spec));
}

TEST(TestGfx1151WmmaAttentionPaged, DeclinesAPageOfAHeadPastI32Bytes)
{
    PagedSpec spec;
    spec.kvHeads = 1;
    // One head's page spans (page - 1) * row + D elements.
    spec.cacheRow = 70'000'000;
    ASSERT_TRUE(matches(spec));
    spec.cacheRow = 72'000'000;
    EXPECT_FALSE(matches(spec));
}

TEST(TestGfx1151WmmaAttentionPaged, DeclinesAPageCountPastI32)
{
    PagedSpec spec;
    spec.numBlocks = int64_t{1} << 31; // the kernel's num_blocks is I32
    EXPECT_FALSE(matches(spec));
    spec.numBlocks = (int64_t{1} << 31) - 1;
    EXPECT_TRUE(matches(spec));
}

TEST(TestGfx1151WmmaAttentionPaged, DeclinesMalformedPagedGraphs)
{
    struct Case
    {
        const char* what;
        PagedSpec spec;
    };
    std::vector<Case> cases;
    const auto add = [&cases](const char* what, auto&& edit, PagedSpec spec = PagedSpec{}) {
        edit(spec);
        cases.push_back({what, spec});
    };
    add("no seq_len_kv", [](PagedSpec& s) { s.seqLenKv = false; });
    add("no max_seq_len_kv", [](PagedSpec& s) { s.maxSeqLenKv.reset(); });
    add("max_seq_len_kv past the table", [](PagedSpec& s) { s.maxSeqLenKv = 129; });
    add("a page size that is not a power of two", [](PagedSpec& s) { s.pageSize = 48; });
    add("separate K and V tables", [](PagedSpec& s) { s.separateVTable = true; });
    add("a strided page-table row", [](PagedSpec& s) { s.tableEntryStride = 2; });
    add("an INT64 page table", [](PagedSpec& s) { s.tableDtype = data_objects::DataType::INT64; });
    add("an additive bias", [](PagedSpec& s) { s.bias = true; });
    add("ALiBi", [](PagedSpec& s) { s.alibi = true; });
    add("a head dim off the 8-element grid", [](PagedSpec& s) { s.headSize = 76; });
    add("a ragged Q without seq_len_q", [](PagedSpec& s) { s.seqLenQ = false; }, raggedGraph());
    add(
        "a ragged Q in element-unit offsets",
        [](PagedSpec& s) { s.qMultiplier = 1; },
        raggedGraph());
    add("O with its own offsets", [](PagedSpec& s) { s.oOwnOffsets = true; }, raggedGraph());
    add("stats of a ragged Q", [](PagedSpec& s) { s.returnLse = true; }, raggedGraph());
    // One page count bounds both caches' page ids.
    add("V with more pages than K", [](PagedSpec& s) { s.vNumBlocks = s.numBlocks * 2; });
    add("V with fewer pages than K", [](PagedSpec& s) { s.vNumBlocks = s.numBlocks / 2; });
    ASSERT_TRUE(matches(PagedSpec{}));
    ASSERT_TRUE(matches(raggedGraph()));
    for(const auto& c : cases)
    {
        EXPECT_FALSE(matches(c.spec)) << c.what;
    }
}

// -----------------------------------------------------------------------------
// Kernel match and score
// -----------------------------------------------------------------------------

TEST(TestGfx1151WmmaAttentionPaged, OnlyTheObjectOfThePageSizeAndQueryFormServes)
{
    const PagedSpec dense;
    PagedSpec denseUsed;
    denseUsed.seqLenQ = true;
    const PagedSpec ragged = raggedGraph();

    EXPECT_TRUE(serves(dense, pagedObject(128, 16, 1, 0)));
    EXPECT_TRUE(serves(denseUsed, pagedObject(128, 16, 1, 1)));
    EXPECT_TRUE(serves(ragged, pagedObject(128, 16, 0, 1)));
    for(const auto& graph : {dense, denseUsed, ragged})
    {
        EXPECT_FALSE(serves(graph, pagedObject(128, 32, 1, 0)));
        EXPECT_FALSE(serves(graph, pagedObject(128, 32, 0, 1)));
        // Dense and decode objects never serve a paged graph.
        EXPECT_FALSE(serves(graph, denseObject(128)));
        EXPECT_FALSE(serves(graph, decodeObject(128)));
    }
    EXPECT_FALSE(serves(dense, pagedObject(128, 16, 1, 1)));
    EXPECT_FALSE(serves(dense, pagedObject(128, 16, 0, 1)));
    EXPECT_FALSE(serves(ragged, pagedObject(128, 16, 1, 0)));
    // A narrower bucket cannot hold the head; a wider one can.
    EXPECT_FALSE(serves(dense, pagedObject(96, 16, 1, 0)));
    EXPECT_TRUE(serves(dense, pagedObject(256, 16, 1, 0)));
}

// -----------------------------------------------------------------------------
// Attention sinks: sink_token [1, Hq, 1, 1] in the Q data type selects the paged sinks
// objects, which take sink_ptr before the query form's operands.
// -----------------------------------------------------------------------------

PagedSpec withSinks(PagedSpec spec)
{
    spec.sinks = true;
    return spec;
}

ObjectSpec sinksObject(ObjectSpec spec)
{
    spec.sinks = 1;
    return spec;
}

PagedSpec denseUsedGraph()
{
    PagedSpec spec;
    spec.seqLenQ = true;
    return spec;
}

TEST(TestGfx1151WmmaAttentionPaged, BindsAttentionSinksInEveryQueryForm)
{
    PagedSpec half;
    half.dtype = data_objects::DataType::HALF;
    half.interleavedCache = false;
    for(const auto& graph : {PagedSpec{}, denseUsedGraph(), raggedGraph(), half})
    {
        const auto plain = Fixture(graph).bound();
        ASSERT_TRUE(plain.has_value());
        EXPECT_FALSE(tryGetBoundInt(*plain, SINK_UID_TOKEN).has_value());
        const auto bound = Fixture(withSinks(graph)).bound();
        ASSERT_TRUE(bound.has_value());
        EXPECT_EQ(boundInt(*bound, SINK_UID_TOKEN), SINK_UID);
        EXPECT_EQ(boundInt(*bound, LAYOUT_TOKEN), boundInt(*plain, LAYOUT_TOKEN));
    }
}

TEST(TestGfx1151WmmaAttentionPaged, DeclinesSinksTheKernelsCannotRead)
{
    struct Case
    {
        const char* what;
        PagedSpec spec;
    };
    std::vector<Case> cases;
    const auto add = [&cases](const char* what, auto&& edit) {
        PagedSpec spec = withSinks(PagedSpec{});
        edit(spec);
        cases.push_back({what, spec});
    };
    // The kernels load the sink in the I/O data type.
    add("a FLOAT sink", [](PagedSpec& s) { s.sinkDtype = data_objects::DataType::FLOAT; });
    add("a HALF sink for a BFLOAT16 query",
        [](PagedSpec& s) { s.sinkDtype = data_objects::DataType::HALF; });
    add("one sink per K/V head", [](PagedSpec& s) { s.sinkHeads = s.kvHeads; });
    add("a strided head axis", [](PagedSpec& s) { s.sinkHeadStride = 2; });
    ASSERT_TRUE(matches(withSinks(PagedSpec{})));
    for(const auto& c : cases)
    {
        EXPECT_FALSE(matches(c.spec)) << c.what;
    }
}

TEST(TestGfx1151WmmaAttentionPaged, OnlyTheSinksObjectServesASinksGraph)
{
    struct Case
    {
        PagedSpec graph;
        int64_t denseQ;
        int64_t seqUsedQ;
    };
    for(const auto& c :
        {Case{PagedSpec{}, 1, 0}, Case{denseUsedGraph(), 1, 1}, Case{raggedGraph(), 0, 1}})
    {
        const auto plain = pagedObject(128, 16, c.denseQ, c.seqUsedQ);
        EXPECT_TRUE(serves(withSinks(c.graph), sinksObject(plain)));
        EXPECT_FALSE(serves(withSinks(c.graph), plain));
        EXPECT_FALSE(serves(c.graph, sinksObject(plain)));
        EXPECT_TRUE(serves(c.graph, plain));
        // The page size and query form still decide among the sinks objects.
        EXPECT_FALSE(
            serves(withSinks(c.graph), sinksObject(pagedObject(128, 32, c.denseQ, c.seqUsedQ))));
        EXPECT_FALSE(serves(withSinks(c.graph),
                            sinksObject(pagedObject(128, 16, 1 - c.denseQ, c.seqUsedQ))));
    }
    // sinks is 0 or 1.
    auto malformed = pagedObject(128, 16, 1, 0);
    malformed.sinks = 2;
    EXPECT_FALSE(serves(withSinks(PagedSpec{}), malformed));
    EXPECT_FALSE(serves(PagedSpec{}, malformed));
}

// A paged graph is the prefill engine's whatever its query length, even a decode-shaped
// one (one query row, a small batch, long sequences): the dense decode rule, which reads
// a dense key length, never declines it, and the decode objects are dense.
TEST(TestGfx1151WmmaAttentionPaged, ThePrefillEngineServesAPagedDecodeShapedGraph)
{
    PagedSpec decodeShaped;
    decodeShaped.batch = 1;
    decodeShaped.queryLength = 1;
    decodeShaped.maxBlocksPerSeq = 512;
    decodeShaped.numBlocks = 512;
    decodeShaped.maxSeqLenKv = 8192;
    PagedSpec decodeShapedUsed = decodeShaped;
    decodeShapedUsed.seqLenQ = true;

    EXPECT_TRUE(serves(decodeShaped, pagedObject(128, 16, 1, 0)));
    EXPECT_TRUE(serves(decodeShapedUsed, pagedObject(128, 16, 1, 1)));
    EXPECT_TRUE(serves(PagedSpec{}, pagedObject(128, 16, 1, 0)));
}

TEST(TestGfx1151WmmaAttentionPaged, TheDecodeEngineDeclinesEveryPagedGraph)
{
    PagedSpec decodeShaped;
    decodeShaped.batch = 1;
    decodeShaped.maxBlocksPerSeq = 512;
    decodeShaped.numBlocks = 512;
    decodeShaped.maxSeqLenKv = 8192;
    // Sinks objects are paged prefill objects, so graphs with sinks are declined too.
    for(const auto& graph : {PagedSpec{},
                             decodeShaped,
                             denseUsedGraph(),
                             raggedGraph(),
                             withSinks(PagedSpec{}),
                             withSinks(denseUsedGraph()),
                             withSinks(raggedGraph())})
    {
        const Fixture fixture(graph);
        ASSERT_TRUE(fixture.bound().has_value());
        // Its graph matcher binds no paged graph ...
        EXPECT_FALSE(fixture.decodeBound().has_value());
        // ... and its kernel matcher admits no object for one, its own or a paged one.
        EXPECT_FALSE(decodeEngineServes(graph, decodeObject(128)));
        EXPECT_FALSE(decodeEngineServes(graph, pagedObject(128, 16, 1, 0)));
        EXPECT_FALSE(decodeEngineServes(graph, pagedObject(128, 16, 0, 1)));
        EXPECT_FALSE(decodeEngineServes(graph, sinksObject(pagedObject(128, 16, 1, 0))));
    }
}

TEST(TestGfx1151WmmaAttentionPaged, ScorePrefersTheTightestBucket)
{
    const Fixture fixture(PagedSpec{});
    const auto bound = fixture.bound();
    ASSERT_TRUE(bound.has_value());
    const auto score = [&](const ObjectSpec& object) {
        return hipdnn_plugin_sdk::ingestor::ScoreRegistry::resolve(std::string(SCORE_SYMBOL))(
            fixture.context(), *bound, makeKernel(object));
    };
    EXPECT_GT(score(pagedObject(128, 16, 1, 0)), score(pagedObject(256, 16, 1, 0)));
}

// -----------------------------------------------------------------------------
// Dispatch: workspace and the paged prefill ABI
// -----------------------------------------------------------------------------

const hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>& dispatchHandler()
{
    registerNativeIngestorSymbols();
    const auto* handler = hipdnn_plugin_sdk::ingestor::DispatchRegistry<Handle>::resolve(
        std::string(DISPATCH_SYMBOL));
    if(handler == nullptr)
    {
        throw std::logic_error("hipkernel.gfx1151_wmma_attention.dispatch is not registered");
    }
    return *handler;
}

TEST(TestGfx1151WmmaAttentionPaged, NeedsNoWorkspace)
{
    for(const auto& [graph, object] : {std::pair{PagedSpec{}, pagedObject(128, 16, 1, 0)},
                                       std::pair{raggedGraph(), pagedObject(128, 16, 0, 1)}})
    {
        const Fixture fixture(graph);
        const auto bound = fixture.bound();
        ASSERT_TRUE(bound.has_value());
        EXPECT_EQ(dispatchHandler().workspaceBytes(fixture.context(), *bound, makeKernel(object)),
                  0U);
    }
}

KernelArgument pointerArgument(const std::string& name)
{
    return KernelArgument{"global_buffer", 8, 0, name};
}

KernelArgument scalarArgument(const std::string& name)
{
    return KernelArgument{"by_value", 4, 0, name};
}

/// wmma_fmha_fwd_v2._declare_params of a shipped paged object.
std::vector<KernelArgument> rockePagedAbi(int64_t denseQ, int64_t seqUsedQ, bool sinks = false)
{
    std::vector<KernelArgument> abi;
    for(const char* name : {"Q", "K", "V", "O", "LSE"})
    {
        abi.push_back(pointerArgument(name));
    }
    abi.push_back(scalarArgument("scale_log2"));
    for(const char* name : {"batch",
                            "seqlen_q",
                            "seqlen_k",
                            "num_query_heads",
                            "num_kv_heads",
                            "num_v_heads",
                            "head_dim_q",
                            "head_dim_v",
                            "bottom_right",
                            "window_left",
                            "window_right",
                            "write_lse"})
    {
        abi.push_back(scalarArgument(name));
    }
    for(const char* tensor : {"q", "k", "v", "o", "lse"})
    {
        for(const char* axis : {"batch", "token", "head"})
        {
            abi.push_back(scalarArgument(std::string("stride_") + tensor + "_" + axis));
        }
    }
    if(sinks)
    {
        abi.push_back(pointerArgument("sink_ptr"));
    }
    if(denseQ == 0)
    {
        abi.push_back(pointerArgument("cu_seqlens_q"));
    }
    if(seqUsedQ != 0)
    {
        abi.push_back(pointerArgument("seqused_q"));
    }
    abi.push_back(pointerArgument("seqused_k"));
    abi.push_back(pointerArgument("block_table"));
    for(const char* name : {"block_table_stride", "stride_k_block", "stride_v_block", "num_blocks"})
    {
        abi.push_back(scalarArgument(name));
    }
    return abi;
}

/// What prepare() throws for @p object against @p graph with the descriptor recording
/// @p recorded; the archive does not exist, so it always throws.
std::string prepareFailure(const PagedSpec& graph,
                           const ObjectSpec& object,
                           const std::vector<KernelArgument>& recorded)
{
    const Fixture fixture(graph);
    const auto bound = fixture.bound();
    EXPECT_TRUE(bound.has_value());
    if(!bound.has_value())
    {
        return {};
    }
    auto kernel = makeKernel(object);
    kernel.name = "wmma_attention.paged_signature_fixture";
    kernel.source.kind = KernelSourceKind::KPACK;
    kernel.source.library = "there-is-no-archive-here.kpack";
    kernel.source.tocKey = "toc#0";
    kernel.source.symbol = "wmma_fmha_fwd_v2";
    kernel.source.signature = recorded;
    const auto tree = std::filesystem::temp_directory_path().string();
    kernel.originDirectory = tree;
    kernel.treeRoot = tree;
    try
    {
        dispatchHandler().prepare(fixture.context(), *bound, kernel);
    }
    catch(const std::exception& error)
    {
        return error.what();
    }
    return {};
}

TEST(TestGfx1151WmmaAttentionPaged, PrepareMarshalsThePagedRockeAbi)
{
    PagedSpec denseUsed;
    denseUsed.seqLenQ = true;
    struct Case
    {
        const char* what;
        PagedSpec graph;
        int64_t denseQ;
        int64_t seqUsedQ;
    };
    for(const auto& c : {Case{"dense Q", PagedSpec{}, 1, 0},
                         Case{"dense Q with seq_len_q", denseUsed, 1, 1},
                         Case{"ragged Q", raggedGraph(), 0, 1}})
    {
        const auto object = pagedObject(128, 16, c.denseQ, c.seqUsedQ);
        const auto failure = prepareFailure(c.graph, object, rockePagedAbi(c.denseQ, c.seqUsedQ));
        ASSERT_FALSE(failure.empty()) << c.what;
        EXPECT_EQ(failure.find("is packaged with arguments"), std::string::npos)
            << c.what << ": " << failure;
        EXPECT_NE(failure.find(", symbol 'wmma_fmha_fwd_v2':"), std::string::npos)
            << c.what << ": " << failure;

        // The dense prefill ABI (no page lookup) is refused.
        auto dense = rockePagedAbi(1, 0);
        dense.resize(dense.size() - 5);
        EXPECT_NE(prepareFailure(c.graph, object, dense).find("is packaged with arguments"),
                  std::string::npos)
            << c.what;
    }
}

TEST(TestGfx1151WmmaAttentionPaged, PrepareMarshalsTheSinkBeforeTheQueryForm)
{
    struct Case
    {
        const char* what;
        PagedSpec graph;
        int64_t denseQ;
        int64_t seqUsedQ;
    };
    for(const auto& c : {Case{"dense Q", PagedSpec{}, 1, 0},
                         Case{"dense Q with seq_len_q", denseUsedGraph(), 1, 1},
                         Case{"ragged Q", raggedGraph(), 0, 1}})
    {
        const auto object = sinksObject(pagedObject(128, 16, c.denseQ, c.seqUsedQ));
        const auto graph = withSinks(c.graph);
        const auto failure
            = prepareFailure(graph, object, rockePagedAbi(c.denseQ, c.seqUsedQ, true));
        ASSERT_FALSE(failure.empty()) << c.what;
        EXPECT_EQ(failure.find("is packaged with arguments"), std::string::npos)
            << c.what << ": " << failure;
        EXPECT_NE(failure.find(", symbol 'wmma_fmha_fwd_v2':"), std::string::npos)
            << c.what << ": " << failure;

        // A descriptor without sink_ptr, or with it after the page lookup, is refused.
        EXPECT_NE(prepareFailure(graph, object, rockePagedAbi(c.denseQ, c.seqUsedQ))
                      .find("is packaged with arguments"),
                  std::string::npos)
            << c.what;
        auto late = rockePagedAbi(c.denseQ, c.seqUsedQ);
        late.push_back(pointerArgument("sink_ptr"));
        EXPECT_NE(prepareFailure(graph, object, late).find("is packaged with arguments"),
                  std::string::npos)
            << c.what;

        // A sinks object prepared for a graph without sinks is refused before any load.
        EXPECT_NE(prepareFailure(c.graph, object, rockePagedAbi(c.denseQ, c.seqUsedQ, true))
                      .find("sink use"),
                  std::string::npos)
            << c.what;
    }
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
