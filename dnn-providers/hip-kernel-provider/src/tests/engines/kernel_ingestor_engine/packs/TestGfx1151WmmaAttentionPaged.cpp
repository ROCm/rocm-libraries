// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "tests/engines/kernel_ingestor_engine/packs/Gfx1151WmmaAttentionTestSupport.hpp"

/**
 * @brief hipkernel:Gfx1151WmmaAttention on the cuDNN paged-K/V SDPA graph: which graphs
 *        the native matcher binds (dense or ragged Q, one page table or separate K and V
 *        tables, sequence lengths, arbitrary container strides incl. vLLM's interleaved
 *        K|V cache, attention sinks), which valid graphs it declines and the cause and
 *        reason it logs for each (one-sided paging, K and V tables of different widths or
 *        row strides, packed page tables, per-side block sizes, page sizes no object ships
 *        for, operands declaring less than the alignment the kernels load, features the
 *        kernels lack), what it binds, which catalog object serves each, and the kernel
 *        ABI prepare() marshals. Paged graphs are the prefill engine's alone:
 *        hipkernel:Gfx1151WmmaAttentionDecode, whose objects are dense, declines every one
 *        of them.
 *
 * No device: the GPU half (numerics against hipDNN's CPU SDPA reference, HIP-graph
 * capture and replay) is IntegrationGpuGfx1151WmmaAttentionPaged.cpp.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

using namespace gfx1151_wmma_attention;
using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;

// Restated rather than shared with the pack: these strings are the contract between the
// matcher that writes a token and the dispatch that reads it.
constexpr std::string_view LAYOUT_TOKEN = "gfx1151_wmma_attention.layout";
constexpr std::string_view PAGE_SIZE_TOKEN = "gfx1151_wmma_attention.page_size";
constexpr std::string_view PAGE_TABLE_UID_TOKEN = "gfx1151_wmma_attention.page_table.uid";
constexpr std::string_view PAGE_TABLE_V_UID_TOKEN = "gfx1151_wmma_attention.page_table_v.uid";
constexpr std::string_view PAGE_TABLE_STRIDE_TOKEN = "gfx1151_wmma_attention.page_table.stride";
constexpr std::string_view SEQ_LEN_KV_UID_TOKEN = "gfx1151_wmma_attention.seq_len_kv.uid";
constexpr std::string_view SEQ_LEN_Q_UID_TOKEN = "gfx1151_wmma_attention.seq_len_q.uid";
constexpr std::string_view Q_OFFSET_UID_TOKEN = "gfx1151_wmma_attention.q_offset.uid";
constexpr std::string_view MAX_SEQ_LEN_KV_TOKEN = "gfx1151_wmma_attention.max_seq_len_kv";
constexpr std::string_view SINK_UID_TOKEN = "gfx1151_wmma_attention.sink.uid";
constexpr std::string_view SINK_F32_TOKEN = "gfx1151_wmma_attention.sink.f32";

// GraphLayout in the pack.
constexpr int64_t LAYOUT_PAGED_DENSE_Q = 1;
constexpr int64_t LAYOUT_PAGED_RAGGED_Q = 2;

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
/// cache: dense Q [B, Hq, 1, D] (BSHD) with seq_len_q and seq_len_kv, K and V the two
/// halves of one interleaved cache [num_blocks, Hkv, page, 2D] viewed as
/// [num_blocks, Hkv, page, D], one block table (separateVTable: a second, V table).
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
    bool seqLenQ = true;
    bool seqLenKv = true;
    bool returnLse = false;
    bool raggedLse = false; // stats ragged on Q's offsets, token-major as in the golden bundles
    // Attention sinks: sink_token [1, sinkHeads, 1, 1] (Hq when absent) in sinkDtype (Q's
    // when absent) with head stride sinkHeadStride.
    bool sinks = false;
    std::optional<data_objects::DataType> sinkDtype;
    std::optional<int64_t> sinkHeads;
    int64_t sinkHeadStride = 1;
    // Valid graphs the kernels cannot read, and defects: the engine declines each.
    bool pageK = true;
    bool pageV = true;
    bool separateVTable = false;
    // The V table's width and row stride (the K table's when absent), and whether it is
    // packed; they apply with separateVTable.
    std::optional<int64_t> vTableBlocksPerSeq;
    std::optional<int64_t> vTableRowStride;
    bool packedVTable = false;
    bool packedTable = false;
    std::optional<int64_t> vPageSize; // pageSize when absent
    int64_t tableEntryStride = 1;
    data_objects::DataType tableDtype = data_objects::DataType::INT32;
    // The data types of seq_len_kv, seq_len_q and the ragged offsets.
    data_objects::DataType seqLenKvDtype = data_objects::DataType::INT32;
    data_objects::DataType seqLenQDtype = data_objects::DataType::INT32;
    data_objects::DataType offsetDtype = data_objects::DataType::INT32;
    std::optional<int64_t> qMultiplier; // the token stride when absent
    bool oOwnOffsets = false;
    bool alibi = false;
    bool bias = false;
    int64_t leftBound = -1;
    std::optional<int64_t> vNumBlocks; // numBlocks when absent
    // V's container strides (K's layout at V's page size when absent).
    std::optional<std::vector<int64_t>> vStrides;
    // Declared device-pointer alignment in bytes, by tensor uid (16 when absent).
    std::map<int64_t, int64_t> alignment;
    // Further SDPA attributes (FP8, dropout and other features the kernels lack).
    AttributeEdit extraAttributes;
};

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
    const std::vector<int64_t> vCacheDims{spec.vNumBlocks.value_or(spec.numBlocks),
                                          spec.kvHeads,
                                          spec.vPageSize.value_or(spec.pageSize),
                                          d};
    const int64_t row = spec.cacheRow.value_or(spec.interleavedCache ? 2 * d : d);
    const std::vector<int64_t> cacheStrides{
        spec.kvHeads * spec.pageSize * row, spec.pageSize * row, row, 1};
    const int64_t vPage = vCacheDims[2];
    const std::vector<int64_t> vCacheStrides = spec.vStrides.value_or(
        std::vector<int64_t>{spec.kvHeads * vPage * row, vPage * row, row, 1});
    const std::vector<int64_t> tableDims{b, 1, spec.maxBlocksPerSeq, 1};
    const std::vector<int64_t> tableStrides{
        spec.maxBlocksPerSeq * spec.tableEntryStride, 1, spec.tableEntryStride, 1};
    const int64_t vTableWidth = spec.vTableBlocksPerSeq.value_or(spec.maxBlocksPerSeq);
    const std::vector<int64_t> vTableDims{b, 1, vTableWidth, 1};
    const std::vector<int64_t> vTableStrides{
        spec.vTableRowStride.value_or(vTableWidth * spec.tableEntryStride),
        1,
        spec.tableEntryStride,
        1};
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
        const int64_t alignment = declaredAlignment(spec.alignment, uid);
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
                                                          alignment,
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
        tensor(V_UID, spec.dtype, vCacheStrides, vCacheDims),
        tensor(O_UID, spec.dtype, qStrides, qDims, oRagged, spec.raggedQ ? token : 1),
        tensor(PAGE_TABLE_UID,
               spec.tableDtype,
               tableStrides,
               tableDims,
               spec.packedTable ? ::flatbuffers::Optional<int64_t>(Q_OFFSET_UID)
                                : ::flatbuffers::nullopt,
               spec.packedTable ? spec.maxBlocksPerSeq : 1),
        tensor(SEQ_LEN_KV_UID, spec.seqLenKvDtype, vectorStrides, vectorDims),
        tensor(SEQ_LEN_Q_UID, spec.seqLenQDtype, vectorStrides, vectorDims),
        tensor(Q_OFFSET_UID, spec.offsetDtype, vectorStrides, offsetDims),
        tensor(O_OFFSET_UID, data_objects::DataType::INT32, vectorStrides, offsetDims),
        tensor(PAGE_TABLE_V_UID,
               spec.tableDtype,
               vTableStrides,
               vTableDims,
               spec.packedVTable ? ::flatbuffers::Optional<int64_t>(Q_OFFSET_UID)
                                 : ::flatbuffers::nullopt,
               spec.packedVTable ? vTableWidth : 1),
        tensor(SINK_UID, spec.sinkDtype.value_or(spec.dtype), sinkStrides, sinkDims),
        tensor(BIAS_UID, data_objects::DataType::FLOAT, biasStrides, biasDims),
    };
    if(spec.returnLse)
    {
        const std::vector<int64_t> raggedLseStrides{
            spec.queryLength * spec.queryHeads, 1, spec.queryHeads, 1};
        tensors.push_back(
            spec.raggedLse ? tensor(LSE_UID,
                                    data_objects::DataType::FLOAT,
                                    raggedLseStrides,
                                    lseDims,
                                    ::flatbuffers::Optional<int64_t>(Q_OFFSET_UID),
                                    spec.queryHeads)
                           : tensor(LSE_UID, data_objects::DataType::FLOAT, lseStrides, lseDims));
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
    if(spec.pageK)
    {
        attributes.add_page_table_k_tensor_uid(PAGE_TABLE_UID);
    }
    if(spec.pageV)
    {
        attributes.add_page_table_v_tensor_uid(spec.separateVTable ? PAGE_TABLE_V_UID
                                                                   : PAGE_TABLE_UID);
    }
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
    if(spec.extraAttributes)
    {
        spec.extraAttributes(attributes);
    }

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

/// @p spec on a gfx1100 (48 WGP) device.
EngineGraphFixture fixtureFor(const PagedSpec& spec)
{
    return EngineGraphFixture{buildPagedGraph(spec), deviceProperties("gfx1100", 48)};
}

bool matches(const PagedSpec& spec)
{
    return fixtureFor(spec).bound().has_value();
}

/// A shipped paged prefill object (configs/gfx1151_wmma_attention.yaml), BF16.
KernelSpec pagedObject(int64_t headSize, int64_t page, int64_t denseQ, int64_t seqUsedQ)
{
    KernelSpec spec;
    spec.dtype = "BF16";
    spec.headSize = headSize;
    spec.headSizeV = headSize;
    spec.blockM = headSize > 128 ? 64 : 128;
    spec.layout = "paged";
    spec.pageBlockSize = page;
    spec.denseQ = denseQ;
    spec.seqUsedQ = seqUsedQ;
    return spec;
}

/// A shipped dense prefill object, BF16.
KernelSpec denseObject(int64_t headSize)
{
    KernelSpec spec;
    spec.dtype = "BF16";
    spec.headSize = headSize;
    spec.headSizeV = headSize;
    spec.blockM = headSize > 128 ? 64 : 128;
    return spec;
}

/// Whether the kernel matcher of the engine @p symbols names admits @p object for
/// @p graph, given the tokens the prefill engine's graph matcher bound (the decode
/// engine binds none).
bool engineServes(const PackSymbols& symbols, const PagedSpec& graph, const KernelSpec& object)
{
    const auto fixture = fixtureFor(graph);
    const auto bound = fixture.bound();
    return bound.has_value()
           && matchesKernel(symbols, fixture.context(), makeKernel(object), *bound);
}

bool serves(const PagedSpec& graph, const KernelSpec& object)
{
    return engineServes(PREFILL_SYMBOLS, graph, object);
}

bool decodeEngineServes(const PagedSpec& graph, const KernelSpec& object)
{
    return engineServes(DECODE_SYMBOLS, graph, object);
}

/// The shipped decode object of @p headSize: four waves, runtime masks, dense.
KernelSpec decodeObject(int64_t headSize)
{
    KernelSpec spec = denseObject(headSize);
    spec.kind = "decode";
    spec.maskMode = "runtime";
    spec.blockSize = 128;
    spec.blockM = 16;
    return spec;
}

PagedSpec raggedGraph()
{
    PagedSpec spec;
    spec.raggedQ = true;
    spec.queryLength = 300; // the longest sequence's query rows
    return spec;
}

/// What the engine @p symbols names logged at INFO when it declined @p spec.
LoggedDecline loggedDecline(const PagedSpec& spec, const PackSymbols& symbols = PREFILL_SYMBOLS)
{
    return recordDecline([&] { return fixtureFor(spec).bound(symbols).has_value(); });
}

/// @p spec with attention sinks: sink_token [1, Hq, 1, 1] in the Q data type.
PagedSpec withSinks(PagedSpec spec)
{
    spec.sinks = true;
    return spec;
}

/// @p spec's sinks variant, which takes sink_ptr.
KernelSpec sinksObject(KernelSpec spec)
{
    spec.sinks = 1;
    return spec;
}

// -----------------------------------------------------------------------------
// Graph match
// -----------------------------------------------------------------------------

TEST(TestGfx1151WmmaAttentionPaged, BindsUniformDecodeOverTheInterleavedCache)
{
    const auto fixture = fixtureFor(PagedSpec{});
    const auto bound = fixture.bound();
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(boundInt(*bound, LAYOUT_TOKEN), LAYOUT_PAGED_DENSE_Q);
    EXPECT_EQ(boundInt(*bound, PAGE_SIZE_TOKEN), 16);
    EXPECT_EQ(boundInt(*bound, PAGE_TABLE_UID_TOKEN), PAGE_TABLE_UID);
    EXPECT_EQ(boundInt(*bound, PAGE_TABLE_STRIDE_TOKEN), 8);
    EXPECT_EQ(boundInt(*bound, SEQ_LEN_KV_UID_TOKEN), SEQ_LEN_KV_UID);
    EXPECT_EQ(boundInt(*bound, SEQ_LEN_Q_UID_TOKEN), SEQ_LEN_Q_UID);
    EXPECT_EQ(boundInt(*bound, MAX_SEQ_LEN_KV_TOKEN), 128);
    EXPECT_FALSE(tryGetBoundInt(*bound, Q_OFFSET_UID_TOKEN).has_value());
}

TEST(TestGfx1151WmmaAttentionPaged, BindsSpeculativeDecodeWithStats)
{
    PagedSpec spec;
    spec.queryLength = 3; // speculative decode, k = 2
    spec.returnLse = true;
    const auto fixture = fixtureFor(spec);
    const auto bound = fixture.bound();
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(boundInt(*bound, LAYOUT_TOKEN), LAYOUT_PAGED_DENSE_Q);
    EXPECT_EQ(boundInt(*bound, SEQ_LEN_Q_UID_TOKEN), SEQ_LEN_Q_UID);
}

// An unset max_seq_len_kv is valid (cuDNN infers it); the kernels need it only as the bound
// on seq_len_kv, so the keys one table row maps serve.
TEST(TestGfx1151WmmaAttentionPaged, InfersAnUnsetMaxSeqLenKvFromTheTable)
{
    PagedSpec spec;
    spec.maxSeqLenKv.reset();
    spec.maxBlocksPerSeq = 5;
    spec.pageSize = 32;
    const auto bound = fixtureFor(spec).bound();
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(boundInt(*bound, MAX_SEQ_LEN_KV_TOKEN), 160);
}

TEST(TestGfx1151WmmaAttentionPaged, BindsARaggedQueryAndItsOffsets)
{
    const auto fixture = fixtureFor(raggedGraph());
    const auto bound = fixture.bound();
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(boundInt(*bound, LAYOUT_TOKEN), LAYOUT_PAGED_RAGGED_Q);
    EXPECT_EQ(boundInt(*bound, SEQ_LEN_Q_UID_TOKEN), SEQ_LEN_Q_UID);
    EXPECT_EQ(boundInt(*bound, Q_OFFSET_UID_TOKEN), Q_OFFSET_UID);
}

// cuDNN's page_table_k and page_table_v may be two tensors: the kernel reads K pages from
// one and V pages from the other, at one row offset.
TEST(TestGfx1151WmmaAttentionPaged, BindsSeparateKAndVPageTables)
{
    for(auto spec : {PagedSpec{}, raggedGraph()})
    {
        spec.separateVTable = true;
        const auto bound = fixtureFor(spec).bound();
        ASSERT_TRUE(bound.has_value()) << spec.raggedQ;
        EXPECT_EQ(boundInt(*bound, PAGE_TABLE_UID_TOKEN), PAGE_TABLE_UID);
        EXPECT_EQ(boundInt(*bound, PAGE_TABLE_V_UID_TOKEN), PAGE_TABLE_V_UID);
        EXPECT_EQ(boundInt(*bound, PAGE_TABLE_STRIDE_TOKEN), 8);
    }
    // One table bound to both sides binds the same uid twice.
    const auto shared = fixtureFor(PagedSpec{}).bound();
    ASSERT_TRUE(shared.has_value());
    EXPECT_EQ(boundInt(*shared, PAGE_TABLE_V_UID_TOKEN), PAGE_TABLE_UID);
    // One sequence never steps a row, so the row strides may differ.
    PagedSpec single;
    single.batch = 1;
    single.separateVTable = true;
    single.vTableRowStride = 32;
    EXPECT_TRUE(matches(single));
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

// V is its own container: its page, head and token strides need not be K's (the GPU
// suite's padded_v cases check the kernel reads V through them). Each is checked on V.
TEST(TestGfx1151WmmaAttentionPaged, BindsAVContainerWithItsOwnStrides)
{
    PagedSpec spec;
    spec.interleavedCache = false;
    const int64_t row = spec.headSize + 8;
    const int64_t head = spec.pageSize * row + 8;
    spec.vStrides = std::vector<int64_t>{spec.kvHeads * head + 16, head, row, 1};
    EXPECT_TRUE(matches(spec));
    for(const auto& misaligned : {std::vector<int64_t>{spec.kvHeads * head + 4, head, row, 1},
                                  std::vector<int64_t>{spec.kvHeads * head, head + 4, row, 1},
                                  std::vector<int64_t>{spec.kvHeads * head, head, row - 4, 1}})
    {
        auto off = spec;
        off.vStrides = misaligned;
        EXPECT_FALSE(matches(off));
    }
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
    add("no seq_len_q", [](PagedSpec& s) { s.seqLenQ = false; });
    add("a ragged Q without seq_len_q", [](PagedSpec& s) { s.seqLenQ = false; }, raggedGraph());
    add("max_seq_len_kv past the table", [](PagedSpec& s) { s.maxSeqLenKv = 129; });
    add("a page size that is not a power of two", [](PagedSpec& s) { s.pageSize = 48; });
    add("a page size no object ships for", [](PagedSpec& s) { s.pageSize = 8; });
    // Valid paged graphs the kernels cannot read: they page K and V through tables of one
    // width and row stride.
    add("only K paged", [](PagedSpec& s) { s.pageV = false; });
    add("only V paged", [](PagedSpec& s) { s.pageK = false; });
    add("K and V tables of different widths", [](PagedSpec& s) {
        s.separateVTable = true;
        s.vTableBlocksPerSeq = 4;
        s.vTableRowStride = s.maxBlocksPerSeq; // the K table's row stride
        s.maxSeqLenKv.reset();
    });
    add("K and V tables of different row strides", [](PagedSpec& s) {
        s.separateVTable = true;
        s.vTableRowStride = 16;
    });
    add("a packed page table", [](PagedSpec& s) { s.packedTable = true; });
    add("a packed V page table", [](PagedSpec& s) {
        s.separateVTable = true;
        s.packedVTable = true;
    });
    add("V pages of another block size", [](PagedSpec& s) { s.vPageSize = 32; });
    add("a strided page-table row", [](PagedSpec& s) { s.tableEntryStride = 2; });
    add("an INT64 page table", [](PagedSpec& s) { s.tableDtype = data_objects::DataType::INT64; });
    add("an additive bias", [](PagedSpec& s) { s.bias = true; });
    add("ALiBi", [](PagedSpec& s) { s.alibi = true; });
    add("a head dim off the 8-element grid", [](PagedSpec& s) { s.headSize = 76; });
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

// hipDNN reports a declined graph only as "no engine configurations", so each valid paged
// graph the kernels cannot serve logs its cause, under the declining engine's name, and
// each check its own reason.
TEST(TestGfx1151WmmaAttentionPaged, LogsTheCauseOfEachDecline)
{
    struct Case
    {
        std::string what;
        PagedSpec spec;
        const char* cause;
        std::string reason; // a part of the logged reason
    };
    std::vector<Case> cases;
    const auto add = [&cases](const std::string& what,
                              const char* cause,
                              const std::string& reason,
                              auto&& edit,
                              PagedSpec spec = PagedSpec{}) {
        edit(spec);
        cases.push_back({what, spec, cause, reason});
    };
    const auto setting = [](AttributeUid uid) {
        return [uid](PagedSpec& s) { s.extraAttributes = settingUid(uid); };
    };
    add("no seq_len_q", "seq_len_q", "needs seq_len_q", [](PagedSpec& s) { s.seqLenQ = false; });
    add("no seq_len_kv", "seq_len_kv", "needs seq_len_kv", [](PagedSpec& s) {
        s.seqLenKv = false;
    });
    add("an INT64 seq_len_q", "seq_len_q", "INT32 [B, 1, 1, 1]", [](PagedSpec& s) {
        s.seqLenQDtype = data_objects::DataType::INT64;
    });
    add("an INT64 seq_len_kv", "seq_len_kv", "INT32 [B, 1, 1, 1]", [](PagedSpec& s) {
        s.seqLenKvDtype = data_objects::DataType::INT64;
    });
    add(
        "INT64 ragged offsets",
        "ragged",
        "INT32 [B+1, 1, 1, 1]",
        [](PagedSpec& s) { s.offsetDtype = data_objects::DataType::INT64; },
        raggedGraph());
    add("only K paged", "one_sided_paging", "only K is paged", [](PagedSpec& s) {
        s.pageV = false;
    });
    add("only V paged", "one_sided_paging", "only V is paged", [](PagedSpec& s) {
        s.pageK = false;
    });
    add("a packed page table", "page_table", "packed", [](PagedSpec& s) { s.packedTable = true; });
    add("a packed V page table", "page_table", "packed", [](PagedSpec& s) {
        s.separateVTable = true;
        s.packedVTable = true;
    });
    add("a strided page-table row",
        "page_table",
        "page_table_k must be a device INT32 [B, 1, M, 1]",
        [](PagedSpec& s) { s.tableEntryStride = 2; });
    add("an INT64 page table",
        "page_table",
        "page_table_k must be a device INT32 [B, 1, M, 1]",
        [](PagedSpec& s) { s.tableDtype = data_objects::DataType::INT64; });
    add("V pages of another block size",
        "block_size",
        "differ in block size (16 and 32)",
        [](PagedSpec& s) { s.vPageSize = 32; });
    add("V with more pages than K", "shape", "differ in page count (64 and 128)", [](PagedSpec& s) {
        s.vNumBlocks = s.numBlocks * 2;
    });
    add("a page size that is not a power of two",
        "block_size",
        "a page size of 48 is not a power of two",
        [](PagedSpec& s) { s.pageSize = 48; });
    for(const int64_t page : {8, 128, 256})
    {
        add("an unshipped page size of " + std::to_string(page),
            "block_size",
            "no paged object ships for a page size of " + std::to_string(page),
            [page](PagedSpec& s) { s.pageSize = page; });
    }
    add("K and V tables of different widths",
        "page_table",
        "differ in width (8 and 4)",
        [](PagedSpec& s) {
            s.separateVTable = true;
            s.vTableBlocksPerSeq = 4;
            s.vTableRowStride = s.maxBlocksPerSeq; // the K table's row stride
            s.maxSeqLenKv.reset();
        });
    add("K and V tables of different row strides",
        "page_table",
        "differ in row stride (8 and 16)",
        [](PagedSpec& s) {
            s.separateVTable = true;
            s.vTableRowStride = 16;
        });
    add("an additive bias", "attn_mask", "no additive attention mask", [](PagedSpec& s) {
        s.bias = true;
    });
    add(
        "an additive bias with a ragged Q",
        "attn_mask",
        "no additive attention mask",
        [](PagedSpec& s) { s.bias = true; },
        raggedGraph());
    add(
        "stats of a ragged Q",
        "stats",
        "no stats for a ragged Q",
        [](PagedSpec& s) { s.returnLse = true; },
        raggedGraph());
    add(
        "ragged stats of a ragged Q",
        "stats",
        "no stats for a ragged Q",
        [](PagedSpec& s) {
            s.returnLse = true;
            s.raggedLse = true;
        },
        raggedGraph());
    add(
        "a ragged Q in element-unit offsets",
        "ragged",
        "ragged_offset_multiplier must equal the token stride",
        [](PagedSpec& s) { s.qMultiplier = 1; },
        raggedGraph());
    add(
        "O with its own offsets",
        "ragged",
        "Q and O must share one ragged offset tensor",
        [](PagedSpec& s) { s.oOwnOffsets = true; },
        raggedGraph());
    add("ALiBi", "alibi", "no per-head slopes tensor", [](PagedSpec& s) { s.alibi = true; });
    using Builder = data_objects::SdpaAttributesBuilder;
    for(const auto& [name, uid] : std::vector<std::pair<const char*, AttributeUid>>{
            {"descale_q", &Builder::add_descale_q_tensor_uid},
            {"descale_k", &Builder::add_descale_k_tensor_uid},
            {"descale_v", &Builder::add_descale_v_tensor_uid},
            {"descale_s", &Builder::add_descale_s_tensor_uid},
            {"scale_s", &Builder::add_scale_s_tensor_uid},
            {"scale_o", &Builder::add_scale_o_tensor_uid},
            {"amax_s", &Builder::add_amax_s_tensor_uid},
            {"amax_o", &Builder::add_amax_o_tensor_uid}})
    {
        add(name, "fp8", "FP8 descale / amax operands", setting(uid));
        add(std::string(name) + " with a ragged Q",
            "fp8",
            "FP8 descale / amax operands",
            setting(uid),
            raggedGraph());
    }
    for(const auto& [name, uid] : std::vector<std::pair<const char*, AttributeUid>>{
            {"seed", &Builder::add_seed_tensor_uid},
            {"offset", &Builder::add_offset_tensor_uid},
            {"dropout_mask", &Builder::add_dropout_mask_tensor_uid},
            {"dropout_scale", &Builder::add_dropout_scale_tensor_uid},
            {"rng_dump", &Builder::add_rng_dump_tensor_uid}})
    {
        add(name, "dropout", "no dropout", setting(uid));
    }
    add("a dropout probability", "dropout", "no dropout", [](PagedSpec& s) {
        s.extraAttributes = [](Builder& attributes) { attributes.add_dropout_probability(0.1F); };
    });
    add("block_mask", "feature", "a block mask", setting(&Builder::add_block_mask_tensor_uid));
    add("max", "feature", "max / sum_exp outputs", setting(&Builder::add_max_tensor_uid));
    add("sum_exp", "feature", "max / sum_exp outputs", setting(&Builder::add_sum_exp_tensor_uid));
    for(const auto implementation : {data_objects::AttentionImplementation::COMPOSITE,
                                     data_objects::AttentionImplementation::UNIFIED})
    {
        add(std::string("a forced ") + data_objects::EnumNameAttentionImplementation(implementation)
                + " implementation",
            "feature",
            "a forced implementation",
            [implementation](PagedSpec& s) {
                s.extraAttributes = [implementation](Builder& attributes) {
                    attributes.add_implementation(implementation);
                };
            });
    }
    ASSERT_FALSE(cases.empty());
    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.what);
        expectDecline(loggedDecline(c.spec), PREFILL_ENGINE, c.cause, c.reason);
    }
}

// The kernels load Q / K / V / O rows as 16-byte vectors, the INT32 operands and the stats
// as 4-byte words and the sink with 16-bit loads (two for a FLOAT sink). A graph may
// declare a lower pointer alignment (engine API 1.3.0, which the engine's 1.4.0 admits);
// the matcher declines it with its cause, so hipDNN can pick another engine at plan time
// instead of the plan failing at execute.
TEST(TestGfx1151WmmaAttentionPaged, DeclinesOperandsDeclaringLessThanTheKernelsLoad)
{
    PagedSpec separate;
    separate.separateVTable = true;
    for(const auto& base : {PagedSpec{}, raggedGraph(), separate, withSinks(PagedSpec{})})
    {
        auto widerAlignment = base;
        for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
        {
            widerAlignment.alignment[uid] = 32;
        }
        for(const int64_t uid :
            {PAGE_TABLE_UID, PAGE_TABLE_V_UID, SEQ_LEN_Q_UID, SEQ_LEN_KV_UID, Q_OFFSET_UID})
        {
            widerAlignment.alignment[uid] = 4;
        }
        widerAlignment.alignment[SINK_UID] = 2;
        EXPECT_TRUE(matches(widerAlignment));
    }
    PagedSpec stats;
    stats.returnLse = true;
    stats.alignment[LSE_UID] = 4;
    EXPECT_TRUE(matches(stats));
    auto floatSink = withSinks(PagedSpec{});
    floatSink.sinkDtype = data_objects::DataType::FLOAT;
    floatSink.alignment[SINK_UID] = 4;
    EXPECT_TRUE(matches(floatSink));
    const auto declining = [](PagedSpec spec, int64_t uid, int64_t bytes) {
        spec.alignment[uid] = bytes;
        return spec;
    };
    for(const auto& [role, uid] : std::vector<std::pair<std::string, int64_t>>{
            {"Q", Q_UID}, {"K", K_UID}, {"V", V_UID}, {"O", O_UID}})
    {
        for(const int64_t bytes : {8, 24})
        {
            SCOPED_TRACE(role + ", " + std::to_string(bytes) + " bytes");
            const auto reason = role + " declares a " + std::to_string(bytes) + "-byte alignment";
            for(const auto& base : {PagedSpec{}, raggedGraph()})
            {
                expectDecline(loggedDecline(declining(base, uid, bytes)),
                              PREFILL_ENGINE,
                              "alignment",
                              reason);
            }
        }
    }
    for(const auto& [role, spec, uid] : std::vector<std::tuple<std::string, PagedSpec, int64_t>>{
            {"page_table_k", PagedSpec{}, PAGE_TABLE_UID},
            {"page_table_v", separate, PAGE_TABLE_V_UID},
            {"seq_len_q", PagedSpec{}, SEQ_LEN_Q_UID},
            {"seq_len_kv", PagedSpec{}, SEQ_LEN_KV_UID},
            {"ragged_offset", raggedGraph(), Q_OFFSET_UID}})
    {
        SCOPED_TRACE(role);
        expectDecline(loggedDecline(declining(spec, uid, 2)),
                      PREFILL_ENGINE,
                      "alignment",
                      role + " declares a 2-byte alignment");
    }
    // The stats are FP32 words: they must declare a multiple of 4 bytes.
    for(const int64_t bytes : {1, 2})
    {
        SCOPED_TRACE("Stats, " + std::to_string(bytes) + " bytes");
        expectDecline(loggedDecline(declining(stats, LSE_UID, bytes)),
                      PREFILL_ENGINE,
                      "alignment",
                      "Stats declares a " + std::to_string(bytes) + "-byte alignment");
    }
    // A Q-dtype sink loads as one 16-bit word, a FLOAT one as two: a multiple of 2 or 4
    // bytes.
    for(const int64_t bytes : {1, 3})
    {
        SCOPED_TRACE("sink_token, " + std::to_string(bytes) + " bytes");
        for(const auto& base : {withSinks(PagedSpec{}), withSinks(raggedGraph())})
        {
            expectDecline(loggedDecline(declining(base, SINK_UID, bytes)),
                          PREFILL_ENGINE,
                          "alignment",
                          "sink_token declares a " + std::to_string(bytes) + "-byte alignment");
        }
    }
    for(const int64_t bytes : {1, 2})
    {
        SCOPED_TRACE("FLOAT sink_token, " + std::to_string(bytes) + " bytes");
        expectDecline(loggedDecline(declining(floatSink, SINK_UID, bytes)),
                      PREFILL_ENGINE,
                      "alignment",
                      "sink_token declares a " + std::to_string(bytes) + "-byte alignment");
    }
}

// -----------------------------------------------------------------------------
// Kernel match and score
// -----------------------------------------------------------------------------

TEST(TestGfx1151WmmaAttentionPaged, OnlyTheObjectOfThePageSizeAndQueryFormServes)
{
    const PagedSpec dense;
    const PagedSpec ragged = raggedGraph();

    EXPECT_TRUE(serves(dense, pagedObject(128, 16, 1, 1)));
    EXPECT_TRUE(serves(ragged, pagedObject(128, 16, 0, 1)));
    for(const auto& graph : {dense, ragged})
    {
        EXPECT_FALSE(serves(graph, pagedObject(128, 32, 1, 1)));
        EXPECT_FALSE(serves(graph, pagedObject(128, 32, 0, 1)));
        // Paged graphs always carry seq_len_q: an object without seqused_q serves none.
        EXPECT_FALSE(serves(graph, pagedObject(128, 16, 1, 0)));
        EXPECT_FALSE(serves(graph, pagedObject(128, 16, 0, 0)));
        // Dense and decode objects never serve a paged graph.
        EXPECT_FALSE(serves(graph, denseObject(128)));
        EXPECT_FALSE(serves(graph, decodeObject(128)));
    }
    EXPECT_FALSE(serves(dense, pagedObject(128, 16, 0, 1)));
    EXPECT_FALSE(serves(ragged, pagedObject(128, 16, 1, 1)));
    // A narrower bucket cannot hold the head; a wider one can.
    EXPECT_FALSE(serves(dense, pagedObject(96, 16, 1, 1)));
    EXPECT_TRUE(serves(dense, pagedObject(256, 16, 1, 1)));
}

// -----------------------------------------------------------------------------
// Attention sinks: sink_token [1, Hq, 1, 1], FLOAT or in the Q data type, selects the
// paged sinks objects, which take sink_ptr and sink_is_f32 before the query form's
// operands.
// -----------------------------------------------------------------------------

TEST(TestGfx1151WmmaAttentionPaged, BindsAttentionSinksInEveryQueryForm)
{
    PagedSpec half;
    half.dtype = data_objects::DataType::HALF;
    half.interleavedCache = false;
    for(const auto& graph : {PagedSpec{}, raggedGraph(), half})
    {
        const auto plain = fixtureFor(graph).bound();
        ASSERT_TRUE(plain.has_value());
        EXPECT_FALSE(tryGetBoundInt(*plain, SINK_UID_TOKEN).has_value());
        EXPECT_FALSE(tryGetBoundInt(*plain, SINK_F32_TOKEN).has_value());
        const auto bound = fixtureFor(withSinks(graph)).bound();
        ASSERT_TRUE(bound.has_value());
        EXPECT_EQ(boundInt(*bound, SINK_UID_TOKEN), SINK_UID);
        EXPECT_EQ(boundInt(*bound, SINK_F32_TOKEN), 0);
        EXPECT_EQ(boundInt(*bound, LAYOUT_TOKEN), boundInt(*plain, LAYOUT_TOKEN));
        // hipDNN's FLOAT sink_token binds the same object with sink_is_f32 set.
        auto floatSink = withSinks(graph);
        floatSink.sinkDtype = data_objects::DataType::FLOAT;
        const auto f32 = fixtureFor(floatSink).bound();
        ASSERT_TRUE(f32.has_value());
        EXPECT_EQ(boundInt(*f32, SINK_UID_TOKEN), SINK_UID);
        EXPECT_EQ(boundInt(*f32, SINK_F32_TOKEN), 1);
        EXPECT_EQ(boundInt(*f32, LAYOUT_TOKEN), boundInt(*plain, LAYOUT_TOKEN));
    }
}

TEST(TestGfx1151WmmaAttentionPaged, DeclinesSinksTheKernelsCannotRead)
{
    struct Case
    {
        const char* what;
        PagedSpec spec;
        const char* reason; // a part of the logged reason
    };
    std::vector<Case> cases;
    const auto add = [&cases](const char* what, const char* reason, auto&& edit) {
        PagedSpec spec = withSinks(PagedSpec{});
        edit(spec);
        cases.push_back({what, spec, reason});
    };
    // The kernels read the sink as FLOAT or in the I/O data type.
    add("an INT32 sink", "FLOAT or in the Q data type", [](PagedSpec& s) {
        s.sinkDtype = data_objects::DataType::INT32;
    });
    add("a HALF sink for a BFLOAT16 query", "FLOAT or in the Q data type", [](PagedSpec& s) {
        s.sinkDtype = data_objects::DataType::HALF;
    });
    add("one sink per K/V head", "[1, Hq, 1, 1]", [](PagedSpec& s) { s.sinkHeads = s.kvHeads; });
    add("a strided head axis", "unit head stride", [](PagedSpec& s) { s.sinkHeadStride = 2; });
    ASSERT_TRUE(matches(withSinks(PagedSpec{})));
    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.what);
        expectDecline(loggedDecline(c.spec), PREFILL_ENGINE, "sinks", c.reason);
    }
}

TEST(TestGfx1151WmmaAttentionPaged, OnlyTheSinksObjectServesASinksGraph)
{
    struct Case
    {
        PagedSpec graph;
        int64_t denseQ;
    };
    for(const auto& c : {Case{PagedSpec{}, 1}, Case{raggedGraph(), 0}})
    {
        const auto plain = pagedObject(128, 16, c.denseQ, 1);
        EXPECT_TRUE(serves(withSinks(c.graph), sinksObject(plain)));
        EXPECT_FALSE(serves(withSinks(c.graph), plain));
        EXPECT_FALSE(serves(c.graph, sinksObject(plain)));
        EXPECT_TRUE(serves(c.graph, plain));
        // The page size and query form still decide among the sinks objects.
        EXPECT_FALSE(serves(withSinks(c.graph), sinksObject(pagedObject(128, 32, c.denseQ, 1))));
        EXPECT_FALSE(
            serves(withSinks(c.graph), sinksObject(pagedObject(128, 16, 1 - c.denseQ, 1))));
    }
    // sinks is 0 or 1.
    auto malformed = pagedObject(128, 16, 1, 1);
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

    EXPECT_TRUE(serves(decodeShaped, pagedObject(128, 16, 1, 1)));
    EXPECT_TRUE(serves(PagedSpec{}, pagedObject(128, 16, 1, 1)));
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
                             raggedGraph(),
                             withSinks(PagedSpec{}),
                             withSinks(raggedGraph())})
    {
        const auto fixture = fixtureFor(graph);
        ASSERT_TRUE(fixture.bound().has_value());
        // Its graph matcher binds no paged graph, and says so under its own name ...
        expectDecline(loggedDecline(graph, DECODE_SYMBOLS),
                      DECODE_ENGINE,
                      "page_table",
                      "the decode objects read dense K/V");
        // ... and its kernel matcher admits no object for one, its own or a paged one.
        EXPECT_FALSE(decodeEngineServes(graph, decodeObject(128)));
        EXPECT_FALSE(decodeEngineServes(graph, pagedObject(128, 16, 1, 1)));
        EXPECT_FALSE(decodeEngineServes(graph, pagedObject(128, 16, 0, 1)));
        EXPECT_FALSE(decodeEngineServes(graph, sinksObject(pagedObject(128, 16, 1, 1))));
    }
}

TEST(TestGfx1151WmmaAttentionPaged, ScorePrefersTheTightestBucket)
{
    const auto fixture = fixtureFor(PagedSpec{});
    const auto bound = fixture.bound();
    ASSERT_TRUE(bound.has_value());
    const auto score = [&](const KernelSpec& object) {
        return scoreKernel(PREFILL_SYMBOLS, fixture.context(), makeKernel(object), *bound);
    };
    EXPECT_GT(score(pagedObject(128, 16, 1, 1)), score(pagedObject(256, 16, 1, 1)));
}

// -----------------------------------------------------------------------------
// Dispatch: workspace and the paged prefill ABI
// -----------------------------------------------------------------------------

TEST(TestGfx1151WmmaAttentionPaged, NeedsNoWorkspace)
{
    for(const auto& [graph, object] : {std::pair{PagedSpec{}, pagedObject(128, 16, 1, 1)},
                                       std::pair{raggedGraph(), pagedObject(128, 16, 0, 1)}})
    {
        const auto fixture = fixtureFor(graph);
        const auto bound = fixture.bound();
        ASSERT_TRUE(bound.has_value());
        EXPECT_EQ(dispatchHandler(PREFILL_SYMBOLS)
                      .workspaceBytes(fixture.context(), *bound, makeKernel(object)),
                  0U);
    }
}

/// wmma_fmha_fwd_v2._declare_params of a shipped paged object (all take seqused_q).
std::vector<KernelArgument> rockePagedAbi(int64_t denseQ, bool sinks = false)
{
    auto abi = rockePrefillAbiHead();
    if(sinks)
    {
        abi.push_back(pointerArgument("sink_ptr"));
        appendScalars(abi, {"sink_is_f32"});
    }
    if(denseQ == 0)
    {
        abi.push_back(pointerArgument("cu_seqlens_q"));
    }
    for(const char* name : {"seqused_q", "seqused_k", "block_table", "block_table_v"})
    {
        abi.push_back(pointerArgument(name));
    }
    appendScalars(abi, {"block_table_stride", "stride_k_block", "stride_v_block", "num_blocks"});
    return abi;
}

/// What prepare() throws for @p object against @p graph with the descriptor recording
/// @p recorded; the archive does not exist, so it always throws.
std::string prepareFailure(const PagedSpec& graph,
                           const KernelSpec& object,
                           const std::vector<KernelArgument>& recorded)
{
    return gfx1151_wmma_attention::prepareFailure(fixtureFor(graph),
                                                  kpackDescriptor(object, recorded));
}

TEST(TestGfx1151WmmaAttentionPaged, PrepareMarshalsThePagedRockeAbi)
{
    struct Case
    {
        const char* what;
        PagedSpec graph;
        int64_t denseQ;
    };
    for(const auto& c : {Case{"dense Q", PagedSpec{}, 1}, Case{"ragged Q", raggedGraph(), 0}})
    {
        const auto object = pagedObject(128, 16, c.denseQ, 1);
        const auto failure = prepareFailure(c.graph, object, rockePagedAbi(c.denseQ));
        ASSERT_FALSE(failure.empty()) << c.what;
        EXPECT_EQ(failure.find("is packaged with arguments"), std::string::npos)
            << c.what << ": " << failure;
        EXPECT_NE(failure.find(", symbol 'wmma_fmha_fwd_v2':"), std::string::npos)
            << c.what << ": " << failure;

        // The dense prefill ABI (no live counts or page lookup) is refused.
        auto dense = rockePagedAbi(1);
        dense.resize(dense.size() - 8);
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
    };
    for(const auto& c : {Case{"dense Q", PagedSpec{}, 1}, Case{"ragged Q", raggedGraph(), 0}})
    {
        const auto object = sinksObject(pagedObject(128, 16, c.denseQ, 1));
        const auto graph = withSinks(c.graph);
        const auto failure = prepareFailure(graph, object, rockePagedAbi(c.denseQ, true));
        ASSERT_FALSE(failure.empty()) << c.what;
        EXPECT_EQ(failure.find("is packaged with arguments"), std::string::npos)
            << c.what << ": " << failure;
        EXPECT_NE(failure.find(", symbol 'wmma_fmha_fwd_v2':"), std::string::npos)
            << c.what << ": " << failure;

        // A descriptor without sink_ptr or sink_is_f32, or with the sink after the page
        // lookup, is refused.
        EXPECT_NE(prepareFailure(graph, object, rockePagedAbi(c.denseQ))
                      .find("is packaged with arguments"),
                  std::string::npos)
            << c.what;
        auto noFlag = rockePagedAbi(c.denseQ);
        noFlag.insert(noFlag.begin() + static_cast<std::ptrdiff_t>(rockePrefillAbiHead().size()),
                      pointerArgument("sink_ptr"));
        EXPECT_NE(prepareFailure(graph, object, noFlag).find("is packaged with arguments"),
                  std::string::npos)
            << c.what;
        auto late = rockePagedAbi(c.denseQ);
        late.push_back(pointerArgument("sink_ptr"));
        appendScalars(late, {"sink_is_f32"});
        EXPECT_NE(prepareFailure(graph, object, late).find("is packaged with arguments"),
                  std::string::npos)
            << c.what;

        // A sinks object prepared for a graph without sinks is refused before any load.
        EXPECT_NE(prepareFailure(c.graph, object, rockePagedAbi(c.denseQ, true)).find("sink use"),
                  std::string::npos)
            << c.what;
    }
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
