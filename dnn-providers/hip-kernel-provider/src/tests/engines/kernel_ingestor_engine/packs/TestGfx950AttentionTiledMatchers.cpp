// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
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

#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"

/**
 * @file TestGfx950AttentionTiledMatchers.cpp
 * @brief Applicability negatives for hipkernel:Gfx950AttentionTiled (RUNBOOK 8b).
 *
 * One case per "must decline" row of the rejection checklist in `mining_tiled.md`, in
 * severity order: the silent-wrong-answer rows first, then the declined features. Each
 * is a graph the kernel would accept and compute something plausible-but-wrong for if
 * the matcher did not stop it -- which is exactly why they are C++ tests and not
 * bundles. A bundle for a graph this engine declines is simply served by another engine
 * and proves nothing.
 *
 * Matcher-only: no device, no compile, no launch.
 *
 * FOUR FAMILIES ARE TILED-SPECIFIC and INVERT the dense sibling's rules. Copying that
 * file's expectations across would produce tests that pass while asserting the wrong
 * thing:
 *
 *  - **PAGED AND VARLEN ARE REQUIRED, NOT DECLINED.** The dense engine declines every
 *    graph carrying `page_table_*` or `seq_len_*`. This kernel's ABI declares
 *    `block_tables_ptr` and `seq_lens_ptr` unconditionally and has no dense mode, so a
 *    graph WITHOUT them is the decline. Both directions are tested.
 *  - **SEQUENCE LENGTHS ARE NOT COMPARED.** `UnifiedAttention2DTiledSpec` has no
 *    total_q/max_seqlen_*/batch field, so a variant serves any sequence length. A test
 *    asserting seqlen equality -- which the dense matcher requires -- would here assert
 *    a bug.
 *  - **`block_size` IS DERIVED FROM `K.dims[SEQ_AXIS]`**, because hipDNN has no
 *    page-size scalar. That derivation is the highest-risk arithmetic in the
 *    integration, so it gets both an accept and a decline case plus a legality sweep.
 *  - **SINKS ARE SERVED.** The dense engine declines them for want of a reference;
 *    decision D1 makes rocKE's `ref_paged_attn` the oracle here and it implements
 *    sinks directly.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::DeviceProperties;
using hipdnn_plugin_sdk::ingestor::MatchContext;

constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.gfx950_attention_tiled.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.gfx950_attention_tiled.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx950_attention_tiled.score";

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;
constexpr int64_t PAGE_TABLE_K_UID = 5;
constexpr int64_t PAGE_TABLE_V_UID = 6;
constexpr int64_t SEQ_LEN_Q_UID = 7;
constexpr int64_t SEQ_LEN_KV_UID = 8;
constexpr int64_t SINK_UID = 9;
/// Named by an attribute under test but never inserted into the tensor map; every
/// optional-feature negative only needs the UID to be PRESENT to be declined.
constexpr int64_t EXTRA_UID = 99;

/// The shape every case starts from: a graph a shipped variant serves. bf16, D128,
/// Hq=16/Hkv=2 (the corpus's dominant GQA pair), page size 16, 2 sequences.
constexpr int64_t NUM_QUERY_HEADS = 16;
constexpr int64_t NUM_KV_HEADS = 2;
constexpr int64_t HEAD_SIZE = 128;
constexpr int64_t BLOCK_SIZE = 16;
constexpr int64_t NUM_SEQS = 2;
constexpr int64_t NUM_BLOCKS = 512;
constexpr int64_t MAX_BLOCKS_PER_SEQ = 64;
constexpr int64_t TOTAL_Q = 256;
constexpr float SCALE = 0.08838834764831843F;

/// A fixed gfx950 device, constructed BY VALUE. Never queried from the host: an
/// arch-gated matcher test that calls hipGetDeviceProperties() is vacuous everywhere
/// except whatever arch CI happens to run, and would silently stop testing the one
/// thing it exists to test the moment that hardware changes.
DeviceProperties testDeviceProperties()
{
    DeviceProperties properties;
    properties.gcnArchName = "gfx950";
    properties.warpSize = 64;
    return properties;
}

/// BSHD strides for Q/O's (B, H, S, D) logical dims -- token-major, head fastest.
std::vector<int64_t> bshdStrides(int64_t heads, int64_t sequence, int64_t headSize)
{
    return {sequence * heads * headSize, headSize, heads * headSize, 1};
}

/// BHSD strides -- the layout every shipped SdpaFwd bundle uses, and the one this
/// kernel must decline for Q.
std::vector<int64_t> bhsdStrides(int64_t heads, int64_t sequence, int64_t headSize)
{
    return {heads * sequence * headSize, sequence * headSize, headSize, 1};
}

/// Paged-container strides for K/V's `[num_blocks, page_size, num_kv_heads, head_size]`
/// -- proven from the kernel's own byte strides (attention_tiled_2d.py:1884-1887).
/// A DIFFERENT pattern from bshdStrides, which is why K/V get their own predicate.
std::vector<int64_t> pagedKvStrides(int64_t pageSize, int64_t kvHeads, int64_t headSize)
{
    return {pageSize * kvHeads * headSize, kvHeads * headSize, headSize, 1};
}

/// Everything a case may vary. Defaults describe the servable graph above; each test
/// perturbs exactly ONE field, so a decline is attributable to that field alone.
struct GraphSpec
{
    int64_t numQueryHeads = NUM_QUERY_HEADS;
    int64_t numKvHeads = NUM_KV_HEADS;
    int64_t headSize = HEAD_SIZE;
    int64_t headSizeV = HEAD_SIZE;
    int64_t blockSize = BLOCK_SIZE;
    int64_t numBlocks = NUM_BLOCKS;
    int64_t numSeqs = NUM_SEQS;
    int64_t maxBlocksPerSeq = MAX_BLOCKS_PER_SEQ;
    int64_t totalQ = TOTAL_Q;
    data_objects::DataType dataType = data_objects::DataType::BFLOAT16;
    std::optional<data_objects::DataType> vDataType;
    /// Q in the head-major layout the kernel does not bake.
    bool qBhsd = false;
    /// K/V strided as if they were dense BSHD rather than the paged container.
    bool kvDenseStrides = false;
    bool omitStrides = false;
    /// A page table whose rows are padded, so its row stride is not its inner extent.
    bool paddedPageTable = false;

    // Paged and varlen are REQUIRED here. These toggles exist to test their ABSENCE.
    bool withPageTables = true;
    bool withSeqLens = true;
    /// Give V a differently shaped page table than K's.
    bool mismatchedPageTables = false;

    // Mask, in the modern spelling. Defaults to top-left causal.
    std::optional<int64_t> leftBound = -1;
    std::optional<int64_t> rightBound = 0;
    data_objects::DiagonalAlignment alignment = data_objects::DiagonalAlignment::TOP_LEFT;
    bool causalMaskDeprecated = false;
    bool causalMaskBottomRightDeprecated = false;

    std::optional<float> attnScaleValue = SCALE;
    std::optional<int32_t> maxSeqLenKv;

    // Optional features.
    bool withSink = false;
    std::optional<int64_t> attnMaskUid;
    std::optional<int64_t> scaleTensorUid;
    std::optional<int64_t> blockMaskUid;
    std::optional<int64_t> statsUid;
    std::optional<int64_t> descaleQUid;
    std::optional<float> dropoutProbability;
    std::optional<bool> generateStats;
    bool alibiMask = false;
    bool paddingMask = false;
    data_objects::DataType mmaCoreMode = data_objects::DataType::UNSET;
    data_objects::AttentionImplementation implementation
        = data_objects::AttentionImplementation::AUTO;

    /// Two nodes rather than one; no single-graph kernel serves it.
    bool twoNodes = false;
};

flatbuffers::FlatBufferBuilder buildSdpaGraph(const GraphSpec& spec)
{
    flatbuffers::FlatBufferBuilder builder;

    // Q is [1, Hq, total_q, D] -- a varlen batch presents its packed query rows with
    // the batch axis collapsed, which is why total_q is B*S rather than a spec field.
    const std::vector<int64_t> qDims{1, spec.numQueryHeads, spec.totalQ, spec.headSize};
    // K/V are the PAGED CONTAINER: [num_blocks, page_size, num_kv_heads, head_size]
    // carried on the same four logical axes.
    const std::vector<int64_t> kDims{
        spec.numBlocks, spec.numKvHeads, spec.blockSize, spec.headSize};
    const std::vector<int64_t> vDims{
        spec.numBlocks, spec.numKvHeads, spec.blockSize, spec.headSizeV};
    const std::vector<int64_t> oDims{1, spec.numQueryHeads, spec.totalQ, spec.headSizeV};

    const auto qStrides = spec.qBhsd ? bhsdStrides(spec.numQueryHeads, spec.totalQ, spec.headSize)
                                     :
bshdStrides(spec.numQueryHeads, spec.totalQ, spec.headSize);
const auto kStrides = spec.kvDenseStrides
                          ? bshdStrides(spec.numKvHeads, spec.blockSize, spec.headSize)
                          : pagedKvStrides(spec.blockSize, spec.numKvHeads, spec.headSize);
const auto vStrides = spec.kvDenseStrides
                          ? bshdStrides(spec.numKvHeads, spec.blockSize, spec.headSizeV)
                          : pagedKvStrides(spec.blockSize, spec.numKvHeads, spec.headSizeV);
const auto oStrides = bshdStrides(spec.numQueryHeads, spec.totalQ, spec.headSizeV);

// Null, not empty: the field is omitted entirely, so strides() returns nullptr.
// Applicability runs before anything has validated a caller-supplied graph.
const std::vector<int64_t>* const qStridesPtr = spec.omitStrides ? nullptr : &qStrides;
const std::vector<int64_t>* const kStridesPtr = spec.omitStrides ? nullptr : &kStrides;
const std::vector<int64_t>* const vStridesPtr = spec.omitStrides ? nullptr : &vStrides;
const std::vector<int64_t>* const oStridesPtr = spec.omitStrides ? nullptr : &oStrides;

std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
tensors.push_back(data_objects::CreateTensorAttributesDirect(
    builder, Q_UID, nullptr, spec.dataType, qStridesPtr, &qDims, false));
tensors.push_back(data_objects::CreateTensorAttributesDirect(
    builder, K_UID, nullptr, spec.dataType, kStridesPtr, &kDims, false));
tensors.push_back(data_objects::CreateTensorAttributesDirect(
    builder, V_UID, nullptr, spec.vDataType.value_or(spec.dataType), vStridesPtr, &vDims, false));
tensors.push_back(data_objects::CreateTensorAttributesDirect(
    builder, O_UID, nullptr, spec.dataType, oStridesPtr, &oDims, false));

// The page tables: [num_seqs, max_blocks_per_seq], dense i32 rows. The row stride
// IS the inner extent unless the case deliberately pads it.
const std::vector<int64_t> ptDims{spec.numSeqs, spec.maxBlocksPerSeq};
const std::vector<int64_t> ptStrides{
    spec.paddedPageTable ? spec.maxBlocksPerSeq + 8 : spec.maxBlocksPerSeq, 1};
const std::vector<int64_t> ptVDims{
    spec.numSeqs, spec.mismatchedPageTables ? spec.maxBlocksPerSeq / 2 : spec.maxBlocksPerSeq};
const std::vector<int64_t> ptVStrides{
    spec.mismatchedPageTables ? spec.maxBlocksPerSeq / 2 : spec.maxBlocksPerSeq, 1};
if(spec.withPageTables)
{
    tensors.push_back(data_objects::CreateTensorAttributesDirect(builder,
                                                                 PAGE_TABLE_K_UID,
                                                                 nullptr,
                                                                 data_objects::DataType::INT32,
                                                                 &ptStrides,
                                                                 &ptDims,
                                                                 false));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(builder,
                                                                 PAGE_TABLE_V_UID,
                                                                 nullptr,
                                                                 data_objects::DataType::INT32,
                                                                 &ptVStrides,
                                                                 &ptVDims,
                                                                 false));
}

// The varlen length vectors: one i32 per sequence.
const std::vector<int64_t> lenDims{spec.numSeqs};
const std::vector<int64_t> lenStrides{1};
if(spec.withSeqLens)
{
    tensors.push_back(data_objects::CreateTensorAttributesDirect(builder,
                                                                 SEQ_LEN_Q_UID,
                                                                 nullptr,
                                                                 data_objects::DataType::INT32,
                                                                 &lenStrides,
                                                                 &lenDims,
                                                                 false));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(builder,
                                                                 SEQ_LEN_KV_UID,
                                                                 nullptr,
                                                                 data_objects::DataType::INT32,
                                                                 &lenStrides,
                                                                 &lenDims,
                                                                 false));
}

// Sinks: one f32 per query head.
const std::vector<int64_t> sinkDims{spec.numQueryHeads};
const std::vector<int64_t> sinkStrides{1};
if(spec.withSink)
{
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, SINK_UID, nullptr, data_objects::DataType::FLOAT, &sinkStrides, &sinkDims, false));
}

const auto attributesFor = [&]() {
    data_objects::SdpaAttributesBuilder attributesBuilder(builder);
    attributesBuilder.add_q_tensor_uid(Q_UID);
    attributesBuilder.add_k_tensor_uid(K_UID);
    attributesBuilder.add_v_tensor_uid(V_UID);
    attributesBuilder.add_o_tensor_uid(O_UID);

    if(spec.withPageTables)
    {
        attributesBuilder.add_page_table_k_tensor_uid(PAGE_TABLE_K_UID);
        attributesBuilder.add_page_table_v_tensor_uid(PAGE_TABLE_V_UID);
    }
    if(spec.withSeqLens)
    {
        attributesBuilder.add_seq_len_q_tensor_uid(SEQ_LEN_Q_UID);
        attributesBuilder.add_seq_len_kv_tensor_uid(SEQ_LEN_KV_UID);
    }
    if(spec.withSink)
    {
        attributesBuilder.add_sink_token_tensor_uid(SINK_UID);
    }

    if(spec.leftBound.has_value())
    {
        attributesBuilder.add_left_bound(*spec.leftBound);
    }
    if(spec.rightBound.has_value())
    {
        attributesBuilder.add_right_bound(*spec.rightBound);
    }
    attributesBuilder.add_diagonal_alignment(spec.alignment);
    attributesBuilder.add_causal_mask(spec.causalMaskDeprecated);
    attributesBuilder.add_causal_mask_bottom_right(spec.causalMaskBottomRightDeprecated);
    if(spec.attnScaleValue.has_value())
    {
        attributesBuilder.add_attn_scale_value(*spec.attnScaleValue);
    }
    if(spec.maxSeqLenKv.has_value())
    {
        attributesBuilder.add_max_seq_len_kv(*spec.maxSeqLenKv);
    }

    if(spec.attnMaskUid.has_value())
    {
        attributesBuilder.add_attn_mask_tensor_uid(*spec.attnMaskUid);
    }
    if(spec.scaleTensorUid.has_value())
    {
        attributesBuilder.add_scale_tensor_uid(*spec.scaleTensorUid);
    }
    if(spec.blockMaskUid.has_value())
    {
        attributesBuilder.add_block_mask_tensor_uid(*spec.blockMaskUid);
    }
    if(spec.statsUid.has_value())
    {
        attributesBuilder.add_stats_tensor_uid(*spec.statsUid);
    }
    if(spec.descaleQUid.has_value())
    {
        attributesBuilder.add_descale_q_tensor_uid(*spec.descaleQUid);
    }
    if(spec.dropoutProbability.has_value())
    {
        attributesBuilder.add_dropout_probability(*spec.dropoutProbability);
    }
    if(spec.generateStats.has_value())
    {
        attributesBuilder.add_generate_stats(*spec.generateStats);
    }
    attributesBuilder.add_alibi_mask(spec.alibiMask);
    attributesBuilder.add_padding_mask(spec.paddingMask);
    attributesBuilder.add_mma_core_mode(spec.mmaCoreMode);
    attributesBuilder.add_implementation(spec.implementation);
    return attributesBuilder.Finish();
};

std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
nodes.push_back(data_objects::CreateNodeDirect(builder,
                                               "sdpa",
                                               data_objects::DataType::FLOAT,
                                               data_objects::NodeAttributes::SdpaAttributes,
                                               attributesFor().Union()));
if(spec.twoNodes)
{
    nodes.push_back(data_objects::CreateNodeDirect(builder,
                                                   "sdpa2",
                                                   data_objects::DataType::FLOAT,
                                                   data_objects::NodeAttributes::SdpaAttributes,
                                                   attributesFor().Union()));
}

auto name = builder.CreateString("gfx950_attention_tiled_test");
auto tensorsVector = builder.CreateVector(tensors);
auto nodesVector = builder.CreateVector(nodes);

data_objects::GraphBuilder graphBuilder(builder);
graphBuilder.add_name(name);
graphBuilder.add_tensors(tensorsVector);
graphBuilder.add_nodes(nodesVector);
builder.Finish(graphBuilder.Finish());
return builder;
}

/// Resolving the matcher by the symbol name its DESCRIPTORS carry, rather than calling
/// the C++ function directly, is what makes a descriptor naming a symbol nothing
/// implements fail here rather than at load time on a device.
std::optional<BoundTokens> matchGraph(const GraphSpec& spec)
{
    registerNativeIngestorSymbols();
    const auto matcher = hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
        std::string(GRAPH_MATCHER_SYMBOL));

    auto builder = buildSdpaGraph(spec);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};
    return matcher(context);
}

/// A candidate whose baked metadata describes the default GraphSpec above.
///
/// NOTE WHAT IS ABSENT: no seqlen_q, no seqlen_kv, no batch. The tiled spec has no
/// such fields, so a variant cannot be specialized on them and the matcher must not
/// compare them. This struct's shape IS that claim.
struct KernelSpec
{
    std::string dtype = "BF16";
    int64_t headSize = HEAD_SIZE;
    int64_t blockSize = BLOCK_SIZE;
    int64_t numQueryHeads = NUM_QUERY_HEADS;
    int64_t numKvHeads = NUM_KV_HEADS;
    int64_t numSeqs = NUM_SEQS;
    int64_t slidingWindow = 0;
    int64_t numWarps = 4;
    int64_t blockMPerWarp = 16;
    int64_t tileSize = BLOCK_SIZE * 2;
    int64_t wavesPerEu = 2;
    int64_t useSinks = 0;
    int64_t useAlibi = 0;
    int64_t hasSoftcap = 0;
    int64_t useQqBias = 0;
    int64_t useFp8MfmaQk = 0;
    int64_t useFp8MfmaPv = 0;
    int64_t useRegisterPv = 0;
    /// Emit the capability fields at all. A descriptor predating them omits them
    /// entirely, and the matcher must read that silence as "not built with it" rather
    /// than throwing -- getIntMetadata THROWS on a missing key, so this distinction is
    /// the difference between a routine non-match and an exception escaping the
    /// matcher.
    bool emitCapabilityFields = true;
};

hipdnn_plugin_sdk::ingestor::KernelDefinition makeKernel(const KernelSpec& spec)
{
    hipdnn_plugin_sdk::ingestor::KernelDefinition kernel;
    kernel.kernelId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-0000000071ed");
    kernel.packId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-0000000071ee");
    kernel.dispatchId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-0000000071ef");
    kernel.metadata = {
        {std::string("dtype"), spec.dtype},
        {std::string("head_size"), spec.headSize},
        {std::string("block_size"), spec.blockSize},
        {std::string("num_query_heads"), spec.numQueryHeads},
        {std::string("num_kv_heads"), spec.numKvHeads},
        {std::string("num_seqs"), spec.numSeqs},
        {std::string("sliding_window"), spec.slidingWindow},
        {std::string("num_warps"), spec.numWarps},
        {std::string("block_m_per_warp"), spec.blockMPerWarp},
        {std::string("tile_size"), spec.tileSize},
        {std::string("waves_per_eu"), spec.wavesPerEu},
    };
    if(spec.emitCapabilityFields)
    {
        kernel.metadata[std::string("use_sinks")] = spec.useSinks;
        kernel.metadata[std::string("use_alibi")] = spec.useAlibi;
        kernel.metadata[std::string("has_softcap")] = spec.hasSoftcap;
        kernel.metadata[std::string("use_qq_bias")] = spec.useQqBias;
        kernel.metadata[std::string("use_fp8_mfma_qk")] = spec.useFp8MfmaQk;
        kernel.metadata[std::string("use_fp8_mfma_pv")] = spec.useFp8MfmaPv;
        kernel.metadata[std::string("use_register_pv")] = spec.useRegisterPv;
    }
    return kernel;
}

/// Runs graph_match then kernel_match, as the runtime does: kernel_match reads the
/// tokens graph_match bound, so calling it standalone would test a different function.
bool matchKernel(const GraphSpec& graphSpec, const KernelSpec& kernelSpec)
{
    registerNativeIngestorSymbols();
    const auto graphMatcher = hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
        std::string(GRAPH_MATCHER_SYMBOL));
    const auto kernelMatcher = hipdnn_plugin_sdk::ingestor::KernelMatchRegistry::resolve(
        std::string(KERNEL_MATCHER_SYMBOL));

    auto builder = buildSdpaGraph(graphSpec);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    const auto bound = graphMatcher(context);
    if(!bound.has_value())
    {
        return false;
    }
    return kernelMatcher(context, *bound, makeKernel(kernelSpec));
}

// ===========================================================================
// The accept case. Without this every decline below passes for a matcher stuck
// on `false`, which is the failure mode that reaches a device days later.
// ===========================================================================

TEST(Gfx950AttentionTiledMatchers, AcceptsAPagedVarlenGraphThisEngineServes)
{
    EXPECT_TRUE(matchGraph(GraphSpec{}).has_value());
}

TEST(Gfx950AttentionTiledMatchers, AcceptsTheDefaultGraphAgainstItsOwnVariant)
{
    EXPECT_TRUE(matchKernel(GraphSpec{}, KernelSpec{}));
}

// ===========================================================================
// Tier 1 -- silent wrong answers. Each of these graphs would produce plausible
// but incorrect numbers if the matcher waved it through.
// ===========================================================================

/// THE HIGHEST-RISK ROW IN THE INTEGRATION. `block_size` is derived from
/// K.dims[SEQ_AXIS] because hipDNN has no page-size scalar; a variant baked for a
/// different page size indexes the KV cache with the wrong stride.
TEST(Gfx950AttentionTiledMatchers, DeclinesAVariantBakedForADifferentPageSize)
{
    GraphSpec graph;
    graph.blockSize = 32;
    KernelSpec kernel;
    kernel.blockSize = 16; // the graph's container is 32-wide
    kernel.tileSize = 32;
    EXPECT_FALSE(matchKernel(graph, kernel));
}

TEST(Gfx950AttentionTiledMatchers, AcceptsEachLegalPageSize)
{
    for(const int64_t pageSize : {16, 32, 64})
    {
        GraphSpec graph;
        graph.blockSize = pageSize;
        EXPECT_TRUE(matchGraph(graph).has_value())
            << "page size " << pageSize << " is in supports_tiled_2d's own set";
    }
}

TEST(Gfx950AttentionTiledMatchers, DeclinesAPageSizeOutsideTheLegalSet)
{
    // 8 and 128 are the neighbours of the legal set. Neither may be rounded to a legal
    // value -- a clamp here is a wrong stride, not a smaller tile.
    for(const int64_t pageSize : {8, 128})
    {
        GraphSpec graph;
        graph.blockSize = pageSize;
        EXPECT_FALSE(matchGraph(graph).has_value()) << "page size " << pageSize;
    }
}

/// Q is token-major with no stride kernargs, so a head-major Q is read as if it were
/// token-major: in-bounds reads of the wrong elements, no fault.
TEST(Gfx950AttentionTiledMatchers, DeclinesAHeadMajorQuery)
{
    GraphSpec graph;
    graph.qBhsd = true;
    EXPECT_FALSE(matchGraph(graph).has_value());
}

/// K/V are the paged container, a DIFFERENT stride pattern from Q's. A K strided as if
/// it were dense BSHD walks the cache wrongly.
TEST(Gfx950AttentionTiledMatchers, DeclinesAKvCacheWithNonContainerStrides)
{
    GraphSpec graph;
    graph.kvDenseStrides = true;
    EXPECT_FALSE(matchGraph(graph).has_value());
}

/// The kernel has ONE block_tables_ptr serving both caches, so two differently shaped
/// tables cannot both be represented.
TEST(Gfx950AttentionTiledMatchers, DeclinesMismatchedPageTables)
{
    GraphSpec graph;
    graph.mismatchedPageTables = true;
    EXPECT_FALSE(matchGraph(graph).has_value());
}

/// The block table is indexed `[seq_idx * bt_stride + tile_idx]` with bt_stride in
/// ELEMENTS. A padded table's row stride is not its inner extent, and we cannot see
/// that padding from the graph -- so it is declined rather than guessed.
TEST(Gfx950AttentionTiledMatchers, DeclinesAPaddedPageTable)
{
    GraphSpec graph;
    graph.paddedPageTable = true;
    EXPECT_FALSE(matchGraph(graph).has_value());
}

/// The window WIDTH is a compile-time KV-loop bound. A windowed graph served by a
/// sliding_window=0 binary attends the whole causal triangle -- a wrong answer, not a
/// decline.
TEST(Gfx950AttentionTiledMatchers, DeclinesAWindowedGraphAgainstAnUnwindowedVariant)
{
    GraphSpec graph;
    graph.leftBound = 127; // a 128-token band
    graph.rightBound = 0;
    KernelSpec kernel;
    kernel.slidingWindow = 0;
    EXPECT_FALSE(matchKernel(graph, kernel));
}

/// And the converse: the width must MATCH, not merely be present.
TEST(Gfx950AttentionTiledMatchers, DeclinesAWindowWidthMismatch)
{
    GraphSpec graph;
    graph.leftBound = 127;
    graph.rightBound = 0;
    KernelSpec kernel;
    kernel.slidingWindow = 256; // graph asked for 128
    EXPECT_FALSE(matchKernel(graph, kernel));
}

/// hipDNN's left bound counts tokens strictly before the current one; the spec's
/// sliding_window counts the band including it. The +1 is a real convention
/// difference, and getting it wrong is an off-by-one in the mask.
TEST(Gfx950AttentionTiledMatchers, DerivesWindowWidthAsLeftBoundPlusOne)
{
    GraphSpec graph;
    graph.leftBound = 127;
    graph.rightBound = 0;
    KernelSpec kernel;
    kernel.slidingWindow = 128;
    EXPECT_TRUE(matchKernel(graph, kernel));
}

/// BOTH mask spellings occur in this repo: the shipped bundles use the modern bounds,
/// the model traces set the deprecated boolean. A matcher reading only one passes its
/// own suite and mis-serves the other population.
TEST(Gfx950AttentionTiledMatchers, AcceptsTheDeprecatedCausalBooleanSpelling)
{
    GraphSpec graph;
    graph.leftBound = std::nullopt;
    graph.rightBound = std::nullopt;
    graph.causalMaskDeprecated = true;
    EXPECT_TRUE(matchGraph(graph).has_value());
}

/// A graph setting a boolean AND a real bound is asking for a WINDOW; reporting it as
/// plain causal silently discards the band.
TEST(Gfx950AttentionTiledMatchers, TreatsABoundedLeftEdgeAsAWindowDespiteTheBoolean)
{
    GraphSpec graph;
    graph.causalMaskDeprecated = true;
    graph.leftBound = 63;
    graph.rightBound = 0;
    KernelSpec kernel;
    kernel.slidingWindow = 64; // 63 + 1, not "causal"
    EXPECT_TRUE(matchKernel(graph, kernel));
}

TEST(Gfx950AttentionTiledMatchers, DeclinesBothDeprecatedCausalBooleansAtOnce)
{
    GraphSpec graph;
    graph.causalMaskDeprecated = true;
    graph.causalMaskBottomRightDeprecated = true;
    EXPECT_FALSE(matchGraph(graph).has_value());
}

/// GQA by integer division: a non-divisible pair silently drops the remainder heads.
TEST(Gfx950AttentionTiledMatchers, DeclinesNonDivisibleHeadCounts)
{
    GraphSpec graph;
    graph.numQueryHeads = 15;
    graph.numKvHeads = 2;
    EXPECT_FALSE(matchGraph(graph).has_value());
}

/// supports_tiled_2d bounds the GQA ratio at 16 (attention_tiled_2d.py:952).
TEST(Gfx950AttentionTiledMatchers, DeclinesAGqaRatioAboveSixteen)
{
    GraphSpec graph;
    graph.numQueryHeads = 32;
    graph.numKvHeads = 1; // ratio 32
    EXPECT_FALSE(matchGraph(graph).has_value());
}

/// A sink graph needs a sink binary (the softmax denominator differs), and a sink
/// binary must not serve a plain graph. Both directions.
TEST(Gfx950AttentionTiledMatchers, RequiresSinkFlagToAgreeInBothDirections)
{
    GraphSpec sinkGraph;
    sinkGraph.withSink = true;
    KernelSpec plainKernel;
    KernelSpec sinkKernel;
    sinkKernel.useSinks = 1;

    EXPECT_FALSE(matchKernel(sinkGraph, plainKernel)) << "sink graph on a plain binary";
    EXPECT_FALSE(matchKernel(GraphSpec{}, sinkKernel)) << "plain graph on a sink binary";
    EXPECT_TRUE(matchKernel(sinkGraph, sinkKernel)) << "sinks ARE served by this engine";
}

// ===========================================================================
// Tier 2 -- structurally required inputs. The INVERSE of the dense sibling's
// rules, and the family most likely to be broken by copying that file.
// ===========================================================================

TEST(Gfx950AttentionTiledMatchers, DeclinesAGraphWithNoPageTables)
{
    GraphSpec graph;
    graph.withPageTables = false;
    EXPECT_FALSE(matchGraph(graph).has_value())
        << "the tiled ABI declares block_tables_ptr unconditionally; there is no dense mode";
}

TEST(Gfx950AttentionTiledMatchers, DeclinesAGraphWithNoSequenceLengths)
{
    GraphSpec graph;
    graph.withSeqLens = false;
    EXPECT_FALSE(matchGraph(graph).has_value())
        << "the block->sequence mapping is a binary search over cu_q; there is no uniform path";
}

/// The tiled spec bakes NO sequence length, so one binary serves every length. This is
/// the opposite of the dense matcher's strict equality, and it is why 48 servable
/// corpus shapes resolved to 39 distinct binaries.
TEST(Gfx950AttentionTiledMatchers, ServesDifferentSequenceLengthsFromOneVariant)
{
    for(const int64_t totalQ : {64, 256, 4096})
    {
        GraphSpec graph;
        graph.totalQ = totalQ;
        EXPECT_TRUE(matchKernel(graph, KernelSpec{}))
            << "total_q " << totalQ << " -- the spec has no sequence-length field to specialize on";
    }
}

/// num_seqs IS baked (it drives the binary-search trip count) but as a CAPACITY BOUND:
/// more iterations than needed is harmless, fewer resolves the wrong sequence.
TEST(Gfx950AttentionTiledMatchers, TreatsNumSeqsAsACapacityBound)
{
    GraphSpec smaller;
    smaller.numSeqs = 2;
    KernelSpec roomy;
    roomy.numSeqs = 8;
    EXPECT_TRUE(matchKernel(smaller, roomy)) << "a binary sized for 8 sequences serves 2";

    GraphSpec larger;
    larger.numSeqs = 16;
    KernelSpec tight;
    tight.numSeqs = 8;
    EXPECT_FALSE(matchKernel(larger, tight))
        << "16 sequences would exhaust an 8-sequence binary search and resolve the wrong sequence";
}

// ===========================================================================
// Tier 3 -- declined features. Each maps to a named row of mining_tiled.md.
// ===========================================================================

TEST(Gfx950AttentionTiledMatchers, DeclinesUnsupportedDataTypes)
{
    for(const auto dataType : {data_objects::DataType::FLOAT, data_objects::DataType::INT8})
    {
        GraphSpec graph;
        graph.dataType = dataType;
        EXPECT_FALSE(matchGraph(graph).has_value());
    }
}

TEST(Gfx950AttentionTiledMatchers, DeclinesAnUnsupportedHeadSize)
{
    // 192 is legal in hipDNN and shipped as a quick-tier bundle; the tiled kernel
    // admits only {64,128,256}.
    GraphSpec graph;
    graph.headSize = 192;
    graph.headSizeV = 192;
    EXPECT_FALSE(matchGraph(graph).has_value());
}

TEST(Gfx950AttentionTiledMatchers, DeclinesMismatchedQkAndVHeadSizes)
{
    // hipDNN permits D_qk != D_v; the kernel has ONE head_size.
    GraphSpec graph;
    graph.headSizeV = 64;
    EXPECT_FALSE(matchGraph(graph).has_value());
}

TEST(Gfx950AttentionTiledMatchers, DeclinesAdditiveAttentionBias)
{
    GraphSpec graph;
    graph.attnMaskUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(graph).has_value());
}

TEST(Gfx950AttentionTiledMatchers, DeclinesADeviceResidentScaleTensor)
{
    GraphSpec graph;
    graph.scaleTensorUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(graph).has_value());
}

TEST(Gfx950AttentionTiledMatchers, DeclinesAGraphWithNoScale)
{
    // The scale is a required kernarg with no default. Inventing 1/sqrt(D) would
    // silently override whatever the frontend's omission meant.
    GraphSpec graph;
    graph.attnScaleValue = std::nullopt;
    EXPECT_FALSE(matchGraph(graph).has_value());
}

TEST(Gfx950AttentionTiledMatchers, DeclinesDropoutInEverySpelling)
{
    GraphSpec probability;
    probability.dropoutProbability = 0.1F;
    EXPECT_FALSE(matchGraph(probability).has_value());
}

TEST(Gfx950AttentionTiledMatchers, DeclinesBlockSparseAttention)
{
    GraphSpec graph;
    graph.blockMaskUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(graph).has_value());
}

TEST(Gfx950AttentionTiledMatchers, DeclinesFp8Descaling)
{
    GraphSpec graph;
    graph.descaleQUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(graph).has_value());
}

TEST(Gfx950AttentionTiledMatchers, DeclinesSoftmaxStatsOutputs)
{
    GraphSpec statsTensor;
    statsTensor.statsUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(statsTensor).has_value());

    GraphSpec generateStats;
    generateStats.generateStats = true;
    EXPECT_FALSE(matchGraph(generateStats).has_value());
}

TEST(Gfx950AttentionTiledMatchers, AcceptsAnExplicitlyFalseGenerateStats)
{
    // optional<bool>: an explicit `false` is inert, and declining it would decline
    // graphs that merely spell out the default.
    GraphSpec graph;
    graph.generateStats = false;
    EXPECT_TRUE(matchGraph(graph).has_value());
}

TEST(Gfx950AttentionTiledMatchers, DeclinesAlibiAndPaddingMasks)
{
    GraphSpec alibi;
    alibi.alibiMask = true;
    EXPECT_FALSE(matchGraph(alibi).has_value())
        << "alibi_mask is a bare bool with no slopes UID anywhere in the schema";

    GraphSpec padding;
    padding.paddingMask = true;
    EXPECT_FALSE(matchGraph(padding).has_value());
}

/// `!= UNSET` would be an over-rejection: every shipped SdpaFwd bundle sets "float".
TEST(Gfx950AttentionTiledMatchers, AcceptsFloatMmaCoreModeAndDeclinesOthers)
{
    GraphSpec floatMode;
    floatMode.mmaCoreMode = data_objects::DataType::FLOAT;
    EXPECT_TRUE(matchGraph(floatMode).has_value())
        << "FLOAT is what the builder emits unconditionally, so it is inert";

    GraphSpec halfMode;
    halfMode.mmaCoreMode = data_objects::DataType::HALF;
    EXPECT_FALSE(matchGraph(halfMode).has_value());
}

/// `implementation` is the field the dense pack still leaves UNCHECKED. UNIFIED is
/// exactly what this kernel is; COMPOSITE asks for a decomposition it is not.
TEST(Gfx950AttentionTiledMatchers, HonoursTheImplementationHint)
{
    GraphSpec unified;
    unified.implementation = data_objects::AttentionImplementation::UNIFIED;
    EXPECT_TRUE(matchGraph(unified).has_value());

    GraphSpec composite;
    composite.implementation = data_objects::AttentionImplementation::COMPOSITE;
    EXPECT_FALSE(matchGraph(composite).has_value());
}

/// max_seq_len_kv is the graph's own statement of its paged bound; where present it
/// must agree with the geometry derived from the tensors.
TEST(Gfx950AttentionTiledMatchers, CrossChecksMaxSeqLenKvAgainstTheDerivedGeometry)
{
    GraphSpec agreeing;
    agreeing.maxSeqLenKv = static_cast<int32_t>(MAX_BLOCKS_PER_SEQ * BLOCK_SIZE);
    EXPECT_TRUE(matchGraph(agreeing).has_value());

    GraphSpec exceeding;
    exceeding.maxSeqLenKv = static_cast<int32_t>(MAX_BLOCKS_PER_SEQ * BLOCK_SIZE * 2);
    EXPECT_FALSE(matchGraph(exceeding).has_value())
        << "a declared bound the page table cannot hold means one of the two readings is wrong";
}

TEST(Gfx950AttentionTiledMatchers, DeclinesMultiNodeGraphs)
{
    GraphSpec graph;
    graph.twoNodes = true;
    EXPECT_FALSE(matchGraph(graph).has_value());
}

TEST(Gfx950AttentionTiledMatchers, DeclinesAGraphWithNoStrides)
{
    // Applicability runs on an UNVALIDATED graph: a caller can omit strides entirely,
    // and indexing a null strides vector is a crash rather than a decline.
    GraphSpec graph;
    graph.omitStrides = true;
    EXPECT_FALSE(matchGraph(graph).has_value());
}

// ===========================================================================
// kernel_match: variants that must never be selected.
// ===========================================================================

/// No graph can request softcap or a query-query bias -- neither has a schema field --
/// so such a variant is unreachable by construction. This comparison keeps it from
/// being reached by ACCIDENT: matching a plain graph and computing a softcapped result
/// nothing asked for.
TEST(Gfx950AttentionTiledMatchers, DeclinesVariantsBakedForUnrequestableCapabilities)
{
    KernelSpec softcap;
    softcap.hasSoftcap = 1;
    EXPECT_FALSE(matchKernel(GraphSpec{}, softcap));

    KernelSpec qqBias;
    qqBias.useQqBias = 1;
    EXPECT_FALSE(matchKernel(GraphSpec{}, qqBias));

    KernelSpec alibi;
    alibi.useAlibi = 1;
    EXPECT_FALSE(matchKernel(GraphSpec{}, alibi));
}

TEST(Gfx950AttentionTiledMatchers, DeclinesFp8BakedVariants)
{
    for(auto* field : {&KernelSpec::useFp8MfmaQk, &KernelSpec::useFp8MfmaPv})
    {
        KernelSpec kernel;
        kernel.*field = 1;
        EXPECT_FALSE(matchKernel(GraphSpec{}, kernel));
    }
    KernelSpec registerPv;
    registerPv.useRegisterPv = 1;
    EXPECT_FALSE(matchKernel(GraphSpec{}, registerPv));
}

/// A descriptor predating these fields omits them entirely. getIntMetadata THROWS on a
/// missing key, so reading that silence wrongly turns a routine non-match into an
/// exception escaping the matcher.
TEST(Gfx950AttentionTiledMatchers, ReadsAbsentCapabilityFieldsAsNotBuiltWithIt)
{
    KernelSpec legacy;
    legacy.emitCapabilityFields = false;
    EXPECT_TRUE(matchKernel(GraphSpec{}, legacy));
}

/// A descriptor naming a geometry the builder would not emit describes no compiled
/// artifact, so it cannot be a real candidate.
TEST(Gfx950AttentionTiledMatchers, DeclinesGeometriesTheBuilderRefusesToEmit)
{
    KernelSpec badWarps;
    badWarps.numWarps = 3; // not in {1,2,4,8}
    EXPECT_FALSE(matchKernel(GraphSpec{}, badWarps));

    KernelSpec badBlockM;
    badBlockM.blockMPerWarp = 64; // not in {16,32}
    EXPECT_FALSE(matchKernel(GraphSpec{}, badBlockM));

    // The 1024-thread CTA cap: 8 warps x 32 rows exceeds it.
    KernelSpec capViolation;
    capViolation.numWarps = 8;
    capViolation.blockMPerWarp = 32;
    EXPECT_FALSE(matchKernel(GraphSpec{}, capViolation));

    // tile_size must be a positive multiple of block_size.
    KernelSpec badTile;
    badTile.tileSize = BLOCK_SIZE + 1;
    EXPECT_FALSE(matchKernel(GraphSpec{}, badTile));
}

// ===========================================================================
// score
// ===========================================================================

/// A score returning a CONSTANT makes ranking arbitrary and hides
/// mis-specialization -- and nothing else catches it until the 8e.3 performance pass.
TEST(Gfx950AttentionTiledMatchers, ScoreVariesWithTheRankedField)
{
    registerNativeIngestorSymbols();
    const auto scorer
        = hipdnn_plugin_sdk::ingestor::ScoreRegistry::resolve(std::string(SCORE_SYMBOL));

    auto builder = buildSdpaGraph(GraphSpec{});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};
    const BoundTokens bound;

    KernelSpec narrow;
    narrow.tileSize = BLOCK_SIZE;
    KernelSpec wide;
    wide.tileSize = BLOCK_SIZE * 4;

    const double narrowScore = scorer(context, bound, makeKernel(narrow));
    const double wideScore = scorer(context, bound, makeKernel(wide));
    EXPECT_NE(narrowScore, wideScore);
    EXPECT_GT(wideScore, narrowScore)
        << "a wider KV tile amortises the block-table indirection over more tokens";
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
