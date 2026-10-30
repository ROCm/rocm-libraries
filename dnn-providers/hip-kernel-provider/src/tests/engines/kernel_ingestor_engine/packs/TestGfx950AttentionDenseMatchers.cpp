// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/Catalog.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelHeuristic.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/AttentionDenseTestArch.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/AttentionDenseTestGraphs.hpp"

/**
 * @file TestGfx950AttentionDenseMatchers.cpp
 * @brief Applicability negatives for hipkernel:Gfx950AttentionDense.
 *
 * One case per applicability rule the matcher must decline, in severity order:
 * silent-wrong-answer cases first, then faults, then declined features.
 *
 * These are matcher-only: no device, no compile, no launch.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

using namespace attention_dense_graphs;
using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;

constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx950_attention_dense.score";
/// The mask type graph_match derives and kernel_match compares, as the engine names it.
constexpr std::string_view CAUSAL_TOKEN = "gfx950_attention_dense.causal";

/// The bound token prepare() reads the softmax scale from, as an IEEE-754 bit pattern.
constexpr std::string_view SCALE_BITS_TOKEN = "gfx950_attention_dense.scale_bits";

/// graph_match over @p spec. This suite's base graph is GraphSpec's default: bf16, D128,
/// B=2, Hq=Hkv=4, Sq=Skv=256, top-left causal.
std::optional<BoundTokens> matchGraph(const GraphSpec& spec)
{
    return attention_dense_graphs::matchGraph(GFX950_ATTENTION_DENSE, spec);
}

/// KernelSpec spells the fields the engine's KMD declares, as a completed record carries
/// them, so a spec that leaves the tile alone is a legacy record completed to 256/64.
struct KernelSpec
{
    std::string dtype = "BF16";
    int64_t headSize = HEAD_SIZE;
    int64_t seqLenQ = SEQ;
    int64_t seqLenKv = SEQ;
    int64_t batch = BATCH;
    int64_t causal = 1;
    int64_t slidingWindow = 0;
    int64_t ragged = 0;
    int64_t blockM = 256;
    int64_t blockN = 64;
    /// The descriptor id's final byte. Only the ranking cases vary it.
    unsigned idByte = 0xa1;
};

/// A descriptor id ending in @p lastByte. Distinct bytes give distinct ids, ordered by
/// the byte, so a case can choose which candidate the selector's id tie-break favours.
hipdnn_plugin_sdk::ingestor::DescriptorId idEndingIn(unsigned lastByte)
{
    constexpr const char* HEX = "0123456789abcdef";
    std::string text = "00000000-0000-4000-8000-0000000000";
    text.push_back(HEX[(lastByte >> 4U) & 0xFU]);
    text.push_back(HEX[lastByte & 0xFU]);
    return hipdnn_flatbuffers_sdk::utilities::parseUuid(text);
}

hipdnn_plugin_sdk::ingestor::KernelDefinition makeKernel(const KernelSpec& spec)
{
    hipdnn_plugin_sdk::ingestor::KernelDefinition kernel;
    kernel.kernelId = idEndingIn(spec.idByte);
    kernel.packId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-00000000dea2");
    kernel.dispatchId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-00000000dea3");
    kernel.metadata = {
        {std::string("dtype"), spec.dtype},
        {std::string("head_size"), spec.headSize},
        {std::string("seqlen_q"), spec.seqLenQ},
        {std::string("seqlen_kv"), spec.seqLenKv},
        {std::string("batch"), spec.batch},
        {std::string("causal"), spec.causal},
        {std::string("sliding_window"), spec.slidingWindow},
        {std::string("ragged"), spec.ragged},
        {std::string("block_m"), spec.blockM},
        {std::string("block_n"), spec.blockN},
    };
    return kernel;
}

/// Runs graph_match then kernel_match for @p kernel, exactly as a catalog build does.
bool matchesKernelDefinition(const GraphSpec& graphSpec,
                             const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel)
{
    return attention_dense_graphs::matchesKernel(GFX950_ATTENTION_DENSE, graphSpec, kernel);
}

bool matchesKernel(const GraphSpec& graphSpec, const KernelSpec& kernelSpec)
{
    return matchesKernelDefinition(graphSpec, makeKernel(kernelSpec));
}

/// Runs kernel_match alone on @p graphSpec with @p bound, the binding a well-formed graph
/// produced. kernel_match is reached here without graph_match having accepted this graph,
/// which is the only way to exercise its own guards.
bool kernelMatchesWithBinding(const GraphSpec& graphSpec,
                              const BoundTokens& bound,
                              const KernelSpec& kernelSpec)
{
    const Graph graph(GFX950_ATTENTION_DENSE, graphSpec);
    return hipdnn_plugin_sdk::ingestor::KernelMatcherRegistry::resolve(std::string(
        GFX950_ATTENTION_DENSE.kernelMatchSymbol))(graph.context(), bound, makeKernel(kernelSpec));
}

double scoreOf(const KernelSpec& kernelSpec)
{
    const Graph graph(GFX950_ATTENTION_DENSE, GraphSpec{});
    const auto bound = graph.match();
    EXPECT_TRUE(bound.has_value());
    return hipdnn_plugin_sdk::ingestor::ScoreRegistry::resolve(std::string(SCORE_SYMBOL))(
        graph.context(), bound.value_or(BoundTokens{}), makeKernel(kernelSpec));
}

/// A (block_m, block_n) pair.
using Tile = std::pair<int64_t, int64_t>;
using TileSet = std::set<Tile>;

/// The tiles the catalog authors per aligned cohort at each head size.
const std::vector<Tile>& d64Tiles()
{
    static const std::vector<Tile> s_tiles{
        {128, 32}, {128, 64}, {128, 128}, {256, 32}, {256, 64}, {256, 128}, {256, 256}};
    return s_tiles;
}

const std::vector<Tile>& d128Tiles()
{
    static const std::vector<Tile> s_tiles{
        {128, 32}, {128, 64}, {128, 128}, {256, 32}, {256, 64}, {256, 128}};
    return s_tiles;
}

TileSet tileSetOf(const std::vector<Tile>& tiles)
{
    return {tiles.begin(), tiles.end()};
}

KernelSpec withTile(KernelSpec spec, int64_t blockM, int64_t blockN)
{
    spec.blockM = blockM;
    spec.blockN = blockN;
    return spec;
}

/// An aligned record's canonical build inputs: B1, Sq=Skv=512. Not runtime constraints.
KernelSpec canonicalAligned()
{
    KernelSpec spec;
    spec.batch = 1;
    spec.seqLenQ = 512;
    spec.seqLenKv = 512;
    spec.ragged = 0;
    return spec;
}

/// One candidate per tile, sharing @p semantic's other fields, with distinct ids.
std::vector<KernelSpec> cohortOf(const KernelSpec& semantic, const std::vector<Tile>& tiles)
{
    std::vector<KernelSpec> cohort;
    cohort.reserve(tiles.size());
    unsigned idByte = 0x10;
    for(const auto& [blockM, blockN] : tiles)
    {
        auto candidate = withTile(semantic, blockM, blockN);
        candidate.idByte = idByte++;
        cohort.push_back(candidate);
    }
    return cohort;
}

/// BF16/D128/H9/9 noncausal: a cross-attention cohort, used for the long-KV cases.
KernelSpec canonicalD128H9()
{
    KernelSpec spec = canonicalAligned();
    spec.headSize = 128;
    spec.causal = 0;
    return spec;
}

std::vector<KernelSpec> d128H9Cohort()
{
    return cohortOf(canonicalD128H9(), d128Tiles());
}

GraphSpec d128H9Noncausal(int64_t batch, int64_t seqLenQ, int64_t seqLenKv)
{
    GraphSpec graph;
    graph.batch = batch;
    graph.numQueryHeads = 9;
    graph.numKvHeads = 9;
    graph.seqLenQ = seqLenQ;
    graph.seqLenKv = seqLenKv;
    graph.headSize = 128;
    graph.leftBound = std::nullopt;
    graph.rightBound = std::nullopt;
    return graph;
}

/// BF16/D64/H32/32 noncausal: the cohort that carries the D64-only 256/256 tile.
std::vector<KernelSpec> d64H32Cohort()
{
    KernelSpec spec = canonicalAligned();
    spec.headSize = 64;
    spec.causal = 0;
    return cohortOf(spec, d64Tiles());
}

GraphSpec d64H32Noncausal(int64_t seqLenQ, int64_t seqLenKv)
{
    GraphSpec graph;
    graph.batch = 1;
    graph.numQueryHeads = 32;
    graph.numKvHeads = 32;
    graph.seqLenQ = seqLenQ;
    graph.seqLenKv = seqLenKv;
    graph.headSize = 64;
    graph.leftBound = std::nullopt;
    graph.rightBound = std::nullopt;
    return graph;
}

/// BF16/D64/H64/8 top-left causal: the semantic cohort of the B1/S2016 ragged record,
/// whose length 2016 is a multiple of 32 but of no block_m.
GraphSpec d64H64Kv8Causal(int64_t batch, int64_t seqLenQ, int64_t seqLenKv)
{
    GraphSpec graph;
    graph.batch = batch;
    graph.numQueryHeads = 64;
    graph.numKvHeads = 8;
    graph.seqLenQ = seqLenQ;
    graph.seqLenKv = seqLenKv;
    graph.headSize = 64;
    return graph;
}

/// A ragged record as the exact-shape builds author it: B1, Sq=Skv=2016.
KernelSpec raggedRecord2016()
{
    KernelSpec spec;
    spec.headSize = 64;
    spec.causal = 1;
    spec.ragged = 1;
    spec.batch = 1;
    spec.seqLenQ = 2016;
    spec.seqLenKv = 2016;
    spec.idByte = 0xe0;
    return spec;
}

/// The aligned cohort sharing d64H64Kv8Causal's semantic fields.
std::vector<KernelSpec> d64H64Kv8Cohort()
{
    KernelSpec spec = canonicalAligned();
    spec.headSize = 64;
    spec.causal = 1;
    return cohortOf(spec, d64Tiles());
}

/// FP16/D64/H12/12 noncausal at B16, Sq=Skv=197 -- a ViT-B/16 shape no tile divides.
GraphSpec vitGraph197()
{
    GraphSpec graph;
    graph.batch = 16;
    graph.numQueryHeads = 12;
    graph.numKvHeads = 12;
    graph.seqLenQ = 197;
    graph.seqLenKv = 197;
    graph.headSize = 64;
    graph.dataType = data_objects::DataType::HALF;
    graph.leftBound = std::nullopt;
    graph.rightBound = std::nullopt;
    return graph;
}

/// The semantic fields vitGraph197 asks for, as a canonical aligned record.
KernelSpec vitSemantic()
{
    KernelSpec spec = canonicalAligned();
    spec.dtype = "FP16";
    spec.headSize = 64;
    spec.causal = 0;
    return spec;
}

/// Runs graph_match over @p graphSpec, then kernel_match over @p candidates, and returns
/// the survivors' tiles -- ranked through the engine's score symbol and the SDK's
/// NativeKernelHeuristic when @p rank, in authoring order otherwise.
std::vector<Tile> matchCandidates(const GraphSpec& graphSpec,
                                  const std::vector<KernelSpec>& candidates,
                                  bool rank)
{
    const Graph graph(GFX950_ATTENTION_DENSE, graphSpec);
    const auto kernelMatcher = hipdnn_plugin_sdk::ingestor::KernelMatcherRegistry::resolve(
        std::string(GFX950_ATTENTION_DENSE.kernelMatchSymbol));
    const auto context = graph.context();

    const auto bound = graph.match();
    EXPECT_TRUE(bound.has_value()) << "graph_match declined the graph before kernel_match ran";
    if(!bound.has_value())
    {
        return {};
    }

    // The catalog references its entries, so the admitted kernels are owned here and
    // pointed at only once the vector stops growing.
    std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition> admitted;
    for(const auto& spec : candidates)
    {
        auto kernel = makeKernel(spec);
        if(kernelMatcher(context, *bound, kernel))
        {
            admitted.push_back(std::move(kernel));
        }
    }

    hipdnn_plugin_sdk::ingestor::Catalog catalog;
    catalog.bound = *bound;
    for(const auto& kernel : admitted)
    {
        catalog.entries.push_back(&kernel);
    }

    if(rank)
    {
        const hipdnn_plugin_sdk::ingestor::NativeKernelHeuristic heuristic{
            std::string(SCORE_SYMBOL)};
        catalog.entries = heuristic.rank(catalog, context);
    }

    std::vector<Tile> tiles;
    tiles.reserve(catalog.entries.size());
    for(const auto* entry : catalog.entries)
    {
        tiles.emplace_back(entry->getIntMetadata("block_m"), entry->getIntMetadata("block_n"));
    }
    return tiles;
}

TileSet admittedTiles(const GraphSpec& graph, const std::vector<KernelSpec>& candidates)
{
    return tileSetOf(matchCandidates(graph, candidates, /*rank=*/false));
}

/// The order a cold plan build tries the admitted candidates in.
std::vector<Tile> coldOrder(const GraphSpec& graph, const std::vector<KernelSpec>& candidates)
{
    return matchCandidates(graph, candidates, /*rank=*/true);
}

// ---------------------------------------------------------------------------
// Positive controls. Every negative below is only meaningful because these pass.
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseGraphMatch, AcceptsDenseBshdCausalGraph)
{
    EXPECT_TRUE(matchGraph(GraphSpec{}).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, AcceptsNoMaskGraph)
{
    GraphSpec spec;
    spec.leftBound = std::nullopt;
    spec.rightBound = std::nullopt;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, AcceptsGroupedQueryAttention)
{
    GraphSpec spec;
    spec.numQueryHeads = 8;
    spec.numKvHeads = 2;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, AcceptsHeadSize64)
{
    GraphSpec spec;
    spec.headSize = 64;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

// ---------------------------------------------------------------------------
// Silent-wrong-answer cases first
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesBhsdLayout)
{
    // The kernel bakes BSHD strides and takes no stride kernargs; BHSD is wrong elements.
    // One operand at a time runs on both arches in TestAttentionDenseMatchers.cpp.
    GraphSpec spec;
    spec.setEveryLayout(SdpaLayout::BHSD);
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, AcceptsSingleHeadUnderEitherStrideSpelling)
{
    // A single-head tensor is byte-identically BSHD and BHSD; strict compare would
    // decline a graph the kernel serves perfectly and empty the whole catalog.
    GraphSpec bshd;
    bshd.numQueryHeads = 1;
    bshd.numKvHeads = 1;
    EXPECT_TRUE(matchGraph(bshd).has_value());

    GraphSpec bhsd = bshd;
    bhsd.setEveryLayout(SdpaLayout::BHSD);
    EXPECT_TRUE(matchGraph(bhsd).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesPaddedSequenceStride)
{
    // Far likelier in a real graph than a full transpose: BSHD axis ORDER, but token rows
    // padded past heads * headSize. The kernel bakes heads * headSize as the row stride,
    // so every row after the first is read off by the accumulated padding.
    //
    // batch = 1 makes the batch axis unit-extent and therefore exempt from the stride
    // compare, which leaves the sequence-stride clause as the only thing that can catch
    // this. Heads and sequence both stay above 1 so neither of those is exempted away.
    GraphSpec packed;
    packed.batch = 1;
    EXPECT_TRUE(matchGraph(packed).has_value());

    GraphSpec padded = packed;
    padded.operand(Q_UID).layout = SdpaLayout::PADDED_BSHD;
    EXPECT_FALSE(matchGraph(padded).has_value());
}

// ---------------------------------------------------------------------------
// Decline logging. hipDNN tells the caller only "No engine configurations available for
// the graph", so the engine's INFO line is the one place the cause shows. Each line
// carries a cause key in brackets; these tests check the key and the operand named, not
// the wording around them.
// ---------------------------------------------------------------------------

LoggedDecline loggedDecline(const GraphSpec& spec)
{
    return attention_dense_graphs::loggedDecline(GFX950_ATTENTION_DENSE, spec);
}

TEST(TestGfx950AttentionDenseGraphMatch, LogsTheOperandAndTheBakedStridesWhenQIsNotBshd)
{
    GraphSpec spec;
    spec.operand(Q_UID).layout = SdpaLayout::BHSD;
    const auto decline = loggedDecline(spec);
    EXPECT_EQ(decline.cause, "layout");
    EXPECT_NE(decline.detail.find("Q (uid 1)"), std::string::npos) << decline.detail;
    // The BSHD strides for dims [2, 4, 256, 128].
    EXPECT_NE(decline.detail.find("131072, 128, 512, 1"), std::string::npos) << decline.detail;
}

TEST(TestGfx950AttentionDenseGraphMatch, LogsTheOutputWhenOnlyOIsNotBshd)
{
    GraphSpec spec;
    spec.operand(O_UID).layout = SdpaLayout::BHSD;
    const auto decline = loggedDecline(spec);
    EXPECT_EQ(decline.cause, "layout");
    EXPECT_NE(decline.detail.find("O (uid 4)"), std::string::npos) << decline.detail;
}

TEST(TestGfx950AttentionDenseGraphMatch, LogsTheCauseOfEachOtherDecline)
{
    struct Case
    {
        const char* name;
        GraphSpec spec;
        const char* cause;
    };
    std::vector<Case> cases;

    // Two SDPA nodes: an attention graph this engine cannot take whole, so it says why.
    GraphSpec twoNodes;
    twoNodes.twoNodes = true;
    cases.push_back({"two nodes", twoNodes, "node"});

    // Null strides: the operand description must survive them.
    GraphSpec noStrides;
    noStrides.omitStrides = true;
    cases.push_back({"no strides", noStrides, "operand"});

    GraphSpec fp32;
    fp32.dataType = data_objects::DataType::FLOAT;
    cases.push_back({"fp32", fp32, "data_type"});

    GraphSpec d256;
    d256.headSize = 256;
    cases.push_back({"D256", d256, "head_size"});

    GraphSpec window;
    window.leftBound = 128;
    cases.push_back({"sliding window", window, "mask"});

    GraphSpec sinks;
    sinks.sinkTokenUid = EXTRA_UID;
    cases.push_back({"sinks", sinks, "sinks"});

    GraphSpec composite;
    composite.implementation = data_objects::AttentionImplementation::COMPOSITE;
    cases.push_back({"implementation", composite, "implementation"});

    GraphSpec largeScale;
    largeScale.attnScaleValue = 0x1p5F;
    cases.push_back({"scale", largeScale, "scale"});

    for(const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.name);
        const auto decline = loggedDecline(testCase.spec);
        EXPECT_EQ(decline.cause, testCase.cause) << decline.detail;
        EXPECT_FALSE(decline.detail.empty());
    }
}

TEST(TestGfx950AttentionDenseGraphMatch, LogsNoDeclineForAGraphItServes)
{
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    EXPECT_TRUE(matchGraph(GraphSpec{}).has_value());
    EXPECT_FALSE(recorder.hasLogContaining(std::string(GFX950_ATTENTION_DENSE_ENGINE_NAME)
                                           + " declined the graph"));
}

TEST(TestGfx950AttentionDenseGraphMatch, LogsNoDeclineForAGraphWithNoSdpaForwardNode)
{
    // graph_match sees every graph the catalog misses on. One with no SDPA-forward node
    // was never this engine's, so it declines without a line.
    GraphSpec spec;
    spec.pointwiseOnly = true;
    const auto decline = loggedDecline(spec);
    EXPECT_TRUE(decline.cause.empty()) << "[" << decline.cause << "]" << decline.detail;
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesPaddedOutputSequenceStride)
{
    // Exact-equality rule, same as the inputs: a dense-but-padded row stride is not the
    // layout the epilogue bakes. Batch is 1 so the batch axis is unit-extent and exempt,
    // leaving the sequence stride as the only clause that can reject this graph.
    GraphSpec spec;
    spec.batch = 1;
    EXPECT_TRUE(matchGraph(spec).has_value());

    GraphSpec padded = spec;
    padded.operand(O_UID).layout = SdpaLayout::PADDED_BSHD;
    EXPECT_FALSE(matchGraph(padded).has_value());
}

// ---------------------------------------------------------------------------
// One stride axis at a time. hasBshdStrides ANDs four per-axis clauses, and the layouts
// above each break two at once (BHSD: head and sequence; padded: sequence and batch).
// Each case below perturbs exactly one axis of one operand at B=2, H=4, so no axis is
// unit-extent and exempt, and that axis's clause is the only one that can decline it.
// ---------------------------------------------------------------------------

/// Indices into a (B, H, S, D) stride vector.
constexpr std::size_t BATCH_STRIDE = 0;
constexpr std::size_t HEAD_STRIDE = 1;
constexpr std::size_t ELEMENT_STRIDE = 3;

/// The packed BSHD strides of the base shape with the stride at @p axis replaced.
std::vector<int64_t> packedStridesWith(std::size_t axis, int64_t stride)
{
    auto strides = sdpaStrides(SdpaLayout::BSHD, HEADS, SEQ, HEAD_SIZE);
    strides.at(axis) = stride;
    return strides;
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesAPaddedBatchStrideAlone)
{
    // A query sliced out of a larger allocation: rows and heads packed, batches spaced one
    // row further apart. The kernel bakes S * H * D as the batch pitch, so every batch
    // after the first is read one row early.
    const GraphSpec spec;
    EXPECT_TRUE(matchGraph(spec).has_value());

    GraphSpec padded = spec;
    padded.operand(Q_UID).strides
        = packedStridesWith(BATCH_STRIDE, SEQ * HEADS * HEAD_SIZE + HEADS * HEAD_SIZE);
    EXPECT_FALSE(matchGraph(padded).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesAKeyValueCacheViewByItsBatchStride)
{
    // K and V as views of a cache allocated for twice the sequence: each batch's rows are
    // packed, but batches sit a whole cache length apart.
    const GraphSpec spec;
    EXPECT_TRUE(matchGraph(spec).has_value());

    const auto cacheView = packedStridesWith(BATCH_STRIDE, 2 * SEQ * HEADS * HEAD_SIZE);
    GraphSpec keyView = spec;
    keyView.operand(K_UID).strides = cacheView;
    EXPECT_FALSE(matchGraph(keyView).has_value());

    GraphSpec valueView = spec;
    valueView.operand(V_UID).strides = cacheView;
    EXPECT_FALSE(matchGraph(valueView).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesABroadcastBatchStride)
{
    // Batch stride 0: one K broadcast across the batch. The kernel would read a distinct
    // K per batch from memory that holds one.
    const GraphSpec spec;
    EXPECT_TRUE(matchGraph(spec).has_value());

    GraphSpec broadcast = spec;
    broadcast.operand(K_UID).strides = packedStridesWith(BATCH_STRIDE, 0);
    EXPECT_FALSE(matchGraph(broadcast).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesAHeadStrideAlone)
{
    // Rows packed at H * D, heads at a 2 * D pitch: the heads overlap one another.
    const GraphSpec spec;
    EXPECT_TRUE(matchGraph(spec).has_value());

    GraphSpec overlapping = spec;
    overlapping.operand(K_UID).strides = packedStridesWith(HEAD_STRIDE, 2 * HEAD_SIZE);
    EXPECT_FALSE(matchGraph(overlapping).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesABroadcastHeadStride)
{
    // Head stride 0: the MQA-expanded view, one KV head presented as H of them. The
    // kernel would read H distinct heads from memory that holds one.
    const GraphSpec spec;
    EXPECT_TRUE(matchGraph(spec).has_value());

    GraphSpec expanded = spec;
    expanded.operand(K_UID).strides = packedStridesWith(HEAD_STRIDE, 0);
    EXPECT_FALSE(matchGraph(expanded).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesANonUnitElementStrideAlone)
{
    // head_size stride 2 with every other axis packed: legal as a stride vector, and read
    // by the kernel as if contiguous.
    const GraphSpec spec;
    EXPECT_TRUE(matchGraph(spec).has_value());

    GraphSpec strided = spec;
    strided.operand(Q_UID).strides = packedStridesWith(ELEMENT_STRIDE, 2);
    EXPECT_FALSE(matchGraph(strided).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesBsdhLayout)
{
    // Token-major like BSHD but head fastest-varying: same row and batch pitch, with the
    // head and element strides exchanged.
    GraphSpec spec;
    spec.operand(Q_UID).layout = SdpaLayout::BSDH;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesBottomRightCausalWhenSeqLensDiffer)
{
    // Top-left causal clamp != bottom-right when Sq != Skv: serving it is a wrong answer.
    GraphSpec spec;
    spec.seqLenKv = SEQ * 2;
    spec.alignment = data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, AcceptsBottomRightCausalWhenSeqLensMatch)
{
    // The complement: every shipped quick/SdpaFwd causal bundle sets BOTTOM_RIGHT at
    // Sq == Skv; declining it outright declines all of them.
    GraphSpec spec;
    spec.alignment = data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesTheDeprecatedBottomRightBooleanAtUnequalSeqLens)
{
    // causal_mask_bottom_right alone, no bounds: bottom-right causal, which the top-left
    // kernel serves only at Sq == Skv. Read as top-left instead, Sq != Skv would be
    // served with the other corner's mask.
    GraphSpec spec;
    spec.leftBound = std::nullopt;
    spec.rightBound = std::nullopt;
    spec.causalMaskBottomRightDeprecated = true;
    spec.seqLenKv = SEQ * 2;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, ServesTheDeprecatedBottomRightBooleanAtEqualSeqLens)
{
    // The positive neighbour: at Sq == Skv the corners coincide, and the graph is served
    // by the causal candidate and refused by the unmasked one.
    GraphSpec spec;
    spec.leftBound = std::nullopt;
    spec.rightBound = std::nullopt;
    spec.causalMaskBottomRightDeprecated = true;

    KernelSpec causal;
    causal.causal = 1;
    EXPECT_TRUE(matchesKernel(spec, causal));

    KernelSpec unmasked;
    unmasked.causal = 0;
    EXPECT_FALSE(matchesKernel(spec, unmasked));
}

// ---------------------------------------------------------------------------
// Faults and malformed input
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesGraphWithNoStrides)
{
    GraphSpec spec;
    spec.omitStrides = true;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesMultiNodeGraph)
{
    GraphSpec spec;
    spec.twoNodes = true;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesAnOperandWithNoDims)
{
    // dims is optional in the schema, so a verified graph can carry an operand without
    // one. Every extent the matcher reads comes from it.
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        SCOPED_TRACE(uid);
        GraphSpec spec;
        spec.omitDimsUid = uid;
        EXPECT_FALSE(matchGraph(spec).has_value());
    }
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesARaggedOperand)
{
    // A ragged offset tensor packs the sequences back to back; the kernel indexes every
    // operand as dense (B, S, H, D).
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        SCOPED_TRACE(uid);
        GraphSpec spec;
        spec.raggedUid = uid;
        EXPECT_FALSE(matchGraph(spec).has_value());
    }
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesAGraphMissingAnOperandTensor)
{
    // The node names a uid the graph holds no tensor for: there is nothing to read a shape
    // from, and nothing for that operand's pointer argument to address.
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        SCOPED_TRACE(uid);
        GraphSpec spec;
        spec.omitTensorUid = uid;
        EXPECT_FALSE(matchGraph(spec).has_value());
    }
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesANonSdpaNode)
{
    // One node, but not an SDPA one. DeclinesMultiNodeGraph covers the node count; this
    // covers the node's type, without which its attributes would be read as SDPA's.
    GraphSpec spec;
    spec.pointwiseOnly = true;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseKernelMatch, RefusesAGraphWithoutASingleSdpaNode)
{
    // kernel_match re-reads the node instead of trusting graph_match to have vetted it.
    // The binding is the well-formed graph's, so only the node can refuse.
    const auto bound = matchGraph(GraphSpec{});
    ASSERT_TRUE(bound.has_value());
    EXPECT_TRUE(kernelMatchesWithBinding(GraphSpec{}, *bound, KernelSpec{}));

    {
        SCOPED_TRACE("two nodes");
        GraphSpec spec;
        spec.twoNodes = true;
        EXPECT_FALSE(kernelMatchesWithBinding(spec, *bound, KernelSpec{}));
    }
    {
        SCOPED_TRACE("non-SDPA node");
        GraphSpec spec;
        spec.pointwiseOnly = true;
        EXPECT_FALSE(kernelMatchesWithBinding(spec, *bound, KernelSpec{}));
    }
}

TEST(TestGfx950AttentionDenseKernelMatch, RefusesAGraphMissingTheQueryOrKeyTensor)
{
    // kernel_match reads the problem's extents from Q and K itself, so it must find both.
    const auto bound = matchGraph(GraphSpec{});
    ASSERT_TRUE(bound.has_value());
    EXPECT_TRUE(kernelMatchesWithBinding(GraphSpec{}, *bound, KernelSpec{}));

    for(const int64_t uid : {Q_UID, K_UID})
    {
        SCOPED_TRACE(uid);
        GraphSpec spec;
        spec.omitTensorUid = uid;
        EXPECT_FALSE(kernelMatchesWithBinding(spec, *bound, KernelSpec{}));
    }
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesAnOperandOfAnotherRank)
{
    // Every axis index the matcher reads assumes rank 4. A rank-5 K carries all four of
    // them and more, so only the rank check stands between it and a match; rank 3 would
    // have its missing axis read out of bounds.
    const auto packed = sdpaStrides(SdpaLayout::BSHD, HEADS, SEQ, HEAD_SIZE);
    const std::vector<int64_t> rank4Dims{BATCH, HEADS, SEQ, HEAD_SIZE};
    const std::vector<int64_t> rank5Dims{BATCH, HEADS, SEQ, HEAD_SIZE, 1};
    const std::vector<int64_t> rank5Strides{packed.at(0), packed.at(1), packed.at(2), 1, 1};

    GraphSpec control;
    control.operand(K_UID).dims = rank4Dims;
    control.operand(K_UID).strides = packed;
    EXPECT_TRUE(matchGraph(control).has_value());

    GraphSpec bothRank5 = control;
    bothRank5.operand(K_UID).dims = rank5Dims;
    bothRank5.operand(K_UID).strides = rank5Strides;
    EXPECT_FALSE(matchGraph(bothRank5).has_value());

    GraphSpec dimsRank5 = control;
    dimsRank5.operand(K_UID).dims = rank5Dims;
    EXPECT_FALSE(matchGraph(dimsRank5).has_value());

    GraphSpec stridesRank5 = control;
    stridesRank5.operand(K_UID).strides = rank5Strides;
    EXPECT_FALSE(matchGraph(stridesRank5).has_value());

    GraphSpec rank3 = control;
    rank3.operand(K_UID).dims = std::vector<int64_t>{BATCH, HEADS, SEQ};
    rank3.operand(K_UID).strides = std::vector<int64_t>{packed.at(0), packed.at(1), packed.at(2)};
    EXPECT_FALSE(matchGraph(rank3).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesAZeroExtent)
{
    // A zero batch or a zero-length KV sequence. The fixture's strides stay consistent
    // with the zero and no bound is approached, so the extent check is the only one that
    // fails.
    GraphSpec noBatch;
    noBatch.batch = 0;
    EXPECT_FALSE(matchGraph(noBatch).has_value());

    GraphSpec noKeys;
    noKeys.seqLenKv = 0;
    EXPECT_FALSE(matchGraph(noKeys).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesAVirtualOperand)
{
    // A virtual tensor has no device buffer for the launch to hand the kernel.
    GraphSpec spec;
    spec.virtualUid = K_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesAPassByValueOperand)
{
    // A runtime pass-by-value tensor is a scalar, not a buffer the ABI takes a pointer to.
    GraphSpec spec;
    spec.passByValueUid = V_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesUnsupportedHeadSize)
{
    GraphSpec spec;
    spec.headSize = 256;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

// ---------------------------------------------------------------------------
// Cross-operand shape agreement.
//
// The kernel derives ONE problem shape from Q and K and addresses every operand from
// it -- there are no per-tensor extent kernargs. An operand whose dims disagree is
// therefore walked with the wrong bounds: in-bounds wrong elements where the operand
// is larger, an out-of-bounds write where the output is smaller. One axis of one operand
// at a time runs on both arches in TestAttentionDenseMatchers.cpp; the case here has
// extents too large to derive strides from at all.
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesOverflowingOutputExtents)
{
    // O's extents here are positive and rank-4, so the well-formedness predicate passes
    // them through, but S * H * D for those extents is 2^63 -- one past INT64_MAX, and that
    // product is what hasBshdStrides derives O's batch stride from. As ordered, the
    // head-count compare rejects the graph before O's layout is read; the case exists to
    // keep the overflow-capable domain reachable and declined. O's strides are written out
    // as the ordinary dense BSHD spelling of the PROBLEM shape, so no stride value is what
    // rejects this graph.
    const GraphSpec spec;
    EXPECT_TRUE(matchGraph(spec).has_value());

    GraphSpec overflowing = spec;
    overflowing.operand(O_UID).dims
        = std::vector<int64_t>{BATCH, int64_t{1} << 16, int64_t{1} << 31, int64_t{1} << 16};
    overflowing.operand(O_UID).strides = sdpaStrides(SdpaLayout::BSHD, HEADS, SEQ, HEAD_SIZE);
    EXPECT_FALSE(matchGraph(overflowing).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, AbsentAttentionScaleBindsOne)
{
    // cuDNN's default: no attn_scale_value and no scale tensor means no scaling, at every
    // head size.
    for(const int64_t headSize : {int64_t{64}, int64_t{128}})
    {
        SCOPED_TRACE(headSize);
        GraphSpec spec;
        spec.headSize = headSize;
        spec.attnScaleValue = std::nullopt;
        const auto bound = matchGraph(spec);
        ASSERT_TRUE(bound.has_value());
        EXPECT_EQ(
            hipdnn_plugin_sdk::ingestor::tryGetBoundInt(*bound, SCALE_BITS_TOKEN).value_or(-1),
            ieee754Bits(1.0F));
    }

    // An explicit scale still wins over the default.
    GraphSpec explicitScale;
    explicitScale.attnScaleValue = 0.5F;
    const auto bound = matchGraph(explicitScale);
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(*bound, SCALE_BITS_TOKEN).value_or(-1),
              ieee754Bits(0.5F));
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesBothDeprecatedCausalBooleans)
{
    GraphSpec spec;
    spec.causalMaskDeprecated = true;
    spec.causalMaskBottomRightDeprecated = true;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

// ---------------------------------------------------------------------------
// Extents whose products do not fit in int64_t. Every one is declined, and every one
// would be accepted by an unchecked product that wrapped modulo 2^64: the wrapped value
// is 0 or INT64_MIN, which is under any bound and equals the strides written below. The
// fixture forms none of these products either -- strides are written out wherever
// deriving them would overflow. The 32-bit bounds, and K/V and Q/O counts past int64_t,
// run on both arches in TestAttentionDenseMatchers.cpp.
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesAQueryBatchStrideOf2To67)
{
    // Q dims {2, 2^30, 2^30, 128}: S * H * D = 2^67, which does not fit in int64_t, and Q
    // and O carry the batch stride 0 it would wrap to beside packed head, row and element
    // strides. hasBshdStrides forms that product through checkedProduct, which removes the
    // undefined behaviour but is not observable in any verdict: every graph whose S * H * D
    // overflows also overflows B * S * H * D, which the Q bound declines.
    const std::vector<int64_t> wrappedStrides{0, HEAD_SIZE, (int64_t{1} << 30) * HEAD_SIZE, 1};
    GraphSpec spec;
    spec.numQueryHeads = int64_t{1} << 30;
    spec.seqLenQ = int64_t{1} << 30;
    spec.numKvHeads = 1;
    spec.operand(Q_UID).strides = wrappedStrides;
    spec.operand(O_UID).strides = wrappedStrides;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesKeyValueExtentsOf2To97Bytes)
{
    // B = S = H = 2^30 on every operand, D64: K/V is 2^97 bytes and Q is 2^96 elements,
    // and S * H * D = 2^66 would wrap to the batch stride 0 every operand carries. As
    // above, the Q and K/V bounds decline before that product is observable in any verdict.
    constexpr int64_t EXTENT = int64_t{1} << 30;
    const std::vector<int64_t> wrappedStrides{0, 64, EXTENT * 64, 1};
    GraphSpec spec;
    spec.batch = EXTENT;
    spec.numQueryHeads = EXTENT;
    spec.numKvHeads = EXTENT;
    spec.seqLenQ = EXTENT;
    spec.seqLenKv = EXTENT;
    spec.headSize = 64;
    spec.operand(Q_UID).strides = wrappedStrides;
    spec.operand(K_UID).strides = wrappedStrides;
    spec.operand(V_UID).strides = wrappedStrides;
    spec.operand(O_UID).strides = wrappedStrides;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesTheGraphMatchGateCounterexample)
{
    // A graph whose bound products wrap to 0 is declined. Packed BSHD, bf16, H8, D128,
    // unmasked, Q {2^20, 8, 2^34, 128} and K/V {2^20, 8, 2^33, 128}: every stride fits,
    // and both bound products are exactly 2^64, which wraps to 0. Every length is a
    // multiple of every tile, so unchecked the whole catalog would admit it.
    GraphSpec spec;
    spec.batch = int64_t{1} << 20;
    spec.numQueryHeads = 8;
    spec.numKvHeads = 8;
    spec.seqLenQ = int64_t{1} << 34;
    spec.seqLenKv = int64_t{1} << 33;
    spec.leftBound = std::nullopt;
    spec.rightBound = std::nullopt;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

// ---------------------------------------------------------------------------
// Declined optional features
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesAdditiveAttentionMask)
{
    GraphSpec spec;
    spec.attnMaskUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesDeviceResidentScaleTensor)
{
    GraphSpec spec;
    spec.scaleTensorUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesVarlen)
{
    GraphSpec spec;
    spec.seqLenQUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesPagedKv)
{
    GraphSpec spec;
    spec.pageTableKUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesAttentionSinks)
{
    GraphSpec spec;
    spec.sinkTokenUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesBlockSparseMask)
{
    GraphSpec spec;
    spec.blockMaskUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesSoftmaxStatsBothSpellings)
{
    GraphSpec byUid;
    byUid.statsUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(byUid).has_value());

    GraphSpec byFlag;
    byFlag.generateStats = true;
    EXPECT_FALSE(matchGraph(byFlag).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, AcceptsExplicitlyDisabledStats)
{
    // generate_stats is optional<bool>; explicit false is not a request for stats.
    GraphSpec spec;
    spec.generateStats = false;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesDropout)
{
    GraphSpec spec;
    spec.dropoutProbability = 0.1F;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesFp8Descale)
{
    GraphSpec spec;
    spec.descaleQUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesEveryOtherSpellingOfEachDeclinedFeature)
{
    // The cases above decline each feature through one spelling. Every other uid that
    // requests the same feature is set here on its own, so dropping any one of them from
    // its decline is seen.
    using UidField = std::optional<int64_t> GraphSpec::*;
    const std::vector<std::pair<const char*, UidField>> spellings{
        {"seq_len_kv", &GraphSpec::seqLenKvUid},
        {"seed", &GraphSpec::seedUid},
        {"offset", &GraphSpec::offsetUid},
        {"dropout_mask", &GraphSpec::dropoutMaskUid},
        {"dropout_scale", &GraphSpec::dropoutScaleUid},
        {"page_table_v", &GraphSpec::pageTableVUid},
        {"descale_k", &GraphSpec::descaleKUid},
        {"descale_v", &GraphSpec::descaleVUid},
        {"descale_s", &GraphSpec::descaleSUid},
        {"scale_s", &GraphSpec::scaleSUid},
        {"scale_o", &GraphSpec::scaleOUid},
        {"amax_s", &GraphSpec::amaxSUid},
        {"amax_o", &GraphSpec::amaxOUid},
        {"max", &GraphSpec::maxUid},
        {"sum_exp", &GraphSpec::sumExpUid},
        {"rng_dump", &GraphSpec::rngDumpUid},
    };
    for(const auto& [name, field] : spellings)
    {
        SCOPED_TRACE(name);
        GraphSpec spec;
        spec.*field = EXTRA_UID;
        EXPECT_FALSE(matchGraph(spec).has_value());
    }

    // Paged KV's scalar spelling.
    GraphSpec maxSeqLenKv;
    maxSeqLenKv.maxSeqLenKv = static_cast<int32_t>(SEQ);
    EXPECT_FALSE(matchGraph(maxSeqLenKv).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesAlibiMask)
{
    GraphSpec spec;
    spec.alibiMask = true;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesPaddingMask)
{
    GraphSpec spec;
    spec.paddingMask = true;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesNonAutoImplementationHint)
{
    GraphSpec spec;
    spec.implementation = data_objects::AttentionImplementation::COMPOSITE;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, AcceptsTheSixteenBitMmaCoreModes)
{
    // mma_core_mode is the MMA operand precision, and this kernel's operands are the
    // graph's fp16/bf16 inputs. HALF is also what the cuDNN-compat shim sets whenever the
    // caller leaves the field unset, so declining it would decline every shim graph.
    // Each graph is matched against the kernel built for its own dtype.
    for(const auto dataType : {data_objects::DataType::BFLOAT16, data_objects::DataType::HALF})
    {
        for(const auto mode : {data_objects::DataType::UNSET,
                               data_objects::DataType::HALF,
                               data_objects::DataType::BFLOAT16})
        {
            SCOPED_TRACE(std::string("graph ") + data_objects::EnumNameDataType(dataType)
                         + ", mma_core_mode " + data_objects::EnumNameDataType(mode));
            GraphSpec spec;
            spec.dataType = dataType;
            spec.mmaCoreMode = mode;
            KernelSpec kernel;
            kernel.dtype = dataType == data_objects::DataType::HALF ? "FP16" : "BF16";
            EXPECT_TRUE(matchesKernel(spec, kernel));
        }
    }
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesMmaCoreModeFloat)
{
    // An fp32-operand MMA is a computation this kernel never performs.
    for(const auto dataType : {data_objects::DataType::BFLOAT16, data_objects::DataType::HALF})
    {
        SCOPED_TRACE(std::string("graph ") + data_objects::EnumNameDataType(dataType)
                     + ", mma_core_mode FLOAT");
        GraphSpec spec;
        spec.dataType = dataType;
        spec.mmaCoreMode = data_objects::DataType::FLOAT;
        EXPECT_FALSE(matchGraph(spec).has_value());
    }
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesEveryFp8MmaCoreMode)
{
    for(const auto mode : {data_objects::DataType::FP8_E4M3,
                           data_objects::DataType::FP8_E5M2,
                           data_objects::DataType::FP8_E8M0,
                           data_objects::DataType::FP8_E4M3_FNUZ,
                           data_objects::DataType::FP8_E5M2_FNUZ})
    {
        SCOPED_TRACE(data_objects::EnumNameDataType(mode));
        GraphSpec spec;
        spec.mmaCoreMode = mode;
        EXPECT_FALSE(matchGraph(spec).has_value());
    }
}

// ---------------------------------------------------------------------------
// Sliding-window: declined outright.
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesSlidingWindowForSelfAttention)
{
    // Every shipped variant is sliding_window = 0. Serving a windowed graph on a
    // full-length causal binary would attend the whole lower triangle instead of the
    // requested band: wrong numerics, no error.
    GraphSpec spec;
    spec.leftBound = 127;
    spec.rightBound = 0;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesSlidingWindowForCrossAttention)
{
    // Declined for the same reason as the self-attention case above; the unequal
    // sequence lengths are incidental, not the cause.
    GraphSpec spec;
    spec.seqLenKv = SEQ * 2;
    spec.leftBound = 127;
    spec.rightBound = 0;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesWhenDeprecatedBoolIsSetAlongsideBound)
{
    // A real bound wins over the deprecated boolean. causal_mask=true alone is served as
    // plain causal; add left_bound and the graph is asking for a band the catalog cannot
    // supply, so it must decline. If the boolean won instead, the window would be silently
    // widened to the full triangle -- accepted, dispatched, and wrong.
    GraphSpec spec;
    spec.causalMaskDeprecated = true;
    spec.leftBound = 127;
    spec.rightBound = 0;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesDeprecatedCausalWithRightBoundBelowMinusOne)
{
    GraphSpec spec;
    spec.causalMaskDeprecated = true;
    spec.rightBound = -5;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, StillServesPlainDeprecatedCausalWithNoBound)
{
    // The control: bound-wins must not over-fire and decline ordinary deprecated-causal.
    GraphSpec spec;
    spec.causalMaskDeprecated = true;
    spec.leftBound = std::nullopt;
    spec.rightBound = std::nullopt;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

/// Both spellings of one causal mask must bind identically, as causal, so neither reaches a
/// candidate the other would not: served by the causal kernel, refused by the unmasked one.
void expectOneCausalMask(const GraphSpec& bounds, const GraphSpec& flag)
{
    const auto boundsBinding = matchGraph(bounds);
    const auto flagBinding = matchGraph(flag);
    ASSERT_TRUE(boundsBinding.has_value());
    ASSERT_TRUE(flagBinding.has_value());
    EXPECT_EQ(tryGetBoundInt(*boundsBinding, CAUSAL_TOKEN), std::optional<int64_t>{1});
    EXPECT_EQ(tryGetBoundInt(*flagBinding, CAUSAL_TOKEN), std::optional<int64_t>{1});
    EXPECT_TRUE(*boundsBinding == *flagBinding);

    KernelSpec causal;
    causal.causal = 1;
    KernelSpec unmasked;
    unmasked.causal = 0;
    for(const GraphSpec* spec : std::initializer_list<const GraphSpec*>{&bounds, &flag})
    {
        SCOPED_TRACE(spec == &bounds ? "bounds" : "deprecated flag");
        EXPECT_TRUE(matchesKernel(*spec, causal));
        EXPECT_FALSE(matchesKernel(*spec, unmasked));
    }
}

TEST(TestGfx950AttentionDenseGraphMatch, BoundsAndTheDeprecatedFlagSpellTheSameTopLeftCausalMask)
{
    // left -1 / right 0 / TOP_LEFT and causal_mask=true with no bounds are one mask.
    const GraphSpec bounds;
    GraphSpec flag;
    flag.causalMaskDeprecated = true;
    flag.leftBound = std::nullopt;
    flag.rightBound = std::nullopt;
    expectOneCausalMask(bounds, flag);
}

TEST(TestGfx950AttentionDenseGraphMatch,
     BoundsAndTheDeprecatedFlagSpellTheSameBottomRightCausalMask)
{
    // left -1 / right 0 / BOTTOM_RIGHT and causal_mask_bottom_right=true with no bounds are
    // one mask. At the default Sq == Skv the corners coincide, so both are served as causal.
    GraphSpec bounds;
    bounds.alignment = data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    GraphSpec flag;
    flag.causalMaskBottomRightDeprecated = true;
    flag.leftBound = std::nullopt;
    flag.rightBound = std::nullopt;
    expectOneCausalMask(bounds, flag);
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesBidirectionalSlidingWindow)
{
    // A graph with both left_bound and a non-zero right_bound is a bidirectional window.
    // The gfx950 kernel is hard-causal (upper mask only) and has no right-bound field, so
    // serving it would produce silent wrong numerics.
    GraphSpec spec;
    spec.leftBound = 127;
    spec.rightBound = 64;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesCausalWithNonZeroRightBound)
{
    // A graph with causal_mask=true and right_bound > 0 also describes a shape the kernel
    // cannot serve correctly. The right_bound wins over the deprecated causal boolean.
    GraphSpec spec;
    spec.causalMaskDeprecated = true;
    spec.leftBound = std::nullopt;
    spec.rightBound = 64;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, ServesDeprecatedCausalWithCausalBounds)
{
    // The bounds describe the same causal band the boolean does, so the merge agrees.
    GraphSpec spec;
    spec.causalMaskDeprecated = true;

    KernelSpec causal;
    causal.causal = 1;
    EXPECT_TRUE(matchesKernel(spec, causal));
}

TEST(TestGfx950AttentionDenseGraphMatch, ReadsDeprecatedCausalWithBottomRightAlignmentAsBottomRight)
{
    // causal_mask keeps the stored alignment, so this is bottom-right causal: served at
    // Sq == Skv, declined where the corners differ.
    GraphSpec spec;
    spec.causalMaskDeprecated = true;
    spec.leftBound = std::nullopt;
    spec.rightBound = std::nullopt;
    spec.alignment = data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    EXPECT_TRUE(matchGraph(spec).has_value());

    spec.seqLenKv = SEQ * 2;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

// ---------------------------------------------------------------------------
// kernel_match
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseKernelMatch, AcceptsTheCandidateBakedForThisGraph)
{
    EXPECT_TRUE(matchesKernel(GraphSpec{}, KernelSpec{}));
}

TEST(TestGfx950AttentionDenseKernelMatch, RefusesACandidateBakedForAnotherDtype)
{
    KernelSpec kernel;
    kernel.dtype = "FP16";
    EXPECT_FALSE(matchesKernel(GraphSpec{}, kernel));
}

TEST(TestGfx950AttentionDenseKernelMatch, AlignedCandidateAcceptsDifferentBatch)
{
    // Aligned (ragged=0) kernels are shape-generic: batch/seqlen equality is not enforced.
    KernelSpec kernel;
    kernel.batch = BATCH + 1;
    EXPECT_TRUE(matchesKernel(GraphSpec{}, kernel));
}

TEST(TestGfx950AttentionDenseKernelMatch, AlignedCandidateAcceptsDifferentSeqLen)
{
    // Same shape-generic rule: seqlen mismatch is not a rejection for aligned kernels.
    KernelSpec kernel;
    kernel.seqLenKv = SEQ * 2;
    EXPECT_TRUE(matchesKernel(GraphSpec{}, kernel));
}

TEST(TestGfx950AttentionDenseKernelMatch, RefusesACandidateBakedForAnotherHeadSize)
{
    // head_size is baked (the builder's single D), and 256/64 is a legal tile at both
    // head sizes, so only the head-size compare separates the candidates. A D64 binary
    // serving a D128 graph reads half of every row, and the reverse reads past it.
    GraphSpec d64Graph;
    d64Graph.headSize = 64;
    KernelSpec d64Candidate;
    d64Candidate.headSize = 64;
    KernelSpec d128Candidate;
    d128Candidate.headSize = 128;

    EXPECT_TRUE(matchesKernel(d64Graph, d64Candidate));
    EXPECT_FALSE(matchesKernel(d64Graph, d128Candidate));

    const GraphSpec d128Graph;
    EXPECT_TRUE(matchesKernel(d128Graph, d128Candidate));
    EXPECT_FALSE(matchesKernel(d128Graph, d64Candidate));
}

TEST(TestGfx950AttentionDenseKernelMatch, OneCandidateServesEveryHeadCount)
{
    // The metadata carries no head counts: the kernel reads them as kernargs from the
    // graph, so one candidate matches MHA, GQA, MQA and non-power-of-two head counts.
    const KernelSpec kernel;
    for(const auto& [queryHeads, kvHeads] : std::vector<std::pair<int64_t, int64_t>>{
            {1, 1}, {6, 3}, {64, 8}, {9, 9}, {HEADS * 2, HEADS}})
    {
        GraphSpec graph;
        graph.numQueryHeads = queryHeads;
        graph.numKvHeads = kvHeads;
        EXPECT_TRUE(matchesKernel(graph, kernel)) << "Hq=" << queryHeads << " Hkv=" << kvHeads;
    }
}

TEST(TestGfx950AttentionDenseKernelMatch, RefusesACandidateBakedForTheOtherMask)
{
    KernelSpec kernel;
    kernel.causal = 0; // default GraphSpec is causal
    EXPECT_FALSE(matchesKernel(GraphSpec{}, kernel));
}

TEST(TestGfx950AttentionDenseKernelMatch, RefusesARaggedCandidateForAnAlignedGraph)
{
    // A ragged build bakes its shape and pads boundary tiles on-chip. The catalog ships
    // only shape-generic builds, so one is declined even where every length is aligned.
    KernelSpec kernel;
    kernel.ragged = 1;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, kernel));
}

TEST(TestGfx950AttentionDenseKernelMatch, RefusesAnAlignedCandidateForANonMultipleGraph)
{
    // 4000 is a multiple of neither block_m: an aligned build has no boundary handling
    // for the partial final query block.
    GraphSpec graph;
    graph.seqLenQ = 4000;
    graph.seqLenKv = 4000;
    KernelSpec aligned;
    aligned.seqLenQ = 4000;
    aligned.seqLenKv = 4000;
    EXPECT_FALSE(matchesKernel(graph, aligned));
}

TEST(TestGfx950AttentionDenseKernelMatch, RefusesAnAlignedCandidateWhoseTileDoesNotDivideSeqLenKv)
{
    // The candidate's own block_n must divide Skv. 288 is not a multiple of the baseline's
    // 64, so the baseline declines; the BN32 neighbour that does serve it is pinned below.
    GraphSpec graph;
    graph.seqLenKv = 288;
    KernelSpec kernel;
    kernel.seqLenKv = 288;
    EXPECT_FALSE(matchesKernel(graph, kernel));
}

TEST(TestGfx950AttentionDenseKernelMatch, AlignedKernelAcceptsDifferentBatch)
{
    // The shape-generic rule in the other direction: the graph's batch moves, not the KD's.
    GraphSpec graph;
    graph.batch = 4;

    KernelSpec kernel;
    kernel.ragged = 0;
    kernel.batch = 1; // canonical build input, not a runtime constraint

    EXPECT_TRUE(matchesKernel(graph, kernel));
}

TEST(TestGfx950AttentionDenseKernelMatch, RefusesWindowedCandidateForPlainGraph)
{
    // The KV-loop bound is baked at compile time; a windowed variant must not serve
    // a plain graph.
    KernelSpec kernel;
    kernel.slidingWindow = 128;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, kernel));
}

// ---------------------------------------------------------------------------
// Candidate-relative tiles. Each cohort is authored as the catalog authors it --
// canonical B1/Sq512/Skv512 build inputs, one candidate per tile -- and every graph
// below differs from those inputs, so an admitted candidate is runtime-shape reuse.
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseTileMatch, D128CrossAttentionQ384AdmitsExactlyTheBm128Tiles)
{
    // 384 is a multiple of 128 and not of 256; 512 is a multiple of every block_n.
    EXPECT_EQ(admittedTiles(d128H9Noncausal(1, 384, 512), d128H9Cohort()),
              (TileSet{{128, 32}, {128, 64}, {128, 128}}));
}

TEST(TestGfx950AttentionDenseTileMatch, D128CrossAttentionKv192AdmitsBothBlockMsAtBn32And64)
{
    // 192 is a multiple of 32 and 64 but not of 128.
    EXPECT_EQ(admittedTiles(d128H9Noncausal(1, 1024, 192), d128H9Cohort()),
              (TileSet{{128, 32}, {128, 64}, {256, 32}, {256, 64}}));
}

TEST(TestGfx950AttentionDenseTileMatch, D128CrossAttentionKv288And416AdmitOnlyBn32)
{
    // Both are odd multiples of 32: the baseline declines, and only the BN32 tiles serve.
    for(const int64_t seqLenKv : {int64_t{288}, int64_t{416}})
    {
        EXPECT_EQ(admittedTiles(d128H9Noncausal(1, 1024, seqLenKv), d128H9Cohort()),
                  (TileSet{{128, 32}, {256, 32}}))
            << "seqlen_kv " << seqLenKv;
    }
}

TEST(TestGfx950AttentionDenseTileMatch, D128LongKvAdmitsEveryAuthoredTile)
{
    // 62208 = 486 * 128, so every block_n divides it.
    EXPECT_EQ(admittedTiles(d128H9Noncausal(1, 1024, 62208), d128H9Cohort()),
              tileSetOf(d128Tiles()));
}

TEST(TestGfx950AttentionDenseTileMatch, D128GqaCausalRuntimeBatchAdmitsAllSixTiles)
{
    // BF16/D128/H32/8 top-left causal at B3, Sq=Skv=1536: 1536 = 6 * 256 = 12 * 128.
    GraphSpec graph;
    graph.batch = 3;
    graph.numQueryHeads = 32;
    graph.numKvHeads = 8;
    graph.seqLenQ = 1536;
    graph.seqLenKv = 1536;

    KernelSpec semantic = canonicalAligned();
    semantic.causal = 1;

    EXPECT_EQ(admittedTiles(graph, cohortOf(semantic, d128Tiles())), tileSetOf(d128Tiles()));
}

TEST(TestGfx950AttentionDenseTileMatch, D64SelfAttentionAdmitsAllSevenTilesIncludingBn256)
{
    EXPECT_EQ(admittedTiles(d64H32Noncausal(1024, 1024), d64H32Cohort()), tileSetOf(d64Tiles()));
}

TEST(TestGfx950AttentionDenseTileMatch, D64Kv128ExcludesOnlyTheBn256Tile)
{
    EXPECT_EQ(admittedTiles(d64H32Noncausal(1024, 128), d64H32Cohort()),
              (TileSet{{128, 32}, {128, 64}, {128, 128}, {256, 32}, {256, 64}, {256, 128}}));
}

TEST(TestGfx950AttentionDenseTileMatch, RefusesTilesGfx950DoesNotBuildEvenWhereTheyWouldDivide)
{
    // Every length below is a multiple of every block_n, so only the tile rules decide.
    // 128/256 fails block_m % block_n at both head sizes; D128 256/256 passes the Python
    // rules and fails the LDS budget. D64 256/256 is the positive neighbour.
    const GraphSpec d64 = d64H32Noncausal(1024, 1024);
    const GraphSpec d128 = d128H9Noncausal(1, 1024, 1024);

    EXPECT_FALSE(matchesKernel(d64, withTile(d64H32Cohort().front(), 128, 256)));
    EXPECT_FALSE(matchesKernel(d128, withTile(d128H9Cohort().front(), 128, 256)));
    EXPECT_FALSE(matchesKernel(d128, withTile(d128H9Cohort().front(), 256, 256)));
    EXPECT_TRUE(matchesKernel(d64, withTile(d64H32Cohort().front(), 256, 256)));
}

TEST(TestGfx950AttentionDenseTileMatch, BothLengthsAreCheckedAgainstTheCandidateTile)
{
    // One graph per failing length, each beside a tile that the same graph admits.
    // Sq 128 x Skv 512: block_m 256 declines, 128 admits.
    const auto q128 = d128H9Noncausal(1, 128, 512);
    EXPECT_FALSE(matchesKernel(q128, withTile(canonicalD128H9(), 256, 64)));
    EXPECT_TRUE(matchesKernel(q128, withTile(canonicalD128H9(), 128, 64)));
    // Sq 512 x Skv 96: block_n 64 declines, 32 admits.
    const auto kv96 = d128H9Noncausal(1, 512, 96);
    EXPECT_FALSE(matchesKernel(kv96, withTile(canonicalD128H9(), 256, 64)));
    EXPECT_TRUE(matchesKernel(kv96, withTile(canonicalD128H9(), 256, 32)));
}

TEST(TestGfx950AttentionDenseTileMatch, AlternativeTilesKeepTheMaskRules)
{
    // TOP_LEFT_CAUSAL at unequal lengths is served; BOTTOM_RIGHT_CAUSAL at unequal
    // lengths is declined by graph_match for every tile alike, and served at equal ones.
    GraphSpec topLeft;
    topLeft.seqLenQ = 384;
    topLeft.seqLenKv = 512;
    EXPECT_TRUE(matchesKernel(topLeft, withTile(KernelSpec{}, 128, 32)));

    GraphSpec bottomRightUnequal = topLeft;
    bottomRightUnequal.alignment = data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    EXPECT_FALSE(matchGraph(bottomRightUnequal).has_value());

    GraphSpec bottomRightEqual = bottomRightUnequal;
    bottomRightEqual.seqLenKv = 384;
    EXPECT_TRUE(matchesKernel(bottomRightEqual, withTile(KernelSpec{}, 128, 128)));
}

TEST(TestGfx950AttentionDenseTileMatch, AlternativeTilesKeepTheSemanticFieldComparisons)
{
    // A BM128 candidate differing in one semantic field from a graph it would otherwise
    // serve. The unperturbed candidate is the positive neighbour.
    const GraphSpec graph = d128H9Noncausal(1, 384, 512);
    const KernelSpec match = withTile(canonicalD128H9(), 128, 32);
    EXPECT_TRUE(matchesKernel(graph, match));

    KernelSpec otherDtype = match;
    otherDtype.dtype = "FP16";
    EXPECT_FALSE(matchesKernel(graph, otherDtype));

    // Head counts are not metadata: another head configuration still matches.
    GraphSpec otherHeads = graph;
    otherHeads.numQueryHeads = 12;
    otherHeads.numKvHeads = 3;
    EXPECT_TRUE(matchesKernel(otherHeads, match));

    KernelSpec otherMask = match;
    otherMask.causal = 1;
    EXPECT_FALSE(matchesKernel(graph, otherMask));

    // FP16 graph against the FP16 candidate, and D64 against D64: both dtypes and both
    // head sizes are served at the alternative tile.
    GraphSpec fp16Graph = graph;
    fp16Graph.dataType = data_objects::DataType::HALF;
    EXPECT_TRUE(matchesKernel(fp16Graph, otherDtype));

    GraphSpec d64Graph = graph;
    d64Graph.headSize = 64;
    KernelSpec d64Candidate = match;
    d64Candidate.headSize = 64;
    EXPECT_TRUE(matchesKernel(d64Graph, d64Candidate));
}

// ---------------------------------------------------------------------------
// Malformed tile metadata. Each is declined outright -- no default substituted, and no
// division by the value: a zero block_n that reached `Skv % block_n` would fault the
// process rather than fail the expectation.
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseTileMatch, AcceptsARecordCompletedToTheBaselineTile)
{
    // A legacy record's raw form omits the tile; completion supplies 256/64 and the
    // candidate is served. This is the positive neighbour of every case below.
    EXPECT_TRUE(matchesKernel(GraphSpec{}, KernelSpec{}));
}

TEST(TestGfx950AttentionDenseTileMatch, DeclinesARecordWhoseTileWasNeverCompleted)
{
    for(const char* field : {"block_m", "block_n"})
    {
        auto kernel = makeKernel(KernelSpec{});
        kernel.metadata.erase(field);
        EXPECT_FALSE(matchesKernelDefinition(GraphSpec{}, kernel)) << "missing " << field;
    }
}

TEST(TestGfx950AttentionDenseTileMatch, DeclinesZeroNegativeAndUnbuiltTileValues)
{
    // Sq = Skv = 256 is a multiple of every legal tile, so only validation can decline.
    const std::vector<std::pair<int64_t, int64_t>> malformed{
        {0, 64}, {256, 0}, {0, 0}, {-256, 64}, {256, -64}, {64, 64}, {512, 64}, {256, 48}};
    for(const auto& [blockM, blockN] : malformed)
    {
        EXPECT_FALSE(matchesKernel(GraphSpec{}, withTile(KernelSpec{}, blockM, blockN)))
            << blockM << "/" << blockN;
    }
}

TEST(TestGfx950AttentionDenseTileMatch, DeclinesATileFieldOfTheWrongType)
{
    // Each value names 256 or 64 in some other type; none may be read as the integer.
    using hipdnn_plugin_sdk::ingestor::MetadataValue;
    const std::vector<MetadataValue> blockMSpellings{MetadataValue{std::string("256")},
                                                     MetadataValue{256.0},
                                                     MetadataValue{true},
                                                     MetadataValue{std::vector<int64_t>{256}}};
    for(const auto& value : blockMSpellings)
    {
        auto kernel = makeKernel(KernelSpec{});
        kernel.metadata[std::string("block_m")] = value;
        EXPECT_FALSE(matchesKernelDefinition(GraphSpec{}, kernel))
            << "block_m held alternative " << value.index();
    }

    auto kernel = makeKernel(KernelSpec{});
    kernel.metadata[std::string("block_n")] = MetadataValue{std::string("64")};
    EXPECT_FALSE(matchesKernelDefinition(GraphSpec{}, kernel));
}

// ---------------------------------------------------------------------------
// Aligned-only. A ragged record is declined at every shape, and a graph whose lengths
// no tile divides admits nothing; the aligned cohort beside it still serves whatever
// its own tiles divide.
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseTileMatch, DeclinesARaggedCandidateEvenAtItsExactAuthoredShape)
{
    // BF16/D64/H64/8 causal B1 S2016 and FP16/D64/H12/12 noncausal B16 S197: ragged
    // records meeting a graph equal to their metadata on every field.
    EXPECT_FALSE(matchesKernel(d64H64Kv8Causal(1, 2016, 2016), raggedRecord2016()));

    KernelSpec vitRagged = vitSemantic();
    vitRagged.ragged = 1;
    vitRagged.batch = 16;
    vitRagged.seqLenQ = 197;
    vitRagged.seqLenKv = 197;
    EXPECT_FALSE(matchesKernel(vitGraph197(), vitRagged));

    // A ragged record at a length its tile divides is declined too; the same record with
    // ragged=0 is the positive neighbour, so the flag alone is what declines it.
    KernelSpec atMultiple = raggedRecord2016();
    atMultiple.seqLenQ = 2048;
    atMultiple.seqLenKv = 2048;
    EXPECT_FALSE(matchesKernel(d64H64Kv8Causal(1, 2048, 2048), atMultiple));
    atMultiple.ragged = 0;
    EXPECT_TRUE(matchesKernel(d64H64Kv8Causal(1, 2048, 2048), atMultiple));
}

TEST(TestGfx950AttentionDenseTileMatch, DeclinesARaggedFieldThatIsMissingOrMistyped)
{
    // `ragged` is read without a throwing accessor: absent or non-integer declines the
    // candidate rather than faulting the match. Integer 0 is the positive neighbour.
    auto missing = makeKernel(KernelSpec{});
    missing.metadata.erase("ragged");
    EXPECT_FALSE(matchesKernelDefinition(GraphSpec{}, missing));

    auto mistyped = makeKernel(KernelSpec{});
    mistyped.metadata[std::string("ragged")]
        = hipdnn_plugin_sdk::ingestor::MetadataValue{std::string("0")};
    EXPECT_FALSE(matchesKernelDefinition(GraphSpec{}, mistyped));

    EXPECT_TRUE(matchesKernelDefinition(GraphSpec{}, makeKernel(KernelSpec{})));
}

TEST(TestGfx950AttentionDenseTileMatch, NonMultipleGraphsAdmitNoAlignedCandidate)
{
    // 2016 is a multiple of 32 but of no block_m; 197 is a multiple of nothing.
    EXPECT_EQ(admittedTiles(d64H64Kv8Causal(1, 2016, 2016), d64H64Kv8Cohort()), TileSet{});
    EXPECT_EQ(admittedTiles(vitGraph197(), cohortOf(vitSemantic(), d64Tiles())), TileSet{});
}

TEST(TestGfx950AttentionDenseTileMatch, EachLengthIsJudgedAgainstEachCandidateTile)
{
    // The 2016 cohort under one length changed at a time, then both. Each admits exactly
    // the tiles that divide its new lengths -- a batch change alone admits nothing.
    struct Case
    {
        const char* what;
        GraphSpec graph;
        TileSet admitted;
    };
    const std::vector<Case> cases{
        {"batch 2", d64H64Kv8Causal(2, 2016, 2016), TileSet{}},
        // TOP_LEFT_CAUSAL with Sq != Skv is served: 2048 is a multiple of both block_m,
        // 2016 of block_n 32 only.
        {"seqlen_q 2048", d64H64Kv8Causal(1, 2048, 2016), TileSet{{128, 32}, {256, 32}}},
        {"seqlen_kv 2048", d64H64Kv8Causal(1, 2016, 2048), TileSet{}},
        {"both 2048", d64H64Kv8Causal(1, 2048, 2048), tileSetOf(d64Tiles())},
    };
    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.what);
        EXPECT_EQ(admittedTiles(c.graph, d64H64Kv8Cohort()), c.admitted);
    }
}

// ---------------------------------------------------------------------------
// Cold ranking, through the engine's own score symbol and the SDK heuristic that
// consumes it: the order a cold (unbenchmarked) plan build tries candidates in.
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseScore, BaselineFirstThenAscendingTilesWhateverTheIdOrder)
{
    // D64 S1024 admits all seven tiles. Ids are assigned twice -- ascending with the
    // expected order, then descending -- so an order the id tie-break decided would
    // differ between the two runs.
    const std::vector<Tile> expected{
        {256, 64}, {128, 32}, {128, 64}, {128, 128}, {256, 32}, {256, 128}, {256, 256}};
    const GraphSpec graph = d64H32Noncausal(1024, 1024);

    for(const bool descendingIds : {false, true})
    {
        SCOPED_TRACE(descendingIds ? "descending ids" : "ascending ids");
        auto cohort = d64H32Cohort();
        for(std::size_t i = 0; i < cohort.size(); ++i)
        {
            const auto& tile = expected.at(i);
            auto& candidate = cohort.at(i);
            candidate.blockM = tile.first;
            candidate.blockN = tile.second;
            candidate.idByte = static_cast<unsigned>(descendingIds ? std::size_t{0x70} - i
                                                                   : std::size_t{0x10} + i);
        }
        EXPECT_EQ(coldOrder(graph, cohort), expected);
    }
}

TEST(TestGfx950AttentionDenseScore, BestApplicableAlternativeLeadsWhenTheBaselineCannotServe)
{
    // Sq 384 rules out every block_m 256 tile, the baseline included. Ids descend against
    // the authoring order, so the selector's ascending-id tie-break alone would put
    // 128/128 first.
    auto cohort = d128H9Cohort();
    for(std::size_t i = 0; i < cohort.size(); ++i)
    {
        cohort.at(i).idByte = static_cast<unsigned>(std::size_t{0x80} - i);
    }
    EXPECT_EQ(coldOrder(d128H9Noncausal(1, 384, 512), cohort),
              (std::vector<Tile>{{128, 32}, {128, 64}, {128, 128}}));
}

TEST(TestGfx950AttentionDenseScore, ScoresEveryTileDeterministicallyAsAPositiveFiniteWeight)
{
    // The properties the selector relies on beyond the order itself: a scorer that drifts
    // between calls makes plan selection unreproducible; a non-positive weight would read
    // as a refusal, but applicability is kernel_match's job and score only ranks what
    // already matched; and a NaN compares false against everything, so one of them turns
    // the ranking into whatever order the sort happened to visit the candidates in.
    for(const auto& candidate : d64H32Cohort())
    {
        SCOPED_TRACE(std::to_string(candidate.blockM) + "/" + std::to_string(candidate.blockN));
        const double score = scoreOf(candidate);
        EXPECT_EQ(score, scoreOf(candidate));
        EXPECT_GT(score, 0.0);
        EXPECT_TRUE(std::isfinite(score));
    }
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
