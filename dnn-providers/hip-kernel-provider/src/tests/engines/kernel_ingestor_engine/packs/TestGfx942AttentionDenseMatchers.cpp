// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
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
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>

#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"

/**
 * @file TestGfx942AttentionDenseMatchers.cpp
 * @brief Applicability, mask translation and ranking for hipkernel:Gfx942AttentionDense.
 *
 * The cases concentrate on what differs from the gfx950 twin, whose suite pins the
 * shared operand, layout and declined-feature rules:
 *   - head counts are runtime kernel arguments, so a candidate built at one head count
 *     serves every other;
 *   - a causal sliding window of graph left_bound L (L keys, diagonal included) is the
 *     kernel's sliding_window L, and every other band is declined;
 *   - the sliding-window zero-trip guard and tile divisibility are per candidate;
 *   - the cold ranking orders every knob combination, policy arms first.
 *
 * Matcher-only: no device, no compile, no launch.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::DeviceProperties;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::MatchContext;
using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;

constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.gfx942_attention_dense.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.gfx942_attention_dense.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx942_attention_dense.score";
constexpr std::string_view CAUSAL_TOKEN = "gfx942_attention_dense.causal";
constexpr std::string_view SLIDING_WINDOW_TOKEN = "gfx942_attention_dense.sliding_window";

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;
constexpr int64_t EXTRA_UID = 99;

constexpr float SCALE = 0.08838834764831843F;

DeviceProperties testDeviceProperties()
{
    DeviceProperties properties;
    properties.gcnArchName = "gfx942";
    properties.warpSize = 64;
    return properties;
}

/// BSHD strides for (B, H, S, D) LOGICAL dims -- token-major, head varying fastest.
std::vector<int64_t> bshdStrides(int64_t heads, int64_t sequence, int64_t headSize)
{
    return {sequence * heads * headSize, headSize, heads * headSize, 1};
}

std::vector<int64_t> bhsdStrides(int64_t heads, int64_t sequence, int64_t headSize)
{
    return {heads * sequence * headSize, sequence * headSize, headSize, 1};
}

/// Base graph: bf16, D128, B=2, Hq=32, Hkv=8, Sq=Skv=1024, top-left causal.
struct GraphSpec
{
    int64_t batch = 2;
    int64_t numQueryHeads = 32;
    int64_t numKvHeads = 8;
    int64_t seqLenQ = 1024;
    int64_t seqLenKv = 1024;
    int64_t headSize = 128;
    data_objects::DataType dataType = data_objects::DataType::BFLOAT16;
    bool bhsdQuery = false;

    std::optional<int64_t> leftBound = -1;
    std::optional<int64_t> rightBound = 0;
    data_objects::DiagonalAlignment alignment = data_objects::DiagonalAlignment::TOP_LEFT;

    std::optional<float> attnScaleValue = SCALE;
    std::optional<int64_t> sinkTokenUid;
    std::optional<int64_t> statsUid;
    std::optional<int64_t> attnMaskUid;
    std::optional<int64_t> seqLenQUid;
    // The one operand that names a ragged offset tensor (packed sequences).
    std::optional<int64_t> raggedUid;
};

flatbuffers::FlatBufferBuilder buildSdpaGraph(const GraphSpec& spec)
{
    flatbuffers::FlatBufferBuilder builder;

    const std::vector<int64_t> qDims{spec.batch, spec.numQueryHeads, spec.seqLenQ, spec.headSize};
    const std::vector<int64_t> kvDims{spec.batch, spec.numKvHeads, spec.seqLenKv, spec.headSize};
    const auto qStrides = spec.bhsdQuery
                              ? bhsdStrides(spec.numQueryHeads, spec.seqLenQ, spec.headSize)
                              : bshdStrides(spec.numQueryHeads, spec.seqLenQ, spec.headSize);
    const auto oStrides = bshdStrides(spec.numQueryHeads, spec.seqLenQ, spec.headSize);
    const auto kvStrides = bshdStrides(spec.numKvHeads, spec.seqLenKv, spec.headSize);

    const auto operand
        = [&](int64_t uid, const std::vector<int64_t>& strides, const std::vector<int64_t>& dims) {
              return data_objects::CreateTensorAttributesDirect(
                  builder,
                  uid,
                  nullptr,
                  spec.dataType,
                  &strides,
                  &dims,
                  false,
                  data_objects::TensorValue::NONE,
                  0,
                  false,
                  spec.raggedUid == uid ? ::flatbuffers::Optional<int64_t>(EXTRA_UID)
                                        : ::flatbuffers::Optional<int64_t>());
          };
    const std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors{
        operand(Q_UID, qStrides, qDims),
        operand(K_UID, kvStrides, kvDims),
        operand(V_UID, kvStrides, kvDims),
        operand(O_UID, oStrides, qDims),
    };

    data_objects::SdpaAttributesBuilder attributesBuilder(builder);
    attributesBuilder.add_q_tensor_uid(Q_UID);
    attributesBuilder.add_k_tensor_uid(K_UID);
    attributesBuilder.add_v_tensor_uid(V_UID);
    attributesBuilder.add_o_tensor_uid(O_UID);
    if(spec.leftBound.has_value())
    {
        attributesBuilder.add_left_bound(*spec.leftBound);
    }
    if(spec.rightBound.has_value())
    {
        attributesBuilder.add_right_bound(*spec.rightBound);
    }
    attributesBuilder.add_diagonal_alignment(spec.alignment);
    attributesBuilder.add_causal_mask(false);
    attributesBuilder.add_causal_mask_bottom_right(false);
    if(spec.attnScaleValue.has_value())
    {
        attributesBuilder.add_attn_scale_value(*spec.attnScaleValue);
    }
    if(spec.sinkTokenUid.has_value())
    {
        attributesBuilder.add_sink_token_tensor_uid(*spec.sinkTokenUid);
    }
    if(spec.statsUid.has_value())
    {
        attributesBuilder.add_stats_tensor_uid(*spec.statsUid);
    }
    if(spec.attnMaskUid.has_value())
    {
        attributesBuilder.add_attn_mask_tensor_uid(*spec.attnMaskUid);
    }
    if(spec.seqLenQUid.has_value())
    {
        attributesBuilder.add_seq_len_q_tensor_uid(*spec.seqLenQUid);
    }
    attributesBuilder.add_alibi_mask(false);
    attributesBuilder.add_padding_mask(false);
    attributesBuilder.add_mma_core_mode(data_objects::DataType::UNSET);
    attributesBuilder.add_implementation(data_objects::AttentionImplementation::AUTO);
    const auto attributes = attributesBuilder.Finish();

    const std::vector<flatbuffers::Offset<data_objects::Node>> nodes{
        data_objects::CreateNodeDirect(builder,
                                       "sdpa",
                                       data_objects::DataType::FLOAT,
                                       data_objects::NodeAttributes::SdpaAttributes,
                                       attributes.Union())};

    auto name = builder.CreateString("gfx942_attention_dense_test");
    auto tensorsVector = builder.CreateVector(tensors);
    auto nodesVector = builder.CreateVector(nodes);
    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_name(name);
    graphBuilder.add_tensors(tensorsVector);
    graphBuilder.add_nodes(nodesVector);
    builder.Finish(graphBuilder.Finish());
    return builder;
}

/// A graph wrapped with the context the matchers take; holds the buffer alive.
class Graph
{
public:
    explicit Graph(const GraphSpec& spec)
        : _builder(buildSdpaGraph(spec))
        , _graph(_builder.GetBufferPointer(), _builder.GetSize())
        , _properties(testDeviceProperties())
        , _context{_graph, 0, _properties}
    {
        registerNativeIngestorSymbols();
    }

    const MatchContext& context() const
    {
        return _context;
    }

private:
    flatbuffers::FlatBufferBuilder _builder;
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper _graph;
    DeviceProperties _properties;
    MatchContext _context;
};

std::optional<BoundTokens> matchGraph(const GraphSpec& spec)
{
    const Graph graph(spec);
    return hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
        std::string(GRAPH_MATCHER_SYMBOL))(graph.context());
}

/// The fields this engine's KMD declares, as a completed record carries them. The head
/// counts and shape are the canonical build inputs every catalog record carries.
struct KernelSpec
{
    std::string dtype = "BF16";
    int64_t headSize = 128;
    int64_t causal = 1;
    int64_t slidingWindow = 0;
    int64_t ragged = 0;
    int64_t blockM = 256;
    int64_t blockN = 64;
    int64_t ldsRowPad = 8;
    int64_t pvPriority = 0;
};

KernelDefinition makeKernel(const KernelSpec& spec)
{
    KernelDefinition kernel;
    kernel.kernelId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-00000000942a");
    kernel.metadata = {
        {std::string("dtype"), spec.dtype},
        {std::string("head_size"), spec.headSize},
        {std::string("num_query_heads"), int64_t{8}},
        {std::string("num_kv_heads"), int64_t{8}},
        {std::string("seqlen_q"), int64_t{512}},
        {std::string("seqlen_kv"), int64_t{512}},
        {std::string("batch"), int64_t{1}},
        {std::string("causal"), spec.causal},
        {std::string("sliding_window"), spec.slidingWindow},
        {std::string("ragged"), spec.ragged},
        {std::string("block_m"), spec.blockM},
        {std::string("block_n"), spec.blockN},
        {std::string("lds_row_pad"), spec.ldsRowPad},
        {std::string("pv_priority"), spec.pvPriority},
    };
    return kernel;
}

/// graph_match then kernel_match, as a catalog build runs them.
bool matches(const GraphSpec& graphSpec, const KernelSpec& kernelSpec)
{
    const Graph graph(graphSpec);
    const auto bound = hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
        std::string(GRAPH_MATCHER_SYMBOL))(graph.context());
    EXPECT_TRUE(bound.has_value()) << "graph_match declined the graph before kernel_match ran";
    if(!bound.has_value())
    {
        return false;
    }
    return hipdnn_plugin_sdk::ingestor::KernelMatcherRegistry::resolve(
        std::string(KERNEL_MATCHER_SYMBOL))(graph.context(), *bound, makeKernel(kernelSpec));
}

double scoreOf(const KernelSpec& kernelSpec)
{
    const Graph graph(GraphSpec{});
    const auto bound = hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
        std::string(GRAPH_MATCHER_SYMBOL))(graph.context());
    EXPECT_TRUE(bound.has_value());
    return hipdnn_plugin_sdk::ingestor::ScoreRegistry::resolve(std::string(SCORE_SYMBOL))(
        graph.context(), bound.value_or(BoundTokens{}), makeKernel(kernelSpec));
}

GraphSpec windowGraph(int64_t leftBound)
{
    GraphSpec spec;
    spec.leftBound = leftBound;
    spec.rightBound = 0;
    return spec;
}

} // namespace

// =============================================================================
// graph_match: what the engine serves, and the mask it binds
// =============================================================================

TEST(TestGfx942AttentionDenseGraphMatch, BindsTopLeftCausalAsCausalWithNoWindow)
{
    const auto bound = matchGraph(GraphSpec{});
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, CAUSAL_TOKEN), 1);
    EXPECT_EQ(tryGetBoundInt(*bound, SLIDING_WINDOW_TOKEN), 0);
}

TEST(TestGfx942AttentionDenseGraphMatch, BindsNoMaskAsNonCausal)
{
    GraphSpec spec;
    spec.leftBound = -1;
    spec.rightBound = -1;
    const auto bound = matchGraph(spec);
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, CAUSAL_TOKEN), 0);
    EXPECT_EQ(tryGetBoundInt(*bound, SLIDING_WINDOW_TOKEN), 0);
}

TEST(TestGfx942AttentionDenseGraphMatch, ServesNonSquareFullAttention)
{
    GraphSpec spec;
    spec.leftBound = -1;
    spec.rightBound = -1;
    spec.seqLenQ = 512;
    spec.seqLenKv = 2048;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, BindsACausalWindowAsLeftBoundKeys)
{
    // The reference keeps keys (q - L, q], L of them (SEM-001); the kernel's
    // sliding_window counts the same keys: (q - W, q].
    for(const int64_t leftBound : {int64_t{1}, int64_t{128}, int64_t{512}, int64_t{2048}})
    {
        const auto bound = matchGraph(windowGraph(leftBound));
        ASSERT_TRUE(bound.has_value()) << leftBound;
        EXPECT_EQ(tryGetBoundInt(*bound, CAUSAL_TOKEN), 1) << leftBound;
        EXPECT_EQ(tryGetBoundInt(*bound, SLIDING_WINDOW_TOKEN), leftBound) << leftBound;
    }
}

TEST(TestGfx942AttentionDenseGraphMatch, ServesABottomRightCausalMaskOnlyAtEqualLengths)
{
    GraphSpec spec;
    spec.alignment = data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    EXPECT_TRUE(matchGraph(spec).has_value());
    spec.seqLenQ = 512;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, ServesABottomRightWindowOnlyAtEqualLengths)
{
    auto spec = windowGraph(128);
    spec.alignment = data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    const auto bound = matchGraph(spec);
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, SLIDING_WINDOW_TOKEN), 128);
    spec.seqLenQ = 512;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, ServesATopLeftWindowAtUnequalLengths)
{
    auto spec = windowGraph(128);
    spec.seqLenQ = 512;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesEveryBandThatIsNotCausal)
{
    // A lookahead, a non-causal left-only window and a two-sided band: the kernel
    // computes keys up to the diagonal only.
    GraphSpec lookahead;
    lookahead.leftBound = -1;
    lookahead.rightBound = 16;
    EXPECT_FALSE(matchGraph(lookahead).has_value());

    GraphSpec leftOnly;
    leftOnly.leftBound = 128;
    leftOnly.rightBound = -1;
    EXPECT_FALSE(matchGraph(leftOnly).has_value());

    GraphSpec twoSided;
    twoSided.leftBound = 128;
    twoSided.rightBound = 8;
    EXPECT_FALSE(matchGraph(twoSided).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesALeftBoundOfZero)
{
    // left_bound 0 is outside hipDNN's band domain (SEM-005).
    EXPECT_FALSE(matchGraph(windowGraph(0)).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesARaggedOperand)
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

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesAWindowThatLeavesQueryRowsNoKey)
{
    // Row q sees keys (q - W, q]; once seqlen_q - seqlen_kv >= W the last rows see none,
    // and the kernel has no empty-row path. One key short of that is served. 4096/512 is
    // also supports_attention_dense's per-block refusal at 256/64 (the last block's window
    // starts at key tile 58 of 8): on every tile-divisible shape, which kernel match
    // checks first, a block-level failure is a row-level one, so the block guard is
    // reached only through TestGfx942AttentionDenseGeometry.
    for(const auto& [seqLenQ, seqLenKv, served] : {std::tuple{512, 256, false},
                                                   std::tuple{4096, 512, false},
                                                   std::tuple{512, 384, false},
                                                   std::tuple{512, 385, true},
                                                   std::tuple{512, 512, true}})
    {
        SCOPED_TRACE(std::to_string(seqLenQ) + "/" + std::to_string(seqLenKv));
        auto spec = windowGraph(128);
        spec.seqLenQ = seqLenQ;
        spec.seqLenKv = seqLenKv;
        EXPECT_EQ(matchGraph(spec).has_value(), served);

        // Without a window every row keeps key 0.
        GraphSpec causal;
        causal.seqLenQ = seqLenQ;
        causal.seqLenKv = seqLenKv;
        EXPECT_TRUE(matchGraph(causal).has_value());
    }
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesSinksStatsMaskTensorAndVarlen)
{
    GraphSpec sinks;
    sinks.sinkTokenUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(sinks).has_value());

    GraphSpec stats;
    stats.statsUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(stats).has_value());

    GraphSpec attnMask;
    attnMask.attnMaskUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(attnMask).has_value());

    GraphSpec varlen;
    varlen.seqLenQUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(varlen).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesAnUnsupportedOperandOrLayout)
{
    GraphSpec headSize;
    headSize.headSize = 96;
    EXPECT_FALSE(matchGraph(headSize).has_value());

    GraphSpec fp32;
    fp32.dataType = data_objects::DataType::FLOAT;
    EXPECT_FALSE(matchGraph(fp32).has_value());

    GraphSpec bhsd;
    bhsd.bhsdQuery = true;
    EXPECT_FALSE(matchGraph(bhsd).has_value());

    GraphSpec gqa;
    gqa.numQueryHeads = 12;
    gqa.numKvHeads = 8;
    EXPECT_FALSE(matchGraph(gqa).has_value());
}

TEST(TestGfx942AttentionDenseGraphMatch, ServesAnyFiniteScaleAndDeclinesTheRest)
{
    // The body scales before the row max, so a negative or large scale is well defined.
    for(const float scale : {-0.5F, 0.0F, 64.0F})
    {
        GraphSpec spec;
        spec.attnScaleValue = scale;
        EXPECT_TRUE(matchGraph(spec).has_value()) << scale;
    }
    for(const float scale : {std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity()})
    {
        GraphSpec spec;
        spec.attnScaleValue = scale;
        EXPECT_FALSE(matchGraph(spec).has_value()) << scale;
    }
}

// =============================================================================
// kernel_match: the candidate's baked fields against the graph
// =============================================================================

TEST(TestGfx942AttentionDenseKernelMatch, ServesEveryHeadCountFromOneCandidate)
{
    // The candidate records Hq = Hkv = 8; the body reads both at runtime.
    for(const auto& [hq, hkv] : {std::pair<int64_t, int64_t>{8, 8},
                                 std::pair<int64_t, int64_t>{32, 8},
                                 std::pair<int64_t, int64_t>{128, 1},
                                 std::pair<int64_t, int64_t>{40, 40}})
    {
        GraphSpec spec;
        spec.numQueryHeads = hq;
        spec.numKvHeads = hkv;
        EXPECT_TRUE(matches(spec, KernelSpec{})) << hq << "/" << hkv;
    }
}

TEST(TestGfx942AttentionDenseKernelMatch, RefusesADifferentDtypeHeadSizeOrMask)
{
    KernelSpec fp16;
    fp16.dtype = "FP16";
    EXPECT_FALSE(matches(GraphSpec{}, fp16));

    KernelSpec d64;
    d64.headSize = 64;
    EXPECT_FALSE(matches(GraphSpec{}, d64));

    KernelSpec full;
    full.causal = 0;
    EXPECT_FALSE(matches(GraphSpec{}, full));
}

TEST(TestGfx942AttentionDenseKernelMatch, ServesAWindowOnlyFromTheCandidateBuiltForIt)
{
    KernelSpec w128;
    w128.slidingWindow = 128;
    KernelSpec w512;
    w512.slidingWindow = 512;

    EXPECT_TRUE(matches(windowGraph(128), w128));
    EXPECT_FALSE(matches(windowGraph(128), w512));
    EXPECT_FALSE(matches(windowGraph(128), KernelSpec{}));
    EXPECT_FALSE(matches(GraphSpec{}, w128));
    // left_bound 129 asks for 129 keys, which no block_n-aligned build carries.
    EXPECT_FALSE(matches(windowGraph(129), w128));
}

TEST(TestGfx942AttentionDenseKernelMatch, DivisibilityIsAgainstTheCandidatesOwnTile)
{
    GraphSpec spec;
    spec.seqLenQ = 384;
    spec.seqLenKv = 384;
    KernelSpec bm128;
    bm128.blockM = 128;
    EXPECT_TRUE(matches(spec, bm128));
    EXPECT_FALSE(matches(spec, KernelSpec{}));

    spec.seqLenKv = 352;
    KernelSpec bn32;
    bn32.blockM = 128;
    bn32.blockN = 32;
    EXPECT_TRUE(matches(spec, bn32));
    EXPECT_FALSE(matches(spec, bm128));
}

TEST(TestGfx942AttentionDenseKernelMatch, RefusesARaggedOrUnlaunchableCandidate)
{
    KernelSpec ragged;
    ragged.ragged = 1;
    EXPECT_FALSE(matches(GraphSpec{}, ragged));

    KernelSpec badTile;
    badTile.blockM = 64;
    badTile.blockN = 128;
    EXPECT_FALSE(matches(GraphSpec{}, badTile));
}

// =============================================================================
// score: a total cold order, policy arms first
// =============================================================================

TEST(TestGfx942AttentionDenseScore, ThePolicyArmsOutrankEveryAlternative)
{
    const double baseline = scoreOf(KernelSpec{});
    KernelSpec otherTile;
    otherTile.blockM = 128;
    KernelSpec otherPad;
    otherPad.ldsRowPad = 0;
    KernelSpec otherPriority;
    otherPriority.pvPriority = 1;
    EXPECT_GT(baseline, scoreOf(otherTile));
    EXPECT_GT(baseline, scoreOf(otherPad));
    EXPECT_GT(baseline, scoreOf(otherPriority));
    // The tile outranks the pad, which outranks the priority.
    EXPECT_GT(scoreOf(otherPriority), scoreOf(otherPad));
    EXPECT_GT(scoreOf(otherPad), scoreOf(otherTile));
}

TEST(TestGfx942AttentionDenseScore, EveryKnobCombinationScoresDistinctlyAndPositively)
{
    std::set<double> scores;
    std::size_t combinations = 0;
    for(const auto& [blockM, blockN] : {std::pair<int64_t, int64_t>{64, 64},
                                        std::pair<int64_t, int64_t>{128, 64},
                                        std::pair<int64_t, int64_t>{256, 32},
                                        std::pair<int64_t, int64_t>{256, 64},
                                        std::pair<int64_t, int64_t>{512, 64}})
    {
        for(const int64_t pad : {0, 4, 8, 12, 16, 24, 32})
        {
            for(const int64_t priority : {0, 1, 2, 3})
            {
                KernelSpec spec;
                spec.blockM = blockM;
                spec.blockN = blockN;
                spec.ldsRowPad = pad;
                spec.pvPriority = priority;
                const double score = scoreOf(spec);
                EXPECT_GT(score, 0.0);
                scores.insert(score);
                ++combinations;
            }
        }
    }
    EXPECT_EQ(scores.size(), combinations);
}

TEST(TestGfx942AttentionDenseScore, AnUnlaunchableRecordScoresZero)
{
    KernelSpec badTile;
    badTile.blockN = 48;
    EXPECT_EQ(scoreOf(badTile), 0.0);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
