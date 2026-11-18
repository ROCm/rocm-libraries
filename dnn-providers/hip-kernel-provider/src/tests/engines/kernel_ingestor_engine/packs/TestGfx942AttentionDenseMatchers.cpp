// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>

#include "engines/kernel_ingestor_engine/packs/Gfx942AttentionDenseGeometry.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/AttentionDenseTestArch.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/AttentionDenseTestGraphs.hpp"

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
 *   - the stats (LSE) output is bound, and served only by a candidate built with
 *     emit_lse, while the gfx950 twin still declines every stats output;
 *   - the cold ranking orders every knob combination, policy arms first.
 *
 * Matcher-only: no device, no compile, no launch.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

using namespace attention_dense_graphs;
using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;

constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx942_attention_dense.score";
constexpr std::string_view CAUSAL_TOKEN = "gfx942_attention_dense.causal";
constexpr std::string_view SLIDING_WINDOW_TOKEN = "gfx942_attention_dense.sliding_window";
constexpr std::string_view STATS_TOKEN = "gfx942_attention_dense.stats.uid";

/// The stats uid of a graph that binds no stats output.
constexpr int64_t NO_STATS = -1;

/// Base graph: bf16, D128, B=2, Hq=32, Hkv=8, Sq=Skv=1024, top-left causal.
GraphSpec baseGraph()
{
    GraphSpec spec;
    spec.numQueryHeads = 32;
    spec.numKvHeads = 8;
    spec.seqLenQ = 1024;
    spec.seqLenKv = 1024;
    return spec;
}

/// The base graph with the stats output the frontend builds.
GraphSpec statsGraph()
{
    return withStats(baseGraph());
}

std::optional<BoundTokens> matchGraph(const GraphSpec& spec)
{
    return attention_dense_graphs::matchGraph(GFX942_ATTENTION_DENSE, spec);
}

LoggedDecline loggedDecline(const GraphSpec& spec)
{
    return attention_dense_graphs::loggedDecline(GFX942_ATTENTION_DENSE, spec);
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
    int64_t emitLse = 0;
    /// The launch grid: the qb_minor grid unless a case moves it.
    int64_t persistent = 0;
    int64_t numPersistent = 304;
    std::string persistDecode = "auto";
    std::string nonpersistDecode = "qb_minor";
    int64_t ldsNumBuffers = 1;
    int64_t vRegPrefetch = 0;
    /// The body levers at their policy for bf16 D128 on the one-CTA-per-block grid.
    int64_t causalDiagSplit = 0;
    int64_t oStoreWidth = 4;
    std::string pvLoopOrder = "d_major";
    int64_t useCfvst = 1;
    int64_t useExp2Fast = 0;
    int64_t ldsKGroupPad = 8;
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
        {std::string("emit_lse"), spec.emitLse},
        {std::string("persistent"), spec.persistent},
        {std::string("num_persistent"), spec.numPersistent},
        {std::string("persist_decode"), spec.persistDecode},
        {std::string("nonpersist_decode"), spec.nonpersistDecode},
        {std::string("lds_num_buffers"), spec.ldsNumBuffers},
        {std::string("v_reg_prefetch"), spec.vRegPrefetch},
        {std::string("causal_diag_split"), spec.causalDiagSplit},
        {std::string("o_store_width"), spec.oStoreWidth},
        {std::string("pv_loop_order"), spec.pvLoopOrder},
        {std::string("use_cfvst"), spec.useCfvst},
        {std::string("use_exp2_fast"), spec.useExp2Fast},
        {std::string("lds_k_group_pad"), spec.ldsKGroupPad},
    };
    return kernel;
}

bool matches(const GraphSpec& graphSpec, const KernelDefinition& kernel)
{
    return matchesKernel(GFX942_ATTENTION_DENSE, graphSpec, kernel);
}

bool matches(const GraphSpec& graphSpec, const KernelSpec& kernelSpec)
{
    return matches(graphSpec, makeKernel(kernelSpec));
}

/// @p multiProcessorCount is the device's CU count the score reads (0: unresolved).
double scoreOf(const KernelSpec& kernelSpec,
               int multiProcessorCount = 0,
               const GraphSpec& graphSpec = baseGraph())
{
    const Graph graph(GFX942_ATTENTION_DENSE, graphSpec, multiProcessorCount);
    const auto bound = graph.match();
    EXPECT_TRUE(bound.has_value());
    return hipdnn_plugin_sdk::ingestor::ScoreRegistry::resolve(std::string(SCORE_SYMBOL))(
        graph.context(), bound.value_or(BoundTokens{}), makeKernel(kernelSpec));
}

GraphSpec windowGraph(int64_t leftBound)
{
    auto spec = baseGraph();
    spec.leftBound = leftBound;
    spec.rightBound = 0;
    return spec;
}

/// A D128 candidate on the persistent grid at @p decode, 304 CTAs, with the
/// single-instruction exp2 the persistent grid's policy turns on.
KernelSpec persistentKernel(const std::string& decode)
{
    KernelSpec spec;
    spec.persistent = 1;
    spec.persistDecode = decode;
    spec.useExp2Fast = 1;
    return spec;
}

/// One row of POLICY_GOLDENS: what rocKE resolves an unset use_cfvst / use_exp2_fast to.
struct PolicyGolden
{
    const char* dtype;
    int64_t headSize;
    bool persistent;
    bool useCfvst;
    bool useExp2Fast;
};

/// One kernel of the manifest's full enumeration (SCORE_GOLDENS), as its KMD record
/// carries it, and whether it is its cell's policy arm on either grid kind.
struct ScoreGolden
{
    int cell;
    const char* dtype;
    int64_t headSize;
    int64_t causal;
    int64_t slidingWindow;
    int64_t emitLse;
    int64_t blockM;
    int64_t blockN;
    int64_t ldsRowPad;
    int64_t pvPriority;
    int64_t persistent;
    int64_t numPersistent;
    const char* persistDecode;
    const char* nonpersistDecode;
    int64_t ldsNumBuffers;
    int64_t vRegPrefetch;
    int64_t causalDiagSplit;
    int64_t oStoreWidth;
    const char* pvLoopOrder;
    int64_t useCfvst;
    int64_t useExp2Fast;
    int64_t ldsKGroupPad;
    bool gridPolicy;
    bool persistentPolicy;
};

#include "TestGfx942AttentionDenseScoreGoldens.inc"

KernelSpec kernelSpecOf(const ScoreGolden& golden)
{
    KernelSpec spec;
    spec.dtype = golden.dtype;
    spec.headSize = golden.headSize;
    spec.causal = golden.causal;
    spec.slidingWindow = golden.slidingWindow;
    spec.emitLse = golden.emitLse;
    spec.blockM = golden.blockM;
    spec.blockN = golden.blockN;
    spec.ldsRowPad = golden.ldsRowPad;
    spec.pvPriority = golden.pvPriority;
    spec.persistent = golden.persistent;
    spec.numPersistent = golden.numPersistent;
    spec.persistDecode = golden.persistDecode;
    spec.nonpersistDecode = golden.nonpersistDecode;
    spec.ldsNumBuffers = golden.ldsNumBuffers;
    spec.vRegPrefetch = golden.vRegPrefetch;
    spec.causalDiagSplit = golden.causalDiagSplit;
    spec.oStoreWidth = golden.oStoreWidth;
    spec.pvLoopOrder = golden.pvLoopOrder;
    spec.useCfvst = golden.useCfvst;
    spec.useExp2Fast = golden.useExp2Fast;
    spec.ldsKGroupPad = golden.ldsKGroupPad;
    return spec;
}

/// A candidate on the one-CTA-per-block grid at @p order.
KernelSpec gridKernel(const std::string& order)
{
    KernelSpec spec;
    spec.nonpersistDecode = order;
    return spec;
}

/// A causal graph with @p heads query heads over @p kvHeads.
GraphSpec headsGraph(int64_t heads, int64_t kvHeads)
{
    auto spec = baseGraph();
    spec.numQueryHeads = heads;
    spec.numKvHeads = kvHeads;
    return spec;
}

} // namespace

// =============================================================================
// graph_match: what the engine serves, and the mask it binds
// =============================================================================

TEST(TestGfx942AttentionDenseGraphMatch, BindsTopLeftCausalAsCausalWithNoWindow)
{
    const auto bound = matchGraph(baseGraph());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, CAUSAL_TOKEN), 1);
    EXPECT_EQ(tryGetBoundInt(*bound, SLIDING_WINDOW_TOKEN), 0);
}

TEST(TestGfx942AttentionDenseGraphMatch, BindsNoMaskAsNonCausal)
{
    auto spec = baseGraph();
    spec.leftBound = -1;
    spec.rightBound = -1;
    const auto bound = matchGraph(spec);
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, CAUSAL_TOKEN), 0);
    EXPECT_EQ(tryGetBoundInt(*bound, SLIDING_WINDOW_TOKEN), 0);
}

TEST(TestGfx942AttentionDenseGraphMatch, ServesNonSquareFullAttention)
{
    auto spec = baseGraph();
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
    auto spec = baseGraph();
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
    auto lookahead = baseGraph();
    lookahead.leftBound = -1;
    lookahead.rightBound = 16;
    EXPECT_FALSE(matchGraph(lookahead).has_value());

    auto leftOnly = baseGraph();
    leftOnly.leftBound = 128;
    leftOnly.rightBound = -1;
    EXPECT_FALSE(matchGraph(leftOnly).has_value());

    auto twoSided = baseGraph();
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
        auto spec = baseGraph();
        spec.raggedUid = uid;
        EXPECT_FALSE(matchGraph(spec).has_value());
    }
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesAWindowThatLeavesQueryRowsNoKey)
{
    // Row q sees keys (q - W, q]; once seqlen_q - seqlen_kv >= W the last rows see none.
    // The pack declines them (windowLeavesRowsWithoutKeys says why). One key short of that
    // is served. 4096/512 also fails the per-block guard at 256/64 (the last block's window
    // starts at key tile 58 of 8): on every tile-divisible shape, which kernel match checks
    // first, a block-level failure is a row-level one, so the block guard is reached only
    // through TestGfx942AttentionDenseGeometry.
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
        auto causal = baseGraph();
        causal.seqLenQ = seqLenQ;
        causal.seqLenKv = seqLenKv;
        EXPECT_TRUE(matchGraph(causal).has_value());
    }
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesSinksADanglingStatsUidMaskTensorAndVarlen)
{
    auto sinks = baseGraph();
    sinks.sinkTokenUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(sinks).has_value());

    auto stats = baseGraph();
    stats.statsUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(stats).has_value());

    auto attnMask = baseGraph();
    attnMask.attnMaskUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(attnMask).has_value());

    auto varlen = baseGraph();
    varlen.seqLenQUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(varlen).has_value());

    // Cumulative lengths, on each side alone. padding_mask stays false, so only the
    // cu_seq_len decline can reject these.
    auto cuQ = baseGraph();
    cuQ.cuSeqLenQUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(cuQ).has_value());

    auto cuKv = baseGraph();
    cuKv.cuSeqLenKvUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(cuKv).has_value());
}

// =============================================================================
// graph_match: the stats (LSE) output
// =============================================================================

// The FlyDSL rule: a stats uid is served unless generate_stats is explicitly false, since
// both references write the LSE whenever the uid is set; generate_stats true without a uid
// names no tensor to write. A graph without either binds no stats.
TEST(TestGfx942AttentionDenseGraphMatch, BindsStatsByTheUidUnlessGenerateStatsIsFalse)
{
    struct Case
    {
        const char* name;
        bool withUid;
        std::optional<bool> generateStats;
        std::optional<int64_t> bound;
    };
    for(const auto& testCase : {Case{"uid, flag unset", true, std::nullopt, STATS_UID},
                                Case{"uid, flag true", true, true, STATS_UID},
                                Case{"uid, flag false", true, false, std::nullopt},
                                Case{"no uid, flag true", false, true, std::nullopt},
                                Case{"no uid, flag unset", false, std::nullopt, NO_STATS},
                                Case{"no uid, flag false", false, false, NO_STATS}})
    {
        SCOPED_TRACE(testCase.name);
        auto spec = statsGraph();
        spec.generateStats = testCase.generateStats;
        if(!testCase.withUid)
        {
            spec.statsUid.reset();
            spec.statsTensor.reset();
        }
        const auto bound = matchGraph(spec);
        ASSERT_EQ(bound.has_value(), testCase.bound.has_value());
        if(bound.has_value())
        {
            EXPECT_EQ(tryGetBoundInt(*bound, STATS_TOKEN), testCase.bound);
        }
        else
        {
            EXPECT_EQ(loggedDecline(spec).cause, "stats_flag");
        }
    }
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesAStatsTensorThatIsNotFloat)
{
    for(const auto dataType : {data_objects::DataType::BFLOAT16,
                               data_objects::DataType::HALF,
                               data_objects::DataType::DOUBLE})
    {
        SCOPED_TRACE(static_cast<int>(dataType));
        auto spec = statsGraph();
        spec.statsTensor->dataType = dataType;
        const auto decline = loggedDecline(spec);
        EXPECT_EQ(decline.cause, "stats_dtype");
        EXPECT_NE(decline.detail.find("stats (uid 5)"), std::string::npos) << decline.detail;
    }
}

// The kernel writes row (b, h, s) at element (b * Hq + h) * Sq + s and takes no stride
// arguments, so only packed [B, Hq, Sq, 1] is served.
TEST(TestGfx942AttentionDenseGraphMatch, DeclinesAStatsTensorTheKernelDoesNotWrite)
{
    // B = 2, Hq = 32, Sq = 1024.
    struct Case
    {
        const char* name;
        TensorSpec tensor;
    };
    TensorSpec bshdOrdered;
    bshdOrdered.strides = {int64_t{1024} * 32, 1, 32, 1};
    TensorSpec rank3;
    rank3.dims = {2, 32, 1024};
    TensorSpec kvHeads;
    kvHeads.dims = {2, 8, 1024, 1};
    TensorSpec wideRow;
    wideRow.dims = {2, 32, 1024, 2};
    TensorSpec paddedBatch;
    paddedBatch.strides = {int64_t{32} * 1024 + 64, 1024, 1, 1};
    TensorSpec virtualStats;
    virtualStats.isVirtual = true;
    for(const auto& testCase : {Case{"BSHD-ordered strides", bshdOrdered},
                                Case{"rank 3", rank3},
                                Case{"kv heads", kvHeads},
                                Case{"two columns", wideRow},
                                Case{"padded batch stride", paddedBatch},
                                Case{"virtual", virtualStats}})
    {
        SCOPED_TRACE(testCase.name);
        auto spec = statsGraph();
        spec.statsTensor = testCase.tensor;
        EXPECT_EQ(loggedDecline(spec).cause, "stats_layout");
    }

    // A uid that names no tensor, and a negative one, which would read back as no stats.
    for(const int64_t uid : {EXTRA_UID, int64_t{-1}, int64_t{-7}})
    {
        SCOPED_TRACE(uid);
        auto spec = statsGraph();
        spec.statsUid = uid;
        spec.statsTensor->uid = uid == EXTRA_UID ? STATS_UID : uid;
        EXPECT_EQ(loggedDecline(spec).cause, "stats_layout");
    }
}

// A unit-extent axis's stride multiplies an index that is always 0.
TEST(TestGfx942AttentionDenseGraphMatch, ServesPackedStatsWithUnitExtentAxesExempt)
{
    auto oneBatch = statsGraph();
    oneBatch.batch = 1;
    oneBatch.statsTensor->strides = {7, 1024, 1, 3};
    EXPECT_TRUE(matchGraph(oneBatch).has_value());

    auto oneHead = statsGraph();
    oneHead.numQueryHeads = 1;
    oneHead.numKvHeads = 1;
    oneHead.statsTensor->strides = {1024, 5, 1, 1};
    EXPECT_TRUE(matchGraph(oneHead).has_value());

    // The trailing axis is always unit-extent.
    auto anyTrailing = statsGraph();
    anyTrailing.statsTensor->strides = {int64_t{32} * 1024, 1024, 1, 1024};
    const auto bound = matchGraph(anyTrailing);
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, STATS_TOKEN), STATS_UID);
}

// Max, Sum_exp and the dropout RNG dump are outputs the kernel cannot write, with or
// without the LSE beside them.
TEST(TestGfx942AttentionDenseGraphMatch, DeclinesTheRowStatsOutputs)
{
    for(const bool withStats : {true, false})
    {
        SCOPED_TRACE(withStats ? "with stats" : "without stats");
        auto max = withStats ? statsGraph() : baseGraph();
        max.maxUid = EXTRA_UID;
        EXPECT_EQ(loggedDecline(max).cause, "stats_rowstats");

        auto sumExp = withStats ? statsGraph() : baseGraph();
        sumExp.sumExpUid = EXTRA_UID;
        EXPECT_EQ(loggedDecline(sumExp).cause, "stats_rowstats");

        auto rngDump = withStats ? statsGraph() : baseGraph();
        rngDump.rngDumpUid = EXTRA_UID;
        EXPECT_EQ(loggedDecline(rngDump).cause, "stats_rowstats");
    }
}

// The empty-row decline stands for a graph that binds stats too (windowLeavesRowsWithoutKeys
// says why).
TEST(TestGfx942AttentionDenseGraphMatch, KeepsDecliningAWindowThatLeavesRowsNoKeyWithStats)
{
    auto spec = statsGraph();
    spec.leftBound = 128;
    spec.seqLenQ = 512;
    spec.seqLenKv = 384;
    EXPECT_EQ(loggedDecline(spec).cause, "mask");

    spec.seqLenKv = 385;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

// The gfx950 twin has no LSE kernel: every stats output still declines there, under its
// one 'stats' cause, while gfx942 binds the same graph.
TEST(TestGfx942AttentionDenseGraphMatch, LeavesTheGfx950TwinDecliningEveryStatsOutput)
{
    ASSERT_TRUE(matchGraph(GFX950_ATTENTION_DENSE, baseGraph()).has_value())
        << "gfx950 must serve the base graph, or the declines below prove nothing";
    ASSERT_TRUE(matchGraph(statsGraph()).has_value());

    auto flagOnly = baseGraph();
    flagOnly.generateStats = true;
    auto uidOnly = statsGraph();
    uidOnly.generateStats.reset();
    auto max = baseGraph();
    max.maxUid = EXTRA_UID;
    for(const auto& spec : {statsGraph(), uidOnly, flagOnly, max})
    {
        const auto decline = loggedDecline(GFX950_ATTENTION_DENSE, spec);
        EXPECT_EQ(decline.cause, "stats");
        EXPECT_EQ(decline.detail,
                  ": softmax statistics outputs (stats, max, sum_exp, rng_dump or "
                  "generate_stats) are not supported");
    }

    auto disabled = baseGraph();
    disabled.generateStats = false;
    const auto bound = matchGraph(GFX950_ATTENTION_DENSE, disabled);
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(bound->count("gfx950_attention_dense.stats.uid"), 0U);
}

TEST(TestGfx942AttentionDenseGraphMatch, DeclinesAnUnsupportedOperandOrLayout)
{
    auto headSize = baseGraph();
    headSize.headSize = 96;
    EXPECT_FALSE(matchGraph(headSize).has_value());

    auto fp32 = baseGraph();
    fp32.dataType = data_objects::DataType::FLOAT;
    EXPECT_FALSE(matchGraph(fp32).has_value());

    auto bhsd = baseGraph();
    bhsd.operand(Q_UID).layout = SdpaLayout::BHSD;
    EXPECT_FALSE(matchGraph(bhsd).has_value());

    auto gqa = baseGraph();
    gqa.numQueryHeads = 12;
    gqa.numKvHeads = 8;
    EXPECT_FALSE(matchGraph(gqa).has_value());
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
        auto spec = baseGraph();
        spec.numQueryHeads = hq;
        spec.numKvHeads = hkv;
        EXPECT_TRUE(matches(spec, KernelSpec{})) << hq << "/" << hkv;
    }
}

TEST(TestGfx942AttentionDenseKernelMatch, RefusesADifferentDtypeHeadSizeOrMask)
{
    KernelSpec fp16;
    fp16.dtype = "FP16";
    EXPECT_FALSE(matches(baseGraph(), fp16));

    KernelSpec d64;
    d64.headSize = 64;
    EXPECT_FALSE(matches(baseGraph(), d64));

    KernelSpec full;
    full.causal = 0;
    EXPECT_FALSE(matches(baseGraph(), full));
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
    EXPECT_FALSE(matches(baseGraph(), w128));
    // left_bound 129 asks for 129 keys, which no block_n-aligned build carries.
    EXPECT_FALSE(matches(windowGraph(129), w128));
}

TEST(TestGfx942AttentionDenseKernelMatch, DivisibilityIsAgainstTheCandidatesOwnTile)
{
    auto spec = baseGraph();
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
    EXPECT_FALSE(matches(baseGraph(), ragged));

    KernelSpec badTile;
    badTile.blockM = 64;
    badTile.blockN = 128;
    EXPECT_FALSE(matches(baseGraph(), badTile));
}

// LSE is baked at compile time: an emit_lse candidate takes the trailing lse_ptr and
// serves exactly the graphs that bind a stats output.
TEST(TestGfx942AttentionDenseKernelMatch, ServesAStatsGraphOnlyFromAnLseCandidate)
{
    KernelSpec lse;
    lse.emitLse = 1;

    EXPECT_TRUE(matches(statsGraph(), lse));
    EXPECT_FALSE(matches(statsGraph(), KernelSpec{}));
    EXPECT_TRUE(matches(baseGraph(), KernelSpec{}));
    EXPECT_FALSE(matches(baseGraph(), lse));

    // A uid with generate_stats unset is a stats graph too.
    auto uidOnly = statsGraph();
    uidOnly.generateStats.reset();
    EXPECT_TRUE(matches(uidOnly, lse));
    EXPECT_FALSE(matches(uidOnly, KernelSpec{}));
}

// A completed record always carries emit_lse (the KMD default is 0), so an absent,
// mistyped or out-of-range value is a malformed record, declined for either graph.
TEST(TestGfx942AttentionDenseKernelMatch, RefusesACandidateWithoutAValidEmitLse)
{
    auto absent = makeKernel(KernelSpec{});
    absent.metadata.erase("emit_lse");
    auto mistyped = makeKernel(KernelSpec{});
    mistyped.metadata[std::string("emit_lse")] = std::string("0");
    KernelSpec two;
    two.emitLse = 2;
    KernelSpec negative;
    negative.emitLse = -1;

    for(const auto& graph : {baseGraph(), statsGraph()})
    {
        EXPECT_FALSE(matches(graph, absent));
        EXPECT_FALSE(matches(graph, mistyped));
        EXPECT_FALSE(matches(graph, two));
        EXPECT_FALSE(matches(graph, negative));
    }
}

// Every grid the catalog carries serves a graph its decode is a bijection for.
TEST(TestGfx942AttentionDenseKernelMatch, ServesEveryGridWhereItsDecodeIsLegal)
{
    for(const char* order : {"qb_minor", "bt_hkv_minor", "hq_minor_swz"})
    {
        EXPECT_TRUE(matches(baseGraph(), gridKernel(order))) << order;
    }
    for(const char* decode : {"qb_major", "hkv_major", "bt_hkv_minor", "hq_minor_swz"})
    {
        EXPECT_TRUE(matches(baseGraph(), persistentKernel(decode))) << decode;
    }
}

// hq_minor_swz gives each of the 8 XCDs a band of Hq / 8 heads: at Hq % 8 != 0 it drops
// and repeats work items with a success status, so both grids decline such a graph, and
// the persistent one also an odd query-block count under causal balancing. Every other
// grid still serves it.
TEST(TestGfx942AttentionDenseKernelMatch, DeclinesHqMinorSwzWhereItIsNotABijection)
{
    const auto twelve = headsGraph(12, 4);
    EXPECT_FALSE(matches(twelve, gridKernel("hq_minor_swz")));
    EXPECT_FALSE(matches(twelve, persistentKernel("hq_minor_swz")));
    EXPECT_TRUE(matches(twelve, gridKernel("qb_minor")));
    EXPECT_TRUE(matches(twelve, gridKernel("bt_hkv_minor")));
    EXPECT_TRUE(matches(twelve, persistentKernel("bt_hkv_minor")));
    EXPECT_TRUE(matches(headsGraph(16, 2), persistentKernel("hq_minor_swz")));

    // Sq 768 over 256-row blocks: three query blocks, causal and no window.
    auto oddBlocks = headsGraph(16, 2);
    oddBlocks.seqLenQ = 768;
    oddBlocks.seqLenKv = 768;
    EXPECT_FALSE(matches(oddBlocks, persistentKernel("hq_minor_swz")));
    EXPECT_TRUE(matches(oddBlocks, gridKernel("hq_minor_swz")));
    EXPECT_TRUE(matches(oddBlocks, persistentKernel("qb_major")));
}

// bt_hkv_minor divides gqa * t by Hq: with Hq = 32768 over one KV head that dividend reaches
// 2^30, where the decode's fast division is wrong, while the extents are still legal.
TEST(TestGfx942AttentionDenseKernelMatch, DeclinesAPersistentDecodePastItsDividendBound)
{
    auto extreme = headsGraph(32768, 1);
    extreme.batch = 1;
    extreme.seqLenQ = 256;
    extreme.seqLenKv = 256;
    EXPECT_FALSE(matches(extreme, persistentKernel("bt_hkv_minor")));
    EXPECT_TRUE(matches(extreme, persistentKernel("qb_major")));
    EXPECT_TRUE(matches(extreme, gridKernel("bt_hkv_minor")));
}

// A completed record always carries the grid fields, so an absent or unknown value, an
// `auto` or gqa_pair decode (their kernel arguments are unknown here) or a CTA count below
// one is a malformed record.
TEST(TestGfx942AttentionDenseKernelMatch, RefusesACandidateWithoutAGridItLaunches)
{
    std::vector<KernelDefinition> malformed;
    for(const char* field : {"persistent", "nonpersist_decode"})
    {
        auto absent = makeKernel(KernelSpec{});
        absent.metadata.erase(field);
        malformed.push_back(absent);
    }
    KernelSpec two;
    two.persistent = 2;
    malformed.push_back(makeKernel(two));
    malformed.push_back(makeKernel(gridKernel("auto")));
    malformed.push_back(makeKernel(gridKernel("persistent")));
    for(const char* decode : {"auto", "gqa_pair", "gqa_pair_2phase", ""})
    {
        malformed.push_back(makeKernel(persistentKernel(decode)));
    }
    auto noCtas = persistentKernel("qb_major");
    noCtas.numPersistent = 0;
    malformed.push_back(makeKernel(noCtas));
    auto mistyped = makeKernel(persistentKernel("qb_major"));
    mistyped.metadata[std::string("persist_decode")] = int64_t{0};
    malformed.push_back(mistyped);

    for(const auto& kernel : malformed)
    {
        EXPECT_FALSE(matches(baseGraph(), kernel));
    }
    // persist_decode and num_persistent are read only on the persistent grid.
    auto gridWithAuto = gridKernel("bt_hkv_minor");
    gridWithAuto.numPersistent = 0;
    EXPECT_TRUE(matches(baseGraph(), gridWithAuto));
}

// =============================================================================
// score: a total cold order, policy arms first
// =============================================================================

// U3: persistent is the cold choice at D128 without a narrow window where its 304 CTAs are
// a multiple of the device's CU count (304 or 38 CUs); otherwise the qb_minor grid comes
// first and persistent after every block order.
TEST(TestGfx942AttentionDenseScore, PersistentLeadsOnlyWhereItsCtasFillTheDevice)
{
    const auto persistent = persistentKernel("bt_hkv_minor");
    for(const int cus : {304, 38})
    {
        EXPECT_GT(scoreOf(persistent, cus), scoreOf(KernelSpec{}, cus)) << cus;
        EXPECT_GT(scoreOf(KernelSpec{}, cus), scoreOf(gridKernel("bt_hkv_minor"), cus)) << cus;
    }
    for(const int cus : {228, 80, 0})
    {
        EXPECT_GT(scoreOf(KernelSpec{}, cus), scoreOf(gridKernel("bt_hkv_minor"), cus)) << cus;
        EXPECT_GT(scoreOf(gridKernel("hq_minor_swz"), cus), scoreOf(persistent, cus)) << cus;
    }

    // Not at D64, and not under a window of 256 keys or fewer.
    auto d64 = persistent;
    d64.headSize = 64;
    KernelSpec d64Grid;
    d64Grid.headSize = 64;
    auto d64Graph = baseGraph();
    d64Graph.headSize = 64;
    EXPECT_GT(scoreOf(d64Grid, 304, d64Graph), scoreOf(d64, 304, d64Graph));
    for(const int64_t window : {int64_t{128}, int64_t{256}})
    {
        auto narrow = persistentKernel("qb_major");
        narrow.slidingWindow = window;
        KernelSpec narrowGrid;
        narrowGrid.slidingWindow = window;
        EXPECT_GT(scoreOf(narrowGrid, 304, windowGraph(window)),
                  scoreOf(narrow, 304, windowGraph(window)))
            << window;
    }
    auto wide = persistentKernel("bt_hkv_minor");
    wide.slidingWindow = 512;
    KernelSpec wideGrid;
    wideGrid.slidingWindow = 512;
    EXPECT_GT(scoreOf(wide, 304, windowGraph(512)), scoreOf(wideGrid, 304, windowGraph(512)));
}

// Every grid crossed with the tile and the persistent pipeline levers: distinct, positive
// scores on both a device persistent leads on and one it does not, the policy arm first.
TEST(TestGfx942AttentionDenseScore, EveryGridAndPipelineArmScoresDistinctly)
{
    std::vector<KernelSpec> grids;
    for(const char* order : {"qb_minor", "bt_hkv_minor", "hq_minor_swz"})
    {
        grids.push_back(gridKernel(order));
    }
    for(const char* decode : {"qb_major", "hkv_major", "bt_hkv_minor", "hq_minor_swz"})
    {
        grids.push_back(persistentKernel(decode));
    }
    for(const int cus : {304, 228})
    {
        std::set<double> scores;
        std::size_t combinations = 0;
        double best = 0.0;
        for(const auto& grid : grids)
        {
            for(const int64_t blockM : {int64_t{128}, int64_t{256}})
            {
                for(const int64_t buffers : {int64_t{1}, int64_t{2}})
                {
                    for(const int64_t prefetch : {int64_t{0}, int64_t{1}})
                    {
                        auto spec = grid;
                        spec.blockM = blockM;
                        spec.ldsNumBuffers = buffers;
                        spec.vRegPrefetch = prefetch;
                        const double score = scoreOf(spec, cus);
                        EXPECT_GT(score, 0.0);
                        best = std::max(best, score);
                        scores.insert(score);
                        ++combinations;
                    }
                }
            }
        }
        EXPECT_EQ(scores.size(), combinations) << cus;
        const auto policy = cus == 304 ? persistentKernel("bt_hkv_minor") : KernelSpec{};
        EXPECT_EQ(best, scoreOf(policy, cus)) << cus;
    }
}

// Among persistent arms of one tile, the decode the registry's base arm pins for the mask
// comes first: bt_hkv_minor where the decodes balance causal work at the build shape,
// qb_major elsewhere.
TEST(TestGfx942AttentionDenseScore, ThePinnedDecodeLeadsThePersistentArms)
{
    for(const char* other : {"qb_major", "hkv_major", "hq_minor_swz"})
    {
        EXPECT_GT(scoreOf(persistentKernel("bt_hkv_minor"), 304),
                  scoreOf(persistentKernel(other), 304))
            << other;
    }
    auto unmasked = baseGraph();
    unmasked.leftBound.reset();
    unmasked.rightBound.reset();
    for(const char* other : {"bt_hkv_minor", "hkv_major", "hq_minor_swz"})
    {
        auto pinned = persistentKernel("qb_major");
        pinned.causal = 0;
        auto alternative = persistentKernel(other);
        alternative.causal = 0;
        EXPECT_GT(scoreOf(pinned, 304, unmasked), scoreOf(alternative, 304, unmasked)) << other;
    }
}

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

// The body levers' policies restate rocKE's default_use_cfvst and default_use_exp2_fast per
// dtype, head size and grid kind (TestGfx942AttentionDenseScoreGoldens.inc).
TEST(TestGfx942AttentionDenseScore, TheBodyPoliciesRestateThePython)
{
    for(const auto& golden : POLICY_GOLDENS)
    {
        EXPECT_EQ(gfx942AttentionDenseCfvstPolicy(golden.dtype, golden.headSize), golden.useCfvst)
            << golden.dtype << " D" << golden.headSize;
        EXPECT_EQ(
            gfx942AttentionDenseExp2FastPolicy(golden.dtype, golden.headSize, golden.persistent),
            golden.useExp2Fast)
            << golden.dtype << " D" << golden.headSize << " persistent " << golden.persistent;
    }
}

// Each body lever ranks its policy arm first and every alternative after the levers before
// it in the lexicographic order: one alternative of a later lever outranks one of an earlier.
TEST(TestGfx942AttentionDenseScore, EveryBodyLeverPlacesItsPolicyArmFirst)
{
    const double policy = scoreOf(KernelSpec{});
    KernelSpec diagSplit;
    diagSplit.causalDiagSplit = 1;
    KernelSpec narrowStore;
    narrowStore.oStoreWidth = 2;
    KernelSpec narrowestStore;
    narrowestStore.oStoreWidth = 1;
    KernelSpec kMajor;
    kMajor.pvLoopOrder = "k_major";
    KernelSpec noCfvst;
    noCfvst.useCfvst = 0;
    KernelSpec exp2Fast;
    exp2Fast.useExp2Fast = 1;
    // In order: diag split, O store (its alternatives ascending: 1 ahead of 2, so 2 is the
    // worse), PV loop order, cfvst, exp2_fast.
    const std::vector<KernelSpec> alternatives{
        diagSplit, narrowStore, narrowestStore, kMajor, noCfvst, exp2Fast};
    for(std::size_t i = 0; i < alternatives.size(); ++i)
    {
        EXPECT_GT(policy, scoreOf(alternatives[i])) << i;
        if(i + 1 < alternatives.size())
        {
            EXPECT_LT(scoreOf(alternatives[i]), scoreOf(alternatives[i + 1])) << i;
        }
    }
    // The K-group pad, last, at D64 where it is a real lever: 8 first, then 0, 16, 24, 32.
    auto d64Graph = baseGraph();
    d64Graph.headSize = 64;
    KernelSpec d64;
    d64.headSize = 64;
    d64.useCfvst = 0;
    d64.useExp2Fast = 1;
    double previous = scoreOf(d64, 0, d64Graph);
    for(const int64_t pad : {0, 16, 24, 32})
    {
        auto spec = d64;
        spec.ldsKGroupPad = pad;
        const double score = scoreOf(spec, 0, d64Graph);
        EXPECT_LT(score, previous) << pad;
        previous = score;
    }
    // The exp2_fast policy follows the grid: on the persistent grid exp2_fast on leads.
    auto persistentOff = persistentKernel("bt_hkv_minor");
    persistentOff.useExp2Fast = 0;
    EXPECT_GT(scoreOf(persistentKernel("bt_hkv_minor"), 304), scoreOf(persistentOff, 304));
}

TEST(TestGfx942AttentionDenseScore, AMalformedBodyLeverScoresZero)
{
    KernelSpec unknownOrder;
    unknownOrder.pvLoopOrder = "auto";
    KernelSpec badCfvst;
    badCfvst.useCfvst = 2;
    KernelSpec badExp2Fast;
    badExp2Fast.useExp2Fast = -1;
    KernelSpec badStore;
    badStore.oStoreWidth = 0;
    KernelSpec badPad;
    badPad.ldsKGroupPad = -8;
    for(const auto& spec : {unknownOrder, badCfvst, badExp2Fast, badStore, badPad})
    {
        EXPECT_EQ(scoreOf(spec), 0.0);
    }
    auto missing = makeKernel(KernelSpec{});
    missing.metadata.erase("use_exp2_fast");
    const Graph graph(GFX942_ATTENTION_DENSE, baseGraph());
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::ScoreRegistry::resolve(std::string(SCORE_SYMBOL))(
                  graph.context(), BoundTokens{}, missing),
              0.0);
}

// E2 over the manifest's full enumeration (SCORE_GOLDENS): in every cell each kernel scores
// distinctly and positively, and the cell's policy arm scores strictly highest -- the
// persistent base arm where persistent leads (D128 with no window or one wider than 256 keys,
// 304 % CUs == 0), the cell default elsewhere.
TEST(TestGfx942AttentionDenseScore, ThePolicyArmLeadsEveryCellOfTheFullEnumeration)
{
    const auto score
        = hipdnn_plugin_sdk::ingestor::ScoreRegistry::resolve(std::string(SCORE_SYMBOL));
    for(const int cus : {304, 228})
    {
        const Graph graph(GFX942_ATTENTION_DENSE, baseGraph(), cus);
        std::map<int, std::vector<std::pair<double, const ScoreGolden*>>> cells;
        for(const auto& golden : SCORE_GOLDENS)
        {
            cells[golden.cell].emplace_back(
                score(graph.context(), BoundTokens{}, makeKernel(kernelSpecOf(golden))), &golden);
        }
        EXPECT_EQ(cells.size(), 48U);
        for(const auto& [cell, scored] : cells)
        {
            std::set<double> distinct;
            const auto* best = scored.front().second;
            double bestScore = scored.front().first;
            for(const auto& [value, golden] : scored)
            {
                EXPECT_GT(value, 0.0) << cell;
                distinct.insert(value);
                if(value > bestScore)
                {
                    bestScore = value;
                    best = golden;
                }
            }
            EXPECT_EQ(distinct.size(), scored.size()) << "cell " << cell << " CUs " << cus;
            const bool persistentLeads = cus == 304 && best->headSize == 128
                                         && (best->slidingWindow == 0 || best->slidingWindow > 256);
            EXPECT_TRUE(persistentLeads ? best->persistentPolicy : best->gridPolicy)
                << "cell " << cell << " CUs " << cus << ": the best arm is block " << best->blockM
                << "/" << best->blockN << " persistent " << best->persistent;
        }
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
