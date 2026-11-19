// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>

#include "tests/engines/kernel_ingestor_engine/packs/AttentionDenseTestArch.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/AttentionDenseTestGraphs.hpp"

/**
 * @file TestAttentionDenseMatchers.cpp
 * @brief The graph and kernel match rules the gfx942 and gfx950 dense packs share, run
 *        on both arches.
 *
 * Both packs bind the same graphMatches/kernelMatches (AttentionDenseNative.cpp), so a
 * rule that holds on one must hold on the other. The per-arch suites
 * (TestGfx94xAttentionDenseMatchers.cpp) keep what differs: masks, stats, tiles, ranking.
 * Each decline here is checked by the cause the matcher logs, so a graph declined for an
 * earlier, unrelated reason cannot pass for the rule under test.
 *
 * Matcher-only: no device, no compile, no launch.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

using namespace attention_dense_graphs;

/// The kernel's 32-bit offset limit: K/V bytes and Q/O elements must stay below it.
constexpr int64_t OFFSET_LIMIT = int64_t{1} << 31;

/// The graph every case here perturbs: the default at a 0.125 scale.
GraphSpec baseGraph()
{
    GraphSpec spec;
    spec.attnScaleValue = 0.125F;
    return spec;
}

/// A D128 bm256/bn64 BF16 causal record for @p spec, completed as @p arch's KMD defaults
/// complete it. gfx950 compares the baked head counts, so they are the graph's.
KernelDefinition makeKernel(const AttentionDenseTestArch& arch, const GraphSpec& spec)
{
    KernelDefinition kernel;
    kernel.kernelId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-00000000d5a1");
    kernel.name = std::string(arch.kernelName);
    kernel.metadata = completedAttentionDenseMetadata(arch, spec.headSize, 256, 64);
    kernel.metadata[std::string("dtype")] = std::string("BF16");
    kernel.metadata[std::string("num_query_heads")] = spec.numQueryHeads;
    kernel.metadata[std::string("num_kv_heads")] = spec.numKvHeads;
    kernel.metadata[std::string("seqlen_q")] = spec.seqLenQ;
    kernel.metadata[std::string("seqlen_kv")] = spec.seqLenKv;
    kernel.metadata[std::string("batch")] = spec.batch;
    kernel.metadata[std::string("causal")] = int64_t{1};
    kernel.metadata[std::string("sliding_window")] = int64_t{0};
    kernel.metadata[std::string("ragged")] = int64_t{0};
    if(arch.completesGridOrder)
    {
        // gfx942's body knobs, at the bf16 D128 policy.
        kernel.metadata[std::string("lds_row_pad")] = int64_t{8};
        kernel.metadata[std::string("pv_priority")] = int64_t{0};
        kernel.metadata[std::string("emit_lse")] = int64_t{0};
        kernel.metadata[std::string("lds_num_buffers")] = int64_t{1};
        kernel.metadata[std::string("v_reg_prefetch")] = int64_t{0};
        kernel.metadata[std::string("causal_diag_split")] = int64_t{0};
        kernel.metadata[std::string("o_store_width")] = int64_t{4};
        kernel.metadata[std::string("pv_loop_order")] = std::string("d_major");
        kernel.metadata[std::string("use_cfvst")] = int64_t{1};
        kernel.metadata[std::string("use_exp2_fast")] = int64_t{0};
        kernel.metadata[std::string("lds_k_group_pad")] = int64_t{8};
    }
    return kernel;
}

class TestAttentionDenseMatch : public ::testing::TestWithParam<const AttentionDenseTestArch*>
{
protected:
    static const AttentionDenseTestArch& arch()
    {
        return *GetParam();
    }
};

} // namespace

/// Every negative below perturbs this graph, so it must be served end to end.
TEST_P(TestAttentionDenseMatch, ServesTheBaseGraphAndRecord)
{
    const auto spec = baseGraph();
    EXPECT_TRUE(matchesKernel(arch(), spec, makeKernel(arch(), spec)));
}

// =============================================================================
// The node's compute data type. The kernel accumulates in FLOAT, which meets FLOAT, HALF
// and BFLOAT16; UNSET leaves the choice to the provider.
// =============================================================================

TEST_P(TestAttentionDenseMatch, ServesTheComputeTypesItsFloatAccumulationMeets)
{
    for(const auto computeType : {data_objects::DataType::UNSET,
                                  data_objects::DataType::FLOAT,
                                  data_objects::DataType::HALF,
                                  data_objects::DataType::BFLOAT16})
    {
        auto spec = baseGraph();
        spec.computeType = computeType;
        EXPECT_TRUE(matchGraph(arch(), spec).has_value())
            << data_objects::EnumNameDataType(computeType);
    }
}

TEST_P(TestAttentionDenseMatch, DeclinesEveryOtherComputeDataType)
{
    for(const auto computeType : data_objects::EnumValuesDataType())
    {
        if(computeType == data_objects::DataType::UNSET
           || computeType == data_objects::DataType::FLOAT
           || computeType == data_objects::DataType::HALF
           || computeType == data_objects::DataType::BFLOAT16)
        {
            continue;
        }
        SCOPED_TRACE(data_objects::EnumNameDataType(computeType));
        auto spec = baseGraph();
        spec.computeType = computeType;
        const auto decline = loggedDecline(arch(), spec);
        EXPECT_EQ(decline.cause, "compute_data_type") << decline.detail;
        EXPECT_NE(decline.detail.find("accumulates in FLOAT"), std::string::npos) << decline.detail;
    }
}

// =============================================================================
// The softmax scale. Both arches serve [2^-64, 2^4], bounds included; the shared
// isSupportedScale (AttentionDenseNative.cpp) says why. Every other scale declines under
// 'scale' with the range in its reason.
// =============================================================================

TEST_P(TestAttentionDenseMatch, ServesOnlyTheSupportedScaleRange)
{
    for(const float scale : {0x1p-64F, 0x1p4F})
    {
        auto spec = baseGraph();
        spec.attnScaleValue = scale;
        EXPECT_TRUE(matchGraph(arch(), spec).has_value()) << "scale " << scale;
    }

    // The float just past each bound first, then zero, negatives, far-off scales, NaN and
    // the infinities.
    for(const float scale : {0x1.fffffep-65F,
                             0x1.000002p4F,
                             0.0F,
                             -0.0F,
                             -0.5F,
                             1e-30F,
                             0x1p-65F,
                             0x1p5F,
                             1e30F,
                             std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity()})
    {
        SCOPED_TRACE(scale);
        auto spec = baseGraph();
        spec.attnScaleValue = scale;
        const auto decline = loggedDecline(arch(), spec);
        EXPECT_EQ(decline.cause, "scale") << decline.detail;
        EXPECT_NE(decline.detail.find("is outside the supported [2^-64, 2^4]"), std::string::npos)
            << decline.detail;
    }
}

// =============================================================================
// kv_segments. A split-KV build launches kv_segments times the grid plus a combine the
// dispatch does not make, so only a single segment is served. gfx942's KMD declares the
// field (default 1); gfx950's does not, so a record without it is a single-segment build.
// =============================================================================

TEST_P(TestAttentionDenseMatch, ServesOnlyASingleKvSegment)
{
    const auto spec = baseGraph();

    EXPECT_TRUE(matchesKernel(arch(), spec, makeKernel(arch(), spec))) << "kv_segments absent";

    auto single = makeKernel(arch(), spec);
    single.metadata[std::string("kv_segments")] = int64_t{1};
    EXPECT_TRUE(matchesKernel(arch(), spec, single)) << "kv_segments 1";

    for(const int64_t segments : {int64_t{2}, int64_t{4}, int64_t{0}, int64_t{-1}})
    {
        auto split = makeKernel(arch(), spec);
        split.metadata[std::string("kv_segments")] = segments;
        EXPECT_FALSE(matchesKernel(arch(), spec, split)) << "kv_segments " << segments;
    }

    auto mistyped = makeKernel(arch(), spec);
    mistyped.metadata[std::string("kv_segments")] = std::string("1");
    EXPECT_FALSE(matchesKernel(arch(), spec, mistyped)) << "kv_segments as a string";
}

// =============================================================================
// 32-bit addressing. K/V is bounded in bytes, Q/O in elements; O must have Q's dims, so
// the Q bound is the O bound. One head, so the sequence length carries the whole count.
//
// Beside the whole-tensor bound the stride predicate holds each slab to the kernel's own
// offsets: Q and O to fewer than 2^31 elements, K and V to a reach below 2^31 bytes that
// counts the two tiles of slack a load may run past the data. The graph match asks with the
// loosest tile (one row), so it declines the graphs no tile could serve; each candidate asks
// again with its own block_n.
// =============================================================================

TEST_P(TestAttentionDenseMatch, QueryElementBoundAdmitsTheLastGraphUnderIt)
{
    for(const int64_t headSize : {int64_t{64}, int64_t{128}})
    {
        SCOPED_TRACE("head_size " + std::to_string(headSize));
        auto spec = baseGraph();
        spec.batch = 1;
        spec.numQueryHeads = 1;
        spec.numKvHeads = 1;
        spec.headSize = headSize;

        spec.seqLenQ = OFFSET_LIMIT / headSize - 1;
        EXPECT_TRUE(matchGraph(arch(), spec).has_value()) << "2^31 - head_size elements";

        spec.seqLenQ = OFFSET_LIMIT / headSize;
        EXPECT_EQ(loggedDecline(arch(), spec).cause, "size_limit") << "exactly 2^31 elements";
    }
}

TEST_P(TestAttentionDenseMatch, KeyValueByteBoundAdmitsTheLastGraphUnderIt)
{
    // Two bytes an element, so the bound falls at half the Q/O sequence length: that pins
    // the bytes factor. A load may run two rows past the data even at the loosest tile, so
    // the last admitted length is two rows short of the whole-tensor bound: its reach is
    // exactly 2^31 bytes, and its last element starts at 2^31 - 2. One row longer, the whole
    // tensor is still under 2^31 bytes and the slab rule alone declines it.
    for(const int64_t headSize : {int64_t{64}, int64_t{128}})
    {
        SCOPED_TRACE("head_size " + std::to_string(headSize));
        auto spec = baseGraph();
        spec.batch = 1;
        spec.numQueryHeads = 1;
        spec.numKvHeads = 1;
        spec.headSize = headSize;

        spec.seqLenKv = OFFSET_LIMIT / (headSize * 2) - 2;
        EXPECT_TRUE(matchGraph(arch(), spec).has_value()) << "a reach of exactly 2^31 bytes";

        spec.seqLenKv = OFFSET_LIMIT / (headSize * 2) - 1;
        EXPECT_EQ(loggedDecline(arch(), spec).cause, "size_limit")
            << "one row past the reach, the whole tensor under 2^31 bytes";

        spec.seqLenKv = OFFSET_LIMIT / (headSize * 2);
        EXPECT_EQ(loggedDecline(arch(), spec).cause, "size_limit") << "exactly 2^31 bytes";
    }
}

TEST_P(TestAttentionDenseMatch, ACandidateServesTheLargestTiledGraphUnderEachBound)
{
    // The bounds, not the bm256/bn64 tile, are what decide at these sizes: the largest
    // tile-aligned Q and K/V under them are served end to end.
    auto query = baseGraph();
    query.batch = 1;
    query.numQueryHeads = 1;
    query.numKvHeads = 1;
    query.seqLenQ = OFFSET_LIMIT / HEAD_SIZE - 256;
    EXPECT_TRUE(matchesKernel(arch(), query, makeKernel(arch(), query)));

    // K/V: the bm256/bn64 candidate reaches two tiles of 64 rows past the data, so the
    // largest tile-aligned length is 128 rows under the whole-tensor bound.
    GraphSpec keyValue = query;
    keyValue.seqLenQ = SEQ;
    keyValue.seqLenKv = OFFSET_LIMIT / (HEAD_SIZE * 2) - 128;
    EXPECT_TRUE(matchesKernel(arch(), keyValue, makeKernel(arch(), keyValue)));
}

namespace
{

/// makeKernel's record with its block_n moved to @p blockN (block_m stays 256).
KernelDefinition
    makeKernelWithTile(const AttentionDenseTestArch& arch, const GraphSpec& spec, int64_t blockN)
{
    auto kernel = makeKernel(arch, spec);
    kernel.metadata[std::string("block_n")] = blockN;
    return kernel;
}

} // namespace

TEST_P(TestAttentionDenseMatch, TheKeyValueSlackFollowsTheCandidatesOwnTile)
{
    // The graph match cannot know the tile, so it asks with one-row tiles; each candidate asks
    // again with its own block_n, whose reach is the padded length plus two of its tiles.
    auto spec = baseGraph();
    spec.batch = 1;
    spec.numQueryHeads = 1;
    spec.numKvHeads = 1;
    spec.seqLenQ = SEQ;

    // 2^23 - 128 rows: a multiple of every tile below. The reach is 2^31 bytes at a tile of
    // 64, which fits; the 128-row tile reaches 128 rows further and does not.
    spec.seqLenKv = OFFSET_LIMIT / (HEAD_SIZE * 2) - 128;
    ASSERT_TRUE(matchGraph(arch(), spec).has_value());
    EXPECT_TRUE(matchesKernel(arch(), spec, makeKernelWithTile(arch(), spec, 32)));
    EXPECT_TRUE(matchesKernel(arch(), spec, makeKernelWithTile(arch(), spec, 64)));
    EXPECT_FALSE(matchesKernel(arch(), spec, makeKernelWithTile(arch(), spec, 128)));

    // 2^23 - 64 rows: the graph is served (a one-row tile reaches two rows), the 32-row tile
    // fits, and the 64-row tile reaches 64 rows past the limit.
    spec.seqLenKv = OFFSET_LIMIT / (HEAD_SIZE * 2) - 64;
    ASSERT_TRUE(matchGraph(arch(), spec).has_value());
    EXPECT_TRUE(matchesKernel(arch(), spec, makeKernelWithTile(arch(), spec, 32)));
    EXPECT_FALSE(matchesKernel(arch(), spec, makeKernelWithTile(arch(), spec, 64)));
}

// Products past int64_t. An unchecked product would wrap to INT64_MIN, under any bound,
// so each graph is declined only because checkedProduct reports the overflow. Each
// operand's own strides fit, so the layout check passes first.

TEST_P(TestAttentionDenseMatch, DeclinesAKeyValueByteCountPastInt64)
{
    // B 2^24, Skv 2^32, D64: K/V is 2^63 bytes; Q is 2^30 elements.
    auto spec = baseGraph();
    spec.batch = int64_t{1} << 24;
    spec.numQueryHeads = 1;
    spec.numKvHeads = 1;
    spec.seqLenQ = 1;
    spec.seqLenKv = int64_t{1} << 32;
    spec.headSize = 64;
    EXPECT_EQ(loggedDecline(arch(), spec).cause, "size_limit");
}

TEST_P(TestAttentionDenseMatch, DeclinesAQueryElementCountPastInt64)
{
    // B 2^23, Sq 2^34, D64: Q is 2^63 elements; K/V is 2^30 bytes.
    auto spec = baseGraph();
    spec.batch = int64_t{1} << 23;
    spec.numQueryHeads = 1;
    spec.numKvHeads = 1;
    spec.seqLenQ = int64_t{1} << 34;
    spec.seqLenKv = 1;
    spec.headSize = 64;
    EXPECT_EQ(loggedDecline(arch(), spec).cause, "size_limit");
}

// =============================================================================
// Per-operand rules. The kernel takes one shape and no strides, so each operand must
// agree with the others and be dense BSHD; one bad operand declines the graph.
// =============================================================================

TEST_P(TestAttentionDenseMatch, DeclinesEachOperandThatDisagreesWithTheShape)
{
    // Each row moves one axis of one operand, whose strides stay dense for its own dims, so
    // only the cross-operand clause can decline it. Q's batch is the problem's, so in the
    // "Q batch" row V, compared first, is the operand that disagrees: agreement is judged
    // against Q, not against a majority of the operands.
    struct Case
    {
        const char* name;
        int64_t uid;
        const char* role;
        std::vector<int64_t> dims;
    };
    const std::vector<Case> cases{
        {"V batch", V_UID, "V", {3, HEADS, SEQ, HEAD_SIZE}},
        {"V heads", V_UID, "V", {BATCH, 2, SEQ, HEAD_SIZE}},
        {"V more heads", V_UID, "V", {BATCH, HEADS * 2, SEQ, HEAD_SIZE}},
        {"V sequence", V_UID, "V", {BATCH, HEADS, 128, HEAD_SIZE}},
        {"V head size", V_UID, "V", {BATCH, HEADS, SEQ, 64}},
        {"K batch", K_UID, "K", {3, HEADS, SEQ, HEAD_SIZE}},
        {"K head size", K_UID, "K", {BATCH, HEADS, SEQ, 64}},
        {"O batch", O_UID, "O", {3, HEADS, SEQ, HEAD_SIZE}},
        {"O heads", O_UID, "O", {BATCH, 2, SEQ, HEAD_SIZE}},
        {"O more heads", O_UID, "O", {BATCH, HEADS * 2, SEQ, HEAD_SIZE}},
        {"O sequence", O_UID, "O", {BATCH, HEADS, 128, HEAD_SIZE}},
        {"O longer sequence", O_UID, "O", {BATCH, HEADS, SEQ * 2, HEAD_SIZE}},
        {"O head size", O_UID, "O", {BATCH, HEADS, SEQ, 64}},
        {"Q batch", Q_UID, "V", {3, HEADS, SEQ, HEAD_SIZE}},
    };
    for(const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.name);
        auto spec = baseGraph();
        spec.operand(testCase.uid).dims = testCase.dims;
        const auto decline = loggedDecline(arch(), spec);
        EXPECT_EQ(decline.cause, "shape") << decline.detail;
        // The line names the operand that broke the rule first.
        EXPECT_EQ(decline.detail.rfind(std::string(": ") + testCase.role + " (uid ", 0), 0U)
            << decline.detail;
    }
}

TEST_P(TestAttentionDenseMatch, DeclinesKeyHeadsThatDoNotDivideTheQueryHeads)
{
    // Integer division would drop the remainder heads silently.
    for(const auto& [queryHeads, kvHeads] :
        {std::pair<int64_t, int64_t>{HEADS, 3}, std::pair<int64_t, int64_t>{6, 4}})
    {
        SCOPED_TRACE(std::to_string(queryHeads) + "/" + std::to_string(kvHeads));
        auto spec = baseGraph();
        spec.numQueryHeads = queryHeads;
        spec.numKvHeads = kvHeads;
        EXPECT_EQ(loggedDecline(arch(), spec).cause, "gqa");
    }
}

// =============================================================================
// Launch-time layout rules. Each operand's strides, alignment and slab go through the stride
// predicate on both arches, whether or not the arch's kernels read them; a graph it declines
// is declined for the predicate's own reason, before any arch's packed-only rule. One operand
// at a time, beside operands that pass, so only its clause stands between the graph and the
// launch.
// =============================================================================

namespace
{

/// The cause the graph match logs for @p spec.
std::string declineCause(const AttentionDenseTestArch& arch, const GraphSpec& spec)
{
    return loggedDecline(arch, spec).cause;
}

/// A graph whose operands are not packed BSHD but are an admissible layout: served, by the
/// grid body of either arch, which reads the strides.
void expectServed(const AttentionDenseTestArch& arch, const GraphSpec& spec)
{
    EXPECT_TRUE(matchGraph(arch, spec).has_value());
    EXPECT_TRUE(matchesKernel(arch, spec, makeKernel(arch, spec)));
}

} // namespace

TEST_P(TestAttentionDenseMatch, DeclinesAZeroOrNegativeStrideOnAnAxisThatSteps)
{
    // The default graph has extents above 1 everywhere, so every axis steps.
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        for(size_t axis = 0; axis < 3; ++axis)
        {
            for(const int64_t value : {int64_t{0}, int64_t{-8}, int64_t{-1024}})
            {
                SCOPED_TRACE("uid " + std::to_string(uid) + " axis " + std::to_string(axis)
                             + " stride " + std::to_string(value));
                auto spec = baseGraph();
                auto strides = spec.stridesOf(uid);
                strides[axis] = value;
                spec.operand(uid).strides = strides;
                EXPECT_EQ(declineCause(arch(), spec), "stride_alignment");
            }
        }
    }
}

TEST_P(TestAttentionDenseMatch, DeclinesAFourElementPadAndAnyStrideOffTheEightElementGrid)
{
    // The hipDNN alignment default is 16 bytes: a stride off the 8-element grid is declined
    // whatever the kernel could tolerate.
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        for(size_t axis = 0; axis < 3; ++axis)
        {
            for(const int64_t pad : {int64_t{4}, int64_t{1}, int64_t{2}})
            {
                SCOPED_TRACE("uid " + std::to_string(uid) + " axis " + std::to_string(axis)
                             + " pad " + std::to_string(pad));
                auto spec = baseGraph();
                auto strides = spec.stridesOf(uid);
                strides[axis] += pad;
                spec.operand(uid).strides = strides;
                EXPECT_EQ(declineCause(arch(), spec), "stride_alignment");
            }
        }
    }
}

TEST_P(TestAttentionDenseMatch, DeclinesATokenStrideBelowTheHeadSize)
{
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        SCOPED_TRACE("uid " + std::to_string(uid));
        auto spec = baseGraph();
        auto strides = spec.stridesOf(uid);
        strides[2] = HEAD_SIZE - 8;
        spec.operand(uid).strides = strides;
        EXPECT_EQ(declineCause(arch(), spec), "stride_alignment");
    }
}

TEST_P(TestAttentionDenseMatch, DeclinesAHeadDimStrideOtherThanOne)
{
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        SCOPED_TRACE("uid " + std::to_string(uid));
        auto spec = baseGraph();
        auto strides = spec.stridesOf(uid);
        strides[3] = 2;
        spec.operand(uid).strides = strides;
        EXPECT_EQ(declineCause(arch(), spec), "head_dim_stride");
    }
}

TEST_P(TestAttentionDenseMatch, DeclinesAnOperandThatOverlapsItself)
{
    // Strides that are positive and on the grid but fold two index tuples onto one element:
    // a head stride below the head size, and a token stride below the heads it spans.
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        SCOPED_TRACE("uid " + std::to_string(uid));
        auto heads = baseGraph();
        auto strides = heads.stridesOf(uid);
        strides[1] = HEAD_SIZE / 2;
        heads.operand(uid).strides = strides;
        EXPECT_EQ(declineCause(arch(), heads), "overlap");

        auto tokens = baseGraph();
        strides = tokens.stridesOf(uid);
        strides[2] = HEAD_SIZE;
        tokens.operand(uid).strides = strides;
        EXPECT_EQ(declineCause(arch(), tokens), "overlap") << "token rows over the head rows";
    }
}

TEST_P(TestAttentionDenseMatch, DeclinesABaseThatIsNotSixteenByteAligned)
{
    // The match's own alignment clause (the operand totals, before any layout is judged) holds
    // every base to the 16 bytes the kernels declare on their pointer parameters, so it is the
    // cause here; the predicate's `pointer_alignment` is the same rule's second line, reached
    // where no match ran (prepare, a saved plan).
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        for(const int64_t alignment : {int64_t{1}, int64_t{2}, int64_t{4}, int64_t{8}})
        {
            SCOPED_TRACE("uid " + std::to_string(uid) + " alignment " + std::to_string(alignment));
            auto spec = baseGraph();
            spec.operand(uid).alignment = alignment;
            EXPECT_EQ(declineCause(arch(), spec), "alignment");
        }
    }

    // A coarser declared alignment is fine.
    auto coarse = baseGraph();
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        coarse.operand(uid).alignment = 256;
    }
    EXPECT_TRUE(matchGraph(arch(), coarse).has_value());
}

TEST_P(TestAttentionDenseMatch, DeclinesAStridedGraphThatAlsoUnderDeclaresAlignment)
{
    // A head-major Q (a layout the grid bodies read through their strides) is served when its
    // base declares 16 bytes or more and declined with the alignment cause, not served and not
    // declined for its layout, when it declares less. The same holds for a padded-row graph.
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        for(const SdpaLayout layout : {SdpaLayout::BHSD, SdpaLayout::PADDED_BSHD})
        {
            SCOPED_TRACE("uid " + std::to_string(uid) + " layout "
                         + std::to_string(static_cast<int>(layout)));
            auto strided = baseGraph();
            strided.operand(uid).layout = layout;
            strided.operand(uid).alignment = 16;
            expectServed(arch(), strided);

            for(const int64_t alignment : {int64_t{8}, int64_t{24}})
            {
                SCOPED_TRACE("alignment " + std::to_string(alignment));
                strided.operand(uid).alignment = alignment;
                const auto decline = loggedDecline(arch(), strided);
                EXPECT_EQ(decline.cause, "alignment") << decline.detail;
            }
        }
    }
}

TEST_P(TestAttentionDenseMatch, AnAxisOfExtentOneIsExemptWhateverItsStride)
{
    // One batch, one KV head: the batch stride of every operand and K's and V's head stride
    // multiply an index that is always 0, so a stride of 0, a negative one or one off the
    // grid is no defect there. Both arches serve it (the canonical strides are the packed
    // ones).
    auto spec = baseGraph();
    spec.batch = 1;
    spec.numKvHeads = 1;
    const std::map<int64_t, int64_t> batchStride{
        {Q_UID, 0}, {K_UID, -7}, {V_UID, 12345}, {O_UID, 12345}};
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        auto strides = spec.stridesOf(uid);
        strides[0] = batchStride.at(uid);
        if(uid == K_UID || uid == V_UID)
        {
            strides[1] = 3;
        }
        spec.operand(uid).strides = strides;
    }
    EXPECT_TRUE(matchGraph(arch(), spec).has_value());
}

TEST_P(TestAttentionDenseMatch, DeclinesAStrideOf2p31OrMoreOnEveryAxis)
{
    // The kernel takes each stride as an i32 argument: 2^31 would wrap to negative.
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        for(size_t axis = 0; axis < 3; ++axis)
        {
            SCOPED_TRACE("uid " + std::to_string(uid) + " axis " + std::to_string(axis));
            auto spec = baseGraph();
            auto strides = spec.stridesOf(uid);
            strides[axis] = OFFSET_LIMIT;
            spec.operand(uid).strides = strides;
            EXPECT_EQ(declineCause(arch(), spec), "size_limit");
        }
    }
}

TEST_P(TestAttentionDenseMatch, ANonPackedLayoutIsServed)
{
    // One operand at a time, beside operands that pass their own clauses. Each layout is an
    // admissible one (aligned, positive, non-overlapping strides): the grid body of both arches
    // reads the strides and serves it.
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        SCOPED_TRACE("uid " + std::to_string(uid));
        auto bhsd = baseGraph();
        bhsd.operand(uid).layout = SdpaLayout::BHSD;
        expectServed(arch(), bhsd);

        // One padded batch: dense in every other axis.
        auto paddedBatch = baseGraph();
        auto strides = paddedBatch.stridesOf(uid);
        strides[0] += 8;
        paddedBatch.operand(uid).strides = strides;
        expectServed(arch(), paddedBatch);

        // Rows padded by 8 elements, the layout a row-aligned allocator gives.
        auto paddedRows = baseGraph();
        paddedRows.operand(uid).layout = SdpaLayout::PADDED_BSHD;
        expectServed(arch(), paddedRows);
    }
}

TEST_P(TestAttentionDenseMatch, PackedQkvAndGroupedQueryLayoutsAreServed)
{
    // Q, K and V cut out of one [B, S, 3, H, D] buffer: every token stride is 3 * H * D and
    // every batch stride S times that, the head stride D. O is its own packed tensor.
    auto qkv = baseGraph();
    for(const int64_t uid : {Q_UID, K_UID, V_UID})
    {
        qkv.operand(uid).strides = std::vector<int64_t>{
            SEQ * 3 * HEADS * HEAD_SIZE, HEAD_SIZE, 3 * HEADS * HEAD_SIZE, 1};
    }
    expectServed(arch(), qkv);

    // Eight query heads over two KV heads, every operand head-major.
    auto grouped = baseGraph();
    grouped.numQueryHeads = 8;
    grouped.numKvHeads = 2;
    grouped.setEveryLayout(SdpaLayout::BHSD);
    expectServed(arch(), grouped);
}

TEST_P(TestAttentionDenseMatch, DeclinesEachOperandOfAnotherRankOrAZeroExtent)
{
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        SCOPED_TRACE("uid " + std::to_string(uid));
        auto rank3 = baseGraph();
        const auto dims = rank3.dimsOf(uid);
        rank3.operand(uid).dims = std::vector<int64_t>{dims[0], dims[2], dims[1] * dims[3]};
        EXPECT_EQ(loggedDecline(arch(), rank3).cause, "operand");

        for(size_t axis = 0; axis < 4; ++axis)
        {
            auto empty = baseGraph();
            auto zeroed = dims;
            zeroed[axis] = 0;
            empty.operand(uid).dims = zeroed;
            EXPECT_EQ(loggedDecline(arch(), empty).cause, "operand") << "axis " << axis;
        }
    }
}

TEST_P(TestAttentionDenseMatch, DeclinesEachOperandDeclaringTooLittleAlignment)
{
    // Both arches declare q/k/v/o `align 16`, so both hold every operand to 16. A graph that
    // declares no alignment already gets 16 from the schema, so only an explicit
    // under-declaration is turned away. 24 is turned away too: it clears 16 but is not a
    // multiple of it, so only every other such pointer would land right.
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        SCOPED_TRACE("uid " + std::to_string(uid));
        for(const int64_t bytes : {1, 2, 4, 8, 24})
        {
            auto spec = baseGraph();
            spec.alignmentUid = uid;
            spec.alignmentBytes = bytes;
            const auto decline = loggedDecline(arch(), spec);
            EXPECT_EQ(decline.cause, "alignment") << bytes << ": " << decline.detail;
        }

        for(const int64_t bytes : {16, 32, 256})
        {
            auto spec = baseGraph();
            spec.alignmentUid = uid;
            spec.alignmentBytes = bytes;
            EXPECT_TRUE(matchGraph(arch(), spec).has_value()) << bytes;
        }
    }
}

TEST_P(TestAttentionDenseMatch, DeclinesMixedOperandDataTypes)
{
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        SCOPED_TRACE("uid " + std::to_string(uid));
        auto spec = baseGraph();
        spec.operand(uid).dataType = data_objects::DataType::HALF;
        EXPECT_EQ(loggedDecline(arch(), spec).cause, "data_type");
    }

    // The epilogue writes the inputs' element width, so a wider O is the wrong format too.
    auto fp32Output = baseGraph();
    fp32Output.operand(O_UID).dataType = data_objects::DataType::FLOAT;
    EXPECT_EQ(loggedDecline(arch(), fp32Output).cause, "data_type");

    // One type throughout, but not one the kernel is built for.
    auto fp32 = baseGraph();
    fp32.dataType = data_objects::DataType::FLOAT;
    EXPECT_EQ(loggedDecline(arch(), fp32).cause, "data_type");

    auto fp16 = baseGraph();
    fp16.dataType = data_objects::DataType::HALF;
    EXPECT_TRUE(matchGraph(arch(), fp16).has_value());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestAttentionDenseMatch,
                         ::testing::Values(&GFX942_ATTENTION_DENSE, &GFX950_ATTENTION_DENSE),
                         attentionDenseArchName);

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
