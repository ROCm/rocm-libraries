// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

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
 * The additive-bias cases at the end run on the arches that serve one, which is both
 * (biasServingArches()), so this file is the only home for them. A bias case added to a
 * per-arch suite would check on one arch a rule that governs both.
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
    // the bytes factor.
    for(const int64_t headSize : {int64_t{64}, int64_t{128}})
    {
        SCOPED_TRACE("head_size " + std::to_string(headSize));
        auto spec = baseGraph();
        spec.batch = 1;
        spec.numQueryHeads = 1;
        spec.numKvHeads = 1;
        spec.headSize = headSize;

        spec.seqLenKv = OFFSET_LIMIT / (headSize * 2) - 1;
        EXPECT_TRUE(matchGraph(arch(), spec).has_value()) << "2^31 - 2 * head_size bytes";

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

    GraphSpec keyValue = query;
    keyValue.seqLenQ = SEQ;
    keyValue.seqLenKv = OFFSET_LIMIT / (HEAD_SIZE * 2) - 64;
    EXPECT_TRUE(matchesKernel(arch(), keyValue, makeKernel(arch(), keyValue)));
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

TEST_P(TestAttentionDenseMatch, DeclinesEachOperandThatIsNotDenseBshd)
{
    // One operand at a time, beside operands that pass their own clauses, so only its clause
    // stands between the graph and a launch that reads wrong elements in bounds.
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        SCOPED_TRACE("uid " + std::to_string(uid));
        auto spec = baseGraph();
        spec.operand(uid).layout = SdpaLayout::BHSD;
        EXPECT_EQ(loggedDecline(arch(), spec).cause, "layout");

        // One padded batch: dense in every other axis.
        auto padded = baseGraph();
        auto strides = padded.stridesOf(uid);
        strides[0] += 8;
        padded.operand(uid).strides = strides;
        EXPECT_EQ(loggedDecline(arch(), padded).cause, "layout");
    }
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

// =============================================================================
// Features the packs decline. Each is one rule of the shared graph match, so a decline
// that regressed would regress on both arches. Every case reads the cause the matcher
// logged, so a graph declined for an earlier, unrelated reason cannot pass for the
// feature under test.
// =============================================================================

TEST_P(TestAttentionDenseMatch, DeclinesAttentionSinks)
{
    auto spec = baseGraph();
    spec.sinkTokenUid = EXTRA_UID;
    EXPECT_EQ(loggedDecline(arch(), spec).cause, "sinks");
}

TEST_P(TestAttentionDenseMatch, DeclinesBlockSparseMask)
{
    auto spec = baseGraph();
    spec.blockMaskUid = EXTRA_UID;
    EXPECT_EQ(loggedDecline(arch(), spec).cause, "block_mask");
}

TEST_P(TestAttentionDenseMatch, DeclinesFp8Descale)
{
    auto spec = baseGraph();
    spec.descaleQUid = EXTRA_UID;
    EXPECT_EQ(loggedDecline(arch(), spec).cause, "fp8");
}

TEST_P(TestAttentionDenseMatch, DeclinesAlibiMask)
{
    auto spec = baseGraph();
    spec.alibiMask = true;
    EXPECT_EQ(loggedDecline(arch(), spec).cause, "alibi_mask");
}

TEST_P(TestAttentionDenseMatch, DeclinesPaddingMask)
{
    auto spec = baseGraph();
    spec.paddingMask = true;
    EXPECT_EQ(loggedDecline(arch(), spec).cause, "padding_mask");
}

TEST_P(TestAttentionDenseMatch, DeclinesNonAutoImplementationHint)
{
    auto spec = baseGraph();
    spec.implementation = data_objects::AttentionImplementation::COMPOSITE;
    EXPECT_EQ(loggedDecline(arch(), spec).cause, "implementation");
}

TEST_P(TestAttentionDenseMatch, DeclinesPagedKv)
{
    auto spec = baseGraph();
    spec.pageTableKUid = EXTRA_UID;
    EXPECT_EQ(loggedDecline(arch(), spec).cause, "paged_kv");
}

TEST_P(TestAttentionDenseMatch, DeclinesDropout)
{
    auto spec = baseGraph();
    spec.dropoutProbability = 0.1F;
    EXPECT_EQ(loggedDecline(arch(), spec).cause, "dropout");
}

/// The kernel takes the softmax scale as a host value, so a device-resident scale tensor
/// has nowhere to be read from.
TEST_P(TestAttentionDenseMatch, DeclinesDeviceResidentScaleTensor)
{
    auto spec = baseGraph();
    spec.scaleTensorUid = EXTRA_UID;
    EXPECT_EQ(loggedDecline(arch(), spec).cause, "scale_tensor");
}

TEST_P(TestAttentionDenseMatch, DeclinesVarlen)
{
    auto spec = baseGraph();
    spec.seqLenQUid = EXTRA_UID;
    EXPECT_EQ(loggedDecline(arch(), spec).cause, "varlen");
}

/// Cumulative lengths on each side alone. padding_mask stays false in the base graph, so
/// only the varlen rule can reject these.
TEST_P(TestAttentionDenseMatch, DeclinesCuSeqLenOnEachSide)
{
    ASSERT_TRUE(matchGraph(arch(), baseGraph()).has_value()) << "control: the base graph serves";

    auto q = baseGraph();
    q.cuSeqLenQUid = EXTRA_UID;
    EXPECT_EQ(loggedDecline(arch(), q).cause, "varlen");

    auto kv = baseGraph();
    kv.cuSeqLenKvUid = EXTRA_UID;
    EXPECT_EQ(loggedDecline(arch(), kv).cause, "varlen");
}

/// Both arches serve a bias, so an attn_mask uid naming no tensor of the graph is declined
/// by the bias reader for the dangling uid, not as an unserved feature.
TEST_P(TestAttentionDenseMatch, DeclinesAnAttnMaskUidNamingNoTensor)
{
    auto spec = baseGraph();
    spec.attnMaskUid = EXTRA_UID;
    const auto decline = loggedDecline(arch(), spec);
    EXPECT_EQ(decline.cause, "attn_mask") << decline.detail;
    EXPECT_NE(decline.detail.find("uid 99"), std::string::npos) << decline.detail;
}

/// An fp32-operand MMA is a computation this kernel never performs.
TEST_P(TestAttentionDenseMatch, DeclinesMmaCoreModeFloat)
{
    for(const auto dataType : {data_objects::DataType::BFLOAT16, data_objects::DataType::HALF})
    {
        SCOPED_TRACE(std::string("graph ") + data_objects::EnumNameDataType(dataType));
        auto spec = baseGraph();
        spec.dataType = dataType;
        spec.mmaCoreMode = data_objects::DataType::FLOAT;
        EXPECT_EQ(loggedDecline(arch(), spec).cause, "mma_core_mode");
    }
}

TEST_P(TestAttentionDenseMatch, DeclinesEveryFp8MmaCoreMode)
{
    for(const auto mode : {data_objects::DataType::FP8_E4M3,
                           data_objects::DataType::FP8_E5M2,
                           data_objects::DataType::FP8_E8M0,
                           data_objects::DataType::FP8_E4M3_FNUZ,
                           data_objects::DataType::FP8_E5M2_FNUZ})
    {
        SCOPED_TRACE(data_objects::EnumNameDataType(mode));
        auto spec = baseGraph();
        spec.mmaCoreMode = mode;
        EXPECT_EQ(loggedDecline(arch(), spec).cause, "mma_core_mode");
    }
}

/// mma_core_mode is the MMA operand precision, and this kernel's operands are the graph's
/// fp16/bf16 inputs. HALF is also what the cuDNN-compat shim sets whenever the caller
/// leaves the field unset, so declining it would decline every shim graph. Each graph is
/// matched against the kernel built for its own dtype.
TEST_P(TestAttentionDenseMatch, AcceptsTheSixteenBitMmaCoreModes)
{
    for(const auto dataType : {data_objects::DataType::BFLOAT16, data_objects::DataType::HALF})
    {
        for(const auto mode : {data_objects::DataType::UNSET,
                               data_objects::DataType::HALF,
                               data_objects::DataType::BFLOAT16})
        {
            SCOPED_TRACE(std::string("graph ") + data_objects::EnumNameDataType(dataType)
                         + ", mma_core_mode " + data_objects::EnumNameDataType(mode));
            auto spec = baseGraph();
            spec.dataType = dataType;
            spec.mmaCoreMode = mode;
            auto kernel = makeKernel(arch(), spec);
            kernel.metadata[std::string("dtype")] = dataType == data_objects::DataType::HALF
                                                        ? std::string("FP16")
                                                        : std::string("BF16");
            EXPECT_TRUE(matchesKernel(arch(), spec, kernel));
        }
    }
}

/// The cases above decline each feature through one spelling. Every other uid that
/// requests the same feature is set here on its own, so dropping any one of them from its
/// decline is seen.
TEST_P(TestAttentionDenseMatch, DeclinesEveryOtherSpellingOfEachDeclinedFeature)
{
    using UidField = std::optional<int64_t> GraphSpec::*;
    const std::vector<std::pair<const char*, UidField>> spellings{
        {"seq_len_kv", &GraphSpec::seqLenKvUid},
        {"cu_seq_len_q", &GraphSpec::cuSeqLenQUid},
        {"cu_seq_len_kv", &GraphSpec::cuSeqLenKvUid},
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
        auto spec = baseGraph();
        spec.*field = EXTRA_UID;
        EXPECT_FALSE(loggedDecline(arch(), spec).cause.empty());
    }

    // Paged KV's scalar spelling.
    auto maxSeqLenKv = baseGraph();
    maxSeqLenKv.maxSeqLenKv = static_cast<int32_t>(SEQ);
    EXPECT_EQ(loggedDecline(arch(), maxSeqLenKv).cause, "paged_kv");
}

// =============================================================================
// The graph shape: one SDPA-forward node over four dense rank-4 BSHD operands
// =============================================================================

TEST_P(TestAttentionDenseMatch, DeclinesMultiNodeGraph)
{
    auto spec = baseGraph();
    spec.twoNodes = true;
    EXPECT_EQ(loggedDecline(arch(), spec).cause, "node");
}

/// One node, but not an SDPA one. DeclinesMultiNodeGraph covers the node count, this
/// covers the node's type, without which its attributes would be read as SDPA's. A graph
/// with no SDPA-forward node is not this engine's to refuse, so no decline is logged for
/// it (see LogsNoDeclineForAGraphWithNoSdpaForwardNode) and only the refusal is asserted.
TEST_P(TestAttentionDenseMatch, DeclinesANonSdpaNode)
{
    auto spec = baseGraph();
    spec.pointwiseOnly = true;
    EXPECT_FALSE(matchGraph(arch(), spec).has_value());
}

/// The kernel bakes BSHD strides and takes no stride kernargs, so BHSD reads wrong
/// elements. DeclinesEachOperandThatIsNotDenseBshd does this one operand at a time, which
/// is the stricter form; this is the whole-graph spelling a caller would actually send.
TEST_P(TestAttentionDenseMatch, DeclinesBhsdLayout)
{
    auto spec = baseGraph();
    spec.setEveryLayout(SdpaLayout::BHSD);
    EXPECT_EQ(loggedDecline(arch(), spec).cause, "layout");
}

/// Token-major like BSHD but head fastest-varying: same row and batch pitch, with the head
/// and element strides exchanged.
TEST_P(TestAttentionDenseMatch, DeclinesBsdhLayout)
{
    auto spec = baseGraph();
    spec.operand(Q_UID).layout = SdpaLayout::BSDH;
    EXPECT_EQ(loggedDecline(arch(), spec).cause, "layout");
}

/// A single-head tensor is byte-identically BSHD and BHSD, so a strict compare would
/// decline a graph the kernel serves perfectly and empty the whole catalog.
TEST_P(TestAttentionDenseMatch, AcceptsSingleHeadUnderEitherStrideSpelling)
{
    auto bshd = baseGraph();
    bshd.numQueryHeads = 1;
    bshd.numKvHeads = 1;
    EXPECT_TRUE(matchGraph(arch(), bshd).has_value());

    auto bhsd = bshd;
    bhsd.setEveryLayout(SdpaLayout::BHSD);
    EXPECT_TRUE(matchGraph(arch(), bhsd).has_value());
}

TEST_P(TestAttentionDenseMatch, DeclinesGraphWithNoStrides)
{
    auto spec = baseGraph();
    spec.omitStrides = true;
    EXPECT_EQ(loggedDecline(arch(), spec).cause, "operand");
}

/// dims is optional in the schema, so a verified graph can carry an operand without one.
/// Every extent the matcher reads comes from it.
TEST_P(TestAttentionDenseMatch, DeclinesAnOperandWithNoDims)
{
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        SCOPED_TRACE("uid " + std::to_string(uid));
        auto spec = baseGraph();
        spec.omitDimsUid = uid;
        EXPECT_EQ(loggedDecline(arch(), spec).cause, "operand");
    }
}

/// The node names a uid the graph holds no tensor for: there is nothing to read a shape
/// from, and nothing for that operand's pointer argument to address.
TEST_P(TestAttentionDenseMatch, DeclinesAGraphMissingAnOperandTensor)
{
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        SCOPED_TRACE("uid " + std::to_string(uid));
        auto spec = baseGraph();
        spec.omitTensorUid = uid;
        EXPECT_EQ(loggedDecline(arch(), spec).cause, "operand");
    }
}

/// A zero batch or a zero-length KV sequence. The fixture's strides stay consistent with
/// the zero and no bound is approached, so the extent check is the only one that fails.
TEST_P(TestAttentionDenseMatch, DeclinesAZeroExtent)
{
    auto noBatch = baseGraph();
    noBatch.batch = 0;
    EXPECT_EQ(loggedDecline(arch(), noBatch).cause, "operand");

    auto noKeys = baseGraph();
    noKeys.seqLenKv = 0;
    EXPECT_EQ(loggedDecline(arch(), noKeys).cause, "operand");
}

/// A virtual tensor has no device buffer for the launch to hand the kernel.
TEST_P(TestAttentionDenseMatch, DeclinesAVirtualOperand)
{
    auto spec = baseGraph();
    spec.virtualUid = K_UID;
    EXPECT_EQ(loggedDecline(arch(), spec).cause, "operand");
}

/// A runtime pass-by-value tensor is a scalar, not a buffer the ABI takes a pointer to.
TEST_P(TestAttentionDenseMatch, DeclinesAPassByValueOperand)
{
    auto spec = baseGraph();
    spec.passByValueUid = V_UID;
    EXPECT_EQ(loggedDecline(arch(), spec).cause, "operand");
}

/// A ragged offset tensor packs the sequences back to back, and the kernel indexes every
/// operand as dense (B, S, H, D).
TEST_P(TestAttentionDenseMatch, DeclinesARaggedOperand)
{
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        SCOPED_TRACE("uid " + std::to_string(uid));
        auto spec = baseGraph();
        spec.raggedUid = uid;
        EXPECT_EQ(loggedDecline(arch(), spec).cause, "operand");
    }
}

// =============================================================================
// One stride axis at a time. hasBshdStrides ANDs four per-axis clauses, and the whole
// layouts above each break two at once (BHSD: head and sequence; padded: sequence and
// batch). Each case below perturbs exactly one axis of one operand at B=2, H=4, so no
// axis is unit-extent and exempt, and that axis's clause is the only one that can decline
// it. Each opens with the unperturbed graph as a control, which is what keeps the decline
// attributable here in place of the logged cause the cases above assert.
// =============================================================================

namespace
{

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

} // namespace

TEST_P(TestAttentionDenseMatch, DeclinesPaddedSequenceStride)
{
    // Far likelier in a real graph than a full transpose: BSHD axis ORDER, but token rows
    // padded past heads * headSize. The kernel bakes heads * headSize as the row stride,
    // so every row after the first is read off by the accumulated padding.
    //
    // batch = 1 makes the batch axis unit-extent and therefore exempt from the stride
    // compare, which leaves the sequence-stride clause as the only thing that can catch
    // this. Heads and sequence both stay above 1 so neither of those is exempted away.
    auto packed = baseGraph();
    packed.batch = 1;
    EXPECT_TRUE(matchGraph(arch(), packed).has_value());

    auto padded = packed;
    padded.operand(Q_UID).layout = SdpaLayout::PADDED_BSHD;
    EXPECT_FALSE(matchGraph(arch(), padded).has_value());
}

TEST_P(TestAttentionDenseMatch, DeclinesPaddedOutputSequenceStride)
{
    // Exact-equality rule, same as the inputs: a dense-but-padded row stride is not the
    // layout the epilogue bakes. Batch is 1 so the batch axis is unit-extent and exempt,
    // leaving the sequence stride as the only clause that can reject this graph.
    auto spec = baseGraph();
    spec.batch = 1;
    EXPECT_TRUE(matchGraph(arch(), spec).has_value());

    auto padded = spec;
    padded.operand(O_UID).layout = SdpaLayout::PADDED_BSHD;
    EXPECT_FALSE(matchGraph(arch(), padded).has_value());
}

TEST_P(TestAttentionDenseMatch, DeclinesAPaddedBatchStrideAlone)
{
    // A query sliced out of a larger allocation: rows and heads packed, batches spaced one
    // row further apart. The kernel bakes S * H * D as the batch pitch, so every batch
    // after the first is read one row early.
    const auto spec = baseGraph();
    EXPECT_TRUE(matchGraph(arch(), spec).has_value());

    auto padded = spec;
    padded.operand(Q_UID).strides
        = packedStridesWith(BATCH_STRIDE, SEQ * HEADS * HEAD_SIZE + HEADS * HEAD_SIZE);
    EXPECT_FALSE(matchGraph(arch(), padded).has_value());
}

TEST_P(TestAttentionDenseMatch, DeclinesAKeyValueCacheViewByItsBatchStride)
{
    // K and V as views of a cache allocated for twice the sequence: each batch's rows are
    // packed, but batches sit a whole cache length apart.
    const auto spec = baseGraph();
    EXPECT_TRUE(matchGraph(arch(), spec).has_value());

    const auto cacheView = packedStridesWith(BATCH_STRIDE, 2 * SEQ * HEADS * HEAD_SIZE);
    auto keyView = spec;
    keyView.operand(K_UID).strides = cacheView;
    EXPECT_FALSE(matchGraph(arch(), keyView).has_value());

    auto valueView = spec;
    valueView.operand(V_UID).strides = cacheView;
    EXPECT_FALSE(matchGraph(arch(), valueView).has_value());
}

TEST_P(TestAttentionDenseMatch, DeclinesABroadcastBatchStride)
{
    // Batch stride 0: one K broadcast across the batch. The kernel would read a distinct
    // K per batch from memory that holds one.
    const auto spec = baseGraph();
    EXPECT_TRUE(matchGraph(arch(), spec).has_value());

    auto broadcast = spec;
    broadcast.operand(K_UID).strides = packedStridesWith(BATCH_STRIDE, 0);
    EXPECT_FALSE(matchGraph(arch(), broadcast).has_value());
}

TEST_P(TestAttentionDenseMatch, DeclinesAHeadStrideAlone)
{
    // Rows packed at H * D, heads at a 2 * D pitch: the heads overlap one another.
    const auto spec = baseGraph();
    EXPECT_TRUE(matchGraph(arch(), spec).has_value());

    auto overlapping = spec;
    overlapping.operand(K_UID).strides = packedStridesWith(HEAD_STRIDE, 2 * HEAD_SIZE);
    EXPECT_FALSE(matchGraph(arch(), overlapping).has_value());
}

TEST_P(TestAttentionDenseMatch, DeclinesABroadcastHeadStride)
{
    // Head stride 0: the MQA-expanded view, one KV head presented as H of them. The
    // kernel would read H distinct heads from memory that holds one.
    const auto spec = baseGraph();
    EXPECT_TRUE(matchGraph(arch(), spec).has_value());

    auto expanded = spec;
    expanded.operand(K_UID).strides = packedStridesWith(HEAD_STRIDE, 0);
    EXPECT_FALSE(matchGraph(arch(), expanded).has_value());
}

TEST_P(TestAttentionDenseMatch, DeclinesANonUnitElementStrideAlone)
{
    // head_size stride 2 with every other axis packed: legal as a stride vector, and read
    // by the kernel as if contiguous.
    const auto spec = baseGraph();
    EXPECT_TRUE(matchGraph(arch(), spec).has_value());

    auto strided = spec;
    strided.operand(Q_UID).strides = packedStridesWith(ELEMENT_STRIDE, 2);
    EXPECT_FALSE(matchGraph(arch(), strided).has_value());
}

TEST_P(TestAttentionDenseMatch, DeclinesAnOperandOfAnotherRank)
{
    // Every axis index the matcher reads assumes rank 4. A rank-5 K carries all four of
    // them and more, so only the rank check stands between it and a match; rank 3 would
    // have its missing axis read out of bounds.
    const auto packed = sdpaStrides(SdpaLayout::BSHD, HEADS, SEQ, HEAD_SIZE);
    const std::vector<int64_t> rank4Dims{BATCH, HEADS, SEQ, HEAD_SIZE};
    const std::vector<int64_t> rank5Dims{BATCH, HEADS, SEQ, HEAD_SIZE, 1};
    const std::vector<int64_t> rank5Strides{packed.at(0), packed.at(1), packed.at(2), 1, 1};

    auto control = baseGraph();
    control.operand(K_UID).dims = rank4Dims;
    control.operand(K_UID).strides = packed;
    EXPECT_TRUE(matchGraph(arch(), control).has_value());

    auto bothRank5 = control;
    bothRank5.operand(K_UID).dims = rank5Dims;
    bothRank5.operand(K_UID).strides = rank5Strides;
    EXPECT_FALSE(matchGraph(arch(), bothRank5).has_value());

    auto dimsRank5 = control;
    dimsRank5.operand(K_UID).dims = rank5Dims;
    EXPECT_FALSE(matchGraph(arch(), dimsRank5).has_value());

    auto stridesRank5 = control;
    stridesRank5.operand(K_UID).strides = rank5Strides;
    EXPECT_FALSE(matchGraph(arch(), stridesRank5).has_value());

    auto rank3 = control;
    rank3.operand(K_UID).dims = std::vector<int64_t>{BATCH, HEADS, SEQ};
    rank3.operand(K_UID).strides = std::vector<int64_t>{packed.at(0), packed.at(1), packed.at(2)};
    EXPECT_FALSE(matchGraph(arch(), rank3).has_value());
}

// =============================================================================
// What the graph match logs: one INFO line naming the cause, for a graph this engine
// could have served, and nothing at all for a graph that was never its own.
// =============================================================================

TEST_P(TestAttentionDenseMatch, LogsTheOperandAndTheBakedStridesWhenQIsNotBshd)
{
    auto spec = baseGraph();
    spec.operand(Q_UID).layout = SdpaLayout::BHSD;
    const auto decline = loggedDecline(arch(), spec);
    EXPECT_EQ(decline.cause, "layout");
    EXPECT_NE(decline.detail.find("Q (uid 1)"), std::string::npos) << decline.detail;
    // The BSHD strides for dims [2, 4, 256, 128].
    EXPECT_NE(decline.detail.find("131072, 128, 512, 1"), std::string::npos) << decline.detail;
}

TEST_P(TestAttentionDenseMatch, LogsTheOutputWhenOnlyOIsNotBshd)
{
    auto spec = baseGraph();
    spec.operand(O_UID).layout = SdpaLayout::BHSD;
    const auto decline = loggedDecline(arch(), spec);
    EXPECT_EQ(decline.cause, "layout");
    EXPECT_NE(decline.detail.find("O (uid 4)"), std::string::npos) << decline.detail;
}

TEST_P(TestAttentionDenseMatch, LogsNoDeclineForAGraphItServes)
{
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    EXPECT_TRUE(matchGraph(arch(), baseGraph()).has_value());
    EXPECT_FALSE(recorder.hasLogContaining(std::string(arch().engineName) + " declined the graph"));
}

TEST_P(TestAttentionDenseMatch, LogsNoDeclineForAGraphWithNoSdpaForwardNode)
{
    // graph_match sees every graph the catalog misses on. One with no SDPA-forward node
    // was never this engine's, so it declines without a line.
    auto spec = baseGraph();
    spec.pointwiseOnly = true;
    const auto decline = loggedDecline(arch(), spec);
    EXPECT_TRUE(decline.cause.empty()) << "[" << decline.cause << "]" << decline.detail;
}

TEST_P(TestAttentionDenseMatch, AcceptsGroupedQueryAttention)
{
    auto spec = baseGraph();
    spec.numQueryHeads = 8;
    spec.numKvHeads = 2;
    EXPECT_TRUE(matchGraph(arch(), spec).has_value());
}

TEST_P(TestAttentionDenseMatch, DeclinesAQueryBatchStrideOf2To67)
{
    // Q dims {2, 2^30, 2^30, 128}: S * H * D = 2^67, which does not fit in int64_t, and Q
    // and O carry the batch stride 0 it would wrap to beside packed head, row and element
    // strides. hasBshdStrides forms that product through checkedProduct, which removes the
    // undefined behaviour but is not observable in any verdict: every graph whose S * H * D
    // overflows also overflows B * S * H * D, which the Q bound declines.
    const std::vector<int64_t> wrappedStrides{0, HEAD_SIZE, (int64_t{1} << 30) * HEAD_SIZE, 1};
    auto spec = baseGraph();
    spec.numQueryHeads = int64_t{1} << 30;
    spec.seqLenQ = int64_t{1} << 30;
    spec.numKvHeads = 1;
    spec.operand(Q_UID).strides = wrappedStrides;
    spec.operand(O_UID).strides = wrappedStrides;
    EXPECT_FALSE(matchGraph(arch(), spec).has_value());
}

TEST_P(TestAttentionDenseMatch, DeclinesOverflowingOutputExtents)
{
    // O's extents here are positive and rank-4, so the well-formedness predicate passes
    // them through, but S * H * D for those extents is 2^63 -- one past INT64_MAX, and that
    // product is what hasBshdStrides derives O's batch stride from. As ordered, the
    // head-count compare rejects the graph before O's layout is read; the case exists to
    // keep the overflow-capable domain reachable and declined. O's strides are written out
    // as the ordinary dense BSHD spelling of the PROBLEM shape, so no stride value is what
    // rejects this graph.
    const auto spec = baseGraph();
    EXPECT_TRUE(matchGraph(arch(), spec).has_value());

    auto overflowing = spec;
    overflowing.operand(O_UID).dims
        = std::vector<int64_t>{BATCH, int64_t{1} << 16, int64_t{1} << 31, int64_t{1} << 16};
    overflowing.operand(O_UID).strides = sdpaStrides(SdpaLayout::BSHD, HEADS, SEQ, HEAD_SIZE);
    EXPECT_FALSE(matchGraph(arch(), overflowing).has_value());
}

TEST_P(TestAttentionDenseMatch, DeclinesTheGraphMatchGateCounterexample)
{
    // A graph whose bound products wrap to 0 is declined. Packed BSHD, bf16, H8, D128,
    // unmasked, Q {2^20, 8, 2^34, 128} and K/V {2^20, 8, 2^33, 128}: every stride fits,
    // and both bound products are exactly 2^64, which wraps to 0. Every length is a
    // multiple of every tile, so unchecked the whole catalog would admit it.
    auto spec = baseGraph();
    spec.batch = int64_t{1} << 20;
    spec.numQueryHeads = 8;
    spec.numKvHeads = 8;
    spec.seqLenQ = int64_t{1} << 34;
    spec.seqLenKv = int64_t{1} << 33;
    spec.leftBound = std::nullopt;
    spec.rightBound = std::nullopt;
    EXPECT_FALSE(matchGraph(arch(), spec).has_value());
}

// =============================================================================
// The mask the graph spells, including the deprecated causal booleans. These are the
// bound-independent spellings, so they hold on an arch that serves a sliding window and
// on one that declines every window alike.
// =============================================================================

TEST_P(TestAttentionDenseMatch, AcceptsNoMaskGraph)
{
    auto spec = baseGraph();
    spec.leftBound = std::nullopt;
    spec.rightBound = std::nullopt;
    EXPECT_TRUE(matchGraph(arch(), spec).has_value());
}

TEST_P(TestAttentionDenseMatch, DeclinesBottomRightCausalWhenSeqLensDiffer)
{
    // Top-left causal clamp != bottom-right when Sq != Skv: serving it is a wrong answer.
    auto spec = baseGraph();
    spec.seqLenKv = SEQ * 2;
    spec.alignment = data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    EXPECT_FALSE(matchGraph(arch(), spec).has_value());
}

TEST_P(TestAttentionDenseMatch, AcceptsBottomRightCausalWhenSeqLensMatch)
{
    // The complement: every shipped quick/SdpaFwd causal bundle sets BOTTOM_RIGHT at
    // Sq == Skv; declining it outright declines all of them.
    auto spec = baseGraph();
    spec.alignment = data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    EXPECT_TRUE(matchGraph(arch(), spec).has_value());
}

TEST_P(TestAttentionDenseMatch, DeclinesTheDeprecatedBottomRightBooleanAtUnequalSeqLens)
{
    // causal_mask_bottom_right alone, no bounds: bottom-right causal, which the top-left
    // kernel serves only at Sq == Skv. Read as top-left instead, Sq != Skv would be
    // served with the other corner's mask.
    auto spec = baseGraph();
    spec.leftBound = std::nullopt;
    spec.rightBound = std::nullopt;
    spec.causalMaskBottomRightDeprecated = true;
    spec.seqLenKv = SEQ * 2;
    EXPECT_FALSE(matchGraph(arch(), spec).has_value());
}

TEST_P(TestAttentionDenseMatch, DeclinesBothDeprecatedCausalBooleans)
{
    auto spec = baseGraph();
    spec.causalMaskDeprecated = true;
    spec.causalMaskBottomRightDeprecated = true;
    EXPECT_FALSE(matchGraph(arch(), spec).has_value());
}

TEST_P(TestAttentionDenseMatch, DeclinesDeprecatedCausalWithRightBoundBelowMinusOne)
{
    auto spec = baseGraph();
    spec.causalMaskDeprecated = true;
    spec.rightBound = -5;
    EXPECT_FALSE(matchGraph(arch(), spec).has_value());
}

TEST_P(TestAttentionDenseMatch, StillServesPlainDeprecatedCausalWithNoBound)
{
    // The control: bound-wins must not over-fire and decline ordinary deprecated-causal.
    auto spec = baseGraph();
    spec.causalMaskDeprecated = true;
    spec.leftBound = std::nullopt;
    spec.rightBound = std::nullopt;
    EXPECT_TRUE(matchGraph(arch(), spec).has_value());
}

namespace
{

/// Both spellings of one causal mask must bind identically, as causal, so neither reaches a
/// candidate the other would not: served by the causal kernel, refused by the unmasked one.
void expectOneCausalMask(const AttentionDenseTestArch& arch,
                         const GraphSpec& bounds,
                         const GraphSpec& flag)
{
    using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;
    const auto causalToken = std::string(arch.tokenPrefix) + ".causal";
    const auto boundsBinding = matchGraph(arch, bounds);
    const auto flagBinding = matchGraph(arch, flag);
    ASSERT_TRUE(boundsBinding.has_value());
    ASSERT_TRUE(flagBinding.has_value());
    EXPECT_EQ(tryGetBoundInt(*boundsBinding, causalToken), std::optional<int64_t>{1});
    EXPECT_EQ(tryGetBoundInt(*flagBinding, causalToken), std::optional<int64_t>{1});
    EXPECT_TRUE(*boundsBinding == *flagBinding);

    auto causal = makeKernel(arch, bounds);
    causal.metadata[std::string("causal")] = int64_t{1};
    auto unmasked = makeKernel(arch, bounds);
    unmasked.metadata[std::string("causal")] = int64_t{0};
    for(const GraphSpec* spec : std::initializer_list<const GraphSpec*>{&bounds, &flag})
    {
        SCOPED_TRACE(spec == &bounds ? "bounds" : "deprecated flag");
        EXPECT_TRUE(matchesKernel(arch, *spec, causal));
        EXPECT_FALSE(matchesKernel(arch, *spec, unmasked));
    }
}

} // namespace

/// left -1 / right 0 / BOTTOM_RIGHT and causal_mask_bottom_right=true with no bounds are one
/// mask. At the default Sq == Skv the corners coincide, so both are served as causal.
TEST_P(TestAttentionDenseMatch, BoundsAndTheDeprecatedFlagSpellTheSameBottomRightCausalMask)
{
    auto bounds = baseGraph();
    bounds.alignment = data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    auto flag = baseGraph();
    flag.causalMaskBottomRightDeprecated = true;
    flag.leftBound = std::nullopt;
    flag.rightBound = std::nullopt;
    expectOneCausalMask(arch(), bounds, flag);
}

TEST_P(TestAttentionDenseMatch, BoundsAndTheDeprecatedFlagSpellTheSameTopLeftCausalMask)
{
    // left -1 / right 0 / TOP_LEFT and causal_mask=true with no bounds are one mask.
    const auto bounds = baseGraph();
    auto flag = baseGraph();
    flag.causalMaskDeprecated = true;
    flag.leftBound = std::nullopt;
    flag.rightBound = std::nullopt;
    expectOneCausalMask(arch(), bounds, flag);
}

TEST_P(TestAttentionDenseMatch, DeclinesCausalWithNonZeroRightBound)
{
    // A graph with causal_mask=true and right_bound > 0 also describes a shape the kernel
    // cannot serve correctly. The right_bound wins over the deprecated causal boolean.
    auto spec = baseGraph();
    spec.causalMaskDeprecated = true;
    spec.leftBound = std::nullopt;
    spec.rightBound = 64;
    EXPECT_FALSE(matchGraph(arch(), spec).has_value());
}

TEST_P(TestAttentionDenseMatch, ReadsDeprecatedCausalWithBottomRightAlignmentAsBottomRight)
{
    // causal_mask keeps the stored alignment, so this is bottom-right causal: served at
    // Sq == Skv, declined where the corners differ.
    auto spec = baseGraph();
    spec.causalMaskDeprecated = true;
    spec.leftBound = std::nullopt;
    spec.rightBound = std::nullopt;
    spec.alignment = data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    EXPECT_TRUE(matchGraph(arch(), spec).has_value());

    spec.seqLenKv = SEQ * 2;
    EXPECT_FALSE(matchGraph(arch(), spec).has_value());
}

TEST_P(TestAttentionDenseMatch, AbsentAttentionScaleBindsOne)
{
    // cuDNN's default: no attn_scale_value and no scale tensor means no scaling, at every
    // head size.
    for(const int64_t headSize : {int64_t{64}, int64_t{128}})
    {
        SCOPED_TRACE(headSize);
        auto spec = baseGraph();
        spec.headSize = headSize;
        spec.attnScaleValue = std::nullopt;
        const auto bound = matchGraph(arch(), spec);
        ASSERT_TRUE(bound.has_value());
        EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(
                      *bound, std::string(arch().tokenPrefix) + ".scale_bits")
                      .value_or(-1),
                  ieee754Bits(1.0F));
    }

    // An explicit scale still wins over the default.
    auto explicitScale = baseGraph();
    explicitScale.attnScaleValue = 0.5F;
    const auto bound = matchGraph(arch(), explicitScale);
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(
                  *bound, std::string(arch().tokenPrefix) + ".scale_bits")
                  .value_or(-1),
              ieee754Bits(0.5F));
}

INSTANTIATE_TEST_SUITE_P(,
                         TestAttentionDenseMatch,
                         ::testing::Values(&GFX942_ATTENTION_DENSE, &GFX950_ATTENTION_DENSE),
                         attentionDenseArchName);

// =============================================================================
// The additive bias (attn_mask), on the arches that serve one
// =============================================================================

namespace
{

/// The int32_t element count: a bias element index below it is addressable in 32 bits.
constexpr int64_t INT32_LIMIT = int64_t{1} << 31;

/// The bound value of @p arch's bias token @p suffix (".bias", ".bias.uid", ...), or -1.
int64_t
    boundBias(const AttentionDenseTestArch& arch, const BoundTokens& bound, std::string_view suffix)
{
    return hipdnn_plugin_sdk::ingestor::tryGetBoundInt(
               bound, std::string(arch.tokenPrefix) + std::string(suffix))
        .value_or(-1);
}

/// The arches that serve an additive bias.
class TestAttentionDenseBiasMatch : public ::testing::TestWithParam<const AttentionDenseTestArch*>
{
protected:
    static const AttentionDenseTestArch& arch()
    {
        return *GetParam();
    }
};

} // namespace

TEST_P(TestAttentionDenseBiasMatch, BindsAFullBiasAndItsStrides)
{
    const auto bound = matchGraph(arch(), withBias({BATCH, HEADS, SEQ, SEQ}));
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(boundBias(arch(), *bound, ".bias"), 1);
    EXPECT_EQ(boundBias(arch(), *bound, ".bias.uid"), BIAS_UID);
    EXPECT_EQ(boundBias(arch(), *bound, ".bias.stride_batch"), HEADS * SEQ * SEQ);
    EXPECT_EQ(boundBias(arch(), *bound, ".bias.stride_head"), SEQ * SEQ);
    EXPECT_EQ(boundBias(arch(), *bound, ".bias.stride_query"), SEQ);
}

/// An extent-1 axis broadcasts through stride 0, whatever stride the tensor writes on it:
/// the kernel multiplies every axis index by its stride, and only a zero keeps a broadcast
/// axis on its one row.
TEST_P(TestAttentionDenseBiasMatch, BindsStrideZeroOnEveryBroadcastAxis)
{
    constexpr int64_t ANY = 12345;
    const auto keyRow = matchGraph(arch(), withBias({1, 1, 1, SEQ}, {{ANY, ANY, ANY, 1}}));
    ASSERT_TRUE(keyRow.has_value());
    EXPECT_EQ(boundBias(arch(), *keyRow, ".bias.stride_batch"), 0);
    EXPECT_EQ(boundBias(arch(), *keyRow, ".bias.stride_head"), 0);
    EXPECT_EQ(boundBias(arch(), *keyRow, ".bias.stride_query"), 0);

    const auto perBatch
        = matchGraph(arch(), withBias({BATCH, 1, SEQ, SEQ}, {{SEQ * SEQ, ANY, SEQ, 1}}));
    ASSERT_TRUE(perBatch.has_value());
    EXPECT_EQ(boundBias(arch(), *perBatch, ".bias.stride_batch"), SEQ * SEQ);
    EXPECT_EQ(boundBias(arch(), *perBatch, ".bias.stride_head"), 0);
    EXPECT_EQ(boundBias(arch(), *perBatch, ".bias.stride_query"), SEQ);

    const auto perHead = matchGraph(arch(), withBias({1, HEADS, SEQ, SEQ}));
    ASSERT_TRUE(perHead.has_value());
    EXPECT_EQ(boundBias(arch(), *perHead, ".bias.stride_batch"), 0);
    EXPECT_EQ(boundBias(arch(), *perHead, ".bias.stride_head"), SEQ * SEQ);
    EXPECT_EQ(boundBias(arch(), *perHead, ".bias.stride_query"), SEQ);

    // A broadcast axis's written stride is never used, so one the range check would refuse
    // on a full axis (negative, or 2^31 and above) is accepted there.
    const auto unusable
        = matchGraph(arch(), withBias({1, 1, SEQ, SEQ}, {{-1, int64_t{1} << 40, SEQ, 1}}));
    ASSERT_TRUE(unusable.has_value());
    EXPECT_EQ(boundBias(arch(), *unusable, ".bias.stride_batch"), 0);
    EXPECT_EQ(boundBias(arch(), *unusable, ".bias.stride_head"), 0);
}

TEST_P(TestAttentionDenseBiasMatch, BindsNoBiasForAGraphWithoutOne)
{
    const auto bound = matchGraph(arch(), GraphSpec{});
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(boundBias(arch(), *bound, ".bias"), 0);
    EXPECT_EQ(boundBias(arch(), *bound, ".bias.uid"), -1);
}

TEST_P(TestAttentionDenseBiasMatch, DeclinesABiasTheKernelCannotRead)
{
    struct Case
    {
        const char* what;
        GraphSpec spec;
        const char* cause;
    };
    std::vector<Case> cases;

    auto halfBias = withBias({BATCH, HEADS, SEQ, SEQ});
    biasTensor(halfBias).dataType = data_objects::DataType::BFLOAT16;
    cases.push_back({"a bf16 bias (the kernel reads fp32)", halfBias, "attn_mask"});

    cases.push_back({"a rank-3 bias", withBias({HEADS, SEQ, SEQ}), "attn_mask"});
    cases.push_back(
        {"a key extent other than Skv", withBias({BATCH, HEADS, SEQ, SEQ / 2}), "attn_mask"});
    cases.push_back(
        {"a key axis that is not unit-stride",
         withBias({BATCH, HEADS, SEQ, SEQ}, {{HEADS * SEQ * SEQ * 2, SEQ * SEQ * 2, SEQ * 2, 2}}),
         "attn_mask"});
    cases.push_back(
        {"a head extent that is neither 1 nor Hq", withBias({BATCH, 2, SEQ, SEQ}), "attn_mask"});
    cases.push_back(
        {"a query extent that is neither 1 nor Sq", withBias({BATCH, HEADS, 2, SEQ}), "attn_mask"});
    cases.push_back({"a negative stride on a full axis",
                     withBias({BATCH, HEADS, SEQ, SEQ}, {{-HEADS * SEQ * SEQ, SEQ * SEQ, SEQ, 1}}),
                     "attn_mask"});
    cases.push_back({"a stride of 2^31",
                     withBias({BATCH, HEADS, SEQ, SEQ}, {{int64_t{1} << 31, SEQ * SEQ, SEQ, 1}}),
                     "attn_mask"});
    // One less passes the stride check, so only the reach check stops it: the batch axis
    // alone then reaches past element 2^31 - 1.
    cases.push_back(
        {"a stride of 2^31 - 1 on a full axis",
         withBias({BATCH, HEADS, SEQ, SEQ}, {{(int64_t{1} << 31) - 1, SEQ * SEQ, SEQ, 1}}),
         "size_limit"});
    // Every stride fits in i32, but (B-1)*2^30 + (Hq-1)*2^29 is past 2^31.
    cases.push_back(
        {"a last element past 2^31",
         withBias({BATCH, HEADS, SEQ, SEQ}, {{int64_t{1} << 30, int64_t{1} << 29, SEQ, 1}}),
         "size_limit"});

    auto virtualBias = withBias({BATCH, HEADS, SEQ, SEQ});
    biasTensor(virtualBias).isVirtual = true;
    cases.push_back({"a virtual bias", virtualBias, "attn_mask"});

    auto raggedBias = withBias({BATCH, HEADS, SEQ, SEQ});
    biasTensor(raggedBias).raggedOffsetUid = EXTRA_UID;
    cases.push_back({"a ragged bias", raggedBias, "attn_mask"});

    auto byValueBias = withBias({BATCH, HEADS, SEQ, SEQ});
    biasTensor(byValueBias).passByValue = true;
    cases.push_back({"a pass-by-value bias", byValueBias, "attn_mask"});

    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.what);
        const auto decline = loggedDecline(arch(), c.spec);
        EXPECT_EQ(decline.cause, c.cause) << decline.detail;
        EXPECT_NE(decline.detail.find("attn_mask"), std::string::npos) << decline.detail;
    }
}

/// The last element the kernel reaches is (Skv-1) + (Sq-1)*sq + (Hq-1)*hs + (B-1)*bs. A
/// batch stride that puts it at 2^31 - 1 is the largest bias the kernel can address; one
/// more element is refused.
TEST_P(TestAttentionDenseBiasMatch, AdditiveBiasReachAdmitsTheLastGraphUnderIt)
{
    ASSERT_EQ(BATCH, 2) << "the batch stride below is (B-1) * stride with B-1 = 1";
    const int64_t headStride = SEQ * SEQ;
    const int64_t rest = (SEQ - 1) + (SEQ - 1) * SEQ + (HEADS - 1) * headStride;
    const int64_t lastFittingBatchStride = INT32_LIMIT - 1 - rest;

    const auto under = matchGraph(
        arch(), withBias({BATCH, HEADS, SEQ, SEQ}, {{lastFittingBatchStride, headStride, SEQ, 1}}));
    ASSERT_TRUE(under.has_value()) << "last element 2^31 - 1";
    EXPECT_EQ(boundBias(arch(), *under, ".bias.stride_batch"), lastFittingBatchStride);

    const auto decline = loggedDecline(
        arch(),
        withBias({BATCH, HEADS, SEQ, SEQ}, {{lastFittingBatchStride + 1, headStride, SEQ, 1}}));
    EXPECT_EQ(decline.cause, "size_limit") << "last element exactly 2^31: " << decline.detail;
}

namespace
{

// Every case above has Hq = Hkv and Sq = Skv, where a bias extent read from the wrong side
// of the graph (key/value heads for query heads, the key length for the query length) would
// match too. The cases below use a GQA cross-attention graph, where the four differ.
constexpr int64_t GQA_Q_HEADS = 8;
constexpr int64_t GQA_KV_HEADS = 2;
constexpr int64_t CROSS_SQ = 256;
constexpr int64_t CROSS_SKV = 512;

/// The base graph at Hq = 8, Hkv = 2, Sq = 256, Skv = 512 with an fp32 bias of @p dims.
GraphSpec crossAttentionWithBias(std::vector<int64_t> dims,
                                 const std::optional<std::vector<int64_t>>& strides = {})
{
    GraphSpec spec = withBias(std::move(dims), strides);
    spec.numQueryHeads = GQA_Q_HEADS;
    spec.numKvHeads = GQA_KV_HEADS;
    spec.seqLenQ = CROSS_SQ;
    spec.seqLenKv = CROSS_SKV;
    return spec;
}

} // namespace

/// The bias a graph carries is [B, Hq, Sq, Skv]: query heads and query rows on the middle
/// axes, the key length last.
TEST_P(TestAttentionDenseBiasMatch, BindsABiasSizedFromTheQueryExtentsOfACrossAttentionGraph)
{
    const auto bound
        = matchGraph(arch(), crossAttentionWithBias({BATCH, GQA_Q_HEADS, CROSS_SQ, CROSS_SKV}));
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(boundBias(arch(), *bound, ".bias.stride_batch"), GQA_Q_HEADS * CROSS_SQ * CROSS_SKV);
    EXPECT_EQ(boundBias(arch(), *bound, ".bias.stride_head"), CROSS_SQ * CROSS_SKV);
    EXPECT_EQ(boundBias(arch(), *bound, ".bias.stride_query"), CROSS_SKV);

    // Each axis at 1 broadcasts, whatever its neighbours are.
    EXPECT_TRUE(
        matchGraph(arch(), crossAttentionWithBias({1, 1, CROSS_SQ, CROSS_SKV})).has_value());
    EXPECT_TRUE(matchGraph(arch(), crossAttentionWithBias({1, 1, 1, CROSS_SKV})).has_value());
}

TEST_P(TestAttentionDenseBiasMatch, DeclinesABiasSizedFromAnotherExtentOfACrossAttentionGraph)
{
    struct Case
    {
        const char* what;
        std::vector<int64_t> dims;
    };
    const std::vector<Case> cases{
        {"key/value heads where query heads belong", {BATCH, GQA_KV_HEADS, CROSS_SQ, CROSS_SKV}},
        {"the key length where the query length belongs",
         {BATCH, GQA_Q_HEADS, CROSS_SKV, CROSS_SKV}},
        {"the query length where the key length belongs", {BATCH, GQA_Q_HEADS, CROSS_SQ, CROSS_SQ}},
    };
    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.what);
        const auto decline = loggedDecline(arch(), crossAttentionWithBias(c.dims));
        EXPECT_EQ(decline.cause, "attn_mask") << decline.detail;
    }
}

/// The last element the kernel reaches is sized by the query heads and the query rows, not
/// by the key/value heads or the key length. Each graph below is on one side of the 2^31
/// limit by the correct extents and on the other by the wrong ones.
TEST_P(TestAttentionDenseBiasMatch, AdditiveBiasReachIsSizedByQueryHeadsAndQueryRows)
{
    // Reach (Sq - 1) * 5e6 + (Skv - 1) = 1.275e9 + 511: under the limit. Sized by Skv
    // instead of Sq it is 2.555e9, over it.
    constexpr int64_t QUERY_STRIDE = 5'000'000;
    EXPECT_TRUE(
        matchGraph(arch(),
                   crossAttentionWithBias({1, 1, CROSS_SQ, CROSS_SKV}, {{0, 0, QUERY_STRIDE, 1}}))
            .has_value())
        << "a reach sized from the key length instead of the query length";

    // Reach (Hq - 1) * 335544320 + (Skv - 1) = 2.349e9: over the limit. Sized by Hkv
    // instead of Hq it is 3.4e8, under it.
    constexpr int64_t FAR_HEAD_STRIDE = (int64_t{1} << 28) + (int64_t{1} << 26);
    const auto decline = loggedDecline(
        arch(),
        crossAttentionWithBias({1, GQA_Q_HEADS, 1, CROSS_SKV}, {{0, FAR_HEAD_STRIDE, 1, 1}}));
    EXPECT_EQ(decline.cause, "size_limit") << decline.detail;
}

/// A bias graph is served by bias candidates only, and a plain graph by plain ones: their
/// scores differ, and so do their argument lists. A record without the field is a plain
/// candidate (the KMD default 0).
TEST_P(TestAttentionDenseBiasMatch, PairsBiasGraphsWithBiasCandidatesOnly)
{
    const auto biasGraph = withBias({BATCH, HEADS, SEQ, SEQ});
    const auto withField = [](int64_t value) {
        auto kernel = makeKernel(arch(), baseGraph());
        kernel.metadata[std::string("use_additive_bias")] = value;
        return kernel;
    };
    const auto biasKernel = withField(1);
    const auto plainKernel = withField(0);
    auto legacyKernel = makeKernel(arch(), baseGraph());
    legacyKernel.metadata.erase(std::string("use_additive_bias"));

    EXPECT_TRUE(matchesKernel(arch(), biasGraph, biasKernel));
    EXPECT_FALSE(matchesKernel(arch(), biasGraph, plainKernel));
    EXPECT_FALSE(matchesKernel(arch(), biasGraph, legacyKernel));

    EXPECT_FALSE(matchesKernel(arch(), GraphSpec{}, biasKernel));
    EXPECT_TRUE(matchesKernel(arch(), GraphSpec{}, plainKernel));
    EXPECT_TRUE(matchesKernel(arch(), GraphSpec{}, legacyKernel));

    // A field that is neither 0 nor 1, or not an integer, is a malformed record: it serves
    // no graph.
    auto outOfRange = withField(2);
    auto notAnInteger = withField(0);
    notAnInteger.metadata[std::string("use_additive_bias")] = std::string("1");
    for(const auto* malformed : {&outOfRange, &notAnInteger})
    {
        EXPECT_FALSE(matchesKernel(arch(), biasGraph, *malformed));
        EXPECT_FALSE(matchesKernel(arch(), GraphSpec{}, *malformed));
    }
}

/// The catalog ships no bias kernel with a sliding window: its bias cells are causal or
/// plain with window 0, and its windowed kernels carry no bias. A bias graph with a window
/// must therefore find no candidate (so the engine is not applicable and no plan is built),
/// and in particular not the window-less bias kernel, which would run the full causal
/// attention, nor the window kernel without the bias, which would drop it. gfx950 declines
/// the window at graph_match, so nothing reaches kernel_match; gfx942 binds the window and
/// kernel_match refuses each of those candidates, and would serve a bias kernel built with
/// that window if the catalog gained one (the control that keeps this from passing for an
/// unrelated reason).
TEST_P(TestAttentionDenseBiasMatch, ServesNoBiasGraphWithASlidingWindowFromTheShippedKernels)
{
    static constexpr int64_t WINDOW = 128;
    const auto windowed = [](GraphSpec spec) {
        spec.leftBound = WINDOW;
        spec.rightBound = 0;
        return spec;
    };
    const auto candidate = [&](int64_t bias, int64_t window) {
        auto kernel = makeKernel(arch(), baseGraph());
        kernel.metadata[std::string("use_additive_bias")] = bias;
        kernel.metadata[std::string("sliding_window")] = window;
        return kernel;
    };
    const auto servedBy = [&](const GraphSpec& spec, const KernelDefinition& kernel) {
        const Graph graph(arch(), spec);
        const auto bound = graph.match();
        return bound.has_value()
               && hipdnn_plugin_sdk::ingestor::KernelMatcherRegistry::resolve(
                   std::string(arch().kernelMatchSymbol))(graph.context(), *bound, kernel);
    };

    const auto biasWindowGraph = windowed(withBias({BATCH, HEADS, SEQ, SEQ}));
    EXPECT_EQ(matchGraph(arch(), biasWindowGraph).has_value(), arch().servesSlidingWindow);

    // The three shapes the catalog ships for this tile: plain, bias without a window, and a
    // window without a bias.
    for(const auto& [bias, window] : {std::pair{int64_t{0}, int64_t{0}},
                                      std::pair{int64_t{1}, int64_t{0}},
                                      std::pair{int64_t{0}, WINDOW}})
    {
        SCOPED_TRACE("candidate use_additive_bias " + std::to_string(bias) + ", sliding_window "
                     + std::to_string(window));
        EXPECT_FALSE(servedBy(biasWindowGraph, candidate(bias, window)));
    }

    if(arch().servesSlidingWindow)
    {
        // Each half of the combination is served on its own, by the candidate built for it.
        EXPECT_TRUE(servedBy(windowed(baseGraph()), candidate(0, WINDOW)));
        EXPECT_TRUE(servedBy(withBias({BATCH, HEADS, SEQ, SEQ}), candidate(1, 0)));
        // So the matcher is not what refuses the combination: a bias kernel built with the
        // window would serve it, and its absence from the catalog is what leaves none.
        EXPECT_TRUE(servedBy(biasWindowGraph, candidate(1, WINDOW)));
    }
}

/// kernel_match reads the bias flag from the binding instead of assuming one; a binding
/// that lacks it (a renamed token) is refused rather than read as "no bias".
TEST_P(TestAttentionDenseBiasMatch, RefusesABindingWithoutTheBiasFlag)
{
    const Graph graph(arch(), GraphSpec{});
    auto bound = graph.match();
    ASSERT_TRUE(bound.has_value());
    auto plainKernel = makeKernel(arch(), baseGraph());
    plainKernel.metadata[std::string("use_additive_bias")] = int64_t{0};
    const auto kernelMatches = [&](const BoundTokens& binding, const KernelDefinition& kernel) {
        return hipdnn_plugin_sdk::ingestor::KernelMatcherRegistry::resolve(
            std::string(arch().kernelMatchSymbol))(graph.context(), binding, kernel);
    };
    ASSERT_TRUE(kernelMatches(*bound, plainKernel));

    bound->erase(std::string(arch().tokenPrefix) + ".bias");
    EXPECT_FALSE(kernelMatches(*bound, plainKernel));

    // A legacy record carries no use_additive_bias field at all. Without the token it is
    // refused as well, rather than the pair of absences reading as "no bias on either side".
    auto legacyKernel = makeKernel(arch(), baseGraph());
    legacyKernel.metadata.erase(std::string("use_additive_bias"));
    EXPECT_FALSE(kernelMatches(*bound, legacyKernel));
}

INSTANTIATE_TEST_SUITE_P(,
                         TestAttentionDenseBiasMatch,
                         ::testing::ValuesIn(biasServingArches()),
                         attentionDenseArchName);

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
