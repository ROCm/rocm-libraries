// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <optional>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "dispatcher/SdpaGraphAdapter.hpp"
#include "dispatcher/SdpaProblem.hpp"
#include "tests/dispatcher/SdpaGraphFixture.hpp"

namespace rocke_client::dispatcher
{
namespace
{

using hipdnn_flatbuffers_sdk::data_objects::AttentionImplementation;
using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment;
using test::buildSdpaGraph;
using test::SdpaGraphConfig;

// ---------------------------------------------------------------------------
// Accept path: graphs the FMHA-fwd-MFMA family can serve normalize to a problem
// ---------------------------------------------------------------------------

TEST(TestSdpaGraphAdapter, TranslatesValidBshdGraph)
{
    const auto fixture = buildSdpaGraph(SdpaGraphConfig{});
    const std::optional<SdpaProblem> problem = translate(fixture.graphWrapper());

    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->op, "sdpa_fwd");
    EXPECT_EQ(problem->dtype, "fp16");
    EXPECT_EQ(problem->layout, TensorLayout::BSHD);
    EXPECT_EQ(problem->batch, 2);
    EXPECT_EQ(problem->numQueryHeads, 4);
    EXPECT_EQ(problem->numKvHeads, 4);
    EXPECT_EQ(problem->seqlenQ, 64);
    EXPECT_EQ(problem->seqlenK, 64);
    EXPECT_EQ(problem->headSize, 64);
    EXPECT_EQ(problem->maskMode, "none");
    EXPECT_DOUBLE_EQ(problem->dropoutProbability, 0.0);
    EXPECT_FALSE(problem->paddingMask);
    EXPECT_FALSE(problem->alibiMask);
    EXPECT_EQ(problem->scalePolicy, "default_1_over_sqrt_d");
    // arch is filled by the dispatcher, not the adapter.
    EXPECT_TRUE(problem->arch.empty());
}

TEST(TestSdpaGraphAdapter, CapturesShapeForNonSquareProblem)
{
    SdpaGraphConfig config;
    config.batch = 5;
    config.numQueryHeads = 8;
    config.numKvHeads = 2; // GQA: fewer KV heads is a valid shape, gated by selection
    config.seqlenQ = 128;
    config.seqlenK = 256;
    config.headSizeQK = 128;
    config.headSizeV = 128;

    const auto fixture = buildSdpaGraph(config);
    const std::optional<SdpaProblem> problem = translate(fixture.graphWrapper());

    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->batch, 5);
    EXPECT_EQ(problem->numQueryHeads, 8);
    EXPECT_EQ(problem->numKvHeads, 2);
    EXPECT_EQ(problem->seqlenQ, 128);
    EXPECT_EQ(problem->seqlenK, 256);
    EXPECT_EQ(problem->headSize, 128);
}

TEST(TestSdpaGraphAdapter, InfersBhsdLayout)
{
    SdpaGraphConfig config;
    config.bshd = false;
    const auto fixture = buildSdpaGraph(config);
    const std::optional<SdpaProblem> problem = translate(fixture.graphWrapper());

    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->layout, TensorLayout::BHSD);
}

TEST(TestSdpaGraphAdapter, MapsBf16Dtype)
{
    SdpaGraphConfig config;
    config.dtype = DataType::BFLOAT16;
    const auto fixture = buildSdpaGraph(config);
    const std::optional<SdpaProblem> problem = translate(fixture.graphWrapper());

    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->dtype, "bf16");
}

TEST(TestSdpaGraphAdapter, CapturesMaskModesIntoProblem)
{
    {
        SdpaGraphConfig config;
        config.causalMask = true;
        const auto fixture = buildSdpaGraph(config);
        const auto problem = translate(fixture.graphWrapper());
        ASSERT_TRUE(problem.has_value());
        EXPECT_EQ(problem->maskMode, "causal_top_left");
    }
    {
        SdpaGraphConfig config;
        config.causalMaskBottomRight = true;
        const auto fixture = buildSdpaGraph(config);
        const auto problem = translate(fixture.graphWrapper());
        ASSERT_TRUE(problem.has_value());
        EXPECT_EQ(problem->maskMode, "causal_bottom_right");
    }
    {
        // Sliding-window bounds.
        SdpaGraphConfig config;
        config.leftBound = 64;
        config.rightBound = 64;
        const auto fixture = buildSdpaGraph(config);
        const auto problem = translate(fixture.graphWrapper());
        ASSERT_TRUE(problem.has_value());
        EXPECT_EQ(problem->maskMode, "sliding_window");
    }
}

// Pins the local mask classification to asm_sdpa_engine::plan_utils::getMaskType:
// the deprecated causal booleans take precedence, an unset bound is unbounded, and
// left=-1/right=0 selects the diagonal-alignment causal variant.
TEST(TestSdpaGraphAdapter, ClassifiesBoundsBasedCausalByDiagonalAlignment)
{
    {
        SdpaGraphConfig config;
        config.leftBound = -1;
        config.rightBound = 0;
        config.diagonalAlignment = DiagonalAlignment::TOP_LEFT;
        const auto fixture = buildSdpaGraph(config);
        const auto problem = translate(fixture.graphWrapper());
        ASSERT_TRUE(problem.has_value());
        EXPECT_EQ(problem->maskMode, "causal_top_left");
    }
    {
        SdpaGraphConfig config;
        config.leftBound = -1;
        config.rightBound = 0;
        config.diagonalAlignment = DiagonalAlignment::BOTTOM_RIGHT;
        const auto fixture = buildSdpaGraph(config);
        const auto problem = translate(fixture.graphWrapper());
        ASSERT_TRUE(problem.has_value());
        EXPECT_EQ(problem->maskMode, "causal_bottom_right");
    }
}

// ---------------------------------------------------------------------------
// Reject path: structural gates
// ---------------------------------------------------------------------------

TEST(TestSdpaGraphAdapter, RejectsNonSdpaNode)
{
    SdpaGraphConfig config;
    config.sdpaNode = false;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, RejectsMultiNodeGraph)
{
    SdpaGraphConfig config;
    config.nodeCount = 2;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, RejectsMissingOutputTensor)
{
    SdpaGraphConfig config;
    config.omitOutputTensor = true;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, RejectsNonRank4Tensor)
{
    SdpaGraphConfig config;
    config.queryRank3 = true;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, RejectsMixedElementTypes)
{
    SdpaGraphConfig config;
    config.dtype = DataType::HALF;
    config.keyDtype = DataType::BFLOAT16;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, RejectsMismatchedVHeadSize)
{
    // V's head dim differs from Q/K: the family serves a single head_size.
    SdpaGraphConfig config;
    config.headSizeQK = 64;
    config.headSizeV = 128;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, RejectsNonContiguousLayout)
{
    SdpaGraphConfig config;
    config.nonContiguousStrides = true; // packing matches neither BSHD nor BHSD
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, RejectsMismatchedKvLayout)
{
    // K packed with a different layout than Q; inferring layout from Q alone would
    // wrongly accept it.
    SdpaGraphConfig config;
    config.mismatchKeyLayout = true;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, RejectsMismatchedBatch)
{
    SdpaGraphConfig config; // default batch 2
    config.mismatchBatch = 4; // K batch != Q batch
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, RejectsMismatchedVSeqlen)
{
    SdpaGraphConfig config; // default seqlenK 64
    config.mismatchVSeqlen = 128; // V seqlen != K seqlen
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, RejectsMismatchedOHeads)
{
    SdpaGraphConfig config; // default numQueryHeads 4
    config.mismatchOHeads = 8; // O heads != Q heads
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, RejectsOverrideShapeEnabled)
{
    // Execute-time override shapes can diverge from the matched compile-time dims;
    // the family serves fixed prebuilt shapes.
    SdpaGraphConfig config;
    config.overrideShapeEnabled = true;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

// ---------------------------------------------------------------------------
// Reject path: dtype allowlist
// ---------------------------------------------------------------------------

TEST(TestSdpaGraphAdapter, RejectsFp32Dtype)
{
    // fp32 is outside the FMHA-fwd-MFMA family (fp16/bf16 only).
    SdpaGraphConfig config;
    config.dtype = DataType::FLOAT;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, RejectsUnsupportedDtype)
{
    SdpaGraphConfig config;
    config.dtype = DataType::INT32;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, RejectsNonFp32ComputeDtype)
{
    // The fmha_fwd_mfma family accumulates in fp32; a non-fp32 compute_data_type
    // has no matching instance.
    SdpaGraphConfig config;
    config.computeDataType = DataType::HALF;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, RejectsUnsetComputeDtype)
{
    // compute_data_type must be declared fp32 explicitly; UNSET is not accepted.
    SdpaGraphConfig config;
    config.computeDataType = DataType::UNSET;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, RejectsExplicitMmaCoreMode)
{
    // mma_core_mode has no selection axis and no consumer today; any explicit value
    // is declined (including fp32).
    for(const DataType mode : {DataType::FLOAT, DataType::HALF})
    {
        SdpaGraphConfig config;
        config.mmaCoreMode = mode;
        const auto fixture = buildSdpaGraph(config);
        EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
    }
}

TEST(TestSdpaGraphAdapter, RejectsCompositeImplementation)
{
    // The fused family cannot honor an explicit COMPOSITE (decomposed) request.
    SdpaGraphConfig config;
    config.implementation = AttentionImplementation::COMPOSITE;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, AcceptsUnifiedImplementation)
{
    // UNIFIED permits a fused kernel; accepted.
    SdpaGraphConfig config;
    config.implementation = AttentionImplementation::UNIFIED;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_TRUE(translate(fixture.graphWrapper()).has_value());
}

// ---------------------------------------------------------------------------
// Reject path: capability gates the family cannot serve today
// ---------------------------------------------------------------------------

TEST(TestSdpaGraphAdapter, RejectsContradictoryMaskFlags)
{
    SdpaGraphConfig config;
    config.causalMask = true;
    config.causalMaskBottomRight = true;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, RejectsNonZeroDropout)
{
    SdpaGraphConfig config;
    config.dropoutProbability = 0.25f;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, AcceptsExplicitZeroDropout)
{
    // dropout explicitly set to 0 is "no dropout" and is accepted.
    SdpaGraphConfig config;
    config.dropoutProbability = 0.0f;
    const auto fixture = buildSdpaGraph(config);
    const auto problem = translate(fixture.graphWrapper());
    ASSERT_TRUE(problem.has_value());
    EXPECT_DOUBLE_EQ(problem->dropoutProbability, 0.0);
}

TEST(TestSdpaGraphAdapter, RejectsAlibiMask)
{
    SdpaGraphConfig config;
    config.alibiMask = true;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, RejectsPaddingMask)
{
    SdpaGraphConfig config;
    config.paddingMask = true;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, RejectsExplicitScaleValue)
{
    SdpaGraphConfig config;
    config.attnScaleValue = 0.5f;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, RejectsExplicitScaleTensor)
{
    SdpaGraphConfig config;
    config.includeScaleTensor = true;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

// ---------------------------------------------------------------------------
// Reject path: unrepresentable feature tensors / outputs (no selection key)
// ---------------------------------------------------------------------------

TEST(TestSdpaGraphAdapter, RejectsAdditiveAttnMask)
{
    SdpaGraphConfig config;
    config.attnMaskTensorUid = 900;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, RejectsPagedKv)
{
    SdpaGraphConfig config;
    config.pageTableKTensorUid = 901;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, RejectsVariableLength)
{
    SdpaGraphConfig config;
    config.seqLenQTensorUid = 902;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, RejectsBlockMask)
{
    SdpaGraphConfig config;
    config.blockMaskTensorUid = 903;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, RejectsSinkToken)
{
    SdpaGraphConfig config;
    config.sinkTokenTensorUid = 904;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, RejectsStatsOutput)
{
    SdpaGraphConfig config;
    config.statsTensorUid = 905;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, RejectsFp8Descale)
{
    SdpaGraphConfig config;
    config.descaleQTensorUid = 906;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, RejectsGenerateStats)
{
    SdpaGraphConfig config;
    config.generateStats = true;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, AcceptsGenerateStatsFalse)
{
    // generate_stats explicitly false does not request stats and is accepted.
    SdpaGraphConfig config;
    config.generateStats = false;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_TRUE(translate(fixture.graphWrapper()).has_value());
}

TEST(TestSdpaGraphAdapter, RejectsMaxSeqLenKv)
{
    SdpaGraphConfig config;
    config.maxSeqLenKv = 4096;
    const auto fixture = buildSdpaGraph(config);
    EXPECT_FALSE(translate(fixture.graphWrapper()).has_value());
}

} // namespace
} // namespace rocke_client::dispatcher
