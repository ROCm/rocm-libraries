// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "ConfigHelpers.hpp"
#include "GraphTest.hpp"
#include "asm_fmha_v3_fwd_configs.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaFwdPlanBuilder.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaPlanUtils.hpp"
#include "hip_kernel_provider_common/HipDeviceUtils.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace asm_sdpa_engine
{
namespace
{

class TestSdpaFwdPlanBuilder : public ::testing::Test
{
protected:
    SdpaFwdPlanBuilder _planBuilder;
    Handle _handle;
};

TEST_F(TestSdpaFwdPlanBuilder, IsApplicableReturnsFalseForNonSdpaGraph)
{
    // Create a batchnorm inference graph - this does not use SDPA attributes
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

auto createSdpaFwdGraph(const std::vector<int64_t>& qDims = {4, 8, 256, 128},
                        const std::vector<int64_t>& vDims = {4, 8, 256, 128},
                        hipdnn_flatbuffers_sdk::data_objects::DataType dataType
                        = hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                        bool withAttnMask = false,
                        bool withScale = false,
                        bool withStats = false,
                        bool alibiMask = false,
                        bool paddingMask = false,
                        bool causalMask = false)
{
    if(qDims.size() != 4 || vDims.size() != 4)
    {
        throw std::runtime_error("Q, K and V tensors must have a dimension of 4");
    }
    const std::vector<int64_t> kDims = {qDims[0], vDims[1], vDims[2], qDims[3]};
    const std::vector<int64_t> oDims = {qDims[0], qDims[1], qDims[2], vDims[3]};

    return hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        qDims,
        hipdnn_data_sdk::utilities::generateStrides(qDims),
        kDims,
        hipdnn_data_sdk::utilities::generateStrides(kDims),
        vDims,
        hipdnn_data_sdk::utilities::generateStrides(vDims),
        oDims,
        hipdnn_data_sdk::utilities::generateStrides(oDims),
        dataType,
        withAttnMask,
        withScale,
        withStats,
        alibiMask,
        paddingMask,
        causalMask);
}

TEST_F(TestSdpaFwdPlanBuilder, IsApplicableAvailableKernels)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());

    for(const auto& test : getCompatibleGraphsForArch(deviceString, cfg_fmha_fwd))
    {
        EXPECT_TRUE(_planBuilder.isApplicable(_handle, test.graphWrapper())) << test.message;
    }
}

TEST_F(TestSdpaFwdPlanBuilder, IsApplicableSdpaVariations)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    const std::vector<std::pair<GraphTest, bool>> applicabilityTests = {
        {GraphTest{createSdpaFwdGraph(), "Valid test with q head dim 128"}, true},
        {GraphTest{createSdpaFwdGraph({4, 8, 256, 192}), "Valid test with q head dim 192"}, true},
        {GraphTest{createSdpaFwdGraph({4, 8, 256, 100}), "Final dimension not 128"}, false},
        {GraphTest{createSdpaFwdGraph({4, 8, 256, 128}, {4, 8, 256, 128}, DataType::HALF),
                   "Half precision tensor data type"},
         false},
        {GraphTest{createSdpaFwdGraph({4, 8, 256, 128}, {4, 8, 256, 128}, DataType::BFLOAT16, true),
                   "attn_mask = true"},
         false},
        {GraphTest{createSdpaFwdGraph(
                       {4, 8, 256, 128}, {4, 8, 256, 128}, DataType::BFLOAT16, false, true),
                   "scale = true"},
         true},
        {GraphTest{
             createSdpaFwdGraph(
                 {4, 8, 256, 128}, {4, 8, 256, 128}, DataType::BFLOAT16, false, true, false, true),
             "alibi_mask = true"},
         false},
        {GraphTest{createSdpaFwdGraph({4, 8, 256, 128},
                                      {4, 8, 256, 128},
                                      DataType::BFLOAT16,
                                      false,
                                      true,
                                      false,
                                      false,
                                      true),
                   "padding_mask = true"},
         false},
        {GraphTest{createSdpaFwdGraph({4, 8, 256, 128},
                                      {4, 8, 256, 128},
                                      DataType::BFLOAT16,
                                      false,
                                      true,
                                      false,
                                      false,
                                      false,
                                      true),
                   "causal_mask = true"},
         false}};

    for(const auto& [test, applicability] : applicabilityTests)
    {
        EXPECT_EQ(_planBuilder.isApplicable(_handle, test.graphWrapper()), applicability)
            << test.message;
    }
}

TEST_F(TestSdpaFwdPlanBuilder, GetMaxWorkspaceSizeCalculatesCorrectly)
{
    // Create an SDPA graph with known dimensions (withStats = false by default)
    auto builder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph();

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the workspace size from the plan builder
    const Settings settings;
    const size_t workspaceSize = _planBuilder.getMaxWorkspaceSize(_handle, graphWrapper, settings);

    // Forward-only kernel uses LDS internally, no external workspace needed
    // LSE (when present) is an optional output tensor (stats_tensor_uid), not workspace
    // The default test graph has withStats = false, so workspace should be 0
    EXPECT_EQ(workspaceSize, 0u);
}

// =============================================================================
// Canonical mask-attribute policy (plan_utils::getMaskType)
// =============================================================================
//
// These tests exercise the shared mask-precedence policy directly through
// plan_utils::getMaskType rather than through isApplicable. When a deprecated
// causal boolean is set it takes precedence over the modern bounds trio; only
// setting both deprecated booleans at once throws. The policy is
// hardware-agnostic (it runs before any device dispatch and independent of the
// kernel registry), so testing the helper keeps the assertions meaningful on
// any device — including this gfx950 box. Driving the policy through
// isApplicable would not discriminate the policy result from the unrelated "no
// matching kernel" rejection that gfx950 produces for causal configurations
// (the gfx950 forward registry carries NO_MASK rows only).

// Build a forward SDPA graph that sets the deprecated causal booleans and the
// modern bounds trio explicitly, so contradictory combinations can be
// constructed. Returns the FlatBufferBuilder owning the graph buffer.
flatbuffers::FlatBufferBuilder createSdpaFwdGraphWithMask(
    bool causalMask,
    bool causalMaskBottomRight,
    flatbuffers::Optional<int64_t> leftBound,
    flatbuffers::Optional<int64_t> rightBound,
    hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment diagAlignment)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<TensorAttributes>> tensorAttributes;

    const std::vector<int64_t> dims = {4, 8, 256, 128};
    const std::vector<int64_t> strides = hipdnn_data_sdk::utilities::generateStrides(dims);

    int64_t uid = 1;
    const auto qUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, qUid, "q", DataType::BFLOAT16, &strides, &dims));
    const auto kUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, kUid, "k", DataType::BFLOAT16, &strides, &dims));
    const auto vUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, vUid, "v", DataType::BFLOAT16, &strides, &dims));
    const auto oUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, oUid, "o", DataType::BFLOAT16, &strides, &dims));

    const auto sdpaAttributes
        = CreateSdpaAttributes(builder,
                               qUid,
                               kUid,
                               vUid,
                               oUid,
                               flatbuffers::nullopt, // attn_mask_tensor_uid
                               flatbuffers::nullopt, // scale_tensor_uid
                               flatbuffers::nullopt, // seq_len_q_tensor_uid
                               flatbuffers::nullopt, // seq_len_kv_tensor_uid
                               flatbuffers::nullopt, // seed_tensor_uid
                               flatbuffers::nullopt, // offset_tensor_uid
                               flatbuffers::nullopt, // dropout_mask_tensor_uid
                               flatbuffers::nullopt, // dropout_scale_tensor_uid
                               flatbuffers::nullopt, // page_table_k_tensor_uid
                               flatbuffers::nullopt, // page_table_v_tensor_uid
                               flatbuffers::nullopt, // block_mask_tensor_uid
                               flatbuffers::nullopt, // sink_token_tensor_uid
                               flatbuffers::nullopt, // descale_q_tensor_uid
                               flatbuffers::nullopt, // descale_k_tensor_uid
                               flatbuffers::nullopt, // descale_v_tensor_uid
                               flatbuffers::nullopt, // descale_s_tensor_uid
                               flatbuffers::nullopt, // scale_s_tensor_uid
                               flatbuffers::nullopt, // scale_o_tensor_uid
                               flatbuffers::nullopt, // stats_tensor_uid
                               flatbuffers::nullopt, // max_tensor_uid
                               flatbuffers::nullopt, // sum_exp_tensor_uid
                               flatbuffers::nullopt, // rng_dump_tensor_uid
                               flatbuffers::nullopt, // amax_s_tensor_uid
                               flatbuffers::nullopt, // amax_o_tensor_uid
                               flatbuffers::nullopt, // generate_stats
                               false, // alibi_mask
                               false, // padding_mask
                               causalMask,
                               causalMaskBottomRight,
                               flatbuffers::nullopt, // dropout_probability
                               flatbuffers::nullopt, // attn_scale_value
                               leftBound,
                               rightBound,
                               flatbuffers::nullopt, // max_seq_len_kv
                               diagAlignment,
                               DataType::FLOAT, // mma_core_mode
                               AttentionImplementation::AUTO);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "sdpa_fwd",
                                     DataType::BFLOAT16,
                                     NodeAttributes::SdpaAttributes,
                                     sdpaAttributes.Union()));

    const auto graphOffset = CreateGraphDirect(builder,
                                               "test",
                                               DataType::FLOAT,
                                               DataType::HALF,
                                               DataType::BFLOAT16,
                                               &tensorAttributes,
                                               &nodes);
    builder.Finish(graphOffset);
    return builder;
}

// Resolve the SDPA attributes from a graph buffer and classify the mask.
plan_utils::MaskType classifyMask(const flatbuffers::FlatBufferBuilder& builder)
{
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());
    const auto& attrs = graphWrapper.nodeWrappers()
                            .front()
                            ->attributesAs<hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes>();
    return plan_utils::getMaskType(attrs);
}

TEST_F(TestSdpaFwdPlanBuilder, IsApplicable_RejectsCausalMaskAndBottomRightSetTogether)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    // Both deprecated causal booleans set is a contradiction.
    auto builder = createSdpaFwdGraphWithMask(
        /*causalMask=*/true,
        /*causalMaskBottomRight=*/true,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        DiagonalAlignment::TOP_LEFT);

    EXPECT_THROW(classifyMask(builder), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestSdpaFwdPlanBuilder, IsApplicable_PrefersCausalMaskOverWindowBounds)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    // causal_mask=true takes precedence over the bounds trio, even though the
    // bounds describe a sliding window (left=64, right=64): result is causal.
    auto builder = createSdpaFwdGraphWithMask(
        /*causalMask=*/true,
        /*causalMaskBottomRight=*/false,
        flatbuffers::Optional<int64_t>(64),
        flatbuffers::Optional<int64_t>(64),
        DiagonalAlignment::TOP_LEFT);

    plan_utils::MaskType maskType = plan_utils::MaskType::NO_MASK;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::TOP_LEFT_CAUSAL);
}

TEST_F(TestSdpaFwdPlanBuilder, IsApplicable_PrefersBottomRightCausalOverTopLeftBounds)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    // causal_mask_bottom_right=true takes precedence over the bounds trio, even
    // though the trio (left=-1, right=0, diag=TOP_LEFT) would derive
    // TOP_LEFT_CAUSAL: result is bottom-right causal.
    auto builder = createSdpaFwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/true,
        flatbuffers::Optional<int64_t>(-1),
        flatbuffers::Optional<int64_t>(0),
        DiagonalAlignment::TOP_LEFT);

    plan_utils::MaskType maskType = plan_utils::MaskType::NO_MASK;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::BOTTOM_RIGHT_CAUSAL);
}

TEST_F(TestSdpaFwdPlanBuilder, IsApplicable_AcceptsConsistentCausalMaskAndBounds)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    // causal_mask=true with a consistent bounds trio (left=-1, right=0,
    // diag=TOP_LEFT -> TOP_LEFT_CAUSAL) must not throw and classify as causal.
    auto builder = createSdpaFwdGraphWithMask(
        /*causalMask=*/true,
        /*causalMaskBottomRight=*/false,
        flatbuffers::Optional<int64_t>(-1),
        flatbuffers::Optional<int64_t>(0),
        DiagonalAlignment::TOP_LEFT);

    plan_utils::MaskType maskType = plan_utils::MaskType::NO_MASK;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::TOP_LEFT_CAUSAL);
}

TEST_F(TestSdpaFwdPlanBuilder, IsApplicable_PrefersBottomRightCausalOverWindowBounds)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    // causal_mask_bottom_right=true takes precedence over the bounds trio, even
    // though a symmetric sliding window (left=64, right=64) would derive
    // WINDOW_GENERIC: result is bottom-right causal.
    auto builder = createSdpaFwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/true,
        flatbuffers::Optional<int64_t>(64),
        flatbuffers::Optional<int64_t>(64),
        DiagonalAlignment::BOTTOM_RIGHT);

    plan_utils::MaskType maskType = plan_utils::MaskType::NO_MASK;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::BOTTOM_RIGHT_CAUSAL);
}

// Modern bounds-trio path (no deprecated boolean set). An unset bound is treated
// as unbounded (-1), so a partially specified trio still derives the mask it
// describes.

TEST_F(TestSdpaFwdPlanBuilder, MaskBoundsTrio_RightZeroLeftUnsetDerivesTopLeftCausal)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    // The canonical causal request: left unbounded (unset) with right_bound=0.
    // An unset left bound must be read as unbounded, not as "no mask".
    auto builder = createSdpaFwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/false,
        flatbuffers::nullopt,
        flatbuffers::Optional<int64_t>(0),
        DiagonalAlignment::TOP_LEFT);

    plan_utils::MaskType maskType = plan_utils::MaskType::NO_MASK;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::TOP_LEFT_CAUSAL);
}

TEST_F(TestSdpaFwdPlanBuilder, MaskBoundsTrio_RightZeroBottomRightDerivesBottomRightCausal)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    // Same causal request with bottom-right diagonal alignment.
    auto builder = createSdpaFwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/false,
        flatbuffers::nullopt,
        flatbuffers::Optional<int64_t>(0),
        DiagonalAlignment::BOTTOM_RIGHT);

    plan_utils::MaskType maskType = plan_utils::MaskType::NO_MASK;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::BOTTOM_RIGHT_CAUSAL);
}

TEST_F(TestSdpaFwdPlanBuilder, MaskBoundsTrio_ExplicitCausalBoundsDeriveTopLeftCausal)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    // Explicit unbounded-left (left=-1) with right_bound=0 matches the unset case.
    auto builder = createSdpaFwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/false,
        flatbuffers::Optional<int64_t>(-1),
        flatbuffers::Optional<int64_t>(0),
        DiagonalAlignment::TOP_LEFT);

    plan_utils::MaskType maskType = plan_utils::MaskType::NO_MASK;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::TOP_LEFT_CAUSAL);
}

TEST_F(TestSdpaFwdPlanBuilder, MaskBoundsTrio_BothUnsetDerivesNoMask)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    auto builder = createSdpaFwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/false,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        DiagonalAlignment::TOP_LEFT);

    plan_utils::MaskType maskType = plan_utils::MaskType::WINDOW_GENERIC;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::NO_MASK);
}

TEST_F(TestSdpaFwdPlanBuilder, MaskBoundsTrio_BothUnboundedDerivesNoMask)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    auto builder = createSdpaFwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/false,
        flatbuffers::Optional<int64_t>(-1),
        flatbuffers::Optional<int64_t>(-1),
        DiagonalAlignment::TOP_LEFT);

    plan_utils::MaskType maskType = plan_utils::MaskType::WINDOW_GENERIC;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::NO_MASK);
}

TEST_F(TestSdpaFwdPlanBuilder, MaskBoundsTrio_SymmetricWindowDerivesWindowGeneric)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    auto builder = createSdpaFwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/false,
        flatbuffers::Optional<int64_t>(64),
        flatbuffers::Optional<int64_t>(64),
        DiagonalAlignment::TOP_LEFT);

    plan_utils::MaskType maskType = plan_utils::MaskType::NO_MASK;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::WINDOW_GENERIC);
}

TEST_F(TestSdpaFwdPlanBuilder, MaskBoundsTrio_LeftOnlyDerivesWindowGeneric)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    // A bounded left with an unset (unbounded) right is a one-sided window.
    auto builder = createSdpaFwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/false,
        flatbuffers::Optional<int64_t>(64),
        flatbuffers::nullopt,
        DiagonalAlignment::TOP_LEFT);

    plan_utils::MaskType maskType = plan_utils::MaskType::NO_MASK;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::WINDOW_GENERIC);
}

} // namespace
} // namespace asm_sdpa_engine
