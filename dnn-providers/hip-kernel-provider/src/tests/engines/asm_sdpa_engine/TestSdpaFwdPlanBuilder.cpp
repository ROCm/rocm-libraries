// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "ConfigHelpers.hpp"
#include "GraphTest.hpp"
#include "asm_fmha_v3_fwd_configs.hpp"
#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaFwdPlanBuilder.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaPlanUtils.hpp"
#include "hip_kernel_provider_common/HipDeviceUtils.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>

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
                        hipdnn_flatbuffers_sdk::data_objects::DataType inputType
                        = hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                        hipdnn_flatbuffers_sdk::data_objects::DataType computeType
                        = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                        bool withAttnMask = false,
                        bool withScale = false,
                        bool withStats = false,
                        bool alibiMask = false,
                        bool paddingMask = false,
                        bool causalMask = false,
                        bool overrideShapeEnabled = false,
                        hipdnn_flatbuffers_sdk::data_objects::DataType mmaCoreMode
                        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET)
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
        inputType,
        computeType,
        withAttnMask,
        withScale,
        withStats,
        alibiMask,
        paddingMask,
        causalMask,
        overrideShapeEnabled,
        mmaCoreMode);
}

TEST_F(TestSdpaFwdPlanBuilder, IsApplicableReturnsFalseForOverrideShapeEnabledGraph)
{
    auto builder = createSdpaFwdGraph({4, 8, 256, 128},
                                      {4, 8, 256, 128},
                                      hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                                      hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                      false,
                                      false,
                                      false,
                                      false,
                                      false,
                                      false,
                                      /*overrideShapeEnabled=*/true);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
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
        {GraphTest{
             createSdpaFwdGraph(
                 {4, 8, 256, 128}, {4, 8, 256, 128}, DataType::BFLOAT16, DataType::FLOAT, true),
             "attn_mask = true"},
         false},
        {GraphTest{createSdpaFwdGraph({4, 8, 256, 128},
                                      {4, 8, 256, 128},
                                      DataType::BFLOAT16,
                                      DataType::FLOAT,
                                      false,
                                      true),
                   "scale = true"},
         true},
        {GraphTest{createSdpaFwdGraph({4, 8, 256, 128},
                                      {4, 8, 256, 128},
                                      DataType::BFLOAT16,
                                      DataType::FLOAT,
                                      false,
                                      true,
                                      false,
                                      true),
                   "alibi_mask = true"},
         false},
        {GraphTest{createSdpaFwdGraph({4, 8, 256, 128},
                                      {4, 8, 256, 128},
                                      DataType::BFLOAT16,
                                      DataType::FLOAT,
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
                                      DataType::FLOAT,
                                      false,
                                      true,
                                      false,
                                      false,
                                      false,
                                      true),
                   "causal_mask = true"},
         false},
        {GraphTest{createSdpaFwdGraph({4, 8, 256, 128},
                                      {4, 8, 256, 128},
                                      DataType::BFLOAT16,
                                      DataType::BFLOAT16,
                                      false,
                                      true),
                   "compute_data_type != FLOAT"},
         false},
        {GraphTest{createSdpaFwdGraph({4, 8, 256, 128},
                                      {4, 8, 256, 128},
                                      DataType::BFLOAT16,
                                      DataType::FLOAT,
                                      false,
                                      false,
                                      false,
                                      false,
                                      false,
                                      false,
                                      false,
                                      DataType::HALF),
                   "mma_core_mode = HALF"},
         false},
        {GraphTest{createSdpaFwdGraph({4, 8, 256, 128},
                                      {4, 8, 256, 128},
                                      DataType::BFLOAT16,
                                      DataType::FLOAT,
                                      false,
                                      false,
                                      false,
                                      false,
                                      false,
                                      false,
                                      false,
                                      DataType::FLOAT),
                   "mma_core_mode = FLOAT"},
         false}};

    for(const auto& [test, applicability] : applicabilityTests)
    {
        EXPECT_EQ(_planBuilder.isApplicable(_handle, test.graphWrapper()), applicability)
            << test.message;
    }
}

// =============================================================================
// Runtime pass-by-value scale tensor (RFC 0016)
// =============================================================================

// Build a forward SDPA graph with a runtime pass-by-value scale tensor
// (is_runtime_pass_by_value=true, no baked value).
flatbuffers::FlatBufferBuilder createSdpaFwdGraphWithRuntimePbvScale()
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

    // Runtime pass-by-value scale tensor: is_runtime_pass_by_value=true, no value
    const std::vector<int64_t> scaleDims = {1};
    const auto scaleUid = uid++;
    tensorAttributes.push_back(CreateTensorAttributesDirect(builder,
                                                            scaleUid,
                                                            "scale",
                                                            DataType::FLOAT,
                                                            &scaleDims,
                                                            &scaleDims,
                                                            false, // virtual
                                                            TensorValue::NONE, // no baked value
                                                            0, // value offset
                                                            true)); // is_runtime_pass_by_value

    const auto sdpaAttributes = CreateSdpaAttributes(builder,
                                                     qUid,
                                                     kUid,
                                                     vUid,
                                                     oUid,
                                                     flatbuffers::nullopt, // attn_mask
                                                     scaleUid); // scale_tensor_uid

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "sdpa_fwd",
                                     DataType::FLOAT,
                                     NodeAttributes::SdpaAttributes,
                                     sdpaAttributes.Union()));

    auto graphOffset = CreateGraphDirect(builder,
                                         "test",
                                         DataType::FLOAT,
                                         DataType::HALF,
                                         DataType::BFLOAT16,
                                         &tensorAttributes,
                                         &nodes);
    builder.Finish(graphOffset);
    return builder;
}

// Build a forward SDPA graph with a non-pass-by-value scale tensor (a regular
// device tensor, not a scalar). This should be rejected by isApplicable.
flatbuffers::FlatBufferBuilder createSdpaFwdGraphWithNonPbvScaleTensor()
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

    // Regular device tensor — NOT pass-by-value, NOT a scalar
    const std::vector<int64_t> scaleDims = {1};
    const auto scaleUid = uid++;
    tensorAttributes.push_back(CreateTensorAttributesDirect(
        builder, scaleUid, "scale", DataType::FLOAT, &scaleDims, &scaleDims));

    const auto sdpaAttributes = CreateSdpaAttributes(builder,
                                                     qUid,
                                                     kUid,
                                                     vUid,
                                                     oUid,
                                                     flatbuffers::nullopt, // attn_mask
                                                     scaleUid); // scale_tensor_uid

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "sdpa_fwd",
                                     DataType::FLOAT,
                                     NodeAttributes::SdpaAttributes,
                                     sdpaAttributes.Union()));

    auto graphOffset = CreateGraphDirect(builder,
                                         "test",
                                         DataType::FLOAT,
                                         DataType::HALF,
                                         DataType::BFLOAT16,
                                         &tensorAttributes,
                                         &nodes);
    builder.Finish(graphOffset);
    return builder;
}

TEST_F(TestSdpaFwdPlanBuilder, IsApplicableAcceptsRuntimePassByValueScale)
{
    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    auto builder = createSdpaFwdGraphWithRuntimePbvScale();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaFwdPlanBuilder, IsApplicableRejectsNonPassByValueScaleTensor)
{
    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    auto builder = createSdpaFwdGraphWithNonPbvScaleTensor();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaFwdPlanBuilder, IsApplicableAcceptsCompileTimeConstantScaleTensor)
{
    // Existing withScale=true path uses a compile-time constant scale tensor
    // (Float32Value baked in). Verify it still passes after the PBV changes.
    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    auto builder = createSdpaFwdGraph({4, 8, 256, 128},
                                      {4, 8, 256, 128},
                                      hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                                      hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                      /*withAttnMask=*/false,
                                      /*withScale=*/true);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_handle, graphWrapper));
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
                               DataType::UNSET, // mma_core_mode
                               AttentionImplementation::AUTO);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "sdpa_fwd",
                                     DataType::FLOAT,
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

TEST_F(TestSdpaFwdPlanBuilder, IsApplicableRejectsCausalMaskAndBottomRightSetTogether)
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

TEST_F(TestSdpaFwdPlanBuilder, IsApplicablePrefersCausalMaskOverWindowBounds)
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

TEST_F(TestSdpaFwdPlanBuilder, IsApplicablePrefersBottomRightCausalOverTopLeftBounds)
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

TEST_F(TestSdpaFwdPlanBuilder, IsApplicableAcceptsConsistentCausalMaskAndBounds)
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

TEST_F(TestSdpaFwdPlanBuilder, IsApplicablePrefersBottomRightCausalOverWindowBounds)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    // causal_mask_bottom_right=true takes precedence over the bounds trio, even
    // though a symmetric sliding window (left=64, right=64) would derive
    // SLIDING_WINDOW: result is bottom-right causal.
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

TEST_F(TestSdpaFwdPlanBuilder, MaskBoundsTrioRightZeroLeftUnsetDerivesTopLeftCausal)
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

TEST_F(TestSdpaFwdPlanBuilder, MaskBoundsTrioRightZeroBottomRightDerivesBottomRightCausal)
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

TEST_F(TestSdpaFwdPlanBuilder, MaskBoundsTrioExplicitCausalBoundsDeriveTopLeftCausal)
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

TEST_F(TestSdpaFwdPlanBuilder, MaskBoundsTrioBothUnsetDerivesNoMask)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    auto builder = createSdpaFwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/false,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        DiagonalAlignment::TOP_LEFT);

    plan_utils::MaskType maskType = plan_utils::MaskType::SLIDING_WINDOW;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::NO_MASK);
}

TEST_F(TestSdpaFwdPlanBuilder, MaskBoundsTrioBothUnboundedDerivesNoMask)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    auto builder = createSdpaFwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/false,
        flatbuffers::Optional<int64_t>(-1),
        flatbuffers::Optional<int64_t>(-1),
        DiagonalAlignment::TOP_LEFT);

    plan_utils::MaskType maskType = plan_utils::MaskType::SLIDING_WINDOW;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::NO_MASK);
}

TEST_F(TestSdpaFwdPlanBuilder, MaskBoundsTrioSymmetricWindowDerivesSlidingWindow)
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
    EXPECT_EQ(maskType, plan_utils::MaskType::SLIDING_WINDOW);
}

TEST_F(TestSdpaFwdPlanBuilder, MaskBoundsTrioLeftOnlyDerivesSlidingWindow)
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
    EXPECT_EQ(maskType, plan_utils::MaskType::SLIDING_WINDOW);
}

// =============================================================================
// Byte-stride overflow guard
// =============================================================================

// Helper: create a forward SDPA graph with explicit strides for all tensors.
auto createSdpaFwdGraphWithStrides(const std::vector<int64_t>& dims,
                                   const std::vector<int64_t>& qStrides,
                                   const std::vector<int64_t>& kStrides,
                                   const std::vector<int64_t>& vStrides,
                                   const std::vector<int64_t>& oStrides)
{
    const std::vector<int64_t> kDims = {dims[0], dims[1], dims[2], dims[3]};
    const std::vector<int64_t> oDims = {dims[0], dims[1], dims[2], dims[3]};

    return hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        dims,
        qStrides,
        kDims,
        kStrides,
        dims,
        vStrides,
        oDims,
        oStrides,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
}

TEST_F(TestSdpaFwdPlanBuilder, IsApplicableRejectsOversizedByteStrides)
{
    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    // A batch stride just above UINT32_MAX / 2 will overflow when scaled to
    // bytes (stride * sizeof(bf16)).  The engine must decline via isApplicable.
    constexpr int64_t K_OVERFLOW_STRIDE = static_cast<int64_t>(UINT32_MAX) / 2 + 1;
    const std::vector<int64_t> dims = {4, 8, 256, 128};
    const std::vector<int64_t> overflowStrides
        = {K_OVERFLOW_STRIDE, static_cast<int64_t>(256 * 128), 128, 1};
    const std::vector<int64_t> normalStrides = hipdnn_data_sdk::utilities::generateStrides(dims);

    // Q has overflow stride
    auto builderQ = createSdpaFwdGraphWithStrides(
        dims, overflowStrides, normalStrides, normalStrides, normalStrides);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper gwQ(
        builderQ.GetBufferPointer(), builderQ.GetSize());
    EXPECT_FALSE(_planBuilder.isApplicable(_handle, gwQ));

    // K has overflow stride
    auto builderK = createSdpaFwdGraphWithStrides(
        dims, normalStrides, overflowStrides, normalStrides, normalStrides);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper gwK(
        builderK.GetBufferPointer(), builderK.GetSize());
    EXPECT_FALSE(_planBuilder.isApplicable(_handle, gwK));

    // Normal strides pass
    auto builderOk = createSdpaFwdGraphWithStrides(
        dims, normalStrides, normalStrides, normalStrides, normalStrides);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper gwOk(
        builderOk.GetBufferPointer(), builderOk.GetSize());
    EXPECT_TRUE(_planBuilder.isApplicable(_handle, gwOk));
}

// =============================================================================
// buildPlan exception contract (IPlanBuilder::buildPlan)
// =============================================================================

TEST_F(TestSdpaFwdPlanBuilder, BuildPlanThrowsForUnsupportedDtype)
{
    SKIP_IF_NO_DEVICES();

    // HALF is not a supported input dtype for the forward ASM SDPA kernels, so
    // the registry lookup will fail and buildPlan must throw
    // HipdnnPluginException rather than silently returning.
    auto builder = createSdpaFwdGraph(
        {4, 8, 256, 128}, {4, 8, 256, 128}, hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    Context ctx;
    const hipdnn_test_sdk::utilities::MockEngineConfig mockEngineConfig;

    EXPECT_THROW(_planBuilder.buildPlan(_handle, graphWrapper, mockEngineConfig, ctx),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestSdpaFwdPlanBuilder, BuildPlanThrowsOnEmptyKernelKey)
{
    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    // Use an unsupported head dimension so the kernel registry lookup returns
    // an empty key.  isApplicable would reject this, but buildPlan must also
    // throw rather than crashing via cfg_fmha_fwd.at("").
    auto builder = createSdpaFwdGraph({4, 8, 256, 100});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper gw(builder.GetBufferPointer(),
                                                                        builder.GetSize());

    // Forward ignores engineConfig — a null-buffer wrapper suffices.
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper dummyCfg(nullptr, 0);
    Context context;
    EXPECT_THROW(_planBuilder.buildPlan(_handle, gw, dummyCfg, context),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

// =============================================================================
// initializeExecutionSettings
// =============================================================================

TEST_F(TestSdpaFwdPlanBuilder, InitializeExecutionSettingsIsNoOp)
{
    auto builder = createSdpaFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper gw(builder.GetBufferPointer(),
                                                                        builder.GetSize());

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper dummyCfg(nullptr, 0);
    Settings settings;
    // Must not throw or log an error — forward has no knobs.
    EXPECT_NO_THROW(_planBuilder.initializeExecutionSettings(_handle, gw, dummyCfg, settings));
}

} // namespace
} // namespace asm_sdpa_engine
