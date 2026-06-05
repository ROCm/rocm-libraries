// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipdnnMiopenHandle.hpp"
#include "HipdnnMiopenSettings.hpp"
#include "engines/plans/MiopenBatchnormBwdPlan.hpp"
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

using namespace miopen_plugin;

TEST(TestBatchnormBwdParams, InitializesAllTensorsFromValidGraph)
{
    // Create a valid batchnorm graph
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the batchnorm node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params
    const BatchnormBwdParams params(*attrs, graph.getTensorMap());

    // All required tensors should be initialized
    EXPECT_NO_THROW(params.x());
    EXPECT_NO_THROW(params.dy());
    EXPECT_NO_THROW(params.dx());
    EXPECT_NO_THROW(params.scale());
    EXPECT_NO_THROW(params.dscale());
    EXPECT_NO_THROW(params.dbias());

    // Optional tensors should be present
    auto& meanOpt = params.optMean();
    auto& varOpt = params.optInvVariance();
    EXPECT_TRUE(meanOpt.has_value());
    EXPECT_TRUE(varOpt.has_value());
}

TEST(TestBatchnormBwdParams, HandlesMissingOptionalTensors)
{
    // Create a valid batchnorm graph and remove mean/variance from tensor map
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph(
        {1, 1, 1, 1}, {1, 1, 1, 1}, false // Set has_optional_attributes to false
    );
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the batchnorm node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    const auto& tensorMap = graph.getTensorMap();
    const BatchnormBwdParams params(*attrs, tensorMap);

    // Optional tensors should not be present
    EXPECT_FALSE(params.optMean().has_value());
    EXPECT_FALSE(params.optInvVariance().has_value());
}

TEST(TestBatchnormBwdParams, InitializesFusedActivationBiasWithAllTensors)
{
    // Create a fused batchnorm backward + activation + bias graph
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferActBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the three required nodes
    const auto& batchnormInfNode = graph.getNode(0);
    const auto& pointwiseNode = graph.getNode(1);
    const auto& batchnormBwdNode = graph.getNode(2);

    auto* batchnormInfAttrs = batchnormInfNode.attributes_as_BatchnormInferenceAttributes();
    auto* pointwiseAttrs = pointwiseNode.attributes_as_PointwiseAttributes();
    auto* batchnormBwdAttrs = batchnormBwdNode.attributes_as_BatchnormBackwardAttributes();

    ASSERT_NE(batchnormInfAttrs, nullptr);
    ASSERT_NE(pointwiseAttrs, nullptr);
    ASSERT_NE(batchnormBwdAttrs, nullptr);

    // Construct fused params
    const BatchnormBwdParams params(
        *batchnormBwdAttrs, *pointwiseAttrs, *batchnormInfAttrs, graph.getTensorMap());

    // All required tensors should be initialized
    EXPECT_NO_THROW(params.x());
    EXPECT_NO_THROW(params.dy());
    EXPECT_NO_THROW(params.dx());
    EXPECT_NO_THROW(params.scale());
    EXPECT_NO_THROW(params.dscale());
    EXPECT_NO_THROW(params.dbias());

    // Optional fusion-specific tensors should be present
    EXPECT_TRUE(params.optActivation().has_value());
    EXPECT_TRUE(params.optBias().has_value());
    EXPECT_EQ(params.dy().uid(), pointwiseAttrs->in_0_tensor_uid());

    // Optional mean/variance should be present if provided
    EXPECT_TRUE(params.optMean().has_value());
    EXPECT_TRUE(params.optInvVariance().has_value());
}

TEST(TestBatchnormBwdParams, FusedParamsHandlesMissingOptionalMeanVariance)
{
    // Create a fused graph without optional mean/variance
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferActBwdGraph(
        {1, 1, 1, 1}, {1, 1, 1, 1}, false);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& batchnormInfNode = graph.getNode(0);
    const auto& pointwiseNode = graph.getNode(1);
    const auto& batchnormBwdNode = graph.getNode(2);

    auto* batchnormInfAttrs = batchnormInfNode.attributes_as_BatchnormInferenceAttributes();
    auto* pointwiseAttrs = pointwiseNode.attributes_as_PointwiseAttributes();
    auto* batchnormBwdAttrs = batchnormBwdNode.attributes_as_BatchnormBackwardAttributes();

    ASSERT_NE(batchnormInfAttrs, nullptr);
    ASSERT_NE(pointwiseAttrs, nullptr);
    ASSERT_NE(batchnormBwdAttrs, nullptr);

    const BatchnormBwdParams params(
        *batchnormBwdAttrs, *pointwiseAttrs, *batchnormInfAttrs, graph.getTensorMap());

    // Fusion-specific tensors should still be present
    EXPECT_TRUE(params.optActivation().has_value());
    EXPECT_TRUE(params.optBias().has_value());

    // Optional mean/variance should not be present
    EXPECT_FALSE(params.optMean().has_value());
    EXPECT_FALSE(params.optInvVariance().has_value());
}

TEST(TestBatchnormBwdPlan, FusedModeHasActivationAndBias)
{
    // Create a fused batchnorm backward + activation + bias graph
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferActBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& batchnormInfNode = graph.getNode(0);
    const auto& pointwiseNode = graph.getNode(1);
    const auto& batchnormBwdNode = graph.getNode(2);

    auto* batchnormInfAttrs = batchnormInfNode.attributes_as_BatchnormInferenceAttributes();
    auto* pointwiseAttrs = pointwiseNode.attributes_as_PointwiseAttributes();
    auto* batchnormBwdAttrs = batchnormBwdNode.attributes_as_BatchnormBackwardAttributes();

    ASSERT_NE(batchnormInfAttrs, nullptr);
    ASSERT_NE(pointwiseAttrs, nullptr);
    ASSERT_NE(batchnormBwdAttrs, nullptr);

    const BatchnormBwdParams params(
        *batchnormBwdAttrs, *pointwiseAttrs, *batchnormInfAttrs, graph.getTensorMap());

    // In fused mode, activation and bias should be present
    EXPECT_TRUE(params.optActivation().has_value());
    EXPECT_TRUE(params.optBias().has_value());
}

TEST(TestBatchnormBwdPlan, GetWorkspaceSizeReturnsZeroForFusedMode)
{
    SKIP_IF_NO_DEVICES();

    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferActBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& batchnormInfNode = graph.getNode(0);
    const auto& pointwiseNode = graph.getNode(1);
    const auto& batchnormBwdNode = graph.getNode(2);

    auto* batchnormInfAttrs = batchnormInfNode.attributes_as_BatchnormInferenceAttributes();
    auto* pointwiseAttrs = pointwiseNode.attributes_as_PointwiseAttributes();
    auto* batchnormBwdAttrs = batchnormBwdNode.attributes_as_BatchnormBackwardAttributes();

    ASSERT_NE(batchnormInfAttrs, nullptr);
    ASSERT_NE(pointwiseAttrs, nullptr);
    ASSERT_NE(batchnormBwdAttrs, nullptr);

    BatchnormBwdParams params(
        *batchnormBwdAttrs, *pointwiseAttrs, *batchnormInfAttrs, graph.getTensorMap());
    const HipdnnMiopenSettings executionSettings;
    const BatchnormBwdPlan plan(std::move(params), executionSettings);

    const HipdnnMiopenHandle handle;
    EXPECT_EQ(plan.getWorkspaceSize(handle), 0);
}

TEST(TestBatchnormBwdPlan, GetWorkspaceSizeReturnsZero)
{
    SKIP_IF_NO_DEVICES();

    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    BatchnormBwdParams params(*attrs, graph.getTensorMap());
    const HipdnnMiopenSettings executionSettings;
    const BatchnormBwdPlan plan(std::move(params), executionSettings);

    const HipdnnMiopenHandle handle;
    EXPECT_EQ(plan.getWorkspaceSize(handle), 0);
}

TEST(TestBatchnormBwdPlan, OptionalTensorsAreNotPresentInBasicMode)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    const BatchnormBwdParams params(*attrs, graph.getTensorMap());

    // Basic mode should not have activation or bias
    EXPECT_FALSE(params.optActivation().has_value());
    EXPECT_FALSE(params.optBias().has_value());
}
