// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "engines/hip_mlops_engine/plans/batchnorm/BatchnormBwdPlan.hpp"

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hip_kernel_provider::batchnorm;

TEST(TestBatchnormBwdParams, InitializesAllTensorsFromValidBwdGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    const BatchnormBwdParams params(*attrs, graph.getTensorMap());
    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.dy(), nullptr);
    EXPECT_NE(params.dx(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.dscale(), nullptr);
    EXPECT_NE(params.dbias(), nullptr);
    EXPECT_TRUE(params.hasSavedStats());
    EXPECT_NE(params.savedMean(), nullptr);
    EXPECT_NE(params.savedInvVariance(), nullptr);
    EXPECT_FALSE(params.optActivation().has_value());
    EXPECT_EQ(params.bias(), nullptr);
}

TEST(TestBatchnormBwdParams, HandlesMissingOptionalSavedStats)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph(
        {1, 1, 1, 1}, {1, 1, 1, 1}, false);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    const BatchnormBwdParams params(*attrs, graph.getTensorMap());
    EXPECT_FALSE(params.hasSavedStats());
    EXPECT_EQ(params.savedMean(), nullptr);
    EXPECT_EQ(params.savedInvVariance(), nullptr);
}

TEST(TestBatchnormBwdParams, InitializesFusedActivationWithAllTensors)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferActBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    ASSERT_EQ(graph.nodeCount(), 3u);

    const auto& bnInfNode = graph.getNode(0);
    const auto& pointwiseNode = graph.getNode(1);
    const auto& bnBwdNode = graph.getNode(2);

    auto* bnInfAttrs = bnInfNode.attributes_as_BatchnormInferenceAttributes();
    auto* pointwiseAttrs = pointwiseNode.attributes_as_PointwiseAttributes();
    auto* bnBwdAttrs = bnBwdNode.attributes_as_BatchnormBackwardAttributes();

    ASSERT_NE(bnInfAttrs, nullptr);
    ASSERT_NE(pointwiseAttrs, nullptr);
    ASSERT_NE(bnBwdAttrs, nullptr);

    const BatchnormBwdParams params(
        *bnBwdAttrs, *pointwiseAttrs, *bnInfAttrs, graph.getTensorMap());

    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.dy(), nullptr);
    EXPECT_NE(params.dx(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.dscale(), nullptr);
    EXPECT_NE(params.dbias(), nullptr);
    EXPECT_TRUE(params.hasSavedStats());
    EXPECT_TRUE(params.optActivation().has_value());
    EXPECT_NE(params.bias(), nullptr);
    EXPECT_EQ(params.dy()->uid(), pointwiseAttrs->in_0_tensor_uid());
}

TEST(TestBatchnormBwdPlan, GetWorkspaceSizeReturnsZero)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    BatchnormBwdParams params(*attrs, graph.getTensorMap());
    const BatchnormBwdPlan plan(std::move(params));
    const HipKernelHandle handle;
    EXPECT_EQ(plan.getWorkspaceSize(handle), 0u);
}

TEST(TestBatchnormBwdPlan, FusedModeHasActivationAndBias)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferActBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& bnInfNode = graph.getNode(0);
    const auto& pointwiseNode = graph.getNode(1);
    const auto& bnBwdNode = graph.getNode(2);

    auto* bnInfAttrs = bnInfNode.attributes_as_BatchnormInferenceAttributes();
    auto* pointwiseAttrs = pointwiseNode.attributes_as_PointwiseAttributes();
    auto* bnBwdAttrs = bnBwdNode.attributes_as_BatchnormBackwardAttributes();

    ASSERT_NE(bnInfAttrs, nullptr);
    ASSERT_NE(pointwiseAttrs, nullptr);
    ASSERT_NE(bnBwdAttrs, nullptr);

    const BatchnormBwdParams params(
        *bnBwdAttrs, *pointwiseAttrs, *bnInfAttrs, graph.getTensorMap());
    EXPECT_TRUE(params.optActivation().has_value());
    EXPECT_NE(params.bias(), nullptr);
}
