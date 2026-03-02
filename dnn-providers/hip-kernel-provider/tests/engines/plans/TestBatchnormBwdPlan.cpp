// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/plans/BatchnormBwdPlan.hpp"
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hip_kernel_plugin;

TEST(TestBatchnormBwdParams, InitializesAllTensorsFromValidBwdGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    EXPECT_NO_THROW(BatchnormBwdParams(*attrs, graph.getTensorMap()));

    BatchnormBwdParams params(*attrs, graph.getTensorMap());
    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.dy(), nullptr);
    EXPECT_NE(params.dx(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.dscale(), nullptr);
    EXPECT_NE(params.dbias(), nullptr);
    EXPECT_NE(params.savedMean(), nullptr);
    EXPECT_NE(params.savedInvVariance(), nullptr);
    EXPECT_EQ(params.optActivation(), std::nullopt);
    EXPECT_EQ(params.bias(), nullptr);
}

TEST(TestBatchnormBwdParams, InitializesFusedActivationWithAllTensors)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferActBwdGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

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

    BatchnormBwdParams params(*bnInfAttrs, *pointwiseAttrs, *bnBwdAttrs, graph.getTensorMap());

    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.dy(), nullptr);
    EXPECT_NE(params.dx(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.dscale(), nullptr);
    EXPECT_NE(params.dbias(), nullptr);
    EXPECT_NE(params.savedMean(), nullptr);
    EXPECT_NE(params.savedInvVariance(), nullptr);
    EXPECT_TRUE(params.optActivation().has_value());
    EXPECT_NE(params.bias(), nullptr);
}
