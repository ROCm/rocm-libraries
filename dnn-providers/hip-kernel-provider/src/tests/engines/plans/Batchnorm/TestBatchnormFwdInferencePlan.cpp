// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "engines/plans/BatchnormFwdInferencePlan.hpp"

#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hip_kernel_provider;

// ============================================================================
// BatchnormFwdInferenceParams - construction from valid graph data
// ============================================================================

TEST(TestBatchnormFwdInferenceParams, ConstructsFromSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributes();

    EXPECT_NO_THROW(BatchnormFwdInferenceParams params(attr, graph.getTensorMap()));
}

TEST(TestBatchnormFwdInferenceParams, HasCorrectTensorPointersForSingleNode)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributes();

    BatchnormFwdInferenceParams params(attr, graph.getTensorMap());

    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.y(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.bias(), nullptr);
    EXPECT_NE(params.estMean(), nullptr);
    EXPECT_NE(params.invVariance(), nullptr);
    EXPECT_FALSE(params.optActivation().has_value());
    EXPECT_EQ(params.activationOut(), nullptr);
}

TEST(TestBatchnormFwdInferenceParams, ConstructsFromFusedInferenceActivationGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdInferActGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    const auto& node0 = graph.getNodeWrapper(0);
    const auto& node1 = graph.getNodeWrapper(1);

    const auto& inferenceAttr
        = node0.attributesAs<hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes>();
    const auto& activationAttr
        = node1.attributesAs<hipdnn_data_sdk::data_objects::PointwiseAttributes>();

    EXPECT_NO_THROW(
        BatchnormFwdInferenceParams params(inferenceAttr, activationAttr, graph.getTensorMap()));
}

TEST(TestBatchnormFwdInferenceParams, FusedParamsHasActivation)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdInferActGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    const auto& node0 = graph.getNodeWrapper(0);
    const auto& node1 = graph.getNodeWrapper(1);

    const auto& inferenceAttr
        = node0.attributesAs<hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes>();
    const auto& activationAttr
        = node1.attributesAs<hipdnn_data_sdk::data_objects::PointwiseAttributes>();

    BatchnormFwdInferenceParams params(inferenceAttr, activationAttr, graph.getTensorMap());

    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.y(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.bias(), nullptr);
    EXPECT_NE(params.estMean(), nullptr);
    EXPECT_NE(params.invVariance(), nullptr);
    EXPECT_TRUE(params.optActivation().has_value());
    EXPECT_NE(params.activationOut(), nullptr);
}
