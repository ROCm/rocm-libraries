/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */
/* SPDX-License-Identifier:  MIT */

#include <gtest/gtest.h>
#include <numeric>

#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/EnginePluginApi.h>
#include <hipdnn_sdk/plugin/test_utils/MockGraph.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>

#include "HipdnnEnginePluginHandle.hpp"
#include "engines/plans/MiopenBatchnormPlanBuilder.hpp"

#include "mocks/MockHipdnnEnginePluginExecutionContext.hpp"

using namespace miopen_legacy_plugin;
using namespace hipdnn_plugin;

class TestMiopenBatchnormPlanBuilder : public ::testing::Test
{
protected:
    MiopenBatchnormPlanBuilder _planBuilder;
    HipdnnEnginePluginHandle _dummyHandle;
};

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsFalseForMultiNodeGraph)
{
    MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(2));

    bool applicable = _planBuilder.isApplicable(_dummyHandle, mockGraph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsFalseForUnsupportedAttributes)
{
    MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillOnce(::testing::Return(1));
    EXPECT_CALL(mockGraph, hasOnlySupportedAttributes(::testing::_))
        .WillOnce(::testing::Return(false));

    bool applicable = _planBuilder.isApplicable(_dummyHandle, mockGraph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsTrueForSupportedSingleNodeGraph)
{
    MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillOnce(::testing::Return(1));
    EXPECT_CALL(mockGraph, hasOnlySupportedAttributes(::testing::_))
        .WillOnce(::testing::Return(true));

    bool applicable = _planBuilder.isApplicable(_dummyHandle, mockGraph);

    EXPECT_TRUE(applicable);
}

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsTrueForFusedThreeNodeGraph)
{
    // Use a real flatbuffer graph with valid fusion pattern
    auto builder = hipdnn_sdk::test_utilities::createValidBatchnormInferActBwdGraph();
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    bool applicable = _planBuilder.isApplicable(_dummyHandle, graph);

    EXPECT_TRUE(applicable);
}

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsFalseForIncorrectThreeNodeOrder)
{
    // Create a graph with 3 nodes but in wrong order
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;

    // Wrong order: activation -> batchnorm inference -> batchnorm backward
    auto node0 = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder, "pointwise", hipdnn_sdk::data_objects::NodeAttributes::PointwiseAttributes, 0);
    nodes.push_back(node0);

    auto node1 = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_inference",
        hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        0);
    nodes.push_back(node1);

    auto node2 = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_backward",
        hipdnn_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        0);
    nodes.push_back(node2);

    auto graphOffset
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "test",
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::HALF,
                                                      hipdnn_sdk::data_objects::DataType::BFLOAT16,
                                                      &tensorAttributes,
                                                      &nodes);
    builder.Finish(graphOffset);

    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    bool applicable = _planBuilder.isApplicable(_dummyHandle, graph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsFalseForFourNodeGraph)
{
    MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(4));

    bool applicable = _planBuilder.isApplicable(_dummyHandle, mockGraph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormPlanBuilder, GetWorkspaceSizeReturnsExpectedValue)
{
    MockGraph mockGraph;

    size_t workspaceSize = _planBuilder.getWorkspaceSize(_dummyHandle, mockGraph);

    EXPECT_EQ(workspaceSize, 0u);
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanSetsPlanForSupportedInferenceNode)
{
    // Use a real flatbuffer graph with a valid batchnorm inference node
    auto builder = hipdnn_sdk::test_utilities::createValidBatchnormInferenceGraph();
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    HipdnnEnginePluginExecutionContext ctx;

    // Should not throw
    EXPECT_NO_THROW(_planBuilder.buildPlan(_dummyHandle, graph, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanSetsPlanForSupportedBackwardNode)
{
    // Use a real flatbuffer graph with a valid batchnorm backward node
    auto builder = hipdnn_sdk::test_utilities::createValidBatchnormBwdGraph();
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    HipdnnEnginePluginExecutionContext ctx;

    // Should not throw
    EXPECT_NO_THROW(_planBuilder.buildPlan(_dummyHandle, graph, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanSetsPlanForFusedThreeNodeGraph)
{
    // Use a real flatbuffer graph with valid fusion pattern
    auto builder = hipdnn_sdk::test_utilities::createValidBatchnormInferActBwdGraph();
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    HipdnnEnginePluginExecutionContext ctx;

    // Should not throw
    EXPECT_NO_THROW(_planBuilder.buildPlan(_dummyHandle, graph, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanThrowsForUnsupportedNodeType)
{
    // Create a graph with a node of unsupported type
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;

    // Node with NONE attributes type
    auto node = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder, "unsupported", hipdnn_sdk::data_objects::NodeAttributes::NONE, 0);
    nodes.push_back(node);

    auto graphOffset
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "test",
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::HALF,
                                                      hipdnn_sdk::data_objects::DataType::BFLOAT16,
                                                      &tensorAttributes,
                                                      &nodes);
    builder.Finish(graphOffset);

    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_THROW(_planBuilder.buildPlan(_dummyHandle, graph, ctx),
                 hipdnn_plugin::HipdnnPluginException);
    EXPECT_FALSE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanThrowsForMalformedInferenceAttributes)
{
    // Create a graph with batchnorm inference node but malformed attributes
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;

    // Node with BatchnormInferenceAttributes type but null attributes pointer
    auto node = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "malformed_bn_inference",
        hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        0);
    nodes.push_back(node);

    auto graphOffset
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "test",
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::HALF,
                                                      hipdnn_sdk::data_objects::DataType::BFLOAT16,
                                                      &tensorAttributes,
                                                      &nodes);
    builder.Finish(graphOffset);

    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_THROW(_planBuilder.buildPlan(_dummyHandle, graph, ctx),
                 hipdnn_plugin::HipdnnPluginException);
    EXPECT_FALSE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanThrowsForMalformedBackwardAttributes)
{
    // Create a graph with batchnorm backward node but malformed attributes
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;

    // Node with BatchnormBackwardAttributes type but null attributes pointer
    auto node = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "malformed_bn_backward",
        hipdnn_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        0);
    nodes.push_back(node);

    auto graphOffset
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "test",
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::HALF,
                                                      hipdnn_sdk::data_objects::DataType::BFLOAT16,
                                                      &tensorAttributes,
                                                      &nodes);
    builder.Finish(graphOffset);

    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_THROW(_planBuilder.buildPlan(_dummyHandle, graph, ctx),
                 hipdnn_plugin::HipdnnPluginException);
    EXPECT_FALSE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanThrowsForMalformedFusedGraphFirstNode)
{
    // Create a 3-node graph with malformed first node (batchnorm inference)
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;

    // Malformed batchnorm inference (null attributes)
    auto node0 = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "malformed_bn_inference",
        hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        0);
    nodes.push_back(node0);

    // Valid pointwise
    auto node1 = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder, "pointwise", hipdnn_sdk::data_objects::NodeAttributes::PointwiseAttributes, 0);
    nodes.push_back(node1);

    // Valid batchnorm backward
    auto node2 = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_backward",
        hipdnn_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        0);
    nodes.push_back(node2);

    auto graphOffset
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "test",
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::HALF,
                                                      hipdnn_sdk::data_objects::DataType::BFLOAT16,
                                                      &tensorAttributes,
                                                      &nodes);
    builder.Finish(graphOffset);

    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_THROW(_planBuilder.buildPlan(_dummyHandle, graph, ctx),
                 hipdnn_plugin::HipdnnPluginException);
    EXPECT_FALSE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanThrowsForMalformedFusedGraphSecondNode)
{
    // Create a 3-node graph with malformed second node (pointwise)
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;

    // Valid batchnorm inference
    auto node0 = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_inference",
        hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        0);
    nodes.push_back(node0);

    // Malformed pointwise (null attributes)
    auto node1 = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "malformed_pointwise",
        hipdnn_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        0);
    nodes.push_back(node1);

    // Valid batchnorm backward
    auto node2 = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_backward",
        hipdnn_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        0);
    nodes.push_back(node2);

    auto graphOffset
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "test",
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::HALF,
                                                      hipdnn_sdk::data_objects::DataType::BFLOAT16,
                                                      &tensorAttributes,
                                                      &nodes);
    builder.Finish(graphOffset);

    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_THROW(_planBuilder.buildPlan(_dummyHandle, graph, ctx),
                 hipdnn_plugin::HipdnnPluginException);
    EXPECT_FALSE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanThrowsForMalformedFusedGraphThirdNode)
{
    // Create a 3-node graph with malformed third node (batchnorm backward)
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;

    // Valid batchnorm inference
    auto node0 = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_inference",
        hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        0);
    nodes.push_back(node0);

    // Valid pointwise
    auto node1 = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder, "pointwise", hipdnn_sdk::data_objects::NodeAttributes::PointwiseAttributes, 0);
    nodes.push_back(node1);

    // Malformed batchnorm backward (null attributes)
    auto node2 = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "malformed_bn_backward",
        hipdnn_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        0);
    nodes.push_back(node2);

    auto graphOffset
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "test",
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::HALF,
                                                      hipdnn_sdk::data_objects::DataType::BFLOAT16,
                                                      &tensorAttributes,
                                                      &nodes);
    builder.Finish(graphOffset);

    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_THROW(_planBuilder.buildPlan(_dummyHandle, graph, ctx),
                 hipdnn_plugin::HipdnnPluginException);
    EXPECT_FALSE(ctx.hasValidPlan());
}
