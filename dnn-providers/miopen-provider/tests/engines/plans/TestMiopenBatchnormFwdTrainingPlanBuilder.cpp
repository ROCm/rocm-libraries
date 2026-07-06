// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <memory>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipdnnMiopenHandle.hpp"
#include "engines/plans/MiopenBatchnormFwdTrainingPlanBuilder.hpp"

using namespace miopen_plugin;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

class TestMiopenBatchnormFwdTrainingPlanBuilder : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        _dummyHandle = std::make_unique<HipdnnMiopenHandle>();
    }

    MiopenBatchnormFwdTrainingPlanBuilder _planBuilder;
    std::unique_ptr<HipdnnMiopenHandle> _dummyHandle;
};

// ============================================================================
// Basic Applicability Tests
// ============================================================================

TEST_F(TestMiopenBatchnormFwdTrainingPlanBuilder, IsApplicableReturnsTrueForValidSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const bool applicable = _planBuilder.isApplicable(*_dummyHandle, graph);

    EXPECT_TRUE(applicable);
}

TEST_F(TestMiopenBatchnormFwdTrainingPlanBuilder, IsApplicableReturnsTrueForValidTwoNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingActivGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const bool applicable = _planBuilder.isApplicable(*_dummyHandle, graph);

    EXPECT_TRUE(applicable);
}

TEST_F(TestMiopenBatchnormFwdTrainingPlanBuilder, IsApplicableReturnsFalseForThreeNodeGraph)
{
    const MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(3));

    const bool applicable = _planBuilder.isApplicable(*_dummyHandle, mockGraph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormFwdTrainingPlanBuilder, IsApplicableReturnsFalseForWrongNodeType)
{
    // Create a graph with wrong node type
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // Node with wrong type (Convolution instead of Batchnorm)
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "conv",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes,
        0);
    nodes.push_back(node);

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const bool applicable = _planBuilder.isApplicable(*_dummyHandle, graph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormFwdTrainingPlanBuilder, IsApplicableReturnsFalseForUnsupportedComputeType)
{
    const flatbuffers::FlatBufferBuilder builder
        = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingActivGraph();

    auto mutableGraph
        = hipdnn_flatbuffers_sdk::data_objects::GetMutableGraph(builder.GetBufferPointer());
    mutableGraph->mutable_nodes()->GetMutableObject(1)->mutate_compute_data_type(
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

// ============================================================================
// Two-Node Fusion Tests
// ============================================================================

TEST_F(TestMiopenBatchnormFwdTrainingPlanBuilder, IsApplicableReturnsFalseForNonReluActivation)
{
    // Create a graph with unsupported activation (e.g., SIGMOID_FWD)
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingActivGraph(
        false, false, hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SIGMOID_FWD);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const bool applicable = _planBuilder.isApplicable(*_dummyHandle, graph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormFwdTrainingPlanBuilder, IsApplicableReturnsFalseForWrongNodeOrder)
{
    // Create a graph with 2 nodes but in wrong order (activation -> batchnorm)
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // Wrong order: activation first, then batchnorm
    auto node0 = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "pointwise",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        0);
    nodes.push_back(node0);

    auto node1 = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_training",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        0);
    nodes.push_back(node1);

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const bool applicable = _planBuilder.isApplicable(*_dummyHandle, graph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormFwdTrainingPlanBuilder,
       IsApplicableReturnsFalseWhenBnOutputDoesNotMatchActivationInput)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> strides = {1, 3, 14, 14};
    const std::vector<int64_t> dims = {1, 3, 14, 14};
    const std::vector<int64_t> derivedStrides = {1, 3, 1, 1};
    const std::vector<int64_t> derivedDims = {1, 3, 1, 1};

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &strides, &dims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        2,
        "y_bn",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims,
        true));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        6,
        "y_wrong",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims,
        true));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "bias",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    const hipdnn_flatbuffers_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributes(
        builder,
        5,
        builder.CreateString("epsilon"),
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        0,
        0,
        false,
        hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        7,
        "final_out",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims));

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // BN node with output tensor 2
    auto bnormAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormAttributes(builder,
                                                                          1,
                                                                          3,
                                                                          4,
                                                                          5,
                                                                          0,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          2, // BN output
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnormAttributes.Union()));

    // Activation with input tensor 6 (wrong - should be 2)
    auto actAttr = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        6, // Wrong input - doesn't match BN output
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        7);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "act",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        actAttr.Union()));

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const bool applicable = _planBuilder.isApplicable(*_dummyHandle, graph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormFwdTrainingPlanBuilder,
       IsApplicableReturnsFalseWhenBnOutputIsNotVirtualInFusion)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> strides = {1, 3, 14, 14};
    const std::vector<int64_t> dims = {1, 3, 14, 14};
    const std::vector<int64_t> derivedStrides = {1, 3, 1, 1};
    const std::vector<int64_t> derivedDims = {1, 3, 1, 1};

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &strides, &dims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        2,
        "y_bn",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims,
        false)); // Non-virtual - wrong!
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "bias",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    const hipdnn_flatbuffers_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributes(
        builder,
        5,
        builder.CreateString("epsilon"),
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        0,
        0,
        false,
        hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        7,
        "final_out",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims));

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    auto bnormAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormAttributes(builder,
                                                                          1,
                                                                          3,
                                                                          4,
                                                                          5,
                                                                          0,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          2,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnormAttributes.Union()));

    auto actAttr = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        7);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "act",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        actAttr.Union()));

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const bool applicable = _planBuilder.isApplicable(*_dummyHandle, graph);

    EXPECT_FALSE(applicable);
}

// ============================================================================
// Workspace and Plan Building Tests
// ============================================================================

TEST_F(TestMiopenBatchnormFwdTrainingPlanBuilder, GetWorkspaceSizeReturnsZero)
{
    const MockGraph mockGraph;

    const HipdnnMiopenSettings settings;
    const size_t workspaceSize
        = _planBuilder.getMaxWorkspaceSize(*_dummyHandle, mockGraph, settings);

    EXPECT_EQ(workspaceSize, 0u);
}

TEST_F(TestMiopenBatchnormFwdTrainingPlanBuilder, BuildPlanSetsPlanForValidSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    HipdnnMiopenContext ctx;
    const MockEngineConfig mockEngineConfig;

    EXPECT_NO_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, mockEngineConfig, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormFwdTrainingPlanBuilder, BuildPlanSetsPlanForValidTwoNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingActivGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    HipdnnMiopenContext ctx;
    const MockEngineConfig mockEngineConfig;

    EXPECT_NO_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, mockEngineConfig, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormFwdTrainingPlanBuilder, BuildPlanThrowsForUnsupportedNodeCount)
{
    // Create a graph with 3 nodes (unsupported)
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    for(int i = 0; i < 3; i++)
    {
        auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
            builder,
            "node",
            hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
            hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::NONE,
            0);
        nodes.push_back(node);
    }

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    HipdnnMiopenContext ctx;
    const MockEngineConfig mockEngineConfig;

    EXPECT_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, mockEngineConfig, ctx),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_FALSE(ctx.hasValidPlan());
}

// ============================================================================
// Validator Error Propagation Tests
// ============================================================================

TEST_F(TestMiopenBatchnormFwdTrainingPlanBuilder, IsApplicableReturnsFalseForInvalidLayout)
{
    // Create a graph with 3D tensors (invalid layout)
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> strides3D = {1, 3, 14}; // 3D instead of 4D/5D
    const std::vector<int64_t> dims3D = {1, 3, 14};
    const std::vector<int64_t> derivedStrides = {1, 3, 1};
    const std::vector<int64_t> derivedDims = {1, 3, 1};

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        1,
        "x",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides3D,
        &dims3D));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        2,
        "y",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides3D,
        &dims3D));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "bias",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    const hipdnn_flatbuffers_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributes(
        builder,
        5,
        builder.CreateString("epsilon"),
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        0,
        0,
        false,
        hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    auto bnormAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormAttributes(builder,
                                                                          1,
                                                                          3,
                                                                          4,
                                                                          5,
                                                                          0,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          2,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnormAttributes.Union());
    nodes.push_back(node);

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const bool applicable = _planBuilder.isApplicable(*_dummyHandle, graph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormFwdTrainingPlanBuilder, IsApplicableReturnsFalseForInvalidDataType)
{
    // Create a graph with INT32 data type (unsupported)
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> strides = {588, 196, 14, 1};
    const std::vector<int64_t> dims = {1, 3, 14, 14};
    const std::vector<int64_t> derivedStrides = {3, 1, 1, 1};
    const std::vector<int64_t> derivedDims = {1, 3, 1, 1};

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_flatbuffers_sdk::data_objects::DataType::INT32, &strides, &dims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", hipdnn_flatbuffers_sdk::data_objects::DataType::INT32, &strides, &dims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_flatbuffers_sdk::data_objects::DataType::INT32,
        &derivedStrides,
        &derivedDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "bias",
        hipdnn_flatbuffers_sdk::data_objects::DataType::INT32,
        &derivedStrides,
        &derivedDims));

    const hipdnn_flatbuffers_sdk::data_objects::Int32Value epsilonVal(1);
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributes(
        builder,
        5,
        builder.CreateString("epsilon"),
        hipdnn_flatbuffers_sdk::data_objects::DataType::INT32,
        0,
        0,
        false,
        hipdnn_flatbuffers_sdk::data_objects::TensorValue::Int32Value,
        builder.CreateStruct(epsilonVal).Union()));

    auto bnormAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormAttributes(builder,
                                                                          1,
                                                                          3,
                                                                          4,
                                                                          5,
                                                                          0,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          2,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnormAttributes.Union());
    nodes.push_back(node);

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const bool applicable = _planBuilder.isApplicable(*_dummyHandle, graph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormFwdTrainingPlanBuilder, IsApplicableReturnsFalseForInvalidShape)
{
    // Create a graph with wrong affine parameter shape
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> strides = {588, 196, 14, 1};
    const std::vector<int64_t> dims = {1, 3, 14, 14};
    const std::vector<int64_t> wrongStrides = {5, 1, 1, 1}; // Wrong channel count
    const std::vector<int64_t> wrongDims = {1, 5, 1, 1};

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &strides, &dims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &strides, &dims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &wrongStrides,
        &wrongDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "bias",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &wrongStrides,
        &wrongDims));

    const hipdnn_flatbuffers_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributes(
        builder,
        5,
        builder.CreateString("epsilon"),
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        0,
        0,
        false,
        hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    auto bnormAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormAttributes(builder,
                                                                          1,
                                                                          3,
                                                                          4,
                                                                          5,
                                                                          0,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          2,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnormAttributes.Union());
    nodes.push_back(node);

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const bool applicable = _planBuilder.isApplicable(*_dummyHandle, graph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormFwdTrainingPlanBuilder, IsApplicableReturnsFalseForNonPackedTensor)
{
    // Create a graph with non-packed tensor (strides don't match packed layout)
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> nonPackedStrides = {1000, 3, 14, 14}; // Non-packed strides
    const std::vector<int64_t> dims = {1, 3, 14, 14};
    const std::vector<int64_t> derivedStrides = {1, 3, 1, 1};
    const std::vector<int64_t> derivedDims = {1, 3, 1, 1};

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        1,
        "x",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &nonPackedStrides,
        &dims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        2,
        "y",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &nonPackedStrides,
        &dims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "bias",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    const hipdnn_flatbuffers_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributes(
        builder,
        5,
        builder.CreateString("epsilon"),
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        0,
        0,
        false,
        hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    auto bnormAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormAttributes(builder,
                                                                          1,
                                                                          3,
                                                                          4,
                                                                          5,
                                                                          0,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          2,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnormAttributes.Union());
    nodes.push_back(node);

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const bool applicable = _planBuilder.isApplicable(*_dummyHandle, graph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormFwdTrainingPlanBuilder,
       IsApplicableReturnsFalseForInsufficientSpatialDims)
{
    // Create a graph with only 1 spatial dimension (NC layout, need at least 2)
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> strides = {3, 1}; // Only N and C dimensions
    const std::vector<int64_t> dims = {1, 3};
    const std::vector<int64_t> derivedStrides = {3, 1};
    const std::vector<int64_t> derivedDims = {1, 3};

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &strides, &dims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &strides, &dims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "bias",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    const hipdnn_flatbuffers_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributes(
        builder,
        5,
        builder.CreateString("epsilon"),
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        0,
        0,
        false,
        hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    auto bnormAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormAttributes(builder,
                                                                          1,
                                                                          3,
                                                                          4,
                                                                          5,
                                                                          0,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          2,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnormAttributes.Union());
    nodes.push_back(node);

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const bool applicable = _planBuilder.isApplicable(*_dummyHandle, graph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormFwdTrainingPlanBuilder, IsApplicableReturnsFalseForMixedLayouts)
{
    // Create a graph with mixed NCHW and NHWC layouts
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> stridesNCHW = {588, 196, 14, 1}; // NCHW
    const std::vector<int64_t> dimsNCHW = {1, 3, 14, 14};
    const std::vector<int64_t> stridesNHWC = {3, 14L * 14L * 3L, 14L * 3L, 3}; // NHWC
    const std::vector<int64_t> dimsNHWC = {1, 3, 14, 14};
    const std::vector<int64_t> derivedStrides = {1, 3, 1, 1};
    const std::vector<int64_t> derivedDims = {1, 3, 1, 1};

    // Input in NCHW
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        1,
        "x",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &stridesNCHW,
        &dimsNCHW));
    // Output in NHWC (mixed layouts)
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        2,
        "y",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &stridesNHWC,
        &dimsNHWC));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "bias",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    const hipdnn_flatbuffers_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributes(
        builder,
        5,
        builder.CreateString("epsilon"),
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        0,
        0,
        false,
        hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    auto bnormAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormAttributes(builder,
                                                                          1,
                                                                          3,
                                                                          4,
                                                                          5,
                                                                          0,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          2,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnormAttributes.Union());
    nodes.push_back(node);

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const bool applicable = _planBuilder.isApplicable(*_dummyHandle, graph);

    EXPECT_FALSE(applicable);
}
