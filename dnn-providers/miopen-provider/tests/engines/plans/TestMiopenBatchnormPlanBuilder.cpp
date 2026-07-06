// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <memory>
#include <numeric>
#include <unordered_set>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_test_sdk/utilities/MockNode.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipdnnMiopenHandle.hpp"
#include "engines/plans/MiopenBatchnormPlanBuilder.hpp"

#include "mocks/MockHipdnnMiopenContext.hpp"

using namespace miopen_plugin;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

//tests in here
namespace
{

void createBatchnormFusionTensorAttributes(
    flatbuffers::FlatBufferBuilder& builder,
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>&
        tensorAttributes,
    int64_t numTensors,
    const std::unordered_set<int64_t>& virtualTensorIds)
{
    static const std::unordered_set<int64_t> s_derivedDimTensorIds = {2, 3, 4, 5, 8, 9, 12};

    const std::vector<int64_t> dims = {1, 3, 224, 224};
    const std::vector<int64_t> strides = {150528, 50176, 224, 1};
    const std::vector<int64_t> derivedDims = {1, 3, 1, 1};
    const std::vector<int64_t> derivedStrides = {3, 1, 1, 1};

    for(int64_t i = 1; i <= numTensors; ++i)
    {
        const auto isDerived = s_derivedDimTensorIds.count(i) > 0;
        const auto& tensorDims = isDerived ? derivedDims : dims;
        const auto& tensorStrides = isDerived ? derivedStrides : strides;
        const auto isVirtual = virtualTensorIds.count(i) > 0;

        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                i,
                ("tensor_" + std::to_string(i)).c_str(),
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                &tensorStrides,
                &tensorDims,
                isVirtual));
    }
}

} // namespace

class TestMiopenBatchnormPlanBuilder : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        _dummyHandle = std::make_unique<HipdnnMiopenHandle>();
    }

    MiopenBatchnormPlanBuilder _planBuilder;
    std::unique_ptr<HipdnnMiopenHandle> _dummyHandle;
    MockEngineConfig _mockEngineConfig;
};

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsFalseForGraphWithUnsupportedNodeCount)
{
    const MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(4));
    // nodeWrappers is only used in an all_of check which will pass when it's empty
    std::vector<std::unique_ptr<INodeWrapper>> nodeWrappers;
    EXPECT_CALL(mockGraph, nodeWrappers()).WillRepeatedly(::testing::ReturnRef(nodeWrappers));

    const bool applicable = _planBuilder.isApplicable(*_dummyHandle, mockGraph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsFalseForUnsupportedAttributes)
{
    const MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(1));
    EXPECT_CALL(mockGraph, hasOnlySupportedAttributes(::testing::_))
        .WillOnce(::testing::Return(false));
    // nodeWrappers is only used in an all_of check which will pass when it's empty
    std::vector<std::unique_ptr<INodeWrapper>> nodeWrappers;
    EXPECT_CALL(mockGraph, nodeWrappers()).WillRepeatedly(::testing::ReturnRef(nodeWrappers));

    const bool applicable = _planBuilder.isApplicable(*_dummyHandle, mockGraph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsTrueForFusedTwoNodeGraph)
{
    // Use a real flatbuffer graph with valid fusion pattern
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdInferActGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const bool applicable = _planBuilder.isApplicable(*_dummyHandle, graph);

    EXPECT_TRUE(applicable);
}

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsTrueForFusedThreeNodeGraph)
{
    // Use a real flatbuffer graph with valid fusion pattern
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferActBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const bool applicable = _planBuilder.isApplicable(*_dummyHandle, graph);

    EXPECT_TRUE(applicable);
}

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsFalseForIncorrectTwoNodeOrder)
{
    // Create a graph with 2 nodes but in wrong order
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // Wrong order: activation -> batchnorm inference
    auto node0 = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "pointwise",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        0);
    nodes.push_back(node0);

    auto node1 = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_inference",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
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

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsFalseForIncorrectThreeNodeOrder)
{
    // Create a graph with 3 nodes but in wrong order
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // Wrong order: activation -> batchnorm inference -> batchnorm backward
    auto node0 = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "pointwise",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        0);
    nodes.push_back(node0);

    auto node1 = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_inference",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        0);
    nodes.push_back(node1);

    auto node2 = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_backward",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        0);
    nodes.push_back(node2);

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

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsFalseForTwoNodeFusionUnsupportedActivatio)
{
    // Fusion graph with unsupported activation (e.g., MUL instead of RELU_FWD)
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    createBatchnormFusionTensorAttributes(builder, tensorAttributes, 7, {6});

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // BN inference
    auto bnInfAttr = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormInferenceAttributes(
        builder, 1, 4, 5, 2, 3, 6);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_inf",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        bnInfAttr.Union()));

    // Activation with unsupported MUL
    auto actAttr = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::MUL, // Unsupported!
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        6,
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

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

// ============================================================================
// Single-Node Backward Tests - Happy & Unhappy Paths
// ============================================================================

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsTrueForValidSingleNodeBackward)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const bool applicable = _planBuilder.isApplicable(*_dummyHandle, graph);

    EXPECT_TRUE(applicable);
}

TEST_F(TestMiopenBatchnormPlanBuilder, BackwardIsApplicableReturnsFalseForInvalidIoDataType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph(
        {150528, 50176, 224, 1},
        {1, 3, 224, 224},
        true,
        hipdnn_flatbuffers_sdk::data_objects::DataType::UINT8); // Invalid IO data type
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBatchnormPlanBuilder, BackwardIsApplicableReturnsFalseForInvalidScaleBiasDataType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph(
        {150528, 50176, 224, 1},
        {1, 3, 224, 224},
        true,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF); // Invalid scale/bias type
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBatchnormPlanBuilder, BackwardIsApplicableReturnsFalseForInvalidStatDataType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph(
        {150528, 50176, 224, 1},
        {1, 3, 224, 224},
        true,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF); // Invalid mean/variance type
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBatchnormPlanBuilder, BackwardIsApplicableReturnsFalseForInvalidLayout)
{
    // Create backward graph with 3D tensors in unsupported stride order (not NCL or NLC)
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    // Invalid stride order: {0, 1, 2} which is neither NCL {2, 1, 0} nor NLC {2, 0, 1}
    const std::vector<int64_t> strides3D = {1, 14, 42};
    const std::vector<int64_t> dims3D = {1, 3, 14};
    const std::vector<int64_t> derivedStrides = {3, 1, 1};
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
        "dy",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides3D,
        &dims3D));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "dx",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides3D,
        &dims3D));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "scale",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        5,
        "dscale",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        6,
        "dbias",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    auto bnBwdAttr = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormBackwardAttributes(
        builder,
        2,
        1,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        4,
        flatbuffers::Offset<flatbuffers::Vector<int64_t>>(),
        3,
        5,
        6);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_bwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        bnBwdAttr.Union());
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

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBatchnormPlanBuilder, BackwardIsApplicableReturnsFalseForInsufficientSpatialDims)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph(
        {3, 1, 1, 1}, {1, 3, 1, 1}); // Batch * spatial = 1 (invalid for training)
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

// ============================================================================
// Validator Error Propagation Tests - Inference
// ============================================================================

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsFalseForInvalidLayoutInference)
{
    // Create an inference graph with 3D tensors (invalid layout)
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> strides3D = {42, 14, 1}; // 3D instead of 4D/5D
    const std::vector<int64_t> dims3D = {1, 3, 14};
    const std::vector<int64_t> derivedStrides = {3, 1, 1, 1};
    const std::vector<int64_t> derivedDims = {1, 3, 1, 1};

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
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        5,
        "mean",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        6,
        "inv_variance",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    auto bnormAttributes = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormInferenceAttributes(
        builder, 1, 5, 6, 3, 4, 2);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
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

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

// ============================================================================
// Validator Error Propagation Tests - Backward
// ============================================================================

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsFalseForNonPackedTensorBackward)
{
    // Create a backward graph with non-packed tensors
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> nonPackedStrides = {1000, 300, 14, 1}; // Non-packed
    const std::vector<int64_t> dims = {1, 3, 14, 14};
    const std::vector<int64_t> derivedStrides = {3, 1, 1, 1};
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
        "dy",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &nonPackedStrides,
        &dims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "dx",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &nonPackedStrides,
        &dims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "scale",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        5,
        "dscale",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        6,
        "dbias",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    auto bnBwdAttr = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormBackwardAttributes(
        builder,
        2,
        1,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        4,
        flatbuffers::Offset<flatbuffers::Vector<int64_t>>(),
        3,
        5,
        6);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_bwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        bnBwdAttr.Union());
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

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsFalseForThreeNodeUnsupportedActivation)
{
    // Fusion graph with unsupported activation (e.g., SIGMOID_BWD instead of RELU_BWD)
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    createBatchnormFusionTensorAttributes(builder, tensorAttributes, 12, {10, 11});

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // BN inference
    auto bnInfAttr = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormInferenceAttributes(
        builder, 1, 4, 5, 2, 3, 10);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_inf",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        bnInfAttr.Union()));

    // Activation with unsupported SIGMOID_BWD
    auto actAttr = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SIGMOID_BWD, // Unsupported!
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        6,
        flatbuffers::Optional<int64_t>(10),
        flatbuffers::nullopt,
        11);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "act",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        actAttr.Union()));

    // BN backward
    auto bnBwdAttr = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormBackwardAttributes(
        builder,
        11,
        1,
        flatbuffers::Optional<int64_t>(4),
        flatbuffers::Optional<int64_t>(5),
        2,
        flatbuffers::Offset<flatbuffers::Vector<int64_t>>(),
        7,
        8,
        9);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_bwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        bnBwdAttr.Union()));

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

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsFalseWhenActivationMissingIn1)
{
    // Fusion graph where activation doesn't have in_1 tensor (required for backward)
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    createBatchnormFusionTensorAttributes(builder, tensorAttributes, 12, {10, 11});

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // BN inference
    auto bnInfAttr = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormInferenceAttributes(
        builder, 1, 4, 5, 2, 3, 10);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_inf",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        bnInfAttr.Union()));

    // Activation without in_1 (missing forward activation input!)
    auto actAttr = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        6,
        flatbuffers::nullopt, // Missing in_1!
        flatbuffers::nullopt,
        11);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "act",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        actAttr.Union()));

    // BN backward
    auto bnBwdAttr = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormBackwardAttributes(
        builder,
        11,
        1,
        flatbuffers::Optional<int64_t>(4),
        flatbuffers::Optional<int64_t>(5),
        2,
        flatbuffers::Offset<flatbuffers::Vector<int64_t>>(),
        7,
        8,
        9);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_bwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        bnBwdAttr.Union()));

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

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsFalseForFourNodeGraph)
{
    const MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(4));
    // nodeWrappers is only used in an all_of check which will pass when it's empty
    std::vector<std::unique_ptr<INodeWrapper>> nodeWrappers;
    EXPECT_CALL(mockGraph, nodeWrappers()).WillRepeatedly(::testing::ReturnRef(nodeWrappers));

    const bool applicable = _planBuilder.isApplicable(*_dummyHandle, mockGraph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsFalseForUnsupportedComputeType)
{
    const MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(2));
    auto nodeA = std::make_unique<MockNode>();
    auto nodeB = std::make_unique<MockNode>();
    EXPECT_CALL(*nodeA, computeDataType())
        .WillOnce(::testing::Return(hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT));
    EXPECT_CALL(*nodeB, computeDataType())
        .WillOnce(::testing::Return(hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16));

    std::vector<std::unique_ptr<INodeWrapper>> nodeWrappers;
    nodeWrappers.emplace_back(std::move(nodeA));
    nodeWrappers.emplace_back(std::move(nodeB));
    EXPECT_CALL(mockGraph, nodeWrappers()).WillRepeatedly(::testing::ReturnRef(nodeWrappers));

    const bool applicable = _planBuilder.isApplicable(*_dummyHandle, mockGraph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormPlanBuilder, GetMaxWorkspaceSizeReturnsExpectedValue)
{
    const MockGraph mockGraph;

    const HipdnnMiopenSettings settings;
    const size_t workspaceSize
        = _planBuilder.getMaxWorkspaceSize(*_dummyHandle, mockGraph, settings);

    EXPECT_EQ(workspaceSize, 0u);
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanSetsPlanForSupportedInferenceNode)
{
    // Use a real flatbuffer graph with a valid batchnorm node
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    HipdnnMiopenContext ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, _mockEngineConfig, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanSetsPlanForSupportedInferenceWithVarianceNode)
{
    // Use a real flatbuffer graph with a valid batchnorm variance node
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormWithVarianceInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    HipdnnMiopenContext ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, _mockEngineConfig, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanSetsPlanForSupportedBackwardNode)
{
    // Use a real flatbuffer graph with a valid batchnorm backward node
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    HipdnnMiopenContext ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, _mockEngineConfig, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanSetsPlanForFusedTwoNodeGraph)
{
    // Use a real flatbuffer graph with valid fusion pattern
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdInferActGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    HipdnnMiopenContext ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, _mockEngineConfig, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanSetsPlanForFusedThreeNodeGraph)
{
    // Use a real flatbuffer graph with valid fusion pattern
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferActBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    HipdnnMiopenContext ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, _mockEngineConfig, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanThrowsForUnsupportedNodeType)
{
    // Create a graph with a node of unsupported type
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // Node with NONE attributes type
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "unsupported",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::NONE,
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

    HipdnnMiopenContext ctx;

    EXPECT_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, _mockEngineConfig, ctx),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_FALSE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanThrowsForMalformedInferenceAttributes)
{
    // Create a graph with batchnorm inference node but malformed attributes
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // Node with BatchnormInferenceAttributes type but null attributes pointer
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "malformed_bn_inference",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
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
    HipdnnMiopenContext ctx;

    EXPECT_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, _mockEngineConfig, ctx),
                 std::invalid_argument);
    EXPECT_FALSE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanThrowsForMalformedBackwardAttributes)
{
    // Create a graph with batchnorm backward node but malformed attributes
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // Node with BatchnormBackwardAttributes type but null attributes pointer
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "malformed_bn_backward",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
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
    HipdnnMiopenContext ctx;

    EXPECT_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, _mockEngineConfig, ctx),
                 std::invalid_argument);
    EXPECT_FALSE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanThrowsForMalformedTwoNodeFusedGraphFirstNode)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    createBatchnormFusionTensorAttributes(builder, tensorAttributes, 7, {6});

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_inf",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        0)); // Malformed!

    auto actAttr = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        6,
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

    HipdnnMiopenContext ctx;

    EXPECT_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, _mockEngineConfig, ctx),
                 std::invalid_argument);
    EXPECT_FALSE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanThrowsForMalformedTwoNodeFusedGraphSecondNode)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    createBatchnormFusionTensorAttributes(builder, tensorAttributes, 7, {6});

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // BN inference
    auto bnInfAttr = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormInferenceAttributes(
        builder, 1, 4, 5, 2, 3, 6);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_inf",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        bnInfAttr.Union()));

    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "act",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        0)); //malformed!

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

    EXPECT_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, _mockEngineConfig, ctx),
                 std::invalid_argument);
    EXPECT_FALSE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanThrowsForMalformedFusedGraphFirstNode)
{
    // Create a 3-node graph with malformed first node (batchnorm inference)
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    createBatchnormFusionTensorAttributes(builder, tensorAttributes, 12, {10, 11});

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // Malformed batchnorm inference (null attributes)
    auto node0 = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "malformed_bn_inference",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        0);
    nodes.push_back(node0);

    // Valid pointwise
    auto actAttr = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        6,
        flatbuffers::Optional<int64_t>(10),
        flatbuffers::nullopt,
        11);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "pointwise",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        actAttr.Union()));

    // Valid batchnorm backward
    auto bnBwdAttr = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormBackwardAttributes(
        builder,
        11,
        1,
        flatbuffers::Optional<int64_t>(4),
        flatbuffers::Optional<int64_t>(5),
        2,
        flatbuffers::Offset<flatbuffers::Vector<int64_t>>(),
        7,
        8,
        9);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_backward",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        bnBwdAttr.Union()));

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

    EXPECT_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, _mockEngineConfig, ctx),
                 std::invalid_argument);
    EXPECT_FALSE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanThrowsForMalformedFusedGraphSecondNode)
{
    // Create a 3-node graph with malformed second node (pointwise)
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    createBatchnormFusionTensorAttributes(builder, tensorAttributes, 12, {10, 11});

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // Valid batchnorm inference
    auto bnInfAttr = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormInferenceAttributes(
        builder, 1, 4, 5, 2, 3, 10);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_inference",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        bnInfAttr.Union()));

    // Malformed pointwise (null attributes)
    auto node1 = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "malformed_pointwise",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        0);
    nodes.push_back(node1);

    // Valid batchnorm backward
    auto bnBwdAttr = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormBackwardAttributes(
        builder,
        11,
        1,
        flatbuffers::Optional<int64_t>(4),
        flatbuffers::Optional<int64_t>(5),
        2,
        flatbuffers::Offset<flatbuffers::Vector<int64_t>>(),
        7,
        8,
        9);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_backward",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        bnBwdAttr.Union()));

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

    EXPECT_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, _mockEngineConfig, ctx),
                 std::invalid_argument);
    EXPECT_FALSE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanThrowsForMalformedFusedGraphThirdNode)
{
    // Create a 3-node graph with malformed third node (batchnorm backward)
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    createBatchnormFusionTensorAttributes(builder, tensorAttributes, 12, {10, 11});

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // Valid batchnorm inference
    auto bnInfAttr = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormInferenceAttributes(
        builder, 1, 4, 5, 2, 3, 10);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_inference",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        bnInfAttr.Union()));

    // Valid pointwise
    auto actAttr = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        6,
        flatbuffers::Optional<int64_t>(10),
        flatbuffers::nullopt,
        11);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "pointwise",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        actAttr.Union()));

    // Malformed batchnorm backward (null attributes)
    auto node2 = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "malformed_bn_backward",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        0);
    nodes.push_back(node2);

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

    EXPECT_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, _mockEngineConfig, ctx),
                 std::invalid_argument);
    EXPECT_FALSE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormPlanBuilder,
       IsApplicableReturnsFalseWhenBnInferenceOutputDoesNotMatchActivationInput)
{
    // Fusion graph where BN inference output doesn't connect to activation
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    createBatchnormFusionTensorAttributes(builder, tensorAttributes, 12, {10, 11});

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // BN inference with output uid=10
    auto bnInfAttr = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormInferenceAttributes(
        builder, 1, 4, 5, 2, 3, 10);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_inf",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        bnInfAttr.Union()));

    // Activation with forward input uid=999
    auto actAttr = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        6,
        flatbuffers::Optional<int64_t>(999), // Wrong forward input
        flatbuffers::nullopt,
        11);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "act",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        actAttr.Union()));

    // BN backward
    auto bnBwdAttr = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormBackwardAttributes(
        builder,
        11,
        1,
        flatbuffers::Optional<int64_t>(4),
        flatbuffers::Optional<int64_t>(5),
        2,
        flatbuffers::Offset<flatbuffers::Vector<int64_t>>(),
        7,
        8,
        9);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_bwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        bnBwdAttr.Union()));

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

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBatchnormPlanBuilder,
       IsApplicableReturnsFalseWhenActivationOutputDoesNotMatchBnBackwardDy)
{
    // Fusion graph where activation output doesn't connect to BN backward dy
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    createBatchnormFusionTensorAttributes(builder, tensorAttributes, 12, {10, 11});

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // BN inference
    auto bnInfAttr = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormInferenceAttributes(
        builder, 1, 4, 5, 2, 3, 10);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_inf",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        bnInfAttr.Union()));

    // Activation with output uid=11
    auto actAttr = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        6,
        flatbuffers::Optional<int64_t>(10),
        flatbuffers::nullopt,
        11);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "act",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        actAttr.Union()));

    // BN backward with dy=999
    auto bnBwdAttr = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormBackwardAttributes(
        builder,
        999, // Wrong dy
        1,
        flatbuffers::Optional<int64_t>(4),
        flatbuffers::Optional<int64_t>(5),
        2,
        flatbuffers::Offset<flatbuffers::Vector<int64_t>>(),
        7,
        8,
        9);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_bwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        bnBwdAttr.Union()));

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

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsFalseWhenBnBackwardXDiffersFromInference)
{
    // Fusion graph where BN backward uses different X than BN inference
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    createBatchnormFusionTensorAttributes(builder, tensorAttributes, 13, {10, 11});

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // BN inference with x=1
    auto bnInfAttr = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormInferenceAttributes(
        builder, 1, 4, 5, 2, 3, 10);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_inf",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        bnInfAttr.Union()));

    // Activation
    auto actAttr = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        6,
        flatbuffers::Optional<int64_t>(10),
        flatbuffers::nullopt,
        11);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "act",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        actAttr.Union()));

    // BN backward with x=12 (wrong - should be 1 like BN inference)
    auto bnBwdAttr = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormBackwardAttributes(
        builder,
        11,
        12, // Wrong X
        flatbuffers::Optional<int64_t>(4),
        flatbuffers::Optional<int64_t>(5),
        2,
        flatbuffers::Offset<flatbuffers::Vector<int64_t>>(),
        7,
        8,
        9);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_bwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        bnBwdAttr.Union()));

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

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBatchnormPlanBuilder,
       IsApplicableReturnsFalseWhenBnBackwardScaleDiffersFromInference)
{
    // Fusion graph where BN backward uses different scale than BN inference
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    createBatchnormFusionTensorAttributes(builder, tensorAttributes, 13, {10, 11});

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // BN inference with scale=2
    auto bnInfAttr = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormInferenceAttributes(
        builder, 1, 4, 5, 2, 3, 10);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_inf",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        bnInfAttr.Union()));

    // Activation
    auto actAttr = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        6,
        flatbuffers::Optional<int64_t>(10),
        flatbuffers::nullopt,
        11);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "act",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        actAttr.Union()));

    // BN backward with scale=12 (wrong - should be 2 like BN inference)
    auto bnBwdAttr = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormBackwardAttributes(
        builder,
        11,
        1,
        flatbuffers::Optional<int64_t>(4),
        flatbuffers::Optional<int64_t>(5),
        12, // Wrong scale
        flatbuffers::Offset<flatbuffers::Vector<int64_t>>(),
        7,
        8,
        9);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_bwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        bnBwdAttr.Union()));

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

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsFalseWhenBnInferenceOutputIsNotVirtual)
{
    // Fusion graph where BN inference output tensor is non-virtual (should be virtual)
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    createBatchnormFusionTensorAttributes(builder, tensorAttributes, 12, {11});

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // BN inference with output uid=10 (which is non-virtual - wrong!)
    auto bnInfAttr = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormInferenceAttributes(
        builder, 1, 4, 5, 2, 3, 10);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_inf",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        bnInfAttr.Union()));

    // Activation
    auto actAttr = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        6,
        flatbuffers::Optional<int64_t>(10),
        flatbuffers::nullopt,
        11);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "act",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        actAttr.Union()));

    // BN backward
    auto bnBwdAttr = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormBackwardAttributes(
        builder,
        11,
        1,
        flatbuffers::Optional<int64_t>(4),
        flatbuffers::Optional<int64_t>(5),
        2,
        flatbuffers::Offset<flatbuffers::Vector<int64_t>>(),
        7,
        8,
        9);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_bwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        bnBwdAttr.Union()));

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

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsFalseWhenActivationOutputIsNotVirtual)
{
    // Fusion graph where activation output tensor is non-virtual (should be virtual)
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    createBatchnormFusionTensorAttributes(builder, tensorAttributes, 12, {10});

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // BN inference
    auto bnInfAttr = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormInferenceAttributes(
        builder, 1, 4, 5, 2, 3, 10);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_inf",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        bnInfAttr.Union()));

    // Activation with output=11 (which is non-virtual - wrong!)
    auto actAttr = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        6,
        flatbuffers::Optional<int64_t>(10),
        flatbuffers::nullopt,
        11);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "act",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        actAttr.Union()));

    // BN backward
    auto bnBwdAttr = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormBackwardAttributes(
        builder,
        11,
        1,
        flatbuffers::Optional<int64_t>(4),
        flatbuffers::Optional<int64_t>(5),
        2,
        flatbuffers::Offset<flatbuffers::Vector<int64_t>>(),
        7,
        8,
        9);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_bwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        bnBwdAttr.Union()));

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

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBatchnormPlanBuilder,
       IsApplicableReturnsFalseForTwoNodeFusionWhenInfOutputNotMatchingActInput)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    createBatchnormFusionTensorAttributes(builder, tensorAttributes, 7, {6});

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // BN inference
    auto bnInfAttr = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormInferenceAttributes(
        builder, 1, 4, 5, 2, 3, 6);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_inf",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        bnInfAttr.Union()));

    auto actAttr = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        5, //wrong input
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

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBatchnormPlanBuilder,
       IsApplicableReturnsFalseForTwoNodeFusionWhenInfOutputIsNonVirtual)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    createBatchnormFusionTensorAttributes(builder, tensorAttributes, 7, {}); //node 6 not virtual

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // BN inference
    auto bnInfAttr = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormInferenceAttributes(
        builder, 1, 4, 5, 2, 3, 6);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_inf",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        bnInfAttr.Union()));

    auto actAttr = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        6,
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

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

// ============================================================================
// Variance Extension Fusion Tests
// ============================================================================

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsTrueForFusedTwoNodeGraphWithVariance)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    createBatchnormFusionTensorAttributes(builder, tensorAttributes, 7, {6});

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // BN inference with variance
    auto bnInfAttr
        = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormInferenceAttributesVarianceExt(
            builder, 1, 2, 3, 4, 5, 6);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_inf_var",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::
            BatchnormInferenceAttributesVarianceExt,
        bnInfAttr.Union()));

    // Activation
    auto actAttr = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        6,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        7);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "act",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
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

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBatchnormPlanBuilder,
       IsApplicableReturnsFalseForFusedTwoNodeGraphWithVarianceAndBrokenConnectivity)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    createBatchnormFusionTensorAttributes(builder, tensorAttributes, 7, {6});

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // BN inference with variance
    auto bnInfAttr
        = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormInferenceAttributesVarianceExt(
            builder, 1, 2, 3, 4, 5, 6);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_inf_var",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::
            BatchnormInferenceAttributesVarianceExt,
        bnInfAttr.Union()));

    // Activation with wrong input
    auto actAttr = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        5, // Wrong input!
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        7);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "act",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
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

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBatchnormPlanBuilder,
       IsApplicableReturnsFalseForFusedTwoNodeGraphWithVarianceAndNonVirtualIntermediate)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    createBatchnormFusionTensorAttributes(builder, tensorAttributes, 7, {}); // No virtual tensors

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // BN inference with variance
    auto bnInfAttr
        = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormInferenceAttributesVarianceExt(
            builder, 1, 2, 3, 4, 5, 6);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_inf_var",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::
            BatchnormInferenceAttributesVarianceExt,
        bnInfAttr.Union()));

    // Activation
    auto actAttr = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        6,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        7);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "act",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
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

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBatchnormPlanBuilder,
       IsApplicableReturnsFalseForFusedTwoNodeGraphWithVarianceAndUnsupportedActivation)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    createBatchnormFusionTensorAttributes(builder, tensorAttributes, 7, {6});

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // BN inference with variance
    auto bnInfAttr
        = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormInferenceAttributesVarianceExt(
            builder, 1, 2, 3, 4, 5, 6);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn_inf_var",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::
            BatchnormInferenceAttributesVarianceExt,
        bnInfAttr.Union()));

    // Activation with unsupported MUL
    auto actAttr = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::MUL, // Unsupported!
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        6,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        7);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "act",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
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

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

// ============================================================================
// 3D Tests
// ============================================================================

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsTrueFor3dNclInference)
{
    const std::vector<int64_t> dims3D = {1, 3, 14};
    const std::vector<int64_t> strides3D = {42, 14, 1};

    auto builder
        = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph(strides3D, dims3D);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsTrueFor3dNlcBackward)
{
    const std::vector<int64_t> dims3D = {2, 3, 14};
    const std::vector<int64_t> strides3D = {42, 1, 3};

    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph(strides3D, dims3D);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsTrueFor3dNclVarianceInference)
{
    const std::vector<int64_t> dims3D = {1, 3, 14};
    const std::vector<int64_t> strides3D = {42, 14, 1};

    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormWithVarianceInferenceGraph(
        strides3D, dims3D);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsTrueFor3dNlcFusedTwoNodeGraph)
{
    const std::vector<int64_t> dims3D = {1, 3, 14};
    const std::vector<int64_t> strides3D = {42, 1, 3};

    auto builder
        = hipdnn_test_sdk::utilities::createValidBatchnormFwdInferActGraph(strides3D, dims3D);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsTrueFor3dNclFusedThreeNodeGraph)
{
    const std::vector<int64_t> dims3D = {2, 3, 14};
    const std::vector<int64_t> strides3D = {42, 14, 1};

    auto builder
        = hipdnn_test_sdk::utilities::createValidBatchnormInferActBwdGraph(strides3D, dims3D);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanSetsPlanFor3dNclInference)
{
    const std::vector<int64_t> dims3D = {1, 3, 14};
    const std::vector<int64_t> strides3D = {42, 14, 1};

    auto builder
        = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph(strides3D, dims3D);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    HipdnnMiopenContext ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, _mockEngineConfig, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanSetsPlanFor3dNlcBackward)
{
    const std::vector<int64_t> dims3D = {2, 3, 14};
    const std::vector<int64_t> strides3D = {42, 1, 3};

    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph(strides3D, dims3D);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    HipdnnMiopenContext ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, _mockEngineConfig, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanSetsPlanFor3dNclVarianceInference)
{
    const std::vector<int64_t> dims3D = {1, 3, 14};
    const std::vector<int64_t> strides3D = {42, 14, 1};

    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormWithVarianceInferenceGraph(
        strides3D, dims3D);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    HipdnnMiopenContext ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, _mockEngineConfig, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanSetsPlanFor3dNlcFusedTwoNodeGraph)
{
    const std::vector<int64_t> dims3D = {1, 3, 14};
    const std::vector<int64_t> strides3D = {42, 1, 3};

    auto builder
        = hipdnn_test_sdk::utilities::createValidBatchnormFwdInferActGraph(strides3D, dims3D);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    HipdnnMiopenContext ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, _mockEngineConfig, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanSetsPlanFor3dNclFusedThreeNodeGraph)
{
    const std::vector<int64_t> dims3D = {2, 3, 14};
    const std::vector<int64_t> strides3D = {42, 14, 1};

    auto builder
        = hipdnn_test_sdk::utilities::createValidBatchnormInferActBwdGraph(strides3D, dims3D);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    HipdnnMiopenContext ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, _mockEngineConfig, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}
