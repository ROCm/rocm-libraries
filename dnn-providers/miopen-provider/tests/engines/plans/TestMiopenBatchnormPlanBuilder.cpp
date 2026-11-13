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
#include "engines/plans/MiopenBatchnormFwdTrainingPlanBuilder.hpp"
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

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsFalseForOverrideShapeEnabledGraph)
{
    // Execute-time override shapes can diverge from the compile-time dims this
    // builder matched exactly; the graph opts in, so the builder must decline.
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormWithVarianceInferenceGraph(
        {150528, 50176, 224, 1},
        {1, 3, 224, 224},
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        /*overrideShapeEnabled=*/true);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
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

namespace
{

namespace fb = hipdnn_flatbuffers_sdk::data_objects;

// Running-statistics roles of a batchnorm training node, one bit each.
constexpr unsigned PREV_MEAN = 1U << 0;
constexpr unsigned PREV_VAR = 1U << 1;
constexpr unsigned MOMENTUM = 1U << 2;
constexpr unsigned NEXT_MEAN = 1U << 3;
constexpr unsigned NEXT_VAR = 1U << 4;
constexpr unsigned ALL_RUNNING_STATS = PREV_MEAN | PREV_VAR | MOMENTUM | NEXT_MEAN | NEXT_VAR;

// Single-node batchnorm training graph whose running-statistics roles and saved
// mean / inv_variance outputs are selected independently.
flatbuffers::FlatBufferBuilder createBatchnormTrainingGraphWithStats(unsigned runningStatsMask,
                                                                     bool saveMean,
                                                                     bool saveInvVariance)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors;

    const std::vector<int64_t> dims = {1, 3, 14, 14};
    const std::vector<int64_t> strides = {588, 196, 14, 1};
    const std::vector<int64_t> derivedDims = {1, 3, 1, 1};
    const std::vector<int64_t> derivedStrides = {3, 1, 1, 1};
    const std::vector<int64_t> scalarDims = {1};

    const auto addTensor = [&](int64_t uid,
                               const char* name,
                               const std::vector<int64_t>& tensorStrides,
                               const std::vector<int64_t>& tensorDims) {
        tensors.push_back(fb::CreateTensorAttributesDirect(
            builder, uid, name, fb::DataType::FLOAT, &tensorStrides, &tensorDims));
    };
    const auto addScalar = [&](int64_t uid, const char* name, float value) {
        const fb::Float32Value scalar(value);
        tensors.push_back(fb::CreateTensorAttributesDirect(builder,
                                                           uid,
                                                           name,
                                                           fb::DataType::FLOAT,
                                                           &scalarDims,
                                                           &scalarDims,
                                                           false,
                                                           fb::TensorValue::Float32Value,
                                                           builder.CreateStruct(scalar).Union()));
    };
    const auto optionalUid = [](bool present, int64_t uid) {
        return present ? flatbuffers::Optional<int64_t>(uid) : flatbuffers::nullopt;
    };

    addTensor(1, "x", strides, dims);
    addTensor(2, "y", strides, dims);
    addTensor(3, "scale", derivedStrides, derivedDims);
    addTensor(4, "bias", derivedStrides, derivedDims);
    addScalar(5, "epsilon", 1e-5f);

    if(saveMean)
    {
        addTensor(6, "mean", derivedStrides, derivedDims);
    }
    if(saveInvVariance)
    {
        addTensor(7, "inv_variance", derivedStrides, derivedDims);
    }
    if((runningStatsMask & PREV_MEAN) != 0U)
    {
        addTensor(8, "prev_running_mean", derivedStrides, derivedDims);
    }
    if((runningStatsMask & PREV_VAR) != 0U)
    {
        addTensor(9, "prev_running_variance", derivedStrides, derivedDims);
    }
    if((runningStatsMask & MOMENTUM) != 0U)
    {
        addScalar(10, "momentum", 0.1f);
    }
    if((runningStatsMask & NEXT_MEAN) != 0U)
    {
        addTensor(11, "next_running_mean", derivedStrides, derivedDims);
    }
    if((runningStatsMask & NEXT_VAR) != 0U)
    {
        addTensor(12, "next_running_variance", derivedStrides, derivedDims);
    }

    const auto attributes
        = fb::CreateBatchnormAttributes(builder,
                                        1,
                                        3,
                                        4,
                                        5,
                                        0,
                                        optionalUid((runningStatsMask & PREV_MEAN) != 0U, 8),
                                        optionalUid((runningStatsMask & PREV_VAR) != 0U, 9),
                                        optionalUid((runningStatsMask & MOMENTUM) != 0U, 10),
                                        2,
                                        optionalUid(saveMean, 6),
                                        optionalUid(saveInvVariance, 7),
                                        optionalUid((runningStatsMask & NEXT_MEAN) != 0U, 11),
                                        optionalUid((runningStatsMask & NEXT_VAR) != 0U, 12));

    std::vector<flatbuffers::Offset<fb::Node>> nodes;
    nodes.push_back(fb::CreateNodeDirect(builder,
                                         "batchnorm_training",
                                         fb::DataType::FLOAT,
                                         fb::NodeAttributes::BatchnormAttributes,
                                         attributes.Union()));

    builder.Finish(fb::CreateGraphDirect(builder,
                                         "test",
                                         fb::DataType::FLOAT,
                                         fb::DataType::HALF,
                                         fb::DataType::BFLOAT16,
                                         &tensors,
                                         &nodes));
    return builder;
}

struct BnTrainingStatsParam
{
    unsigned runningStatsMask;
    bool saveMean;
    bool saveInvVariance;
};

std::string bnTrainingStatsParamName(const ::testing::TestParamInfo<BnTrainingStatsParam>& info)
{
    const auto& p = info.param;
    const auto role = [&](unsigned bit, const char* name) {
        return std::string((p.runningStatsMask & bit) != 0U ? "_" : "_No") + name;
    };
    return role(PREV_MEAN, "PrevMean") + role(PREV_VAR, "PrevVar") + role(MOMENTUM, "Momentum")
           + role(NEXT_MEAN, "NextMean") + role(NEXT_VAR, "NextVar")
           + (p.saveMean ? "_SaveMean" : "_NoSaveMean")
           + (p.saveInvVariance ? "_SaveInvVar" : "_NoSaveInvVar");
}

std::vector<BnTrainingStatsParam> allBnTrainingStatsParams()
{
    std::vector<BnTrainingStatsParam> params;
    for(unsigned mask = 0; mask <= ALL_RUNNING_STATS; ++mask)
    {
        for(const bool saveMean : {false, true})
        {
            for(const bool saveInvVariance : {false, true})
            {
                params.push_back({mask, saveMean, saveInvVariance});
            }
        }
    }
    return params;
}

enum class BnSingleNodeMode
{
    INFERENCE,
    INFERENCE_WITH_VARIANCE,
    BACKWARD,
    TRAINING,
};

struct BnVirtualRoleParam
{
    BnSingleNodeMode mode;
    int64_t uid;
    const char* role;
};

const char* modeName(BnSingleNodeMode mode)
{
    switch(mode)
    {
    case BnSingleNodeMode::INFERENCE:
        return "Inference";
    case BnSingleNodeMode::INFERENCE_WITH_VARIANCE:
        return "InferenceWithVariance";
    case BnSingleNodeMode::BACKWARD:
        return "Backward";
    case BnSingleNodeMode::TRAINING:
        return "Training";
    default:
        return "Unknown";
    }
}

flatbuffers::FlatBufferBuilder createSingleNodeGraph(BnSingleNodeMode mode)
{
    namespace utils = hipdnn_test_sdk::utilities;
    switch(mode)
    {
    case BnSingleNodeMode::INFERENCE:
        return utils::createValidBatchnormInferenceGraph();
    case BnSingleNodeMode::INFERENCE_WITH_VARIANCE:
        return utils::createValidBatchnormWithVarianceInferenceGraph();
    case BnSingleNodeMode::BACKWARD:
        return utils::createValidBatchnormBwdGraph();
    case BnSingleNodeMode::TRAINING:
        return utils::createValidBatchnormFwdTrainingGraph();
    default:
        throw std::logic_error("unknown batchnorm mode");
    }
}

// Returns the graph with one tensor flagged virtual; a negative uid leaves the graph unchanged.
flatbuffers::FlatBufferBuilder createSingleNodeGraphWithVirtualTensor(BnSingleNodeMode mode,
                                                                      int64_t virtualUid)
{
    return hipdnn_test_sdk::utilities::mutateGraph(createSingleNodeGraph(mode),
                                                   [virtualUid](fb::GraphT& graph) {
                                                       for(auto& tensor : graph.tensors)
                                                       {
                                                           if(tensor->uid == virtualUid)
                                                           {
                                                               tensor->virtual_ = true;
                                                           }
                                                       }
                                                   });
}

// Every tensor role the single-node plan reads or writes. Uids follow the test-sdk graph helpers.
const std::vector<BnVirtualRoleParam>& singleNodeVirtualRoles()
{
    static const std::vector<BnVirtualRoleParam> s_roles = {
        {BnSingleNodeMode::INFERENCE, 1, "X"},
        {BnSingleNodeMode::INFERENCE, 2, "Y"},
        {BnSingleNodeMode::INFERENCE, 3, "Scale"},
        {BnSingleNodeMode::INFERENCE, 4, "Bias"},
        {BnSingleNodeMode::INFERENCE, 5, "Mean"},
        {BnSingleNodeMode::INFERENCE, 6, "InvVariance"},
        {BnSingleNodeMode::INFERENCE_WITH_VARIANCE, 1, "X"},
        {BnSingleNodeMode::INFERENCE_WITH_VARIANCE, 2, "Y"},
        {BnSingleNodeMode::INFERENCE_WITH_VARIANCE, 3, "Scale"},
        {BnSingleNodeMode::INFERENCE_WITH_VARIANCE, 4, "Bias"},
        {BnSingleNodeMode::INFERENCE_WITH_VARIANCE, 5, "Mean"},
        {BnSingleNodeMode::INFERENCE_WITH_VARIANCE, 6, "Variance"},
        {BnSingleNodeMode::BACKWARD, 1, "X"},
        {BnSingleNodeMode::BACKWARD, 2, "Dy"},
        {BnSingleNodeMode::BACKWARD, 3, "Dx"},
        {BnSingleNodeMode::BACKWARD, 4, "Scale"},
        {BnSingleNodeMode::BACKWARD, 5, "Dscale"},
        {BnSingleNodeMode::BACKWARD, 6, "Dbias"},
        {BnSingleNodeMode::BACKWARD, 7, "Mean"},
        {BnSingleNodeMode::BACKWARD, 8, "InvVariance"},
        {BnSingleNodeMode::TRAINING, 1, "X"},
        {BnSingleNodeMode::TRAINING, 2, "Y"},
        {BnSingleNodeMode::TRAINING, 3, "Scale"},
        {BnSingleNodeMode::TRAINING, 4, "Bias"},
        {BnSingleNodeMode::TRAINING, 6, "SavedMean"},
        {BnSingleNodeMode::TRAINING, 7, "SavedInvVariance"},
    };
    return s_roles;
}

} // namespace

// The single-node training plan only honours a complete running-statistics set or
// none, and saved mean / inv_variance only as a pair; anything else is dropped
// silently at execute time, so the builder must decline it.
class TestMiopenBatchnormPlanBuilderTrainingStats
    : public ::testing::TestWithParam<BnTrainingStatsParam>
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        _dummyHandle = std::make_unique<HipdnnMiopenHandle>();
    }

    std::unique_ptr<HipdnnMiopenHandle> _dummyHandle;
};

TEST_P(TestMiopenBatchnormPlanBuilderTrainingStats, ServesOnlyGraphsWhoseStatsThePlanHonours)
{
    const auto& p = GetParam();
    auto builder
        = createBatchnormTrainingGraphWithStats(p.runningStatsMask, p.saveMean, p.saveInvVariance);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const bool planHonoursStats = p.runningStatsMask == 0U && p.saveMean == p.saveInvVariance;

    EXPECT_EQ(MiopenBatchnormPlanBuilder{}.isApplicable(*_dummyHandle, graph), planHonoursStats);
}

// The training builder serves the complete running-statistics set, but, like the plan
// it builds, only a complete one.
TEST_P(TestMiopenBatchnormPlanBuilderTrainingStats,
       TrainingBuilderServesOnlyCompleteOrNoRunningStats)
{
    const auto& p = GetParam();
    auto builder
        = createBatchnormTrainingGraphWithStats(p.runningStatsMask, p.saveMean, p.saveInvVariance);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const bool planHonoursStats
        = (p.runningStatsMask == 0U || p.runningStatsMask == ALL_RUNNING_STATS)
          && p.saveMean == p.saveInvVariance;

    EXPECT_EQ(MiopenBatchnormFwdTrainingPlanBuilder{}.isApplicable(*_dummyHandle, graph),
              planHonoursStats);
}

INSTANTIATE_TEST_SUITE_P(AllRunningStatsAndSavedStatsSubsets,
                         TestMiopenBatchnormPlanBuilderTrainingStats,
                         ::testing::ValuesIn(allBnTrainingStatsParams()),
                         bnTrainingStatsParamName);

class TestMiopenBatchnormPlanBuilderVirtualRole
    : public ::testing::TestWithParam<BnVirtualRoleParam>
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        _dummyHandle = std::make_unique<HipdnnMiopenHandle>();
    }

    MiopenBatchnormPlanBuilder _planBuilder;
    std::unique_ptr<HipdnnMiopenHandle> _dummyHandle;
};

// The single-node plans execute against caller-supplied device buffers, which a virtual
// tensor never has.
TEST_P(TestMiopenBatchnormPlanBuilderVirtualRole, DeclinesVirtualTensor)
{
    const auto& p = GetParam();

    auto servedBuilder = createSingleNodeGraphWithVirtualTensor(p.mode, -1);
    const GraphWrapper served(servedBuilder.GetBufferPointer(), servedBuilder.GetSize());
    ASSERT_TRUE(_planBuilder.isApplicable(*_dummyHandle, served));

    auto virtualBuilder = createSingleNodeGraphWithVirtualTensor(p.mode, p.uid);
    const GraphWrapper graph(virtualBuilder.GetBufferPointer(), virtualBuilder.GetSize());
    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

INSTANTIATE_TEST_SUITE_P(EveryRolePerMode,
                         TestMiopenBatchnormPlanBuilderVirtualRole,
                         ::testing::ValuesIn(singleNodeVirtualRoles()),
                         [](const ::testing::TestParamInfo<BnVirtualRoleParam>& info) {
                             return std::string(modeName(info.param.mode)) + "_" + info.param.role;
                         });
