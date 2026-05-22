// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <memory>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipdnnMiopenHandle.hpp"
#include "engines/plans/MiopenPointwisePlanBuilder.hpp"

using namespace miopen_plugin;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

namespace
{

flatbuffers::FlatBufferBuilder
    createPointwiseGraphWithIoDtype(hipdnn_flatbuffers_sdk::data_objects::DataType ioDtype)
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    std::vector<int64_t> dims = {1, 3, 4, 4};
    std::vector<int64_t> strides = {48, 16, 4, 1};

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "input", ioDtype, &strides, &dims, false));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "output", ioDtype, &strides, &dims, false));

    auto pwAttr = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        1,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "pointwise",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        pwAttr.Union()));

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);
    return builder;
}

flatbuffers::FlatBufferBuilder createPointwiseGraphWithDims(std::vector<int64_t> inputDims,
                                                            std::vector<int64_t> inputStrides,
                                                            std::vector<int64_t> outputDims,
                                                            std::vector<int64_t> outputStrides)
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        1,
        "input",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &inputStrides,
        &inputDims,
        false));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        2,
        "output",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &outputStrides,
        &outputDims,
        false));

    auto pwAttr = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        1,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "pointwise",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        pwAttr.Union()));

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);
    return builder;
}

flatbuffers::FlatBufferBuilder
    createReluGraphWithParams(flatbuffers::Optional<float> lowerClip,
                              flatbuffers::Optional<float> upperClip,
                              flatbuffers::Optional<float> lowerClipSlope)
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    std::vector<int64_t> dims = {1, 3, 4, 4};
    std::vector<int64_t> strides = {48, 16, 4, 1};

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        1,
        "input",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims,
        false));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        2,
        "output",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims,
        false));

    auto pwAttr = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        lowerClip,
        upperClip,
        lowerClipSlope,
        flatbuffers::nullopt,
        1,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "pointwise",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        pwAttr.Union()));

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);
    return builder;
}

flatbuffers::FlatBufferBuilder
    createPointwiseGraph(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode mode,
                         hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType
                         = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                         bool virtualInput = false,
                         bool virtualOutput = false)
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    std::vector<int64_t> dims = {1, 3, 4, 4};
    std::vector<int64_t> strides = {48, 16, 4, 1};

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        1,
        "input",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims,
        virtualInput));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        2,
        "output",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims,
        virtualOutput));

    auto pwAttr
        = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(builder,
                                                                          mode,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          1,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          2);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "pointwise",
        computeDataType,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        pwAttr.Union()));

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);

    return builder;
}

} // namespace

class TestMiopenPointwisePlanBuilder : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        _dummyHandle = std::make_unique<HipdnnMiopenHandle>();
    }

    MiopenPointwisePlanBuilder _planBuilder;
    std::unique_ptr<HipdnnMiopenHandle> _dummyHandle;
    MockEngineConfig _mockEngineConfig;
};

TEST_F(TestMiopenPointwisePlanBuilder, IsApplicableReturnsFalseForMultiNodeGraph)
{
    MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(2));

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, mockGraph));
}

TEST_F(TestMiopenPointwisePlanBuilder, IsApplicableReturnsFalseForUnsupportedAttributes)
{
    MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(1));
    EXPECT_CALL(mockGraph, hasOnlySupportedAttributes(::testing::_))
        .WillOnce(::testing::Return(false));

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, mockGraph));
}

TEST_F(TestMiopenPointwisePlanBuilder, IsApplicableReturnsTrueForValidReluFwdGraph)
{
    auto builder
        = createPointwiseGraph(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenPointwisePlanBuilder, IsApplicableReturnsFalseForUnsupportedMode)
{
    auto builder = createPointwiseGraph(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenPointwisePlanBuilder, IsApplicableReturnsFalseForNonFloatComputeType)
{
    auto builder
        = createPointwiseGraph(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
                               hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenPointwisePlanBuilder, IsApplicableReturnsFalseForVirtualInputTensor)
{
    auto builder
        = createPointwiseGraph(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
                               hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                               true,
                               false);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenPointwisePlanBuilder, IsApplicableReturnsFalseForVirtualOutputTensor)
{
    auto builder
        = createPointwiseGraph(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
                               hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                               false,
                               true);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenPointwisePlanBuilder, GetMaxWorkspaceSizeReturnsZero)
{
    auto builder
        = createPointwiseGraph(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    HipdnnMiopenSettings settings;
    EXPECT_EQ(_planBuilder.getMaxWorkspaceSize(*_dummyHandle, graph, settings), 0u);
}

TEST_F(TestMiopenPointwisePlanBuilder, GetCustomKnobsReturnsEmpty)
{
    auto builder
        = createPointwiseGraph(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto knobs = _planBuilder.getCustomKnobs(*_dummyHandle, graph);
    EXPECT_TRUE(knobs.empty());
}

TEST_F(TestMiopenPointwisePlanBuilder, BuildPlanDoesNotThrowForValidGraph)
{
    auto builder
        = createPointwiseGraph(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    HipdnnMiopenContext ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, _mockEngineConfig, ctx));
}

TEST_F(TestMiopenPointwisePlanBuilder, IsApplicableReturnsTrueForHalfIoDtype)
{
    auto builder
        = createPointwiseGraphWithIoDtype(hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenPointwisePlanBuilder, IsApplicableReturnsFalseForBfloat16IoDtype)
{
    auto builder
        = createPointwiseGraphWithIoDtype(hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenPointwisePlanBuilder, IsApplicableReturnsTrueForRank1Tensor)
{
    std::vector<int64_t> dims = {16};
    std::vector<int64_t> strides = {1};
    auto builder = createPointwiseGraphWithDims(dims, strides, dims, strides);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenPointwisePlanBuilder, IsApplicableReturnsFalseForRank5Tensor)
{
    std::vector<int64_t> dims = {1, 2, 3, 4, 5};
    std::vector<int64_t> strides = {120, 60, 20, 5, 1};
    auto builder = createPointwiseGraphWithDims(dims, strides, dims, strides);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenPointwisePlanBuilder, IsApplicableReturnsFalseForMismatchedElementCount)
{
    std::vector<int64_t> inputDims = {1, 3, 4, 4};
    std::vector<int64_t> inputStrides = {48, 16, 4, 1};
    std::vector<int64_t> outputDims = {1, 3, 4, 8};
    std::vector<int64_t> outputStrides = {96, 32, 8, 1};
    auto builder = createPointwiseGraphWithDims(inputDims, inputStrides, outputDims, outputStrides);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenPointwisePlanBuilder, IsApplicableReturnsTrueForReluWithUpperClip)
{
    auto builder = createReluGraphWithParams(flatbuffers::nullopt, 1.0f, flatbuffers::nullopt);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenPointwisePlanBuilder, IsApplicableReturnsTrueForReluWithLowerClipSlope)
{
    auto builder = createReluGraphWithParams(flatbuffers::nullopt, flatbuffers::nullopt, 0.1f);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenPointwisePlanBuilder, IsApplicableReturnsTrueForReluWithLowerAndUpperClip)
{
    auto builder = createReluGraphWithParams(-1.0f, 1.0f, flatbuffers::nullopt);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenPointwisePlanBuilder, IsApplicableReturnsFalseForReluWithNonZeroLowerClipOnly)
{
    auto builder = createReluGraphWithParams(0.5f, flatbuffers::nullopt, flatbuffers::nullopt);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}
