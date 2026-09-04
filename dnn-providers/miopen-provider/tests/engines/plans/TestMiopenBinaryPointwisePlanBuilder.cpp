// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipdnnMiopenHandle.hpp"
#include "engines/plans/MiopenBinaryPointwisePlanBuilder.hpp"

using namespace miopen_plugin;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

namespace
{

flatbuffers::FlatBufferBuilder
    createPointwiseGraphWithIoDtype(hipdnn_flatbuffers_sdk::data_objects::DataType ioDtype,
                                    hipdnn_flatbuffers_sdk::data_objects::PointwiseMode mode)
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    const std::vector<int64_t> strides = {48, 16, 4, 1};

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "input0", ioDtype, &strides, &dims, false));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "input1", ioDtype, &strides, &dims, false));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "output", ioDtype, &strides, &dims, false));

    auto pwAttr
        = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(builder,
                                                                          mode,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          1,
                                                                          2,
                                                                          flatbuffers::nullopt,
                                                                          3);

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
    createPointwiseGraphWithDims(const std::vector<int64_t>& inputDims,
                                 const std::vector<int64_t>& inputStrides,
                                 const std::vector<int64_t>& outputDims,
                                 const std::vector<int64_t>& outputStrides,
                                 hipdnn_flatbuffers_sdk::data_objects::PointwiseMode mode)
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        1,
        "input0",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &inputStrides,
        &inputDims,
        false));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        2,
        "input1",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &inputStrides,
        &inputDims,
        false));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "output",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &outputStrides,
        &outputDims,
        false));

    auto pwAttr
        = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(builder,
                                                                          mode,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          1,
                                                                          2,
                                                                          flatbuffers::nullopt,
                                                                          3);

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

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    const std::vector<int64_t> strides = {48, 16, 4, 1};

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        1,
        "input0",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims,
        virtualInput));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        2,
        "input1",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims,
        false));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
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
                                                                          2,
                                                                          flatbuffers::nullopt,
                                                                          3);

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

class TestMiopenBinaryPointwisePlanBuilder : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        _dummyHandle = std::make_unique<HipdnnMiopenHandle>();
    }

    MiopenBinaryPointwisePlanBuilder _planBuilder;
    std::unique_ptr<HipdnnMiopenHandle> _dummyHandle;
    MockEngineConfig _mockEngineConfig;
};

// =========================================================================
// Structural Constraints Verification
// =========================================================================

TEST_F(TestMiopenBinaryPointwisePlanBuilder, IsApplicableReturnsFalseForMultiNodeGraph)
{
    const MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(2));

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, mockGraph));
}

TEST_F(TestMiopenBinaryPointwisePlanBuilder, IsApplicableReturnsFalseForUnsupportedAttributes)
{
    const MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(1));
    EXPECT_CALL(mockGraph, hasOnlySupportedAttributes(::testing::_))
        .WillOnce(::testing::Return(false));

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, mockGraph));
}

TEST_F(TestMiopenBinaryPointwisePlanBuilder, IsApplicableReturnsFalseForNonFloatComputeType)
{
    auto builder = createPointwiseGraph(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD,
                                        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBinaryPointwisePlanBuilder, IsApplicableReturnsFalseForVirtualInputTensor)
{
    auto builder = createPointwiseGraph(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD,
                                        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                        true,
                                        false);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBinaryPointwisePlanBuilder, IsApplicableReturnsFalseForVirtualOutputTensor)
{
    auto builder = createPointwiseGraph(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD,
                                        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                        false,
                                        true);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBinaryPointwisePlanBuilder, GetMaxWorkspaceSizeReturnsZero)
{
    auto builder = createPointwiseGraph(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const HipdnnMiopenSettings settings;
    EXPECT_EQ(_planBuilder.getMaxWorkspaceSize(*_dummyHandle, graph, settings), 0u);
}

TEST_F(TestMiopenBinaryPointwisePlanBuilder, GetCustomKnobsReturnsEmpty)
{
    auto builder = createPointwiseGraph(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto knobs = _planBuilder.getCustomKnobs(*_dummyHandle, graph);
    EXPECT_TRUE(knobs.empty());
}

TEST_F(TestMiopenBinaryPointwisePlanBuilder, IsApplicableReturnsFalseForBfloat16IoDtype)
{
    auto builder
        = createPointwiseGraphWithIoDtype(hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                                          hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBinaryPointwisePlanBuilder, IsApplicableReturnsFalseForRank5Tensor)
{
    const std::vector<int64_t> dims = {1, 2, 3, 4, 5};
    const std::vector<int64_t> strides = {120, 60, 20, 5, 1};
    auto builder = createPointwiseGraphWithDims(
        dims, strides, dims, strides, hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBinaryPointwisePlanBuilder, IsApplicableReturnsFalseForMismatchedElementCount)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> inputStrides = {48, 16, 4, 1};
    const std::vector<int64_t> outputDims = {1, 3, 4, 8};
    const std::vector<int64_t> outputStrides = {96, 32, 8, 1};
    auto builder
        = createPointwiseGraphWithDims(inputDims,
                                       inputStrides,
                                       outputDims,
                                       outputStrides,
                                       hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBinaryPointwisePlanBuilder, IsApplicableReturnsFalseForMismatchedStrideRank)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    const std::vector<int64_t> invalidStrides = {16, 4, 1};

    auto builder
        = createPointwiseGraphWithDims(dims,
                                       invalidStrides,
                                       dims,
                                       invalidStrides,
                                       hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

// =========================================================================
// ADD Operation Tests
// =========================================================================

TEST_F(TestMiopenBinaryPointwisePlanBuilder, IsApplicableReturnsTrueForValidAddGraph)
{
    auto builder = createPointwiseGraph(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBinaryPointwisePlanBuilder, BuildPlanDoesNotThrowForValidAddGraph)
{
    auto builder = createPointwiseGraph(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    HipdnnMiopenContext ctx;
    EXPECT_NO_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, _mockEngineConfig, ctx));
}

TEST_F(TestMiopenBinaryPointwisePlanBuilder, IsApplicableReturnsTrueForAddHalfIoDtype)
{
    auto builder
        = createPointwiseGraphWithIoDtype(hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                                          hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBinaryPointwisePlanBuilder, IsApplicableReturnsTrueForAddRank1Tensor)
{
    const std::vector<int64_t> dims = {16};
    const std::vector<int64_t> strides = {1};
    auto builder = createPointwiseGraphWithDims(
        dims, strides, dims, strides, hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

// =========================================================================
// SUB Operation Tests
// =========================================================================

TEST_F(TestMiopenBinaryPointwisePlanBuilder, IsApplicableReturnsTrueForValidSubGraph)
{
    auto builder = createPointwiseGraph(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SUB);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBinaryPointwisePlanBuilder, BuildPlanDoesNotThrowForValidSubGraph)
{
    auto builder = createPointwiseGraph(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SUB);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    HipdnnMiopenContext ctx;
    EXPECT_NO_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, _mockEngineConfig, ctx));
}

TEST_F(TestMiopenBinaryPointwisePlanBuilder, IsApplicableReturnsTrueForSubHalfIoDtype)
{
    auto builder
        = createPointwiseGraphWithIoDtype(hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                                          hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SUB);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBinaryPointwisePlanBuilder, IsApplicableReturnsTrueForSubRank1Tensor)
{
    const std::vector<int64_t> dims = {16};
    const std::vector<int64_t> strides = {1};
    auto builder = createPointwiseGraphWithDims(
        dims, strides, dims, strides, hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SUB);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

// =========================================================================
// MUL Operation Tests
// =========================================================================

TEST_F(TestMiopenBinaryPointwisePlanBuilder, IsApplicableReturnsTrueForValidMulGraph)
{
    auto builder = createPointwiseGraph(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::MUL);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBinaryPointwisePlanBuilder, BuildPlanDoesNotThrowForValidMulGraph)
{
    auto builder = createPointwiseGraph(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::MUL);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    HipdnnMiopenContext ctx;
    EXPECT_NO_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, _mockEngineConfig, ctx));
}

TEST_F(TestMiopenBinaryPointwisePlanBuilder, IsApplicableReturnsTrueForMulHalfIoDtype)
{
    auto builder
        = createPointwiseGraphWithIoDtype(hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                                          hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::MUL);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenBinaryPointwisePlanBuilder, IsApplicableReturnsTrueForMulRank1Tensor)
{
    const std::vector<int64_t> dims = {16};
    const std::vector<int64_t> strides = {1};
    auto builder = createPointwiseGraphWithDims(
        dims, strides, dims, strides, hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::MUL);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}
