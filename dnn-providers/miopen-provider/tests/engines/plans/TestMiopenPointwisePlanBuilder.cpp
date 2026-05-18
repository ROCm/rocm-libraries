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
