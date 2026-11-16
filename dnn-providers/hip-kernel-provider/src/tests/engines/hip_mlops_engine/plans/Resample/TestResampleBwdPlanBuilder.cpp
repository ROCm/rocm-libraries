// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdio>
#include <gtest/gtest.h>

#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"
#include "engines/hip_mlops_engine/plans/resample/ResampleBwdPlanBuilder.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockDevicePropertyProvider.hpp"
#include "mocks/MockKernelCompiler.hpp"
#include "mocks/MockRunnableKernel.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/resample_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>

using hipdnn_test_sdk::utilities::MockEngineConfig;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hip_kernel_provider;
using namespace hip_kernel_provider::resample;

namespace
{

// A 1x1 pooling window with stride 1, so dx has the dims of dy: [1, 1, height, width].
flatbuffers::FlatBufferBuilder createResampleBwdGraphWithGradientSize(int64_t height, int64_t width)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> dims = {1, 1, height, width};
    const std::vector<int64_t> strides = {height * width, height * width, width, 1};
    const std::vector<int64_t> padding = {0, 0};
    const std::vector<int64_t> unit = {1, 1};

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "dy", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &strides, &dims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "dx", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &strides, &dims));

    auto resampleAttr = hipdnn_flatbuffers_sdk::data_objects::CreateResampleBwdAttributesDirect(
        builder,
        1,
        2,
        ::flatbuffers::nullopt,
        &padding,
        &padding,
        &unit,
        &unit,
        hipdnn_flatbuffers_sdk::data_objects::ResampleMode::AVGPOOL_EXCLUDE_PADDING,
        hipdnn_flatbuffers_sdk::data_objects::PaddingMode::ZERO_PAD);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "resample_bwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ResampleBwdAttributes,
        resampleAttr.Union()));

    builder.Finish(hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &tensorAttributes,
        &nodes));
    return builder;
}

} // namespace

class TestResampleBwdPlanBuilder : public ::testing::Test
{
protected:
    MockKernelCompiler _mockKernelCompiler;
    MockDevicePropertyProvider _mockDevicePropertyProvider;
    ResampleBwdPlanBuilder _planBuilder{_mockKernelCompiler, _mockDevicePropertyProvider};
    Handle _dummyHandle;
    MockEngineConfig _mockEngineConfig;

    void setupMockCompileChain()
    {
        hipDeviceProp_t deviceProps = {};
        deviceProps.multiProcessorCount = 60;
        deviceProps.warpSize = 64;
        std::snprintf(deviceProps.gcnArchName, sizeof(deviceProps.gcnArchName), "%s", "gfx942");

        EXPECT_CALL(_mockDevicePropertyProvider, getDeviceProperties())
            .WillOnce(::testing::Return(deviceProps));

        auto mockKernel = std::make_unique<MockRunnableKernel>();
        EXPECT_CALL(*mockKernel, setBlockSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
        EXPECT_CALL(*mockKernel, setGridSize(::testing::_, ::testing::_, ::testing::_)).Times(1);

        auto mockProgram = std::make_unique<MockCompiledProgram>();
        EXPECT_CALL(*mockProgram, getKernel(::testing::_))
            .WillOnce(::testing::Return(::testing::ByMove(std::move(mockKernel))));

        EXPECT_CALL(_mockKernelCompiler, compile(::testing::_, ::testing::_))
            .WillOnce(::testing::Return(::testing::ByMove(std::move(mockProgram))));
    }
};

// ============================================================================
// isApplicable - valid graphs
// ============================================================================

TEST_F(TestResampleBwdPlanBuilder, IsApplicableReturnsTrueForValidGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestResampleBwdPlanBuilder, IsApplicableReturnsTrueWithoutOptionalAttributes)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph(
        false, ResampleMode::AVGPOOL_EXCLUDE_PADDING);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_dummyHandle, graph));
}

// 16777215 * 256 = 2^32 - 256 dx elements is 2^24 - 1 blocks of 256 threads.
TEST_F(TestResampleBwdPlanBuilder, IsApplicableReturnsTrueForGradientAtThreadLimit)
{
    auto builder = createResampleBwdGraphWithGradientSize(16777215, 256);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_dummyHandle, graph));
}

// 401 * 10710641 = 2^32 - 255 dx elements needs 2^24 blocks, which is 2^32 threads.
TEST_F(TestResampleBwdPlanBuilder, IsApplicableReturnsFalseForGradientOverThreadLimit)
{
    auto builder = createResampleBwdGraphWithGradientSize(401, 10710641);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

// ============================================================================
// isApplicable - invalid graphs
// ============================================================================

TEST_F(TestResampleBwdPlanBuilder, IsNotApplicableForNonResampleGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestResampleBwdPlanBuilder, IsNotApplicableForNonF32ComputeType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();

    // Set the compute data type of the node to HALF
    auto* mutableGraph
        = hipdnn_flatbuffers_sdk::data_objects::GetMutableGraph(builder.GetBufferPointer());
    mutableGraph->mutable_nodes()->GetMutableObject(0)->mutate_compute_data_type(
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

// ============================================================================
// buildPlan - valid graphs
// ============================================================================

TEST_F(TestResampleBwdPlanBuilder, BuildPlanSetsPlanForSingleNodeGraph)
{
    setupMockCompileChain();

    auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    Context ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(_dummyHandle, graph, _mockEngineConfig, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

// ============================================================================
// getMaxWorkspaceSize
// ============================================================================

TEST_F(TestResampleBwdPlanBuilder, GetMaxWorkspaceSizeReturnsZero)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const Settings settings;

    EXPECT_EQ(_planBuilder.getMaxWorkspaceSize(_dummyHandle, graph, settings), 0u);
}

// ============================================================================
// getCustomKnobs
// ============================================================================

TEST_F(TestResampleBwdPlanBuilder, GetCustomKnobsReturnsEmpty)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    auto knobs = _planBuilder.getCustomKnobs(_dummyHandle, graph);
    EXPECT_TRUE(knobs.empty());
}
