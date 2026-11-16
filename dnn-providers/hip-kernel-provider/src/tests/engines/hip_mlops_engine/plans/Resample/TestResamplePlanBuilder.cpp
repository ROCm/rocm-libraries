// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdio>
#include <gtest/gtest.h>

#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"
#include "engines/hip_mlops_engine/plans/resample/ResamplePlanBuilder.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockDevicePropertyProvider.hpp"
#include "mocks/MockKernelCompiler.hpp"
#include "mocks/MockRunnableKernel.hpp"

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>

using hipdnn_test_sdk::utilities::MockEngineConfig;
using namespace hip_kernel_provider;
using namespace hip_kernel_provider::resample;

namespace
{

// A 1x1 pooling window with stride 1, so y has the dims of x: [1, 1, height, width].
flatbuffers::FlatBufferBuilder createResampleFwdGraphWithOutputSize(int64_t height, int64_t width)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> dims = {1, 1, height, width};
    const std::vector<int64_t> strides = {height * width, height * width, width, 1};
    const std::vector<int64_t> padding = {0, 0};
    const std::vector<int64_t> unit = {1, 1};

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &strides, &dims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &strides, &dims));

    auto resampleAttr = hipdnn_flatbuffers_sdk::data_objects::CreateResampleFwdAttributesDirect(
        builder,
        1,
        2,
        ::flatbuffers::nullopt,
        &padding,
        &padding,
        &unit,
        &unit,
        hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL,
        hipdnn_flatbuffers_sdk::data_objects::PaddingMode::ZERO_PAD,
        ::flatbuffers::nullopt);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "resample_fwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ResampleFwdAttributes,
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

class TestResamplePlanBuilder : public ::testing::Test
{
protected:
    MockKernelCompiler _mockKernelCompiler;
    MockDevicePropertyProvider _mockDevicePropertyProvider;
    ResamplePlanBuilder _planBuilder{_mockKernelCompiler, _mockDevicePropertyProvider};
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

TEST_F(TestResamplePlanBuilder, IsApplicableReturnsTrueForValidGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_dummyHandle, graph));
}

// 16777215 * 256 = 2^32 - 256 output elements is 2^24 - 1 blocks of 256 threads.
TEST_F(TestResamplePlanBuilder, IsApplicableReturnsTrueForOutputAtThreadLimit)
{
    auto builder = createResampleFwdGraphWithOutputSize(16777215, 256);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_dummyHandle, graph));
}

// 401 * 10710641 = 2^32 - 255 output elements needs 2^24 blocks, which is 2^32 threads.
TEST_F(TestResamplePlanBuilder, IsApplicableReturnsFalseForOutputOverThreadLimit)
{
    auto builder = createResampleFwdGraphWithOutputSize(401, 10710641);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestResamplePlanBuilder, IsApplicableReturnsFalseForNonResampleGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestResamplePlanBuilder, BuildPlanSetsPlanForSingleNodeGraph)
{
    setupMockCompileChain();

    auto builder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    Context ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(_dummyHandle, graph, _mockEngineConfig, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestResamplePlanBuilder, GetMaxWorkspaceSizeReturnsZero)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const Settings settings;

    EXPECT_EQ(_planBuilder.getMaxWorkspaceSize(_dummyHandle, graph, settings), 0u);
}

TEST_F(TestResamplePlanBuilder, GetCustomKnobsReturnsEmpty)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    auto knobs = _planBuilder.getCustomKnobs(_dummyHandle, graph);
    EXPECT_TRUE(knobs.empty());
}
