// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdio>
#include <gtest/gtest.h>

#include "HipKernelContext.hpp"
#include "HipKernelHandle.hpp"
#include "engines/plans/batchnorm/BatchnormPlanBuilder.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockDevicePropertyProvider.hpp"
#include "mocks/MockKernelCompiler.hpp"
#include "mocks/MockRunnableKernel.hpp"

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_test_sdk/utilities/MockNode.hpp>

using hipdnn_test_sdk::utilities::MockEngineConfig;

namespace hip_kernel_provider::batchnorm::test
{

class TestBatchnormPlanBuilder : public ::testing::Test
{
protected:
    MockKernelCompiler _mockKernelCompiler;
    MockDevicePropertyProvider _mockDevicePropertyProvider;
    BatchnormPlanBuilder _planBuilder{_mockKernelCompiler, _mockDevicePropertyProvider};
    HipKernelHandle _dummyHandle;
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

TEST_F(TestBatchnormPlanBuilder, IsApplicableReturnsTrueForValidInferenceGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestBatchnormPlanBuilder, IsApplicableReturnsTrueForValidBwdGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestBatchnormPlanBuilder, IsApplicableReturnsTrueForValidFwdTrainingGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestBatchnormPlanBuilder, IsApplicableReturnsTrueForValidThreeNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferActBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_dummyHandle, graph));
}

// ============================================================================
// buildPlan - valid graphs
// ============================================================================

TEST_F(TestBatchnormPlanBuilder, BuildPlanSetsPlanForSingleNodeInference)
{
    setupMockCompileChain();

    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    HipKernelContext ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(_dummyHandle, graph, _mockEngineConfig, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestBatchnormPlanBuilder, BuildPlanSetsPlanForSingleNodeBwd)
{
    setupMockCompileChain();

    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    HipKernelContext ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(_dummyHandle, graph, _mockEngineConfig, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestBatchnormPlanBuilder, BuildPlanSetsPlanForSingleNodeFwdTraining)
{
    setupMockCompileChain();

    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    HipKernelContext ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(_dummyHandle, graph, _mockEngineConfig, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestBatchnormPlanBuilder, BuildPlanSetsPlanForFusedBackwardGraph)
{
    setupMockCompileChain();

    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferActBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    HipKernelContext ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(_dummyHandle, graph, _mockEngineConfig, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestBatchnormPlanBuilder, GetMaxWorkspaceSizeReturnsZero)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const HipKernelSettings settings;

    EXPECT_EQ(_planBuilder.getMaxWorkspaceSize(_dummyHandle, graph, settings), 0u);
}

TEST_F(TestBatchnormPlanBuilder, GetMaxWorkspaceSizeReturnsZeroForFwdTrainingGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const HipKernelSettings settings;

    EXPECT_EQ(_planBuilder.getMaxWorkspaceSize(_dummyHandle, graph, settings), 0u);
}

// ============================================================================
// getCustomKnobs
// ============================================================================

TEST_F(TestBatchnormPlanBuilder, GetCustomKnobsReturnsEmpty)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    auto knobs = _planBuilder.getCustomKnobs(_dummyHandle, graph);
    EXPECT_TRUE(knobs.empty());
}

TEST_F(TestBatchnormPlanBuilder, GetCustomKnobsReturnsEmptyForFwdTrainingGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    auto knobs = _planBuilder.getCustomKnobs(_dummyHandle, graph);
    EXPECT_TRUE(knobs.empty());
}

} // namespace hip_kernel_provider::batchnorm::test
