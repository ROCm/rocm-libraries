// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdio>
#include <gtest/gtest.h>

#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/plans/RMSnorm/RMSnormPlanBuilder.hpp"
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
using namespace hip_kernel_provider;
using namespace hip_kernel_provider::rmsnorm;

class TestRMSnormPlanBuilder : public ::testing::Test
{
protected:
    MockKernelCompiler _mockKernelCompiler;
    MockDevicePropertyProvider _mockDevicePropertyProvider;
    RMSnormPlanBuilder _planBuilder{_mockKernelCompiler, _mockDevicePropertyProvider};
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

TEST_F(TestRMSnormPlanBuilder, IsApplicableReturnsTrueForValidInferenceGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_dummyHandle, graph));
}
// ============================================================================
// isApplicable - invalid graphs
// ============================================================================

TEST_F(TestRMSnormPlanBuilder, IsApplicableReturnsFalseForThreeNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

// ============================================================================
// buildPlan - valid graphs
// ============================================================================

TEST_F(TestRMSnormPlanBuilder, BuildPlanSetsPlanForSingleNodeInference)
{
    setupMockCompileChain();

    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    Context ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(_dummyHandle, graph, _mockEngineConfig, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

// ============================================================================
// getMaxWorkspaceSize
// ============================================================================

TEST_F(TestRMSnormPlanBuilder, GetMaxWorkspaceSizeReturnsZero)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const Settings settings;

    EXPECT_EQ(_planBuilder.getMaxWorkspaceSize(_dummyHandle, graph, settings), 0u);
}

// ============================================================================
// getCustomKnobs
// ============================================================================

TEST_F(TestRMSnormPlanBuilder, GetCustomKnobsReturnsEmpty)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    auto knobs = _planBuilder.getCustomKnobs(_dummyHandle, graph);
    EXPECT_TRUE(knobs.empty());
}
