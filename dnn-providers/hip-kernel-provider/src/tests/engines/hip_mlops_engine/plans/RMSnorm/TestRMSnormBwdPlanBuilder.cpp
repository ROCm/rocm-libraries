// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "HipKernelContext.hpp"
#include "HipKernelHandle.hpp"
#include "engines/hip_mlops_engine/plans/RMSnorm/RMSnormBwdPlanBuilder.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockDevicePropertyProvider.hpp"
#include "mocks/MockKernelCompiler.hpp"
#include "mocks/MockRunnableKernel.hpp"

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::rmsnorm;
using hipdnn_test_sdk::utilities::MockEngineConfig;

class TestRMSnormBwdPlanBuilder : public ::testing::Test
{
protected:
    MockKernelCompiler _mockKernelCompiler;
    MockDevicePropertyProvider _mockDevicePropertyProvider;
    RMSnormBwdPlanBuilder _planBuilder{_mockKernelCompiler, _mockDevicePropertyProvider};
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

        // First mock kernel for BwdData
        auto mockKernel1 = std::make_unique<MockRunnableKernel>();
        EXPECT_CALL(*mockKernel1, setBlockSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
        EXPECT_CALL(*mockKernel1, setGridSize(::testing::_, ::testing::_, ::testing::_)).Times(1);

        // Second mock kernel for BwdWeightBias
        auto mockKernel2 = std::make_unique<MockRunnableKernel>();
        EXPECT_CALL(*mockKernel2, setBlockSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
        EXPECT_CALL(*mockKernel2, setGridSize(::testing::_, ::testing::_, ::testing::_)).Times(1);

        auto mockProgram = std::make_unique<MockCompiledProgram>();
        EXPECT_CALL(*mockProgram, getKernel(::testing::_))
            .WillOnce(::testing::Return(::testing::ByMove(std::move(mockKernel1))))
            .WillOnce(::testing::Return(::testing::ByMove(std::move(mockKernel2))));

        EXPECT_CALL(_mockKernelCompiler, compile(::testing::_, ::testing::_))
            .WillOnce(::testing::Return(::testing::ByMove(std::move(mockProgram))));
    }
};

// ============================================================================
// isApplicable - valid graphs
// ============================================================================

TEST_F(TestRMSnormBwdPlanBuilder, IsApplicableReturnsTrueForValidSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestRMSnormBwdPlanBuilder, IsApplicableReturnsTrueWithoutOptionalAttributes)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph(
        {150528, 50176, 224, 1}, {1, 3, 224, 224}, false);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_dummyHandle, graph));
}

// ============================================================================
// isApplicable - invalid graphs
// ============================================================================

TEST_F(TestRMSnormBwdPlanBuilder, IsNotApplicableForBatchnormGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestRMSnormBwdPlanBuilder, IsNotApplicableForNonF32ComputeType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph(
        {150528, 50176, 224, 1},
        {1, 3, 224, 224},
        true,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

// ============================================================================
// buildPlan - valid graphs
// ============================================================================

TEST_F(TestRMSnormBwdPlanBuilder, BuildPlanSetsPlanForSingleNodeGraph)
{
    setupMockCompileChain();

    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    HipKernelContext ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(_dummyHandle, graph, _mockEngineConfig, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

// ============================================================================
// getMaxWorkspaceSize
// ============================================================================

TEST_F(TestRMSnormBwdPlanBuilder, GetMaxWorkspaceSizeReturnsZero)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const HipKernelSettings settings;

    EXPECT_EQ(_planBuilder.getMaxWorkspaceSize(_dummyHandle, graph, settings), 0u);
}

// ============================================================================
// getCustomKnobs
// ============================================================================

TEST_F(TestRMSnormBwdPlanBuilder, GetCustomKnobsReturnsEmpty)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    auto knobs = _planBuilder.getCustomKnobs(_dummyHandle, graph);
    EXPECT_TRUE(knobs.empty());
}
