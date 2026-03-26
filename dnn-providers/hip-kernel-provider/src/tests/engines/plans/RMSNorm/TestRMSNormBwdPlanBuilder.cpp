// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "HipKernelContext.hpp"
#include "HipKernelHandle.hpp"
#include "engines/plans/rmsnorm/RMSNormBwdPlanBuilder.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockDevicePropertyProvider.hpp"
#include "mocks/MockKernelCompiler.hpp"
#include "mocks/MockRunnableKernel.hpp"

#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>

using namespace hip_kernel_provider;
using hipdnn_test_sdk::utilities::MockEngineConfig;

class TestRMSNormBwdPlanBuilder : public ::testing::Test
{
protected:
    MockKernelCompiler _mockKernelCompiler;
    MockDevicePropertyProvider _mockDevicePropertyProvider;
    RMSNormBwdPlanBuilder _planBuilder{_mockKernelCompiler, _mockDevicePropertyProvider};
    HipKernelHandle _dummyHandle;
    MockEngineConfig _mockEngineConfig;
};

// ============================================================================
// isApplicable - valid graphs
// ============================================================================

TEST_F(TestRMSNormBwdPlanBuilder, IsApplicableReturnsTrueForValidSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestRMSNormBwdPlanBuilder, IsApplicableReturnsTrueWithoutOptionalAttributes)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph(
        {150528, 50176, 224, 1}, {1, 3, 224, 224}, false);
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_dummyHandle, graph));
}

// ============================================================================
// isApplicable - invalid graphs
// ============================================================================

TEST_F(TestRMSNormBwdPlanBuilder, IsNotApplicableForBatchnormGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestRMSNormBwdPlanBuilder, IsNotApplicableForNonF32ComputeType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph(
        {150528, 50176, 224, 1},
        {1, 3, 224, 224},
        true,
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::DataType::HALF);
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

// ============================================================================
// buildPlan - valid graphs
// ============================================================================

TEST_F(TestRMSNormBwdPlanBuilder, BuildPlanSetsPlanForSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());
    HipKernelContext ctx;

    EXPECT_CALL(_mockDevicePropertyProvider, getDeviceProperties())
        .WillOnce(::testing::Return(hipDeviceProp_t{}));

    EXPECT_NO_THROW(_planBuilder.buildPlan(_dummyHandle, graph, _mockEngineConfig, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

// ============================================================================
// getMaxWorkspaceSize
// ============================================================================

TEST_F(TestRMSNormBwdPlanBuilder, GetMaxWorkspaceSizeReturnsZero)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());
    HipKernelSettings settings;

    EXPECT_EQ(_planBuilder.getMaxWorkspaceSize(_dummyHandle, graph, settings), 0u);
}

// ============================================================================
// getCustomKnobs
// ============================================================================

TEST_F(TestRMSNormBwdPlanBuilder, GetCustomKnobsReturnsEmpty)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    auto knobs = _planBuilder.getCustomKnobs(_dummyHandle, graph);
    EXPECT_TRUE(knobs.empty());
}
