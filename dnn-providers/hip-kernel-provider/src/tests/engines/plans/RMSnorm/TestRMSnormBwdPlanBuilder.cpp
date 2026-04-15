// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "HipKernelContext.hpp"
#include "HipKernelHandle.hpp"
#include "engines/plans/RMSnorm/RMSnormBwdPlanBuilder.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockDevicePropertyProvider.hpp"
#include "mocks/MockKernelCompiler.hpp"
#include "mocks/MockRunnableKernel.hpp"

#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
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
};

// ============================================================================
// isApplicable - valid graphs
// ============================================================================

TEST_F(TestRMSnormBwdPlanBuilder, IsApplicableReturnsTrueForValidSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestRMSnormBwdPlanBuilder, IsApplicableReturnsTrueWithoutOptionalAttributes)
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

TEST_F(TestRMSnormBwdPlanBuilder, IsNotApplicableForBatchnormGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestRMSnormBwdPlanBuilder, IsNotApplicableForNonF32ComputeType)
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

TEST_F(TestRMSnormBwdPlanBuilder, BuildPlanSetsPlanForSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());
    HipKernelContext ctx;

    EXPECT_CALL(_mockDevicePropertyProvider, getDeviceProperties())
        .WillOnce(::testing::Return(hipDeviceProp_t{}));

    EXPECT_THROW(_planBuilder.buildPlan(_dummyHandle, graph, _mockEngineConfig, ctx),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

// ============================================================================
// getMaxWorkspaceSize
// ============================================================================

TEST_F(TestRMSnormBwdPlanBuilder, GetMaxWorkspaceSizeReturnsZero)
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

TEST_F(TestRMSnormBwdPlanBuilder, GetCustomKnobsReturnsEmpty)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    auto knobs = _planBuilder.getCustomKnobs(_dummyHandle, graph);
    EXPECT_TRUE(knobs.empty());
}
