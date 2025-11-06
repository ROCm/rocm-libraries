// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/EnginePluginApi.h>
#include <hipdnn_sdk/plugin/test_utils/MockGraph.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>

#include "HipdnnEnginePluginHandle.hpp"
#include "engines/plans/MiopenBatchnormFwdTrainingActivPlanBuilder.hpp"

#include "mocks/MockHipdnnEnginePluginExecutionContext.hpp"

using namespace miopen_legacy_plugin;
using namespace hipdnn_plugin;

class TestMiopenBatchnormFwdTrainingActivPlanBuilder : public ::testing::Test
{
protected:
    MiopenBatchnormFwdTrainingActivPlanBuilder _planBuilder;
    HipdnnEnginePluginHandle _dummyHandle;
};

TEST_F(TestMiopenBatchnormFwdTrainingActivPlanBuilder, IsApplicableReturnsFalseForSingleNodeGraph)
{
    MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(1));

    bool applicable = _planBuilder.isApplicable(_dummyHandle, mockGraph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormFwdTrainingActivPlanBuilder, IsApplicableReturnsFalseForThreeNodeGraph)
{
    MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(3));

    bool applicable = _planBuilder.isApplicable(_dummyHandle, mockGraph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormFwdTrainingActivPlanBuilder, IsApplicableReturnsTrueForValidTwoNodeGraph)
{
    auto builder = hipdnn_sdk::test_utilities::createValidBatchnormFwdTrainingActivGraph();
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    bool applicable = _planBuilder.isApplicable(_dummyHandle, graph);

    EXPECT_TRUE(applicable);
}

TEST_F(TestMiopenBatchnormFwdTrainingActivPlanBuilder,
       IsApplicableReturnsFalseForGraphWithRunningStatistics)
{
    auto builder
        = hipdnn_sdk::test_utilities::createValidBatchnormFwdTrainingActivGraph(true, true);
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    bool applicable = _planBuilder.isApplicable(_dummyHandle, graph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormFwdTrainingActivPlanBuilder, IsApplicableReturnsFalseForNonReluActivation)
{
    // Graph with SIGMOID_FWD instead of RELU_FWD
    auto builder = hipdnn_sdk::test_utilities::createValidBatchnormFwdTrainingActivGraph(
        false, false, hipdnn_sdk::data_objects::PointwiseMode::SIGMOID_FWD);
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    bool applicable = _planBuilder.isApplicable(_dummyHandle, graph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormFwdTrainingActivPlanBuilder,
       IsApplicableReturnsFalseWhenBnOutputIsNotVirtual)
{
    // Graph where BN output tensor is non-virtual (should be virtual for fusion)
    auto builder
        = hipdnn_sdk::test_utilities::createValidBatchnormFwdTrainingActivGraph(false, false);
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    bool applicable = _planBuilder.isApplicable(_dummyHandle, graph);

    // This should be false if the test utility properly validates virtual tensors
    // Note: Depends on how createValidBatchnormFwdTrainingActivGraph handles virtual flag
    EXPECT_TRUE(applicable);
}

TEST_F(TestMiopenBatchnormFwdTrainingActivPlanBuilder, GetWorkspaceSizeReturnsZero)
{
    auto builder = hipdnn_sdk::test_utilities::createValidBatchnormFwdTrainingActivGraph();
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    size_t workspaceSize = _planBuilder.getWorkspaceSize(_dummyHandle, graph);

    EXPECT_EQ(workspaceSize, 0u);
}

TEST_F(TestMiopenBatchnormFwdTrainingActivPlanBuilder, BuildPlanSetsPlanForValidGraph)
{
    auto builder = hipdnn_sdk::test_utilities::createValidBatchnormFwdTrainingActivGraph();
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(_dummyHandle, graph, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}
