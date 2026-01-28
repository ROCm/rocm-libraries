// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <miopen/miopen.h>

#include "HipdnnEnginePluginHandle.hpp"
#include "engines/plans/MiopenConvPlanBuilder.hpp"

using namespace miopen_legacy_plugin;
using namespace hipdnn_plugin_sdk;
using namespace hipdnn_test_sdk::utilities;

class TestMiopenConvPlanBuilder : public ::testing::Test
{
protected:
    MiopenConvPlanBuilder _planBuilder;
    HipdnnEnginePluginHandle _dummyHandle;
};

class TestGpuMiopenConvPlanBuilder : public TestMiopenConvPlanBuilder
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        ASSERT_EQ(miopenCreate(&_handle.miopenHandle), miopenStatusSuccess);
    }

    void TearDown() override
    {
        if(_handle.miopenHandle != nullptr)
        {
            EXPECT_EQ(miopenDestroy(_handle.miopenHandle), miopenStatusSuccess);
        }
    }

    HipdnnEnginePluginHandle _handle;
};

TEST_F(TestMiopenConvPlanBuilder, IsApplicableReturnsFalseForMultiNodeGraph)
{
    MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(2));

    bool applicable = _planBuilder.isApplicable(_dummyHandle, mockGraph);
    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenConvPlanBuilder, IsApplicableReturnsFalseForUnsupportedGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    bool applicable = _planBuilder.isApplicable(_dummyHandle, graph);
    EXPECT_FALSE(applicable);
}

TEST_F(TestGpuMiopenConvPlanBuilder, IsApplicableReturnsTrueForSupportedGraph)
{
    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
        hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        bool applicable = _planBuilder.isApplicable(_handle, graph);
        EXPECT_TRUE(applicable);
    }

    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
        hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        bool applicable = _planBuilder.isApplicable(_handle, graph);
        EXPECT_TRUE(applicable);
    }

    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvWrwGraph();
        hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        bool applicable = _planBuilder.isApplicable(_handle, graph);
        EXPECT_TRUE(applicable);
    }
}

TEST_F(TestMiopenConvPlanBuilder, GetWorkspaceSizeThrowsForMultiNodeGraph)
{
    MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(2));

    EXPECT_THROW(_planBuilder.getMaxWorkspaceSize(_dummyHandle, mockGraph),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestMiopenConvPlanBuilder, GetWorkspaceSizeRangeThrowsForMultiNodeGraph)
{
    MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(2));

    EXPECT_THROW(_planBuilder.getWorkspaceSizeRange(_dummyHandle, mockGraph),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestMiopenConvPlanBuilder, GetWorkspaceSizeThrowsForUnsupportedGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_THROW(_planBuilder.getMaxWorkspaceSize(_dummyHandle, graph),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestMiopenConvPlanBuilder, GetWorkspaceSizeRangeThrowsForUnsupportedGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_THROW(_planBuilder.getWorkspaceSizeRange(_dummyHandle, graph),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestGpuMiopenConvPlanBuilder, GetWorkspaceSizeReturnsValueForSupportedGraph)
{
    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
        hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_NO_THROW(_planBuilder.getMaxWorkspaceSize(_handle, graph));
    }

    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
        hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_NO_THROW(_planBuilder.getMaxWorkspaceSize(_handle, graph));
    }

    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvWrwGraph();
        hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_NO_THROW(_planBuilder.getMaxWorkspaceSize(_handle, graph));
    }
}

TEST_F(TestMiopenConvPlanBuilder, BuildPlanThrowsForMultiNodeGraph)
{
    MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(2));
    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_THROW(_planBuilder.buildPlan(_dummyHandle, mockGraph, ctx),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_FALSE(ctx.hasValidPlan());
}

TEST_F(TestMiopenConvPlanBuilder, BuildPlanThrowsForUnsupportedGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_THROW(_planBuilder.buildPlan(_dummyHandle, graph, ctx),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_FALSE(ctx.hasValidPlan());
}

TEST_F(TestMiopenConvPlanBuilder, IsApplicableReturnsFalseForUnsupportedComputeType)
{
    flatbuffers::FlatBufferBuilder builder
        = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();

    auto mutableGraph = hipdnn_data_sdk::data_objects::GetMutableGraph(builder.GetBufferPointer());
    mutableGraph->mutable_nodes()->GetMutableObject(0)->mutate_compute_data_type(
        hipdnn_data_sdk::data_objects::DataType::HALF);

    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestGpuMiopenConvPlanBuilder, BuildPlanCreatesValidPlanForSupportedGraph)
{
    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
        hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
        HipdnnEnginePluginExecutionContext ctx;

        EXPECT_NO_THROW(_planBuilder.buildPlan(_handle, graph, ctx));
        EXPECT_TRUE(ctx.hasValidPlan());
    }

    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
        hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
        HipdnnEnginePluginExecutionContext ctx;

        EXPECT_NO_THROW(_planBuilder.buildPlan(_handle, graph, ctx));
        EXPECT_TRUE(ctx.hasValidPlan());
    }

    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvWrwGraph();
        hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
        HipdnnEnginePluginExecutionContext ctx;

        EXPECT_NO_THROW(_planBuilder.buildPlan(_handle, graph, ctx));
        EXPECT_TRUE(ctx.hasValidPlan());
    }
}

TEST_F(TestGpuMiopenConvPlanBuilder, GetWorkspaceSizeRangeReturnsValidRangeForConvFwd)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    WorkspaceSizeRange range;
    EXPECT_NO_THROW(range = _planBuilder.getWorkspaceSizeRange(_handle, graph));

    EXPECT_LE(range.min, range.max);

    size_t maxWorkspace = _planBuilder.getMaxWorkspaceSize(_handle, graph);
    EXPECT_EQ(range.max, maxWorkspace);
}

TEST_F(TestGpuMiopenConvPlanBuilder, GetWorkspaceSizeRangeReturnsValidRangeForConvBwd)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    WorkspaceSizeRange range;
    EXPECT_NO_THROW(range = _planBuilder.getWorkspaceSizeRange(_handle, graph));

    EXPECT_LE(range.min, range.max);

    size_t maxWorkspace = _planBuilder.getMaxWorkspaceSize(_handle, graph);
    EXPECT_EQ(range.max, maxWorkspace);
}

TEST_F(TestGpuMiopenConvPlanBuilder, GetWorkspaceSizeRangeReturnsValidRangeForConvWrw)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvWrwGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    WorkspaceSizeRange range;
    EXPECT_NO_THROW(range = _planBuilder.getWorkspaceSizeRange(_handle, graph));

    EXPECT_LE(range.min, range.max);

    size_t maxWorkspace = _planBuilder.getMaxWorkspaceSize(_handle, graph);
    EXPECT_EQ(range.max, maxWorkspace);
}

TEST_F(TestGpuMiopenConvPlanBuilder, WorkspaceSizeRespectsLimitFwd)
{
    // Note: This test requires a graph for which MIOpen has multiple solutions
    // with at least two different workspace sizes. Otherwise, we cannot verify
    // that the workspace limit is enforced.
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    WorkspaceSizeRange range = _planBuilder.getWorkspaceSizeRange(_handle, graph);
    ASSERT_NE(range.min, range.max)
        << "No workspace size range available for testing";

    // TODO: We should force MIOpen to select an algorithm that uses more than the minimum workspace.
    HipdnnEnginePluginExecutionContext ctxUnlimited;
    _planBuilder.buildPlan(_handle, graph, ctxUnlimited);
    size_t workspaceUnlimited = ctxUnlimited.plan().getWorkspaceSize(_handle);

    ASSERT_GT(workspaceUnlimited, range.min)
        << "Unlimited workspace is already at minimum, cannot test limit";

    HipdnnEnginePluginExecutionContext ctxLimited;
    ctxLimited.setWorkspaceSizeLimit(range.min);
    _planBuilder.buildPlan(_handle, graph, ctxLimited);

    size_t workspaceLimited = ctxLimited.plan().getWorkspaceSize(_handle);
    EXPECT_EQ(workspaceLimited, range.min);
}

TEST_F(TestGpuMiopenConvPlanBuilder, WorkspaceSizeRespectsLimitBwd)
{
    // Note: This test requires a graph for which MIOpen has multiple solutions
    // with at least two different workspace sizes. Otherwise, we cannot verify
    // that the workspace limit is enforced.
    auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    WorkspaceSizeRange range = _planBuilder.getWorkspaceSizeRange(_handle, graph);
    ASSERT_NE(range.min, range.max)
        << "No workspace size range available for testing";

    // TODO: We should force MIOpen to select an algorithm that uses more than the minimum workspace.
    HipdnnEnginePluginExecutionContext ctxUnlimited;
    _planBuilder.buildPlan(_handle, graph, ctxUnlimited);
    size_t workspaceUnlimited = ctxUnlimited.plan().getWorkspaceSize(_handle);

    ASSERT_GT(workspaceUnlimited, range.min)
        << "Unlimited workspace is already at minimum, cannot test limit";

    HipdnnEnginePluginExecutionContext ctxLimited;
    ctxLimited.setWorkspaceSizeLimit(range.min);
    _planBuilder.buildPlan(_handle, graph, ctxLimited);

    size_t workspaceLimited = ctxLimited.plan().getWorkspaceSize(_handle);
    EXPECT_EQ(workspaceLimited, range.min);
}

TEST_F(TestGpuMiopenConvPlanBuilder, WorkspaceSizeRespectsLimitWrw)
{
    // Note: This test requires a graph for which MIOpen has multiple solutions
    // with at least two different workspace sizes. Otherwise, we cannot verify
    // that the workspace limit is enforced.
    auto builder = hipdnn_test_sdk::utilities::createValidConvWrwGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    WorkspaceSizeRange range = _planBuilder.getWorkspaceSizeRange(_handle, graph);
    ASSERT_NE(range.min, range.max)
        << "No workspace size range available for testing";

    // TODO: We should force MIOpen to select an algorithm that uses more than the minimum workspace.
    HipdnnEnginePluginExecutionContext ctxUnlimited;
    _planBuilder.buildPlan(_handle, graph, ctxUnlimited);
    size_t workspaceUnlimited = ctxUnlimited.plan().getWorkspaceSize(_handle);

    ASSERT_GT(workspaceUnlimited, range.min)
        << "Unlimited workspace is already at minimum, cannot test limit";

    HipdnnEnginePluginExecutionContext ctxLimited;
    ctxLimited.setWorkspaceSizeLimit(range.min);
    _planBuilder.buildPlan(_handle, graph, ctxLimited);

    size_t workspaceLimited = ctxLimited.plan().getWorkspaceSize(_handle);
    EXPECT_EQ(workspaceLimited, range.min);
}
