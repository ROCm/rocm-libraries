// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <miopen/miopen.h>

#include "HipdnnEnginePluginHandle.hpp"
#include "engines/plans/MiopenConvPlanBuilder.hpp"

using namespace miopen_plugin;
using namespace hipdnn_data_sdk::flatbuffer_utilities;
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

    void executePlan(const miopen_plugin::IPlan& plan, const hipdnn_plugin_sdk::IGraph& graph)
    {
        size_t workspaceSize = plan.getWorkspaceSize(_handle);
        hipdnn_data_sdk::utilities::Workspace workspace(workspaceSize);

        std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers;
        std::vector<std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>> tensors;

        const auto& tensorMap = graph.getTensorMap();
        for(const auto& [uid, tensorAttrPtr] : tensorMap)
        {
            if(!tensorAttrPtr->virtual_())
            {
                auto dims = hipdnn_data_sdk::utilities::convertFlatBufferVectorToStdVector(
                    tensorAttrPtr->dims());
                auto strides = hipdnn_data_sdk::utilities::convertFlatBufferVectorToStdVector(
                    tensorAttrPtr->strides());

                auto tensor = hipdnn_data_sdk::utilities::createTensor(tensorAttrPtr->data_type(),
                                                                       dims,
                                                                       strides);

                deviceBuffers.push_back({tensorAttrPtr->uid(), tensor->rawDeviceData()});
                tensors.push_back(std::move(tensor));
            }
        }

        EXPECT_NO_THROW(plan.execute(_handle,
                                     deviceBuffers.data(),
                                     static_cast<uint32_t>(deviceBuffers.size()),
                                     workspace.get()));
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
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    bool applicable = _planBuilder.isApplicable(_dummyHandle, graph);
    EXPECT_FALSE(applicable);
}

TEST_F(TestGpuMiopenConvPlanBuilder, IsApplicableReturnsTrueForSupportedGraph)
{
    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
        hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                  builder.GetSize());

        bool applicable = _planBuilder.isApplicable(_handle, graph);
        EXPECT_TRUE(applicable);
    }

    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
        hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                  builder.GetSize());

        bool applicable = _planBuilder.isApplicable(_handle, graph);
        EXPECT_TRUE(applicable);
    }

    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvWrwGraph();
        hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                  builder.GetSize());

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
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

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
        hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                  builder.GetSize());

        EXPECT_NO_THROW(_planBuilder.getMaxWorkspaceSize(_handle, graph));
    }

    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
        hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                  builder.GetSize());

        EXPECT_NO_THROW(_planBuilder.getMaxWorkspaceSize(_handle, graph));
    }

    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvWrwGraph();
        hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                  builder.GetSize());

        EXPECT_NO_THROW(_planBuilder.getMaxWorkspaceSize(_handle, graph));
    }
}

TEST_F(TestMiopenConvPlanBuilder, BuildPlanThrowsForMultiNodeGraph)
{
    MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(2));
    HipdnnEnginePluginExecutionContext ctx;
    MockEngineConfig mockEngineConfig;

    EXPECT_THROW(_planBuilder.buildPlan(_dummyHandle, mockGraph, mockEngineConfig, ctx),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_FALSE(ctx.hasValidPlan());
}

TEST_F(TestMiopenConvPlanBuilder, BuildPlanThrowsForUnsupportedGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());
    HipdnnEnginePluginExecutionContext ctx;
    MockEngineConfig mockEngineConfig;

    EXPECT_THROW(_planBuilder.buildPlan(_dummyHandle, graph, mockEngineConfig, ctx),
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

    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());
    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestGpuMiopenConvPlanBuilder, BuildPlanCreatesValidPlanForSupportedGraph)
{
    MockEngineConfig mockEngineConfig;

    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
        hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                  builder.GetSize());
        HipdnnEnginePluginExecutionContext ctx;

        EXPECT_NO_THROW(_planBuilder.buildPlan(_handle, graph, mockEngineConfig, ctx));
        EXPECT_TRUE(ctx.hasValidPlan());
    }

    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
        hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                  builder.GetSize());
        HipdnnEnginePluginExecutionContext ctx;

        EXPECT_NO_THROW(_planBuilder.buildPlan(_handle, graph, mockEngineConfig, ctx));
        EXPECT_TRUE(ctx.hasValidPlan());
    }

    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvWrwGraph();
        hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                  builder.GetSize());
        HipdnnEnginePluginExecutionContext ctx;

        EXPECT_NO_THROW(_planBuilder.buildPlan(_handle, graph, mockEngineConfig, ctx));
        EXPECT_TRUE(ctx.hasValidPlan());
    }
}

TEST_F(TestGpuMiopenConvPlanBuilder, ActualWorkspaceSizeIsWithinRangeFwd)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto range = _planBuilder.getWorkspaceSizeRange(_handle, graph);

    HipdnnEnginePluginExecutionContext ctx;
    _planBuilder.buildPlan(_handle, graph, ctx);

    size_t actualWorkspace = ctx.plan().getWorkspaceSize(_handle);

    EXPECT_GE(actualWorkspace, range.min);
    EXPECT_LE(actualWorkspace, range.max);
}

TEST_F(TestGpuMiopenConvPlanBuilder, ActualWorkspaceSizeIsWithinRangeBwd)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto range = _planBuilder.getWorkspaceSizeRange(_handle, graph);

    HipdnnEnginePluginExecutionContext ctx;
    _planBuilder.buildPlan(_handle, graph, ctx);

    size_t actualWorkspace = ctx.plan().getWorkspaceSize(_handle);

    EXPECT_GE(actualWorkspace, range.min);
    EXPECT_LE(actualWorkspace, range.max);
}

TEST_F(TestGpuMiopenConvPlanBuilder, ActualWorkspaceSizeIsWithinRangeWrw)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvWrwGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto range = _planBuilder.getWorkspaceSizeRange(_handle, graph);

    HipdnnEnginePluginExecutionContext ctx;
    _planBuilder.buildPlan(_handle, graph, ctx);

    size_t actualWorkspace = ctx.plan().getWorkspaceSize(_handle);

    EXPECT_GE(actualWorkspace, range.min);
    EXPECT_LE(actualWorkspace, range.max);
}

TEST_F(TestGpuMiopenConvPlanBuilder, PlanExecutesWithMinWorkspaceLimitFwd)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto range = _planBuilder.getWorkspaceSizeRange(_handle, graph);
    ASSERT_NE(range.min, range.max) << "No workspace size range available for testing";

    HipdnnEnginePluginExecutionContext ctx1;
    ctx1.setDebugMode(HipdnnEnginePluginExecutionContext::DebugMode::LOG_ALL_FOUND_PLAN_ALGORITHMS);
    _planBuilder.buildPlan(_handle, graph, ctx1);
    executePlan(ctx1.plan(), graph);

    HipdnnEnginePluginExecutionContext ctx2;
    ctx2.setDebugMode(HipdnnEnginePluginExecutionContext::DebugMode::LOG_ALL_FOUND_PLAN_ALGORITHMS);
    ctx2.setWorkspaceSizeLimit(range.min);
    _planBuilder.buildPlan(_handle, graph, ctx2);
    executePlan(ctx2.plan(), graph);
}

TEST_F(TestGpuMiopenConvPlanBuilder, PlanExecutesWithMinWorkspaceLimitBwd)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto range = _planBuilder.getWorkspaceSizeRange(_handle, graph);
    ASSERT_NE(range.min, range.max) << "No workspace size range available for testing";

    HipdnnEnginePluginExecutionContext ctx1;
    ctx1.setDebugMode(HipdnnEnginePluginExecutionContext::DebugMode::LOG_ALL_FOUND_PLAN_ALGORITHMS);
    _planBuilder.buildPlan(_handle, graph, ctx1);
    executePlan(ctx1.plan(), graph);

    HipdnnEnginePluginExecutionContext ctx2;
    ctx2.setDebugMode(HipdnnEnginePluginExecutionContext::DebugMode::LOG_ALL_FOUND_PLAN_ALGORITHMS);
    ctx2.setWorkspaceSizeLimit(range.min);
    _planBuilder.buildPlan(_handle, graph, ctx2);
    executePlan(ctx2.plan(), graph);
}

TEST_F(TestGpuMiopenConvPlanBuilder, PlanExecutesWithMinWorkspaceLimitWrw)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvWrwGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto range = _planBuilder.getWorkspaceSizeRange(_handle, graph);
    ASSERT_NE(range.min, range.max) << "No workspace size range available for testing";

    HipdnnEnginePluginExecutionContext ctx1;
    ctx1.setDebugMode(HipdnnEnginePluginExecutionContext::DebugMode::LOG_ALL_FOUND_PLAN_ALGORITHMS);
    _planBuilder.buildPlan(_handle, graph, ctx1);
    executePlan(ctx1.plan(), graph);

    HipdnnEnginePluginExecutionContext ctx2;
    ctx2.setDebugMode(HipdnnEnginePluginExecutionContext::DebugMode::LOG_ALL_FOUND_PLAN_ALGORITHMS);
    ctx2.setWorkspaceSizeLimit(range.min);
    _planBuilder.buildPlan(_handle, graph, ctx2);
    executePlan(ctx2.plan(), graph);
}
