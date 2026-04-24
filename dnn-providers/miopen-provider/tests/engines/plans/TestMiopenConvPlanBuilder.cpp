// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <memory>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>
#include <miopen/miopen.h>

#include "HipdnnMiopenHandle.hpp"
#include "engines/plans/MiopenConvPlanBuilder.hpp"

using namespace miopen_plugin;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace hipdnn_test_sdk::utilities;

class TestMiopenConvPlanBuilder : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        _dummyHandle = std::make_unique<HipdnnMiopenHandle>();
    }

    MiopenConvPlanBuilder _planBuilder;
    std::unique_ptr<HipdnnMiopenHandle> _dummyHandle;
};

class TestGpuMiopenConvPlanBuilder : public TestMiopenConvPlanBuilder
{
protected:
    void SetUp() override
    {
        TestMiopenConvPlanBuilder::SetUp();
        // GTEST_SKIP() in the base only unwinds the base SetUp() frame; without this
        // guard, _handle would still be constructed and miopenCreate() would throw on
        // no-device, turning the intended skip into a fixture failure.
        if(IsSkipped())
        {
            return;
        }
        _handle = std::make_unique<HipdnnMiopenHandle>();
    }

    void executePlan(const hipdnn_plugin_sdk::IPlan<HipdnnMiopenHandle>& plan,
                     const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph)
    {
        size_t workspaceSize = plan.getWorkspaceSize(*_handle);
        hipdnn_data_sdk::utilities::Workspace workspace(workspaceSize);

        std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers;
        std::vector<std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>> tensors;

        const auto& tensorMap = graph.getTensorMap();
        for(const auto& [uid, tensorAttrPtr] : tensorMap)
        {
            if(!tensorAttrPtr->virtual_())
            {
                auto dims = hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(
                    tensorAttrPtr->dims());
                auto strides
                    = hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(
                        tensorAttrPtr->strides());

                auto tensor = hipdnn_test_sdk::detail::createTensor(
                    tensorAttrPtr->data_type(), dims, strides);

                deviceBuffers.push_back({tensorAttrPtr->uid(), tensor->rawDeviceData()});
                tensors.push_back(std::move(tensor));
            }
        }

        EXPECT_NO_THROW(plan.execute(*_handle,
                                     deviceBuffers.data(),
                                     static_cast<uint32_t>(deviceBuffers.size()),
                                     workspace.get()));
    }

    std::unique_ptr<HipdnnMiopenHandle> _handle;
};

TEST_F(TestMiopenConvPlanBuilder, IsApplicableReturnsFalseForMultiNodeGraph)
{
    MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(2));

    bool applicable = _planBuilder.isApplicable(*_dummyHandle, mockGraph);
    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenConvPlanBuilder, IsApplicableReturnsFalseForUnsupportedGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    bool applicable = _planBuilder.isApplicable(*_dummyHandle, graph);
    EXPECT_FALSE(applicable);
}

TEST_F(TestGpuMiopenConvPlanBuilder, IsApplicableReturnsTrueForSupportedGraph)
{
    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
        hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                         builder.GetSize());

        bool applicable = _planBuilder.isApplicable(*_handle, graph);
        EXPECT_TRUE(applicable);
    }

    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
        hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                         builder.GetSize());

        bool applicable = _planBuilder.isApplicable(*_handle, graph);
        EXPECT_TRUE(applicable);
    }

    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvWrwGraph();
        hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                         builder.GetSize());

        bool applicable = _planBuilder.isApplicable(*_handle, graph);
        EXPECT_TRUE(applicable);
    }
}

TEST_F(TestMiopenConvPlanBuilder, GetWorkspaceSizeThrowsForMultiNodeGraph)
{
    MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(2));

    HipdnnMiopenSettings settings;
    EXPECT_THROW(_planBuilder.getMaxWorkspaceSize(*_dummyHandle, mockGraph, settings),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestMiopenConvPlanBuilder, GetWorkspaceSizeRangeThrowsForMultiNodeGraph)
{
    MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(2));

    EXPECT_THROW(_planBuilder.getWorkspaceSizeRange(*_dummyHandle, mockGraph),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestMiopenConvPlanBuilder, GetWorkspaceSizeThrowsForUnsupportedGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    HipdnnMiopenSettings settings;
    EXPECT_THROW(_planBuilder.getMaxWorkspaceSize(*_dummyHandle, graph, settings),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestMiopenConvPlanBuilder, GetWorkspaceSizeRangeThrowsForUnsupportedGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    EXPECT_THROW(_planBuilder.getWorkspaceSizeRange(*_dummyHandle, graph),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestGpuMiopenConvPlanBuilder, GetWorkspaceSizeReturnsValueForSupportedGraph)
{
    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
        hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                         builder.GetSize());

        HipdnnMiopenSettings settings;
        EXPECT_NO_THROW(_planBuilder.getMaxWorkspaceSize(*_handle, graph, settings));
    }

    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
        hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                         builder.GetSize());

        HipdnnMiopenSettings settings;
        EXPECT_NO_THROW(_planBuilder.getMaxWorkspaceSize(*_handle, graph, settings));
    }

    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvWrwGraph();
        hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                         builder.GetSize());

        HipdnnMiopenSettings settings;
        EXPECT_NO_THROW(_planBuilder.getMaxWorkspaceSize(*_handle, graph, settings));
    }
}

TEST_F(TestMiopenConvPlanBuilder, BuildPlanThrowsForMultiNodeGraph)
{
    MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(2));
    MockEngineConfig mockEngineConfig;
    HipdnnMiopenContext ctx;

    EXPECT_THROW(_planBuilder.buildPlan(*_dummyHandle, mockGraph, mockEngineConfig, ctx),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_FALSE(ctx.hasValidPlan());
}

TEST_F(TestMiopenConvPlanBuilder, BuildPlanThrowsForUnsupportedGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());
    MockEngineConfig mockEngineConfig;
    HipdnnMiopenContext ctx;

    EXPECT_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, mockEngineConfig, ctx),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_FALSE(ctx.hasValidPlan());
}

TEST_F(TestMiopenConvPlanBuilder, IsApplicableReturnsFalseForUnsupportedComputeType)
{
    flatbuffers::FlatBufferBuilder builder
        = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();

    auto mutableGraph
        = hipdnn_flatbuffers_sdk::data_objects::GetMutableGraph(builder.GetBufferPointer());
    mutableGraph->mutable_nodes()->GetMutableObject(0)->mutate_compute_data_type(
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());
    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestGpuMiopenConvPlanBuilder, BuildPlanCreatesValidPlanForSupportedGraph)
{
    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
        hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                         builder.GetSize());
        MockEngineConfig mockEngineConfig;
        HipdnnMiopenContext ctx;

        EXPECT_NO_THROW(_planBuilder.buildPlan(*_handle, graph, mockEngineConfig, ctx));
        EXPECT_TRUE(ctx.hasValidPlan());
    }

    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
        hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                         builder.GetSize());
        MockEngineConfig mockEngineConfig;
        HipdnnMiopenContext ctx;

        EXPECT_NO_THROW(_planBuilder.buildPlan(*_handle, graph, mockEngineConfig, ctx));
        EXPECT_TRUE(ctx.hasValidPlan());
    }

    {
        auto builder = hipdnn_test_sdk::utilities::createValidConvWrwGraph();
        hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                         builder.GetSize());
        MockEngineConfig mockEngineConfig;
        HipdnnMiopenContext ctx;

        EXPECT_NO_THROW(_planBuilder.buildPlan(*_handle, graph, mockEngineConfig, ctx));
        EXPECT_TRUE(ctx.hasValidPlan());
    }
}

TEST_F(TestGpuMiopenConvPlanBuilder, ActualWorkspaceSizeIsWithinRangeFwd)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    auto range = _planBuilder.getWorkspaceSizeRange(*_handle, graph);

    MockEngineConfig mockEngineConfig;
    HipdnnMiopenContext ctx;
    _planBuilder.buildPlan(*_handle, graph, mockEngineConfig, ctx);

    size_t actualWorkspace = ctx.plan().getWorkspaceSize(*_handle);

    EXPECT_GE(actualWorkspace, range.min);
    EXPECT_LE(actualWorkspace, range.max);
}

TEST_F(TestGpuMiopenConvPlanBuilder, ActualWorkspaceSizeIsWithinRangeBwd)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    auto range = _planBuilder.getWorkspaceSizeRange(*_handle, graph);

    MockEngineConfig mockEngineConfig;
    HipdnnMiopenContext ctx;
    _planBuilder.buildPlan(*_handle, graph, mockEngineConfig, ctx);

    size_t actualWorkspace = ctx.plan().getWorkspaceSize(*_handle);

    EXPECT_GE(actualWorkspace, range.min);
    EXPECT_LE(actualWorkspace, range.max);
}

TEST_F(TestGpuMiopenConvPlanBuilder, ActualWorkspaceSizeIsWithinRangeWrw)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvWrwGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    auto range = _planBuilder.getWorkspaceSizeRange(*_handle, graph);

    MockEngineConfig mockEngineConfig;
    HipdnnMiopenContext ctx;
    _planBuilder.buildPlan(*_handle, graph, mockEngineConfig, ctx);

    size_t actualWorkspace = ctx.plan().getWorkspaceSize(*_handle);

    EXPECT_GE(actualWorkspace, range.min);
    EXPECT_LE(actualWorkspace, range.max);
}

TEST_F(TestGpuMiopenConvPlanBuilder, PlanExecutesWithMinWorkspaceLimitFwd)
{
    // Configuration matching multiple MIOpen solvers with different workspace requirements
    std::vector<int64_t> xDims = {2, 16, 28, 28};
    auto xStrides = hipdnn_data_sdk::utilities::generateStrides(xDims);
    std::vector<int64_t> wDims = {32, 16, 3, 3};
    auto wStrides = hipdnn_data_sdk::utilities::generateStrides(wDims);
    std::vector<int64_t> yDims = {2, 32, 28, 28};
    auto yStrides = hipdnn_data_sdk::utilities::generateStrides(yDims);
    std::vector<int64_t> convPrePadding = {1, 1};
    std::vector<int64_t> convPostPadding = {1, 1};
    std::vector<int64_t> convStrides = {1, 1};
    std::vector<int64_t> convDilation = {1, 1};

    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph(xDims,
                                                                       xStrides,
                                                                       wDims,
                                                                       wStrides,
                                                                       yDims,
                                                                       yStrides,
                                                                       convPrePadding,
                                                                       convPostPadding,
                                                                       convStrides,
                                                                       convDilation);
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    auto range = _planBuilder.getWorkspaceSizeRange(*_handle, graph);
    ASSERT_NE(range.min, range.max) << "No workspace size range available for testing";

    MockEngineConfig mockEngineConfig;
    HipdnnMiopenSettings executionSettings1;
    HipdnnMiopenContext ctx1;
    ctx1.setExecutionSettings(executionSettings1);
    _planBuilder.buildPlan(*_handle, graph, mockEngineConfig, ctx1);
    executePlan(ctx1.plan(), graph);

    HipdnnMiopenSettings executionSettings2;
    executionSettings2.setWorkspaceSizeLimit(range.min);
    HipdnnMiopenContext ctx2;
    ctx2.setExecutionSettings(executionSettings2);
    _planBuilder.buildPlan(*_handle, graph, mockEngineConfig, ctx2);
    executePlan(ctx2.plan(), graph);
}

TEST_F(TestGpuMiopenConvPlanBuilder, PlanExecutesWithMinWorkspaceLimitBwd)
{
    // Configuration matching multiple MIOpen solvers with different workspace requirements
    std::vector<int64_t> dxDims = {2, 16, 28, 28};
    auto dxStrides = hipdnn_data_sdk::utilities::generateStrides(dxDims);
    std::vector<int64_t> wDims = {32, 16, 3, 3};
    auto wStrides = hipdnn_data_sdk::utilities::generateStrides(wDims);
    std::vector<int64_t> dyDims = {2, 32, 28, 28};
    auto dyStrides = hipdnn_data_sdk::utilities::generateStrides(dyDims);
    std::vector<int64_t> convPrePadding = {1, 1};
    std::vector<int64_t> convPostPadding = {1, 1};
    std::vector<int64_t> convStrides = {1, 1};
    std::vector<int64_t> convDilation = {1, 1};

    auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph(dxDims,
                                                                       dxStrides,
                                                                       wDims,
                                                                       wStrides,
                                                                       dyDims,
                                                                       dyStrides,
                                                                       convPrePadding,
                                                                       convPostPadding,
                                                                       convStrides,
                                                                       convDilation);
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    auto range = _planBuilder.getWorkspaceSizeRange(*_handle, graph);
    ASSERT_NE(range.min, range.max) << "No workspace size range available for testing";

    MockEngineConfig mockEngineConfig;
    HipdnnMiopenSettings executionSettings1;
    HipdnnMiopenContext ctx1;
    ctx1.setExecutionSettings(executionSettings1);
    _planBuilder.buildPlan(*_handle, graph, mockEngineConfig, ctx1);
    executePlan(ctx1.plan(), graph);

    HipdnnMiopenSettings executionSettings2;
    executionSettings2.setWorkspaceSizeLimit(range.min);
    HipdnnMiopenContext ctx2;
    ctx2.setExecutionSettings(executionSettings2);
    _planBuilder.buildPlan(*_handle, graph, mockEngineConfig, ctx2);
    executePlan(ctx2.plan(), graph);
}

TEST_F(TestGpuMiopenConvPlanBuilder, PlanExecutesWithMinWorkspaceLimitWrw)
{
    // Configuration matching multiple MIOpen solvers with different workspace requirements
    std::vector<int64_t> xDims = {2, 16, 28, 28};
    auto xStrides = hipdnn_data_sdk::utilities::generateStrides(xDims);
    std::vector<int64_t> dwDims = {32, 16, 3, 3};
    auto dwStrides = hipdnn_data_sdk::utilities::generateStrides(dwDims);
    std::vector<int64_t> dyDims = {2, 32, 28, 28};
    auto dyStrides = hipdnn_data_sdk::utilities::generateStrides(dyDims);
    std::vector<int64_t> convPrePadding = {1, 1};
    std::vector<int64_t> convPostPadding = {1, 1};
    std::vector<int64_t> convStrides = {1, 1};
    std::vector<int64_t> convDilation = {1, 1};

    auto builder = hipdnn_test_sdk::utilities::createValidConvWrwGraph(xDims,
                                                                       xStrides,
                                                                       dwDims,
                                                                       dwStrides,
                                                                       dyDims,
                                                                       dyStrides,
                                                                       convPrePadding,
                                                                       convPostPadding,
                                                                       convStrides,
                                                                       convDilation);
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    auto range = _planBuilder.getWorkspaceSizeRange(*_handle, graph);
    ASSERT_NE(range.min, range.max) << "No workspace size range available for testing";

    MockEngineConfig mockEngineConfig;
    HipdnnMiopenSettings executionSettings1;
    HipdnnMiopenContext ctx1;
    ctx1.setExecutionSettings(executionSettings1);
    _planBuilder.buildPlan(*_handle, graph, mockEngineConfig, ctx1);
    executePlan(ctx1.plan(), graph);

    HipdnnMiopenSettings executionSettings2;
    executionSettings2.setWorkspaceSizeLimit(range.min);
    HipdnnMiopenContext ctx2;
    ctx2.setExecutionSettings(executionSettings2);
    _planBuilder.buildPlan(*_handle, graph, mockEngineConfig, ctx2);
    executePlan(ctx2.plan(), graph);
}

TEST_F(TestGpuMiopenConvPlanBuilder, GetCustomKnobsReturnsWorkspaceSizeLimitKnob)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    auto customKnobs = _planBuilder.getCustomKnobs(*_handle, graph);
    const auto range = _planBuilder.getWorkspaceSizeRange(*_handle, graph);

    auto workspaceKnobIt
        = std::find_if(customKnobs.begin(),
                       customKnobs.end(),
                       [](const hipdnn_flatbuffers_sdk::data_objects::KnobT& knob) {
                           return knob.knob_id == hipdnn_plugin_sdk::WORKSPACE_SIZE_LIMIT_KNOB_NAME;
                       });

    ASSERT_NE(workspaceKnobIt, customKnobs.end());

    const auto& workspaceKnob = *workspaceKnobIt;
    EXPECT_EQ(workspaceKnob.description, "Workspace size limit in bytes");

    // Check default value (should be max)
    ASSERT_TRUE(workspaceKnob.default_value.AsIntValue() != nullptr);
    EXPECT_EQ(workspaceKnob.default_value.AsIntValue()->value, static_cast<int64_t>(range.max));

    // Check constraint
    ASSERT_TRUE(workspaceKnob.constraint.AsIntConstraint() != nullptr);
    const auto& constraint = *workspaceKnob.constraint.AsIntConstraint();
    EXPECT_EQ(constraint.min_value, static_cast<int64_t>(range.min));
    EXPECT_EQ(constraint.max_value, static_cast<int64_t>(range.max));
    EXPECT_EQ(constraint.step, 1);
}

TEST_F(TestMiopenConvPlanBuilder, GetCustomKnobsReturnsEmptyWhenNotApplicable)
{
    MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(0));

    auto customKnobs = _planBuilder.getCustomKnobs(*_dummyHandle, mockGraph);

    EXPECT_TRUE(customKnobs.empty());
}

TEST_F(TestGpuMiopenConvPlanBuilder,
       InitializeExecutionSettingsThrowsOnInvalidWorkspaceSizeLimitKnobType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    flatbuffers::FlatBufferBuilder configBuilder;
    auto knobIdOffset = configBuilder.CreateString("global.workspace_size_limit");
    auto stringValueOffset = configBuilder.CreateString("invalid_value");
    auto knobValue
        = hipdnn_flatbuffers_sdk::data_objects::CreateStringValue(configBuilder, stringValueOffset);
    hipdnn_flatbuffers_sdk::data_objects::KnobSettingBuilder knobSettingBuilder(configBuilder);
    knobSettingBuilder.add_knob_id(knobIdOffset);
    knobSettingBuilder.add_value_type(hipdnn_flatbuffers_sdk::data_objects::KnobValue::StringValue);
    knobSettingBuilder.add_value(knobValue.Union());
    auto knobSetting = knobSettingBuilder.Finish();

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::KnobSetting>> knobsVector;
    knobsVector.push_back(knobSetting);
    auto knobs = configBuilder.CreateVector(knobsVector);

    auto engineConfig
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(configBuilder, 1, knobs);
    configBuilder.Finish(engineConfig);

    auto buffer = configBuilder.Release();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper configWrapper(buffer.data(),
                                                                                    buffer.size());

    HipdnnMiopenSettings executionSettings;
    EXPECT_THROW(
        _planBuilder.initializeExecutionSettings(*_handle, graph, configWrapper, executionSettings),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestGpuMiopenConvPlanBuilder, InitializeExecutionSettingsThrowsOnNegativeWorkspaceSizeLimit)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    flatbuffers::FlatBufferBuilder configBuilder;
    auto knobIdOffset = configBuilder.CreateString("global.workspace_size_limit");
    auto knobValue = hipdnn_flatbuffers_sdk::data_objects::CreateIntValue(configBuilder, -1);
    hipdnn_flatbuffers_sdk::data_objects::KnobSettingBuilder knobSettingBuilder(configBuilder);
    knobSettingBuilder.add_knob_id(knobIdOffset);
    knobSettingBuilder.add_value_type(hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue);
    knobSettingBuilder.add_value(knobValue.Union());
    auto knobSetting = knobSettingBuilder.Finish();

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::KnobSetting>> knobsVector;
    knobsVector.push_back(knobSetting);
    auto knobs = configBuilder.CreateVector(knobsVector);

    auto engineConfig
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(configBuilder, 1, knobs);
    configBuilder.Finish(engineConfig);

    auto buffer = configBuilder.Release();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper configWrapper(buffer.data(),
                                                                                    buffer.size());

    HipdnnMiopenSettings executionSettings;
    EXPECT_THROW(
        _planBuilder.initializeExecutionSettings(*_handle, graph, configWrapper, executionSettings),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestGpuMiopenConvPlanBuilder,
       InitializeExecutionSettingsThrowsOnWorkspaceSizeLimitBelowRange)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto range = _planBuilder.getWorkspaceSizeRange(*_handle, graph);

    // Skip if min is 0 (already tested in BuildPlanThrowsOnNegativeWorkspaceSizeLimit)
    if(range.min == 0)
    {
        GTEST_SKIP() << "Skipping below-range test when min is 0";
    }

    flatbuffers::FlatBufferBuilder configBuilder;
    auto knobIdOffset = configBuilder.CreateString("global.workspace_size_limit");
    auto knobValue = hipdnn_flatbuffers_sdk::data_objects::CreateIntValue(
        configBuilder, static_cast<int64_t>(range.min) - 1);
    hipdnn_flatbuffers_sdk::data_objects::KnobSettingBuilder knobSettingBuilder(configBuilder);
    knobSettingBuilder.add_knob_id(knobIdOffset);
    knobSettingBuilder.add_value_type(hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue);
    knobSettingBuilder.add_value(knobValue.Union());
    auto knobSetting = knobSettingBuilder.Finish();

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::KnobSetting>> knobsVector;
    knobsVector.push_back(knobSetting);
    auto knobs = configBuilder.CreateVector(knobsVector);

    auto engineConfig
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(configBuilder, 1, knobs);
    configBuilder.Finish(engineConfig);

    auto buffer = configBuilder.Release();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper configWrapper(buffer.data(),
                                                                                    buffer.size());

    HipdnnMiopenSettings executionSettings;
    EXPECT_THROW(
        _planBuilder.initializeExecutionSettings(*_handle, graph, configWrapper, executionSettings),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestGpuMiopenConvPlanBuilder,
       InitializeExecutionSettingsThrowsOnWorkspaceSizeLimitAboveRange)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto range = _planBuilder.getWorkspaceSizeRange(*_handle, graph);

    if(range.max >= std::numeric_limits<int64_t>::max())
    {
        GTEST_SKIP() << "Skipping above-range test when max >= INT64_MAX";
    }

    flatbuffers::FlatBufferBuilder configBuilder;
    auto knobIdOffset = configBuilder.CreateString("global.workspace_size_limit");
    auto knobValue = hipdnn_flatbuffers_sdk::data_objects::CreateIntValue(
        configBuilder, static_cast<int64_t>(range.max) + 1);
    hipdnn_flatbuffers_sdk::data_objects::KnobSettingBuilder knobSettingBuilder(configBuilder);
    knobSettingBuilder.add_knob_id(knobIdOffset);
    knobSettingBuilder.add_value_type(hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue);
    knobSettingBuilder.add_value(knobValue.Union());
    auto knobSetting = knobSettingBuilder.Finish();

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::KnobSetting>> knobsVector;
    knobsVector.push_back(knobSetting);
    auto knobs = configBuilder.CreateVector(knobsVector);

    auto engineConfig
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(configBuilder, 1, knobs);
    configBuilder.Finish(engineConfig);

    auto buffer = configBuilder.Release();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper configWrapper(buffer.data(),
                                                                                    buffer.size());

    HipdnnMiopenSettings executionSettings;
    EXPECT_THROW(
        _planBuilder.initializeExecutionSettings(*_handle, graph, configWrapper, executionSettings),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestGpuMiopenConvPlanBuilder, InitializeExecutionSettingsSetsWorkspaceSizeLimit)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto range = _planBuilder.getWorkspaceSizeRange(*_handle, graph);
    const auto testWorkspaceSize = range.min + ((range.max - range.min) / 2);

    flatbuffers::FlatBufferBuilder configBuilder;
    auto knobIdOffset = configBuilder.CreateString("global.workspace_size_limit");
    auto knobValue = hipdnn_flatbuffers_sdk::data_objects::CreateIntValue(
        configBuilder, static_cast<int64_t>(testWorkspaceSize));
    hipdnn_flatbuffers_sdk::data_objects::KnobSettingBuilder knobSettingBuilder(configBuilder);
    knobSettingBuilder.add_knob_id(knobIdOffset);
    knobSettingBuilder.add_value_type(hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue);
    knobSettingBuilder.add_value(knobValue.Union());
    auto knobSetting = knobSettingBuilder.Finish();

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::KnobSetting>> knobsVector;
    knobsVector.push_back(knobSetting);
    auto knobs = configBuilder.CreateVector(knobsVector);

    auto engineConfig
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(configBuilder, 1, knobs);
    configBuilder.Finish(engineConfig);

    auto buffer = configBuilder.Release();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper configWrapper(buffer.data(),
                                                                                    buffer.size());

    HipdnnMiopenSettings executionSettings;
    EXPECT_NO_THROW(_planBuilder.initializeExecutionSettings(
        *_handle, graph, configWrapper, executionSettings));

    MockEngineConfig mockEngineConfig;
    HipdnnMiopenContext ctx;
    ctx.setExecutionSettings(executionSettings);

    EXPECT_NO_THROW(_planBuilder.buildPlan(*_handle, graph, mockEngineConfig, ctx));
    auto workspaceLimit = executionSettings.workspaceSizeLimit();
    ASSERT_TRUE(workspaceLimit.has_value());
    if(workspaceLimit.has_value())
    {
        EXPECT_EQ(*workspaceLimit, testWorkspaceSize);
    }
}

TEST_F(TestGpuMiopenConvPlanBuilder,
       InitializeExecutionSettingsDefaultsWorkspaceSizeLimitWhenNoKnob)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    flatbuffers::FlatBufferBuilder configBuilder;
    auto engineConfig
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(configBuilder, 1, 0);
    configBuilder.Finish(engineConfig);

    auto buffer = configBuilder.Release();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper configWrapper(buffer.data(),
                                                                                    buffer.size());

    HipdnnMiopenSettings executionSettings;
    EXPECT_NO_THROW(_planBuilder.initializeExecutionSettings(
        *_handle, graph, configWrapper, executionSettings));

    MockEngineConfig mockEngineConfig;
    HipdnnMiopenContext ctx;
    ctx.setExecutionSettings(executionSettings);

    EXPECT_NO_THROW(_planBuilder.buildPlan(*_handle, graph, mockEngineConfig, ctx));
    EXPECT_FALSE(executionSettings.workspaceSizeLimit().has_value());
}

TEST_F(TestGpuMiopenConvPlanBuilder, InitializeExecutionSettingsSetsDefaultWorkspaceSize)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    flatbuffers::FlatBufferBuilder configBuilder;
    auto engineConfig
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(configBuilder, 1, 0);
    configBuilder.Finish(engineConfig);

    auto buffer = configBuilder.Release();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper configWrapper(buffer.data(),
                                                                                    buffer.size());

    HipdnnMiopenSettings settings;
    _planBuilder.initializeExecutionSettings(*_handle, graph, configWrapper, settings);

    ASSERT_TRUE(settings.defaultWorkspaceSize().has_value());

    HipdnnMiopenSettings freshSettings;
    auto expected = _planBuilder.getMaxWorkspaceSize(*_handle, graph, freshSettings);

    // NOLINTNEXTLINE(bugprone-unchecked-optional-access,-warnings-as-errors)
    EXPECT_EQ(settings.defaultWorkspaceSize().value(), expected);
}

TEST_F(TestGpuMiopenConvPlanBuilder, InitializeExecutionSettingsSetDefaultWhenLimitIsSet)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto range = _planBuilder.getWorkspaceSizeRange(*_handle, graph);

    flatbuffers::FlatBufferBuilder configBuilder;
    auto knobIdOffset = configBuilder.CreateString("global.workspace_size_limit");
    auto knobValue = hipdnn_flatbuffers_sdk::data_objects::CreateIntValue(
        configBuilder, static_cast<int64_t>(range.max));
    hipdnn_flatbuffers_sdk::data_objects::KnobSettingBuilder knobSettingBuilder(configBuilder);
    knobSettingBuilder.add_knob_id(knobIdOffset);
    knobSettingBuilder.add_value_type(hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue);
    knobSettingBuilder.add_value(knobValue.Union());
    auto knobSetting = knobSettingBuilder.Finish();

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::KnobSetting>> knobsVector;
    knobsVector.push_back(knobSetting);
    auto knobs = configBuilder.CreateVector(knobsVector);

    auto engineConfig
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(configBuilder, 1, knobs);
    configBuilder.Finish(engineConfig);

    auto buffer = configBuilder.Release();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper configWrapper(buffer.data(),
                                                                                    buffer.size());

    HipdnnMiopenSettings settings;
    _planBuilder.initializeExecutionSettings(*_handle, graph, configWrapper, settings);

    EXPECT_TRUE(settings.defaultWorkspaceSize().has_value());
}

TEST_F(TestGpuMiopenConvPlanBuilder, GetMaxWorkspaceSizeReturnsCachedDefault)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const size_t cachedValue = 42;
    HipdnnMiopenSettings settings;
    settings.setDefaultWorkspaceSize(cachedValue);

    auto result = _planBuilder.getMaxWorkspaceSize(*_handle, graph, settings);
    EXPECT_EQ(result, cachedValue);
}

TEST(TestHipdnnMiopenSettings, SelectedWorkspaceSizeReturnsZeroWhenNothingSet)
{
    HipdnnMiopenSettings settings;
    EXPECT_EQ(settings.selectedWorkspaceSize(), 0u);
}

TEST(TestHipdnnMiopenSettings, SelectedWorkspaceSizeReturnsDefaultWhenOnlyDefaultSet)
{
    HipdnnMiopenSettings settings;
    settings.setDefaultWorkspaceSize(1024);
    EXPECT_EQ(settings.selectedWorkspaceSize(), 1024u);
}

TEST(TestHipdnnMiopenSettings, SelectedWorkspaceSizeReturnsLimitWhenOnlyLimitSet)
{
    HipdnnMiopenSettings settings;
    settings.setWorkspaceSizeLimit(512);
    EXPECT_EQ(settings.selectedWorkspaceSize(), 512u);
}

TEST(TestHipdnnMiopenSettings, SelectedWorkspaceSizeReturnsLimitWhenBothSet)
{
    HipdnnMiopenSettings settings;
    settings.setDefaultWorkspaceSize(1024);
    settings.setWorkspaceSizeLimit(512);
    EXPECT_EQ(settings.selectedWorkspaceSize(), 512u);
}

TEST_F(TestGpuMiopenConvPlanBuilder, InitializeExecutionSettingsSetsDefaultWorkspaceSizeBwd)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    flatbuffers::FlatBufferBuilder configBuilder;
    auto engineConfig
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(configBuilder, 1, 0);
    configBuilder.Finish(engineConfig);

    auto buffer = configBuilder.Release();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper configWrapper(buffer.data(),
                                                                                    buffer.size());

    HipdnnMiopenSettings settings;
    _planBuilder.initializeExecutionSettings(*_handle, graph, configWrapper, settings);

    ASSERT_TRUE(settings.defaultWorkspaceSize().has_value());

    HipdnnMiopenSettings freshSettings;
    auto expected = _planBuilder.getMaxWorkspaceSize(*_handle, graph, freshSettings);

    // NOLINTNEXTLINE(bugprone-unchecked-optional-access,-warnings-as-errors)
    EXPECT_EQ(settings.defaultWorkspaceSize().value(), expected);
}

TEST_F(TestGpuMiopenConvPlanBuilder, InitializeExecutionSettingsSetsDefaultWorkspaceSizeWrw)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvWrwGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    flatbuffers::FlatBufferBuilder configBuilder;
    auto engineConfig
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(configBuilder, 1, 0);
    configBuilder.Finish(engineConfig);

    auto buffer = configBuilder.Release();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper configWrapper(buffer.data(),
                                                                                    buffer.size());

    HipdnnMiopenSettings settings;
    _planBuilder.initializeExecutionSettings(*_handle, graph, configWrapper, settings);

    ASSERT_TRUE(settings.defaultWorkspaceSize().has_value());

    HipdnnMiopenSettings freshSettings;
    auto expected = _planBuilder.getMaxWorkspaceSize(*_handle, graph, freshSettings);

    // NOLINTNEXTLINE(bugprone-unchecked-optional-access,-warnings-as-errors)
    EXPECT_EQ(settings.defaultWorkspaceSize().value(), expected);
}

TEST_F(TestGpuMiopenConvPlanBuilder, GetMaxWorkspaceSizeReturnsCachedDefaultBwd)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const size_t cachedValue = 42;
    HipdnnMiopenSettings settings;
    settings.setDefaultWorkspaceSize(cachedValue);

    auto result = _planBuilder.getMaxWorkspaceSize(*_handle, graph, settings);
    EXPECT_EQ(result, cachedValue);
}

TEST_F(TestGpuMiopenConvPlanBuilder, GetMaxWorkspaceSizeReturnsCachedDefaultWrw)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvWrwGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const size_t cachedValue = 42;
    HipdnnMiopenSettings settings;
    settings.setDefaultWorkspaceSize(cachedValue);

    auto result = _planBuilder.getMaxWorkspaceSize(*_handle, graph, settings);
    EXPECT_EQ(result, cachedValue);
}
