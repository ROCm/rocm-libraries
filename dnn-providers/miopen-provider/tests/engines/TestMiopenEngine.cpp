// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <memory>
#include <set>

#include <hipdnn_data_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/EngineDetailsWrapper.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>

#include "engines/MiopenEngine.hpp"
#include "mocks/MockHipdnnEnginePluginExecutionContext.hpp"
#include "mocks/MockPlanBuilder.hpp"
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>

using namespace miopen_legacy_plugin;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_plugin_sdk;

TEST(TestMiopenEngine, ConstructorAndId)
{
    MiopenEngine engine(42);
    EXPECT_EQ(engine.id(), 42);
}

TEST(TestMiopenEngine, WorkspaceSizeReturnsZeroIfNoPlanBuilders)
{
    MiopenEngine engine(1);

    MockGraph mockGraph;

    HipdnnEnginePluginHandle dummyHandle;
    EXPECT_EQ(engine.getMaxWorkspaceSize(dummyHandle, mockGraph), 0u);
}

TEST(TestMiopenEngine, WorkspaceSizeReturnsPlanBuilderWorkspace)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder, getMaxWorkspaceSize(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(1337u));

    MiopenEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    MockGraph mockGraph;

    HipdnnEnginePluginHandle dummyHandle;
    EXPECT_EQ(engine.getMaxWorkspaceSize(dummyHandle, mockGraph), 1337u);
}

TEST(TestMiopenEngine, WorkspaceSizeReturnsMaxPlanBuilderWorkspace)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder, getMaxWorkspaceSize(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(1337u));
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder2, getMaxWorkspaceSize(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(45000u));

    MiopenEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    MockGraph mockGraph;

    HipdnnEnginePluginHandle dummyHandle;
    EXPECT_EQ(engine.getMaxWorkspaceSize(dummyHandle, mockGraph), 45000u);
}

TEST(TestMiopenEngine, WorkspaceSizeReturnsZeroIfNoPlanBuilderApplicable)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(false));

    MiopenEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    MockGraph mockGraph;

    HipdnnEnginePluginHandle dummyHandle;
    EXPECT_EQ(engine.getMaxWorkspaceSize(dummyHandle, mockGraph), 0u);
}

TEST(TestMiopenEngine, IsApplicableReturnsTrueIfAnyPlanBuilderApplicable)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();

    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));

    MiopenEngine engine(0);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    MockGraph mockGraph;
    auto graphBuilder = hipdnn_test_sdk::utilities::createEmptyValidGraph();

    HipdnnEnginePluginHandle dummyHandle;
    EXPECT_TRUE(engine.isApplicable(dummyHandle, mockGraph));
}

TEST(TestMiopenEngine, IsApplicableReturnsAfterTheFirstApplicablePlanBuilder)
{
    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    EXPECT_CALL(*mockPlanBuilder1, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_, ::testing::_)).Times(0);

    MiopenEngine engine(0);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    MockGraph mockGraph;
    auto graphBuilder = hipdnn_test_sdk::utilities::createEmptyValidGraph();

    HipdnnEnginePluginHandle dummyHandle;
    EXPECT_TRUE(engine.isApplicable(dummyHandle, mockGraph));
}

TEST(TestMiopenEngine, IsApplicableReturnsFalseIfNoPlanBuilders)
{
    MiopenEngine engine(0);

    MockGraph mockGraph;
    auto graphBuilder = hipdnn_test_sdk::utilities::createEmptyValidGraph();

    HipdnnEnginePluginHandle dummyHandle;
    EXPECT_FALSE(engine.isApplicable(dummyHandle, mockGraph));
}

TEST(TestMiopenEngine, IsApplicableReturnsFalseIfNoPlanBuilderApplicable)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(false));

    MiopenEngine engine(0);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    MockGraph mockGraph;
    auto graphBuilder = hipdnn_test_sdk::utilities::createEmptyValidGraph();

    HipdnnEnginePluginHandle dummyHandle;
    EXPECT_FALSE(engine.isApplicable(dummyHandle, mockGraph));
}

TEST(TestMiopenEngine, GetDetailsReturnsSerializedEngineDetails)
{
    MiopenEngine engine(1);
    HipdnnEnginePluginHandle dummyHandle;
    MockGraph mockGraph;

    hipdnnPluginConstData_t result;
    engine.getDetails(dummyHandle, mockGraph, result);

    hipdnn_plugin_sdk::EngineDetailsWrapper engineDetails(result.ptr, result.size);
    EXPECT_EQ(engineDetails.engineId(), 1);
}

TEST(TestMiopenEngine, InitializeExecutionContextInvokesFirstApplicablePlanBuilder)
{
    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    // Only the first plan builder is applicable
    EXPECT_CALL(*mockPlanBuilder1, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder1, buildPlan(::testing::_, ::testing::_, ::testing::_)).Times(1);
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_, ::testing::_)).Times(0);
    EXPECT_CALL(*mockPlanBuilder2, buildPlan(::testing::_, ::testing::_, ::testing::_)).Times(0);

    MiopenEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    MockGraph mockGraph;
    HipdnnEnginePluginHandle dummyHandle;
    MockHipdnnEnginePluginExecutionContext ctx;
    MockEngineConfig mockConfig;
    EXPECT_CALL(mockConfig, isValid()).WillRepeatedly(::testing::Return(false));

    engine.initializeExecutionContext(dummyHandle, mockGraph, mockConfig, ctx);
}

TEST(TestMiopenEngine, GetDetailsIncludesBenchmarkingKnob)
{
    MiopenEngine engine(1);
    HipdnnEnginePluginHandle dummyHandle;
    MockGraph mockGraph;

    hipdnnPluginConstData_t result;
    engine.getDetails(dummyHandle, mockGraph, result);

    hipdnn_plugin_sdk::EngineDetailsWrapper engineDetails(result.ptr, result.size);

    bool foundBenchmarkingKnob = false;
    const auto& knobWrappers = engineDetails.knobWrappers();
    for(const auto& knobWrapper : knobWrappers)
    {
        const auto& knob = knobWrapper->getKnob();
        if(std::string(knob.knob_id()->c_str()) == "global.benchmarking")
        {
            foundBenchmarkingKnob = true;
            EXPECT_EQ(knob.default_value_type(), hipdnn_data_sdk::data_objects::KnobValue::IntValue);
            break;
        }
    }
    EXPECT_TRUE(foundBenchmarkingKnob);
}

TEST(TestMiopenEngine, InitializeExecutionContextSetsBenchmarkingEnabled)
{
    MiopenEngine engine(1);
    MockGraph mockGraph;
    HipdnnEnginePluginHandle dummyHandle;
    MockHipdnnEnginePluginExecutionContext ctx;

    flatbuffers::FlatBufferBuilder builder;
    auto knobIdOffset = builder.CreateString("global.benchmarking");
    auto knobValue = hipdnn_data_sdk::data_objects::CreateIntValue(builder, 1);
    hipdnn_data_sdk::data_objects::KnobSettingBuilder knobSettingBuilder(builder);
    knobSettingBuilder.add_knob_id(knobIdOffset);
    knobSettingBuilder.add_value_type(hipdnn_data_sdk::data_objects::KnobValue::IntValue);
    knobSettingBuilder.add_value(knobValue.Union());
    auto knobSetting = knobSettingBuilder.Finish();

    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::KnobSetting>> knobsVector;
    knobsVector.push_back(knobSetting);
    auto knobs = builder.CreateVector(knobsVector);

    auto engineConfig = hipdnn_data_sdk::data_objects::CreateEngineConfig(builder, 1, knobs);
    builder.Finish(engineConfig);

    auto buffer = builder.Release();
    hipdnn_plugin_sdk::EngineConfigWrapper configWrapper(buffer.data(), buffer.size());

    engine.initializeExecutionContext(dummyHandle, mockGraph, configWrapper, ctx);

    EXPECT_TRUE(ctx.benchmarkingEnabled());
}

TEST(TestMiopenEngine, InitializeExecutionContextSetsBenchmarkingDisabled)
{
    MiopenEngine engine(1);
    MockGraph mockGraph;
    HipdnnEnginePluginHandle dummyHandle;
    MockHipdnnEnginePluginExecutionContext ctx;

    flatbuffers::FlatBufferBuilder builder;
    auto knobIdOffset = builder.CreateString("global.benchmarking");
    auto knobValue
        = hipdnn_data_sdk::data_objects::CreateIntValue(builder, static_cast<int64_t>(0));
    hipdnn_data_sdk::data_objects::KnobSettingBuilder knobSettingBuilder(builder);
    knobSettingBuilder.add_knob_id(knobIdOffset);
    knobSettingBuilder.add_value_type(hipdnn_data_sdk::data_objects::KnobValue::IntValue);
    knobSettingBuilder.add_value(knobValue.Union());
    auto knobSetting = knobSettingBuilder.Finish();

    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::KnobSetting>> knobsVector;
    knobsVector.push_back(knobSetting);
    auto knobs = builder.CreateVector(knobsVector);

    auto engineConfig = hipdnn_data_sdk::data_objects::CreateEngineConfig(builder, 1, knobs);
    builder.Finish(engineConfig);

    auto buffer = builder.Release();
    hipdnn_plugin_sdk::EngineConfigWrapper configWrapper(buffer.data(), buffer.size());

    engine.initializeExecutionContext(dummyHandle, mockGraph, configWrapper, ctx);

    EXPECT_FALSE(ctx.benchmarkingEnabled());
}

TEST(TestMiopenEngine, InitializeExecutionContextDefaultsBenchmarkingDisabledWhenConfigInvalid)
{
    MiopenEngine engine(1);
    MockGraph mockGraph;
    HipdnnEnginePluginHandle dummyHandle;
    MockHipdnnEnginePluginExecutionContext ctx;
    MockEngineConfig mockConfig;

    EXPECT_CALL(mockConfig, isValid()).WillRepeatedly(::testing::Return(false));

    engine.initializeExecutionContext(dummyHandle, mockGraph, mockConfig, ctx);

    EXPECT_FALSE(ctx.benchmarkingEnabled());
}

TEST(TestMiopenEngine, InitializeExecutionContextDefaultsBenchmarkingDisabledWhenNoKnobs)
{
    MiopenEngine engine(1);
    MockGraph mockGraph;
    HipdnnEnginePluginHandle dummyHandle;
    MockHipdnnEnginePluginExecutionContext ctx;

    flatbuffers::FlatBufferBuilder builder;
    auto engineConfig = hipdnn_data_sdk::data_objects::CreateEngineConfig(builder, 1, 0);
    builder.Finish(engineConfig);

    auto buffer = builder.Release();
    hipdnn_plugin_sdk::EngineConfigWrapper configWrapper(buffer.data(), buffer.size());

    engine.initializeExecutionContext(dummyHandle, mockGraph, configWrapper, ctx);

    EXPECT_FALSE(ctx.benchmarkingEnabled());
}

TEST(TestMiopenEngine, GetDetailsIncludesWorkspaceSizeLimitKnob)
{
    MiopenEngine engine(1);
    HipdnnEnginePluginHandle dummyHandle;
    MockGraph mockGraph;

    hipdnnPluginConstData_t result;
    engine.getDetails(dummyHandle, mockGraph, result);

    hipdnn_plugin_sdk::EngineDetailsWrapper engineDetails(result.ptr, result.size);

    bool foundWorkspaceSizeLimitKnob = false;
    const auto& knobWrappers = engineDetails.knobWrappers();
    for(const auto& knobWrapper : knobWrappers)
    {
        const auto& knob = knobWrapper->getKnob();
        if(std::string(knob.knob_id()->c_str()) == "global.workspace_size_limit")
        {
            foundWorkspaceSizeLimitKnob = true;
            EXPECT_EQ(knob.default_value_type(), hipdnn_data_sdk::data_objects::KnobValue::IntValue);
            break;
        }
    }
    EXPECT_TRUE(foundWorkspaceSizeLimitKnob);
}

TEST(TestMiopenEngine, InitializeExecutionContextSetsWorkspaceSizeLimit)
{
    constexpr auto WORKSPACE_SIZE_LIMIT = 1024 * 1024;
    MiopenEngine engine(1);
    MockGraph mockGraph;
    HipdnnEnginePluginHandle dummyHandle;
    MockHipdnnEnginePluginExecutionContext ctx;

    flatbuffers::FlatBufferBuilder builder;
    auto knobIdOffset = builder.CreateString("global.workspace_size_limit");
    auto knobValue = hipdnn_data_sdk::data_objects::CreateIntValue(builder, WORKSPACE_SIZE_LIMIT);
    hipdnn_data_sdk::data_objects::KnobSettingBuilder knobSettingBuilder(builder);
    knobSettingBuilder.add_knob_id(knobIdOffset);
    knobSettingBuilder.add_value_type(hipdnn_data_sdk::data_objects::KnobValue::IntValue);
    knobSettingBuilder.add_value(knobValue.Union());
    auto knobSetting = knobSettingBuilder.Finish();

    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::KnobSetting>> knobsVector;
    knobsVector.push_back(knobSetting);
    auto knobs = builder.CreateVector(knobsVector);

    auto engineConfig = hipdnn_data_sdk::data_objects::CreateEngineConfig(builder, 1, knobs);
    builder.Finish(engineConfig);

    auto buffer = builder.Release();
    hipdnn_plugin_sdk::EngineConfigWrapper configWrapper(buffer.data(), buffer.size());

    engine.initializeExecutionContext(dummyHandle, mockGraph, configWrapper, ctx);

    EXPECT_TRUE(ctx.workspaceSizeLimit().has_value());
    EXPECT_EQ(ctx.workspaceSizeLimit().value(), WORKSPACE_SIZE_LIMIT);
}

TEST(TestMiopenEngine, InitializeExecutionContextSetsWorkspaceSizeLimitToZero)
{
    MiopenEngine engine(1);
    MockGraph mockGraph;
    HipdnnEnginePluginHandle dummyHandle;
    MockHipdnnEnginePluginExecutionContext ctx;

    flatbuffers::FlatBufferBuilder builder;
    auto knobIdOffset = builder.CreateString("global.workspace_size_limit");
    auto knobValue = hipdnn_data_sdk::data_objects::CreateIntValue(builder, static_cast<int64_t>(0));
    hipdnn_data_sdk::data_objects::KnobSettingBuilder knobSettingBuilder(builder);
    knobSettingBuilder.add_knob_id(knobIdOffset);
    knobSettingBuilder.add_value_type(hipdnn_data_sdk::data_objects::KnobValue::IntValue);
    knobSettingBuilder.add_value(knobValue.Union());
    auto knobSetting = knobSettingBuilder.Finish();

    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::KnobSetting>> knobsVector;
    knobsVector.push_back(knobSetting);
    auto knobs = builder.CreateVector(knobsVector);

    auto engineConfig = hipdnn_data_sdk::data_objects::CreateEngineConfig(builder, 1, knobs);
    builder.Finish(engineConfig);

    auto buffer = builder.Release();
    hipdnn_plugin_sdk::EngineConfigWrapper configWrapper(buffer.data(), buffer.size());

    engine.initializeExecutionContext(dummyHandle, mockGraph, configWrapper, ctx);

    EXPECT_TRUE(ctx.workspaceSizeLimit().has_value());
    EXPECT_EQ(ctx.workspaceSizeLimit().value(), 0u);
}

TEST(TestMiopenEngine, InitializeExecutionContextDefaultsWorkspaceSizeLimitToUnlimitedWhenConfigInvalid)
{
    MiopenEngine engine(1);
    MockGraph mockGraph;
    HipdnnEnginePluginHandle dummyHandle;
    MockHipdnnEnginePluginExecutionContext ctx;
    MockEngineConfig mockConfig;

    EXPECT_CALL(mockConfig, isValid()).WillRepeatedly(::testing::Return(false));

    engine.initializeExecutionContext(dummyHandle, mockGraph, mockConfig, ctx);

    EXPECT_FALSE(ctx.workspaceSizeLimit().has_value());
}

TEST(TestMiopenEngine, InitializeExecutionContextDefaultsWorkspaceSizeLimitToUnlimitedWhenNoKnobs)
{
    MiopenEngine engine(1);
    MockGraph mockGraph;
    HipdnnEnginePluginHandle dummyHandle;
    MockHipdnnEnginePluginExecutionContext ctx;

    flatbuffers::FlatBufferBuilder builder;
    auto engineConfig = hipdnn_data_sdk::data_objects::CreateEngineConfig(builder, 1, 0);
    builder.Finish(engineConfig);

    auto buffer = builder.Release();
    hipdnn_plugin_sdk::EngineConfigWrapper configWrapper(buffer.data(), buffer.size());

    engine.initializeExecutionContext(dummyHandle, mockGraph, configWrapper, ctx);

    EXPECT_FALSE(ctx.workspaceSizeLimit().has_value());
}

TEST(TestMiopenEngine, GetDetailsUsesWorkspaceSizeRangeFromPlanBuilder)
{
    constexpr size_t MIN_WORKSPACE = 1024;
    constexpr size_t MAX_WORKSPACE = 8192;

    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder, getWorkspaceSizeRange(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(WorkspaceSizeRange{MIN_WORKSPACE, MAX_WORKSPACE}));

    MiopenEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    MockGraph mockGraph;
    HipdnnEnginePluginHandle dummyHandle;

    hipdnnPluginConstData_t result;
    engine.getDetails(dummyHandle, mockGraph, result);

    hipdnn_plugin_sdk::EngineDetailsWrapper engineDetails(result.ptr, result.size);

    bool foundWorkspaceSizeLimitKnob = false;
    const auto& knobWrappers = engineDetails.knobWrappers();
    for(const auto& knobWrapper : knobWrappers)
    {
        const auto& knob = knobWrapper->getKnob();
        if(std::string(knob.knob_id()->c_str()) == "global.workspace_size_limit")
        {
            foundWorkspaceSizeLimitKnob = true;
            EXPECT_EQ(knob.default_value_type(), hipdnn_data_sdk::data_objects::KnobValue::IntValue);
            auto defaultValue
                = knob.default_value_as<hipdnn_data_sdk::data_objects::IntValue>()->value();
            auto constraint = knob.constraint_as<hipdnn_data_sdk::data_objects::IntConstraint>();
            auto minValue = constraint->min_value();
            auto maxValue = constraint->max_value();

            EXPECT_EQ(defaultValue, static_cast<int64_t>(MAX_WORKSPACE));
            EXPECT_EQ(minValue, static_cast<int64_t>(MIN_WORKSPACE));
            EXPECT_EQ(maxValue, static_cast<int64_t>(MAX_WORKSPACE));
            break;
        }
    }
    EXPECT_TRUE(foundWorkspaceSizeLimitKnob);
}

TEST(TestMiopenEngine, InitializeExecutionContextSkipsNonApplicableBuilders)
{
    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    // First plan builder not applicable, second is
    EXPECT_CALL(*mockPlanBuilder1, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(false));
    EXPECT_CALL(*mockPlanBuilder1, buildPlan(::testing::_, ::testing::_, ::testing::_)).Times(0);
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder2, buildPlan(::testing::_, ::testing::_, ::testing::_)).Times(1);

    MiopenEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    MockGraph mockGraph;
    HipdnnEnginePluginHandle dummyHandle;
    MockHipdnnEnginePluginExecutionContext ctx;
    MockEngineConfig mockConfig;
    EXPECT_CALL(mockConfig, isValid()).WillRepeatedly(::testing::Return(false));

    engine.initializeExecutionContext(dummyHandle, mockGraph, mockConfig, ctx);
}

TEST(TestMiopenEngine, InitializeExecutionContextDoesNotCallBuildPlanIfNoApplicableBuilders)
{
    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    EXPECT_CALL(*mockPlanBuilder1, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(false));
    EXPECT_CALL(*mockPlanBuilder1, buildPlan(::testing::_, ::testing::_, ::testing::_)).Times(0);
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(false));
    EXPECT_CALL(*mockPlanBuilder2, buildPlan(::testing::_, ::testing::_, ::testing::_)).Times(0);

    MiopenEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    MockGraph mockGraph;
    HipdnnEnginePluginHandle dummyHandle;
    MockHipdnnEnginePluginExecutionContext ctx;
    MockEngineConfig mockConfig;
    EXPECT_CALL(mockConfig, isValid()).WillRepeatedly(::testing::Return(false));

    engine.initializeExecutionContext(dummyHandle, mockGraph, mockConfig, ctx);
}
