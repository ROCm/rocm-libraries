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

using namespace miopen_plugin;
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
    EXPECT_EQ(engine.getWorkspaceSize(dummyHandle, mockGraph), 0u);
}

TEST(TestMiopenEngine, WorkspaceSizeReturnsPlanBuilderWorkspace)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder, getWorkspaceSize(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(1337u));

    MiopenEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    MockGraph mockGraph;

    HipdnnEnginePluginHandle dummyHandle;
    EXPECT_EQ(engine.getWorkspaceSize(dummyHandle, mockGraph), 1337u);
}

TEST(TestMiopenEngine, WorkspaceSizeReturnsMaxPlanBuilderWorkspace)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder, getWorkspaceSize(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(1337u));
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder2, getWorkspaceSize(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(45000u));

    MiopenEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    MockGraph mockGraph;

    HipdnnEnginePluginHandle dummyHandle;
    EXPECT_EQ(engine.getWorkspaceSize(dummyHandle, mockGraph), 45000u);
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
    EXPECT_EQ(engine.getWorkspaceSize(dummyHandle, mockGraph), 0u);
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

TEST(TestMiopenEngine, GetDetailsContainsBenchmarkingKnob)
{
    MiopenEngine engine(1);
    HipdnnEnginePluginHandle dummyHandle;
    MockGraph mockGraph;

    hipdnnPluginConstData_t result;
    engine.getDetails(dummyHandle, mockGraph, result);

    hipdnn_plugin_sdk::EngineDetailsWrapper engineDetails(result.ptr, result.size);
    ASSERT_EQ(engineDetails.knobCount(), 1u);

    const auto& knob = engineDetails.getKnobByName("global.benchmarking");
    EXPECT_EQ(knob.knobIdStr(), "global.benchmarking");
    EXPECT_EQ(knob.description(), "Enable benchmarking");

    ASSERT_TRUE(knob.hasDefaultValue());
    EXPECT_EQ(knob.defaultValueType(), hipdnn_data_sdk::data_objects::KnobValue::IntValue);
    const auto& defaultValue = knob.defaultValueAs<hipdnn_data_sdk::data_objects::IntValue>();
    EXPECT_EQ(defaultValue.value(), 0);

    ASSERT_TRUE(knob.hasConstraint());
    EXPECT_EQ(knob.constraintType(), hipdnn_data_sdk::data_objects::KnobConstraint::IntConstraint);
    const auto& constraint = knob.constraintAs<hipdnn_data_sdk::data_objects::IntConstraint>();
    EXPECT_EQ(constraint.min_value(), 0);
    EXPECT_EQ(constraint.max_value(), 1);
    EXPECT_EQ(constraint.step(), 1);
}

TEST(TestMiopenEngine, GetDetailsOnlyUsesFirstPlanBuilderCustomKnobs)
{
    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    // Set up first plan builder to return a custom knob
    hipdnn_data_sdk::data_objects::KnobT knob1;
    knob1.knob_id = "custom.knob1";
    knob1.description = "First custom knob";
    hipdnn_data_sdk::data_objects::IntValueT defaultValue1;
    defaultValue1.value = 1;
    knob1.default_value.Set(defaultValue1);

    std::vector<hipdnn_data_sdk::data_objects::KnobT> customKnobs1;
    customKnobs1.push_back(knob1);

    EXPECT_CALL(*mockPlanBuilder1, getCustomKnobs(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(customKnobs1));

    // Set up second plan builder to also return a custom knob (this should be ignored)
    hipdnn_data_sdk::data_objects::KnobT knob2;
    knob2.knob_id = "custom.knob2";
    knob2.description = "Second custom knob";
    hipdnn_data_sdk::data_objects::IntValueT defaultValue2;
    defaultValue2.value = 2;
    knob2.default_value.Set(defaultValue2);

    std::vector<hipdnn_data_sdk::data_objects::KnobT> customKnobs2;
    customKnobs2.push_back(knob2);

    // This should NOT be called because we break after first non-empty custom knobs
    EXPECT_CALL(*mockPlanBuilder2, getCustomKnobs(::testing::_, ::testing::_)).Times(0);

    MiopenEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    HipdnnEnginePluginHandle dummyHandle;
    MockGraph mockGraph;

    hipdnnPluginConstData_t result;
    engine.getDetails(dummyHandle, mockGraph, result);

    hipdnn_plugin_sdk::EngineDetailsWrapper engineDetails(result.ptr, result.size);

    // Should have 2 knobs: benchmarking (always present) + custom.knob1 (from first builder)
    ASSERT_EQ(engineDetails.knobCount(), 2u);

    // Verify benchmarking knob is present
    const auto& benchmarkingKnob = engineDetails.getKnobByName("global.benchmarking");
    EXPECT_EQ(benchmarkingKnob.knobIdStr(), "global.benchmarking");

    // Verify first custom knob is present
    const auto& customKnob1 = engineDetails.getKnobByName("custom.knob1");
    EXPECT_EQ(customKnob1.knobIdStr(), "custom.knob1");
    EXPECT_EQ(customKnob1.description(), "First custom knob");

    // Verify second custom knob is NOT present (would throw if we tried to access it)
    EXPECT_THROW(engineDetails.getKnobByName("custom.knob2"), std::out_of_range);
}

TEST(TestMiopenEngine, InitializeExecutionContextInvokesFirstApplicablePlanBuilder)
{
    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    // Only the first plan builder is applicable
    EXPECT_CALL(*mockPlanBuilder1, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder1,
                buildPlan(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(1);
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_, ::testing::_)).Times(0);
    EXPECT_CALL(*mockPlanBuilder2,
                buildPlan(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(0);

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

TEST(TestMiopenEngine, InitializeExecutionContextSkipsNonApplicableBuilders)
{
    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    // First plan builder not applicable, second is
    EXPECT_CALL(*mockPlanBuilder1, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(false));
    EXPECT_CALL(*mockPlanBuilder1,
                buildPlan(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(0);
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder2,
                buildPlan(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(1);

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
    EXPECT_CALL(*mockPlanBuilder1,
                buildPlan(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(0);
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(false));
    EXPECT_CALL(*mockPlanBuilder2,
                buildPlan(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(0);

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
