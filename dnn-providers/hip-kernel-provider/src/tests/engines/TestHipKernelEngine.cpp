// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <memory>

#include <hipdnn_data_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/EngineDetailsWrapper.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "engines/HipKernelEngine.hpp"
#include "mocks/MockCompilablePlan.hpp"
#include "mocks/MockPlan.hpp"
#include "mocks/MockPlanBuilder.hpp"

using namespace hip_kernel_provider;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::flatbuffer_utilities;

TEST(TestHipKernelEngine, ConstructorAndId)
{
    HipKernelEngine engine(42);
    EXPECT_EQ(engine.id(), 42);
}

// ============================================================================
// Workspace Size
// ============================================================================

TEST(TestHipKernelEngine, WorkspaceSizeReturnsZeroIfNoPlanBuilders)
{
    HipKernelEngine engine(1);

    HipKernelHandle dummyHandle;
    MockGraph mockGraph;
    MockEngineConfig mockConfig;

    EXPECT_EQ(engine.getMaxWorkspaceSize(dummyHandle, mockGraph, mockConfig), 0u);
}

TEST(TestHipKernelEngine, WorkspaceSizeReturnsPlanBuilderWorkspace)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder,
                initializeExecutionSettings(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(1);
    EXPECT_CALL(*mockPlanBuilder, getMaxWorkspaceSize(::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Return(1337u));

    HipKernelEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    HipKernelHandle dummyHandle;
    MockGraph mockGraph;
    MockEngineConfig mockConfig;

    EXPECT_EQ(engine.getMaxWorkspaceSize(dummyHandle, mockGraph, mockConfig), 1337u);
}

TEST(TestHipKernelEngine, WorkspaceSizeReturnsMaxPlanBuilderWorkspace)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder,
                initializeExecutionSettings(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(1);
    EXPECT_CALL(*mockPlanBuilder, getMaxWorkspaceSize(::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Return(1337u));

    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder2,
                initializeExecutionSettings(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(1);
    EXPECT_CALL(*mockPlanBuilder2, getMaxWorkspaceSize(::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Return(45000u));

    HipKernelEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    HipKernelHandle dummyHandle;
    MockGraph mockGraph;
    MockEngineConfig mockConfig;

    EXPECT_EQ(engine.getMaxWorkspaceSize(dummyHandle, mockGraph, mockConfig), 45000u);
}

TEST(TestHipKernelEngine, WorkspaceSizeReturnsZeroIfNoPlanBuilderApplicable)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(false));

    HipKernelEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    HipKernelHandle dummyHandle;
    MockGraph mockGraph;
    MockEngineConfig mockConfig;

    EXPECT_EQ(engine.getMaxWorkspaceSize(dummyHandle, mockGraph, mockConfig), 0u);
}

// ============================================================================
// IsApplicable
// ============================================================================

TEST(TestHipKernelEngine, IsApplicableReturnsTrueIfAnyPlanBuilderApplicable)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();

    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));

    HipKernelEngine engine(0);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    MockGraph mockGraph;
    HipKernelHandle dummyHandle;
    EXPECT_TRUE(engine.isApplicable(dummyHandle, mockGraph));
}

TEST(TestHipKernelEngine, IsApplicableReturnsAfterTheFirstApplicablePlanBuilder)
{
    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    EXPECT_CALL(*mockPlanBuilder1, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_, ::testing::_)).Times(0);

    HipKernelEngine engine(0);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    MockGraph mockGraph;
    HipKernelHandle dummyHandle;
    EXPECT_TRUE(engine.isApplicable(dummyHandle, mockGraph));
}

TEST(TestHipKernelEngine, IsApplicableReturnsFalseIfNoPlanBuilders)
{
    HipKernelEngine engine(0);

    MockGraph mockGraph;
    HipKernelHandle dummyHandle;
    EXPECT_FALSE(engine.isApplicable(dummyHandle, mockGraph));
}

TEST(TestHipKernelEngine, IsApplicableReturnsFalseIfNoPlanBuilderApplicable)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(false));

    HipKernelEngine engine(0);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    MockGraph mockGraph;
    HipKernelHandle dummyHandle;
    EXPECT_FALSE(engine.isApplicable(dummyHandle, mockGraph));
}

// ============================================================================
// GetDetails
// ============================================================================

TEST(TestHipKernelEngine, GetDetailsReturnsSerializedEngineDetails)
{
    HipKernelEngine engine(1);
    HipKernelHandle dummyHandle;
    MockGraph mockGraph;

    hipdnnPluginConstData_t result;
    engine.getDetails(dummyHandle, mockGraph, result);

    EngineDetailsWrapper engineDetails(result.ptr, result.size);
    EXPECT_EQ(engineDetails.engineId(), 1);
}

TEST(TestHipKernelEngine, GetDetailsContainsBenchmarkingKnob)
{
    HipKernelEngine engine(1);
    HipKernelHandle dummyHandle;
    MockGraph mockGraph;

    hipdnnPluginConstData_t result;
    engine.getDetails(dummyHandle, mockGraph, result);

    EngineDetailsWrapper engineDetails(result.ptr, result.size);
    ASSERT_EQ(engineDetails.knobCount(), 1u);

    const auto& knob = engineDetails.getKnobByName("global.benchmarking");
    EXPECT_EQ(knob.knobId(), "global.benchmarking");
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

TEST(TestHipKernelEngine, GetDetailsOnlyUsesFirstPlanBuilderCustomKnobs)
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

    // Second plan builder should NOT be queried (we break after first non-empty custom knobs)
    EXPECT_CALL(*mockPlanBuilder2, getCustomKnobs(::testing::_, ::testing::_)).Times(0);

    HipKernelEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    HipKernelHandle dummyHandle;
    MockGraph mockGraph;

    hipdnnPluginConstData_t result;
    engine.getDetails(dummyHandle, mockGraph, result);

    EngineDetailsWrapper engineDetails(result.ptr, result.size);

    // Should have 2 knobs: benchmarking (always present) + custom.knob1 (from first builder)
    ASSERT_EQ(engineDetails.knobCount(), 2u);

    const auto& benchmarkingKnob = engineDetails.getKnobByName("global.benchmarking");
    EXPECT_EQ(benchmarkingKnob.knobId(), "global.benchmarking");

    const auto& customKnob = engineDetails.getKnobByName("custom.knob1");
    EXPECT_EQ(customKnob.knobId(), "custom.knob1");
    EXPECT_EQ(customKnob.description(), "First custom knob");

    // Second builder's knob should NOT be present
    EXPECT_THROW(engineDetails.getKnobByName("custom.knob2"), std::out_of_range);
}

// ============================================================================
// InitializeExecutionContext
// ============================================================================

TEST(TestHipKernelEngine, InitializeExecutionContextInvokesFirstApplicablePlanBuilder)
{
    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    EXPECT_CALL(*mockPlanBuilder1, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder1,
                buildPlan(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(1);
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_, ::testing::_)).Times(0);
    EXPECT_CALL(*mockPlanBuilder2,
                buildPlan(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(0);

    HipKernelEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    MockGraph mockGraph;
    HipKernelHandle dummyHandle;
    HipKernelContext ctx;
    MockEngineConfig mockConfig;
    EXPECT_CALL(mockConfig, isValid()).WillRepeatedly(::testing::Return(false));

    engine.initializeExecutionContext(dummyHandle, mockGraph, mockConfig, ctx);
}

TEST(TestHipKernelEngine, InitializeExecutionContextSetsBenchmarkingEnabled)
{
    HipKernelEngine engine(1);
    MockGraph mockGraph;
    HipKernelHandle dummyHandle;
    HipKernelContext ctx;

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
    EngineConfigWrapper configWrapper(buffer.data(), buffer.size());

    engine.initializeExecutionContext(dummyHandle, mockGraph, configWrapper, ctx);

    EXPECT_TRUE(ctx.executionSettings().isBenchmarkingEnabled());
}

TEST(TestHipKernelEngine, InitializeExecutionContextSetsBenchmarkingDisabled)
{
    HipKernelEngine engine(1);
    MockGraph mockGraph;
    HipKernelHandle dummyHandle;
    HipKernelContext ctx;

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
    EngineConfigWrapper configWrapper(buffer.data(), buffer.size());

    engine.initializeExecutionContext(dummyHandle, mockGraph, configWrapper, ctx);

    EXPECT_FALSE(ctx.executionSettings().isBenchmarkingEnabled());
}

TEST(TestHipKernelEngine, InitializeExecutionContextDefaultsBenchmarkingDisabledWhenConfigInvalid)
{
    HipKernelEngine engine(1);
    MockGraph mockGraph;
    HipKernelHandle dummyHandle;
    HipKernelContext ctx;
    MockEngineConfig mockConfig;

    EXPECT_CALL(mockConfig, isValid()).WillRepeatedly(::testing::Return(false));

    engine.initializeExecutionContext(dummyHandle, mockGraph, mockConfig, ctx);

    EXPECT_FALSE(ctx.executionSettings().isBenchmarkingEnabled());
}

TEST(TestHipKernelEngine, InitializeExecutionContextDefaultsBenchmarkingDisabledWhenNoKnobs)
{
    HipKernelEngine engine(1);
    MockGraph mockGraph;
    HipKernelHandle dummyHandle;
    HipKernelContext ctx;

    flatbuffers::FlatBufferBuilder builder;
    auto engineConfig = hipdnn_data_sdk::data_objects::CreateEngineConfig(builder, 1, 0);
    builder.Finish(engineConfig);

    auto buffer = builder.Release();
    EngineConfigWrapper configWrapper(buffer.data(), buffer.size());

    engine.initializeExecutionContext(dummyHandle, mockGraph, configWrapper, ctx);

    EXPECT_FALSE(ctx.executionSettings().isBenchmarkingEnabled());
}

TEST(TestHipKernelEngine, InitializeExecutionContextBenchmarkingRemainsDisabledOnInvalidKnobType)
{
    HipKernelEngine engine(1);
    MockGraph mockGraph;
    HipKernelHandle dummyHandle;
    HipKernelContext ctx;

    flatbuffers::FlatBufferBuilder builder;
    auto knobIdOffset = builder.CreateString("global.benchmarking");
    auto stringValueOffset = builder.CreateString("invalid_value");
    auto knobValue = hipdnn_data_sdk::data_objects::CreateStringValue(builder, stringValueOffset);
    hipdnn_data_sdk::data_objects::KnobSettingBuilder knobSettingBuilder(builder);
    knobSettingBuilder.add_knob_id(knobIdOffset);
    knobSettingBuilder.add_value_type(hipdnn_data_sdk::data_objects::KnobValue::StringValue);
    knobSettingBuilder.add_value(knobValue.Union());
    auto knobSetting = knobSettingBuilder.Finish();

    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::KnobSetting>> knobsVector;
    knobsVector.push_back(knobSetting);
    auto knobs = builder.CreateVector(knobsVector);

    auto engineConfig = hipdnn_data_sdk::data_objects::CreateEngineConfig(builder, 1, knobs);
    builder.Finish(engineConfig);

    auto buffer = builder.Release();
    EngineConfigWrapper configWrapper(buffer.data(), buffer.size());

    engine.initializeExecutionContext(dummyHandle, mockGraph, configWrapper, ctx);

    EXPECT_FALSE(ctx.executionSettings().isBenchmarkingEnabled());
}

TEST(TestHipKernelEngine, InitializeExecutionContextSkipsNonApplicableBuilders)
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

    HipKernelEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    MockGraph mockGraph;
    HipKernelHandle dummyHandle;
    HipKernelContext ctx;
    MockEngineConfig mockConfig;
    EXPECT_CALL(mockConfig, isValid()).WillRepeatedly(::testing::Return(false));

    engine.initializeExecutionContext(dummyHandle, mockGraph, mockConfig, ctx);
}

TEST(TestHipKernelEngine, InitializeExecutionContextDoesNotCallBuildPlanIfNoApplicableBuilders)
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

    HipKernelEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    MockGraph mockGraph;
    HipKernelHandle dummyHandle;
    HipKernelContext ctx;
    MockEngineConfig mockConfig;
    EXPECT_CALL(mockConfig, isValid()).WillRepeatedly(::testing::Return(false));

    engine.initializeExecutionContext(dummyHandle, mockGraph, mockConfig, ctx);
}

// ============================================================================
// ICompilablePlan integration
// ============================================================================

TEST(TestHipKernelEngine, InitializeExecutionContextDoesNotCompileNonCompilablePlan)
{
    auto mockPlan = std::make_unique<MockPlan>();

    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));

    // When buildPlan is called, set a non-compilable plan on the context
    EXPECT_CALL(*mockPlanBuilder, buildPlan(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillOnce([plan = std::move(mockPlan)](
                      const HipKernelHandle&,
                      const hipdnn_data_sdk::flatbuffer_utilities::IGraph&,
                      const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig&,
                      HipKernelContext& ctx) mutable { ctx.setPlan(std::move(plan)); });

    HipKernelEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    MockGraph mockGraph;
    HipKernelHandle dummyHandle;
    HipKernelContext ctx;
    MockEngineConfig mockConfig;
    EXPECT_CALL(mockConfig, isValid()).WillRepeatedly(::testing::Return(false));

    engine.initializeExecutionContext(dummyHandle, mockGraph, mockConfig, ctx);

    // Plan was set but compile should not have been called (not an ICompilablePlan)
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST(GpuTestHipKernelEngine, InitializeExecutionContextCompilesCompilablePlan)
{
    SKIP_IF_NO_DEVICES();

    auto mockCompilablePlan = std::make_unique<MockCompilablePlan>();
    auto* mockCompilablePlanPtr = mockCompilablePlan.get();

    EXPECT_CALL(*mockCompilablePlanPtr, compile(::testing::_)).Times(1);

    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));

    // When buildPlan is called, set the compilable plan on the context
    EXPECT_CALL(*mockPlanBuilder, buildPlan(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillOnce([plan = std::move(mockCompilablePlan)](
                      const HipKernelHandle&,
                      const hipdnn_data_sdk::flatbuffer_utilities::IGraph&,
                      const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig&,
                      HipKernelContext& ctx) mutable { ctx.setPlan(std::move(plan)); });

    HipKernelEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    MockGraph mockGraph;
    HipKernelHandle dummyHandle;
    HipKernelContext ctx;
    MockEngineConfig mockConfig;
    EXPECT_CALL(mockConfig, isValid()).WillRepeatedly(::testing::Return(false));

    engine.initializeExecutionContext(dummyHandle, mockGraph, mockConfig, ctx);

    // Plan should be set and compile() should have been called (verified by EXPECT_CALL)
    EXPECT_TRUE(ctx.hasValidPlan());
}
