// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <memory>

#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineDetailsWrapper.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "engines/hip_mlops_engine/HipMlopsEngine.hpp"
#include "mocks/MockPlanBuilder.hpp"

using namespace hip_kernel_provider;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

TEST(TestHipMlopsEngine, ConstructorAndId)
{
    const HipMlopsEngine engine(42);
    EXPECT_EQ(engine.id(), 42);
}

// ============================================================================
// Workspace Size
// ============================================================================

TEST(TestHipMlopsEngine, WorkspaceSizeReturnsZeroIfNoPlanBuilders)
{
    const HipMlopsEngine engine(1);

    const Handle dummyHandle;
    const MockGraph mockGraph;
    const MockEngineConfig mockConfig;

    EXPECT_EQ(engine.getMaxWorkspaceSize(dummyHandle, mockGraph, mockConfig), 0u);
}

TEST(TestHipMlopsEngine, WorkspaceSizeReturnsPlanBuilderWorkspace)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder,
                initializeExecutionSettings(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(1);
    EXPECT_CALL(*mockPlanBuilder, getMaxWorkspaceSize(::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Return(1337u));

    HipMlopsEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    const Handle dummyHandle;
    const MockGraph mockGraph;
    const MockEngineConfig mockConfig;

    EXPECT_EQ(engine.getMaxWorkspaceSize(dummyHandle, mockGraph, mockConfig), 1337u);
}

TEST(TestHipMlopsEngine, WorkspaceSizeReturnsMaxPlanBuilderWorkspace)
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

    HipMlopsEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    const Handle dummyHandle;
    const MockGraph mockGraph;
    const MockEngineConfig mockConfig;

    EXPECT_EQ(engine.getMaxWorkspaceSize(dummyHandle, mockGraph, mockConfig), 45000u);
}

TEST(TestHipMlopsEngine, WorkspaceSizeReturnsZeroIfNoPlanBuilderApplicable)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(false));

    HipMlopsEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    const Handle dummyHandle;
    const MockGraph mockGraph;
    const MockEngineConfig mockConfig;

    EXPECT_EQ(engine.getMaxWorkspaceSize(dummyHandle, mockGraph, mockConfig), 0u);
}

// ============================================================================
// IsApplicable
// ============================================================================

TEST(TestHipMlopsEngine, IsApplicableReturnsTrueIfAnyPlanBuilderApplicable)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();

    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));

    HipMlopsEngine engine(0);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    const MockGraph mockGraph;
    Handle dummyHandle;
    EXPECT_TRUE(engine.isApplicable(dummyHandle, mockGraph));
}

TEST(TestHipMlopsEngine, IsApplicableReturnsAfterTheFirstApplicablePlanBuilder)
{
    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    EXPECT_CALL(*mockPlanBuilder1, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_, ::testing::_)).Times(0);

    HipMlopsEngine engine(0);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    const MockGraph mockGraph;
    Handle dummyHandle;
    EXPECT_TRUE(engine.isApplicable(dummyHandle, mockGraph));
}

TEST(TestHipMlopsEngine, IsApplicableReturnsFalseIfNoPlanBuilders)
{
    const HipMlopsEngine engine(0);

    const MockGraph mockGraph;
    Handle dummyHandle;
    EXPECT_FALSE(engine.isApplicable(dummyHandle, mockGraph));
}

TEST(TestHipMlopsEngine, IsApplicableReturnsFalseIfNoPlanBuilderApplicable)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(false));

    HipMlopsEngine engine(0);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    const MockGraph mockGraph;
    Handle dummyHandle;
    EXPECT_FALSE(engine.isApplicable(dummyHandle, mockGraph));
}

// ============================================================================
// GetDetails
// ============================================================================

TEST(TestHipMlopsEngine, GetDetailsReturnsSerializedEngineDetails)
{
    const HipMlopsEngine engine(1);
    Handle dummyHandle;
    const MockGraph mockGraph;

    hipdnnPluginConstData_t result;
    engine.getDetails(dummyHandle, mockGraph, result);

    const EngineDetailsWrapper engineDetails(result.ptr, result.size);
    EXPECT_EQ(engineDetails.engineId(), 1);
}

TEST(TestHipMlopsEngine, GetDetailsOnlyUsesFirstPlanBuilderCustomKnobs)
{
    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    // Set up first plan builder to return a custom knob
    hipdnn_flatbuffers_sdk::data_objects::KnobT knob1;
    knob1.knob_id = "custom.knob1";
    knob1.description = "First custom knob";
    hipdnn_flatbuffers_sdk::data_objects::IntValueT defaultValue1;
    defaultValue1.value = 1;
    knob1.default_value.Set(defaultValue1);

    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> customKnobs1;
    customKnobs1.push_back(knob1);

    EXPECT_CALL(*mockPlanBuilder1, getCustomKnobs(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(customKnobs1));

    // Second plan builder should NOT be queried (we break after first non-empty custom knobs)
    EXPECT_CALL(*mockPlanBuilder2, getCustomKnobs(::testing::_, ::testing::_)).Times(0);

    HipMlopsEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    Handle dummyHandle;
    const MockGraph mockGraph;

    hipdnnPluginConstData_t result;
    engine.getDetails(dummyHandle, mockGraph, result);

    const EngineDetailsWrapper engineDetails(result.ptr, result.size);

    // Should have 1 knob: custom.knob1 (from first builder)
    ASSERT_EQ(engineDetails.knobCount(), 1u);

    const auto& customKnob = engineDetails.getKnobByName("custom.knob1");
    EXPECT_EQ(customKnob.knobId(), "custom.knob1");
    EXPECT_EQ(customKnob.description(), "First custom knob");

    // Second builder's knob should NOT be present
    EXPECT_THROW(engineDetails.getKnobByName("custom.knob2"), std::out_of_range);
}

TEST(TestHipMlopsEngine, GetDetailsSkipsPlanBuildersWithEmptyCustomKnobs)
{
    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    // First plan builder returns no custom knobs: the engine should skip it
    // (the `continue` branch) and use the next builder's knobs.
    EXPECT_CALL(*mockPlanBuilder1, getCustomKnobs(::testing::_, ::testing::_))
        .WillOnce(
            ::testing::Return(std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT>{}));

    hipdnn_flatbuffers_sdk::data_objects::KnobT knob2;
    knob2.knob_id = "custom.knob2";
    knob2.description = "Second custom knob";
    hipdnn_flatbuffers_sdk::data_objects::IntValueT defaultValue2;
    defaultValue2.value = 2;
    knob2.default_value.Set(defaultValue2);

    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> customKnobs2;
    customKnobs2.push_back(knob2);

    EXPECT_CALL(*mockPlanBuilder2, getCustomKnobs(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(customKnobs2));

    HipMlopsEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    Handle dummyHandle;
    const MockGraph mockGraph;

    hipdnnPluginConstData_t result;
    engine.getDetails(dummyHandle, mockGraph, result);

    const EngineDetailsWrapper engineDetails(result.ptr, result.size);

    ASSERT_EQ(engineDetails.knobCount(), 1u);
    const auto& customKnob = engineDetails.getKnobByName("custom.knob2");
    EXPECT_EQ(customKnob.knobId(), "custom.knob2");
    EXPECT_EQ(customKnob.description(), "Second custom knob");
}

TEST(TestHipMlopsEngine, GetDetailsSerializesAllKnobsFromFirstNonEmptyPlanBuilder)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();

    hipdnn_flatbuffers_sdk::data_objects::KnobT knobA;
    knobA.knob_id = "custom.knobA";
    knobA.description = "Knob A";
    hipdnn_flatbuffers_sdk::data_objects::IntValueT defaultValueA;
    defaultValueA.value = 1;
    knobA.default_value.Set(defaultValueA);

    hipdnn_flatbuffers_sdk::data_objects::KnobT knobB;
    knobB.knob_id = "custom.knobB";
    knobB.description = "Knob B";
    hipdnn_flatbuffers_sdk::data_objects::IntValueT defaultValueB;
    defaultValueB.value = 2;
    knobB.default_value.Set(defaultValueB);

    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> customKnobs;
    customKnobs.push_back(knobA);
    customKnobs.push_back(knobB);

    EXPECT_CALL(*mockPlanBuilder, getCustomKnobs(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(customKnobs));

    HipMlopsEngine engine(7);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    Handle dummyHandle;
    const MockGraph mockGraph;

    hipdnnPluginConstData_t result;
    engine.getDetails(dummyHandle, mockGraph, result);

    const EngineDetailsWrapper engineDetails(result.ptr, result.size);

    ASSERT_EQ(engineDetails.knobCount(), 2u);
    EXPECT_EQ(engineDetails.getKnobByName("custom.knobA").description(), "Knob A");
    EXPECT_EQ(engineDetails.getKnobByName("custom.knobB").description(), "Knob B");
}

TEST(TestHipMlopsEngine, WorkspaceSizeIgnoresNonApplicablePlanBuilders)
{
    auto applicableBuilder = std::make_unique<MockPlanBuilder>();
    auto nonApplicableBuilder = std::make_unique<MockPlanBuilder>();

    EXPECT_CALL(*applicableBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*applicableBuilder,
                initializeExecutionSettings(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(1);
    EXPECT_CALL(*applicableBuilder, getMaxWorkspaceSize(::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Return(2048u));

    // The non-applicable builder must not contribute to (or be queried for) the
    // workspace size.
    EXPECT_CALL(*nonApplicableBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(false));
    EXPECT_CALL(*nonApplicableBuilder,
                initializeExecutionSettings(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(0);
    EXPECT_CALL(*nonApplicableBuilder,
                getMaxWorkspaceSize(::testing::_, ::testing::_, ::testing::_))
        .Times(0);

    HipMlopsEngine engine(1);
    engine.addPlanBuilder(std::move(applicableBuilder));
    engine.addPlanBuilder(std::move(nonApplicableBuilder));

    const Handle dummyHandle;
    const MockGraph mockGraph;
    const MockEngineConfig mockConfig;

    EXPECT_EQ(engine.getMaxWorkspaceSize(dummyHandle, mockGraph, mockConfig), 2048u);
}

// ============================================================================
// InitializeExecutionContext
// ============================================================================

TEST(TestHipMlopsEngine, InitializeExecutionContextInvokesFirstApplicablePlanBuilder)
{
    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    EXPECT_CALL(*mockPlanBuilder1, isApplicable(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder1,
                initializeExecutionSettings(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(1);
    EXPECT_CALL(*mockPlanBuilder1,
                buildPlan(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(1);
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_, ::testing::_)).Times(0);
    EXPECT_CALL(*mockPlanBuilder2,
                initializeExecutionSettings(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(0);
    EXPECT_CALL(*mockPlanBuilder2,
                buildPlan(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(0);

    HipMlopsEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    const MockGraph mockGraph;
    const Handle dummyHandle;
    Context ctx;
    const MockEngineConfig mockConfig;
    EXPECT_CALL(mockConfig, isValid()).WillRepeatedly(::testing::Return(false));

    engine.initializeExecutionContext(dummyHandle, mockGraph, mockConfig, ctx);
}

TEST(TestHipMlopsEngine, InitializeExecutionContextSkipsNonApplicableBuilders)
{
    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    // First plan builder not applicable, second is
    EXPECT_CALL(*mockPlanBuilder1, isApplicable(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(false));
    EXPECT_CALL(*mockPlanBuilder1,
                initializeExecutionSettings(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(0);
    EXPECT_CALL(*mockPlanBuilder1,
                buildPlan(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(0);
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder2,
                initializeExecutionSettings(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(1);
    EXPECT_CALL(*mockPlanBuilder2,
                buildPlan(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(1);

    HipMlopsEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    const MockGraph mockGraph;
    const Handle dummyHandle;
    Context ctx;
    const MockEngineConfig mockConfig;
    EXPECT_CALL(mockConfig, isValid()).WillRepeatedly(::testing::Return(false));

    engine.initializeExecutionContext(dummyHandle, mockGraph, mockConfig, ctx);
}

TEST(TestHipMlopsEngine, InitializeExecutionContextDoesNotCallBuildPlanIfNoApplicableBuilders)
{
    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    EXPECT_CALL(*mockPlanBuilder1, isApplicable(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(false));
    EXPECT_CALL(*mockPlanBuilder1,
                initializeExecutionSettings(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(0);
    EXPECT_CALL(*mockPlanBuilder1,
                buildPlan(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(0);
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(false));
    EXPECT_CALL(*mockPlanBuilder2,
                initializeExecutionSettings(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(0);
    EXPECT_CALL(*mockPlanBuilder2,
                buildPlan(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(0);

    HipMlopsEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    const MockGraph mockGraph;
    const Handle dummyHandle;
    Context ctx;
    const MockEngineConfig mockConfig;
    EXPECT_CALL(mockConfig, isValid()).WillRepeatedly(::testing::Return(false));

    engine.initializeExecutionContext(dummyHandle, mockGraph, mockConfig, ctx);
}
