// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <memory>

#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineDetailsWrapper.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "engines/hip_kernel_engine/HipKernelEngine.hpp"
#include "mocks/MockPlanBuilder.hpp"

using namespace hip_kernel_provider;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

TEST(TestHipKernelEngine, ConstructorAndId)
{
    const HipKernelEngine engine(42);
    EXPECT_EQ(engine.id(), 42);
}

// ============================================================================
// Workspace Size
// ============================================================================

TEST(TestHipKernelEngine, WorkspaceSizeReturnsZeroIfNoPlanBuilders)
{
    const HipKernelEngine engine(1);

    const HipKernelHandle dummyHandle;
    const MockGraph mockGraph;
    const MockEngineConfig mockConfig;

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

    const HipKernelHandle dummyHandle;
    const MockGraph mockGraph;
    const MockEngineConfig mockConfig;

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

    const HipKernelHandle dummyHandle;
    const MockGraph mockGraph;
    const MockEngineConfig mockConfig;

    EXPECT_EQ(engine.getMaxWorkspaceSize(dummyHandle, mockGraph, mockConfig), 45000u);
}

TEST(TestHipKernelEngine, WorkspaceSizeReturnsZeroIfNoPlanBuilderApplicable)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(false));

    HipKernelEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    const HipKernelHandle dummyHandle;
    const MockGraph mockGraph;
    const MockEngineConfig mockConfig;

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

    const MockGraph mockGraph;
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

    const MockGraph mockGraph;
    HipKernelHandle dummyHandle;
    EXPECT_TRUE(engine.isApplicable(dummyHandle, mockGraph));
}

TEST(TestHipKernelEngine, IsApplicableReturnsFalseIfNoPlanBuilders)
{
    const HipKernelEngine engine(0);

    const MockGraph mockGraph;
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

    const MockGraph mockGraph;
    HipKernelHandle dummyHandle;
    EXPECT_FALSE(engine.isApplicable(dummyHandle, mockGraph));
}

// ============================================================================
// GetDetails
// ============================================================================

TEST(TestHipKernelEngine, GetDetailsReturnsSerializedEngineDetails)
{
    const HipKernelEngine engine(1);
    HipKernelHandle dummyHandle;
    const MockGraph mockGraph;

    hipdnnPluginConstData_t result;
    engine.getDetails(dummyHandle, mockGraph, result);

    const EngineDetailsWrapper engineDetails(result.ptr, result.size);
    EXPECT_EQ(engineDetails.engineId(), 1);
}

TEST(TestHipKernelEngine, GetDetailsOnlyUsesFirstPlanBuilderCustomKnobs)
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

    HipKernelEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    HipKernelHandle dummyHandle;
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

// ============================================================================
// InitializeExecutionContext
// ============================================================================

TEST(TestHipKernelEngine, InitializeExecutionContextInvokesFirstApplicablePlanBuilder)
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

    HipKernelEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    const MockGraph mockGraph;
    const HipKernelHandle dummyHandle;
    HipKernelContext ctx;
    const MockEngineConfig mockConfig;
    EXPECT_CALL(mockConfig, isValid()).WillRepeatedly(::testing::Return(false));

    engine.initializeExecutionContext(dummyHandle, mockGraph, mockConfig, ctx);
}

TEST(TestHipKernelEngine, InitializeExecutionContextSkipsNonApplicableBuilders)
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

    HipKernelEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    const MockGraph mockGraph;
    const HipKernelHandle dummyHandle;
    HipKernelContext ctx;
    const MockEngineConfig mockConfig;
    EXPECT_CALL(mockConfig, isValid()).WillRepeatedly(::testing::Return(false));

    engine.initializeExecutionContext(dummyHandle, mockGraph, mockConfig, ctx);
}

TEST(TestHipKernelEngine, InitializeExecutionContextDoesNotCallBuildPlanIfNoApplicableBuilders)
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

    HipKernelEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    const MockGraph mockGraph;
    const HipKernelHandle dummyHandle;
    HipKernelContext ctx;
    const MockEngineConfig mockConfig;
    EXPECT_CALL(mockConfig, isValid()).WillRepeatedly(::testing::Return(false));

    engine.initializeExecutionContext(dummyHandle, mockGraph, mockConfig, ctx);
}
