// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <memory>
#include <set>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineDetailsWrapper.hpp>
#include <hipdnn_plugin_sdk/BehaviorNote.h>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>

#include "engines/PlanBuilderEngine.hpp"
#include "mocks/MockHipdnnEnginePluginExecutionContext.hpp"
#include "mocks/MockPlanBuilder.hpp"

using namespace hipblaslt_plugin;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_plugin_sdk;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

TEST(TestPlanBuilderEngine, ConstructorAndId)
{
    PlanBuilderEngine const engine(42);
    EXPECT_EQ(engine.id(), 42);
}

TEST(TestPlanBuilderEngine, WorkspaceSizeReturnsZeroIfNoPlanBuilders)
{
    PlanBuilderEngine const engine(1);

    MockGraph const mockGraph;

    HipdnnEnginePluginHandle const dummyHandle;
    EXPECT_EQ(engine.getWorkspaceSize(dummyHandle, mockGraph), 0u);
}

TEST(TestPlanBuilderEngine, WorkspaceSizeReturnsPlanBuilderWorkspace)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder, getWorkspaceSize(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(1337u));

    PlanBuilderEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    MockGraph const mockGraph;

    HipdnnEnginePluginHandle const dummyHandle;
    EXPECT_EQ(engine.getWorkspaceSize(dummyHandle, mockGraph), 1337u);
}

TEST(TestPlanBuilderEngine, WorkspaceSizeReturnsMaxPlanBuilderWorkspace)
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

    PlanBuilderEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    MockGraph const mockGraph;

    HipdnnEnginePluginHandle const dummyHandle;
    EXPECT_EQ(engine.getWorkspaceSize(dummyHandle, mockGraph), 45000u);
}

TEST(TestPlanBuilderEngine, WorkspaceSizeReturnsZeroIfNoPlanBuilderApplicable)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(false));

    PlanBuilderEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    MockGraph const mockGraph;

    HipdnnEnginePluginHandle const dummyHandle;
    EXPECT_EQ(engine.getWorkspaceSize(dummyHandle, mockGraph), 0u);
}

TEST(TestPlanBuilderEngine, IsApplicableReturnsTrueIfAnyPlanBuilderApplicable)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();

    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));

    PlanBuilderEngine engine(0);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    MockGraph const mockGraph;
    auto graphBuilder = hipdnn_test_sdk::utilities::createEmptyValidGraph();

    HipdnnEnginePluginHandle dummyHandle;
    EXPECT_TRUE(engine.isApplicable(dummyHandle, mockGraph));
}

TEST(TestPlanBuilderEngine, IsApplicableReturnsAfterTheFirstApplicablePlanBuilder)
{
    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    EXPECT_CALL(*mockPlanBuilder1, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_, ::testing::_)).Times(0);

    PlanBuilderEngine engine(0);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    MockGraph const mockGraph;
    auto graphBuilder = hipdnn_test_sdk::utilities::createEmptyValidGraph();

    HipdnnEnginePluginHandle dummyHandle;
    EXPECT_TRUE(engine.isApplicable(dummyHandle, mockGraph));
}

TEST(TestPlanBuilderEngine, IsApplicableReturnsFalseIfNoPlanBuilders)
{
    PlanBuilderEngine const engine(0);

    MockGraph const mockGraph;
    auto graphBuilder = hipdnn_test_sdk::utilities::createEmptyValidGraph();

    HipdnnEnginePluginHandle dummyHandle;
    EXPECT_FALSE(engine.isApplicable(dummyHandle, mockGraph));
}

TEST(TestPlanBuilderEngine, IsApplicableReturnsFalseIfNoPlanBuilderApplicable)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(false));

    PlanBuilderEngine engine(0);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    MockGraph const mockGraph;
    auto graphBuilder = hipdnn_test_sdk::utilities::createEmptyValidGraph();

    HipdnnEnginePluginHandle dummyHandle;
    EXPECT_FALSE(engine.isApplicable(dummyHandle, mockGraph));
}

TEST(TestPlanBuilderEngine, GetDetailsReturnsSerializedEngineDetails)
{
    PlanBuilderEngine const engine(hipdnn_data_sdk::utilities::HIPBLASLT_ENGINE_ID);
    HipdnnEnginePluginHandle dummyHandle;

    hipdnnPluginConstData_t result;
    engine.getDetails(dummyHandle, result);

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineDetailsWrapper const engineDetails(
        result.ptr, result.size);
    EXPECT_EQ(engineDetails.engineId(), hipdnn_data_sdk::utilities::HIPBLASLT_ENGINE_ID);
}

TEST(TestPlanBuilderEngine, GetDetailsReportsGraphCaptureSupport)
{
    PlanBuilderEngine const engine(hipdnn_data_sdk::utilities::HIPBLASLT_ENGINE_ID);
    HipdnnEnginePluginHandle dummyHandle;

    hipdnnPluginConstData_t result;
    engine.getDetails(dummyHandle, result);

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineDetailsWrapper const engineDetails(
        result.ptr, result.size);
    EXPECT_THAT(
        engineDetails.behaviorNotes(),
        ::testing::Contains(static_cast<int32_t>(HIPDNN_BEHAVIOR_NOTE_SUPPORTS_GRAPH_CAPTURE)));
}

TEST(TestPlanBuilderEngine, InitializeExecutionContextInvokesFirstApplicablePlanBuilder)
{
    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    // Only the first plan builder is applicable
    EXPECT_CALL(*mockPlanBuilder1, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder1, buildPlan(::testing::_, ::testing::_, ::testing::_)).Times(1);
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_, ::testing::_)).Times(0);
    EXPECT_CALL(*mockPlanBuilder2, buildPlan(::testing::_, ::testing::_, ::testing::_)).Times(0);

    PlanBuilderEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    MockGraph const mockGraph;
    HipdnnEnginePluginHandle const dummyHandle;
    MockHipdnnEnginePluginExecutionContext ctx;

    engine.initializeExecutionContext(dummyHandle, mockGraph, ctx);
}

TEST(TestPlanBuilderEngine, InitializeExecutionContextSkipsNonApplicableBuilders)
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

    PlanBuilderEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    MockGraph const mockGraph;
    HipdnnEnginePluginHandle const dummyHandle;
    MockHipdnnEnginePluginExecutionContext ctx;

    engine.initializeExecutionContext(dummyHandle, mockGraph, ctx);
}

TEST(TestPlanBuilderEngine, InitializeExecutionContextDoesNotCallBuildPlanIfNoApplicableBuilders)
{
    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    EXPECT_CALL(*mockPlanBuilder1, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(false));
    EXPECT_CALL(*mockPlanBuilder1, buildPlan(::testing::_, ::testing::_, ::testing::_)).Times(0);
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(false));
    EXPECT_CALL(*mockPlanBuilder2, buildPlan(::testing::_, ::testing::_, ::testing::_)).Times(0);

    PlanBuilderEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    MockGraph const mockGraph;
    HipdnnEnginePluginHandle const dummyHandle;
    MockHipdnnEnginePluginExecutionContext ctx;

    engine.initializeExecutionContext(dummyHandle, mockGraph, ctx);
}
