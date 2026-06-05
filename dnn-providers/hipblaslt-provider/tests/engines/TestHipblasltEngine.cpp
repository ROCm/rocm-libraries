// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <memory>
#include <set>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineDetailsWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>

#include "engines/HipblasltEngine.hpp"
#include "mocks/MockHipdnnEnginePluginExecutionContext.hpp"
#include "mocks/MockPlanBuilder.hpp"

using namespace hipblaslt_plugin;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_plugin_sdk;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

TEST(TestHipblasltEngine, ConstructorAndId)
{
    HipblasltEngine const engine(42);
    EXPECT_EQ(engine.id(), 42);
}

TEST(TestHipblasltEngine, WorkspaceSizeReturnsZeroIfNoPlanBuilders)
{
    HipblasltEngine const engine(1);

    MockGraph const mockGraph;

    HipdnnEnginePluginHandle const dummyHandle;
    EXPECT_EQ(engine.getWorkspaceSize(dummyHandle, mockGraph), 0u);
}

TEST(TestHipblasltEngine, WorkspaceSizeReturnsPlanBuilderWorkspace)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder, getWorkspaceSize(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(1337u));

    HipblasltEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    MockGraph const mockGraph;

    HipdnnEnginePluginHandle const dummyHandle;
    EXPECT_EQ(engine.getWorkspaceSize(dummyHandle, mockGraph), 1337u);
}

TEST(TestHipblasltEngine, WorkspaceSizeReturnsMaxPlanBuilderWorkspace)
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

    HipblasltEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    MockGraph const mockGraph;

    HipdnnEnginePluginHandle const dummyHandle;
    EXPECT_EQ(engine.getWorkspaceSize(dummyHandle, mockGraph), 45000u);
}

TEST(TestHipblasltEngine, WorkspaceSizeReturnsZeroIfNoPlanBuilderApplicable)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(false));

    HipblasltEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    MockGraph const mockGraph;

    HipdnnEnginePluginHandle const dummyHandle;
    EXPECT_EQ(engine.getWorkspaceSize(dummyHandle, mockGraph), 0u);
}

TEST(TestHipblasltEngine, IsApplicableReturnsTrueIfAnyPlanBuilderApplicable)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();

    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));

    HipblasltEngine engine(0);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    MockGraph const mockGraph;
    auto graphBuilder = hipdnn_test_sdk::utilities::createEmptyValidGraph();

    HipdnnEnginePluginHandle dummyHandle;
    EXPECT_TRUE(engine.isApplicable(dummyHandle, mockGraph));
}

TEST(TestHipblasltEngine, IsApplicableReturnsAfterTheFirstApplicablePlanBuilder)
{
    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    EXPECT_CALL(*mockPlanBuilder1, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_, ::testing::_)).Times(0);

    HipblasltEngine engine(0);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    MockGraph const mockGraph;
    auto graphBuilder = hipdnn_test_sdk::utilities::createEmptyValidGraph();

    HipdnnEnginePluginHandle dummyHandle;
    EXPECT_TRUE(engine.isApplicable(dummyHandle, mockGraph));
}

TEST(TestHipblasltEngine, IsApplicableReturnsFalseIfNoPlanBuilders)
{
    HipblasltEngine const engine(0);

    MockGraph const mockGraph;
    auto graphBuilder = hipdnn_test_sdk::utilities::createEmptyValidGraph();

    HipdnnEnginePluginHandle dummyHandle;
    EXPECT_FALSE(engine.isApplicable(dummyHandle, mockGraph));
}

TEST(TestHipblasltEngine, IsApplicableReturnsFalseIfNoPlanBuilderApplicable)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(false));

    HipblasltEngine engine(0);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    MockGraph const mockGraph;
    auto graphBuilder = hipdnn_test_sdk::utilities::createEmptyValidGraph();

    HipdnnEnginePluginHandle dummyHandle;
    EXPECT_FALSE(engine.isApplicable(dummyHandle, mockGraph));
}

TEST(TestHipblasltEngine, GetDetailsReturnsSerializedEngineDetails)
{
    HipblasltEngine const engine(hipdnn_data_sdk::utilities::HIPBLASLT_ENGINE_ID);
    HipdnnEnginePluginHandle dummyHandle;

    hipdnnPluginConstData_t result;
    engine.getDetails(dummyHandle, result);

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineDetailsWrapper const engineDetails(
        result.ptr, result.size);
    EXPECT_EQ(engineDetails.engineId(), hipdnn_data_sdk::utilities::HIPBLASLT_ENGINE_ID);
}

TEST(TestHipblasltEngine, InitializeExecutionContextInvokesFirstApplicablePlanBuilder)
{
    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    // Only the first plan builder is applicable
    EXPECT_CALL(*mockPlanBuilder1, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder1, buildPlan(::testing::_, ::testing::_, ::testing::_)).Times(1);
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_, ::testing::_)).Times(0);
    EXPECT_CALL(*mockPlanBuilder2, buildPlan(::testing::_, ::testing::_, ::testing::_)).Times(0);

    HipblasltEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    MockGraph const mockGraph;
    HipdnnEnginePluginHandle const dummyHandle;
    MockHipdnnEnginePluginExecutionContext ctx;

    engine.initializeExecutionContext(dummyHandle, mockGraph, ctx);
}

TEST(TestHipblasltEngine, InitializeExecutionContextSkipsNonApplicableBuilders)
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

    HipblasltEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    MockGraph const mockGraph;
    HipdnnEnginePluginHandle const dummyHandle;
    MockHipdnnEnginePluginExecutionContext ctx;

    engine.initializeExecutionContext(dummyHandle, mockGraph, ctx);
}

TEST(TestHipblasltEngine, InitializeExecutionContextDoesNotCallBuildPlanIfNoApplicableBuilders)
{
    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    EXPECT_CALL(*mockPlanBuilder1, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(false));
    EXPECT_CALL(*mockPlanBuilder1, buildPlan(::testing::_, ::testing::_, ::testing::_)).Times(0);
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(false));
    EXPECT_CALL(*mockPlanBuilder2, buildPlan(::testing::_, ::testing::_, ::testing::_)).Times(0);

    HipblasltEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    MockGraph const mockGraph;
    HipdnnEnginePluginHandle const dummyHandle;
    MockHipdnnEnginePluginExecutionContext ctx;

    engine.initializeExecutionContext(dummyHandle, mockGraph, ctx);
}
