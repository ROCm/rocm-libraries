// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// TEMPLATE ADAPTATION: Demonstrates the testing pattern for PlanBuilders. Key test categories:
// (1) isApplicable: verify true for matching graphs, false for non-matching.
// (2) getCustomKnobs: verify knob IDs and types.
// (3) buildPlan: verify a valid plan is created (uses mock compilation chain).
// Adapt these tests or discard and replace with tests appropriate for your PlanBuilder.

#include <gtest/gtest.h>

#include <memory>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include "TestHelpers.hpp"
#include "engines/plans/ReluPlan.hpp"
#include "engines/plans/ReluPlanBuilder.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockKernelCompiler.hpp"
#include "mocks/MockRunnableKernel.hpp"

using namespace example_provider;
using namespace example_provider::test_helpers;
using ::testing::_;
using ::testing::Return;

class ReluPlanBuilderTest : public ::testing::Test
{
protected:
    MockKernelCompiler mockCompiler;
    ExampleProviderHandle handle;

    std::unique_ptr<ReluPlanBuilder> planBuilder;

    void SetUp() override
    {
        planBuilder = std::make_unique<ReluPlanBuilder>(mockCompiler);
    }
};

TEST_F(ReluPlanBuilderTest, IsApplicable_SingleNodeReluFwd_ReturnsTrue)
{
    auto fbb = createReluFwdGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(fbb.GetBufferPointer(),
                                                                     fbb.GetSize());
    ASSERT_TRUE(graph.isValid());
    EXPECT_TRUE(planBuilder->isApplicable(handle, graph));
}

TEST_F(ReluPlanBuilderTest, IsApplicable_NonReluPointwise_ReturnsFalse)
{
    auto fbb = createNonReluPointwiseGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(fbb.GetBufferPointer(),
                                                                     fbb.GetSize());
    ASSERT_TRUE(graph.isValid());
    EXPECT_FALSE(planBuilder->isApplicable(handle, graph));
}

TEST_F(ReluPlanBuilderTest, IsApplicable_MultiNodeGraph_ReturnsFalse)
{
    auto fbb = createMultiNodeReluGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(fbb.GetBufferPointer(),
                                                                     fbb.GetSize());
    ASSERT_TRUE(graph.isValid());
    EXPECT_FALSE(planBuilder->isApplicable(handle, graph));
}

TEST_F(ReluPlanBuilderTest, IsApplicable_ConvFwdGraph_ReturnsFalse)
{
    auto fbb = createConvFwdGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(fbb.GetBufferPointer(),
                                                                     fbb.GetSize());
    ASSERT_TRUE(graph.isValid());
    EXPECT_FALSE(planBuilder->isApplicable(handle, graph));
}

TEST_F(ReluPlanBuilderTest, GetMaxWorkspaceSize_ReturnsZero)
{
    auto fbb = createReluFwdGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(fbb.GetBufferPointer(),
                                                                     fbb.GetSize());
    ExampleProviderSettings settings;
    EXPECT_EQ(planBuilder->getMaxWorkspaceSize(handle, graph, settings), 0u);
}

TEST_F(ReluPlanBuilderTest, GetCustomKnobs_ReturnsNegativeSlopeKnob)
{
    auto fbb = createReluFwdGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(fbb.GetBufferPointer(),
                                                                     fbb.GetSize());
    auto knobs = planBuilder->getCustomKnobs(handle, graph);
    ASSERT_EQ(knobs.size(), 1u);
    EXPECT_EQ(knobs[0].knob_id, "example.relu.negative_slope");
}

TEST_F(ReluPlanBuilderTest, InitializeExecutionSettings_NegativeSlopeKnob_StoresInSettings)
{
    auto graphFbb = createReluFwdGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(graphFbb.GetBufferPointer(),
                                                                     graphFbb.GetSize());

    auto configFbb = createEngineConfigWithFloatKnob(0, "example.relu.negative_slope", 0.5);
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper config(
        configFbb.GetBufferPointer(), configFbb.GetSize());

    ExampleProviderSettings settings;
    planBuilder->initializeExecutionSettings(handle, graph, config, settings);

    EXPECT_DOUBLE_EQ(settings.reluNegativeSlope, 0.5);
}

TEST_F(ReluPlanBuilderTest, BuildPlan_SetsPlanOnContext)
{
    auto graphFbb = createReluFwdGraph({1, 1, 4});
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(graphFbb.GetBufferPointer(),
                                                                     graphFbb.GetSize());

    auto configFbb = createEngineConfig(0);
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper config(
        configFbb.GetBufferPointer(), configFbb.GetSize());

    // Set up mock expectations for buildPlan
    auto compiledProgram = std::make_unique<MockCompiledProgram>();
    auto* rawProgram = compiledProgram.get();
    auto kernel = std::make_unique<MockRunnableKernel>();

    EXPECT_CALL(mockCompiler, compile("ReluForward.cpp", _))
        .WillOnce(Return(testing::ByMove(std::move(compiledProgram))));
    EXPECT_CALL(*rawProgram, getRunnableKernel("relu_forward_kernel"))
        .WillOnce(Return(testing::ByMove(std::move(kernel))));

    ExampleProviderContext context;
    planBuilder->buildPlan(handle, graph, config, context);

    // After buildPlan, context should have a valid plan
    EXPECT_TRUE(context.hasValidPlan());
}
