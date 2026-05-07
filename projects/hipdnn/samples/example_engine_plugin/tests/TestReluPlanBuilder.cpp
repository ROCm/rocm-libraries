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
using ::testing::_; // NOLINT(bugprone-reserved-identifier)
using ::testing::Return;

class ReluPlanBuilderTest : public ::testing::Test
{
protected:
    MockKernelCompiler _mockCompiler;
    ExampleProviderHandle _handle;

    std::unique_ptr<ReluPlanBuilder> _planBuilder;

    void SetUp() override
    {
        _planBuilder = std::make_unique<ReluPlanBuilder>(_mockCompiler);
    }
};

TEST_F(ReluPlanBuilderTest, IsApplicable_SingleNodeReluFwd_ReturnsTrue)
{
    auto fbb = createReluFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(fbb.GetBufferPointer(),
                                                                           fbb.GetSize());
    ASSERT_TRUE(graph.isValid());
    EXPECT_TRUE(_planBuilder->isApplicable(_handle, graph));
}

TEST_F(ReluPlanBuilderTest, IsApplicable_NonReluPointwise_ReturnsFalse)
{
    auto fbb = createNonReluPointwiseGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(fbb.GetBufferPointer(),
                                                                           fbb.GetSize());
    ASSERT_TRUE(graph.isValid());
    EXPECT_FALSE(_planBuilder->isApplicable(_handle, graph));
}

TEST_F(ReluPlanBuilderTest, IsApplicable_MultiNodeGraph_ReturnsFalse)
{
    auto fbb = createMultiNodeReluGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(fbb.GetBufferPointer(),
                                                                           fbb.GetSize());
    ASSERT_TRUE(graph.isValid());
    EXPECT_FALSE(_planBuilder->isApplicable(_handle, graph));
}

TEST_F(ReluPlanBuilderTest, IsApplicable_ConvFwdGraph_ReturnsFalse)
{
    auto fbb = createConvFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(fbb.GetBufferPointer(),
                                                                           fbb.GetSize());
    ASSERT_TRUE(graph.isValid());
    EXPECT_FALSE(_planBuilder->isApplicable(_handle, graph));
}

TEST_F(ReluPlanBuilderTest, GetMaxWorkspaceSize_ReturnsZero)
{
    auto fbb = createReluFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(fbb.GetBufferPointer(),
                                                                           fbb.GetSize());
    const ExampleProviderSettings settings;
    EXPECT_EQ(_planBuilder->getMaxWorkspaceSize(_handle, graph, settings), 0u);
}

TEST_F(ReluPlanBuilderTest, GetCustomKnobs_ReturnsNegativeSlopeKnob)
{
    auto fbb = createReluFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(fbb.GetBufferPointer(),
                                                                           fbb.GetSize());
    const auto knobs = _planBuilder->getCustomKnobs(_handle, graph);
    ASSERT_EQ(knobs.size(), 1u);
    EXPECT_EQ(knobs[0].knob_id, "example.relu.negative_slope");
}

TEST_F(ReluPlanBuilderTest, InitializeExecutionSettings_NegativeSlopeKnob_StoresInSettings)
{
    auto graphFbb = createReluFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        graphFbb.GetBufferPointer(), graphFbb.GetSize());

    auto configFbb = createEngineConfigWithFloatKnob(0, "example.relu.negative_slope", 0.5);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper config(
        configFbb.GetBufferPointer(), configFbb.GetSize());

    ExampleProviderSettings settings;
    _planBuilder->initializeExecutionSettings(_handle, graph, config, settings);

    EXPECT_DOUBLE_EQ(settings.reluNegativeSlope, 0.5);
}

TEST_F(ReluPlanBuilderTest, BuildPlan_SetsPlanOnContext)
{
    auto graphFbb = createReluFwdGraph({1, 1, 4});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        graphFbb.GetBufferPointer(), graphFbb.GetSize());

    auto configFbb = createEngineConfig(0);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper config(
        configFbb.GetBufferPointer(), configFbb.GetSize());

    // Set up mock expectations for buildPlan
    auto compiledProgram = std::make_unique<MockCompiledProgram>();
    auto* rawProgram = compiledProgram.get();
    auto kernel = std::make_unique<MockRunnableKernel>();

    EXPECT_CALL(_mockCompiler, compile("ReluForward.cpp", _))
        .WillOnce(Return(testing::ByMove(std::move(compiledProgram))));
    EXPECT_CALL(*rawProgram, getRunnableKernel("relu_forward_kernel"))
        .WillOnce(Return(testing::ByMove(std::move(kernel))));

    ExampleProviderContext context;
    _planBuilder->buildPlan(_handle, graph, config, context);

    // After buildPlan, context should have a valid plan
    EXPECT_TRUE(context.hasValidPlan());
}
