// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// TEMPLATE REFERENCE: Second PlanBuilder test example. This uses the same testing
// pattern as TestReluPlanBuilder.cpp but for a convolution operation.

#include <gtest/gtest.h>

#include <memory>

#include <hipdnn_data_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include "TestHelpers.hpp"
#include "engines/plans/ConvFwdPlan.hpp"
#include "engines/plans/ConvFwdPlanBuilder.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockKernelCompiler.hpp"
#include "mocks/MockRunnableKernel.hpp"

using namespace example_provider;
using namespace example_provider::test_helpers;
using ::testing::_;
using ::testing::Return;

class ConvFwdPlanBuilderTest : public ::testing::Test
{
protected:
    MockKernelCompiler mockCompiler;
    ExampleProviderHandle handle;

    std::unique_ptr<ConvFwdPlanBuilder> planBuilder;

    void SetUp() override
    {
        planBuilder = std::make_unique<ConvFwdPlanBuilder>(mockCompiler);
    }
};

TEST_F(ConvFwdPlanBuilderTest, IsApplicable_SingleNodeConvFwd_ReturnsTrue)
{
    auto fbb = createConvFwdGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(fbb.GetBufferPointer(),
                                                              fbb.GetSize());
    ASSERT_TRUE(graph.isValid());
    EXPECT_TRUE(planBuilder->isApplicable(handle, graph));
}

TEST_F(ConvFwdPlanBuilderTest, IsApplicable_ReluFwdGraph_ReturnsFalse)
{
    auto fbb = createReluFwdGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(fbb.GetBufferPointer(),
                                                              fbb.GetSize());
    ASSERT_TRUE(graph.isValid());
    EXPECT_FALSE(planBuilder->isApplicable(handle, graph));
}

TEST_F(ConvFwdPlanBuilderTest, IsApplicable_NonReluPointwise_ReturnsFalse)
{
    auto fbb = createNonReluPointwiseGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(fbb.GetBufferPointer(),
                                                              fbb.GetSize());
    ASSERT_TRUE(graph.isValid());
    EXPECT_FALSE(planBuilder->isApplicable(handle, graph));
}

TEST_F(ConvFwdPlanBuilderTest, IsApplicable_MultiNodeGraph_ReturnsFalse)
{
    auto fbb = createMultiNodeConvGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(fbb.GetBufferPointer(),
                                                              fbb.GetSize());
    ASSERT_TRUE(graph.isValid());
    EXPECT_FALSE(planBuilder->isApplicable(handle, graph));
}

TEST_F(ConvFwdPlanBuilderTest, IsApplicable_NonUnitDilation_ReturnsFalse)
{
    // Create a ConvFwd graph with dilation={2,2} on a large enough input (8x8)
    // so the output dimensions remain positive: outH = (8 - (2*(3-1)+1)) / 1 + 1 = 4
    auto fbb = createConvFwdGraph(1, 2, 3, 1, 1, 8, 8, 1, 3, 3, 0, 0, 1, 1, 2, 2);
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(fbb.GetBufferPointer(),
                                                              fbb.GetSize());
    ASSERT_TRUE(graph.isValid());
    EXPECT_FALSE(planBuilder->isApplicable(handle, graph));
}

TEST_F(ConvFwdPlanBuilderTest, GetMaxWorkspaceSize_ReturnsZero)
{
    auto fbb = createConvFwdGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(fbb.GetBufferPointer(),
                                                              fbb.GetSize());
    ExampleProviderSettings settings;
    EXPECT_EQ(planBuilder->getMaxWorkspaceSize(handle, graph, settings), 0u);
}

TEST_F(ConvFwdPlanBuilderTest, GetCustomKnobs_ReturnsBlockSizeKnob)
{
    auto fbb = createConvFwdGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(fbb.GetBufferPointer(),
                                                              fbb.GetSize());
    auto knobs = planBuilder->getCustomKnobs(handle, graph);
    ASSERT_EQ(knobs.size(), 1u);
    EXPECT_EQ(knobs[0].knob_id, "BLOCK_SIZE");
}

TEST_F(ConvFwdPlanBuilderTest, BuildPlan_SetsPlanOnContext)
{
    auto graphFbb = createConvFwdGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(graphFbb.GetBufferPointer(),
                                                              graphFbb.GetSize());

    auto configFbb = createEngineConfig(0);
    hipdnn_data_sdk::flatbuffer_utilities::EngineConfigWrapper config(configFbb.GetBufferPointer(),
                                                                      configFbb.GetSize());

    // Set up mock expectations for buildPlan
    auto compiledProgram = std::make_unique<MockCompiledProgram>();
    auto* rawProgram = compiledProgram.get();
    auto kernel = std::make_unique<MockRunnableKernel>();

    EXPECT_CALL(mockCompiler, compile("ConvForwardNaive.cpp", _))
        .WillOnce(Return(testing::ByMove(std::move(compiledProgram))));
    EXPECT_CALL(*rawProgram, getRunnableKernel("conv_forward_naive_kernel"))
        .WillOnce(Return(testing::ByMove(std::move(kernel))));

    ExampleProviderContext context;
    planBuilder->buildPlan(handle, graph, config, context);

    EXPECT_TRUE(context.hasValidPlan());
}
