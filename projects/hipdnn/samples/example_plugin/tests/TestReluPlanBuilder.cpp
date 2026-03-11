// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstring>
#include <memory>

#include <hipdnn_data_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include "TestHelpers.hpp"
#include "engines/plans/ReluPlan.hpp"
#include "engines/plans/ReluPlanBuilder.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockDevicePropertyProvider.hpp"
#include "mocks/MockKernelCompiler.hpp"
#include "mocks/MockRunnableKernel.hpp"

using namespace example_plugin;
using namespace example_plugin::test_helpers;
using ::testing::_;
using ::testing::Return;

class ReluPlanBuilderTest : public ::testing::Test
{
protected:
    MockKernelCompiler mockCompiler;
    MockDevicePropertyProvider mockDeviceProps;
    ExamplePluginHandle handle;

    std::unique_ptr<ReluPlanBuilder> planBuilder;

    void SetUp() override
    {
        planBuilder = std::make_unique<ReluPlanBuilder>(mockCompiler, mockDeviceProps);
    }
};

TEST_F(ReluPlanBuilderTest, IsApplicable_SingleNodeReluFwd_ReturnsTrue)
{
    auto fbb = createReluFwdGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(fbb.GetBufferPointer(),
                                                              fbb.GetSize());
    ASSERT_TRUE(graph.isValid());
    EXPECT_TRUE(planBuilder->isApplicable(handle, graph));
}

TEST_F(ReluPlanBuilderTest, IsApplicable_NonReluPointwise_ReturnsFalse)
{
    auto fbb = createNonReluPointwiseGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(fbb.GetBufferPointer(),
                                                              fbb.GetSize());
    ASSERT_TRUE(graph.isValid());
    EXPECT_FALSE(planBuilder->isApplicable(handle, graph));
}

TEST_F(ReluPlanBuilderTest, IsApplicable_MultiNodeGraph_ReturnsFalse)
{
    auto fbb = createMultiNodeReluGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(fbb.GetBufferPointer(),
                                                              fbb.GetSize());
    ASSERT_TRUE(graph.isValid());
    EXPECT_FALSE(planBuilder->isApplicable(handle, graph));
}

TEST_F(ReluPlanBuilderTest, IsApplicable_ConvFwdGraph_ReturnsFalse)
{
    auto fbb = createConvFwdGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(fbb.GetBufferPointer(),
                                                              fbb.GetSize());
    ASSERT_TRUE(graph.isValid());
    EXPECT_FALSE(planBuilder->isApplicable(handle, graph));
}

TEST_F(ReluPlanBuilderTest, GetMaxWorkspaceSize_ReturnsSizeofFloat)
{
    auto fbb = createReluFwdGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(fbb.GetBufferPointer(),
                                                              fbb.GetSize());
    ExamplePluginSettings settings;
    EXPECT_EQ(planBuilder->getMaxWorkspaceSize(handle, graph, settings), sizeof(float));
}

TEST_F(ReluPlanBuilderTest, GetCustomKnobs_ReturnsNegativeSlopeKnob)
{
    auto fbb = createReluFwdGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(fbb.GetBufferPointer(),
                                                              fbb.GetSize());
    auto knobs = planBuilder->getCustomKnobs(handle, graph);
    ASSERT_EQ(knobs.size(), 1u);
    EXPECT_EQ(knobs[0].knob_id, "example.relu.negative_slope");
}

TEST_F(ReluPlanBuilderTest, BuildPlan_SetsPlanOnContext)
{
    auto graphFbb = createReluFwdGraph({1, 1, 4});
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(graphFbb.GetBufferPointer(),
                                                              graphFbb.GetSize());

    auto configFbb = createEngineConfig(0);
    hipdnn_data_sdk::flatbuffer_utilities::EngineConfigWrapper config(configFbb.GetBufferPointer(),
                                                                      configFbb.GetSize());

    // Set up mock expectations for buildPlan
    hipDeviceProp_t props = {};
    snprintf(props.gcnArchName, sizeof(props.gcnArchName), "%s", "gfx942");
    EXPECT_CALL(mockDeviceProps, getDeviceProperties()).WillOnce(Return(props));

    auto compiledProgram = std::make_unique<MockCompiledProgram>();
    auto* rawProgram = compiledProgram.get();
    auto kernel = std::make_unique<MockRunnableKernel>();

    EXPECT_CALL(mockCompiler, compile("ReluForward.cpp", _))
        .WillOnce(Return(testing::ByMove(std::move(compiledProgram))));
    EXPECT_CALL(*rawProgram, getKernel("relu_forward_kernel"))
        .WillOnce(Return(testing::ByMove(std::move(kernel))));

    ExamplePluginContext context;
    planBuilder->buildPlan(handle, graph, config, context);

    // After buildPlan, context should have a valid plan
    EXPECT_TRUE(context.hasValidPlan());
}
