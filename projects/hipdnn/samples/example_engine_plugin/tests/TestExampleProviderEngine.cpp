// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include "TestHelpers.hpp"
#include "engines/ExampleProviderEngine.hpp"

using namespace example_provider;
using namespace example_provider::test_helpers;

/// Mock PlanBuilder with controllable isApplicable() return value.
class MockPlanBuilder : public hipdnn_plugin_sdk::IPlanBuilder<ExampleProviderHandle,
                                                               ExampleProviderSettings,
                                                               ExampleProviderContext>
{
public:
    explicit MockPlanBuilder(bool applicable)
        : _applicable(applicable)
    {
    }

    bool isApplicable(
        const ExampleProviderHandle& /*handle*/,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/) const override
    {
        return _applicable;
    }

    size_t
        getMaxWorkspaceSize(const ExampleProviderHandle& /*handle*/,
                            const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
                            const ExampleProviderSettings& /*executionSettings*/) const override
    {
        return 0;
    }

    void initializeExecutionSettings(
        const ExampleProviderHandle& /*handle*/,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/,
        ExampleProviderSettings& /*executionSettings*/) const override
    {
    }

    void buildPlan(const ExampleProviderHandle& /*handle*/,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
                   [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::
                       IEngineConfig& /*engineConfig*/,
                   ExampleProviderContext& /*executionContext*/) const override
    {
    }

    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> getCustomKnobs(
        const ExampleProviderHandle& /*handle*/,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/) const override
    {
        return {};
    }

private:
    bool _applicable;
};

class ExampleProviderEngineTest : public ::testing::Test
{
protected:
    ExampleProviderHandle _handle;
};

TEST_F(ExampleProviderEngineTest, IsApplicable_WithMatchingBuilder_ReturnsTrue)
{
    ExampleProviderEngine engine(1);
    engine.addPlanBuilder(std::make_unique<MockPlanBuilder>(true));

    auto fbb = createReluFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(fbb.GetBufferPointer(),
                                                                           fbb.GetSize());
    ASSERT_TRUE(graph.isValid());
    EXPECT_TRUE(engine.isApplicable(_handle, graph));
}

TEST_F(ExampleProviderEngineTest, IsApplicable_WithNoMatchingBuilder_ReturnsFalse)
{
    ExampleProviderEngine engine(1);
    engine.addPlanBuilder(std::make_unique<MockPlanBuilder>(false));

    auto fbb = createReluFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(fbb.GetBufferPointer(),
                                                                           fbb.GetSize());
    ASSERT_TRUE(graph.isValid());
    EXPECT_FALSE(engine.isApplicable(_handle, graph));
}
