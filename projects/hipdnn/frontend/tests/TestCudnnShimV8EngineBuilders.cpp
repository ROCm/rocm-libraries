// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The ported v0.x Engine and EngineConfig builders against the mock hipDNN
// backend: knob projection, the knob maximum, and configs that choose only
// some knobs.
#include <hipdnn_compatibility/cudnn/cudnn_frontend_EngineConfig.h>

#include <gtest/gtest.h>

#include "fake_backend/ShimV8PlanFixture.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

using hipdnn_shim_test::BuildFailure;
using hipdnn_shim_test::failureOf;

constexpr std::int64_t KNOB_ENGINE = 0;

class TestCudnnShimV8EngineBuilders : public hipdnn_shim_test::ShimV8PlanFixture
{
};

TEST_F(TestCudnnShimV8EngineBuilders, KnobMaxValueIsMaximum)
{
    installEngineKnobs(KNOB_ENGINE, {{"prov.tile_size", 8, 0, 64, 8, {}}});
    const fe::OperationGraph_v8 graph = convGraph();

    fe::Engine_v8 built = engine(graph, KNOB_ENGINE);

    ASSERT_EQ(built.get_status(), CUDNN_STATUS_SUCCESS) << built.get_error();
    ASSERT_EQ(built.getSupportedKnobs().size(), 1U);
    const auto& knob = built.getSupportedKnobs().front();
    EXPECT_EQ(knob.getKnobType(), CUDNN_KNOB_TYPE_TILE_SIZE);
    EXPECT_EQ(knob.getMinValue(), 0);
    // Upstream returns the minimum here.
    EXPECT_EQ(knob.getMaxValue(), 64);
    EXPECT_EQ(knob.getStride(), 8);
    EXPECT_EQ(knob.getChoice(), -1);
}

TEST_F(TestCudnnShimV8EngineBuilders, KnobsListVisibleOnly)
{
    // Only TILE_SIZE of the mixed knobs can be shown.
    installEngineKnobs(KNOB_ENGINE, mixedKnobs());
    const fe::OperationGraph_v8 graph = convGraph();

    fe::Engine_v8 built = engine(graph, KNOB_ENGINE);

    ASSERT_EQ(built.get_status(), CUDNN_STATUS_SUCCESS) << built.get_error();
    ASSERT_EQ(built.getSupportedKnobs().size(), 1U);
    EXPECT_EQ(built.getSupportedKnobs().front().getKnobType(), CUDNN_KNOB_TYPE_TILE_SIZE);
    EXPECT_EQ(built.getTag(), graph.getTag());

    // An engine without knobs builds with an empty list.
    fe::Engine_v8 bare = engine(graph, 1);
    ASSERT_EQ(bare.get_status(), CUDNN_STATUS_SUCCESS) << bare.get_error();
    EXPECT_TRUE(bare.getSupportedKnobs().empty());
}

// Upstream sends -1 for every knob the caller leaves unset once any is set.
TEST_F(TestCudnnShimV8EngineBuilders, PartialKnobChoiceBuilds)
{
    installEngineKnobs(KNOB_ENGINE,
                       {{"prov.tile_size", 8, 0, 64, 8, {}}, {"prov.stages", 1, 1, 4, 1, {}}});
    const fe::OperationGraph_v8 graph = convGraph();
    fe::Engine_v8 built = engine(graph, KNOB_ENGINE);
    ASSERT_EQ(built.get_status(), CUDNN_STATUS_SUCCESS) << built.get_error();
    ASSERT_EQ(built.getSupportedKnobs().size(), 2U);
    choose(built, CUDNN_KNOB_TYPE_TILE_SIZE, 16);

    const fe::EngineConfig_v8 config = fe::EngineConfigBuilder_v8().setEngine(built).build();
    ASSERT_EQ(config.get_status(), CUDNN_STATUS_SUCCESS) << config.get_error();
    EXPECT_EQ(config.getTag(), graph.getTag());

    const fe::ExecutionPlan_v8 plan = fe::ExecutionPlanBuilder_v8().setEngineConfig(config).build();
    ASSERT_EQ(plan.get_status(), CUDNN_STATUS_SUCCESS) << plan.get_error();
    // Every stored choice stays in the tag; only TILE_SIZE reaches hipDNN.
    std::string expected = "ConvFwd_eng0";
    for(const auto& knob : built.getSupportedKnobs())
    {
        expected += knobSegment(knob.getKnobType(), knob.getChoice());
    }
    EXPECT_EQ(plan.getTag(), expected);
    EXPECT_NE(plan.getTag().find(knobSegment(CUDNN_KNOB_TYPE_STAGES, -1)), std::string::npos);
    EXPECT_EQ(created(HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR).size(), 1U);
}

TEST_F(TestCudnnShimV8EngineBuilders, GlobalIndexPastEngineCountNotSupported)
{
    const fe::OperationGraph_v8 graph = convGraph();

    for(const std::int64_t index : {std::int64_t{3}, std::int64_t{1000}})
    {
        SCOPED_TRACE(index);
        fe::EngineBuilder_v8 builder;
        builder.setOperationGraph(graph).setGlobalEngineIdx(index);
        const BuildFailure failure
            = failureOf([&builder] { [[maybe_unused]] auto built = builder.build(); });
        ASSERT_TRUE(failure.thrown);
        EXPECT_EQ(failure.status, CUDNN_STATUS_NOT_SUPPORTED);
        EXPECT_NE(failure.what.find("EngineBuilder: global engine index = "), std::string::npos)
            << failure.what;
    }

    // Negative indices are rejected before the graph's engine space is consulted.
    fe::EngineBuilder_v8 negative;
    negative.setOperationGraph(graph).setGlobalEngineIdx(-1);
    const BuildFailure failure
        = failureOf([&negative] { [[maybe_unused]] auto built = negative.build(); });
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(failure.what.find("EngineBuilder: the global engine index must be non-negative"),
              std::string::npos)
        << failure.what;
}

} // namespace
