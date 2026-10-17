// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The ported v0.x Engine and EngineConfig builders against the shim's backend
// C API and the mock hipDNN backend: knob projection, the knob maximum, and
// configs that choose only some knobs.
#include <hipdnn_compatibility/cudnn/cudnn_frontend_EngineConfig.h>

#include <gtest/gtest.h>

#include "fake_backend/ShimV8PlanFixture.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

using hipdnn_shim_test::BuilderFailure;
using hipdnn_shim_test::builderFailureOf;

constexpr std::int64_t KNOB_ENGINE = 0;

std::string knobSegment(cudnnBackendKnobType_t type, std::int64_t value)
{
    return "_k" + std::to_string(static_cast<int>(type)) + "=" + std::to_string(value);
}

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
    // BLOCK_SIZE has explicit valid values, "int_knob" has no cuDNN type and
    // the two STAGES knobs collide: only TILE_SIZE can be shown.
    installEngineKnobs(KNOB_ENGINE,
                       {
                           {"prov.tile_size", 8, 0, 64, 8, {}},
                           {"prov.block_size", 8, 0, 64, 1, {8, 16}},
                           {"prov.int_knob", 1, 0, 10, 1, {}},
                           {"first.stages", 1, 1, 4, 1, {}},
                           {"second.stages", 1, 1, 4, 1, {}},
                       });
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
    auto& knobs = built.getSupportedKnobs();
    ASSERT_EQ(knobs.size(), 2U);
    const auto tileSize = std::find_if(knobs.begin(), knobs.end(), [](const auto& knob) {
        return knob.getKnobType() == CUDNN_KNOB_TYPE_TILE_SIZE;
    });
    ASSERT_NE(tileSize, knobs.end());
    tileSize->setChoice(16);

    const fe::EngineConfig_v8 config = fe::EngineConfigBuilder_v8().setEngine(built).build();
    ASSERT_EQ(config.get_status(), CUDNN_STATUS_SUCCESS) << config.get_error();
    EXPECT_EQ(config.getTag(), graph.getTag());

    const fe::ExecutionPlan_v8 plan = fe::ExecutionPlanBuilder_v8().setEngineConfig(config).build();
    ASSERT_EQ(plan.get_status(), CUDNN_STATUS_SUCCESS) << plan.get_error();
    // Every stored choice stays in the tag; only TILE_SIZE reaches hipDNN.
    std::string expected = "ConvFwd_eng0";
    for(const auto& knob : knobs)
    {
        expected += knobSegment(knob.getKnobType(), knob.getChoice());
    }
    EXPECT_EQ(plan.getTag(), expected);
    EXPECT_NE(plan.getTag().find(knobSegment(CUDNN_KNOB_TYPE_STAGES, -1)), std::string::npos);
    EXPECT_EQ(created(HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR).size(), 1U);
}

TEST_F(TestCudnnShimV8EngineBuilders, GlobalIndexOutOfRangeRejected)
{
    const fe::OperationGraph_v8 graph = convGraph();

    for(const std::int64_t index : {std::int64_t{3}, std::int64_t{1000}})
    {
        SCOPED_TRACE(index);
        fe::EngineBuilder_v8 builder;
        builder.setOperationGraph(graph).setGlobalEngineIdx(index);
        const BuilderFailure failure
            = builderFailureOf([&builder] { [[maybe_unused]] auto built = builder.build(); });
        ASSERT_TRUE(failure.thrown);
        EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
        EXPECT_NE(failure.what.find("cudnnFinalize"), std::string::npos) << failure.what;
    }

    // Negative indices never reach the C API.
    fe::EngineBuilder_v8 negative;
    negative.setOperationGraph(graph).setGlobalEngineIdx(-1);
    const BuilderFailure failure
        = builderFailureOf([&negative] { [[maybe_unused]] auto built = negative.build(); });
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(failure.what.find("CUDNN_ATTR_ENGINE_GLOBAL_INDEX"), std::string::npos)
        << failure.what;
}

} // namespace
