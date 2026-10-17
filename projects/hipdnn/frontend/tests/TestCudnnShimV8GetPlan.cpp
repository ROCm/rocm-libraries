// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The ported EngineConfigGenerator and cudnnGetPlan against the shim's backend
// C API and the mock hipDNN backend.
#include <hipdnn_compatibility/cudnn/cudnn_frontend_get_plan.h>

#include <gtest/gtest.h>

#include "fake_backend/ShimV8PlanFixture.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

fe::EngineConfigList heuristicSource(fe::OperationGraph& graph)
{
    fe::EngineHeuristics_v8 heuristics = fe::EngineHeuristicsBuilder_v8()
                                             .setOperationGraph(graph)
                                             .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                                             .build();
    return heuristics.getEngineConfig(heuristics.getEngineConfigCount());
}

// The heuristic list's last config only.
fe::EngineConfigList lastConfigSource(fe::OperationGraph& graph)
{
    fe::EngineConfigList configs = heuristicSource(graph);
    configs.erase(configs.begin(), configs.end() - 1);
    return configs;
}

class TestCudnnShimV8GetPlan : public hipdnn_shim_test::ShimV8PlanFixture
{
protected:
    static fe::EngineConfigGenerator heuristicGenerator()
    {
        const std::array<fe::GeneratorSource, 1> sources = {heuristicSource};
        return {static_cast<int>(sources.size()), sources.data()};
    }
};

// A plan whose build throws is skipped, as cudnnGetPlan catches per config.
TEST_F(TestCudnnShimV8GetPlan, FailingConfigSkipped)
{
    failPlanFinalizeFor(1);
    fe::OperationGraph graph = convGraph();
    fe::EngineConfigGenerator generator = heuristicGenerator();

    const fe::executionPlans_t plans = generator.cudnnGetPlan(_handle, graph);

    EXPECT_EQ(tagsOf(plans), (std::vector<std::string>{"ConvFwd_eng0", "ConvFwd_eng2"}));
    for(const fe::ExecutionPlan& plan : plans)
    {
        EXPECT_EQ(plan.get_status(), CUDNN_STATUS_SUCCESS) << plan.get_error();
    }
}

TEST_F(TestCudnnShimV8GetPlan, PredicateFiltersPlans)
{
    installEngines(4);
    fe::OperationGraph graph = convGraph();
    fe::EngineConfigGenerator generator = heuristicGenerator();

    // True blocks the plan.
    const fe::executionPlans_t plans
        = generator.cudnnGetPlan(_handle, graph, [](const fe::ExecutionPlan& plan) {
              return plan.getTag() == "ConvFwd_eng1" || plan.getTag() == "ConvFwd_eng3";
          });

    EXPECT_EQ(tagsOf(plans), (std::vector<std::string>{"ConvFwd_eng0", "ConvFwd_eng2"}));
}

TEST_F(TestCudnnShimV8GetPlan, MaxPlansCaps)
{
    installEngines(4);
    fe::OperationGraph graph = convGraph();
    fe::EngineConfigGenerator generator = heuristicGenerator();

    EXPECT_EQ(tagsOf(generator.cudnnGetPlan(_handle, graph, 2)),
              (std::vector<std::string>{"ConvFwd_eng0", "ConvFwd_eng1"}));

    // Only plans that built count toward the cap.
    failPlanFinalizeFor(0);
    EXPECT_EQ(tagsOf(generator.cudnnGetPlan(_handle, graph, 2)),
              (std::vector<std::string>{"ConvFwd_eng1", "ConvFwd_eng2"}));
}

TEST_F(TestCudnnShimV8GetPlan, GeneratorConcatenatesSources)
{
    fe::OperationGraph graph = convGraph();
    const std::array<fe::GeneratorSource, 2> sources = {heuristicSource, lastConfigSource};
    fe::EngineConfigGenerator generator(static_cast<int>(sources.size()), sources.data());

    const fe::EngineConfigList configs = generator.generate_engine_config(graph);
    EXPECT_EQ(indicesOf(configs), (std::vector<std::int64_t>{0, 1, 2, 2}));

    EXPECT_EQ(
        tagsOf(generator.cudnnGetPlan(_handle, graph)),
        (std::vector<std::string>{"ConvFwd_eng0", "ConvFwd_eng1", "ConvFwd_eng2", "ConvFwd_eng2"}));
}

} // namespace
