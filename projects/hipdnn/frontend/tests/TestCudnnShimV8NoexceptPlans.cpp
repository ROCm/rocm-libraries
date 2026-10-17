// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Plan builder and autotune status-return path: with exceptions disabled a
// failed plan keeps its status instead of throwing, and cudnnGetPlan keeps it.
#ifndef NV_CUDNN_DISABLE_EXCEPTION
#error "TestCudnnShimV8NoexceptPlans.cpp must be built with NV_CUDNN_DISABLE_EXCEPTION"
#endif

#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_find_plan.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_get_plan.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "fake_backend/ShimV8PlanFixture.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

using ::testing::HasSubstr;

fe::EngineConfigList heuristicSource(fe::OperationGraph& graph)
{
    fe::EngineHeuristics heuristics = fe::EngineHeuristicsBuilder()
                                          .setOperationGraph(graph)
                                          .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                                          .build();
    return heuristics.getEngineConfig(heuristics.getEngineConfigCount());
}

class TestCudnnShimV8NoexceptPlans : public hipdnn_shim_test::ShimV8PlanFixture
{
protected:
    static fe::EngineConfigGenerator heuristicGenerator()
    {
        const std::array<fe::GeneratorSource, 1> sources = {heuristicSource};
        return {static_cast<int>(sources.size()), sources.data()};
    }
};

TEST_F(TestCudnnShimV8NoexceptPlans, PlanFailureSetsStatus)
{
    failPlanFinalizeFor(1);
    fe::OperationGraph graph = convGraph();
    const fe::EngineConfigList configs = engineConfigs(graph);
    ASSERT_EQ(configs.size(), 3U);

    const fe::ExecutionPlan failed = planFor(configs[1], graph.getTag());
    EXPECT_EQ(failed.get_status(), CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_THAT(failed.get_error(), HasSubstr("cudnnFinalize"));

    const fe::ExecutionPlan withCache = fe::ExecutionPlanBuilder()
                                            .setEngineConfig(configs[0], graph.getTag())
                                            .setKernelCache(std::make_shared<fe::KernelCache>())
                                            .build();
    EXPECT_EQ(withCache.get_status(), CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_THAT(withCache.get_error(), HasSubstr("KERNEL_CACHE"));

    const fe::ExecutionPlan good = planFor(configs[2], graph.getTag());
    EXPECT_EQ(good.get_status(), CUDNN_STATUS_SUCCESS) << good.get_error();
}

// Without exceptions upstream cannot tell a failed plan from a good one, so
// every plan is kept and the caller checks get_status().
TEST_F(TestCudnnShimV8NoexceptPlans, GetPlanKeepsFailedPlans)
{
    failPlanFinalizeFor(1);
    fe::OperationGraph graph = convGraph();
    fe::EngineConfigGenerator generator = heuristicGenerator();

    const fe::executionPlans_t plans = generator.cudnnGetPlan(_handle, graph);

    ASSERT_EQ(plans.size(), 3U);
    EXPECT_EQ(plans[0].get_status(), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(plans[1].get_status(), CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(plans[2].get_status(), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(plans[0].getTag(), "ConvFwd_eng0");
    EXPECT_EQ(plans[2].getTag(), "ConvFwd_eng2");
}

TEST_F(TestCudnnShimV8NoexceptPlans, EmptyFindAndCacheSetsStatus)
{
    for(std::int64_t engine = 0; engine < 3; ++engine)
    {
        failExecuteFor(engine, 0);
    }
    fe::OperationGraph graph = convGraph();
    const fe::VariantPack pack = variantPack();
    fe::EngineConfigGenerator generator = heuristicGenerator();
    fe::ExecutionPlanCache cache("find");

    const fe::ExecutionPlan plan
        = generator.cudnnFindPlanAndCache<fe::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE>(
            _handle, graph, pack, cache);

    // The returned plan is an empty builder's: its status names the missing engine config.
    EXPECT_EQ(plan.get_status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_THAT(plan.get_error(), HasSubstr("CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG"));
    const fe::ExecutionPlan* cached = nullptr;
    EXPECT_FALSE(cache.get_plan_from_cache(graph, cached));
}

} // namespace
