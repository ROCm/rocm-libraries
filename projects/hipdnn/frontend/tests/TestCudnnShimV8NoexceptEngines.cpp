// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The engine-layer builders and get_heuristics_list with exceptions disabled: failures before a
// descriptor exists leave a null raw view, and the heuristics list reports per-mode statuses.
#ifndef NV_CUDNN_DISABLE_EXCEPTION
#error "TestCudnnShimV8NoexceptEngines.cpp must be built with NV_CUDNN_DISABLE_EXCEPTION"
#endif

#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <gtest/gtest.h>

#include "fake_backend/ShimV8PlanFixture.hpp"

#include <string>
#include <vector>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

using fe::AllowAll;

class TestCudnnShimV8NoexceptEngines : public hipdnn_shim_test::ShimV8PlanFixture
{
};

TEST_F(TestCudnnShimV8NoexceptEngines, EarlyBuildFailureLeavesNullRawDesc)
{
    const fe::EngineFallbackList fallback = fe::EngineFallbackListBuilder().build();
    EXPECT_EQ(fallback.get_status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(fallback.get_raw_desc(), nullptr);

    const fe::EngineHeuristics heuristics = fe::EngineHeuristicsBuilder().build();
    EXPECT_EQ(heuristics.get_status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(heuristics.get_raw_desc(), nullptr);

    const fe::Engine negative = fe::EngineBuilder().setGlobalEngineIdx(-1).build();
    EXPECT_EQ(negative.get_status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(negative.get_raw_desc(), nullptr);

    const fe::Engine noGraph = fe::EngineBuilder().setGlobalEngineIdx(0).build();
    EXPECT_EQ(noGraph.get_status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(noGraph.get_raw_desc(), nullptr);

    const fe::EngineConfig config = fe::EngineConfigBuilder().build();
    EXPECT_EQ(config.get_status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(config.get_raw_desc(), nullptr);
}

TEST_F(TestCudnnShimV8NoexceptEngines, HeuristicsListReportsStatuses)
{
    fe::OperationGraph graph = convGraph();
    const std::vector<std::string> modes{"heuristics_mode_a", "heuristics_fallback"};

    fe::EngineConfigList all;
    EXPECT_EQ(
        fe::get_heuristics_list(modes, graph, AllowAll, all, /*evaluate_all=*/true, /*sm_count=*/8),
        (std::vector<cudnnStatus_t>{CUDNN_STATUS_NOT_SUPPORTED, CUDNN_STATUS_NOT_SUPPORTED}));
    EXPECT_TRUE(all.empty());

    fe::EngineConfigList first;
    EXPECT_EQ(fe::get_heuristics_list(
                  modes, graph, AllowAll, first, /*evaluate_all=*/false, /*sm_count=*/8),
              std::vector<cudnnStatus_t>{CUDNN_STATUS_NOT_SUPPORTED});
    EXPECT_TRUE(first.empty());

    fe::EngineConfigList unknown;
    EXPECT_EQ(
        fe::get_heuristics_list(
            std::vector<fe::HeurMode_t>{static_cast<fe::HeurMode_t>(99)}, graph, AllowAll, unknown),
        std::vector<cudnnStatus_t>{CUDNN_STATUS_BAD_PARAM});
    EXPECT_TRUE(unknown.empty());
}

} // namespace
