// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Integration test for end-to-end autotune -> execute workflow.
// Uses the test_autotune_plugin which supports the autotune knob workflow.

#include <algorithm>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend.hpp>

#include "AutotuneIntegrationFixture.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;

namespace
{

using IntegrationAutotuneEndToEnd = hipdnn_tests::AutotuneIntegrationFixture;

// Test: build graph -> add_all_engines -> autotune (AUTO/SINGLE_SHOT) -> execute -> verify success
TEST_F(IntegrationAutotuneEndToEnd, AutotuneAutoSingleShotThenExecute)
{
    ConvGraphBundle bundle;
    createBuiltConvGraph("autotune_test_conv", bundle);

    auto result = bundle.graph->add_all_engines();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    int64_t maxWs = 0;
    result = bundle.graph->get_estimated_max_workspace_size(maxWs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    ASSERT_GE(maxWs, 0);

    const Workspace workspace(static_cast<size_t>(maxWs));

    AutotuneConfig config;
    config.mode = TuneMode::AUTO;
    config.strategy = AutotuneStrategy::SINGLE_SHOT;
    config.warmupIterations = 1;

    std::vector<AutotuneResult> results;
    result = bundle.graph->autotune(
        _handle, bundle.variantPack, workspace.get(), maxWs, config, {}, &results);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify at least one engine succeeded
    assertAnySucceeded(results);

    // Execute with the autotuned plan
    buildWorkspaceAndExecute(bundle);
}

// Test: build graph -> get_engine_configs -> filter by workspace -> add_engine_configs
//       -> autotune -> verify workspace constraint -> execute
TEST_F(IntegrationAutotuneEndToEnd, FilteredAutotuneWithWorkspaceConstraint)
{
    ConvGraphBundle bundle;
    createBuiltConvGraph("autotune_test_conv", bundle);

    // Step 1: Discover available engine configs
    std::vector<EngineConfigInfo> configs;
    auto result = bundle.graph->get_engine_configs(configs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    ASSERT_FALSE(configs.empty()) << "Expected at least one engine config";

    // Step 2: Filter configs by workspace size (256 MB limit)
    const int64_t workspaceLimit = int64_t{256} * 1024 * 1024;
    std::vector<EngineConfigInfo> filteredConfigs;
    for(const auto& cfg : configs)
    {
        if(cfg.estimatedWorkspaceSize <= workspaceLimit)
        {
            filteredConfigs.push_back(cfg);
        }
    }
    ASSERT_FALSE(filteredConfigs.empty())
        << "No engine configs within workspace limit of " << workspaceLimit << " bytes";

    // Step 3: Add only the filtered engines
    result = bundle.graph->add_engine_configs(filteredConfigs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Step 4: Allocate workspace (capped at the limit)
    int64_t maxWs = 0;
    result = bundle.graph->get_estimated_max_workspace_size(maxWs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    const int64_t allocatedWs = std::min(maxWs, workspaceLimit);
    const Workspace workspace(static_cast<size_t>(allocatedWs));

    // Step 5: Autotune with pre-filtered engines
    AutotuneConfig config;
    config.mode = TuneMode::AUTO;
    config.strategy = AutotuneStrategy::SINGLE_SHOT;
    config.warmupIterations = 1;

    std::vector<AutotuneResult> results;
    result = bundle.graph->autotune(
        _handle, bundle.variantPack, workspace.get(), allocatedWs, config, {}, &results);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Step 6: Verify all results respect the workspace constraint
    for(const auto& r : results)
    {
        EXPECT_LE(r.workspaceSize, workspaceLimit)
            << "Engine " << r.engineName << " (id=" << r.engineId
            << ") reported workspaceSize=" << r.workspaceSize
            << " exceeding workspace limit=" << workspaceLimit;
    }

    // Step 7: Execute with the autotuned plan
    buildWorkspaceAndExecute(bundle);
}

// F3-part3: per-strategy GPU smoke tests. These assert only invariants that
// hold regardless of measured time (proving the strategy is wired end-to-end
// on hardware); they deliberately do NOT assert convergence or an exact
// iteration count, which depend on un-steerable real hipEvent timings.
class IntegrationAutotuneStrategySmoke : public hipdnn_tests::AutotuneIntegrationFixture
{
protected:
    void runStrategySmoke(AutotuneStrategy strategy)
    {
        ConvGraphBundle bundle;
        createBuiltConvGraph("autotune_strategy_smoke_conv", bundle);

        auto result = bundle.graph->add_all_engines();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        int64_t maxWs = 0;
        result = bundle.graph->get_estimated_max_workspace_size(maxWs);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        const Workspace workspace(static_cast<size_t>(maxWs));

        AutotuneConfig config;
        config.mode = TuneMode::AUTO;
        config.strategy = strategy;
        config.warmupIterations = 1;
        config.windowSize = 3;
        config.maxIterations = 10;
        config.timedIterations = 5;

        std::vector<AutotuneResult> results;
        result = bundle.graph->autotune(
            _handle, bundle.variantPack, workspace.get(), maxWs, config, {}, &results);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        bool checkedAnySucceeded = false;
        for(const auto& r : results)
        {
            if(!r.succeeded)
            {
                continue;
            }
            checkedAnySucceeded = true;
            EXPECT_GE(r.iterationsRun, 1);
            EXPECT_LE(r.iterationsRun, config.maxIterations);
            EXPECT_GT(r.avgTimeMs, 0.0f);
            EXPECT_GE(r.stddevMs, 0.0f);
            EXPECT_LE(r.minTimeMs, r.avgTimeMs);
        }
        EXPECT_TRUE(checkedAnySucceeded) << "No engine succeeded for strategy smoke test";
    }
};

TEST_F(IntegrationAutotuneStrategySmoke, FixedAverage)
{
    runStrategySmoke(AutotuneStrategy::FIXED_AVERAGE);
}

TEST_F(IntegrationAutotuneStrategySmoke, RunUntilStable)
{
    runStrategySmoke(AutotuneStrategy::RUN_UNTIL_STABLE);
}

TEST_F(IntegrationAutotuneStrategySmoke, SingleShot)
{
    runStrategySmoke(AutotuneStrategy::SINGLE_SHOT);
}

} // namespace
