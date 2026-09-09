// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Integration test for end-to-end autotune -> execute workflow.
// Uses the test_autotune_plugin which supports the autotune knob workflow.

#include <algorithm>
#include <cmath>
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
using IntegrationGpuTimedExecute = hipdnn_tests::AutotuneIntegrationFixture;

TEST_F(IntegrationGpuTimedExecute, ReportsTimingFromActivePlan)
{
    auto bundle = createConvGraph("timed_execute_conv");
    auto error = bundle.graph->build(_handle);
    ASSERT_TRUE(error.is_good()) << error.get_message();
    bundle.buildVariantPack();

    int64_t workspaceSize = 0;
    error = bundle.graph->get_workspace_size(workspaceSize);
    ASSERT_TRUE(error.is_good()) << error.get_message();
    const Workspace workspace(static_cast<size_t>(workspaceSize));

    // Warmup is an explicit caller action, not an extra execution hidden in the API.
    error = bundle.graph->execute(_handle, bundle.variantPack, workspace.get());
    ASSERT_TRUE(error.is_good()) << error.get_message();
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    ExecutionTiming timing;
    error = bundle.graph->execute_timed_ext(_handle, bundle.variantPack, workspace.get(), timing);
    ASSERT_TRUE(error.is_good()) << error.get_message();
    ASSERT_TRUE(timing.elapsedMs.has_value());
    EXPECT_TRUE(std::isfinite(*timing.elapsedMs));
    EXPECT_GT(*timing.elapsedMs, 0.0f);

    // Other tests may have disabled stalling after a watchdog timeout. Device support
    // alone does not prove that this call armed the gate; both valid modes are allowed.
    EXPECT_TRUE(timing.quality == TimingQuality::DEVICE_ONLY
                || timing.quality == TimingQuality::HOST_INCLUDED);
    GTEST_LOG_(INFO) << "timed execute: " << *timing.elapsedMs << " ms, "
                     << (timing.quality == TimingQuality::DEVICE_ONLY ? "device-only"
                                                                      : "host-included");
}

// Per-strategy GPU smoke tests. These assert only invariants that
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
        config.mode = TuneMode::STANDARD;
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

// Covers the maxIterations == windowSize accepted boundary for RUN_UNTIL_STABLE
// end-to-end through production code: the validation gate (maxIterations >=
// windowSize) must ACCEPT the equal case, and autotune must run to completion
// and return OK with results. runStrategySmoke hard-codes a different
// maxIterations, so this drives the same real path with the equal boundary.
TEST_F(IntegrationAutotuneStrategySmoke, RunUntilStableMaxEqualsWindow)
{
    ConvGraphBundle bundle;
    createBuiltConvGraph("autotune_max_equals_window_conv", bundle);

    auto result = bundle.graph->add_all_engines();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    int64_t maxWs = 0;
    result = bundle.graph->get_estimated_max_workspace_size(maxWs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const Workspace workspace(static_cast<size_t>(maxWs));

    AutotuneConfig config;
    config.mode = TuneMode::STANDARD;
    config.strategy = AutotuneStrategy::RUN_UNTIL_STABLE;
    config.warmupIterations = 1;
    config.windowSize = 3;
    config.maxIterations = 3; // equal boundary: maxIterations == windowSize
    config.timedIterations = 5;

    std::vector<AutotuneResult> results;
    result = bundle.graph->autotune(
        _handle, bundle.variantPack, workspace.get(), maxWs, config, {}, &results);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    assertAnySucceeded(results);
}

} // namespace
