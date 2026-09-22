// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Integration test for end-to-end autotune -> execute workflow.
// Uses the test_autotune_plugin which supports the autotune knob workflow.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#define HIPDNN_UNDEF_NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#define HIPDNN_UNDEF_WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#ifdef HIPDNN_UNDEF_NOMINMAX
#undef NOMINMAX
#undef HIPDNN_UNDEF_NOMINMAX
#endif
#ifdef HIPDNN_UNDEF_WIN32_LEAN_AND_MEAN
#undef WIN32_LEAN_AND_MEAN
#undef HIPDNN_UNDEF_WIN32_LEAN_AND_MEAN
#endif
#else
#include <unistd.h>
#endif

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/StallGate.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

#include "AutotuneIntegrationFixture.hpp"
#include "test_plugins/TestPluginEngineIdMap.hpp"

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
                || timing.quality == TimingQuality::UNSTALLED);
    GTEST_LOG_(INFO) << "timed execute: " << *timing.elapsedMs << " ms, "
                     << (timing.quality == TimingQuality::DEVICE_ONLY ? "device-only"
                                                                      : "unstalled");
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

// Regression test for the comparison-local stall fallback in Graph::autotune()
// (Graph.hpp's sweepStalled loop + TimedRunLoop's restartUnstalled signal). The
// test_autotune_plugin's AutotunePluginEngineHostSyncs engine calls
// hipStreamSynchronize on the plugin's own stream from inside executeOpGraph, so once
// the autotune stall gate has armed that stream for its first timed measurement, the
// engine deadlocks itself and only the stall watchdog can release it -- a real,
// self-inflicted stall trip exercised end-to-end.
//
// There is no cross-call or cross-comparison latch any more (StallGate reports only
// per-attempt arm/release/timedOut): a timeout must cost at most one stalled attempt at
// this sweep layer, the whole comparison reruns unstalled exactly once, and a later,
// independent autotune() call must still be able to arm the stall gate for engines that
// do not deadlock.
class IntegrationAutotuneStallRecovery : public hipdnn_tests::AutotuneIntegrationFixture
{
protected:
    // This test requires a usable stall gate. Fail constructor errors loudly; skip only
    // when the capability query succeeds and reports no support.
    static bool stallGateAvailable()
    {
        const hipdnn_data_sdk::utilities::StallGate gate;
        if(gate.isUsable())
        {
            return true;
        }
        EXPECT_EQ(gate.lastError(), hipSuccess)
            << "StallGate construction failed: " << gate.lastOperation() << " returned "
            << hipGetErrorString(gate.lastError());
        return false;
    }

    // Engine discovery is cached across handles. Enable the test-only engine before
    // startup in a fresh process, not by changing the environment after discovery.
    template <typename Body>
    void runIsolated(Body&& body)
    {
        if(!hipdnn_data_sdk::utilities::getEnv("HIPDNN_STALL_RECOVERY_CHILD").empty())
        {
            body();
            return;
        }
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        ASSERT_NE(info, nullptr);
        const std::string testId = std::string(info->test_suite_name()) + "." + info->name();
#if defined(_WIN32)
        std::string selfPath(MAX_PATH, '\0');
        const DWORD length
            = GetModuleFileNameA(nullptr, selfPath.data(), static_cast<DWORD>(selfPath.size()));
        ASSERT_GT(length, 0U);
        ASSERT_LT(length, selfPath.size());
        selfPath.resize(length);
#else
        std::string selfPath(4096, '\0');
        const auto length = readlink("/proc/self/exe", selfPath.data(), selfPath.size() - 1);
        ASSERT_GT(length, 0);
        ASSERT_LT(static_cast<size_t>(length), selfPath.size() - 1);
        selfPath.resize(static_cast<size_t>(length));
#endif
        const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter child(
            "HIPDNN_STALL_RECOVERY_CHILD", "1");
        const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter hostSync(
            "HIPDNN_TEST_AUTOTUNE_HOST_SYNC_ENGINE", "1");
        const std::string command = "\"" + selfPath + "\" --gtest_filter=" + testId;
        ASSERT_EQ(std::system(command.c_str()), 0) << "recovery child failed: " << testId;
    }

    // A timeout must stop the comparison and cause one complete unstalled rerun.
    void runMixedTimingModeRecovery(AutotuneStrategy strategy)
    {
        if(!stallGateAvailable())
        {
            GTEST_SKIP() << "Device does not support hipStreamWaitValue32";
        }

        ConvGraphBundle bundle;
        createBuiltConvGraph("autotune_stall_recovery_conv", bundle);

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
        config.maxIterations = 5;
        config.timedIterations = 3;

        std::vector<AutotuneResult> results;
        const auto start = std::chrono::steady_clock::now();
        result = bundle.graph->autotune(
            _handle, bundle.variantPack, workspace.get(), maxWs, config, {}, &results);
        const auto elapsed = std::chrono::steady_clock::now() - start;
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        // Proves this run genuinely deadlocked and was released by the watchdog,
        // rather than passing because nothing was ever stalled: the host-syncing
        // engine can only complete once StallGate::DEFAULT_TIMEOUT (2 s) elapses and
        // the watchdog writes the release signal itself. A run that took the normal
        // sub-second path for this tiny graph never hit that wait.
        constexpr auto MIN_EXPECTED_STALL = std::chrono::milliseconds(1500);
        EXPECT_GE(elapsed, MIN_EXPECTED_STALL)
            << "run finished in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
            << " ms; the stall watchdog's ~2s timeout was never waited out, so this "
               "run did not exercise the comparison-local stall fallback";

        // Proves the timeout cost at most ONE stalled attempt at this sweep layer: the
        // whole comparison breaks and reruns unstalled immediately, instead of letting
        // every remaining candidate or iteration of the host-syncing engine retry into
        // its own fresh watchdog wait. Two full timeouts would clear this bound.
        constexpr auto MAX_EXPECTED_STALL = std::chrono::milliseconds(3500);
        EXPECT_LT(elapsed, MAX_EXPECTED_STALL)
            << "run took " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
            << " ms; a timeout must cost at most one stalled attempt at this sweep layer";

        // Every succeeded engine must carry the SAME timing quality: a discarded,
        // re-measured sweep is uniformly UNSTALLED, never a mix of
        // DEVICE_ONLY and UNSTALLED results.
        std::optional<TimingQuality> commonQuality;
        bool checkedAnySucceeded = false;
        for(const auto& r : results)
        {
            if(!r.succeeded)
            {
                continue;
            }
            checkedAnySucceeded = true;
            if(!commonQuality.has_value())
            {
                commonQuality = r.timingQuality;
            }
            EXPECT_EQ(r.timingQuality, *commonQuality)
                << "engine " << r.engineName
                << " was ranked with a timing quality that differs from the rest of "
                   "the pass -- the mixed sweep was not discarded";
        }
        ASSERT_TRUE(checkedAnySucceeded) << "No engine succeeded during autotune";
        ASSERT_TRUE(commonQuality.has_value());
        EXPECT_EQ(*commonQuality, TimingQuality::UNSTALLED)
            << "results were not re-measured unstalled after the mixed pass";

        // The winner must come from that same uniformly-measured, re-run pass.
        const auto winner
            = std::find_if(results.begin(), results.end(), [](const AutotuneResult& r) {
                  return r.succeeded && r.rank == 0;
              });
        ASSERT_NE(winner, results.end()) << "No winner selected";
        EXPECT_EQ(winner->timingQuality, TimingQuality::UNSTALLED)
            << "winner was not one of the uniformly-measured results";
    }
};

TEST_F(IntegrationAutotuneStallRecovery, FixedAverageDiscardsMixedTimingSweep)
{
    runIsolated([this] { runMixedTimingModeRecovery(AutotuneStrategy::FIXED_AVERAGE); });
}

TEST_F(IntegrationAutotuneStallRecovery, RunUntilStableDiscardsMixedTimingSweep)
{
    runIsolated([this] { runMixedTimingModeRecovery(AutotuneStrategy::RUN_UNTIL_STABLE); });
}

// A later comparison on healthy engines must remain eligible for device-only timing.
// Filter out the deliberately blocking engine rather than changing discovery mid-handle.
TEST_F(IntegrationAutotuneStallRecovery, LaterIndependentCallIsEligibleToStallAgain)
{
    runIsolated([this] {
        if(!stallGateAvailable())
        {
            GTEST_SKIP() << "Device does not support hipStreamWaitValue32";
        }

        runMixedTimingModeRecovery(AutotuneStrategy::FIXED_AVERAGE);

        ConvGraphBundle bundle;
        createBuiltConvGraph("autotune_stall_recovery_followup_conv", bundle);

        auto result = bundle.graph->add_all_engines();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        int64_t maxWs = 0;
        result = bundle.graph->get_estimated_max_workspace_size(maxWs);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        const Workspace workspace(static_cast<size_t>(maxWs));

        AutotuneConfig config;
        config.mode = TuneMode::STANDARD;
        config.warmupIterations = 1;
        config.timedIterations = 3;
        config.engineIdFilter = {hipdnn_tests::plugin_constants::engineId<AutotunePlugin>(),
                                 hipdnn_tests::plugin_constants::engineId<AutotunePluginEngineB>()};

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
            EXPECT_EQ(r.timingQuality, TimingQuality::DEVICE_ONLY)
                << "engine " << r.engineName
                << " was not measured device-only; an earlier call's timeout must not "
                   "disable stalling for this independent comparison";
        }
        EXPECT_TRUE(checkedAnySucceeded) << "No engine succeeded for the follow-up autotune call";
    });
}

} // namespace
