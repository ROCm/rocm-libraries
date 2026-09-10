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
#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/utilities/StallGate.hpp>
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

// Regression test for the mixed-timing-mode recovery in Graph::autotune()
// (Graph.hpp, the sweepStalled / sawDeviceOnly / sawHostIncluded loop). The
// test_autotune_plugin's AutotunePluginEngineHostSyncs engine calls
// hipStreamSynchronize on the plugin's own stream from inside executeOpGraph,
// so once the autotune stall gate has armed that stream for the timed
// measurement, the engine deadlocks itself and only the stall watchdog can
// release it. That is the exact scenario the recovery exists for: engines
// benchmarked before the timeout are DEVICE_ONLY, engines benchmarked after
// it (including the retried host-syncing one) are HOST_INCLUDED, and ranking
// the two populations against each other could hand the win to a slower
// engine.
//
// Before the recovery existed, this failed: the first sweep pass was kept
// as-is, so the earlier engines' DEVICE_ONLY times were ranked directly
// against the host-syncing engine's HOST_INCLUDED time instead of the whole
// pass being discarded and re-measured unstalled.
class IntegrationAutotuneStallRecovery : public hipdnn_tests::AutotuneIntegrationFixture
{
protected:
    // Mirrors TestProfilingControlDescriptor's stallGateAvailable(): a device
    // without hipStreamWaitValue32 support never arms, so the mixed-mode path
    // this test targets cannot occur and the smoke value stays outside its scope.
    static bool stallGateAvailable()
    {
        const hipdnn_data_sdk::utilities::StallGate gate;
        return gate.isUsable();
    }

    // libhipdnn_backend.so (the ProfilingControlDescriptor / StallGate that actually
    // arms and watches the stream) builds with CXX_VISIBILITY_PRESET hidden, so its
    // copy of StallGate's process-wide latch is private to that shared object:
    // resetStallingDisabledForTesting() called from this executable cannot reach
    // it, and once one run trips the watchdog it stays sticky for every later test
    // in this binary -- exactly the "no reset" problem
    // test_hip_event_bindings.py's HipStallGate child-process test documents for the
    // same class. Isolating each strategy in a freshly exec'd child, instead of a
    // fork, guarantees a never-armed gate for that strategy regardless of test order,
    // and keeps this test's self-inflicted trip from silently disabling stalling for
    // every later test in this binary. The child is this same test, selected by exact
    // name, so its own EXPECT/ASSERT failures are the ones that fail the parent.
    void runIsolated(const char* testId, AutotuneStrategy strategy)
    {
        if(std::getenv("HIPDNN_STALL_RECOVERY_CHILD") == nullptr)
        {
            if(!stallGateAvailable())
            {
                GTEST_SKIP() << "Device does not support hipStreamWaitValue32";
            }
            // /proc/self/exe and GetModuleFileName both give this process's own
            // executable path, which is the only reliable way to re-exec exactly this
            // test binary regardless of where CTest or the CI runner placed it.
#if defined(_WIN32)
            std::string selfPath(MAX_PATH, '\0');
            const DWORD n
                = GetModuleFileNameA(nullptr, selfPath.data(), static_cast<DWORD>(selfPath.size()));
            ASSERT_GT(n, 0U) << "could not resolve this test binary's own path via "
                                "GetModuleFileName";
            selfPath.resize(n);
#else
            std::string selfPath(4096, '\0');
            const ssize_t n = readlink("/proc/self/exe", selfPath.data(), selfPath.size() - 1);
            ASSERT_GT(n, 0) << "could not resolve this test binary's own path via "
                               "/proc/self/exe";
            selfPath.resize(static_cast<size_t>(n));
#endif

            // Set in this process, not via shell "VAR=1 cmd" prefix syntax: that syntax
            // is POSIX-shell-only and cmd.exe does not support it. setenv/_putenv_s is
            // inherited by the child regardless of which shell std::system() invokes.
#if defined(_WIN32)
            _putenv_s("HIPDNN_STALL_RECOVERY_CHILD", "1");
#else
            setenv("HIPDNN_STALL_RECOVERY_CHILD", "1", 1);
#endif
            const std::string command = "\"" + selfPath + "\" --gtest_filter=" + testId;
            const int rc = std::system(command.c_str());

            // setenv/_putenv_s changed THIS process's own environment, not only the
            // spawned child's: unset it now, or the next IntegrationAutotuneStallRecovery
            // test in this same binary would see it already set, skip its own re-exec,
            // and run unisolated in this process instead of a fresh one.
#if defined(_WIN32)
            _putenv_s("HIPDNN_STALL_RECOVERY_CHILD", "");
#else
            unsetenv("HIPDNN_STALL_RECOVERY_CHILD");
#endif

            ASSERT_EQ(rc, 0) << "isolated child run of " << testId
                             << " failed (see its gtest output above)";
            return;
        }

        runMixedTimingModeRecovery(strategy);
    }

    void runMixedTimingModeRecovery(AutotuneStrategy strategy)
    {
        hipdnn_data_sdk::utilities::StallGate::resetStallingDisabledForTesting();
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
               "run did not exercise the mixed-timing-mode recovery";

        // Every succeeded engine must carry the SAME timing quality: a discarded,
        // re-measured sweep is uniformly HOST_INCLUDED, never a mix of
        // DEVICE_ONLY and HOST_INCLUDED results.
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
        EXPECT_EQ(*commonQuality, TimingQuality::HOST_INCLUDED)
            << "results were not re-measured unstalled after the mixed pass";

        // The winner must come from that same uniformly-measured, re-run pass.
        const auto winner
            = std::find_if(results.begin(), results.end(), [](const AutotuneResult& r) {
                  return r.succeeded && r.rank == 0;
              });
        ASSERT_NE(winner, results.end()) << "No winner selected";
        EXPECT_EQ(winner->timingQuality, TimingQuality::HOST_INCLUDED)
            << "winner was not one of the uniformly-measured results";
    }
};

TEST_F(IntegrationAutotuneStallRecovery, FixedAverageDiscardsMixedTimingSweep)
{
    runIsolated("IntegrationAutotuneStallRecovery.FixedAverageDiscardsMixedTimingSweep",
                AutotuneStrategy::FIXED_AVERAGE);
}

TEST_F(IntegrationAutotuneStallRecovery, RunUntilStableDiscardsMixedTimingSweep)
{
    runIsolated("IntegrationAutotuneStallRecovery.RunUntilStableDiscardsMixedTimingSweep",
                AutotuneStrategy::RUN_UNTIL_STABLE);
}

} // namespace
