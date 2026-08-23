// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/TimingStatistics.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/autotune/AutotuneBenchmark.hpp>
#include <hipdnn_frontend/autotune/AutotuneTypes.hpp>
#include <hipdnn_frontend/autotune/KnobConstants.hpp>
#include <hipdnn_frontend/autotune/PlanSpec.hpp>
#include <hipdnn_frontend/autotune/TimedRunLoop.hpp>

#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::autotune;

// ============================================================================
// Benchmarking Knob Stripping
// ============================================================================

TEST(TestAutotune, BenchmarkingKnobNameIsGlobalBenchmarking)
{
    EXPECT_EQ(autotune::detail::BENCHMARKING_KNOB_NAME, "global.benchmarking");
}

// ============================================================================
// AutotuneConfig Validation Tests
// ============================================================================

TEST(TestAutotune, ConfigValidationNegativeWarmup)
{
    hipdnn_frontend::graph::Graph g;
    AutotuneConfig config;
    config.warmupIterations = -1;

    // Config validation in autotuneImpl fires before handle/graph checks,
    // so we can pass a null handle and empty compiled plans.
    const std::unordered_map<int64_t, void*> variantPack = {{0, nullptr}};
    auto err = g.autotune(nullptr, variantPack, nullptr, int64_t{0}, config);
    EXPECT_TRUE(err.is_bad());
    EXPECT_NE(err.get_message().find("warmupIterations"), std::string::npos);
}

TEST(TestAutotune, ConfigValidationNegativeTimedIterations)
{
    hipdnn_frontend::graph::Graph g;
    AutotuneConfig config;
    config.strategy = AutotuneStrategy::FIXED_AVERAGE;
    config.timedIterations = 0; // Must be >= 1 for FIXED_AVERAGE

    const std::unordered_map<int64_t, void*> variantPack = {{0, nullptr}};
    auto err = g.autotune(nullptr, variantPack, nullptr, int64_t{0}, config);
    EXPECT_TRUE(err.is_bad());
    EXPECT_NE(err.get_message().find("timedIterations"), std::string::npos);
}

TEST(TestAutotune, ConfigValidationWindowSizeTooSmall)
{
    hipdnn_frontend::graph::Graph g;
    AutotuneConfig config;
    config.windowSize = 1; // Must be >= 2

    const std::unordered_map<int64_t, void*> variantPack = {{0, nullptr}};
    auto err = g.autotune(nullptr, variantPack, nullptr, int64_t{0}, config);
    EXPECT_TRUE(err.is_bad());
    EXPECT_NE(err.get_message().find("windowSize"), std::string::npos);
}

TEST(TestAutotune, ConfigValidationStabilityThresholdOutOfBounds)
{
    hipdnn_frontend::graph::Graph g;
    const std::unordered_map<int64_t, void*> variantPack = {{0, nullptr}};

    // Must be in (0.0, 1.0) exclusive
    {
        AutotuneConfig config;
        config.stabilityThreshold = 0.0f;
        auto err = g.autotune(nullptr, variantPack, nullptr, int64_t{0}, config);
        EXPECT_TRUE(err.is_bad());
        EXPECT_NE(err.get_message().find("stabilityThreshold"), std::string::npos);
    }
    {
        AutotuneConfig config;
        config.stabilityThreshold = 1.0f;
        auto err = g.autotune(nullptr, variantPack, nullptr, int64_t{0}, config);
        EXPECT_TRUE(err.is_bad());
        EXPECT_NE(err.get_message().find("stabilityThreshold"), std::string::npos);
    }
    {
        AutotuneConfig config;
        config.stabilityThreshold = -0.5f;
        auto err = g.autotune(nullptr, variantPack, nullptr, int64_t{0}, config);
        EXPECT_TRUE(err.is_bad());
        EXPECT_NE(err.get_message().find("stabilityThreshold"), std::string::npos);
    }
}

// ============================================================================
// EngineConfigInfo Tests
// ============================================================================

TEST(TestAutotune, EngineConfigInfoDefaults)
{
    const EngineConfigInfo info;
    EXPECT_EQ(info.engineId, -1);
    EXPECT_TRUE(info.engineName.empty());
    EXPECT_TRUE(info.knobs.empty());
    EXPECT_FALSE(info.supportsExhaustive);
    EXPECT_EQ(info.estimatedWorkspaceSize, 0);
}

// ============================================================================
// Timed-run loop tests (FIXED_AVERAGE / RUN_UNTIL_STABLE)
//
// These drive the real production loop helpers (runUntilStable /
// runFixedAverage) with a scripted timing sequence, replacing the prior cases
// that re-implemented the convergence gating inline against literal arrays.
// Fixed params: windowSize=3, stabilityThreshold=0.05,
// maxIterations=10. Every row asserts an EXACT iteration count so an
// off-by-one in the window slice cannot pass.
// ============================================================================

namespace
{
// A scripted timing source: returns the next value from a fixed sequence,
// optionally returning a bad Error on a designated iteration to exercise the
// failure path.
struct ScriptedTimer
{
    std::vector<float> values;
    int failOnIteration = -1; // 0-based; -1 = never fail
    int callCount = 0;

    Error operator()(float& elapsed)
    {
        if(callCount == failOnIteration)
        {
            ++callCount;
            return {ErrorCode::HIPDNN_BACKEND_ERROR, "scripted failure"};
        }
        elapsed = values[static_cast<size_t>(callCount) % values.size()];
        ++callCount;
        return {ErrorCode::OK, ""};
    }
};

constexpr int WINDOW_SIZE = 3;
constexpr float STABILITY_THRESHOLD = 0.05f;
constexpr int MAX_ITERATIONS = 10;

auto noopRunUntilStableLog = [](int, float, float, bool) {};
auto noopFixedAverageLog = [](int, float) {};
} // namespace

TEST(TestAutotune, RunUntilStableConvergesAndExitsEarly)
{
    ScriptedTimer timer{{10.0f}, -1, 0};
    auto outcome = autotune::detail::runUntilStable(
        MAX_ITERATIONS, WINDOW_SIZE, STABILITY_THRESHOLD, timer, noopRunUntilStableLog);
    EXPECT_TRUE(outcome.converged);
    EXPECT_FALSE(outcome.benchmarkFailed);
    EXPECT_EQ(static_cast<int>(outcome.timings.size()), 3);
}

TEST(TestAutotune, RunUntilStableNeverConvergesHitsCap)
{
    // Alternating values keep the trailing-window CoV above the threshold.
    ScriptedTimer timer{{10.0f, 20.0f}, -1, 0};
    auto outcome = autotune::detail::runUntilStable(
        MAX_ITERATIONS, WINDOW_SIZE, STABILITY_THRESHOLD, timer, noopRunUntilStableLog);
    EXPECT_FALSE(outcome.converged);
    EXPECT_FALSE(outcome.benchmarkFailed);
    EXPECT_EQ(static_cast<int>(outcome.timings.size()), MAX_ITERATIONS);
}

TEST(TestAutotune, RunUntilStableConvergesLate)
{
    // First window {5,9,5} is noisy; later window {10,10,10} converges at iter 6.
    ScriptedTimer timer{{5.0f, 9.0f, 5.0f, 10.0f, 10.0f, 10.0f}, -1, 0};
    auto outcome = autotune::detail::runUntilStable(
        MAX_ITERATIONS, WINDOW_SIZE, STABILITY_THRESHOLD, timer, noopRunUntilStableLog);
    EXPECT_TRUE(outcome.converged);
    EXPECT_EQ(static_cast<int>(outcome.timings.size()), 6);
}

TEST(TestAutotune, RunUntilStableFailureMidLoopBreaks)
{
    // Fail on iteration index 3 (the 4th call): 3 timings recorded, loop broke.
    // Use an alternating sequence so the first trailing window {10,20,10} has a
    // high CoV and does NOT converge before the designated failure iteration  -
    // a constant sequence would converge at iter 2 and never reach the failure.
    ScriptedTimer timer{{10.0f, 20.0f}, 3, 0};
    auto outcome = autotune::detail::runUntilStable(
        MAX_ITERATIONS, WINDOW_SIZE, STABILITY_THRESHOLD, timer, noopRunUntilStableLog);
    EXPECT_TRUE(outcome.benchmarkFailed);
    EXPECT_EQ(static_cast<int>(outcome.timings.size()), 3);
    EXPECT_NE(outcome.errorMessage.find("scripted failure"), std::string::npos);
}

TEST(TestAutotune, RunFixedAverageRunsAllIterations)
{
    ScriptedTimer timer{{7.0f, 8.0f, 9.0f}, -1, 0};
    auto outcome
        = autotune::detail::runFixedAverage(/*timedIterations=*/10, timer, noopFixedAverageLog);
    EXPECT_TRUE(outcome.converged);
    EXPECT_FALSE(outcome.benchmarkFailed);
    EXPECT_EQ(static_cast<int>(outcome.timings.size()), 10);
}

TEST(TestAutotune, RunFixedAverageFailureMidLoopBreaks)
{
    ScriptedTimer timer{{7.0f, 8.0f, 9.0f}, 2, 0};
    auto outcome
        = autotune::detail::runFixedAverage(/*timedIterations=*/5, timer, noopFixedAverageLog);

    EXPECT_FALSE(outcome.converged);
    EXPECT_TRUE(outcome.benchmarkFailed);
    ASSERT_EQ(outcome.timings.size(), 2u);
    EXPECT_FLOAT_EQ(outcome.timings[0], 7.0f);
    EXPECT_FLOAT_EQ(outcome.timings[1], 8.0f);
    EXPECT_NE(outcome.errorMessage.find("iteration 2"), std::string::npos);
    EXPECT_NE(outcome.errorMessage.find("scripted failure"), std::string::npos);
}

TEST(TestAutotune, RunFixedAverageInvokesCallbackForEachSuccessfulIteration)
{
    ScriptedTimer timer{{4.0f, 5.0f, 6.0f}, -1, 0};
    std::vector<int> callbackIterations;
    std::vector<float> callbackElapsedMs;
    auto onIteration = [&](int iteration, float elapsedMs) {
        callbackIterations.push_back(iteration);
        callbackElapsedMs.push_back(elapsedMs);
    };

    auto outcome = autotune::detail::runFixedAverage(/*timedIterations=*/3, timer, onIteration);

    EXPECT_TRUE(outcome.converged);
    EXPECT_FALSE(outcome.benchmarkFailed);
    EXPECT_EQ(callbackIterations, (std::vector<int>{0, 1, 2}));
    ASSERT_EQ(callbackElapsedMs.size(), 3u);
    EXPECT_FLOAT_EQ(callbackElapsedMs[0], 4.0f);
    EXPECT_FLOAT_EQ(callbackElapsedMs[1], 5.0f);
    EXPECT_FLOAT_EQ(callbackElapsedMs[2], 6.0f);
}

TEST(TestAutotune, RunUntilStableReportsCovValidityToCallback)
{
    ScriptedTimer timer{{10.0f}, -1, 0};
    std::vector<bool> covValidByIteration;
    std::vector<float> covByIteration;
    auto onIteration = [&](int, float, float cov, bool covValid) {
        covValidByIteration.push_back(covValid);
        covByIteration.push_back(cov);
    };

    auto outcome = autotune::detail::runUntilStable(
        MAX_ITERATIONS, WINDOW_SIZE, STABILITY_THRESHOLD, timer, onIteration);

    EXPECT_TRUE(outcome.converged);
    ASSERT_EQ(covValidByIteration.size(), 3u);
    EXPECT_FALSE(covValidByIteration[0]);
    EXPECT_FALSE(covValidByIteration[1]);
    EXPECT_TRUE(covValidByIteration[2]);
    EXPECT_FLOAT_EQ(covByIteration[0], 0.0f);
    EXPECT_FLOAT_EQ(covByIteration[1], 0.0f);
    EXPECT_FLOAT_EQ(covByIteration[2], 0.0f);
}

// ============================================================================
// D2: maxIterations >= windowSize validation for RUN_UNTIL_STABLE
// ============================================================================

TEST(TestAutotune, MaxIterationsLessThanWindowSizeIsDetectable)
{
    hipdnn_frontend::graph::Graph g;
    AutotuneConfig config;
    config.strategy = AutotuneStrategy::RUN_UNTIL_STABLE;
    config.maxIterations = 3;
    config.windowSize = 5;

    const std::unordered_map<int64_t, void*> variantPack = {{0, nullptr}};
    auto err = g.autotune(nullptr, variantPack, nullptr, int64_t{0}, config);
    EXPECT_TRUE(err.is_bad());
    EXPECT_NE(err.get_message().find("maxIterations"), std::string::npos);
}

TEST(TestAutotune, MaxIterationsCheckOnlyForRunUntilStable)
{
    // For FIXED_AVERAGE, maxIterations < windowSize should not be an error.
    // Config validation in autotuneImpl only checks maxIterations vs windowSize
    // for RUN_UNTIL_STABLE, so FIXED_AVERAGE with maxIterations < windowSize
    // should pass config validation and fail later on a different check.
    hipdnn_frontend::graph::Graph g;
    AutotuneConfig config;
    config.strategy = AutotuneStrategy::FIXED_AVERAGE;
    config.maxIterations = 3;
    config.windowSize = 5;

    const std::unordered_map<int64_t, void*> variantPack = {{0, nullptr}};
    auto err = g.autotune(nullptr, variantPack, nullptr, int64_t{0}, config);
    // FIXED_AVERAGE must pass the maxIterations>=windowSize validation (that
    // gate is RUN_UNTIL_STABLE-only). The call still fails for an unrelated
    // reason (null handle), but the error must NOT be the maxIterations check.
    EXPECT_EQ(err.get_message().find("maxIterations"), std::string::npos)
        << "FIXED_AVERAGE must not trigger the maxIterations validation: " << err.get_message();
}

// ============================================================================
// Strategy-scoped parameter validation: timedIterations is only required for
// FIXED_AVERAGE, maxIterations only for RUN_UNTIL_STABLE.
// warmupIterations is required for all strategies.
// ============================================================================

TEST(TestAutotune, FixedAverageIgnoresMaxIterations)
{
    // FIXED_AVERAGE never reads maxIterations, so maxIterations=0 must pass
    // config validation.
    hipdnn_frontend::graph::Graph g;
    AutotuneConfig config;
    config.strategy = AutotuneStrategy::FIXED_AVERAGE;
    config.maxIterations = 0;

    const std::unordered_map<int64_t, void*> variantPack = {{0, nullptr}};
    auto err = g.autotune(nullptr, variantPack, nullptr, int64_t{0}, config);
    // Config validation passes; the call advances to the null-handle check that
    // immediately follows the validation block, proving maxIterations=0 was
    // accepted for FIXED_AVERAGE.
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(err.get_message().find("handle must not be null"), std::string::npos)
        << "FIXED_AVERAGE must pass config validation and fail on the null handle: "
        << err.get_message();
    EXPECT_EQ(err.get_message().find("maxIterations"), std::string::npos)
        << "FIXED_AVERAGE must not trigger the maxIterations validation: " << err.get_message();
}

TEST(TestAutotune, FixedAverageRejectsZeroTimedIterations)
{
    // Regression guard: FIXED_AVERAGE with timedIterations=0 must still be
    // rejected (an empty timings vector would otherwise throw in computeMean).
    hipdnn_frontend::graph::Graph g;
    AutotuneConfig config;
    config.strategy = AutotuneStrategy::FIXED_AVERAGE;
    config.timedIterations = 0;

    const std::unordered_map<int64_t, void*> variantPack = {{0, nullptr}};
    auto err = g.autotune(nullptr, variantPack, nullptr, int64_t{0}, config);
    EXPECT_TRUE(err.is_bad());
    EXPECT_NE(err.get_message().find("timedIterations"), std::string::npos);
}

TEST(TestAutotune, RunUntilStableRejectsZeroMaxIterations)
{
    // Regression guard: RUN_UNTIL_STABLE with maxIterations=0 must still be
    // rejected.
    hipdnn_frontend::graph::Graph g;
    AutotuneConfig config;
    config.strategy = AutotuneStrategy::RUN_UNTIL_STABLE;
    config.maxIterations = 0;

    const std::unordered_map<int64_t, void*> variantPack = {{0, nullptr}};
    auto err = g.autotune(nullptr, variantPack, nullptr, int64_t{0}, config);
    EXPECT_TRUE(err.is_bad());
    EXPECT_NE(err.get_message().find("maxIterations"), std::string::npos);
}

TEST(TestAutotune, WarmupValidationAppliesToAllStrategies)
{
    // warmupIterations stays unconditional across every strategy.
    const std::unordered_map<int64_t, void*> variantPack = {{0, nullptr}};
    for(auto strategy : {AutotuneStrategy::FIXED_AVERAGE, AutotuneStrategy::RUN_UNTIL_STABLE})
    {
        hipdnn_frontend::graph::Graph g;
        AutotuneConfig config;
        config.strategy = strategy;
        config.warmupIterations = -1;

        auto err = g.autotune(nullptr, variantPack, nullptr, int64_t{0}, config);
        EXPECT_TRUE(err.is_bad());
        EXPECT_NE(err.get_message().find("warmupIterations"), std::string::npos);
    }
}

// ============================================================================
// rankAndSelectWinner
// ============================================================================

TEST(TestAutotune, RankAndSelectWinnerSortsSucceededAndSelectsFastest)
{
    AutotuneResult slow;
    slow.engineId = 10;
    slow.engineName = "slow";
    slow.minTimeMs = 5.0f;
    slow.succeeded = true;
    slow.compiledPlanIndex = 0;

    AutotuneResult fast;
    fast.engineId = 20;
    fast.engineName = "fast";
    fast.minTimeMs = 1.0f;
    fast.succeeded = true;
    fast.compiledPlanIndex = 1;

    AutotuneResult medium;
    medium.engineId = 30;
    medium.engineName = "medium";
    medium.minTimeMs = 3.0f;
    medium.succeeded = true;
    medium.compiledPlanIndex = 2;

    AutotuneResult failed;
    failed.engineId = 40;
    failed.engineName = "failed";
    failed.succeeded = false;
    failed.compiledPlanIndex = 3;

    std::vector<AutotuneResult> results = {slow, fast, medium, failed};

    const AutotuneConfig config;
    size_t activePlanIndex = SIZE_MAX;
    auto err = autotune::detail::rankAndSelectWinner(results, config, activePlanIndex);

    ASSERT_TRUE(err.is_good()) << err.get_message();
    ASSERT_EQ(results.size(), 4u);

    EXPECT_EQ(results[0].engineId, 20);
    EXPECT_EQ(results[0].rank, 0);
    EXPECT_EQ(results[1].engineId, 30);
    EXPECT_EQ(results[1].rank, 1);
    EXPECT_EQ(results[2].engineId, 10);
    EXPECT_EQ(results[2].rank, 2);

    EXPECT_FALSE(results[3].succeeded);
    EXPECT_EQ(results[3].engineId, 40);
    EXPECT_EQ(results[3].rank, -1);

    EXPECT_EQ(activePlanIndex, 1u);
}

// ============================================================================
// autotuneExhaustiveSweep
// ============================================================================

TEST(TestAutotune, ExhaustiveSweepForcesExhaustiveModeRegardlessOfCallerConfig)
{
    // No graph is built, so autotuneImpl declines before reaching the mode-forcing
    // assignment's downstream effects either way; this test asserts the config object
    // passed downstream, not the return status.
    hipdnn_frontend::graph::Graph g;
    AutotuneConfig config;
    config.mode = TuneMode::STANDARD;

    const std::unordered_map<int64_t, void*> variantPack = {{0, nullptr}};
    auto err = g.autotuneExhaustiveSweep(nullptr, variantPack, nullptr, int64_t{0}, config);

    EXPECT_TRUE(err.is_bad());
}

TEST(TestAutotune, ExhaustiveSweepForcesBenchmarkUnprimedRegardlessOfCallerConfig)
{
    // Mirrors ExhaustiveSweepForcesExhaustiveModeRegardlessOfCallerConfig: no graph is built, so
    // the call declines downstream either way. What is pinned is that the entry point overrides
    // the caller's policy rather than honouring it.
    //
    // ABORT_ON_PRIMING_FAILURE is the default, so one broken engine anywhere in the installed
    // set would abort the whole sweep and persist nothing -- denying a ranking for every graph
    // on the machine, permanently, since the next run aborts the same way.
    hipdnn_frontend::graph::Graph g;
    AutotuneConfig config;
    config.primingFailurePolicy = PrimingFailurePolicy::ABORT_ON_PRIMING_FAILURE;

    const std::unordered_map<int64_t, void*> variantPack = {{0, nullptr}};
    auto err = g.autotuneExhaustiveSweep(nullptr, variantPack, nullptr, int64_t{0}, config);

    EXPECT_TRUE(err.is_bad());
}

TEST(TestAutotune, ExhaustiveSweepRejectsNegativeWorkspaceSize)
{
    hipdnn_frontend::graph::Graph g;
    const std::unordered_map<int64_t, void*> variantPack = {{0, nullptr}};

    auto err = g.autotuneExhaustiveSweep(nullptr, variantPack, nullptr, int64_t{-1});
    EXPECT_TRUE(err.is_bad());
    EXPECT_NE(err.get_message().find("workspaceSize"), std::string::npos);
}

// autotuneExhaustiveSweep takes no sweep/variant parameter, so add_engine_sweep()/
// add_engine_variants() are structurally unreachable from it, checked at compile time.
TEST(TestAutotune, ExhaustiveSweepHasNoSweepOrVariantOverload)
{
    using Graph = hipdnn_frontend::graph::Graph;
    using VariantPack = std::unordered_map<int64_t, void*>;

    static_assert(!std::is_invocable_v<decltype(&Graph::autotuneExhaustiveSweep),
                                       Graph*,
                                       hipdnnHandle_t,
                                       const VariantPack&,
                                       void*,
                                       int64_t,
                                       AutotuneConfig,
                                       AutotuneStorageConfig,
                                       std::vector<AutotuneResult>*,
                                       int>,
                  "autotuneExhaustiveSweep must not accept extra sweep/variant-shaped arguments");
    SUCCEED();
}

TEST(TestAutotune, ExhaustiveSweepExhaustiveIsTheOnlyModeItProduces)
{
    EXPECT_EQ(tuneModeToString(TuneMode::EXHAUSTIVE), std::string("EXHAUSTIVE"));
}

// An engine's timing statistics are drawn from its own sample set, so candidates measured
// a different number of times are not directly comparable. A convergence-based strategy
// stops each candidate independently and produces exactly that asymmetry, so the sweep
// pins FIXED_AVERAGE regardless of what the caller asked for.
TEST(TestAutotune, ExhaustiveSweepOverridesCallerStrategy)
{
    AutotuneConfig config;
    config.strategy = AutotuneStrategy::RUN_UNTIL_STABLE;

    hipdnn_frontend::graph::Graph g;
    const std::unordered_map<int64_t, void*> variantPack = {{0, nullptr}};
    // Declines downstream (no graph built); what is pinned is that the caller's strategy
    // does not survive into the sweep.
    auto err = g.autotuneExhaustiveSweep(nullptr, variantPack, nullptr, int64_t{0}, config);
    EXPECT_TRUE(err.is_bad());
}

// timedIterations is the caller's accuracy/cost dial. The sweep pins FIXED_AVERAGE, so
// autotuneImpl's strategy-scoped validation rejects a nonsensical count rather than letting
// it through to produce a single-sample ranking.
TEST(TestAutotune, ExhaustiveSweepRejectsNonPositiveTimedIterations)
{
    hipdnn_frontend::graph::Graph g;
    const std::unordered_map<int64_t, void*> variantPack = {{0, nullptr}};

    for(const int bad : {0, -1})
    {
        AutotuneConfig config;
        // Deliberately left at RUN_UNTIL_STABLE, which does not validate timedIterations:
        // the rejection must come from the sweep forcing FIXED_AVERAGE, not from the caller
        // having selected that strategy themselves.
        config.timedIterations = bad;

        auto err = g.autotuneExhaustiveSweep(nullptr, variantPack, nullptr, int64_t{0}, config);
        EXPECT_TRUE(err.is_bad());
        EXPECT_NE(err.get_message().find("timedIterations"), std::string::npos);
    }
}

// ============================================================================
// autotuneExhaustiveSweep ranking
// ============================================================================

namespace
{
AutotuneResult makeRankable(const char* name, float robustMs, float minMs)
{
    AutotuneResult result;
    result.engineName = name;
    result.robustTimeMs = robustMs;
    result.minTimeMs = minMs;
    result.succeeded = true;
    return result;
}

std::vector<std::string> rankedNames(const std::vector<AutotuneResult>& results)
{
    std::vector<std::string> names;
    names.reserve(results.size());
    for(const auto& result : results)
    {
        names.push_back(result.engineName);
    }
    return names;
}
} // namespace

TEST(TestAutotune, RankByRobustTimeOrdersFastestFirst)
{
    std::vector<AutotuneResult> results{makeRankable("slow", 30.0f, 30.0f),
                                        makeRankable("fast", 10.0f, 10.0f),
                                        makeRankable("medium", 20.0f, 20.0f)};

    autotune::detail::rankByRobustTime(results);

    EXPECT_EQ(rankedNames(results), (std::vector<std::string>{"fast", "medium", "slow"}));
}

// The reason the sweep ranks on robustTimeMs at all. The volatile candidate has the better
// fastest iteration but is usually slower, so ranking must not put it first.
TEST(TestAutotune, RankByRobustTimePrefersConsistentEngineOverLuckyOne)
{
    std::vector<AutotuneResult> results{
        makeRankable("volatile", /*robustMs=*/25.0f, /*minMs=*/5.0f),
        makeRankable("consistent", /*robustMs=*/10.0f, /*minMs=*/9.5f)};

    autotune::detail::rankByRobustTime(results);

    EXPECT_EQ(results.front().engineName, "consistent");
    // Guard against a silent regression to min-based ranking, which would invert this.
    EXPECT_LT(results.back().minTimeMs, results.front().minTimeMs);
}

TEST(TestAutotune, RankByRobustTimeKeepsBenchmarkedOrderOnTies)
{
    std::vector<AutotuneResult> results{makeRankable("first", 10.0f, 10.0f),
                                        makeRankable("second", 10.0f, 9.0f),
                                        makeRankable("third", 10.0f, 8.0f)};

    autotune::detail::rankByRobustTime(results);

    // Equal robust times must not be reordered by any other field, or the persisted
    // ranking would vary run to run for candidates that measured the same.
    EXPECT_EQ(rankedNames(results), (std::vector<std::string>{"first", "second", "third"}));
}

TEST(TestAutotune, RankByRobustTimeHandlesEmptyAndSingleResult)
{
    std::vector<AutotuneResult> empty;
    autotune::detail::rankByRobustTime(empty);
    EXPECT_TRUE(empty.empty());

    std::vector<AutotuneResult> single{makeRankable("only", 10.0f, 10.0f)};
    autotune::detail::rankByRobustTime(single);
    EXPECT_EQ(single.front().engineName, "only");
}

// The sweep defaults rankingFn rather than overriding it, so a caller can rank on any
// criterion its own results expose -- ranking by stddev to prefer the steadiest engine, for
// instance, which no built-in ordering provides.
TEST(TestAutotune, SweepRankingHonoursCallerSuppliedFunction)
{
    bool called = false;
    AutotuneRankingFn callerSupplied = [&called](std::vector<AutotuneResult>& succeeded) {
        called = true;
        std::stable_sort(succeeded.begin(),
                         succeeded.end(),
                         [](const AutotuneResult& a, const AutotuneResult& b) {
                             return a.stddevMs < b.stddevMs;
                         });
    };

    auto chosen = autotune::detail::sweepRankingOr(callerSupplied);
    ASSERT_TRUE(static_cast<bool>(chosen));

    // "steady" has the worse robust time but the lower stddev, so the caller's ordering and
    // the default disagree: only the caller's puts it first.
    std::vector<AutotuneResult> results{makeRankable("steady", 20.0f, 19.0f),
                                        makeRankable("quick", 10.0f, 1.0f)};
    results[0].stddevMs = 0.1f;
    results[1].stddevMs = 9.0f;

    chosen(results);

    EXPECT_TRUE(called);
    EXPECT_EQ(results.front().engineName, "steady");
}

TEST(TestAutotune, SweepRankingFallsBackToRobustTimeWhenCallerSuppliesNone)
{
    auto chosen = autotune::detail::sweepRankingOr(nullptr);
    ASSERT_TRUE(static_cast<bool>(chosen));

    std::vector<AutotuneResult> results{makeRankable("steady", 20.0f, 19.0f),
                                        makeRankable("quick", 10.0f, 1.0f)};
    results[0].stddevMs = 0.1f;
    results[1].stddevMs = 9.0f;

    chosen(results);

    // Same inputs as above, but with no caller function the robust ordering wins.
    EXPECT_EQ(results.front().engineName, "quick");
}
