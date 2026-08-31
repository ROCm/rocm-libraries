// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
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
    // rejected (an empty timings vector would otherwise throw in mean()).
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
// AutotuneResult factories
//
// The trailing string parameters are same-typed, so each case passes a distinct
// self-identifying value and asserts the exact field it lands in.
// ============================================================================

namespace
{
constexpr int64_t FACTORY_ENGINE_ID = 4242;
constexpr int64_t ESTIMATED_WORKSPACE = 111;
constexpr int64_t COMPILED_WORKSPACE = 222;
constexpr const char* ENGINE_NAME_VALUE = "engine-name-value";
constexpr const char* REASON_VALUE = "reason-value";
constexpr const char* ERROR_MESSAGE_VALUE = "error-value";

std::vector<KnobSetting> factoryKnobSettings()
{
    return {KnobSetting("knob-id-value", int64_t{7})};
}

// Non-default mode and strategy so the pass-through of both is observable.
AutotuneConfig factoryConfig()
{
    AutotuneConfig config;
    config.mode = TuneMode::EXHAUSTIVE;
    config.strategy = AutotuneStrategy::FIXED_AVERAGE;
    return config;
}

// Fields every non-benchmarked factory sets identically.
void expectNonBenchmarkedCommonFields(const AutotuneResult& result)
{
    EXPECT_EQ(result.engineId, FACTORY_ENGINE_ID);
    EXPECT_EQ(result.engineName, ENGINE_NAME_VALUE);
    EXPECT_EQ(result.exhaustiveNotRunReason, REASON_VALUE);
    EXPECT_EQ(result.knobSettings, factoryKnobSettings());
    EXPECT_TRUE(result.supportsExhaustive);
    EXPECT_FALSE(result.ranExhaustive);
    EXPECT_FALSE(result.succeeded);
    EXPECT_EQ(result.rank, -1);
    EXPECT_EQ(result.compiledPlanIndex, -1);
    EXPECT_EQ(result.modeUsed, TuneMode::EXHAUSTIVE);
    EXPECT_EQ(result.strategyUsed, AutotuneStrategy::FIXED_AVERAGE);
}
} // namespace

TEST(TestAutotune, MakeBenchmarkResultAssignsEveryField)
{
    const auto result = autotune::detail::makeBenchmarkResult(FACTORY_ENGINE_ID,
                                                              factoryKnobSettings(),
                                                              ESTIMATED_WORKSPACE,
                                                              COMPILED_WORKSPACE,
                                                              factoryConfig(),
                                                              ENGINE_NAME_VALUE);

    EXPECT_EQ(result.engineId, FACTORY_ENGINE_ID);
    EXPECT_EQ(result.engineName, ENGINE_NAME_VALUE);
    EXPECT_EQ(result.knobSettings, factoryKnobSettings());
    EXPECT_EQ(result.estimatedWorkspaceSize, ESTIMATED_WORKSPACE);
    EXPECT_EQ(result.workspaceSize, COMPILED_WORKSPACE);
    EXPECT_EQ(result.modeUsed, TuneMode::EXHAUSTIVE);
    EXPECT_EQ(result.strategyUsed, AutotuneStrategy::FIXED_AVERAGE);
    EXPECT_TRUE(result.errorMessage.empty());
    EXPECT_TRUE(result.exhaustiveNotRunReason.empty());
}

TEST(TestAutotune, MakeNonBenchmarkedResultAssignsEveryField)
{
    const auto result = autotune::detail::makeNonBenchmarkedResult(FACTORY_ENGINE_ID,
                                                                   factoryKnobSettings(),
                                                                   ESTIMATED_WORKSPACE,
                                                                   COMPILED_WORKSPACE,
                                                                   factoryConfig(),
                                                                   ERROR_MESSAGE_VALUE,
                                                                   /*supportsExhaustive=*/true,
                                                                   /*ranExhaustive=*/false,
                                                                   REASON_VALUE,
                                                                   ENGINE_NAME_VALUE);

    expectNonBenchmarkedCommonFields(result);
    EXPECT_EQ(result.errorMessage, ERROR_MESSAGE_VALUE);
    EXPECT_EQ(result.estimatedWorkspaceSize, ESTIMATED_WORKSPACE);
    EXPECT_EQ(result.workspaceSize, COMPILED_WORKSPACE);
}

TEST(TestAutotune, MakeSkippedResultAssignsEveryField)
{
    constexpr int64_t MAX_WORKSPACE_SIZE = 128;
    const auto result = autotune::detail::makeSkippedResult(FACTORY_ENGINE_ID,
                                                            factoryKnobSettings(),
                                                            ESTIMATED_WORKSPACE,
                                                            COMPILED_WORKSPACE,
                                                            factoryConfig(),
                                                            MAX_WORKSPACE_SIZE,
                                                            /*supportsExhaustive=*/true,
                                                            /*ranExhaustive=*/false,
                                                            REASON_VALUE,
                                                            ENGINE_NAME_VALUE);

    expectNonBenchmarkedCommonFields(result);
    EXPECT_EQ(result.estimatedWorkspaceSize, ESTIMATED_WORKSPACE);
    EXPECT_EQ(result.workspaceSize, COMPILED_WORKSPACE);
    EXPECT_EQ(result.errorMessage, "Workspace size 222 exceeds limit 128");
}

TEST(TestAutotune, MakeBarredResultAssignsEveryField)
{
    const auto result = autotune::detail::makeBarredResult(FACTORY_ENGINE_ID,
                                                           factoryKnobSettings(),
                                                           ESTIMATED_WORKSPACE,
                                                           COMPILED_WORKSPACE,
                                                           factoryConfig(),
                                                           /*supportsExhaustive=*/true,
                                                           /*ranExhaustive=*/false,
                                                           REASON_VALUE,
                                                           ENGINE_NAME_VALUE);

    expectNonBenchmarkedCommonFields(result);
    EXPECT_EQ(result.estimatedWorkspaceSize, ESTIMATED_WORKSPACE);
    EXPECT_EQ(result.workspaceSize, COMPILED_WORKSPACE);
    EXPECT_EQ(result.errorMessage, "Plan barred (engine ID or workspace deselect filter).");
}

TEST(TestAutotune, MakeCompileFailedResultAssignsEveryField)
{
    const auto result = autotune::detail::makeCompileFailedResult(FACTORY_ENGINE_ID,
                                                                  factoryKnobSettings(),
                                                                  ESTIMATED_WORKSPACE,
                                                                  factoryConfig(),
                                                                  ERROR_MESSAGE_VALUE,
                                                                  /*supportsExhaustive=*/true,
                                                                  /*ranExhaustive=*/false,
                                                                  REASON_VALUE,
                                                                  ENGINE_NAME_VALUE);

    expectNonBenchmarkedCommonFields(result);
    EXPECT_EQ(result.errorMessage, ERROR_MESSAGE_VALUE);
    EXPECT_EQ(result.estimatedWorkspaceSize, ESTIMATED_WORKSPACE);
    EXPECT_EQ(result.workspaceSize, -1);
}

TEST(TestAutotune, MakeFinalizeFailedResultAssignsEveryField)
{
    const auto result = autotune::detail::makeFinalizeFailedResult(FACTORY_ENGINE_ID,
                                                                   factoryKnobSettings(),
                                                                   factoryConfig(),
                                                                   ERROR_MESSAGE_VALUE,
                                                                   /*supportsExhaustive=*/true,
                                                                   /*ranExhaustive=*/false,
                                                                   REASON_VALUE,
                                                                   ENGINE_NAME_VALUE);

    expectNonBenchmarkedCommonFields(result);
    EXPECT_EQ(result.errorMessage, ERROR_MESSAGE_VALUE);
    EXPECT_EQ(result.estimatedWorkspaceSize, -1);
    EXPECT_EQ(result.workspaceSize, -1);
}

TEST(TestAutotune, MakeFilteredResultAssignsEveryField)
{
    const auto result = autotune::detail::makeFilteredResult(FACTORY_ENGINE_ID,
                                                             factoryKnobSettings(),
                                                             ESTIMATED_WORKSPACE,
                                                             COMPILED_WORKSPACE,
                                                             factoryConfig(),
                                                             /*supportsExhaustive=*/true,
                                                             /*ranExhaustive=*/false,
                                                             REASON_VALUE,
                                                             ENGINE_NAME_VALUE);

    expectNonBenchmarkedCommonFields(result);
    EXPECT_EQ(result.estimatedWorkspaceSize, ESTIMATED_WORKSPACE);
    EXPECT_EQ(result.workspaceSize, COMPILED_WORKSPACE);
    EXPECT_EQ(result.errorMessage, "Plan excluded by engineIdFilter.");
}

TEST(TestAutotune, MakeBenchmarkResultResolvesAnOmittedEngineName)
{
    // Omitting the name is the arity a caller with no handle has. The registry
    // answers for an engine it carries, and the hexadecimal rendering for one it
    // does not, rather than the result carrying an empty name.
    const auto registered
        = autotune::detail::makeBenchmarkResult(hipdnn_data_sdk::utilities::MIOPEN_ENGINE_ID,
                                                factoryKnobSettings(),
                                                ESTIMATED_WORKSPACE,
                                                COMPILED_WORKSPACE,
                                                factoryConfig());
    EXPECT_EQ(registered.engineName, "MIOPEN_ENGINE");

    const auto unregistered = autotune::detail::makeBenchmarkResult(FACTORY_ENGINE_ID,
                                                                    factoryKnobSettings(),
                                                                    ESTIMATED_WORKSPACE,
                                                                    COMPILED_WORKSPACE,
                                                                    factoryConfig());
    EXPECT_EQ(unregistered.engineName, "0x0000000000001092");
}

TEST(TestAutotune, NonBenchmarkedFactoriesResolveAnOmittedEngineName)
{
    // The shared factory resolves the name, so every wrapper over it inherits the
    // behavior; makeFilteredResult stands in for the five.
    const auto shared
        = autotune::detail::makeNonBenchmarkedResult(hipdnn_data_sdk::utilities::MIOPEN_ENGINE_ID,
                                                     factoryKnobSettings(),
                                                     ESTIMATED_WORKSPACE,
                                                     COMPILED_WORKSPACE,
                                                     factoryConfig(),
                                                     ERROR_MESSAGE_VALUE,
                                                     /*supportsExhaustive=*/true,
                                                     /*ranExhaustive=*/false,
                                                     REASON_VALUE);
    EXPECT_EQ(shared.engineName, "MIOPEN_ENGINE");

    const auto wrapped = autotune::detail::makeFilteredResult(FACTORY_ENGINE_ID,
                                                              factoryKnobSettings(),
                                                              ESTIMATED_WORKSPACE,
                                                              COMPILED_WORKSPACE,
                                                              factoryConfig(),
                                                              /*supportsExhaustive=*/true,
                                                              /*ranExhaustive=*/false,
                                                              REASON_VALUE);
    EXPECT_EQ(wrapped.engineName, "0x0000000000001092");
}

// ============================================================================
// autotuneExhaustiveSweep
// ============================================================================

// mode and primingFailurePolicy are locked because they are what makes the call an
// exhaustive sweep that populates the cache. STANDARD would not prime any engine, and
// ABORT_ON_PRIMING_FAILURE would let one broken engine abandon the sweep and persist
// nothing -- denying a ranking for every graph on the machine, permanently, since the next
// run aborts the same way.
TEST(TestAutotune, SweepConfigLocksModeAndPrimingPolicy)
{
    AutotuneConfig requested;
    requested.mode = TuneMode::STANDARD;
    requested.primingFailurePolicy = PrimingFailurePolicy::ABORT_ON_PRIMING_FAILURE;

    const auto applied = autotune::detail::sweepConfigFrom(requested);

    EXPECT_EQ(applied.mode, TuneMode::EXHAUSTIVE);
    EXPECT_EQ(applied.primingFailurePolicy, PrimingFailurePolicy::BENCHMARK_UNPRIMED);
}

// Everything the two locked fields do not decide stays the caller's.
TEST(TestAutotune, SweepConfigPreservesEveryOtherField)
{
    AutotuneConfig requested;
    requested.strategy = AutotuneStrategy::FIXED_AVERAGE;
    requested.timedIterations = 37;
    requested.warmupIterations = 4;
    requested.maxIterations = 55;
    requested.windowSize = 6;
    requested.stabilityThreshold = 0.02f;
    requested.engineIdFilter = {11, 22};

    const auto applied = autotune::detail::sweepConfigFrom(requested);

    EXPECT_EQ(applied.strategy, AutotuneStrategy::FIXED_AVERAGE);
    EXPECT_EQ(applied.timedIterations, 37);
    EXPECT_EQ(applied.warmupIterations, 4);
    EXPECT_EQ(applied.maxIterations, 55);
    EXPECT_EQ(applied.windowSize, 6);
    EXPECT_FLOAT_EQ(applied.stabilityThreshold, 0.02f);
    EXPECT_EQ(applied.engineIdFilter, (std::vector<int64_t>{11, 22}));
}

// A default-constructed config must come back ready to sweep, so the common call is
// autotuneExhaustiveSweep(handle, variantPack, workspace, size) with nothing else supplied.
TEST(TestAutotune, SweepConfigFromDefaultsIsReadyToSweep)
{
    const auto applied = autotune::detail::sweepConfigFrom(AutotuneConfig{});

    EXPECT_EQ(applied.mode, TuneMode::EXHAUSTIVE);
    EXPECT_EQ(applied.primingFailurePolicy, PrimingFailurePolicy::BENCHMARK_UNPRIMED);
    EXPECT_TRUE(static_cast<bool>(applied.rankingFn));
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

// timedIterations is the caller's accuracy/cost dial, validated by autotuneImpl for the
// strategy that reads it. A count of zero would otherwise leave an empty sample set.
TEST(TestAutotune, ExhaustiveSweepRejectsNonPositiveTimedIterations)
{
    hipdnn_frontend::graph::Graph g;
    const std::unordered_map<int64_t, void*> variantPack = {{0, nullptr}};

    for(const int bad : {0, -1})
    {
        AutotuneConfig config;
        config.strategy = AutotuneStrategy::FIXED_AVERAGE;
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
    const AutotuneRankingFn callerSupplied = [&called](std::vector<AutotuneResult>& succeeded) {
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

// ============================================================================
// Sweep record plan: what a sweep persists, and when persisting is unsound
// ============================================================================
//
// The read path (ConfigBuiltIn::applyExactCacheEntry) derives candidates from the
// pre-compile applicability probe and then rejects an entry if any live candidate is
// missing from its sampled set, or if the stored order is not the same size as the
// candidate list. Both rejections are permanent in practice: a re-tune produces the same
// record, and put() returns UNCHANGED without appending. These tests pin the writer-side
// rules that keep a record appliable.

namespace
{

AutotuneResult makeSucceeded(int64_t engineId, float robustTimeMs)
{
    AutotuneResult r;
    r.engineId = engineId;
    r.succeeded = true;
    r.benchmarked = true;
    r.robustTimeMs = robustTimeMs;
    r.minTimeMs = robustTimeMs;
    return r;
}

/// Reached the timing loop and failed there: an intrinsic, recurring outcome.
AutotuneResult makeMeasuredFailure(int64_t engineId)
{
    AutotuneResult r;
    r.engineId = engineId;
    r.succeeded = false;
    r.benchmarked = true;
    return r;
}

/// Never reached the timing loop for a reason intrinsic to engine+graph, e.g. the plan
/// would not compile or finalize. Also recurring.
AutotuneResult makeIntrinsicMiss(int64_t engineId)
{
    AutotuneResult r;
    r.engineId = engineId;
    r.succeeded = false;
    r.benchmarked = false;
    r.excludedByCaller = false;
    return r;
}

/// Held out of the timing loop by a caller choice: engineIdFilter, a deselect filter, or
/// a workspace budget too small for the compiled plan. Not recurring -- the next call may
/// admit it.
AutotuneResult makeCallerExclusion(int64_t engineId)
{
    AutotuneResult r;
    r.engineId = engineId;
    r.succeeded = false;
    r.benchmarked = false;
    r.excludedByCaller = true;
    return r;
}

} // namespace

TEST(TestAutotuneSweepRecord, SucceededEnginesKeepTheirRankOrder)
{
    const std::vector<AutotuneResult> results{
        makeSucceeded(10, 1.0f), makeSucceeded(20, 2.0f), makeSucceeded(30, 3.0f)};

    const auto plan = autotune::detail::sweepRecordPlanFrom(results);

    EXPECT_TRUE(plan.persistable);
    EXPECT_EQ(plan.order, (std::vector<int64_t>{10, 20, 30}));
}

// An engine that is applicable but always fails is a live candidate on every later run.
// Omitting it from the record makes it a candidate the record cannot account for, which
// rejects the entry on every lookup, forever.
TEST(TestAutotuneSweepRecord, MeasuredFailuresAreRecordedAfterSurvivors)
{
    const std::vector<AutotuneResult> results{
        makeSucceeded(10, 1.0f), makeMeasuredFailure(99), makeSucceeded(20, 2.0f)};

    const auto plan = autotune::detail::sweepRecordPlanFrom(results);

    EXPECT_TRUE(plan.persistable);
    EXPECT_EQ(plan.order, (std::vector<int64_t>{10, 20, 99}));
}

// A plan that never reached the timing loop because it would not compile or finalize.
// The read path's candidate probe runs before any plan is built, so it reports this
// engine too, and the record must name it.
TEST(TestAutotuneSweepRecord, EnginesThatNeverBuiltAreStillRecorded)
{
    const std::vector<AutotuneResult> results{
        makeSucceeded(10, 1.0f), makeIntrinsicMiss(77), makeSucceeded(20, 2.0f)};

    const auto plan = autotune::detail::sweepRecordPlanFrom(results);

    EXPECT_TRUE(plan.persistable);
    EXPECT_EQ(plan.order, (std::vector<int64_t>{10, 20, 77}));
    EXPECT_NE(std::find(plan.order.begin(), plan.order.end(), int64_t{77}), plan.order.end())
        << "an engine that never compiled is still a live candidate at lookup time";
}

// Ordering among the three groups, in one case: survivors, then measured failures, then
// engines that never built. Everything unmeasured must lose to everything measured.
TEST(TestAutotuneSweepRecord, RecordOrdersSurvivorsThenMeasuredFailuresThenUnbuilt)
{
    const std::vector<AutotuneResult> results{makeIntrinsicMiss(77),
                                              makeMeasuredFailure(99),
                                              makeSucceeded(10, 1.0f),
                                              makeSucceeded(20, 2.0f)};

    const auto plan = autotune::detail::sweepRecordPlanFrom(results);

    EXPECT_EQ(plan.order, (std::vector<int64_t>{10, 20, 99, 77}));
}

// Results are one per plan spec, and specs are keyed on (engineId, knobSettings). Knob
// variants therefore produce several results for one engine. The record stores ids only,
// and the read path's order filter is non-consuming set membership, so a duplicate would
// survive it and make the order longer than the candidate set -- rejecting every lookup.
TEST(TestAutotuneSweepRecord, KnobVariantsCollapseToOneIdPerEngine)
{
    const std::vector<AutotuneResult> results{makeSucceeded(10, 1.0f),
                                              makeSucceeded(10, 2.0f),
                                              makeSucceeded(20, 3.0f),
                                              makeSucceeded(10, 4.0f)};

    const auto plan = autotune::detail::sweepRecordPlanFrom(results);

    EXPECT_TRUE(plan.persistable);
    EXPECT_EQ(plan.order, (std::vector<int64_t>{10, 20}));
    EXPECT_EQ(std::count(plan.order.begin(), plan.order.end(), int64_t{10}), 1)
        << "a duplicated engine id makes the stored order un-appliable";
}

// First occurrence wins because results arrive rank-ordered, so it is the engine's best
// showing. A last-wins collapse would rank an engine on its worst knob setting.
TEST(TestAutotuneSweepRecord, DuplicateCollapseKeepsTheBestRankedOccurrence)
{
    // Rank order: 20 (fastest), then 10's good variant, then 10's poor variant.
    const std::vector<AutotuneResult> results{
        makeSucceeded(20, 1.0f), makeSucceeded(10, 2.0f), makeSucceeded(10, 9.0f)};

    const auto plan = autotune::detail::sweepRecordPlanFrom(results);

    EXPECT_EQ(plan.order, (std::vector<int64_t>{20, 10}));
}

// A measured result and an unmeasured one for the same engine must not both land in the
// order; the measured one is the engine's showing.
TEST(TestAutotuneSweepRecord, EngineMeasuredOnOneSpecIsNotAlsoRecordedAsFailed)
{
    const std::vector<AutotuneResult> results{
        makeSucceeded(10, 1.0f), makeMeasuredFailure(10), makeSucceeded(20, 2.0f)};

    const auto plan = autotune::detail::sweepRecordPlanFrom(results);

    EXPECT_EQ(plan.order, (std::vector<int64_t>{10, 20}));
}

// The coverage gate. A caller-excluded engine may be a live candidate next run, so the
// sweep did not cover the set a lookup will see. Recording it would claim a measurement
// that never happened; omitting it makes every later lookup reject the entry.
TEST(TestAutotuneSweepRecord, CallerExcludedCandidateMakesTheSweepUnpersistable)
{
    const std::vector<AutotuneResult> results{
        makeSucceeded(10, 1.0f), makeSucceeded(20, 2.0f), makeCallerExclusion(30)};

    const auto plan = autotune::detail::sweepRecordPlanFrom(results);

    EXPECT_FALSE(plan.persistable)
        << "a filtered or workspace-barred candidate must not produce a persisted ranking";
    EXPECT_EQ(std::find(plan.order.begin(), plan.order.end(), int64_t{30}), plan.order.end())
        << "a never-measured engine must never enter the record's sampled set";
}

// Distinguishes the two unmeasured kinds: an intrinsic miss recurs and is recordable, a
// caller exclusion may not recur and is not.
TEST(TestAutotuneSweepRecord, IntrinsicMissPersistsWhereCallerExclusionDoesNot)
{
    const std::vector<AutotuneResult> intrinsic{makeSucceeded(10, 1.0f), makeIntrinsicMiss(30)};
    const std::vector<AutotuneResult> callerChosen{makeSucceeded(10, 1.0f),
                                                   makeCallerExclusion(30)};

    EXPECT_TRUE(autotune::detail::sweepRecordPlanFrom(intrinsic).persistable);
    EXPECT_FALSE(autotune::detail::sweepRecordPlanFrom(callerChosen).persistable);
}

// One engine, two specs: excluded on one, measured on the other. The record covers that
// id, so this is a full sweep for it. Guards against a naive any_of over the flag.
TEST(TestAutotuneSweepRecord, EngineExcludedOnOneSpecButMeasuredOnAnotherStaysPersistable)
{
    const std::vector<AutotuneResult> results{
        makeSucceeded(10, 1.0f), makeCallerExclusion(10), makeSucceeded(20, 2.0f)};

    const auto plan = autotune::detail::sweepRecordPlanFrom(results);

    EXPECT_TRUE(plan.persistable);
    EXPECT_EQ(plan.order, (std::vector<int64_t>{10, 20}));
}

// The default sweep -- no filters, workspace large enough, one spec per engine -- is the
// configuration the API documents, and it must persist.
TEST(TestAutotuneSweepRecord, FullSweepWithNoExclusionsIsPersistable)
{
    const std::vector<AutotuneResult> results{
        makeSucceeded(10, 1.0f), makeSucceeded(20, 2.0f), makeSucceeded(30, 3.0f)};

    EXPECT_TRUE(autotune::detail::sweepRecordPlanFrom(results).persistable);
}

TEST(TestAutotuneSweepRecord, EmptyResultsProduceAnEmptyPersistablePlan)
{
    const auto plan = autotune::detail::sweepRecordPlanFrom({});

    EXPECT_TRUE(plan.order.empty());
    EXPECT_TRUE(plan.persistable) << "nothing to decline when nothing was swept";
}

// The write outcome is how a caller learns a sweep was not persisted, so the new decline
// must be reportable and distinct from the pre-existing ones.
TEST(TestAutotuneSweepRecord, PartialSweepOutcomeHasADistinctName)
{
    EXPECT_STREQ(
        autotuneCacheWriteOutcomeToString(AutotuneCacheWriteOutcome::NOT_ATTEMPTED_PARTIAL_SWEEP),
        "NOT_ATTEMPTED_PARTIAL_SWEEP");
    EXPECT_STRNE(
        autotuneCacheWriteOutcomeToString(AutotuneCacheWriteOutcome::NOT_ATTEMPTED_PARTIAL_SWEEP),
        autotuneCacheWriteOutcomeToString(
            AutotuneCacheWriteOutcome::NOT_ATTEMPTED_NO_SUCCESSFUL_ENGINE));
}
