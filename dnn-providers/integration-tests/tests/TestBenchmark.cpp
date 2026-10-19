// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

#include "BundleFixtureFiles.hpp"
#include "HarnessTestSupport.hpp"
#include "harness/Benchmark.hpp"

// NOLINTBEGIN(readability-identifier-naming)

using namespace hipdnn_integration_tests;
using namespace hipdnn_integration_tests::bundle;
using hipdnn_frontend::Error;
using hipdnn_frontend::ErrorCode;
using hipdnn_frontend::ExecutionTiming;
using hipdnn_frontend::TimingQuality;

namespace
{

Error ok()
{
    return {ErrorCode::OK, ""};
}

Error broken(const std::string& message)
{
    return {ErrorCode::HIPDNN_BACKEND_ERROR, message};
}

ExecutionTiming measured(float ms, TimingQuality quality = TimingQuality::DEVICE_ONLY)
{
    ExecutionTiming timing;
    timing.elapsedMs = ms;
    timing.quality = quality;
    return timing;
}

ExecutionTiming watchdogTimeout()
{
    ExecutionTiming timing;
    timing.timedOut = true;
    return timing;
}

/// A timer that replays `script` in order, then repeats its last entry.
struct ScriptedTimer
{
    std::vector<ExecutionTiming> script;
    int calls = 0;

    Error operator()(ExecutionTiming& timing)
    {
        const auto index
            = std::min<std::size_t>(static_cast<std::size_t>(calls), script.size() - 1);
        timing = script[index];
        ++calls;
        return ok();
    }
};

BenchmarkSettings settings(int timed, int warmup)
{
    BenchmarkSettings result;
    result.timedIterations = timed;
    result.warmupIterations = warmup;
    return result;
}

BenchmarkResult successfulResult(std::vector<float> samples)
{
    BenchmarkResult result;
    result.engine = "TEST_ENGINE";
    result.samplesMs = std::move(samples);
    result.quality = TimingQuality::DEVICE_ONLY;
    return result;
}

} // namespace

// ---- runBenchmarkLoop ------------------------------------------------------

TEST(TestBenchmarkLoop, RunsWarmupThenCollectsTheRequestedSamples)
{
    int warmups = 0;
    ScriptedTimer timer{{measured(3.0f), measured(1.0f), measured(2.0f)}};

    const auto result = runBenchmarkLoop(
        settings(3, 2),
        [&] {
            ++warmups;
            EXPECT_EQ(timer.calls, 0) << "warmup must finish before timing starts";
            return ok();
        },
        std::ref(timer));

    ASSERT_TRUE(result.ok()) << result.error;
    EXPECT_EQ(warmups, 2);
    EXPECT_EQ(timer.calls, 3);
    EXPECT_EQ(result.samplesMs, (std::vector<float>{3.0f, 1.0f, 2.0f}));
    EXPECT_EQ(result.quality, TimingQuality::DEVICE_ONLY);
    EXPECT_EQ(result.discardedSamples, 0);
}

TEST(TestBenchmarkLoop, ZeroWarmupStartsTimingImmediately)
{
    int warmups = 0;
    ScriptedTimer timer{{measured(1.0f)}};

    const auto result = runBenchmarkLoop(
        settings(1, 0),
        [&] {
            ++warmups;
            return ok();
        },
        std::ref(timer));

    ASSERT_TRUE(result.ok()) << result.error;
    EXPECT_EQ(warmups, 0);
    EXPECT_EQ(timer.calls, 1);
}

TEST(TestBenchmarkLoop, OneUnstalledSampleMarksTheWholeSetUnstalled)
{
    ScriptedTimer timer{{measured(1.0f), measured(1.5f, TimingQuality::UNSTALLED), measured(1.0f)}};

    const auto result = runBenchmarkLoop(settings(3, 0), [] { return ok(); }, std::ref(timer));

    ASSERT_TRUE(result.ok()) << result.error;
    EXPECT_EQ(result.quality, TimingQuality::UNSTALLED);
}

TEST(TestBenchmarkLoop, RunsWithoutAMeasurementAreRetriedAndCounted)
{
    ScriptedTimer timer{{watchdogTimeout(), measured(2.0f), ExecutionTiming{}, measured(4.0f)}};

    const auto result = runBenchmarkLoop(settings(2, 0), [] { return ok(); }, std::ref(timer));

    ASSERT_TRUE(result.ok()) << result.error;
    EXPECT_EQ(timer.calls, 4);
    EXPECT_EQ(result.samplesMs, (std::vector<float>{2.0f, 4.0f}));
    EXPECT_EQ(result.discardedSamples, 2);
    EXPECT_EQ(result.quality, TimingQuality::DEVICE_ONLY);
}

TEST(TestBenchmarkLoop, TooFewMeasurementsFailAfterABoundedNumberOfAttempts)
{
    ScriptedTimer timer{{measured(1.0f), watchdogTimeout()}};

    const auto result = runBenchmarkLoop(settings(5, 0), [] { return ok(); }, std::ref(timer));

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(timer.calls, 10) << "retries are capped at timedIterations extra attempts";
    EXPECT_TRUE(result.samplesMs.empty()) << "a partial set must not be reported as a result";
    EXPECT_NE(result.error.find("9 of 10 timed executions returned no measurement"),
              std::string::npos)
        << result.error;
    EXPECT_NE(result.error.find("collected 1 of 5 samples"), std::string::npos) << result.error;
}

TEST(TestBenchmarkLoop, WarmupFailureStopsBeforeAnyTimedRun)
{
    ScriptedTimer timer{{measured(1.0f)}};

    const auto result
        = runBenchmarkLoop(settings(3, 2), [] { return broken("bad kernel"); }, std::ref(timer));

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(timer.calls, 0);
    EXPECT_EQ(result.error, "warmup execution failed: bad kernel");
}

TEST(TestBenchmarkLoop, TimedFailureFailsAndDropsEarlierSamples)
{
    int calls = 0;
    const auto result = runBenchmarkLoop(
        settings(3, 0),
        [] { return ok(); },
        [&](ExecutionTiming& timing) {
            if(++calls == 2)
            {
                return broken("device lost");
            }
            timing = measured(1.0f);
            return ok();
        });

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(calls, 2);
    EXPECT_TRUE(result.samplesMs.empty());
    EXPECT_EQ(result.error, "timed execution failed: device lost");
}

// ---- BenchmarkResult statistics -------------------------------------------

TEST(TestBenchmarkResult, StatisticsOfAnOddSampleCount)
{
    const auto result = successfulResult({5.0f, 1.0f, 3.0f});

    EXPECT_FLOAT_EQ(result.minMs(), 1.0f);
    EXPECT_FLOAT_EQ(result.medianMs(), 3.0f);
    EXPECT_FLOAT_EQ(result.meanMs(), 3.0f);
}

TEST(TestBenchmarkResult, MedianOfAnEvenSampleCountAveragesTheMiddlePair)
{
    const auto result = successfulResult({4.0f, 1.0f, 3.0f, 2.0f});

    EXPECT_FLOAT_EQ(result.medianMs(), 2.5f);
    EXPECT_FLOAT_EQ(result.meanMs(), 2.5f);
}

TEST(TestBenchmarkResult, AResultWithoutSamplesIsNotOk)
{
    EXPECT_FALSE(BenchmarkResult{}.ok());
}

// ---- resolveBenchmark -----------------------------------------------------

TEST(TestResolveBenchmark, OffWithoutFlagsHasNoSettingsAndNoError)
{
    const auto resolution = resolveBenchmark({});

    EXPECT_FALSE(resolution.settings.has_value());
    EXPECT_FALSE(resolution.error.has_value());
}

TEST(TestResolveBenchmark, EnabledAloneUsesTheDefaults)
{
    BenchmarkRequest request;
    request.enabled = true;

    const auto resolution = resolveBenchmark(request);

    ASSERT_TRUE(resolution.settings.has_value()) << resolution.error.value_or("");
    EXPECT_EQ(resolution.settings->timedIterations, 20);
    EXPECT_EQ(resolution.settings->warmupIterations, 3);
}

TEST(TestResolveBenchmark, ExplicitCountsOverrideTheDefaults)
{
    BenchmarkRequest request;
    request.enabled = true;
    request.timedIterations = 7;
    request.warmupIterations = 0;

    const auto resolution = resolveBenchmark(request);

    ASSERT_TRUE(resolution.settings.has_value()) << resolution.error.value_or("");
    EXPECT_EQ(resolution.settings->timedIterations, 7);
    EXPECT_EQ(resolution.settings->warmupIterations, 0);
}

TEST(TestResolveBenchmark, CountsWithoutBenchmarkAreRefused)
{
    BenchmarkRequest iterations;
    iterations.timedIterations = 5;
    BenchmarkRequest warmup;
    warmup.warmupIterations = 1;

    for(const auto& request : {iterations, warmup})
    {
        const auto resolution = resolveBenchmark(request);
        EXPECT_FALSE(resolution.settings.has_value());
        ASSERT_TRUE(resolution.error.has_value());
        EXPECT_NE(resolution.error->find("require --benchmark"), std::string::npos);
    }
}

TEST(TestResolveBenchmark, OutOfRangeCountsAreRefused)
{
    BenchmarkRequest noIterations;
    noIterations.enabled = true;
    noIterations.timedIterations = 0;
    BenchmarkRequest negativeWarmup;
    negativeWarmup.enabled = true;
    negativeWarmup.warmupIterations = -1;

    EXPECT_EQ(resolveBenchmark(noIterations).error,
              std::optional<std::string>("--benchmark-iterations must be at least 1"));
    EXPECT_EQ(resolveBenchmark(negativeWarmup).error,
              std::optional<std::string>("--benchmark-warmup must not be negative"));
}

TEST(TestResolveBenchmark, ModesThatDoNotExecuteAreRefused)
{
    BenchmarkRequest skip;
    skip.skipGraphValidation = true;
    BenchmarkRequest capture;
    capture.captureBundles = true;
    BenchmarkRequest write;
    write.writeSupportClaims = true;

    for(auto request : {skip, capture, write})
    {
        // Without --benchmark these modes are untouched.
        EXPECT_FALSE(resolveBenchmark(request).error.has_value());

        request.enabled = true;
        const auto resolution = resolveBenchmark(request);
        EXPECT_FALSE(resolution.settings.has_value());
        ASSERT_TRUE(resolution.error.has_value());
        EXPECT_NE(resolution.error->find("do not execute the graph"), std::string::npos);
    }
}

// ---- reportBenchmark ------------------------------------------------------

TEST(TestReportBenchmark, RecordsTheResultAsTestProperties)
{
    reportBenchmark(successfulResult({0.5f, 0.25f, 1.0f}));

    const auto* result = ::testing::UnitTest::GetInstance()->current_test_info()->result();
    std::map<std::string, std::string> properties;
    for(int i = 0; i < result->test_property_count(); ++i)
    {
        const auto& property = result->GetTestProperty(i);
        properties[property.key()] = property.value();
    }

    EXPECT_EQ(properties["benchmark_engine"], "TEST_ENGINE");
    EXPECT_EQ(properties["benchmark_timing"], "DEVICE_ONLY");
    EXPECT_EQ(properties["benchmark_samples"], "3");
    EXPECT_EQ(properties["benchmark_min_ms"], "0.2500");
    EXPECT_EQ(properties["benchmark_median_ms"], "0.5000");
    EXPECT_EQ(properties["benchmark_mean_ms"], "0.5833");
}

TEST(TestReportBenchmark, AFailedBenchmarkFailsTheTest)
{
    BenchmarkResult failed;
    failed.error = "timed execution failed: device lost";

    EXPECT_NONFATAL_FAILURE(reportBenchmark(failed),
                            "benchmark failed: timed execution failed: device lost");
    EXPECT_NONFATAL_FAILURE(reportBenchmark(BenchmarkResult{}),
                            "benchmark failed: no samples were collected");
}

// ---- bundle harness -------------------------------------------------------

namespace
{

/// Expectations are checked after the harness returns: a gmock or EXPECT failure
/// raised inside driveHarness() lands in its intercepted results, not in this test.
class TestBundleHarnessBenchmark : public ::testing::Test
{
protected:
    std::optional<hipdnn_test_sdk::utilities::ScopedDirectory> _scopedDir;
    testing_support::HarnessMocks _mocks;

    /// What the mocked runner returns from benchmark().
    BenchmarkResult _benchmarkResult = successfulResult({1.0f});
    /// The settings of every benchmark() call, and whether its variant pack held
    /// the verified output tensor.
    std::vector<BenchmarkSettings> _benchmarkCalls;
    bool _sawVerifiedOutput = false;

    void SetUp() override
    {
        testing_support::ensureTestConfigInitialized();
        _scopedDir.emplace(hipdnn_test_sdk::utilities::claimScratchDirectory("bundle_benchmark"));

        ON_CALL(_mocks.engineRunner, benchmark(::testing::_, ::testing::_, ::testing::_))
            .WillByDefault(
                [this](GraphSession&, VariantPack& variantPack, const BenchmarkSettings& s) {
                    _benchmarkCalls.push_back(s);
                    _sawVerifiedOutput = variantPack.count(fixtures::K_OUTPUT_UID) == 1;
                    return _benchmarkResult;
                });
    }

    /// Drives a golden bundle through the real harness. The mocked engine writes
    /// `engineOutput`; the golden output is fixtures::K_OUTPUT_VALUE.
    void run(float engineOutput,
             std::optional<BenchmarkSettings> benchmark,
             ::testing::TestPartResultArray* results)
    {
        testing_support::engineWrites(_mocks.engineRunner, &fixtures::writeOutput, engineOutput);

        auto policy = testing_support::hostPolicy(VerificationMode::AUTO);
        policy.benchmark = benchmark;
        IntegrationBundleVerificationHarness harness(_mocks.dependencies(policy));
        harness.setBundle(fixtures::loadBundle(_scopedDir->path(), "bench", true), "bench");

        testing_support::driveHarness(harness, results);
    }
};

} // namespace

TEST_F(TestBundleHarnessBenchmark, AVerifiedPassIsTimedWithThePolicySettings)
{
    ::testing::TestPartResultArray results;
    run(fixtures::K_OUTPUT_VALUE, settings(4, 1), &results);

    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    EXPECT_FALSE(testing_support::anySkipped(results));
    ASSERT_EQ(_benchmarkCalls.size(), 1u);
    EXPECT_EQ(_benchmarkCalls[0].timedIterations, 4);
    EXPECT_EQ(_benchmarkCalls[0].warmupIterations, 1);
    EXPECT_TRUE(_sawVerifiedOutput) << "the benchmark must run on the verified tensors";
}

TEST_F(TestBundleHarnessBenchmark, AMismatchIsNotTimed)
{
    ::testing::TestPartResultArray results;
    run(fixtures::K_OUTPUT_VALUE + 100.0f, settings(4, 1), &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_TRUE(_benchmarkCalls.empty());
}

TEST_F(TestBundleHarnessBenchmark, WithoutBenchmarkNothingIsTimed)
{
    ::testing::TestPartResultArray results;
    run(fixtures::K_OUTPUT_VALUE, std::nullopt, &results);

    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    EXPECT_TRUE(_benchmarkCalls.empty());
}

TEST_F(TestBundleHarnessBenchmark, ABenchmarkFailureFailsAVerifiedTest)
{
    _benchmarkResult = BenchmarkResult{};
    _benchmarkResult.error = "timed execution failed: device lost";

    ::testing::TestPartResultArray results;
    run(fixtures::K_OUTPUT_VALUE, settings(4, 1), &results);

    EXPECT_EQ(_benchmarkCalls.size(), 1u);
    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_NE(testing_support::allMessages(results).find("benchmark failed: timed execution "
                                                         "failed: device lost"),
              std::string::npos)
        << testing_support::allMessages(results);
}

// NOLINTEND(readability-identifier-naming)
