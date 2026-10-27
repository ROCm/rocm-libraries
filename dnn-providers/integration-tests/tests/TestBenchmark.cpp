// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

#include "BundleFixtureFiles.hpp"
#include "HarnessTestSupport.hpp"
#include "harness/Benchmark.hpp"
#include "harness/BenchmarkReport.hpp"

// NOLINTBEGIN(readability-identifier-naming)

using namespace hipdnn_integration_tests;
using namespace hipdnn_integration_tests::bundle;
using hipdnn_frontend::AutotuneResult;
using hipdnn_frontend::TimingQuality;

namespace
{

BenchmarkSettings settings(int timed, int warmup)
{
    BenchmarkSettings result;
    result.timedIterations = timed;
    result.warmupIterations = warmup;
    return result;
}

AutotuneResult successfulResult()
{
    AutotuneResult result;
    result.engineName = "TEST_ENGINE";
    result.succeeded = true;
    result.iterationsRun = 3;
    result.minTimeMs = 0.25f;
    result.robustTimeMs = 0.375f;
    result.avgTimeMs = 0.5f;
    result.stddevMs = 0.125f;
    result.timingQuality = TimingQuality::DEVICE_ONLY;
    return result;
}

AutotuneResult failedResult(std::string message)
{
    AutotuneResult result;
    result.errorMessage = std::move(message);
    return result;
}

} // namespace

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
    EXPECT_EQ(resolution.settings->warmupIterations, 30);
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

// ---- selectEngineResult ---------------------------------------------------

namespace
{

AutotuneResult resultFor(int64_t engineId, bool excludedByCaller = false)
{
    auto result = successfulResult();
    result.engineId = engineId;
    result.excludedByCaller = excludedByCaller;
    return result;
}

const hipdnn_frontend::Error K_GOOD{hipdnn_frontend::ErrorCode::OK, ""};
const hipdnn_frontend::Error K_NO_ENGINE_SUCCEEDED{hipdnn_frontend::ErrorCode::HIPDNN_BACKEND_ERROR,
                                                   "no engine succeeded"};

} // namespace

TEST(TestSelectEngineResult, ReturnsTheResultOfTheRequestedEngine)
{
    auto other = resultFor(7);
    other.minTimeMs = 99.0f;
    auto wanted = resultFor(42);
    wanted.minTimeMs = 1.0f;

    const auto selected = selectEngineResult(42, {other, wanted}, K_GOOD);

    EXPECT_EQ(selected.engineId, 42);
    EXPECT_TRUE(selected.succeeded);
    EXPECT_FLOAT_EQ(selected.minTimeMs, 1.0f);
}

TEST(TestSelectEngineResult, TheEngineOwnFailureWinsOverTheAutotuneStatus)
{
    auto failed = resultFor(42);
    failed.succeeded = false;
    failed.errorMessage = "Benchmark failed on iteration 2";

    const auto selected = selectEngineResult(42, {failed}, K_NO_ENGINE_SUCCEEDED);

    EXPECT_FALSE(selected.succeeded);
    EXPECT_EQ(selected.errorMessage, "Benchmark failed on iteration 2");
}

TEST(TestSelectEngineResult, NeverReportsAnExcludedOrDifferentEngine)
{
    for(const auto& results : {std::vector<AutotuneResult>{resultFor(42, true)},
                               std::vector<AutotuneResult>{resultFor(7)}})
    {
        const auto selected = selectEngineResult(42, results, K_GOOD);

        EXPECT_FALSE(selected.succeeded);
        EXPECT_EQ(selected.engineId, 42);
        EXPECT_EQ(selected.errorMessage, "autotune returned no result for the engine under test");
    }
}

TEST(TestSelectEngineResult, NoResultsReportsTheAutotuneStatus)
{
    const auto good = selectEngineResult(42, {}, K_GOOD);
    const auto bad = selectEngineResult(42, {}, K_NO_ENGINE_SUCCEEDED);

    EXPECT_FALSE(good.succeeded);
    EXPECT_EQ(good.errorMessage, "autotune returned no result for the engine under test");
    EXPECT_FALSE(bad.succeeded);
    EXPECT_EQ(bad.errorMessage, "no engine succeeded");
}

// ---- toString(TimingQuality) ------------------------------------------------

TEST(TestTimingQualityName, NamesEveryValueAndRefusesUnknownValues)
{
    EXPECT_STREQ(toString(TimingQuality::DEVICE_ONLY), "DEVICE_ONLY");
    EXPECT_STREQ(toString(TimingQuality::UNSTALLED), "UNSTALLED");
    EXPECT_STREQ(toString(TimingQuality::INVALID), "INVALID");
    EXPECT_THROW(toString(static_cast<TimingQuality>(99)), std::invalid_argument);
}

// ---- reportBenchmark ------------------------------------------------------

TEST(TestReportBenchmark, RecordsTheResultAsProperties)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter forceBenchmarking(
        hipdnn_plugin_sdk::FORCE_BENCHMARKING_ENV_NAME, "1");
    auto benchmarked = successfulResult();
    benchmarked.supportsExhaustive = true;

    reportBenchmark(benchmarked);

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
    EXPECT_EQ(properties["benchmark_robust_ms"], "0.3750");
    EXPECT_EQ(properties["benchmark_mean_ms"], "0.5000");
    EXPECT_EQ(properties["benchmark_stddev_ms"], "0.1250");
    EXPECT_EQ(properties["benchmark_tuned"], "1");
}

// The harness sets no knobs, so only HIPDNN_FORCE_BENCHMARKING can turn benchmarking
// on, and only an engine with the global.benchmarking knob reads it.
TEST(TestReportBenchmark, TunedOnlyWhenForcedOnAnEngineWithTheKnob)
{
    struct Case
    {
        bool engineHasKnob;
        const char* forceBenchmarking;
        bool tuned;
    };
    const std::array<Case, 5> cases{{
        {true, "", false},
        {true, "1", true},
        {true, "0", false},
        {false, "1", false},
        {false, "", false},
    }};

    for(const auto& c : cases)
    {
        const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter forceBenchmarking(
            hipdnn_plugin_sdk::FORCE_BENCHMARKING_ENV_NAME, c.forceBenchmarking);
        auto result = successfulResult();
        result.supportsExhaustive = c.engineHasKnob;

        EXPECT_EQ(selectedByBenchmarking(result), c.tuned)
            << "knob=" << c.engineHasKnob << " HIPDNN_FORCE_BENCHMARKING='" << c.forceBenchmarking
            << "'";
    }
}

TEST(TestReportBenchmark, AFailedBenchmarkAddsAFailure)
{
    EXPECT_NONFATAL_FAILURE(reportBenchmark(failedResult("Benchmark failed on iteration 2")),
                            "benchmark failed: Benchmark failed on iteration 2");
    EXPECT_NONFATAL_FAILURE(reportBenchmark(AutotuneResult{}),
                            "benchmark failed: the engine was not benchmarked");
}

// ---- bundle harness -------------------------------------------------------

namespace
{

/// One way the harness verifies a bundle. Each way has its own benchmark hook.
struct OracleCase
{
    const char* name;
    VerificationMode mode;
    bool hasGoldenOutput;
    Verifier verifier;

    friend void PrintTo(const OracleCase& oracleCase, std::ostream* os)
    {
        *os << oracleCase.name;
    }
};

/// AUTO with golden data and explicit GOLDEN compare against the golden file.
/// AUTO without golden data falls through to the reference oracle chain.
const std::array<OracleCase, 3> ORACLE_CASES{{
    {"AutoGolden", VerificationMode::AUTO, true, Verifier::GOLDEN},
    {"AutoReference", VerificationMode::AUTO, false, Verifier::GPU_REFERENCE},
    {"GoldenMode", VerificationMode::GOLDEN, true, Verifier::GOLDEN},
}};

/// Expectations are checked after the harness returns: a gmock or EXPECT failure
/// raised inside driveHarness() lands in its intercepted results, not in this test.
class TestBundleHarnessBenchmark : public ::testing::TestWithParam<OracleCase>
{
protected:
    std::optional<hipdnn_test_sdk::utilities::ScopedDirectory> _scopedDir;
    testing_support::HarnessMocks _mocks;

    /// What the mocked runner returns from benchmark().
    AutotuneResult _benchmarkResult = successfulResult();
    /// The settings of every benchmark() call, and whether its variant pack held
    /// the verified output tensor.
    std::vector<BenchmarkSettings> _benchmarkCalls;
    bool _sawVerifiedOutput = false;
    /// When set, the mocked runner throws it from benchmark().
    std::optional<std::string> _benchmarkThrows;
    /// The verifier the harness reported for each bundle it ran.
    std::vector<Verifier> _verifiers;

    void SetUp() override
    {
        using ::testing::_;
        testing_support::ensureTestConfigInitialized();
        _scopedDir.emplace(hipdnn_test_sdk::utilities::claimScratchDirectory("bundle_benchmark"));
        testing_support::captureVerifiers(_mocks.reporter, _verifiers);

        ON_CALL(_mocks.engineRunner, benchmark(_, _, _))
            .WillByDefault(
                [this](GraphSession&, VariantPack& variantPack, const BenchmarkSettings& s) {
                    _benchmarkCalls.push_back(s);
                    _sawVerifiedOutput = variantPack.count(fixtures::K_OUTPUT_UID) == 1;
                    if(_benchmarkThrows.has_value())
                    {
                        throw std::runtime_error(*_benchmarkThrows);
                    }
                    return _benchmarkResult;
                });

        // Both references produce the golden value, so the reference path and the
        // golden path agree on what a match is.
        for(auto* reference : {&_mocks.cpuReference, &_mocks.gpuReference})
        {
            ON_CALL(*reference, execute(_, _, _))
                .WillByDefault([](void*, size_t, const VariantPack& variantPack) {
                    auto* ptr = static_cast<float*>(variantPack.at(fixtures::K_OUTPUT_UID));
                    std::fill(ptr, ptr + fixtures::K_OUTPUT_ELEMS, fixtures::K_OUTPUT_VALUE);
                });
        }
    }

    /// Drives a bundle through the real harness with the parameter's oracle. The
    /// mocked engine writes `engineOutput`; the expected output is
    /// fixtures::K_OUTPUT_VALUE.
    void run(float engineOutput,
             std::optional<BenchmarkSettings> benchmark,
             ::testing::TestPartResultArray* results)
    {
        testing_support::engineWrites(_mocks.engineRunner, &fixtures::writeOutput, engineOutput);

        auto policy = testing_support::hostPolicy(GetParam().mode);
        policy.benchmark = benchmark;
        IntegrationBundleVerificationHarness harness(_mocks.dependencies(policy));
        harness.setBundle(
            fixtures::loadBundle(_scopedDir->path(), "bench", GetParam().hasGoldenOutput), "bench");

        testing_support::driveHarness(harness, results);
    }
};

} // namespace

TEST_P(TestBundleHarnessBenchmark, AVerifiedPassIsTimedWithThePolicySettings)
{
    ::testing::TestPartResultArray results;
    run(fixtures::K_OUTPUT_VALUE, settings(4, 1), &results);

    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    EXPECT_FALSE(testing_support::anySkipped(results));
    ASSERT_EQ(_benchmarkCalls.size(), 1u);
    EXPECT_EQ(_benchmarkCalls[0].timedIterations, 4);
    EXPECT_EQ(_benchmarkCalls[0].warmupIterations, 1);
    EXPECT_TRUE(_sawVerifiedOutput) << "the benchmark must run on the verified tensors";
    EXPECT_EQ(_verifiers, std::vector<Verifier>{GetParam().verifier});
}

TEST_P(TestBundleHarnessBenchmark, AMismatchIsNotTimed)
{
    ::testing::TestPartResultArray results;
    run(fixtures::K_OUTPUT_VALUE + 100.0f, settings(4, 1), &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_TRUE(_benchmarkCalls.empty());
}

TEST_P(TestBundleHarnessBenchmark, WithoutBenchmarkAMatchPassesUntimed)
{
    ::testing::TestPartResultArray results;
    run(fixtures::K_OUTPUT_VALUE, std::nullopt, &results);

    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    EXPECT_FALSE(testing_support::anySkipped(results));
    EXPECT_TRUE(_benchmarkCalls.empty());
}

TEST_P(TestBundleHarnessBenchmark, WithoutBenchmarkAMismatchStillFails)
{
    ::testing::TestPartResultArray results;
    run(fixtures::K_OUTPUT_VALUE + 100.0f, std::nullopt, &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_TRUE(_benchmarkCalls.empty());
}

TEST_P(TestBundleHarnessBenchmark, ABenchmarkFailureFailsAVerifiedBundle)
{
    _benchmarkResult = failedResult("timed execution failed: device lost");

    ::testing::TestPartResultArray results;
    run(fixtures::K_OUTPUT_VALUE, settings(4, 1), &results);

    EXPECT_EQ(_benchmarkCalls.size(), 1u);
    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_NE(testing_support::allMessages(results).find("benchmark failed: timed execution "
                                                         "failed: device lost"),
              std::string::npos)
        << testing_support::allMessages(results);
}

// A throw from the benchmark must not turn the verified PASS into a harness failure,
// which would also change the support-claim verdict.
TEST_P(TestBundleHarnessBenchmark, AThrowingBenchmarkFailsButKeepsTheVerifiedOutcome)
{
    _benchmarkThrows = "hipMalloc failed";

    ::testing::TestPartResultArray results;
    run(fixtures::K_OUTPUT_VALUE, settings(4, 1), &results);

    EXPECT_EQ(_benchmarkCalls.size(), 1u);
    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_NE(testing_support::allMessages(results).find("benchmark failed: hipMalloc failed"),
              std::string::npos)
        << testing_support::allMessages(results);
    EXPECT_EQ(_verifiers, std::vector<Verifier>{GetParam().verifier})
        << "the outcome the harness recorded must still be the verified pass";
}

INSTANTIATE_TEST_SUITE_P(Oracles,
                         TestBundleHarnessBenchmark,
                         ::testing::ValuesIn(ORACLE_CASES),
                         [](const ::testing::TestParamInfo<OracleCase>& info) {
                             return std::string(info.param.name);
                         });

// NOLINTEND(readability-identifier-naming)
