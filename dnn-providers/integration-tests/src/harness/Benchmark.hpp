// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/autotune/AutotuneTypes.hpp>

namespace hipdnn_integration_tests
{

/// How --benchmark times one verified graph.
struct BenchmarkSettings
{
    static constexpr int DEFAULT_TIMED_ITERATIONS = 20;
    // The GPU idles between tests (input fill, reference, comparison), so each test
    // starts cold. On MI210, quick-tier bundles read 4-60% slow with 1-10 warmup
    // executions and settle after about 30. More timed iterations do not fix this.
    static constexpr int DEFAULT_WARMUP_ITERATIONS = 30;

    int timedIterations = DEFAULT_TIMED_ITERATIONS;
    int warmupIterations = DEFAULT_WARMUP_ITERATIONS;
};

/// The benchmark flags as typed, plus the flags that stop a test before it executes.
struct BenchmarkRequest
{
    bool enabled = false;
    std::optional<int> timedIterations;
    std::optional<int> warmupIterations;
    bool skipGraphValidation = false;
    bool captureBundles = false;
    bool writeSupportClaims = false;
};

struct BenchmarkResolution
{
    std::optional<BenchmarkSettings> settings;
    std::optional<std::string> error;
};

/// Settings for --benchmark, or the reason the flags cannot be used together.
///
/// A mode that stops a test before it executes would leave --benchmark timing
/// nothing and the run green. Those combinations are refused instead.
inline BenchmarkResolution resolveBenchmark(const BenchmarkRequest& request)
{
    if(!request.enabled)
    {
        if(request.timedIterations.has_value() || request.warmupIterations.has_value())
        {
            return {std::nullopt,
                    "--benchmark-iterations and --benchmark-warmup require --benchmark"};
        }
        return {};
    }

    if(request.skipGraphValidation || request.captureBundles || request.writeSupportClaims)
    {
        return {std::nullopt,
                "--benchmark cannot be combined with --skip-graph-validation, "
                "--capture-bundles or --write-support-claims: those modes do not "
                "execute the graph"};
    }

    BenchmarkSettings settings;
    settings.timedIterations = request.timedIterations.value_or(settings.timedIterations);
    settings.warmupIterations = request.warmupIterations.value_or(settings.warmupIterations);
    if(settings.timedIterations < 1)
    {
        return {std::nullopt, "--benchmark-iterations must be at least 1"};
    }
    if(settings.warmupIterations < 0)
    {
        return {std::nullopt, "--benchmark-warmup must not be negative"};
    }
    return {settings, std::nullopt};
}

inline const char* toString(hipdnn_frontend::TimingQuality quality)
{
    switch(quality)
    {
    case hipdnn_frontend::TimingQuality::DEVICE_ONLY:
        return "DEVICE_ONLY";
    case hipdnn_frontend::TimingQuality::UNSTALLED:
        return "UNSTALLED";
    case hipdnn_frontend::TimingQuality::INVALID:
    default:
        return "INVALID";
    }
}

/// Publishes the autotune result of one engine for the running test: a
/// "[ BENCHMARK ]" line on stdout and benchmark_* properties in --gtest_output
/// files. A result that did not succeed fails the test.
inline void reportBenchmark(const hipdnn_frontend::AutotuneResult& result)
{
    if(!result.succeeded)
    {
        ADD_FAILURE() << "benchmark failed: "
                      << (result.errorMessage.empty() ? "the engine was not benchmarked"
                                                      : result.errorMessage);
        return;
    }

    const auto ms = [](float value) {
        std::ostringstream out;
        out << std::fixed << std::setprecision(4) << value;
        return out.str();
    };

    const auto minMs = ms(result.minTimeMs);
    const auto robustMs = ms(result.robustTimeMs);
    const auto meanMs = ms(result.avgTimeMs);
    const auto stddevMs = ms(result.stddevMs);

    ::testing::Test::RecordProperty("benchmark_engine", result.engineName);
    ::testing::Test::RecordProperty("benchmark_timing", toString(result.timingQuality));
    ::testing::Test::RecordProperty("benchmark_samples", result.iterationsRun);
    ::testing::Test::RecordProperty("benchmark_min_ms", minMs);
    ::testing::Test::RecordProperty("benchmark_robust_ms", robustMs);
    ::testing::Test::RecordProperty("benchmark_mean_ms", meanMs);
    ::testing::Test::RecordProperty("benchmark_stddev_ms", stddevMs);

    std::string testName = "<unknown test>";
    if(const auto* info = ::testing::UnitTest::GetInstance()->current_test_info())
    {
        testName = std::string(info->test_suite_name()) + "." + info->name();
    }

    std::cout << "[ BENCHMARK ] " << testName << ": engine=" << result.engineName
              << " timing=" << toString(result.timingQuality) << " samples=" << result.iterationsRun
              << " min=" << minMs << " ms robust=" << robustMs << " ms mean=" << meanMs
              << " ms stddev=" << stddevMs << " ms\n";
}

} // namespace hipdnn_integration_tests
