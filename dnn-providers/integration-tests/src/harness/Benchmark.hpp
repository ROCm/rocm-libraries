// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>

namespace hipdnn_integration_tests
{

/// How --benchmark times one verified graph.
struct BenchmarkSettings
{
    static constexpr int DEFAULT_TIMED_ITERATIONS = 20;
    static constexpr int DEFAULT_WARMUP_ITERATIONS = 3;

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

/// The timings of one graph on one engine.
struct BenchmarkResult
{
    std::string engine;
    std::vector<float> samplesMs;

    /// DEVICE_ONLY only when every sample was. One UNSTALLED sample makes the set
    /// UNSTALLED, because that sample can include host submission time.
    hipdnn_frontend::TimingQuality quality = hipdnn_frontend::TimingQuality::INVALID;

    /// Timed runs that returned no measurement (stall watchdog timeout or a negative
    /// reading). They are retried and are not in samplesMs.
    int discardedSamples = 0;

    /// Why the benchmark failed. Empty on success.
    std::string error;

    bool ok() const
    {
        return error.empty() && !samplesMs.empty();
    }

    float minMs() const
    {
        return samplesMs.empty() ? 0.0f : *std::min_element(samplesMs.begin(), samplesMs.end());
    }

    float medianMs() const
    {
        if(samplesMs.empty())
        {
            return 0.0f;
        }
        auto sorted = samplesMs;
        std::sort(sorted.begin(), sorted.end());
        const std::size_t mid = sorted.size() / 2;
        return sorted.size() % 2 == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2.0f;
    }

    float meanMs() const
    {
        if(samplesMs.empty())
        {
            return 0.0f;
        }
        const double sum = std::accumulate(samplesMs.begin(), samplesMs.end(), 0.0);
        return static_cast<float>(sum / static_cast<double>(samplesMs.size()));
    }
};

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

using BenchmarkRunOnce = std::function<hipdnn_frontend::Error()>;
using BenchmarkTimeOnce = std::function<hipdnn_frontend::Error(hipdnn_frontend::ExecutionTiming&)>;

/// Runs `warmupIterations` untimed executions, then collects `timedIterations` valid
/// samples. A timed run with no measurement is retried, up to `timedIterations` extra
/// attempts. Any execution error stops the benchmark and is returned in `error`.
///
/// Takes callables rather than a graph so the loop is testable without a device.
inline BenchmarkResult runBenchmarkLoop(const BenchmarkSettings& settings,
                                        const BenchmarkRunOnce& runOnce,
                                        const BenchmarkTimeOnce& timeOnce)
{
    BenchmarkResult result;

    for(int i = 0; i < settings.warmupIterations; ++i)
    {
        if(auto status = runOnce(); !status.is_good())
        {
            result.error = "warmup execution failed: " + status.get_message();
            return result;
        }
    }

    bool anyUnstalled = false;
    const int maxAttempts = 2 * settings.timedIterations;
    for(int attempt = 0; attempt < maxAttempts
                         && static_cast<int>(result.samplesMs.size()) < settings.timedIterations;
        ++attempt)
    {
        hipdnn_frontend::ExecutionTiming timing;
        if(auto status = timeOnce(timing); !status.is_good())
        {
            result.error = "timed execution failed: " + status.get_message();
            result.samplesMs.clear();
            return result;
        }
        if(timing.quality == hipdnn_frontend::TimingQuality::INVALID || !timing.elapsedMs)
        {
            ++result.discardedSamples;
            continue;
        }
        anyUnstalled = anyUnstalled || timing.quality == hipdnn_frontend::TimingQuality::UNSTALLED;
        result.samplesMs.push_back(*timing.elapsedMs);
    }

    if(static_cast<int>(result.samplesMs.size()) < settings.timedIterations)
    {
        result.error = std::to_string(result.discardedSamples) + " of "
                       + std::to_string(maxAttempts)
                       + " timed executions returned no measurement; collected "
                       + std::to_string(result.samplesMs.size()) + " of "
                       + std::to_string(settings.timedIterations) + " samples";
        result.samplesMs.clear();
        return result;
    }

    result.quality = anyUnstalled ? hipdnn_frontend::TimingQuality::UNSTALLED
                                  : hipdnn_frontend::TimingQuality::DEVICE_ONLY;
    return result;
}

/// Publishes a benchmark for the running test: a "[ BENCHMARK ]" line on stdout and
/// benchmark_* properties in --gtest_output files. A failed benchmark fails the test.
inline void reportBenchmark(const BenchmarkResult& result)
{
    if(!result.ok())
    {
        ADD_FAILURE() << "benchmark failed: "
                      << (result.error.empty() ? "no samples were collected" : result.error);
        return;
    }

    const auto ms = [](float value) {
        std::ostringstream out;
        out << std::fixed << std::setprecision(4) << value;
        return out.str();
    };

    const auto minMs = ms(result.minMs());
    const auto medianMs = ms(result.medianMs());
    const auto meanMs = ms(result.meanMs());

    ::testing::Test::RecordProperty("benchmark_engine", result.engine);
    ::testing::Test::RecordProperty("benchmark_timing", toString(result.quality));
    ::testing::Test::RecordProperty("benchmark_samples", static_cast<int>(result.samplesMs.size()));
    ::testing::Test::RecordProperty("benchmark_min_ms", minMs);
    ::testing::Test::RecordProperty("benchmark_median_ms", medianMs);
    ::testing::Test::RecordProperty("benchmark_mean_ms", meanMs);

    std::string testName = "<unknown test>";
    if(const auto* info = ::testing::UnitTest::GetInstance()->current_test_info())
    {
        testName = std::string(info->test_suite_name()) + "." + info->name();
    }

    std::cout << "[ BENCHMARK ] " << testName << ": engine=" << result.engine
              << " timing=" << toString(result.quality) << " samples=" << result.samplesMs.size()
              << " min=" << minMs << " ms median=" << medianMs << " ms mean=" << meanMs << " ms";
    if(result.discardedSamples > 0)
    {
        std::cout << " discarded=" << result.discardedSamples;
    }
    std::cout << '\n';
}

} // namespace hipdnn_integration_tests
