// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace hipdnn_integration_tests
{

struct ReferenceBenchmarkRequest
{
    bool enabled = false;
    bool skipGraphValidation = false;
    bool captureBundles = false;
    bool writeSupportClaims = false;
};

inline std::optional<std::string>
    validateReferenceBenchmark(const ReferenceBenchmarkRequest& request)
{
    if(request.enabled
       && (request.skipGraphValidation || request.captureBundles || request.writeSupportClaims))
    {
        return "--benchmark-reference cannot be combined with --skip-graph-validation, "
               "--capture-bundles or --write-support-claims: those modes do not "
               "execute the reference";
    }
    return std::nullopt;
}

struct ReferenceBenchmarkResult
{
    std::string reference;
    std::string validator;
    std::string status = "pending";
    double prepareMs = 0.0;
    double executeMs = 0.0;
    double materializeMs = 0.0;

    double totalMs() const
    {
        return prepareMs + executeMs + materializeMs;
    }
};

class ReferenceBenchmarkError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

class ReferenceBenchmarkTimer
{
public:
    using Clock = std::chrono::steady_clock;
    using Now = std::function<Clock::time_point()>;

    explicit ReferenceBenchmarkTimer(
        std::function<void()> synchronize = [] {}, Now now = [] { return Clock::now(); })
        : _synchronize(std::move(synchronize))
        , _now(std::move(now))
    {
    }

    template <typename Work>
    void measure(double& elapsedMs, Work&& work) const
    {
        _synchronize();
        const auto start = _now();
        std::forward<Work>(work)();
        _synchronize();
        elapsedMs += std::chrono::duration<double, std::milli>(_now() - start).count();
    }

private:
    std::function<void()> _synchronize;
    Now _now;
};

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

} // namespace hipdnn_integration_tests
