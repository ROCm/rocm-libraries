// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/autotune/AutotuneTypes.hpp>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>

#include "harness/CurrentTestName.hpp"

namespace hipdnn_integration_tests
{

inline const char* toString(hipdnn_frontend::TimingQuality quality)
{
    switch(quality)
    {
    case hipdnn_frontend::TimingQuality::DEVICE_ONLY:
        return "DEVICE_ONLY";
    case hipdnn_frontend::TimingQuality::UNSTALLED:
        return "UNSTALLED";
    case hipdnn_frontend::TimingQuality::INVALID:
        return "INVALID";
    default:
        // An unknown value must not print as a valid quality.
        throw std::invalid_argument("unknown TimingQuality value "
                                    + std::to_string(static_cast<int>(quality)));
    }
}

/// Picks the result of the engine under test from Graph::autotune() output.
///
/// autotune() returns an error when no engine succeeds, but the engine's own result
/// says why, so that result wins over `status`. A result for another engine, or one
/// the caller excluded, is never used: it would report the wrong engine's timing.
inline hipdnn_frontend::AutotuneResult
    selectEngineResult(int64_t engineId,
                       const std::vector<hipdnn_frontend::AutotuneResult>& results,
                       const hipdnn_frontend::Error& status)
{
    for(const auto& result : results)
    {
        if(result.engineId == engineId && !result.excludedByCaller)
        {
            return result;
        }
    }
    hipdnn_frontend::AutotuneResult failed;
    failed.engineId = engineId;
    failed.errorMessage = status.is_good() ? "autotune returned no result for the engine under test"
                                           : status.get_message();
    return failed;
}

/// True when the engine selected its solution by benchmarking candidates rather than
/// by its heuristics. The harness sets no knobs, so global.benchmarking keeps its
/// default (0 in every hipDNN provider). Only HIPDNN_FORCE_BENCHMARKING turns it on,
/// and only for engines that expose the knob (AutotuneResult::supportsExhaustive).
inline bool selectedByBenchmarking(const hipdnn_frontend::AutotuneResult& result)
{
    return result.supportsExhaustive
           && hipdnn_plugin_sdk::benchmarkingOverrideFromEnv().value_or(false);
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
    const int tuned = selectedByBenchmarking(result) ? 1 : 0;

    ::testing::Test::RecordProperty("benchmark_engine", result.engineName);
    ::testing::Test::RecordProperty("benchmark_timing", toString(result.timingQuality));
    ::testing::Test::RecordProperty("benchmark_samples", result.iterationsRun);
    ::testing::Test::RecordProperty("benchmark_min_ms", minMs);
    ::testing::Test::RecordProperty("benchmark_robust_ms", robustMs);
    ::testing::Test::RecordProperty("benchmark_mean_ms", meanMs);
    ::testing::Test::RecordProperty("benchmark_stddev_ms", stddevMs);
    ::testing::Test::RecordProperty("benchmark_tuned", tuned);

    auto testName = currentTestName();
    if(testName.empty())
    {
        testName = "<unknown test>";
    }

    std::cout << "[ BENCHMARK ] " << testName << ": engine=" << result.engineName
              << " tuned=" << tuned << " timing=" << toString(result.timingQuality)
              << " samples=" << result.iterationsRun << " min=" << minMs
              << " ms robust=" << robustMs << " ms mean=" << meanMs << " ms stddev=" << stddevMs
              << " ms\n";
}

} // namespace hipdnn_integration_tests
