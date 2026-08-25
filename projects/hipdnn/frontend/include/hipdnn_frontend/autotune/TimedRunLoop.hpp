// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TimedRunLoop.hpp
 * @brief Host-testable timed-run loop helpers for autotune benchmarking
 *
 * Implements the FIXED_AVERAGE and RUN_UNTIL_STABLE timed-iteration loops as
 * pure helpers parameterized on a timing callable. The only GPU dependency of
 * these loops in production is a single per-iteration timing call, so injecting
 * that call as a callable makes the convergence/counting logic testable on the
 * host with a scripted timing sequence (real hipEvent timings cannot be steered
 * deterministically). Per-iteration logging stays at the production call site,
 * supplied via the onIteration callback, so the helpers stay free of GPU and
 * member state.
 */

#pragma once

#include <string>
#include <vector>

#include "hipdnn_frontend/autotune/BenchmarkStatistics.hpp"

namespace hipdnn_frontend::autotune
{

/**
 * @brief Result of a timed-run loop.
 *
 * @c timings holds every successfully measured iteration (in iteration order).
 * @c converged is set by RUN_UNTIL_STABLE when the trailing-window CoV drops
 * below the threshold, and by FIXED_AVERAGE when all iterations succeed.
 * @c benchmarkFailed / @c errorMessage carry the failure path back to the
 * caller so it can mark the engine failed without aborting the autotune run.
 */
struct TimedRunOutcome
{
    std::vector<float> timings;
    bool converged = false;
    bool benchmarkFailed = false;
    std::string errorMessage;
};

namespace detail
{

// RUN_UNTIL_STABLE timed loop: run until the trailing-window CoV converges or
// maxIterations is reached.
//
// TimeOnceFn is a callable (float& elapsed) -> Error with no GPU and no member
// state; it returns a bad Error to signal a benchmark failure. OnIterationFn is
// a callable (int iter, float elapsed, float cov, bool covValid) -> void,
// invoked once per successful iteration for logging. windowSize is the number
// of trailing samples used for the CoV check; stabilityThreshold is the CoV
// value below which the loop is considered converged. cov/covValid passed to
// onIteration are only meaningful once timings.size() >= windowSize.
template <typename TimeOnceFn, typename OnIterationFn>
TimedRunOutcome runUntilStable(int maxIterations,
                               int windowSize,
                               float stabilityThreshold,
                               TimeOnceFn&& timeOnce,
                               OnIterationFn&& onIteration)
{
    TimedRunOutcome outcome;
    outcome.timings.reserve(static_cast<size_t>(maxIterations));

    bool converged = false;
    for(int t = 0; t < maxIterations; ++t)
    {
        float elapsed = 0.0f;
        auto benchErr = timeOnce(elapsed);
        if(benchErr.is_bad())
        {
            outcome.errorMessage = "Benchmark failed on iteration " + std::to_string(t) + ": "
                                   + benchErr.get_message();
            outcome.benchmarkFailed = true;
            break;
        }
        outcome.timings.push_back(elapsed);

        // Compute CoV for convergence check and logging.
        float cov = 0.0f;
        bool covValid = false;
        bool convergedThisIter = false;
        if(static_cast<int>(outcome.timings.size()) >= windowSize)
        {
            const std::vector<float> window(outcome.timings.end() - windowSize,
                                            outcome.timings.end());
            cov = computeCoefficientOfVariation(window);
            covValid = true;
            if(cov < stabilityThreshold)
            {
                convergedThisIter = true;
            }
        }

        onIteration(t, elapsed, cov, covValid);

        if(convergedThisIter)
        {
            converged = true;
            break;
        }
    }

    outcome.converged = converged;
    return outcome;
}

// FIXED_AVERAGE timed loop: run exactly timedIterations and average.
//
// TimeOnceFn is a callable (float& elapsed) -> Error; OnIterationFn is a
// callable (int iter, float elapsed) -> void invoked once per iteration for
// logging.
template <typename TimeOnceFn, typename OnIterationFn>
TimedRunOutcome
    runFixedAverage(int timedIterations, TimeOnceFn&& timeOnce, OnIterationFn&& onIteration)
{
    TimedRunOutcome outcome;
    outcome.timings.reserve(static_cast<size_t>(timedIterations));

    for(int t = 0; t < timedIterations; ++t)
    {
        float elapsed = 0.0f;
        auto benchErr = timeOnce(elapsed);
        if(benchErr.is_bad())
        {
            outcome.errorMessage = "Benchmark failed on iteration " + std::to_string(t) + ": "
                                   + benchErr.get_message();
            outcome.benchmarkFailed = true;
            break;
        }
        outcome.timings.push_back(elapsed);
        onIteration(t, elapsed);
    }

    // FIXED_AVERAGE converges iff every iteration succeeded.
    outcome.converged = !outcome.benchmarkFailed;
    return outcome;
}

} // namespace detail
} // namespace hipdnn_frontend::autotune
