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

#include <cmath>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/TimingStatistics.hpp>
#include <hipdnn_frontend/Types.hpp>

namespace hipdnn_frontend::autotune
{

/**
 * @brief Result of a timed-run loop.
 *
 * @c timings holds every successfully measured iteration (in iteration order).
 * @c converged is set by RUN_UNTIL_STABLE when the trailing-window CoV drops
 * below the threshold, and by FIXED_AVERAGE when all iterations succeed;
 * neither a benchmark failure nor a restart request counts as converged.
 * @c benchmarkFailed / @c errorMessage carry the failure path back to the
 * caller so it can mark the engine failed without aborting the autotune run.
 * @c restartUnstalled is set when a stalled pass hit a timeout or a valid
 * UNSTALLED measurement: the loop stopped immediately without recording that
 * sample, and the caller must discard the whole comparison and rerun it
 * unstalled rather than trust this partial result.
 * @c finalQuality is the TimingQuality of the last recorded sample (INVALID
 * if none was recorded), useful for a caller that wants to know which mode
 * produced @c timings without re-deriving it from the timing callback itself.
 */
struct TimedRunOutcome
{
    std::vector<float> timings;
    bool converged = false;
    bool benchmarkFailed = false;
    std::string errorMessage;
    bool restartUnstalled = false;
    TimingQuality finalQuality = TimingQuality::INVALID;
};

namespace detail
{

enum class TimedSampleOutcome
{
    RECORD, // A valid measurement for this pass: append it and keep going.
    RESTART_UNSTALLED, // Stalled pass hit a timeout or a valid UNSTALLED result: stop now,
    // discard this sample, caller reruns the whole comparison unstalled.
    MALFORMED // A real Error, or a result that cannot happen for this pass: benchmark failure.
};

// Classifies one ExecutionTiming against the pass it was measured in.
//
// A stalled pass expects DEVICE_ONLY; a timeout or a valid UNSTALLED measurement means the
// stall could not hold for this candidate, so the loop must stop immediately (without
// recording or logging the sample) and the caller reruns every candidate unstalled. An
// unstalled pass expects UNSTALLED only -- it never arms, so it cannot time out and cannot
// itself trigger another restart; anything else from it is a malformed contract from below.
inline TimedSampleOutcome classifyTimedSample(const ExecutionTiming& timing, bool stalled)
{
    if(timing.timedOut)
    {
        return stalled && timing.quality == TimingQuality::INVALID && !timing.elapsedMs.has_value()
                   ? TimedSampleOutcome::RESTART_UNSTALLED
                   : TimedSampleOutcome::MALFORMED;
    }
    if(!timing.elapsedMs.has_value() || !std::isfinite(*timing.elapsedMs)
       || *timing.elapsedMs < 0.0f)
    {
        return TimedSampleOutcome::MALFORMED;
    }
    if(timing.quality == (stalled ? TimingQuality::DEVICE_ONLY : TimingQuality::UNSTALLED))
    {
        return TimedSampleOutcome::RECORD;
    }
    return stalled && timing.quality == TimingQuality::UNSTALLED
               ? TimedSampleOutcome::RESTART_UNSTALLED
               : TimedSampleOutcome::MALFORMED;
}

// RUN_UNTIL_STABLE timed loop: run until the trailing-window CoV converges or
// maxIterations is reached.
//
// TimeOnceFn is a callable (ExecutionTiming&) -> Error with no GPU and no member state; it
// returns a bad Error to signal a benchmark failure. OnIterationFn is a callable (int iter,
// float elapsed, float cov, bool covValid) -> void, invoked once per successfully recorded
// iteration for logging. windowSize is the number of trailing samples used for the CoV
// check; stabilityThreshold is the CoV value below which the loop is considered converged.
// cov/covValid passed to onIteration are only meaningful once timings.size() >= windowSize.
// `stalled` selects which TimedSampleOutcome this pass expects; see classifyTimedSample.
template <typename TimeOnceFn, typename OnIterationFn>
TimedRunOutcome runUntilStable(int maxIterations,
                               int windowSize,
                               float stabilityThreshold,
                               bool stalled,
                               TimeOnceFn&& timeOnce,
                               OnIterationFn&& onIteration)
{
    TimedRunOutcome outcome;
    outcome.timings.reserve(static_cast<size_t>(maxIterations));

    bool converged = false;
    for(int t = 0; t < maxIterations; ++t)
    {
        ExecutionTiming timing;
        auto benchErr = timeOnce(timing);
        if(benchErr.is_bad())
        {
            outcome.errorMessage = "Benchmark failed on iteration " + std::to_string(t) + ": "
                                   + benchErr.get_message();
            outcome.benchmarkFailed = true;
            break;
        }

        const auto sampleOutcome = classifyTimedSample(timing, stalled);
        if(sampleOutcome == TimedSampleOutcome::RESTART_UNSTALLED)
        {
            outcome.restartUnstalled = true;
            break;
        }
        if(sampleOutcome == TimedSampleOutcome::MALFORMED)
        {
            outcome.errorMessage = "Benchmark iteration " + std::to_string(t)
                                   + " reported a malformed timing result";
            outcome.benchmarkFailed = true;
            break;
        }

        const float elapsed = *timing.elapsedMs;
        outcome.timings.push_back(elapsed);
        outcome.finalQuality = timing.quality;

        // Compute CoV for convergence check and logging.
        float cov = 0.0f;
        bool covValid = false;
        bool convergedThisIter = false;
        if(static_cast<int>(outcome.timings.size()) >= windowSize)
        {
            const std::vector<float> window(outcome.timings.end() - windowSize,
                                            outcome.timings.end());
            cov = ::hipdnn_data_sdk::utilities::detail::coefficientOfVariation(window);
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
// TimeOnceFn is a callable (ExecutionTiming&) -> Error; OnIterationFn is a callable (int
// iter, float elapsed) -> void invoked once per successfully recorded iteration for
// logging. `stalled` selects which TimedSampleOutcome this pass expects; see
// classifyTimedSample.
template <typename TimeOnceFn, typename OnIterationFn>
TimedRunOutcome runFixedAverage(int timedIterations,
                                bool stalled,
                                TimeOnceFn&& timeOnce,
                                OnIterationFn&& onIteration)
{
    TimedRunOutcome outcome;
    outcome.timings.reserve(static_cast<size_t>(timedIterations));

    for(int t = 0; t < timedIterations; ++t)
    {
        ExecutionTiming timing;
        auto benchErr = timeOnce(timing);
        if(benchErr.is_bad())
        {
            outcome.errorMessage = "Benchmark failed on iteration " + std::to_string(t) + ": "
                                   + benchErr.get_message();
            outcome.benchmarkFailed = true;
            break;
        }

        const auto sampleOutcome = classifyTimedSample(timing, stalled);
        if(sampleOutcome == TimedSampleOutcome::RESTART_UNSTALLED)
        {
            outcome.restartUnstalled = true;
            break;
        }
        if(sampleOutcome == TimedSampleOutcome::MALFORMED)
        {
            outcome.errorMessage = "Benchmark iteration " + std::to_string(t)
                                   + " reported a malformed timing result";
            outcome.benchmarkFailed = true;
            break;
        }

        const float elapsed = *timing.elapsedMs;
        outcome.timings.push_back(elapsed);
        outcome.finalQuality = timing.quality;
        onIteration(t, elapsed);
    }

    // FIXED_AVERAGE converges iff every iteration succeeded without a restart request.
    outcome.converged = !outcome.benchmarkFailed && !outcome.restartUnstalled;
    return outcome;
}

} // namespace detail
} // namespace hipdnn_frontend::autotune
