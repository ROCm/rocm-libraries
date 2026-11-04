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

#include <algorithm>
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
 * @c restartUnstalled is set when a stalled pass hit a timeout, a valid
 * UNSTALLED measurement, or exhausted its negative-reading budget for one
 * candidate: the loop stopped immediately without recording that sample, and the
 * caller must discard the whole comparison and rerun it unstalled rather than
 * trust this partial result.
 * @c finalQuality is the TimingQuality of the last recorded sample (INVALID
 * if none was recorded), useful for a caller that wants to know which mode
 * produced @c timings without re-deriving it from the timing callback itself.
 * A finite negative elapsed reading is not a failure: it is re-measured in the same slot
 * within negativeReadingRetryBudget(), and @c discardedNegativeSamples counts those
 * re-measurements. @c timings never contains a negative sample. Exhausting the budget
 * sets @c restartUnstalled in a stalled pass and @c benchmarkFailed in an unstalled one.
 */
struct TimedRunOutcome
{
    std::vector<float> timings;
    bool converged = false;
    bool benchmarkFailed = false;
    std::string errorMessage;
    bool restartUnstalled = false;
    TimingQuality finalQuality = TimingQuality::INVALID;
    int discardedNegativeSamples = 0;
};

namespace detail
{

enum class TimedSampleOutcome
{
    RECORD, // A valid measurement for this pass: append it and keep going.
    RESTART_UNSTALLED, // Stalled pass hit a timeout or a valid UNSTALLED result: stop now,
    // discard this sample, caller reruns the whole comparison unstalled.
    TRANSIENT_INVALID, // A finite negative elapsed reading: not a real error, just a bad
    // sample. Retry with a fresh measurement (see measureOneSample), bounded by
    // negativeReadingRetryBudget() so a candidate can never spin forever.
    MALFORMED // A real Error, or a result that cannot happen for this pass: benchmark failure.
};

// Classifies one ExecutionTiming against the pass it was measured in.
//
// A stalled pass expects DEVICE_ONLY; a timeout or a valid UNSTALLED measurement means the
// stall could not hold for this candidate, so the loop must stop immediately (without
// recording or logging the sample) and the caller reruns every candidate unstalled. An
// unstalled pass expects UNSTALLED only -- it never arms, so it cannot time out and cannot
// itself trigger another restart; anything else from it is a malformed contract from below.
//
// A finite negative elapsed time -- whether reported as an empty elapsedMs (the production
// executeWithPlanTimed() shape) or as a negative value directly (the scripted test seam) --
// is an invalid sample rather than a failure: TRANSIENT_INVALID, not MALFORMED. A non-finite
// (NaN/Inf) elapsedMs remains MALFORMED; that can only reach here as a scripted value, since
// the production path reports it as a bad Error before classification ever runs.
inline TimedSampleOutcome classifyTimedSample(const ExecutionTiming& timing, bool stalled)
{
    if(timing.timedOut)
    {
        return stalled && timing.quality == TimingQuality::INVALID && !timing.elapsedMs.has_value()
                   ? TimedSampleOutcome::RESTART_UNSTALLED
                   : TimedSampleOutcome::MALFORMED;
    }
    if(!timing.elapsedMs.has_value())
    {
        return TimedSampleOutcome::TRANSIENT_INVALID;
    }
    if(!std::isfinite(*timing.elapsedMs))
    {
        return TimedSampleOutcome::MALFORMED;
    }
    if(*timing.elapsedMs < 0.0f)
    {
        return TimedSampleOutcome::TRANSIENT_INVALID;
    }
    if(timing.quality == (stalled ? TimingQuality::DEVICE_ONLY : TimingQuality::UNSTALLED))
    {
        return TimedSampleOutcome::RECORD;
    }
    return stalled && timing.quality == TimingQuality::UNSTALLED
               ? TimedSampleOutcome::RESTART_UNSTALLED
               : TimedSampleOutcome::MALFORMED;
}

// Negative-reading retries per candidate in a stalled pass. Exhausting them restarts the
// comparison unstalled.
inline constexpr int K_STALLED_EXTRA_ATTEMPTS_PER_CANDIDATE = 2;

// An unstalled pass cannot restart, and the Windows HIP runtime reads negative at a steady
// rate (ROCm/rocm-systems#12925), so a fixed budget fails healthy candidates more often the
// more samples they take. The unstalled budget scales with the requested sample count,
// which also caps total attempts at twice that count.
inline int negativeReadingRetryBudget(int sampleSlots, bool stalled)
{
    return stalled ? K_STALLED_EXTRA_ATTEMPTS_PER_CANDIDATE
                   : std::max(sampleSlots, K_STALLED_EXTRA_ATTEMPTS_PER_CANDIDATE);
}

// Outcome of resolving one output sample slot: either a valid measurement, a
// restart-unstalled request, or an unrecoverable failure (a real Error, a malformed
// contract violation, or a transient sample in an unstalled pass after the retry budget
// above is exhausted).
enum class MeasureOutcome
{
    RECORDED,
    RESTART_UNSTALLED,
    FAILED
};

// Resolves one output sample: calls timeOnce, transparently retrying a TRANSIENT_INVALID
// sample with a fresh measurement (bounded by extraAttemptsRemaining, shared across the
// whole candidate run and decremented in place). `attempt` is used only for log/error-message
// text; on RECORDED, `timing` holds the recorded measurement.
template <typename TimeOnceFn>
MeasureOutcome measureOneSample(TimeOnceFn&& timeOnce,
                                bool stalled,
                                int attempt,
                                int& extraAttemptsRemaining,
                                ExecutionTiming& timing,
                                std::string& errorMessage)
{
    for(;;)
    {
        auto benchErr = timeOnce(timing);
        if(benchErr.is_bad())
        {
            errorMessage = "Benchmark failed on iteration " + std::to_string(attempt) + ": "
                           + benchErr.get_message();
            return MeasureOutcome::FAILED;
        }

        const auto sampleOutcome = classifyTimedSample(timing, stalled);
        if(sampleOutcome == TimedSampleOutcome::RECORD)
        {
            return MeasureOutcome::RECORDED;
        }
        if(sampleOutcome == TimedSampleOutcome::RESTART_UNSTALLED)
        {
            return MeasureOutcome::RESTART_UNSTALLED;
        }
        if(sampleOutcome == TimedSampleOutcome::TRANSIENT_INVALID)
        {
            if(extraAttemptsRemaining == 0)
            {
                // Repeated negative readings under the stall gate say the stalled span
                // is unusable for this workload, not that the candidate is broken.
                // Dropping it would silently remove the fastest candidates (the ones
                // closest to the timer floor), so re-measure the comparison unstalled.
                if(stalled)
                {
                    return MeasureOutcome::RESTART_UNSTALLED;
                }
                errorMessage = "Benchmark iteration " + std::to_string(attempt)
                               + " reported an invalid (negative) elapsed time after the "
                                 "retry budget for this candidate was exhausted";
                return MeasureOutcome::FAILED;
            }
            --extraAttemptsRemaining;
            continue; // Fresh measurement replaces the invalid one in the same slot.
        }
        errorMessage = "Benchmark iteration " + std::to_string(attempt)
                       + " reported a malformed timing result";
        return MeasureOutcome::FAILED;
    }
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

    const int retryBudget = negativeReadingRetryBudget(maxIterations, stalled);
    int extraAttemptsRemaining = retryBudget;
    bool converged = false;
    for(int t = 0; t < maxIterations; ++t)
    {
        ExecutionTiming timing;
        const auto measured = measureOneSample(
            timeOnce, stalled, t, extraAttemptsRemaining, timing, outcome.errorMessage);
        if(measured == MeasureOutcome::RESTART_UNSTALLED)
        {
            outcome.restartUnstalled = true;
            break;
        }
        if(measured == MeasureOutcome::FAILED)
        {
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
    outcome.discardedNegativeSamples = retryBudget - extraAttemptsRemaining;
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

    const int retryBudget = negativeReadingRetryBudget(timedIterations, stalled);
    int extraAttemptsRemaining = retryBudget;
    for(int t = 0; t < timedIterations; ++t)
    {
        ExecutionTiming timing;
        const auto measured = measureOneSample(
            timeOnce, stalled, t, extraAttemptsRemaining, timing, outcome.errorMessage);
        if(measured == MeasureOutcome::RESTART_UNSTALLED)
        {
            outcome.restartUnstalled = true;
            break;
        }
        if(measured == MeasureOutcome::FAILED)
        {
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
    outcome.discardedNegativeSamples = retryBudget - extraAttemptsRemaining;
    return outcome;
}

} // namespace detail
} // namespace hipdnn_frontend::autotune
