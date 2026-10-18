// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/ScopedResource.hpp>
#include <hipdnn_data_sdk/utilities/StallGate.hpp>
#include <hipdnn_data_sdk/utilities/TimingStatistics.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

/// The sampling, timer and stall-restart machinery shared by BenchmarkPlan (one catalog's
/// candidates) and SlotProgramBenchmark (whole slot programs). Both time an IPlan through its
/// execute() only; neither touches a dispatcher, PreparedDispatch, or HIP launch API.
namespace hipdnn_plugin_sdk::ingestor
{

/// Sampling counts: one untimed warmup and seven valid timed runs per candidate,
/// matching MIOpen's EvaluateInvokers before replacement attempts.
constexpr int BENCHMARK_WARMUP_RUNS = 1;
constexpr int BENCHMARK_ITERATIONS = 7;

/// A finite negative elapsed time is a bad HIP event reading, not proof a candidate is
/// broken: the Windows HIP runtime reports them (ROCm/rocm-systems#12925).
/// sampleRuns() discards one and re-measures the same slot, using at most this many
/// extra attempts per candidate in each sampling pass. NaN, Inf, and a backend timing
/// error are never retried this way.
constexpr int MAX_NEGATIVE_SAMPLE_RETRIES = 2;

// A zero iteration count would leave sampleRuns()'s reduction at its DBL_MAX seed
// and report that as a real measurement, which reads as a successful benchmark rather
// than the honest no-usable-candidate path.
static_assert(BENCHMARK_ITERATIONS > 0, "benchmarking must time at least one iteration");
static_assert(BENCHMARK_WARMUP_RUNS >= 0);

namespace detail
{

/// A hipEvent_t that destroys itself, or an empty ScopedResource if creation failed.
inline hipdnn_data_sdk::utilities::ScopedResource<hipEvent_t> createScopedHipEvent()
{
    hipEvent_t event = nullptr;
    if(hipEventCreate(&event) != hipSuccess)
    {
        return {};
    }
    // Nothing actionable on a destroy failure; the event is being discarded anyway.
    return {event, [](hipEvent_t handle) { static_cast<void>(hipEventDestroy(handle)); }};
}

/// The event pair one timer records into. Created once and re-recorded on every sample:
/// hipEventRecord() overwrites prior state.
struct HipEventPair
{
    hipdnn_data_sdk::utilities::ScopedResource<hipEvent_t> start = createScopedHipEvent();
    hipdnn_data_sdk::utilities::ScopedResource<hipEvent_t> stop = createScopedHipEvent();

    bool isUsable() const
    {
        return !start.isEmpty() && !stop.isEmpty();
    }
};

/// One timer's answer for one sample: an elapsed time when the launch could be
/// timed, whether a stall gate actually held the stream during it, and whether a
/// stall watchdog released the gate late instead of the host. A timeout always
/// leaves elapsedMs unset; healthy elapsed is finite and >= 0, including zero.
struct TimingResult
{
    std::optional<double> elapsedMs;
    bool stallUsed = false;
    bool timedOut = false;
};

/// One candidate's sampled timings. Only `robustMeanMs` decides the ranking; the rest
/// are logged as UHD training signal (RFC 0019.13 §8.3).
struct CandidateTiming
{
    double robustMeanMs = 0.0;
    double minMs = 0.0;
    double avgMs = 0.0;
    double stddevMs = 0.0;
    int iterations = 0;
};

/// What one sample attempt produced: a usable timing, an unusable candidate (neither
/// entered nor blamed on the stall gate), or a request to restart the whole
/// comparison unstalled.
struct SampleOutcome
{
    /// The candidate's timings; unset when it could not be timed or a sample was
    /// malformed. Either way scores the candidate unusable.
    std::optional<CandidateTiming> timing;
    /// Why @c timing is unset: the `reason` of the candidate's failed log record.
    std::string failureReason;
    /// True when the timed span cannot be trusted as a stalled measurement: either a
    /// watchdog timeout, or a valid sample the timer reports as unstalled while a
    /// stalled pass requested one. Neither is about the candidate itself, so the
    /// whole comparison must stop sampling immediately and restart unstalled.
    bool restartUnstalled = false;
};

/// Times one execute() of a plan, returning elapsed milliseconds or nullopt when it could not
/// be timed. An injected timer owns its synchronization and measurement method; no HIP
/// resources are acquired for it.
template <typename THandle>
using InjectedTimer = std::function<std::optional<double>(
    const IPlan<THandle>&, const THandle&, const hipdnnPluginDeviceBuffer_t*, uint32_t, void*)>;

/// The default timer: HIP events on handle.getStream() rather than the null stream,
/// so a plan on a non-default stream still measures its own work. @p events and
/// @p gate are owned by runWithStallRestart() for the lifetime of one comparison; this
/// closure only borrows them by reference, so nothing it touches outlives that call.
template <typename THandle>
auto makeDefaultTimer(std::optional<HipEventPair>& events,
                      std::optional<hipdnn_data_sdk::utilities::StallGate>& gate)
{
    return [&events, &gate, reusable = true](const IPlan<THandle>& plan,
                                             const THandle& handle,
                                             const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                             uint32_t numDeviceBuffers,
                                             void* workspace,
                                             bool stalled) mutable -> TimingResult {
        TimingResult result;
        if(!reusable)
        {
            return result;
        }

        if(!events.has_value())
        {
            events.emplace();
        }
        if(!events->isUsable())
        {
            // Discard the unusable pair so the next sample retries creation. Caching
            // it would turn one transient hipEventCreate failure into a permanently
            // untimeable comparison: every candidate's first timed iteration would
            // return unusable and selection would silently fall back to the ranked
            // front for the plan's whole life.
            events.reset();
            return result;
        }

        const auto start = events->start.get();
        const auto stop = events->stop.get();
        const auto stream = handle.getStream();

        bool armed = false;
        if(stalled)
        {
            armed = gate->arm(stream);
            // The previous sample is drained, and a failed arm enqueues no new
            // wait. Continue timing; a valid unstalled sample makes the caller
            // restart the comparison without stalling.
        }
        result.stallUsed = armed;

        // Release+drain on every exit once armed, even a throw out of plan.execute()
        // below: an escaping exception or an early return with the gate still armed
        // would leave the stream stalled, and a later arm() on the same gate requires
        // the previously armed stream to have already drained past its wait packet.
        // `drained` is only set once hipEventSynchronize(stop) below actually
        // succeeds -- release() alone does not prove the stream drained -- so a
        // synchronize failure still leaves this guard armed to release and drain by
        // its own hipStreamSynchronize instead of trusting the failed call.
        bool drained = false;
        struct ReleaseAndDrainOnUnwind
        {
            hipdnn_data_sdk::utilities::StallGate* gatePtr;
            hipStream_t stream;
            bool armed;
            const bool* drained;
            bool& reusable;
            ~ReleaseAndDrainOnUnwind()
            {
                if(armed && !*drained)
                {
                    gatePtr->release();
                    reusable = hipStreamSynchronize(stream) == hipSuccess;
                }
            }
        } const releaseGuard{armed ? &*gate : nullptr, stream, armed, &drained, reusable};

        if(hipEventRecord(start, stream) != hipSuccess)
        {
            return result;
        }

        plan.execute(handle, deviceBuffers, numDeviceBuffers, workspace);

        if(hipEventRecord(stop, stream) != hipSuccess)
        {
            return result;
        }

        // Must precede the synchronize: a still-stalled stream never signals stop.
        if(armed)
        {
            gate->release();
        }

        if(hipEventSynchronize(stop) != hipSuccess)
        {
            return result;
        }
        drained = true;

        // The watchdog permits work to run before host submission finishes.
        // Its elapsed span is not a device-only measurement, regardless of cause.
        if(armed && gate->timedOut())
        {
            result.timedOut = true;
            return result;
        }

        float elapsedMs = 0.0F;
        if(hipEventElapsedTime(&elapsedMs, start, stop) != hipSuccess)
        {
            return result;
        }
        result.elapsedMs = static_cast<double>(elapsedMs);
        return result;
    };
}

/// min, mean, population stddev, and count alongside the ranking statistic. Population
/// stddev because the samples are every iteration that ran, not a draw.
inline CandidateTiming summarize(const std::vector<double>& samples)
{
    CandidateTiming timing;
    timing.iterations = static_cast<int>(samples.size());
    timing.robustMeanMs = hipdnn_data_sdk::utilities::detail::robustMean(samples);
    timing.minMs = *std::min_element(samples.begin(), samples.end());

    double total = 0.0;
    for(const double sample : samples)
    {
        total += sample;
    }
    timing.avgMs = total / static_cast<double>(samples.size());

    double squaredError = 0.0;
    for(const double sample : samples)
    {
        const double error = sample - timing.avgMs;
        squaredError += error * error;
    }
    timing.stddevMs = std::sqrt(squaredError / static_cast<double>(samples.size()));

    return timing;
}

/// The timings of BENCHMARK_ITERATIONS timed executes of @p plan, after BENCHMARK_WARMUP_RUNS
/// untimed ones. Any timed iteration that could not actually be
/// measured stalled while @p stalled was requested -- a watchdog timeout, or a valid
/// sample the timer reports as unstalled -- returns immediately with restartUnstalled
/// set; the two populations must never be averaged together. Both checks run before
/// the sign check below, so a negative sample can never hide a restart the comparison
/// actually needs. A backend error (no elapsed time at all) or a non-finite sample
/// (NaN/Inf) scores the candidate unusable immediately, with no retry. A finite
/// negative sample is instead treated as a bad HIP event reading: it is discarded and
/// the same slot re-measured, using at most MAX_NEGATIVE_SAMPLE_RETRIES extra attempts
/// for this pass; exhausting that budget scores the candidate unusable without the
/// malformed value ever reaching the reduction, ranking, or cache.
///
/// Ranks by robustMean() rather than the fastest sample, so an occasionally lucky but
/// usually slower kernel cannot win. Each failure path names its reason; the caller
/// logs it once the comparison settles.
/// @param describe Names the timed candidate for the log lines; called only when one is written.
/// @param injected The injected timer, or empty to time with @p timer.
template <typename THandle, typename TDescribe, typename TDefaultTimer>
SampleOutcome sampleRuns(const IPlan<THandle>& plan,
                         const TDescribe& describe,
                         const THandle& handle,
                         const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                         uint32_t numDeviceBuffers,
                         void* workspace,
                         const InjectedTimer<THandle>& injected,
                         TDefaultTimer& timer,
                         bool stalled)
{
    try
    {
        for(int warmup = 0; warmup < BENCHMARK_WARMUP_RUNS; ++warmup)
        {
            plan.execute(handle, deviceBuffers, numDeviceBuffers, workspace);
        }

        // Only the default HIP timer needs a warmup drain. Injected timers own
        // their synchronization and can run without a HIP device.
        if(!injected && hipStreamSynchronize(handle.getStream()) != hipSuccess)
        {
            HIPDNN_PLUGIN_LOG_WARN("ingestor: benchmarking candidate '"
                                   << describe()
                                   << "' failed to drain its warmup runs; scored unusable");
            return {std::nullopt, "warmup-not-drained", false};
        }

        std::vector<double> samples;
        samples.reserve(BENCHMARK_ITERATIONS);
        int negativeSampleRetriesLeft = MAX_NEGATIVE_SAMPLE_RETRIES;
        for(int iteration = 0; iteration < BENCHMARK_ITERATIONS;)
        {
            const TimingResult sample
                = injected
                      ? TimingResult{injected(
                                         plan, handle, deviceBuffers, numDeviceBuffers, workspace),
                                     false,
                                     false}
                      : timer(plan, handle, deviceBuffers, numDeviceBuffers, workspace, stalled);

            if(sample.timedOut)
            {
                HIPDNN_PLUGIN_LOG_WARN("ingestor: benchmarking candidate '"
                                       << describe()
                                       << "' hit a stall watchdog timeout; the whole "
                                          "comparison will restart unstalled");
                return {std::nullopt, "stall-watchdog-timeout", true};
            }
            if(!sample.elapsedMs.has_value())
            {
                HIPDNN_PLUGIN_LOG_WARN("ingestor: benchmarking candidate '"
                                       << describe()
                                       << "' failed to time a launch; scored unusable");
                return {std::nullopt, "launch-not-timed", false};
            }
            if(stalled && !sample.stallUsed)
            {
                // A genuine measurement, but not a stalled one (unsupported device, a
                // transient arm failure, ...). Mixing it into a pass whose earlier
                // samples were actually stalled would rank two incomparable
                // populations exactly like a watchdog timeout, so it gets the same
                // whole-comparison restart rather than being accepted here. Checked
                // before the sign check below: a negative elapsed time must never
                // mask this restart behind a per-sample retry.
                HIPDNN_PLUGIN_LOG_WARN(
                    "ingestor: benchmarking candidate '"
                    << describe()
                    << "' measured without an actual stall during a stalled pass; the "
                       "whole comparison will restart unstalled");
                return {std::nullopt, "measured-unstalled", true};
            }
            if(!std::isfinite(*sample.elapsedMs))
            {
                // NaN/Inf is never a transient timing artifact -- it means the event
                // pair itself is broken -- so it is scored unusable immediately, with
                // no retry, exactly like the backend error above.
                HIPDNN_PLUGIN_LOG_WARN("ingestor: benchmarking candidate '"
                                       << describe()
                                       << "' reported a non-finite elapsed time; scored unusable");
                return {std::nullopt, "non-finite-elapsed-time", false};
            }
            if(*sample.elapsedMs < 0.0)
            {
                // A finite negative elapsed time is a known HIP-event-timing artifact
                // near the clock's resolution floor, not proof the candidate is
                // broken: discard it and re-measure the same slot rather than
                // condemning the whole candidate on one bad reading. A persistently
                // negative candidate still exhausts the retry budget and falls
                // through to the unusable return below, never entering the
                // reduction, ranking, or cache.
                if(negativeSampleRetriesLeft > 0)
                {
                    --negativeSampleRetriesLeft;
                    HIPDNN_PLUGIN_LOG_WARN("ingestor: benchmarking candidate '"
                                           << describe()
                                           << "' reported a negative elapsed time; "
                                              "discarding it and re-measuring");
                    continue;
                }
                HIPDNN_PLUGIN_LOG_WARN(
                    "ingestor: benchmarking candidate '"
                    << describe() << "' reported a negative elapsed time after exhausting "
                    << MAX_NEGATIVE_SAMPLE_RETRIES << " retries; scored unusable");
                return {std::nullopt, "negative-elapsed-time", false};
            }
            samples.push_back(*sample.elapsedMs);
            ++iteration;
        }
        return {summarize(samples), {}, false};
    }
    catch(const std::exception& error)
    {
        HIPDNN_PLUGIN_LOG_WARN("ingestor: benchmarking candidate '"
                               << describe() << "' threw and is scored unusable: " << error.what());
        return {std::nullopt, error.what(), false};
    }
    catch(...)
    {
        // IKernelDispatchHandler::launch() and an injected Timer are both extension
        // points with no exception-type contract. Letting a non-std::exception escape
        // would leave the plan unresolved, so every later execute() would re-run the
        // whole comparison and re-hit this candidate. Score it unusable like any other
        // failure instead.
        HIPDNN_PLUGIN_LOG_WARN("ingestor: benchmarking candidate '"
                               << describe()
                               << "' threw a non-standard exception and is scored unusable");
        return {std::nullopt, "non-standard-exception", false};
    }
}

/// Runs one comparison, restarting it unstalled when stalled timing proves unavailable.
///
/// An injected timer is used as-is; it already owns whatever it needs. Without one, the default
/// timer's events and stall gate are local to this call: created here, reused across every
/// sample and pass, and freed when this function returns.
///
/// A candidate that could not actually be measured stalled -- whether a stall watchdog timed
/// out, or the gate simply never held the stream (unsupported device, a transient arm failure)
/// -- cannot be mixed with the rest of the pass's genuinely stalled measurements: that would
/// rank two incomparable populations. So @p pass stops sampling the instant either happens --
/// candidates not yet reached are never sampled stalled at all -- and returns nullopt, and
/// every candidate is re-measured unstalled from scratch. This policy is local to this one
/// comparison: it never reads or writes any state shared with another comparison, another
/// gate, or a standalone timed execution elsewhere.
///
/// @param pass Called as `pass(defaultTimer, stalled)`; returns the comparison's result as an
///        optional, empty when a sample asked for the restart. It is never asked to restart
///        again once @c stalled is false.
template <typename THandle, typename TPass>
auto runWithStallRestart(bool hasInjectedTimer, TPass&& pass)
{
    std::optional<HipEventPair> defaultEvents;
    std::optional<hipdnn_data_sdk::utilities::StallGate> defaultGate;
    bool stalled = false;
    if(!hasInjectedTimer)
    {
        defaultGate.emplace();
        stalled = defaultGate->isUsable();
    }
    auto defaultTimer = makeDefaultTimer<THandle>(defaultEvents, defaultGate);

    for(;;)
    {
        auto result = pass(defaultTimer, stalled);
        if(result.has_value())
        {
            return std::move(*result);
        }

        HIPDNN_PLUGIN_LOG_WARN(
            "ingestor: stalled timing was unavailable or timed out during benchmarking. "
            "Discarding the pass and re-measuring every candidate unstalled.");
        // The retry runs with stalled=false, so no sample from it can trigger
        // another restart: this can happen at most once.
        stalled = false;
    }
}

} // namespace detail
} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
