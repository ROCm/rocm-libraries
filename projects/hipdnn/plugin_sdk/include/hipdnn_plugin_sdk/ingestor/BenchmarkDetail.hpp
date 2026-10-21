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
#include <nlohmann/json.hpp>

#include <hipdnn_data_sdk/utilities/ScopedResource.hpp>
#include <hipdnn_data_sdk/utilities/StallGate.hpp>
#include <hipdnn_data_sdk/utilities/TimingStatistics.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
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

// With zero iterations, the reduction would report its DBL_MAX seed as a real time.
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

/// The event pair one timer records into. Created once and re-recorded on every sample.
struct HipEventPair
{
    hipdnn_data_sdk::utilities::ScopedResource<hipEvent_t> start = createScopedHipEvent();
    hipdnn_data_sdk::utilities::ScopedResource<hipEvent_t> stop = createScopedHipEvent();

    bool isUsable() const
    {
        return !start.isEmpty() && !stop.isEmpty();
    }
};

/// One timer sample: the elapsed time when timed, whether the stall gate held the
/// stream, and whether the watchdog (not the host) released it. A timeout leaves
/// elapsedMs unset.
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

/// One candidate's result: usable timings, unusable (timing unset), or a request to
/// restart the whole comparison unstalled (see sampleRuns()).
struct SampleOutcome
{
    std::optional<CandidateTiming> timing;
    /// Why @c timing is unset: the `reason` of the candidate's failed log record.
    std::string failureReason;
    bool restartUnstalled = false;
};

/// The fields common to every candidate log row, layered over the candidate's @p features.
/// `benchmark` plus `device` identify one problem, as the winner cache keys on both; `device` is
/// omitted when unidentified. The envelope is written last so a feature key can never overwrite
/// the row's identity.
inline nlohmann::json candidateRecord(const nlohmann::json& features,
                                      const std::string& benchmarkId,
                                      const std::string& deviceIdentity,
                                      const DescriptorId& kernelId,
                                      const DescriptorId& packId,
                                      const DescriptorId& dispatchId)
{
    nlohmann::json record = features;
    record["event"] = "ingestor.benchmark.candidate";
    record["benchmark"] = benchmarkId;
    if(!deviceIdentity.empty())
    {
        record["device"] = deviceIdentity;
    }
    record["kernel"] = toString(kernelId);
    record["pack"] = toString(packId);
    record["dispatch"] = toString(dispatchId);
    return record;
}

/// Marks @p record `ok` with the timings a UHD is trained on.
inline void annotateTiming(nlohmann::json& record, const CandidateTiming& timing)
{
    record["status"] = "ok";
    record["min_ms"] = timing.minMs;
    record["avg_ms"] = timing.avgMs;
    record["stddev_ms"] = timing.stddevMs;
    record["robust_mean_ms"] = timing.robustMeanMs;
    record["iters"] = timing.iterations;
}

/// Marks @p record `failed` with its reason (RFC 0019.13 §8.3).
inline void annotateFailure(nlohmann::json& record, const std::string& reason)
{
    record["status"] = "failed";
    record["reason"] = reason;
}

/// Times one execute() of a plan, returning elapsed milliseconds or nullopt when it could not
/// be timed. An injected timer owns its synchronization and measurement method; no HIP
/// resources are acquired for it.
template <typename THandle>
using InjectedTimer = std::function<std::optional<double>(
    const IPlan<THandle>&, const THandle&, const hipdnnPluginDeviceBuffer_t*, uint32_t, void*)>;

/// The default timer: HIP events on handle.getStream(), so a plan on a non-default
/// stream measures its own work. Borrows @p events and @p gate, which runWithStallRestart()
/// owns for one comparison.
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
            // Drop the pair so the next sample retries creation; caching it would let
            // one transient failure make the whole comparison untimeable.
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
            // A failed arm enqueues nothing; the resulting unstalled sample makes the
            // caller restart the comparison.
        }
        result.stallUsed = armed;

        // Once armed, release and drain on every exit, including a throw from
        // execute(): a stalled stream never drains, and re-arming requires the previous
        // wait to have retired. Only a successful hipEventSynchronize(stop) sets
        // `drained`; release() alone does not prove it.
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

        // A watchdog release lets work run before submission finishes, so the span is
        // not device-only.
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

/// Times BENCHMARK_ITERATIONS executes of @p plan after BENCHMARK_WARMUP_RUNS untimed ones
/// and ranks by robustMean(), so an occasionally lucky kernel cannot win on one fast
/// sample.
///
/// - In a stalled pass, a watchdog timeout or an unstalled sample returns
///   restartUnstalled: the two populations must never be mixed. These checks precede
///   the sign check, so a negative sample cannot hide them.
/// - A missing or non-finite time scores the candidate unusable, with no retry.
/// - A finite negative time is re-measured in the same slot, at most
///   MAX_NEGATIVE_SAMPLE_RETRIES times per candidate in each pass. The Windows HIP
///   runtime reports negative spans (ROCm/rocm-systems#12925), and for short stalled
///   work it can report them persistently. Exhausting the budget restarts a stalled
///   pass unstalled, because dropping the candidate would exclude the fastest
///   kernels; in an unstalled pass it scores the candidate unusable. A negative value
///   never reaches the ranking or cache.
///
/// Each failure path names its reason; the caller logs it once the comparison settles.
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
                // Measured but not stalled (unsupported device, transient arm failure):
                // restart like a timeout.
                HIPDNN_PLUGIN_LOG_WARN(
                    "ingestor: benchmarking candidate '"
                    << describe()
                    << "' measured without an actual stall during a stalled pass; the "
                       "whole comparison will restart unstalled");
                return {std::nullopt, "measured-unstalled", true};
            }
            if(!std::isfinite(*sample.elapsedMs))
            {
                // NaN/Inf means the event pair is broken: no retry.
                HIPDNN_PLUGIN_LOG_WARN("ingestor: benchmarking candidate '"
                                       << describe()
                                       << "' reported a non-finite elapsed time; scored unusable");
                return {std::nullopt, "non-finite-elapsed-time", false};
            }
            if(*sample.elapsedMs < 0.0)
            {
                // Re-measure in place; see the sampleRuns() doc for the budget.
                if(negativeSampleRetriesLeft > 0)
                {
                    --negativeSampleRetriesLeft;
                    HIPDNN_PLUGIN_LOG_WARN("ingestor: benchmarking candidate '"
                                           << describe()
                                           << "' reported a negative elapsed time; "
                                              "discarding it and re-measuring");
                    continue;
                }
                if(stalled)
                {
                    HIPDNN_PLUGIN_LOG_WARN("ingestor: benchmarking candidate '"
                                           << describe()
                                           << "' reported a negative stalled elapsed time "
                                              "after exhausting "
                                           << MAX_NEGATIVE_SAMPLE_RETRIES
                                           << " retries; the whole comparison will "
                                              "restart unstalled");
                    return {std::nullopt, "negative-stalled-elapsed-time", true};
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
        // launch() and an injected Timer have no exception contract. An escaping
        // exception would leave the plan unresolved and re-run the comparison on
        // every execute().
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
/// Stalled and unstalled samples are not comparable. If any candidate cannot be measured
/// stalled (see sampleRuns()), @p pass stops sampling, returns nullopt, and every candidate is
/// re-measured unstalled. This policy is local to this comparison.
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
            "ingestor: stalled timing was unavailable, timed out, or kept reading "
            "negative during benchmarking. "
            "Discarding the pass and re-measuring every candidate unstalled.");
        // The unstalled pass cannot request a restart, so this runs at most once.
        stalled = false;
    }
}

} // namespace detail
} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
