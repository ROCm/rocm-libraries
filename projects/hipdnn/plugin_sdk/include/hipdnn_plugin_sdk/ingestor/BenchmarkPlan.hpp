// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/ScopedResource.hpp>
#include <hipdnn_data_sdk/utilities/StallGate.hpp>
#include <hipdnn_data_sdk/utilities/TimingStatistics.hpp>
#include <hipdnn_plugin_sdk/EnginePluginTypeTraits.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/WinnerCache.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// Sampling counts: one untimed warmup and seven valid timed runs per candidate,
/// matching MIOpen's EvaluateInvokers before replacement attempts.
constexpr int BENCHMARK_WARMUP_RUNS = 1;
constexpr int BENCHMARK_ITERATIONS = 7;

/// Extra attempts per candidate to replace finite negative samples, a HIP event timing
/// artifact. See sampleCandidate() for the policy.
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

} // namespace detail

/// An IPlan owning one GenericPlan per knob-filtered catalog entry. Times each candidate on
/// the first execute() and delegates every call to the fastest. Timing goes through
/// IPlan::execute() only; this class touches no dispatcher or HIP launch API.
template <typename THandle>
class BenchmarkPlan : public IPlan<THandle>
{
    static_assert(HasGetStream<THandle>::value,
                  "BenchmarkPlan requires THandle to have a 'hipStream_t getStream() const' "
                  "method: the default timer brackets each comparison's candidates with HIP "
                  "events on that stream. Required even when a Timer is injected, since the "
                  "default-timer path is still compiled for every THandle.");

public:
    /// A sub-plan and its kernel ids. Typed on IPlan so tests can substitute doubles; IPlan
    /// has no kernel accessor, so the ids ride along: kernelId for logging,
    /// packId/dispatchId to validate a cached ranking on a later run.
    struct Candidate
    {
        DescriptorId kernelId;
        std::unique_ptr<IPlan<THandle>> plan;
        DescriptorId packId{};
        DescriptorId dispatchId{};
    };

    /// Times one execute() of a candidate, returning elapsed milliseconds or nullopt
    /// when it could not be timed. An injected timer owns its synchronization and
    /// measurement method; no HIP resources are acquired for it.
    using Timer = std::function<std::optional<double>(
        const IPlan<THandle>&, const THandle&, const hipdnnPluginDeviceBuffer_t*, uint32_t, void*)>;
    /// Receives every usable candidate in ranked order once the winner is resolved. Absent
    /// means no caching.
    using RecordRankingFn = std::function<void(std::vector<RankedEntry>)>;

    /// @param handle Sizes every sub-plan's workspace requirement.
    /// @param timer Overrides the default HIP-event timer. Called only from the sampling
    ///        sweep, under _mutex, so it need not be thread-safe. When empty, each
    ///        comparison builds its own events and stall gate (see resolveChosen()).
    /// @throws HipdnnPluginException(INTERNAL_ERROR) if @p candidates is empty.
    BenchmarkPlan(std::vector<Candidate> candidates,
                  const THandle& handle,
                  Timer timer = {},
                  RecordRankingFn recordRanking = {})
        : _candidates(std::move(candidates))
        , _timer(std::move(timer))
        , _recordRanking(std::move(recordRanking))
    {
        if(_candidates.empty())
        {
            throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                        "BenchmarkPlan constructed with no candidates");
        }

        for(const auto& candidate : _candidates)
        {
            _workspaceBytes = std::max(_workspaceBytes, candidate.plan->getWorkspaceSize(handle));
        }
    }

    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    size_t getWorkspaceSize(const THandle& /*handle*/) const override
    {
        return _workspaceBytes;
    }

    /// Sampling runs candidates against the caller's buffers, so a candidate failing
    /// mid-loop can leave a partial result behind. The delegated execute below overwrites
    /// it with the winner's output; never add an early return before that delegation.
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    void execute(const THandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override
    {
        // Lock-free after resolution: _chosen never changes once written, and no
        // benchmarking resource is touched on this path.
        size_t chosen = _chosen.load(std::memory_order_acquire);
        if(chosen == NOT_RESOLVED)
        {
            chosen = resolveChosen(handle, deviceBuffers, numDeviceBuffers, workspace);
        }
        _candidates[chosen].plan->execute(handle, deviceBuffers, numDeviceBuffers, workspace);
    }

private:
    static constexpr size_t NOT_RESOLVED = std::numeric_limits<size_t>::max();
    /// One timer sample: the elapsed time when timed, whether the stall gate held the
    /// stream, and whether the watchdog (not the host) released it. A timeout leaves
    /// elapsedMs unset.
    struct TimingResult
    {
        std::optional<double> elapsedMs;
        bool stallUsed = false;
        bool timedOut = false;
    };

    /// One candidate's result: a usable time, unusable (timeMs unset), or a request to
    /// restart the whole comparison unstalled (see sampleCandidate()).
    struct SampleOutcome
    {
        std::optional<double> timeMs;
        bool restartUnstalled = false;
    };

    /// The default timer: HIP events on handle.getStream(), so a plan on a non-default
    /// stream measures its own work. Borrows @p events and @p gate, which resolveChosen()
    /// owns for one comparison.
    static auto makeDefaultTimer(std::optional<detail::HipEventPair>& events,
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

    /// Resolves and caches _chosen. The lock spans the whole comparison, so a racing
    /// first execute() blocks instead of sampling against another thread's buffers.
    size_t resolveChosen(const THandle& handle,
                         const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                         uint32_t numDeviceBuffers,
                         void* workspace) const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        if(const size_t resolved = _chosen.load(std::memory_order_relaxed);
           resolved != NOT_RESOLVED)
        {
            return resolved;
        }

        // Without an injected timer, the events and stall gate live for this comparison
        // only and are freed before the chosen plan's execute().
        std::optional<detail::HipEventPair> defaultEvents;
        std::optional<hipdnn_data_sdk::utilities::StallGate> defaultGate;
        bool stalled = false;
        if(!_timer)
        {
            defaultGate.emplace();
            stalled = defaultGate->isUsable();
        }
        auto defaultTimer = makeDefaultTimer(defaultEvents, defaultGate);

        // Keep every usable time, so a later knob filter that excludes the winner can
        // still serve the runner-up.
        std::vector<std::pair<double, size_t>> ranked;
        ranked.reserve(_candidates.size());

        // Stalled and unstalled samples are not comparable. If any candidate cannot be
        // measured stalled (see sampleCandidate()), stop the pass and re-measure every
        // candidate unstalled. This policy is local to this comparison.
        for(;;)
        {
            ranked.clear();
            bool restartUnstalled = false;
            for(size_t index = 0; index < _candidates.size(); ++index)
            {
                const auto outcome = sampleCandidate(index,
                                                     handle,
                                                     deviceBuffers,
                                                     numDeviceBuffers,
                                                     workspace,
                                                     defaultTimer,
                                                     stalled);
                if(stalled && outcome.restartUnstalled)
                {
                    // Candidates after this index stay unsampled in this pass.
                    restartUnstalled = true;
                    break;
                }
                if(!outcome.timeMs.has_value())
                {
                    // Omitted, never given a sentinel time that could outrank real ones.
                    continue;
                }
                ranked.emplace_back(*outcome.timeMs, index);
            }

            if(!restartUnstalled)
            {
                break;
            }

            HIPDNN_PLUGIN_LOG_WARN(
                "ingestor: stalled timing was unavailable, timed out, or kept reading "
                "negative during benchmarking. "
                "Discarding the pass and re-measuring every candidate unstalled.");
            // The unstalled pass cannot request a restart, so this runs at most once.
            stalled = false;
        }

        // stable_sort: ties resolve to the lowest candidate index.
        std::stable_sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.first < rhs.first;
        });

        size_t best = 0;
        if(ranked.empty())
        {
            HIPDNN_PLUGIN_LOG_ERROR("ingestor: benchmarking found no usable candidate among "
                                    << _candidates.size() << " kernel(s); defaulting to "
                                    << toString(_candidates.front().kernelId));
            // Nothing is recorded here: an all-unusable sweep has no ranking to cache.
        }
        else
        {
            best = ranked.front().second;
            HIPDNN_PLUGIN_LOG_INFO("ingestor: benchmarking selected kernel "
                                   << toString(_candidates[best].kernelId) << " in "
                                   << ranked.front().first << " ms among " << _candidates.size()
                                   << " candidate(s)");

            if(_recordRanking)
            {
                std::vector<RankedEntry> entries;
                entries.reserve(ranked.size());
                for(const auto& [timeMs, index] : ranked)
                {
                    const auto& candidate = _candidates[index];
                    entries.push_back(RankedEntry{
                        candidate.kernelId, candidate.packId, candidate.dispatchId, timeMs});
                }
                _recordRanking(std::move(entries));
            }
        }

        _chosen.store(best, std::memory_order_release);
        return best;
    }

    /// Times BENCHMARK_ITERATIONS executes after BENCHMARK_WARMUP_RUNS untimed ones and
    /// reduces them with robustMean(), so an occasionally lucky kernel cannot win on one
    /// fast sample.
    ///
    /// - In a stalled pass, a watchdog timeout or an unstalled sample returns
    ///   restartUnstalled: the two populations must never be mixed. These checks precede
    ///   the sign check, so a negative sample cannot hide them.
    /// - A missing or non-finite time scores the candidate unusable, with no retry.
    /// - A finite negative time is re-measured in the same slot, at most
    ///   MAX_NEGATIVE_SAMPLE_RETRIES times per candidate. Negatives appear near the timer
    ///   resolution and, for short stalled work on some runtimes (seen on Windows),
    ///   persistently. Exhausting the budget restarts a stalled pass unstalled, because
    ///   dropping the candidate would exclude the fastest kernels; in an unstalled pass it
    ///   scores the candidate unusable. A negative value never reaches the ranking or
    ///   cache.
    template <typename DefaultTimer>
    SampleOutcome sampleCandidate(size_t index,
                                  const THandle& handle,
                                  const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                  uint32_t numDeviceBuffers,
                                  void* workspace,
                                  DefaultTimer& timer,
                                  bool stalled) const
    {
        const auto& candidate = _candidates[index];
        try
        {
            for(int warmup = 0; warmup < BENCHMARK_WARMUP_RUNS; ++warmup)
            {
                candidate.plan->execute(handle, deviceBuffers, numDeviceBuffers, workspace);
            }

            // Only the default HIP timer needs a warmup drain. Injected timers own
            // their synchronization and can run without a HIP device.
            if(!_timer && hipStreamSynchronize(handle.getStream()) != hipSuccess)
            {
                HIPDNN_PLUGIN_LOG_WARN("ingestor: benchmarking candidate '"
                                       << toString(candidate.kernelId)
                                       << "' failed to drain its warmup runs; scored unusable");
                return {std::nullopt, false};
            }

            std::vector<double> samples;
            samples.reserve(BENCHMARK_ITERATIONS);
            int negativeSampleRetriesLeft = MAX_NEGATIVE_SAMPLE_RETRIES;
            for(int iteration = 0; iteration < BENCHMARK_ITERATIONS;)
            {
                const TimingResult sample = _timer ? TimingResult{_timer(*candidate.plan,
                                                                         handle,
                                                                         deviceBuffers,
                                                                         numDeviceBuffers,
                                                                         workspace),
                                                                  false,
                                                                  false}
                                                   : timer(*candidate.plan,
                                                           handle,
                                                           deviceBuffers,
                                                           numDeviceBuffers,
                                                           workspace,
                                                           stalled);

                if(sample.timedOut)
                {
                    HIPDNN_PLUGIN_LOG_WARN("ingestor: benchmarking candidate '"
                                           << toString(candidate.kernelId)
                                           << "' hit a stall watchdog timeout; the whole "
                                              "comparison will restart unstalled");
                    return {std::nullopt, true};
                }
                if(!sample.elapsedMs.has_value())
                {
                    HIPDNN_PLUGIN_LOG_WARN("ingestor: benchmarking candidate '"
                                           << toString(candidate.kernelId)
                                           << "' failed to time a launch; scored unusable");
                    return {std::nullopt, false};
                }
                if(stalled && !sample.stallUsed)
                {
                    // Measured but not stalled (unsupported device, transient arm failure):
                    // restart like a timeout.
                    HIPDNN_PLUGIN_LOG_WARN(
                        "ingestor: benchmarking candidate '"
                        << toString(candidate.kernelId)
                        << "' measured without an actual stall during a stalled pass; the "
                           "whole comparison will restart unstalled");
                    return {std::nullopt, true};
                }
                if(!std::isfinite(*sample.elapsedMs))
                {
                    // NaN/Inf means the event pair is broken: no retry.
                    HIPDNN_PLUGIN_LOG_WARN(
                        "ingestor: benchmarking candidate '"
                        << toString(candidate.kernelId)
                        << "' reported a non-finite elapsed time; scored unusable");
                    return {std::nullopt, false};
                }
                if(*sample.elapsedMs < 0.0)
                {
                    // Re-measure in place; see the sampleCandidate() doc for the budget.
                    if(negativeSampleRetriesLeft > 0)
                    {
                        --negativeSampleRetriesLeft;
                        HIPDNN_PLUGIN_LOG_WARN("ingestor: benchmarking candidate '"
                                               << toString(candidate.kernelId)
                                               << "' reported a negative elapsed time; "
                                                  "discarding it and re-measuring");
                        continue;
                    }
                    if(stalled)
                    {
                        HIPDNN_PLUGIN_LOG_WARN("ingestor: benchmarking candidate '"
                                               << toString(candidate.kernelId)
                                               << "' reported a negative stalled elapsed time "
                                                  "after exhausting "
                                               << MAX_NEGATIVE_SAMPLE_RETRIES
                                               << " retries; the whole comparison will "
                                                  "restart unstalled");
                        return {std::nullopt, true};
                    }
                    HIPDNN_PLUGIN_LOG_WARN("ingestor: benchmarking candidate '"
                                           << toString(candidate.kernelId)
                                           << "' reported a negative elapsed time after exhausting "
                                           << MAX_NEGATIVE_SAMPLE_RETRIES
                                           << " retries; scored unusable");
                    return {std::nullopt, false};
                }
                samples.push_back(*sample.elapsedMs);
                ++iteration;
            }
            return {hipdnn_data_sdk::utilities::detail::robustMean(samples), false};
        }
        catch(const std::exception& error)
        {
            HIPDNN_PLUGIN_LOG_WARN("ingestor: benchmarking candidate '"
                                   << toString(candidate.kernelId)
                                   << "' threw and is scored unusable: " << error.what());
            return {std::nullopt, false};
        }
        catch(...)
        {
            // launch() and an injected Timer have no exception contract. An escaping
            // exception would leave _chosen unresolved and re-run the comparison on
            // every execute().
            HIPDNN_PLUGIN_LOG_WARN("ingestor: benchmarking candidate '"
                                   << toString(candidate.kernelId)
                                   << "' threw a non-standard exception and is scored unusable");
            return {std::nullopt, false};
        }
    }

    std::vector<Candidate> _candidates;
    Timer _timer;
    RecordRankingFn _recordRanking;
    size_t _workspaceBytes = 0;
    mutable std::atomic<size_t> _chosen{NOT_RESOLVED};
    mutable std::mutex _mutex;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
