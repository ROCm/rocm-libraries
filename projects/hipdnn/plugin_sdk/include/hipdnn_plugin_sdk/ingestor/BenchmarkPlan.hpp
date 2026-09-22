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

/// Sampling counts, matching MIOpen's EvaluateInvokers: one untimed warmup followed by
/// up to eight total runs per candidate.
constexpr int BENCHMARK_WARMUP_RUNS = 1;
constexpr int BENCHMARK_ITERATIONS = 7;

// A zero iteration count would leave sampleCandidate()'s reduction at its DBL_MAX seed
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

} // namespace detail

/// An IPlan owning one GenericPlan per knob-filtered catalog entry. Times each candidate on
/// the first execute() and delegates every call to the one that measured fastest. Wraps
/// GenericPlan rather than widening it, leaving single-kernel construction, workspace
/// query, and the null-prepared check unchanged.
///
/// Timing goes through IPlan::execute() only; this class touches no dispatcher,
/// PreparedDispatch, or HIP launch API.
template <typename THandle>
class BenchmarkPlan : public IPlan<THandle>
{
    static_assert(HasGetStream<THandle>::value,
                  "BenchmarkPlan requires THandle to have a 'hipStream_t getStream() const' "
                  "method: the default timer brackets each comparison's candidates with HIP "
                  "events on that stream. Required even when a Timer is injected, since the "
                  "default-timer path is still compiled for every THandle.");

public:
    /// A sub-plan and the kernel it was built for. The vector is typed on IPlan rather
    /// than GenericPlan so tests can substitute doubles, and IPlan has no kernel
    /// accessor, so the ids ride alongside: kernelId for the selection log,
    /// packId/dispatchId as the staleness cross-check a cached ranking is validated
    /// against on a later run.
    struct Candidate
    {
        DescriptorId kernelId;
        std::unique_ptr<IPlan<THandle>> plan;
        DescriptorId packId{};
        DescriptorId dispatchId{};
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

    /// Times one execute() of a candidate, honoring @p stalled by attempting to hold the
    /// stream stalled through the timed span when true. Defaults to HIP events on the
    /// handle's stream; tests substitute a deterministic timer so selection is provable
    /// without a device.
    using Timer = std::function<TimingResult(const IPlan<THandle>&,
                                             const THandle&,
                                             const hipdnnPluginDeviceBuffer_t*,
                                             uint32_t,
                                             void*,
                                             bool stalled)>;
    /// Invoked once, with every usable candidate in benchmarked order, after sampling
    /// resolves the winner. An absent callback means no caching, so BenchmarkPlan needs
    /// no knowledge of the cache's type or lifetime.
    using RecordRankingFn = std::function<void(std::vector<RankedEntry>)>;

    /// @param handle Sizes every sub-plan's workspace requirement; execute() uses the
    ///        handle its own caller passes.
    /// @param timer Overrides the default HIP-event timer. An empty @p timer keeps the
    ///        default: resolveChosen() then builds one locally, scoped to that one
    ///        comparison, rather than holding HIP events and a stall gate for this
    ///        plan's whole life. Only ever called from the sampling sweep, which holds
    ///        _mutex, so it need not be thread-safe.
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
        // Post-resolution reads take no lock: _chosen never changes once written, so the
        // steady-state path the feature exists for has no serialization point, and no
        // benchmarking resource (event pair, stall gate) is ever touched on it.
        size_t chosen = _chosen.load(std::memory_order_acquire);
        if(chosen == NOT_RESOLVED)
        {
            chosen = resolveChosen(handle, deviceBuffers, numDeviceBuffers, workspace);
        }
        _candidates[chosen].plan->execute(handle, deviceBuffers, numDeviceBuffers, workspace);
    }

private:
    static constexpr size_t NOT_RESOLVED = std::numeric_limits<size_t>::max();

    /// What one sample attempt produced: a usable time, an unusable sample (neither
    /// entered nor blamed on the stall gate), or a request to restart the whole
    /// comparison unstalled.
    struct SampleOutcome
    {
        /// The candidate's representative time; unset when it could not be timed or the
        /// sample was malformed. Either way scores the candidate unusable.
        std::optional<double> timeMs;
        /// True when the timed span cannot be trusted as a stalled measurement: either a
        /// watchdog timeout, or a valid sample the timer reports as unstalled while a
        /// stalled pass requested one. Neither is about the candidate itself, so the
        /// whole comparison must stop sampling immediately and restart unstalled.
        bool restartUnstalled = false;
    };

    /// The default timer: HIP events on handle.getStream() rather than the null stream,
    /// so a plan on a non-default stream still measures its own work. @p events and
    /// @p gate are owned by resolveChosen() for the lifetime of one comparison; this
    /// closure only borrows them by reference, so nothing it touches outlives that call.
    static Timer makeDefaultTimer(std::optional<detail::HipEventPair>& events,
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
                if(!gate.has_value())
                {
                    gate.emplace();
                }
                armed = gate->arm(stream);
                if(!armed && gate->isUsable() && gate->lastError() != hipSuccess)
                {
                    // An acquired gate failed to enqueue its wait. An unavailable gate,
                    // by contrast, still permits plain event timing.
                    return result;
                }
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

    /// Resolves _chosen on the first call and caches it. The lock spans the whole
    /// comparison, so a second thread racing the first execute() blocks instead of
    /// sampling against the first thread's buffers.
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

        // An injected _timer is used as-is; it already owns whatever it needs. Without
        // one, the default timer's events and stall gate are local to this call: created
        // here, reused across every sample and pass below, and freed when this function
        // returns -- before the ordinary chosen-plan execute() that follows.
        std::optional<detail::HipEventPair> defaultEvents;
        std::optional<hipdnn_data_sdk::utilities::StallGate> defaultGate;
        Timer defaultTimer;
        const Timer* timer = &_timer;
        if(!_timer)
        {
            defaultTimer = makeDefaultTimer(defaultEvents, defaultGate);
            timer = &defaultTimer;
        }

        // Every usable candidate's time is retained so a later run whose knob filter
        // excludes the winner can still serve the runner-up.
        std::vector<std::pair<double, size_t>> ranked;
        ranked.reserve(_candidates.size());

        // A candidate that could not actually be measured stalled -- whether a stall
        // watchdog timed out, or the gate simply never held the stream (unsupported
        // device, a transient arm failure) -- cannot be mixed with the rest of this
        // pass's genuinely stalled measurements: that would rank two incomparable
        // populations. So sampling stops the instant either happens -- candidates not
        // yet reached this pass are never sampled stalled at all -- and every candidate
        // is re-measured unstalled from scratch. This policy is local to this one
        // comparison: it never reads or writes any state shared with another
        // comparison, another gate, or a standalone timed execution elsewhere.
        bool stalled = true;
        for(;;)
        {
            ranked.clear();
            bool restartUnstalled = false;
            for(size_t index = 0; index < _candidates.size(); ++index)
            {
                const auto outcome = sampleCandidate(
                    index, handle, deviceBuffers, numDeviceBuffers, workspace, *timer, stalled);
                if(stalled && outcome.restartUnstalled)
                {
                    // Abort this pass immediately rather than finishing it: candidates
                    // after this index are deliberately left unsampled this pass.
                    restartUnstalled = true;
                    break;
                }
                if(!outcome.timeMs.has_value())
                {
                    // Omitted, never appended with a sentinel time: a candidate that failed
                    // to time must never be served ahead of the normal ranked path.
                    continue;
                }
                ranked.emplace_back(*outcome.timeMs, index);
            }

            if(!restartUnstalled)
            {
                break;
            }

            HIPDNN_PLUGIN_LOG_WARN(
                "ingestor: a candidate could not actually be measured stalled partway "
                "through benchmarking (a stall watchdog timeout, or the gate never held "
                "the stream), so this pass mixed stalled and unstalled measurements. "
                "Discarding it and re-measuring every candidate unstalled.");
            // The retry runs with stalled=false, so no sample from it can trigger
            // another restart: this can happen at most once.
            stalled = false;
        }

        // stable_sort, not sort: ties must resolve to the lowest candidate index. A plain
        // std::sort would reorder equal times arbitrarily and silently change which kernel
        // wins.
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

    /// The representative time of BENCHMARK_ITERATIONS timed executes, after
    /// BENCHMARK_WARMUP_RUNS untimed ones. Any timed iteration that could not actually be
    /// measured stalled while @p stalled was requested -- a watchdog timeout, or a valid
    /// sample the timer reports as unstalled -- returns immediately with restartUnstalled
    /// set; the two populations must never be averaged together. A sample that reports
    /// stalled activity while @p stalled was false is a timer inconsistency, not a second
    /// restart opportunity, so it scores the candidate unusable instead. Any other
    /// failure to time (or a malformed sample) scores the candidate unusable too.
    ///
    /// Samples are reduced with robustMean() rather than by taking the fastest: a kernel
    /// that is usually slower but occasionally lucky would win on its best sample and then
    /// serve its typical time on every dispatch the cached ranking covers.
    SampleOutcome sampleCandidate(size_t index,
                                  const THandle& handle,
                                  const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                  uint32_t numDeviceBuffers,
                                  void* workspace,
                                  const Timer& timer,
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
            for(int iteration = 0; iteration < BENCHMARK_ITERATIONS; ++iteration)
            {
                const TimingResult sample = timer(
                    *candidate.plan, handle, deviceBuffers, numDeviceBuffers, workspace, stalled);

                if(!stalled && (sample.timedOut || sample.stallUsed))
                {
                    // The timer reported stalled activity while explicitly asked to run
                    // unstalled. This pass is already the one-shot unstalled retry, so
                    // there is no second restart to fall back to: an inconsistent timer
                    // scores the candidate unusable rather than being trusted.
                    HIPDNN_PLUGIN_LOG_WARN(
                        "ingestor: benchmarking candidate '"
                        << toString(candidate.kernelId)
                        << "' reported stalled activity during an unstalled pass; scored "
                           "unusable");
                    return {std::nullopt, false};
                }
                if(sample.timedOut && (!sample.stallUsed || sample.elapsedMs.has_value()))
                {
                    HIPDNN_PLUGIN_LOG_WARN(
                        "ingestor: inconsistent timeout result; candidate scored unusable");
                    return {std::nullopt, false};
                }
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
                // A malformed sample from either the default HIP-event timer or an
                // injected one must not enter the reduction, ranking, or cache: dropping
                // the whole candidate scores it unusable rather than letting a bogus
                // negative/NaN/infinite value win or corrupt robustMean(). Zero is a
                // valid sample (an unmeasurably fast launch).
                if(!std::isfinite(*sample.elapsedMs) || *sample.elapsedMs < 0.0)
                {
                    HIPDNN_PLUGIN_LOG_WARN(
                        "ingestor: benchmarking candidate '"
                        << toString(candidate.kernelId)
                        << "' reported a non-finite or negative elapsed time; scored unusable");
                    return {std::nullopt, false};
                }
                if(stalled && !sample.stallUsed && sample.elapsedMs.has_value())
                {
                    // A genuine measurement, but not a stalled one (unsupported device, a
                    // transient arm failure, ...). Mixing it into a pass whose earlier
                    // samples were actually stalled would rank two incomparable
                    // populations exactly like a watchdog timeout, so it gets the same
                    // whole-comparison restart rather than being accepted here.
                    HIPDNN_PLUGIN_LOG_WARN(
                        "ingestor: benchmarking candidate '"
                        << toString(candidate.kernelId)
                        << "' measured without an actual stall during a stalled pass; the "
                           "whole comparison will restart unstalled");
                    return {std::nullopt, true};
                }
                samples.push_back(*sample.elapsedMs);
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
            // IKernelDispatchHandler::launch() and an injected Timer are both extension
            // points with no exception-type contract. Letting a non-std::exception escape
            // would leave _chosen unresolved, so every later execute() would re-run the
            // whole comparison and re-hit this candidate. Score it unusable like any other
            // failure instead.
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
