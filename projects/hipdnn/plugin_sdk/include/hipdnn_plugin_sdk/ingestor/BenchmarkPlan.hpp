// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <atomic>
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
                  "method: the default timer brackets each candidate with HIP events on that "
                  "stream. Required even when a Timer is injected, since both arms of the "
                  "constructor's default are instantiated.");

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

    /// Times one execute() of a candidate, returning its elapsed milliseconds or nullopt
    /// if the launch could not be timed. Defaults to HIP events on the handle's stream;
    /// tests substitute a deterministic timer so selection is provable without a device.
    using Timer = std::function<std::optional<double>(
        const IPlan<THandle>&, const THandle&, const hipdnnPluginDeviceBuffer_t*, uint32_t, void*)>;
    /// Invoked once, with every usable candidate in benchmarked order, after sampling
    /// resolves the winner. An absent callback means no caching, so BenchmarkPlan needs
    /// no knowledge of the cache's type or lifetime.
    using RecordRankingFn = std::function<void(std::vector<RankedEntry>)>;

    /// @param handle Sizes every sub-plan's workspace requirement; execute() uses the
    ///        handle its own caller passes.
    /// @param timer Overrides the default HIP-event timer. Only ever called from the
    ///        sampling sweep, which holds _mutex, so it need not be thread-safe.
    /// @throws HipdnnPluginException(INTERNAL_ERROR) if @p candidates is empty.
    BenchmarkPlan(std::vector<Candidate> candidates,
                  const THandle& handle,
                  Timer timer = {},
                  RecordRankingFn recordRanking = {})
        : _candidates(std::move(candidates))
        , _timer(timer ? std::move(timer) : makeHipEventTimer())
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
        // steady-state path the feature exists for has no serialization point.
        size_t chosen = _chosen.load(std::memory_order_acquire);
        if(chosen == NOT_RESOLVED)
        {
            chosen = resolveChosen(handle, deviceBuffers, numDeviceBuffers, workspace);
        }
        _candidates[chosen].plan->execute(handle, deviceBuffers, numDeviceBuffers, workspace);
    }

private:
    static constexpr size_t NOT_RESOLVED = std::numeric_limits<size_t>::max();

    /// The default timer: HIP events on handle.getStream() rather than the null stream,
    /// so a plan on a non-default stream still measures its own work. The event pair is
    /// created on first use and reused for every subsequent sample.
    static Timer makeHipEventTimer()
    {
        auto events = std::make_shared<std::optional<detail::HipEventPair>>();
        return [events](const IPlan<THandle>& plan,
                        const THandle& handle,
                        const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                        uint32_t numDeviceBuffers,
                        void* workspace) -> std::optional<double> {
            if(!events->has_value())
            {
                events->emplace();
            }
            if(!(*events)->isUsable())
            {
                // Discard the unusable pair so the next sample retries creation. Caching
                // it would turn one transient hipEventCreate failure into a permanently
                // untimeable plan: every candidate's first timed iteration would return
                // nullopt, no candidate would score usable, and selection would silently
                // fall back to the ranked front for the plan's whole life.
                events->reset();
                return std::nullopt;
            }

            const auto start = (*events)->start.get();
            const auto stop = (*events)->stop.get();
            const auto stream = handle.getStream();

            if(hipEventRecord(start, stream) != hipSuccess)
            {
                return std::nullopt;
            }

            plan.execute(handle, deviceBuffers, numDeviceBuffers, workspace);

            if(hipEventRecord(stop, stream) != hipSuccess
               || hipEventSynchronize(stop) != hipSuccess)
            {
                return std::nullopt;
            }

            float elapsedMs = 0.0F;
            if(hipEventElapsedTime(&elapsedMs, start, stop) != hipSuccess)
            {
                return std::nullopt;
            }
            return static_cast<double>(elapsedMs);
        };
    }

    /// Resolves _chosen on the first call and caches it. The lock spans the whole sweep,
    /// so a second thread racing the first execute() blocks instead of sampling against
    /// the first thread's buffers.
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

        // Every usable candidate's time is retained so a later run whose knob filter
        // excludes the winner can still serve the runner-up.
        std::vector<std::pair<double, size_t>> ranked;
        ranked.reserve(_candidates.size());

        for(size_t index = 0; index < _candidates.size(); ++index)
        {
            const auto timeMs
                = sampleCandidate(index, handle, deviceBuffers, numDeviceBuffers, workspace);
            if(!timeMs.has_value())
            {
                // Omitted, never appended with a sentinel time: a candidate that failed to
                // time must never be served ahead of the normal ranked path.
                continue;
            }
            ranked.emplace_back(*timeMs, index);
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

protected:
    /// The representative time of BENCHMARK_ITERATIONS timed executes, after
    /// BENCHMARK_WARMUP_RUNS untimed ones. Returns nullopt if the candidate threw or could
    /// not be timed; both score the candidate unusable rather than throwing out of
    /// resolveChosen().
    ///
    /// Samples are reduced with robustMean() rather than by taking the fastest: a kernel
    /// that is usually slower but occasionally lucky would win on its best sample and then
    /// serve its typical time on every dispatch the cached ranking covers.
    std::optional<double> sampleCandidate(size_t index,
                                          const THandle& handle,
                                          const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                          uint32_t numDeviceBuffers,
                                          void* workspace) const
    {
        const auto& candidate = _candidates[index];
        try
        {
            for(int warmup = 0; warmup < BENCHMARK_WARMUP_RUNS; ++warmup)
            {
                candidate.plan->execute(handle, deviceBuffers, numDeviceBuffers, workspace);
            }

            std::vector<double> samples;
            samples.reserve(BENCHMARK_ITERATIONS);
            for(int iteration = 0; iteration < BENCHMARK_ITERATIONS; ++iteration)
            {
                const auto sampleMs
                    = _timer(*candidate.plan, handle, deviceBuffers, numDeviceBuffers, workspace);
                if(!sampleMs.has_value())
                {
                    HIPDNN_PLUGIN_LOG_WARN("ingestor: benchmarking candidate '"
                                           << toString(candidate.kernelId)
                                           << "' failed to time a launch; scored unusable");
                    return std::nullopt;
                }
                samples.push_back(*sampleMs);
            }
            return hipdnn_data_sdk::utilities::detail::robustMean(samples);
        }
        catch(const std::exception& error)
        {
            HIPDNN_PLUGIN_LOG_WARN("ingestor: benchmarking candidate '"
                                   << toString(candidate.kernelId)
                                   << "' threw and is scored unusable: " << error.what());
            return std::nullopt;
        }
        catch(...)
        {
            // IKernelDispatchHandler::launch() and an injected Timer are both extension
            // points with no exception-type contract. Letting a non-std::exception escape
            // would leave _chosen unresolved, so every later execute() would re-run the
            // whole sweep and re-hit this candidate. Score it unusable like any other
            // failure instead.
            HIPDNN_PLUGIN_LOG_WARN("ingestor: benchmarking candidate '"
                                   << toString(candidate.kernelId)
                                   << "' threw a non-standard exception and is scored unusable");
            return std::nullopt;
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
