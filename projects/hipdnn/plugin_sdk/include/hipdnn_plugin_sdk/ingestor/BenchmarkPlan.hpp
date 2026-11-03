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
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include <hip/hip_runtime.h>

#include <hipdnn_plugin_sdk/EnginePluginTypeTraits.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/BenchmarkDetail.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/WinnerCache.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

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
        /// Feature values for this (problem, kernel) pair, merged flat into the candidate's log
        /// record. Keys are namespaced (`q.`, `kernel.`), so they cannot collide with the dotless
        /// envelope keys. Supplied by the caller; empty adds nothing.
        nlohmann::json features = nlohmann::json::object();
    };

    /// Times one execute() of a candidate, returning elapsed milliseconds or nullopt
    /// when it could not be timed. An injected timer owns its synchronization and
    /// measurement method; no HIP resources are acquired for it.
    using Timer = detail::InjectedTimer<THandle>;
    /// Receives every usable candidate in ranked order once the winner is resolved. Absent
    /// means no caching.
    using RecordRankingFn = std::function<void(std::vector<RankedEntry>)>;

    /// @param handle Sizes every sub-plan's workspace requirement.
    /// @param timer Overrides the default HIP-event timer. Called only from the sampling
    ///        sweep, under _mutex, so it need not be thread-safe. When empty, each
    ///        comparison builds its own events and stall gate (see resolveChosen()).
    /// @throws HipdnnPluginException(INTERNAL_ERROR) if @p candidates is empty.
    /// @param benchmarkId Opaque problem identity echoed on every candidate log record so an
    ///        exporter can group one sweep's rows; empty means unidentified.
    /// @param deviceIdentity Opaque device identity, echoed the same way and omitted when
    ///        empty. Rows group by (benchmark, device), matching the winner cache key.
    BenchmarkPlan(std::vector<Candidate> candidates,
                  const THandle& handle,
                  Timer timer = {},
                  RecordRankingFn recordRanking = {},
                  std::string benchmarkId = {},
                  std::string deviceIdentity = {})
        : _candidates(std::move(candidates))
        , _timer(std::move(timer))
        , _recordRanking(std::move(recordRanking))
        , _benchmarkId(std::move(benchmarkId))
        , _deviceIdentity(std::move(deviceIdentity))
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

    /// The candidate the first execute() chose, or nullptr before a candidate is chosen.
    const IPlan<THandle>* chosenPlan() const
    {
        const size_t chosen = _chosen.load(std::memory_order_acquire);
        if(chosen == NOT_RESOLVED)
        {
            return nullptr;
        }
        return _candidates[chosen].plan.get();
    }

    /// The chosen candidate's saveable plan, or nullptr before a candidate is chosen.
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    const GenericPlan<THandle>* saveablePlan() const override
    {
        const IPlan<THandle>* chosen = chosenPlan();
        return chosen != nullptr ? chosen->saveablePlan() : nullptr;
    }

private:
    static constexpr size_t NOT_RESOLVED = std::numeric_limits<size_t>::max();

    using CandidateTiming = detail::CandidateTiming;
    using SampleOutcome = detail::SampleOutcome;

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

        auto outcomes = detail::runWithStallRestart<THandle>(
            static_cast<bool>(_timer),
            [&](auto& defaultTimer, bool stalled) -> std::optional<std::vector<SampleOutcome>> {
                // Outcomes of the settling pass, logged only after it settles: a discarded
                // stalled pass re-measures every candidate and would otherwise log each one
                // twice.
                std::vector<SampleOutcome> passOutcomes;
                passOutcomes.reserve(_candidates.size());
                for(const auto& candidate : _candidates)
                {
                    auto outcome = detail::sampleRuns<THandle>(
                        *candidate.plan,
                        [&candidate] { return toString(candidate.kernelId); },
                        handle,
                        deviceBuffers,
                        numDeviceBuffers,
                        workspace,
                        _timer,
                        defaultTimer,
                        stalled);
                    if(stalled && outcome.restartUnstalled)
                    {
                        // Abort this pass immediately rather than finishing it: candidates
                        // after this one are deliberately left unsampled this pass.
                        return std::nullopt;
                    }
                    passOutcomes.push_back(std::move(outcome));
                }
                return passOutcomes;
            });

        // Keep every usable time, so a later knob filter that excludes the winner can
        // still serve the runner-up. A failed candidate is omitted, never given a sentinel time.
        std::vector<std::pair<double, size_t>> ranked;
        ranked.reserve(_candidates.size());
        for(size_t index = 0; index < outcomes.size(); ++index)
        {
            if(outcomes[index].timing.has_value())
            {
                ranked.emplace_back(outcomes[index].timing->robustMeanMs, index);
            }
        }
        for(size_t index = 0; index < outcomes.size(); ++index)
        {
            const auto& outcome = outcomes[index];
            if(outcome.timing.has_value())
            {
                logCandidateTiming(_candidates[index], *outcome.timing);
            }
            else
            {
                logCandidateFailure(_candidates[index], outcome.failureReason);
            }
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

    /// Logs one JSON record per timed candidate at INFO: the per-kernel measurement a UHD is
    /// trained on, since the cache keeps only the winner. Nothing is built when INFO is off.
    void logCandidateTiming(const Candidate& candidate, const CandidateTiming& timing) const
    {
        if(!HIPDNN_PLUGIN_LOG_IS_INFO_ENABLED())
        {
            return;
        }
        auto record = candidateRecord(candidate);
        detail::annotateTiming(record, timing);
        HIPDNN_PLUGIN_LOG_INFO(record.dump());
    }

    /// Logs a `failed` record with its reason (RFC 0019.13 §8.3). The winner cache drops
    /// failed candidates, so this is the only place a failure is recorded.
    void logCandidateFailure(const Candidate& candidate, const std::string& reason) const
    {
        if(!HIPDNN_PLUGIN_LOG_IS_INFO_ENABLED())
        {
            return;
        }
        auto record = candidateRecord(candidate);
        detail::annotateFailure(record, reason);
        HIPDNN_PLUGIN_LOG_INFO(record.dump());
    }

    /// The fields common to every candidate record, layered over the candidate's features.
    nlohmann::json candidateRecord(const Candidate& candidate) const
    {
        return detail::candidateRecord(candidate.features,
                                       _benchmarkId,
                                       _deviceIdentity,
                                       candidate.kernelId,
                                       candidate.packId,
                                       candidate.dispatchId);
    }

    std::vector<Candidate> _candidates;
    Timer _timer;
    RecordRankingFn _recordRanking;
    std::string _benchmarkId;
    std::string _deviceIdentity;
    size_t _workspaceBytes = 0;
    mutable std::atomic<size_t> _chosen{NOT_RESOLVED};
    mutable std::mutex _mutex;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
