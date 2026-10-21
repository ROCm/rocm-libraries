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
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_plugin_sdk/EnginePluginTypeTraits.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/BenchmarkDetail.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/WinnerCache.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

namespace detail
{

/// The id of a slotted program as an enumerated candidate names it: `<slot>=<kernel id>` for
/// every slot, in slot order, joined by `|`.
/// @param part Called with each slot index; returns that slot's name and kernel id as a tuple of
///        references.
template <typename TPart>
std::string compositeCandidateId(size_t slotCount, const TPart& part)
{
    std::string id;
    for(size_t slot = 0; slot < slotCount; ++slot)
    {
        const auto& [name, kernelId] = part(slot);
        id += (slot == 0 ? "" : "|") + name + "=" + toString(kernelId);
    }
    return id;
}

} // namespace detail

/// The IPlan of a slotted engine with benchmarking enabled (RFC 0021 §7). Slots share one stream
/// and have no time of their own, so it times whole programs, varying one slot at a time, on the
/// first execute() and serves the winners program from then on:
///
///  1. The reference program: every slot at its reference pick, the first kernel in plan-build
///     order. A reference pick that fails at launch is replaced, for this benchmark only, by its
///     slot's next kernel.
///  2. For every slot not already covered by a winner record, each of its other kernels, with
///     the remaining slots at their reference picks: 1 + sum(K - 1) runs, not the product.
///  3. The winners program, each slot at its measured winner. If it fails, the reference
///     program is served and no program record is written.
///
/// A covered slot is not re-measured: its first kernel is its winner. With every slot covered
/// only the winners program is timed, which is how a stale program record is re-established.
/// Each slot's ranking is built from the whole-program times of its runs.
///
/// Timing goes through IPlan::execute() only.
template <typename THandle>
class SlotProgramBenchmark : public IPlan<THandle>
{
    static_assert(HasGetStream<THandle>::value,
                  "SlotProgramBenchmark requires THandle to have a 'hipStream_t getStream() const' "
                  "method: the default timer brackets each comparison's runs with HIP events on "
                  "that stream. Required even when a Timer is injected, since the default-timer "
                  "path is still compiled for every THandle.");

public:
    /// One built kernel of a slot and the ids a record carries for it.
    struct Kernel
    {
        DescriptorId kernelId{};
        DescriptorId packId{};
        DescriptorId dispatchId{};
        std::shared_ptr<GenericPlan<THandle>> plan;
        /// Feature values for this (problem, kernel) pair, merged flat into the kernel's log
        /// record. Empty adds nothing.
        nlohmann::json features = nlohmann::json::object();
    };

    /// One slot, in launch order.
    struct Slot
    {
        std::string name;
        /// The slot's built kernels in plan-build order; never empty. A covered slot needs only
        /// its first.
        std::vector<Kernel> kernels;
        /// A winner record already covers this slot's catalog, so `kernels` is in measured order
        /// and its first kernel is the winner.
        bool covered = false;
    };

    using Timer = detail::InjectedTimer<THandle>;
    /// Each measured slot's name and its winner ranking.
    using SlotRecords = std::vector<std::pair<std::string, WinnerRecord>>;
    /// Invoked once, after the comparison settles, with each measured slot's ranking and, when
    /// the winners program ran, its program record. An empty function means no records.
    using RecordFn = std::function<void(const SlotRecords&, const std::optional<ProgramRecord>&)>;

    /// @param alignedSharedBytes The shared region, already rounded by alignSharedWorkspace().
    /// @param timer Overrides the default HIP-event timer, as for BenchmarkPlan.
    /// @param record The write-back channel, already bound to its keys.
    /// @param benchmarkId Opaque problem identity echoed on every log record.
    /// @param deviceIdentity Opaque device identity, echoed the same way and omitted when empty.
    /// @throws HipdnnPluginException(INTERNAL_ERROR) if @p slots is empty or a slot has no kernel.
    SlotProgramBenchmark(std::vector<Slot> slots,
                         size_t alignedSharedBytes,
                         Timer timer = {},
                         RecordFn record = {},
                         std::string benchmarkId = {},
                         std::string deviceIdentity = {})
        : _slots(std::move(slots))
        , _alignedSharedBytes(alignedSharedBytes)
        , _timer(std::move(timer))
        , _record(std::move(record))
        , _benchmarkId(std::move(benchmarkId))
        , _deviceIdentity(std::move(deviceIdentity))
    {
        if(_slots.empty())
        {
            throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                        "SlotProgramBenchmark constructed with no slots");
        }
        for(const auto& slot : _slots)
        {
            if(slot.kernels.empty())
            {
                throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                            "SlotProgramBenchmark slot '" + slot.name
                                                + "' constructed with no kernels");
            }
            for(const auto& kernel : slot.kernels)
            {
                _maxScratchBytes = std::max(_maxScratchBytes, kernel.plan->scratchBytes());
            }
        }
    }

    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    size_t getWorkspaceSize(const THandle& /*handle*/) const override
    {
        return _alignedSharedBytes + _maxScratchBytes;
    }

    /// Benchmarking runs programs against the caller's buffers, so a failing program can leave a
    /// partial result behind. The delegated execute below overwrites it with the served program's
    /// output; never add an early return before that delegation.
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    void execute(const THandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override
    {
        // Post-resolution reads take no lock, as BenchmarkPlan: _served never changes once set.
        const auto* served = _served.load(std::memory_order_acquire);
        if(served == nullptr)
        {
            served = resolveServed(Launch{&handle, deviceBuffers, numDeviceBuffers, workspace});
        }
        served->execute(handle, deviceBuffers, numDeviceBuffers, workspace);
    }

private:
    using Program = SlotProgramPlan<THandle>;

    /// What every run of one comparison executes against.
    struct Launch
    {
        const THandle* handle;
        const hipdnnPluginDeviceBuffer_t* deviceBuffers;
        uint32_t numDeviceBuffers;
        void* workspace;
    };

    /// One measured kernel of a slot: its index in plan-build order and its whole-program time.
    struct Timed
    {
        size_t kernel;
        detail::CandidateTiming timing;
    };

    /// A run that produced no time, kept for the log written once the comparison settles.
    struct FailedRun
    {
        size_t slot;
        size_t kernel;
        std::string reason;
    };

    /// What one pass of the protocol settled.
    struct Settled
    {
        /// The program to serve, as a kernel index per slot.
        std::vector<size_t> served;
        SlotRecords records;
        std::optional<ProgramRecord> program;
        /// Per slot, the kernels timed in it; empty for a covered slot.
        std::vector<std::vector<Timed>> timed;
        std::vector<FailedRun> failed;
        /// The winners program's own outcome; unset when it was not timed.
        std::optional<detail::SampleOutcome> winnersRun;
        bool referenceUsable = true;
    };

    std::shared_ptr<const Program> makeProgram(const std::vector<size_t>& picks) const
    {
        std::vector<SlotStage<THandle>> stages;
        stages.reserve(_slots.size());
        for(size_t slot = 0; slot < _slots.size(); ++slot)
        {
            stages.push_back({_slots[slot].name, _slots[slot].kernels[picks[slot]].plan});
        }
        return std::make_shared<const Program>(std::move(stages), _alignedSharedBytes);
    }

    /// The composite id of @p picks, as an enumerated candidate names it.
    std::string describe(const std::vector<size_t>& picks) const
    {
        return detail::compositeCandidateId(_slots.size(), [this, &picks](size_t slot) {
            return std::forward_as_tuple(_slots[slot].name,
                                         _slots[slot].kernels[picks[slot]].kernelId);
        });
    }

    /// Times the program of @p picks; empty when the sample asked for the comparison to restart
    /// unstalled.
    template <typename TDefaultTimer>
    std::optional<detail::SampleOutcome> timeProgram(const std::vector<size_t>& picks,
                                                     const Launch& launch,
                                                     TDefaultTimer& timer,
                                                     bool stalled) const
    {
        const auto program = makeProgram(picks);
        auto outcome = detail::sampleRuns<THandle>(
            *program,
            [this, &picks] { return describe(picks); },
            *launch.handle,
            launch.deviceBuffers,
            launch.numDeviceBuffers,
            launch.workspace,
            _timer,
            timer,
            stalled);
        if(stalled && outcome.restartUnstalled)
        {
            return std::nullopt;
        }
        return outcome;
    }

    /// The first slot whose pick at @p picks throws when launched alone, if any. Attributes a
    /// failed program to the slot whose kernel failed to launch.
    std::optional<size_t> firstLaunchFailure(const std::vector<size_t>& picks,
                                             const Launch& launch) const
    {
        for(size_t slot = 0; slot < _slots.size(); ++slot)
        {
            try
            {
                _slots[slot].kernels[picks[slot]].plan->execute(*launch.handle,
                                                                launch.deviceBuffers,
                                                                launch.numDeviceBuffers,
                                                                launch.workspace);
            }
            catch(...)
            {
                return slot;
            }
        }
        return std::nullopt;
    }

    /// One pass of the protocol; empty when a sample asked for the comparison to restart
    /// unstalled.
    template <typename TDefaultTimer>
    std::optional<Settled>
        runProtocol(const Launch& launch, TDefaultTimer& timer, bool stalled) const
    {
        Settled settled;
        settled.timed.resize(_slots.size());
        std::vector<size_t> reference(_slots.size(), 0);

        const bool anyUncovered = std::any_of(
            _slots.begin(), _slots.end(), [](const Slot& slot) { return !slot.covered; });

        std::optional<detail::CandidateTiming> referenceTiming;
        if(anyUncovered)
        {
            for(;;)
            {
                auto outcome = timeProgram(reference, launch, timer, stalled);
                if(!outcome.has_value())
                {
                    return std::nullopt;
                }
                if(outcome->timing.has_value())
                {
                    referenceTiming = outcome->timing;
                    break;
                }

                const auto failedSlot = firstLaunchFailure(reference, launch);
                if(!failedSlot.has_value()
                   || reference[*failedSlot] + 1 >= _slots[*failedSlot].kernels.size())
                {
                    HIPDNN_PLUGIN_LOG_ERROR(
                        "ingestor: benchmarking could not time the reference program "
                        << describe(reference) << ": " << outcome->failureReason
                        << "; serving it without records");
                    settled.referenceUsable = false;
                    settled.served = reference;
                    return settled;
                }
                HIPDNN_PLUGIN_LOG_WARN(
                    "ingestor: benchmarking reference pick '"
                    << toString(_slots[*failedSlot].kernels[reference[*failedSlot]].kernelId)
                    << "' of slot '" << _slots[*failedSlot].name
                    << "' failed to launch; the slot's reference moves to its next kernel");
                settled.failed.push_back(
                    {*failedSlot, reference[*failedSlot], outcome->failureReason});
                ++reference[*failedSlot];
            }

            for(size_t slot = 0; slot < _slots.size(); ++slot)
            {
                if(_slots[slot].covered)
                {
                    continue;
                }
                settled.timed[slot].push_back({reference[slot], *referenceTiming});
                for(size_t kernel = reference[slot] + 1; kernel < _slots[slot].kernels.size();
                    ++kernel)
                {
                    auto picks = reference;
                    picks[slot] = kernel;
                    auto outcome = timeProgram(picks, launch, timer, stalled);
                    if(!outcome.has_value())
                    {
                        return std::nullopt;
                    }
                    if(outcome->timing.has_value())
                    {
                        settled.timed[slot].push_back({kernel, *outcome->timing});
                    }
                    else
                    {
                        settled.failed.push_back({slot, kernel, std::move(outcome->failureReason)});
                    }
                }
            }
        }

        // Each slot's winner, and whether every slot has a measured one to name in a program
        // record: a slot whose kernels all failed has none, and keeps its reference pick.
        std::vector<size_t> winners = reference;
        bool everySlotRanked = true;
        for(size_t slot = 0; slot < _slots.size(); ++slot)
        {
            auto& timed = settled.timed[slot];
            if(_slots[slot].covered)
            {
                continue;
            }
            // stable_sort, not sort: ties must resolve to the lowest kernel index.
            std::stable_sort(timed.begin(), timed.end(), [](const Timed& lhs, const Timed& rhs) {
                return lhs.timing.robustMeanMs < rhs.timing.robustMeanMs;
            });
            if(timed.empty())
            {
                everySlotRanked = false;
                continue;
            }
            winners[slot] = timed.front().kernel;

            WinnerRecord ranking;
            ranking.reserve(timed.size());
            for(const auto& entry : timed)
            {
                const auto& kernel = _slots[slot].kernels[entry.kernel];
                ranking.push_back(RankedEntry{
                    kernel.kernelId, kernel.packId, kernel.dispatchId, entry.timing.robustMeanMs});
            }
            settled.records.emplace_back(_slots[slot].name, std::move(ranking));
        }

        auto winnersRun = timeProgram(winners, launch, timer, stalled);
        if(!winnersRun.has_value())
        {
            return std::nullopt;
        }
        if(winnersRun->timing.has_value())
        {
            settled.served = winners;
            if(everySlotRanked)
            {
                ProgramRecord program;
                program.picks.reserve(_slots.size());
                for(size_t slot = 0; slot < _slots.size(); ++slot)
                {
                    program.picks.push_back(
                        {_slots[slot].name, _slots[slot].kernels[winners[slot]].kernelId});
                }
                program.timeMs = winnersRun->timing->robustMeanMs;
                settled.program = std::move(program);
            }
            if(referenceTiming.has_value()
               && winnersRun->timing->robustMeanMs > referenceTiming->robustMeanMs)
            {
                HIPDNN_PLUGIN_LOG_WARN(
                    "ingestor: benchmarking winners program "
                    << describe(winners) << " took " << winnersRun->timing->robustMeanMs
                    << " ms, slower than the reference program's " << referenceTiming->robustMeanMs
                    << " ms: slots may not be separable");
            }
        }
        else
        {
            HIPDNN_PLUGIN_LOG_WARN(
                "ingestor: benchmarking winners program "
                << describe(winners) << " failed: " << winnersRun->failureReason
                << "; serving the reference program and writing no program record");
            settled.served = reference;
        }
        settled.winnersRun = std::move(winnersRun);
        return settled;
    }

    /// Resolves the served program on the first call and caches it. The lock spans the whole
    /// comparison, so a second thread racing the first execute() blocks instead of running
    /// against the first thread's buffers.
    const Program* resolveServed(const Launch& launch) const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        if(const auto* resolved = _served.load(std::memory_order_relaxed); resolved != nullptr)
        {
            return resolved;
        }

        const auto settled = detail::runWithStallRestart<THandle>(
            static_cast<bool>(_timer), [&](auto& defaultTimer, bool stalled) {
                return runProtocol(launch, defaultTimer, stalled);
            });

        logSettled(settled);
        if(_record && settled.referenceUsable
           && (!settled.records.empty() || settled.program.has_value()))
        {
            _record(settled.records, settled.program);
        }

        _servedOwner = makeProgram(settled.served);
        _served.store(_servedOwner.get(), std::memory_order_release);
        return _servedOwner.get();
    }

    void logSettled(const Settled& settled) const
    {
        if(!HIPDNN_PLUGIN_LOG_IS_INFO_ENABLED())
        {
            return;
        }
        for(size_t slot = 0; slot < _slots.size(); ++slot)
        {
            for(const auto& entry : settled.timed[slot])
            {
                auto record = kernelRecord(slot, entry.kernel);
                record["status"] = "ok";
                record["min_ms"] = entry.timing.minMs;
                record["avg_ms"] = entry.timing.avgMs;
                record["stddev_ms"] = entry.timing.stddevMs;
                record["robust_mean_ms"] = entry.timing.robustMeanMs;
                record["iters"] = entry.timing.iterations;
                HIPDNN_PLUGIN_LOG_INFO(record.dump());
            }
        }
        for(const auto& run : settled.failed)
        {
            auto record = kernelRecord(run.slot, run.kernel);
            record["status"] = "failed";
            record["reason"] = run.reason;
            HIPDNN_PLUGIN_LOG_INFO(record.dump());
        }
        if(settled.winnersRun.has_value())
        {
            nlohmann::json record = nlohmann::json::object();
            record["event"] = "ingestor.benchmark.program";
            record["benchmark"] = _benchmarkId;
            if(!_deviceIdentity.empty())
            {
                record["device"] = _deviceIdentity;
            }
            record["program"] = describe(settled.served);
            if(settled.winnersRun->timing.has_value())
            {
                record["status"] = "ok";
                record["robust_mean_ms"] = settled.winnersRun->timing->robustMeanMs;
            }
            else
            {
                record["status"] = "failed";
                record["reason"] = settled.winnersRun->failureReason;
            }
            HIPDNN_PLUGIN_LOG_INFO(record.dump());
        }
    }

    /// The fields common to every kernel record, layered over the kernel's features. The
    /// envelope is written last so a feature key can never overwrite the row's identity.
    nlohmann::json kernelRecord(size_t slot, size_t kernelIndex) const
    {
        const auto& kernel = _slots[slot].kernels[kernelIndex];
        nlohmann::json record = kernel.features;
        record["event"] = "ingestor.benchmark.candidate";
        record["benchmark"] = _benchmarkId;
        if(!_deviceIdentity.empty())
        {
            record["device"] = _deviceIdentity;
        }
        record["slot"] = _slots[slot].name;
        record["kernel"] = toString(kernel.kernelId);
        record["pack"] = toString(kernel.packId);
        record["dispatch"] = toString(kernel.dispatchId);
        return record;
    }

    std::vector<Slot> _slots;
    size_t _alignedSharedBytes;
    size_t _maxScratchBytes = 0;
    Timer _timer;
    RecordFn _record;
    std::string _benchmarkId;
    std::string _deviceIdentity;
    mutable std::shared_ptr<const Program> _servedOwner;
    mutable std::atomic<const Program*> _served{nullptr};
    mutable std::mutex _mutex;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
