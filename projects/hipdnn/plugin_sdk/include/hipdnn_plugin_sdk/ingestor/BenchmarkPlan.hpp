// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>

#include <hipdnn_plugin_sdk/EnginePluginTypeTraits.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// Timing constants, not knobs: Part 1 fixes the mechanism. These are starting values,
/// not measured ones -- the gfx942 run confirmed sampling executes and selects, but did
/// not measure per-candidate variance, so whether this many iterations makes a pointwise
/// kernel's timing stable is still open (plan §9 uncertainty 1).
constexpr int BENCHMARK_WARMUP_RUNS = 1;
constexpr int BENCHMARK_ITERATIONS = 5;

namespace detail
{

/// Owns at most one hipEvent_t and always destroys it, including when a candidate's
/// execute() throws between construction and destruction.
class ScopedHipEvent
{
public:
    ScopedHipEvent()
    {
        if(hipEventCreate(&_event) != hipSuccess)
        {
            _event = nullptr;
        }
    }

    ~ScopedHipEvent()
    {
        if(_event != nullptr)
        {
            // Nothing actionable in a destructor; the event is being discarded anyway.
            static_cast<void>(hipEventDestroy(_event));
        }
    }

    ScopedHipEvent(const ScopedHipEvent&) = delete;
    ScopedHipEvent& operator=(const ScopedHipEvent&) = delete;
    ScopedHipEvent(ScopedHipEvent&&) = delete;
    ScopedHipEvent& operator=(ScopedHipEvent&&) = delete;

    bool valid() const
    {
        return _event != nullptr;
    }

    hipEvent_t get() const
    {
        return _event;
    }

private:
    hipEvent_t _event = nullptr;
};

} // namespace detail

/// A composite IPlan owning every knob-filtered catalog entry as its own GenericPlan,
/// timing each on the first execute() and delegating every call after to the fastest.
/// Wraps GenericPlan rather than widening it: a single-kernel plan's construction,
/// workspace query, and null-prepared check stay exactly what they are today.
///
/// Timing goes through IPlan::execute() only, on the handle's own stream -- this class
/// never touches a dispatcher, a PreparedDispatch, or any HIP launch API beyond the
/// bracketing events.
template <typename THandle>
class BenchmarkPlan : public IPlan<THandle>
{
public:
    /// One sub-plan plus the kernel it was built for. kernelId rides alongside because
    /// IPlan has no kernel accessor and must not grow one -- it is what the selection
    /// log names and what a future disk-backed cache would persist against.
    struct Candidate
    {
        DescriptorId kernelId;
        std::unique_ptr<IPlan<THandle>> plan;
    };

    /// @param handle Used once, here, to size every sub-plan's workspace requirement;
    ///        execute() always uses the handle its own caller passes.
    /// @throws HipdnnPluginException(INTERNAL_ERROR) if @p candidates is empty.
    BenchmarkPlan(std::vector<Candidate> candidates, const THandle& handle)
        : _candidates(std::move(candidates))
    {
        static_assert(HasGetStream<THandle>::value,
                      "BenchmarkPlan requires THandle to have a 'hipStream_t getStream() const' "
                      "method");

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

    /// Sampling runs candidates against the caller's real buffers, so a candidate that
    /// fails mid-loop can leave a partial result behind. What makes that safe is that
    /// this function always ends with the delegated execute below, which overwrites the
    /// output with the winner's. Never add an early return between resolveChosen() and
    /// that delegation.
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    void execute(const THandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override
    {
        const size_t chosen = resolveChosen(handle, deviceBuffers, numDeviceBuffers, workspace);
        _candidates[chosen].plan->execute(handle, deviceBuffers, numDeviceBuffers, workspace);
    }

private:
    /// Resolves _chosen on the first call under the mutex; every later call returns the
    /// cached winner without re-sampling. Two threads racing the first execute() see one
    /// sampling pass, not two.
    ///
    /// The lock is held across the whole sweep, so a second thread executing with
    /// different buffers blocks until sampling finishes rather than proceeding in
    /// parallel. That is the deliberate trade for sampling exactly once: a
    /// double-checked lock here would let two threads sample concurrently, against each
    /// other's buffers. Only the first execute() pays it.
    size_t resolveChosen(const THandle& handle,
                         const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                         uint32_t numDeviceBuffers,
                         void* workspace) const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        if(_chosen.has_value())
        {
            return *_chosen;
        }

        size_t best = 0;
        double bestTimeMs = std::numeric_limits<double>::max();
        bool anyUsable = false;

        for(size_t index = 0; index < _candidates.size(); ++index)
        {
            const auto timeMs
                = sampleCandidate(index, handle, deviceBuffers, numDeviceBuffers, workspace);
            if(!timeMs.has_value())
            {
                continue;
            }
            anyUsable = true;
            if(*timeMs < bestTimeMs)
            {
                bestTimeMs = *timeMs;
                best = index;
            }
        }

        if(!anyUsable)
        {
            HIPDNN_PLUGIN_LOG_ERROR("ingestor: benchmarking found no usable candidate among "
                                    << _candidates.size() << " kernel(s); defaulting to "
                                    << toString(_candidates.front().kernelId));
            best = 0;
        }
        else
        {
            HIPDNN_PLUGIN_LOG_INFO("ingestor: benchmarking selected kernel "
                                   << toString(_candidates[best].kernelId) << " in " << bestTimeMs
                                   << " ms among " << _candidates.size() << " candidate(s)");
        }

        _chosen = best;
        return best;
    }

    /// The minimum timed execute() over BENCHMARK_ITERATIONS, after BENCHMARK_WARMUP_RUNS
    /// untimed ones, or nullopt if the candidate threw or a HIP event call failed -- both
    /// are a loss for this candidate, never a throw out of resolveChosen().
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

            double bestMs = std::numeric_limits<double>::max();
            for(int iteration = 0; iteration < BENCHMARK_ITERATIONS; ++iteration)
            {
                const auto sampleMs
                    = timeOneExecute(candidate, handle, deviceBuffers, numDeviceBuffers, workspace);
                if(!sampleMs.has_value())
                {
                    HIPDNN_PLUGIN_LOG_WARN("ingestor: benchmarking candidate '"
                                           << toString(candidate.kernelId)
                                           << "' failed to time a launch; scored unusable");
                    return std::nullopt;
                }
                bestMs = std::min(bestMs, *sampleMs);
            }
            return bestMs;
        }
        catch(const std::exception& error)
        {
            HIPDNN_PLUGIN_LOG_WARN("ingestor: benchmarking candidate '"
                                   << toString(candidate.kernelId)
                                   << "' threw and is scored unusable: " << error.what());
            return std::nullopt;
        }
    }

    /// One warmup-free, timed execute() bracketed by hipEvents on handle.getStream() --
    /// never the null stream, or a plan on a non-default stream measures nothing.
    std::optional<double> timeOneExecute(const Candidate& candidate,
                                         const THandle& handle,
                                         const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                         uint32_t numDeviceBuffers,
                                         void* workspace) const
    {
        const detail::ScopedHipEvent start;
        const detail::ScopedHipEvent stop;
        if(!start.valid() || !stop.valid())
        {
            return std::nullopt;
        }

        const auto stream = handle.getStream();
        if(hipEventRecord(start.get(), stream) != hipSuccess)
        {
            return std::nullopt;
        }

        candidate.plan->execute(handle, deviceBuffers, numDeviceBuffers, workspace);

        if(hipEventRecord(stop.get(), stream) != hipSuccess
           || hipEventSynchronize(stop.get()) != hipSuccess)
        {
            return std::nullopt;
        }

        float elapsedMs = 0.0F;
        if(hipEventElapsedTime(&elapsedMs, start.get(), stop.get()) != hipSuccess)
        {
            return std::nullopt;
        }
        return static_cast<double>(elapsedMs);
    }

    std::vector<Candidate> _candidates;
    size_t _workspaceBytes = 0;
    mutable std::optional<size_t> _chosen;
    mutable std::mutex _mutex;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
