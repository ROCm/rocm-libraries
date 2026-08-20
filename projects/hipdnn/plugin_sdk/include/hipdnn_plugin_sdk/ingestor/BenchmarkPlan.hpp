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

#include <hipdnn_data_sdk/utilities/ScopedResource.hpp>
#include <hipdnn_plugin_sdk/EnginePluginTypeTraits.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// Sampling counts, matching MIOpen's EvaluateInvokers: one untimed warmup followed by
/// up to eight total runs per candidate.
constexpr int BENCHMARK_WARMUP_RUNS = 1;
constexpr int BENCHMARK_ITERATIONS = 7;

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

} // namespace detail

/// An IPlan owning one GenericPlan per knob-filtered catalog entry. Times each candidate
/// on the first execute() and delegates every call to the fastest. Wraps GenericPlan
/// rather than widening it, leaving single-kernel construction, workspace query, and the
/// null-prepared check unchanged.
///
/// Timing goes through IPlan::execute() only, bracketed by HIP events on the handle's
/// stream; this class touches no dispatcher, PreparedDispatch, or HIP launch API.
template <typename THandle>
class BenchmarkPlan : public IPlan<THandle>
{
public:
    /// A sub-plan and the kernel it was built for. IPlan has no kernel accessor, so the
    /// id rides alongside for the selection log.
    struct Candidate
    {
        DescriptorId kernelId;
        std::unique_ptr<IPlan<THandle>> plan;
    };

    /// @param handle Sizes every sub-plan's workspace requirement; execute() uses the
    ///        handle its own caller passes.
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

    /// Sampling runs candidates against the caller's buffers, so a candidate failing
    /// mid-loop can leave a partial result behind. The delegated execute below overwrites
    /// it with the winner's output; never add an early return before that delegation.
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
    /// Resolves _chosen on the first call and caches it. The lock spans the whole sweep,
    /// so a second thread racing the first execute() blocks instead of sampling against
    /// the first thread's buffers.
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

    /// The fastest of BENCHMARK_ITERATIONS timed executes, after BENCHMARK_WARMUP_RUNS
    /// untimed ones. Returns nullopt if the candidate threw or a HIP event call failed;
    /// both score the candidate unusable rather than throwing out of resolveChosen().
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

    /// One timed execute(), bracketed by HIP events on handle.getStream() rather than the
    /// null stream, so a plan on a non-default stream still measures its own work.
    std::optional<double> timeOneExecute(const Candidate& candidate,
                                         const THandle& handle,
                                         const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                         uint32_t numDeviceBuffers,
                                         void* workspace) const
    {
        const auto start = detail::createScopedHipEvent();
        const auto stop = detail::createScopedHipEvent();
        if(start.isEmpty() || stop.isEmpty())
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
