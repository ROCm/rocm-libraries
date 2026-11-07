// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/conv/BenchmarkedPlan.hpp"

#include <algorithm>
#include <limits>
#include <utility>

#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "HipdnnEnginePluginHandle.hpp"

namespace hipblaslt_plugin
{

namespace
{

/// Timed runs per candidate after one warm-up; the best is kept.
constexpr int TIMED_RUNS = 3;

void check(hipError_t status, const char* what)
{
    if(status != hipSuccess)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            std::string(what) + " failed: " + hipGetErrorString(status));
    }
}

/// A HIP event, destroyed with its owner.
class Event
{
public:
    Event()
    {
        check(hipEventCreate(&_event), "hipEventCreate");
    }

    ~Event()
    {
        static_cast<void>(hipEventDestroy(_event));
    }

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    Event(Event&&) = delete;
    Event& operator=(Event&&) = delete;

    hipEvent_t get() const
    {
        return _event;
    }

private:
    hipEvent_t _event = nullptr;
};

} // namespace

BenchmarkedPlan::BenchmarkedPlan(std::vector<Candidate> candidates)
    : _candidates(std::move(candidates))
{
    PLUGIN_THROW_IF_FALSE(!_candidates.empty(),
                          HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                          "a benchmarked plan needs a candidate");
    for(const auto& candidate : _candidates)
    {
        PLUGIN_THROW_IF_FALSE(candidate.plan != nullptr,
                              HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                              "a benchmarked plan's candidate has no plan");
    }
}

size_t BenchmarkedPlan::getWorkspaceSize(const HipdnnEnginePluginHandle& handle) const
{
    size_t bytes = 0;
    for(const auto& candidate : _candidates)
    {
        bytes = std::max(bytes, candidate.plan->getWorkspaceSize(handle));
    }
    return bytes;
}

int BenchmarkedPlan::winner() const
{
    const std::lock_guard<std::mutex> lock(_mutex);
    return _winner;
}

void BenchmarkedPlan::execute(const HipdnnEnginePluginHandle& handle,
                              const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                              uint32_t numDeviceBuffers,
                              void* workspace) const
{
    const std::lock_guard<std::mutex> lock(_mutex);
    if(_winner >= 0)
    {
        _candidates[static_cast<size_t>(_winner)].plan->execute(
            handle, deviceBuffers, numDeviceBuffers, workspace);
        return;
    }

    const hipStream_t stream = handle.getStream();
    const Event start;
    const Event stop;
    float best = std::numeric_limits<float>::max();
    int bestIndex = 0;
    for(size_t index = 0; index < _candidates.size(); ++index)
    {
        const auto& plan = *_candidates[index].plan;
        plan.execute(handle, deviceBuffers, numDeviceBuffers, workspace);
        float fastest = std::numeric_limits<float>::max();
        for(int run = 0; run < TIMED_RUNS; ++run)
        {
            check(hipEventRecord(start.get(), stream), "hipEventRecord");
            plan.execute(handle, deviceBuffers, numDeviceBuffers, workspace);
            check(hipEventRecord(stop.get(), stream), "hipEventRecord");
            check(hipEventSynchronize(stop.get()), "hipEventSynchronize");
            float milliseconds = 0.0f;
            check(hipEventElapsedTime(&milliseconds, start.get(), stop.get()),
                  "hipEventElapsedTime");
            fastest = std::min(fastest, milliseconds);
        }
        HIPDNN_PLUGIN_LOG_INFO("Benchmarked candidate '" << _candidates[index].name
                                                         << "': " << fastest << " ms");
        if(fastest < best)
        {
            best = fastest;
            bestIndex = static_cast<int>(index);
        }
    }
    _winner = bestIndex;
    HIPDNN_PLUGIN_LOG_INFO("Kept candidate '" << _candidates[static_cast<size_t>(_winner)].name
                                              << "'");
    // The last timed run was the last candidate's; leave the winner's result in place.
    if(static_cast<size_t>(_winner) + 1 != _candidates.size())
    {
        _candidates[static_cast<size_t>(_winner)].plan->execute(
            handle, deviceBuffers, numDeviceBuffers, workspace);
    }
}

} // namespace hipblaslt_plugin
