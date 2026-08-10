// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "plans/CatalogPlan.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "launch/PluginError.hpp"

namespace aot_catalog_engine
{

namespace
{

// One warmup launch, discarded, then this many timed launches per candidate; the
// median is the candidate's score. Small counts keep first-execute tuning cheap.
constexpr int TIMED_LAUNCHES = 5;

// Build the single-candidate vector for the legacy ctor.
std::vector<PlanCandidate> makeSingle(launch::HipModuleGuard module,
                                      catalog::LaunchMetadata launchMetadata,
                                      catalog::LaunchBindings bindings,
                                      launch::SymbolTable gridSymbols,
                                      size_t workspaceBytes,
                                      std::string kernelName)
{
    std::vector<PlanCandidate> out;
    out.push_back(PlanCandidate{std::move(module),
                                std::move(launchMetadata),
                                std::move(bindings),
                                std::move(gridSymbols),
                                workspaceBytes,
                                std::move(kernelName)});
    return out;
}

} // namespace

CatalogPlan::CatalogPlan(launch::HipModuleGuard module,
                         catalog::LaunchMetadata launchMetadata,
                         catalog::LaunchBindings bindings,
                         launch::SymbolTable gridSymbols,
                         size_t workspaceBytes,
                         std::string kernelName)
    : _candidates(makeSingle(std::move(module),
                             std::move(launchMetadata),
                             std::move(bindings),
                             std::move(gridSymbols),
                             workspaceBytes,
                             std::move(kernelName)))
{
}

CatalogPlan::CatalogPlan(std::vector<PlanCandidate> candidates,
                         catalog::TuneCache* cache,
                         std::string problemKey)
    : _candidates(std::move(candidates))
    , _cache(cache)
    , _problemKey(std::move(problemKey))
{
}

size_t CatalogPlan::getWorkspaceSize(const Handle& /*handle*/) const
{
    // The framework sizes the workspace before we know the winner, so reserve the
    // max any candidate needs -- every candidate uses <= its own <= max.
    size_t maxBytes = 0;
    for(const PlanCandidate& c : _candidates)
    {
        maxBytes = std::max(maxBytes, c.workspaceBytes);
    }
    return maxBytes;
}

void CatalogPlan::launchCandidate(const PlanCandidate& candidate,
                                  const Handle& handle,
                                  const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                  uint32_t numDeviceBuffers,
                                  void* workspace)
{
    // uid -> device pointer, from the runtime device-buffer list.
    std::unordered_map<int64_t, void*> uidToPtr;
    uidToPtr.reserve(numDeviceBuffers);
    for(uint32_t i = 0; i < numDeviceBuffers; ++i)
    {
        uidToPtr[deviceBuffers[i].uid] = deviceBuffers[i].ptr;
    }

    const launch::PointerResolver resolver = [&](int64_t uid) -> uint64_t {
        auto it = uidToPtr.find(uid);
        if(it == uidToPtr.end())
        {
            throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                             "aot-catalog: device buffer uid " + std::to_string(uid)
                                 + " not provided to execute()");
        }
        return reinterpret_cast<uint64_t>(it->second);
    };

    // A kernel that declares a "workspace" pointer argument gets the runtime
    // workspace buffer bound by value here.
    catalog::LaunchBindings bindings = candidate.bindings;
    if(workspace != nullptr)
    {
        bindings.pointerValues["workspace"] = reinterpret_cast<uint64_t>(workspace);
    }

    const std::vector<catalog::ScalarValue> bound
        = launch::bindArgs(candidate.launch.argsSignature, bindings, resolver);
    const std::vector<std::byte> kernargs = launch::packArgs(candidate.launch.argsSignature, bound);
    const launch::Grid grid = launch::evalGrid(candidate.launch.grid, candidate.gridSymbols);

    // HIP_LAUNCH_PARAM buffer mechanism: pass the packed kernarg blob by pointer
    // + size (same pattern as asm_sdpa_engine::launchKernel).
    size_t argSize = kernargs.size();
    // NOLINTNEXTLINE(modernize-avoid-c-arrays) - HIP API requires a C-style array
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      const_cast<std::byte*>(kernargs.data()),
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &argSize,
                      HIP_LAUNCH_PARAM_END};

    const hipError_t err = hipModuleLaunchKernel(candidate.module.function(),
                                                 grid.x,
                                                 grid.y,
                                                 grid.z,
                                                 candidate.launch.block[0],
                                                 candidate.launch.block[1],
                                                 candidate.launch.block[2],
                                                 candidate.launch.sharedMemBytes,
                                                 handle.getStream(),
                                                 nullptr, // kernel args (unused with config)
                                                 config);
    if(err != hipSuccess)
    {
        throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                         "aot-catalog: hipModuleLaunchKernel failed for '" + candidate.symbol
                             + "': " + hipGetErrorString(err));
    }

    HIPDNN_PLUGIN_LOG_INFO("aot-catalog: launched '"
                           << candidate.symbol << "' grid=[" << grid.x << "," << grid.y << ","
                           << grid.z << "] block=[" << candidate.launch.block[0] << ","
                           << candidate.launch.block[1] << "," << candidate.launch.block[2]
                           << "] kernarg_bytes=" << argSize);
}

size_t CatalogPlan::tuneAndSelect(const Handle& handle,
                                  const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                  uint32_t numDeviceBuffers,
                                  void* workspace) const
{
    // hipStream_t is an opaque handle, so the const qualifies the handle value.
    // NOLINTNEXTLINE(misc-misplaced-const)
    const hipStream_t stream = handle.getStream();

    hipEvent_t startEvent = nullptr;
    hipEvent_t stopEvent = nullptr;
    if(hipEventCreate(&startEvent) != hipSuccess || hipEventCreate(&stopEvent) != hipSuccess)
    {
        // Timing unavailable -- fall back to the first candidate (no cache store).
        if(startEvent != nullptr)
        {
            (void)hipEventDestroy(startEvent);
        }
        if(stopEvent != nullptr)
        {
            (void)hipEventDestroy(stopEvent);
        }
        HIPDNN_PLUGIN_LOG_WARN("aot-catalog: hipEventCreate failed; skipping tuning");
        return 0;
    }

    constexpr float NO_TIME = std::numeric_limits<float>::max();
    size_t bestIdx = 0;
    float bestMs = NO_TIME;
    std::ostringstream ranking;

    for(size_t i = 0; i < _candidates.size(); ++i)
    {
        const PlanCandidate& candidate = _candidates[i];
        float median = NO_TIME;
        try
        {
            // Warmup (module code-object load, caches) -- discarded.
            launchCandidate(candidate, handle, deviceBuffers, numDeviceBuffers, workspace);
            (void)hipStreamSynchronize(stream);

            std::vector<float> times;
            times.reserve(static_cast<size_t>(TIMED_LAUNCHES));
            for(int r = 0; r < TIMED_LAUNCHES; ++r)
            {
                if(hipEventRecord(startEvent, stream) != hipSuccess)
                {
                    throw std::runtime_error("hipEventRecord(start) failed");
                }
                launchCandidate(candidate, handle, deviceBuffers, numDeviceBuffers, workspace);
                if(hipEventRecord(stopEvent, stream) != hipSuccess)
                {
                    throw std::runtime_error("hipEventRecord(stop) failed");
                }
                if(hipEventSynchronize(stopEvent) != hipSuccess)
                {
                    throw std::runtime_error("hipEventSynchronize failed");
                }
                float ms = 0.0f;
                if(hipEventElapsedTime(&ms, startEvent, stopEvent) != hipSuccess)
                {
                    throw std::runtime_error("hipEventElapsedTime failed");
                }
                times.push_back(ms);
            }
            std::sort(times.begin(), times.end());
            median = times[times.size() / 2];
        }
        catch(const std::exception& e)
        {
            HIPDNN_PLUGIN_LOG_WARN("aot-catalog: candidate '"
                                   << candidate.symbol
                                   << "' failed while tuning, skipping: " << e.what());
            continue;
        }

        ranking << (i == 0 ? "" : ", ") << candidate.symbol << "=" << median << "ms";
        if(median < bestMs)
        {
            bestMs = median;
            bestIdx = i;
        }
    }

    if(bestMs == NO_TIME)
    {
        HIPDNN_PLUGIN_LOG_WARN(
            "aot-catalog: no candidate timed successfully; falling back to first");
        (void)hipEventDestroy(startEvent);
        (void)hipEventDestroy(stopEvent);
        return 0;
    }

    HIPDNN_PLUGIN_LOG_INFO("aot-catalog: tuned '" << _problemKey << "' -> '"
                                                  << _candidates[bestIdx].symbol << "' (" << bestMs
                                                  << "ms); ranking: " << ranking.str());

    if(_cache != nullptr && !_problemKey.empty())
    {
        _cache->store(_problemKey, _candidates[bestIdx].symbol, static_cast<double>(bestMs));
    }

    (void)hipEventDestroy(startEvent);
    (void)hipEventDestroy(stopEvent);
    return bestIdx;
}

void CatalogPlan::execute(const Handle& handle,
                          const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                          uint32_t numDeviceBuffers,
                          void* workspace) const
{
    // Single-candidate plan: launch directly (no tuning) -- unchanged behavior.
    if(_candidates.size() == 1)
    {
        launchCandidate(_candidates.front(), handle, deviceBuffers, numDeviceBuffers, workspace);
        return;
    }

    // Multi-candidate: use the cached winner if we have one, else tune now.
    size_t winner = std::numeric_limits<size_t>::max();
    if(_cache != nullptr && !_problemKey.empty())
    {
        if(const std::optional<std::string> cached = _cache->lookup(_problemKey);
           cached.has_value())
        {
            for(size_t i = 0; i < _candidates.size(); ++i)
            {
                if(_candidates[i].symbol == *cached)
                {
                    winner = i;
                    break;
                }
            }
            if(winner == std::numeric_limits<size_t>::max())
            {
                HIPDNN_PLUGIN_LOG_WARN("aot-catalog: cached winner '"
                                       << *cached << "' for '" << _problemKey
                                       << "' not among candidates; re-tuning");
            }
        }
    }

    if(winner == std::numeric_limits<size_t>::max())
    {
        winner = tuneAndSelect(handle, deviceBuffers, numDeviceBuffers, workspace);
    }

    // Always launch the winner last so the caller's output buffer holds its
    // result (candidates fully overwrite the output from unchanged inputs).
    launchCandidate(_candidates[winner], handle, deviceBuffers, numDeviceBuffers, workspace);
}

} // namespace aot_catalog_engine
