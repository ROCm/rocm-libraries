// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "plans/RockeClientPlan.hpp"
#include "dispatcher/KpackModuleLoader.hpp"
#include "dispatcher/SdpaGraphAdapter.hpp"
#include "plans/LaunchAbi.hpp"
#include "plans/PluginError.hpp"

#include <rocm_kpack/kpack.h>

#include <array>
#include <cstddef>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocke_client
{
namespace
{

namespace fb = hipdnn_flatbuffers_sdk::flatbuffer_utilities;

void checkHip(hipError_t status, const char* call)
{
    if(status != hipSuccess)
    {
        throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                         std::string(call) + " failed: " + hipGetErrorString(status));
    }
}

void checkKpack(kpack_error_t status, const char* call)
{
    if(status != KPACK_SUCCESS)
    {
        throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                         std::string(call) + " failed with kpack_error_t "
                             + std::to_string(static_cast<int>(status)));
    }
}

// Makes the given device current for its lifetime and restores the previous
// device on scope exit. A HIP module (and the hipFunction_t derived from it) is
// bound to the device that was current when it was loaded, so both the load and
// every launch must run with the handle stream's device current -- which the
// dispatcher already proves need not be the thread-current device.
class ScopedDevice
{
public:
    explicit ScopedDevice(int device)
    {
        checkHip(hipGetDevice(&_previous), "hipGetDevice");
        if(device != _previous)
        {
            checkHip(hipSetDevice(device), "hipSetDevice");
            _restore = true;
        }
    }

    ~ScopedDevice()
    {
        if(_restore)
        {
            // Best-effort restore; a destructor cannot surface a plugin status.
            static_cast<void>(hipSetDevice(_previous));
        }
    }

    ScopedDevice(const ScopedDevice&) = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;
    ScopedDevice(ScopedDevice&&) = delete;
    ScopedDevice& operator=(ScopedDevice&&) = delete;

private:
    int _previous = 0;
    bool _restore = false;
};

std::unordered_map<std::int64_t, void*>
    makeDeviceBufferMap(const hipdnnPluginDeviceBuffer_t* deviceBuffers, uint32_t numDeviceBuffers)
{
    std::unordered_map<std::int64_t, void*> ptrs;
    ptrs.reserve(numDeviceBuffers);
    for(uint32_t index = 0; index < numDeviceBuffers; ++index)
    {
        ptrs[deviceBuffers[index].uid] = deviceBuffers[index].ptr;
    }
    return ptrs;
}

dispatcher::SdpaLaunchInputs buildLaunchInputsOrThrow(const fb::IGraph& graph)
{
    auto inputs = dispatcher::buildSdpaLaunchInputs(graph);
    if(!inputs.has_value())
    {
        throwPluginError(HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                         "rocke-client plan could not build launch inputs from the graph");
    }
    return *inputs;
}

} // namespace

void HipModule::reset(hipModule_t module) noexcept
{
    if(_module != nullptr)
    {
        const auto status = hipModuleUnload(_module);
        if(status != hipSuccess)
        {
            // Destructors cannot surface plugin status; preserve diagnostics on stderr.
            std::cerr << "rocke-client hipModuleUnload failed: " << hipGetErrorString(status)
                      << '\n';
        }
    }
    _module = module;
}

RockeClientPlan::RockeClientPlan(dispatcher::AotInstance instance,
                                 const fb::IGraph& graph,
                                 const RockeClientHandle& handle)
    : _instance(std::move(instance))
{
    // Decode the graph into op-agnostic launch bindings; the SDPA specifics stay
    // in the adapter and the plan holds only generic per-launch data. Grid symbols
    // are sourced from the selected instance's compile spec plus the runtime batch.
    auto inputs = buildLaunchInputsOrThrow(graph);
    _bindings = std::move(inputs.bindings);
    _gridSymbols = dispatcher::sdpaGridSymbols(_instance.compileSpec, inputs.batch);

    checkHip(hipStreamGetDevice(handle.getStream(), &_deviceId), "hipStreamGetDevice");
    const ScopedDevice deviceGuard(_deviceId);

    // Delegate archive open, HSACO extraction, module load and function lookup
    // to the shared kpack loader so kpack_* has a single reference site.
    const auto loaded = dispatcher::loadKernelFromKpack(_instance.runtime.kpackPath,
                                                        _instance.runtime.tocKey,
                                                        _instance.arch,
                                                        _instance.runtime.symbol);
    checkKpack(loaded.kpackError, "loadKernelFromKpack");
    checkHip(loaded.hipError, "loadKernelFromKpack");
    _module.reset(loaded.module);
    _function = loaded.fn;
}

RockeClientPlan::~RockeClientPlan() = default;

size_t RockeClientPlan::getWorkspaceSize(const RockeClientHandle& /*handle*/) const
{
    return 0;
}

void RockeClientPlan::execute(const RockeClientHandle& handle,
                              const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                              uint32_t numDeviceBuffers,
                              void* /*workspace*/) const
{
    if(deviceBuffers == nullptr)
    {
        throwPluginError(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                         "rocke-client execute received null buffers");
    }

    const ScopedDevice deviceGuard(_deviceId);

    const auto ptrs = makeDeviceBufferMap(deviceBuffers, numDeviceBuffers);
    const auto argValues
        = launch::bindArgs(_instance.runtime.launch.argsSignature, _bindings, ptrs);
    auto packed = launch::packArgs(_instance.runtime.launch.argsSignature, argValues);
    auto argSize = packed.size();
    std::array<void*, 5> config = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                                   packed.data(),
                                   HIP_LAUNCH_PARAM_BUFFER_SIZE,
                                   &argSize,
                                   HIP_LAUNCH_PARAM_END};
    const auto grid = launch::evalGrid(_instance.runtime.launch.grid, _gridSymbols);
    const auto& block = _instance.runtime.launch.block;

    checkHip(
        hipModuleLaunchKernel(_function,
                              grid[0],
                              grid[1],
                              grid[2],
                              block[0],
                              block[1],
                              block[2],
                              static_cast<unsigned int>(_instance.runtime.launch.sharedMemBytes),
                              handle.getStream(),
                              nullptr,
                              config.data()),
        "hipModuleLaunchKernel");
}

} // namespace rocke_client
