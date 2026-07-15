// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>

#include "RockeClientHandle.hpp"
#include "dispatcher/AotInstance.hpp"

namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities
{
class IGraph;
}

namespace rocke_client
{

// RAII owner for a HIP module. Unloading reports (but never throws on) an unload
// failure so destruction stays noexcept; a throw partway through the plan
// constructor still unloads any module already held.
class HipModule
{
public:
    HipModule() = default;
    explicit HipModule(hipModule_t module) noexcept
        : _module(module)
    {
    }
    ~HipModule()
    {
        reset();
    }

    HipModule(HipModule&& other) noexcept
        : _module(std::exchange(other._module, nullptr))
    {
    }
    HipModule& operator=(HipModule&& other) noexcept
    {
        if(this != &other)
        {
            reset(std::exchange(other._module, nullptr));
        }
        return *this;
    }

    HipModule(const HipModule&) = delete;
    HipModule& operator=(const HipModule&) = delete;

    void reset(hipModule_t module = nullptr) noexcept;
    hipModule_t get() const noexcept
    {
        return _module;
    }

private:
    hipModule_t _module = nullptr;
};

class RockeClientPlan final : public hipdnn_plugin_sdk::IPlan<RockeClientHandle>
{
public:
    RockeClientPlan(dispatcher::AotInstance instance,
                    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                    const RockeClientHandle& handle);
    ~RockeClientPlan() override;

    RockeClientPlan(const RockeClientPlan&) = delete;
    RockeClientPlan& operator=(const RockeClientPlan&) = delete;
    RockeClientPlan(RockeClientPlan&&) = delete;
    RockeClientPlan& operator=(RockeClientPlan&&) = delete;

    size_t getWorkspaceSize(const RockeClientHandle& handle) const override;

    void execute(const RockeClientHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    dispatcher::AotInstance _instance;
    dispatcher::LaunchBindings _bindings;
    std::unordered_map<std::string, std::int64_t> _gridSymbols;
    int _deviceId = 0;
    HipModule _module;
    hipFunction_t _function = nullptr;
};

} // namespace rocke_client
