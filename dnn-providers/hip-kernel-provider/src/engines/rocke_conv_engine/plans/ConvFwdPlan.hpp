// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_runtime.h>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "ConvFwdParams.hpp"
#include "core/Handle.hpp"

namespace rocke_conv_engine
{

// RAII wrapper for a JIT-compiled HIP module + function pointer.
class ConvModuleGuard
{
public:
    ConvModuleGuard() = default;

    ConvModuleGuard(hipModule_t mod, hipFunction_t fn)
        : _module(mod)
        , _function(fn)
    {
    }

    ~ConvModuleGuard();

    ConvModuleGuard(const ConvModuleGuard&) = delete;
    ConvModuleGuard& operator=(const ConvModuleGuard&) = delete;

    ConvModuleGuard(ConvModuleGuard&& other) noexcept
        : _module(other._module)
        , _function(other._function)
    {
        other._module = nullptr;
        other._function = nullptr;
    }

    ConvModuleGuard& operator=(ConvModuleGuard&& other) noexcept;

    hipFunction_t function() const
    {
        return _function;
    }

private:
    hipModule_t _module = nullptr;
    hipFunction_t _function = nullptr;
};

class ConvFwdPlan : public hipdnn_plugin_sdk::IPlan<Handle>
{
public:
    ConvFwdPlan(ConvModuleGuard kernel, ConvFwdParams params);

    ~ConvFwdPlan() override = default;

    ConvFwdPlan(const ConvFwdPlan&) = delete;
    ConvFwdPlan& operator=(const ConvFwdPlan&) = delete;
    ConvFwdPlan(ConvFwdPlan&&) noexcept = default;
    ConvFwdPlan& operator=(ConvFwdPlan&&) noexcept = default;

    size_t getWorkspaceSize(const Handle& handle) const override;

    void execute(const Handle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    ConvModuleGuard _kernel;
    ConvFwdParams _params;
};

} // namespace rocke_conv_engine
