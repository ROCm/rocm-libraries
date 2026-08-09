// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Loose-.co/HSACO module loader. Replaces PR #9207's kpack loader: a catalog
// KernelEntry points directly at a .co file on disk, which we hipModuleLoad and
// resolve to an exported function. Same RAII shape as
// asm_sdpa_engine::HipModuleGuard, but standalone (no SDPA coupling) so the
// throwaway catalog engine owns its own copy.

#pragma once

#include <hip/hip_runtime.h>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <optional>
#include <string>
#include <utility>

namespace aot_catalog_engine::launch
{

// Owns a hipModule_t (unloads in dtor) and holds its resolved hipFunction_t
// (non-owning; lifetime tied to the module). Move-only.
class HipModuleGuard
{
public:
    HipModuleGuard() = default;

    explicit HipModuleGuard(hipModule_t moduleIn, hipFunction_t functionIn = nullptr)
        : _module(moduleIn)
        , _function(functionIn)
    {
    }

    ~HipModuleGuard()
    {
        unload();
    }

    HipModuleGuard(const HipModuleGuard&) = delete;
    HipModuleGuard& operator=(const HipModuleGuard&) = delete;

    HipModuleGuard(HipModuleGuard&& other) noexcept
        : _module(std::exchange(other._module, nullptr))
        , _function(std::exchange(other._function, nullptr))
    {
    }

    HipModuleGuard& operator=(HipModuleGuard&& other) noexcept
    {
        if(this != &other)
        {
            unload();
            _module = std::exchange(other._module, nullptr);
            _function = std::exchange(other._function, nullptr);
        }
        return *this;
    }

    hipModule_t module() const
    {
        return _module;
    }

    hipFunction_t function() const
    {
        return _function;
    }

private:
    void unload()
    {
        if(_module != nullptr)
        {
            const hipError_t err = hipModuleUnload(_module);
            if(err != hipSuccess)
            {
                HIPDNN_PLUGIN_LOG_ERROR("aot-catalog: failed to unload kernel module, error: "
                                        << hipGetErrorString(err));
            }
            _module = nullptr;
            _function = nullptr;
        }
    }

    hipModule_t _module = nullptr;
    hipFunction_t _function = nullptr;
};

// hipModuleLoad(coPath) + hipModuleGetFunction(funcName). Returns nullopt on
// failure (logging the HIP error); on success the guard owns the module and
// holds the function pointer.
inline std::optional<HipModuleGuard> loadKernelModule(const std::string& coPath,
                                                      const std::string& funcName)
{
    hipModule_t rawModule = nullptr;
    hipError_t err = hipModuleLoad(&rawModule, coPath.c_str());
    if(err != hipSuccess)
    {
        HIPDNN_PLUGIN_LOG_ERROR("aot-catalog: failed to load kernel module: "
                                << coPath << " error: " << hipGetErrorString(err));
        return std::nullopt;
    }

    // NOTE: do not wrap `rawModule` in an owning guard before the returned one is
    // built -- a local guard would unload the module when it goes out of scope on
    // return, leaving the returned guard's function pointer dangling. There are no
    // throwing calls between here and the return, so unload manually on failure.
    hipFunction_t func = nullptr;
    err = hipModuleGetFunction(&func, rawModule, funcName.c_str());
    if(err != hipSuccess)
    {
        HIPDNN_PLUGIN_LOG_ERROR("aot-catalog: failed to resolve symbol '"
                                << funcName << "' in " << coPath
                                << " error: " << hipGetErrorString(err));
        (void)hipModuleUnload(rawModule);
        return std::nullopt;
    }

    return HipModuleGuard(rawModule, func);
}

} // namespace aot_catalog_engine::launch
