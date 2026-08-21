// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <utility>

#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace hip_kernel_provider::compilation
{

/// Move-only owner of a hipModule_t loaded from a kpack code object.
///
/// The pattern is asm_sdpa's HipModuleGuard (SdpaKernelUtils.hpp:147-215), deliberately
/// without that type's bound hipFunction_t: one kpack module can back several kernels
/// that differ only by entry point, which is exactly the sharing AC #6 asks for, so the
/// resolved function belongs to the Kernel and not to the module.
///
/// Not shared with HipModuleGuard: the asm_sdpa path is out of scope for this story and
/// refactoring it to a common base would put an unrelated engine in this diff.
class KpackModule
{
public:
    KpackModule() = default;

    explicit KpackModule(hipModule_t module)
        : _module(module)
    {
    }

    ~KpackModule()
    {
        if(_module != nullptr)
        {
            // Destructors do not throw, so a failed unload is reported and swallowed --
            // the same choice HipModuleGuard makes.
            const hipError_t status = hipModuleUnload(_module);
            if(status != hipSuccess)
            {
                HIPDNN_PLUGIN_LOG_WARN(
                    "hipModuleUnload failed for a kpack module: " << hipGetErrorString(status));
            }
        }
    }

    KpackModule(const KpackModule&) = delete;
    KpackModule& operator=(const KpackModule&) = delete;

    KpackModule(KpackModule&& other) noexcept
        : _module(std::exchange(other._module, nullptr))
    {
    }

    KpackModule& operator=(KpackModule&& other) noexcept
    {
        if(this != &other)
        {
            if(_module != nullptr)
            {
                const hipError_t status = hipModuleUnload(_module);
                if(status != hipSuccess)
                {
                    HIPDNN_PLUGIN_LOG_WARN(
                        "hipModuleUnload failed for a kpack module: " << hipGetErrorString(status));
                }
            }
            _module = std::exchange(other._module, nullptr);
        }
        return *this;
    }

    hipModule_t module() const
    {
        return _module;
    }

private:
    hipModule_t _module = nullptr;
};

} // namespace hip_kernel_provider::compilation

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
