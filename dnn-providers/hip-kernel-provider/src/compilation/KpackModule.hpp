// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "device/ScopedDevice.hpp"

#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace hip_kernel_provider::compilation
{

/// Non-copyable, non-movable owner of a hipModule_t loaded from a kpack code object.
///
/// The pattern is asm_sdpa's HipModuleGuard in SdpaKernelUtils.hpp, deliberately without
/// that type's bound hipFunction_t: one kpack module can back several kernels differing
/// only by entry point, so the resolved function belongs to the Kernel, not the module.
///
/// Immovable, not move-only: it is constructed in place by make_shared and every holder is
/// a shared_ptr<const KpackModule>, which cannot be moved from.
///
/// The ordinal is carried so the unload can name the load's device (see ScopedDevice): a
/// cache entry outlives the dispatch that filled it, so by destruction time the current
/// device is whatever the application last set.
class KpackModule
{
public:
    KpackModule() = default;

    KpackModule(hipModule_t module, int deviceOrdinal)
        : _module(module)
        , _deviceOrdinal(deviceOrdinal)
    {
    }

    ~KpackModule()
    {
        if(_module != nullptr)
        {
            // bound() is not consulted: an unload on the wrong device beats leaking the
            // module, and a destructor has nowhere to report a refusal. ScopedDevice logs it.
            const device::ScopedDevice binding(_deviceOrdinal);

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
    KpackModule(KpackModule&&) = delete;
    KpackModule& operator=(KpackModule&&) = delete;

    hipModule_t module() const
    {
        return _module;
    }

    int deviceOrdinal() const
    {
        return _deviceOrdinal;
    }

private:
    hipModule_t _module = nullptr;
    int _deviceOrdinal = 0;
};

} // namespace hip_kernel_provider::compilation

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
