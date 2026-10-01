// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace hip_kernel_provider::device
{

// Falls back to 0 rather than reporting -- whatever the caller does next hits the same HIP
// failure and reports it in its own terms.
inline int currentDeviceOrdinal()
{
    int ordinal = 0;
    if(hipGetDevice(&ordinal) != hipSuccess)
    {
        return 0;
    }
    return ordinal;
}

// HIP binds a module to the device current at hipModuleLoadData, and binds the unload the
// same way: keying a cache on the ordinal separates entries, it does not place them.
// Restoring keeps asking for a kernel from being a visible side effect.
//
// Nothing here throws -- the caller decides whether a failed bind is fatal by reading
// bound().
class ScopedDevice
{
public:
    explicit ScopedDevice(int ordinal)
    {
        if(hipGetDevice(&_previous) != hipSuccess)
        {
            return;
        }
        if(_previous == ordinal)
        {
            _bound = true;
            return;
        }
        if(hipSetDevice(ordinal) != hipSuccess)
        {
            return;
        }
        _bound = true;
        _restore = true;
    }

    ~ScopedDevice()
    {
        if(_restore)
        {
            const hipError_t status = hipSetDevice(_previous);
            if(status != hipSuccess)
            {
                HIPDNN_PLUGIN_LOG_WARN("could not restore device "
                                       << _previous
                                       << " after a scoped switch: " << hipGetErrorString(status));
            }
        }
    }

    ScopedDevice(const ScopedDevice&) = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;
    ScopedDevice(ScopedDevice&&) = delete;
    ScopedDevice& operator=(ScopedDevice&&) = delete;

    // False means HIP refused, and the caller is on whatever device it started on.
    bool bound() const
    {
        return _bound;
    }

private:
    int _previous = 0;
    bool _bound = false;
    bool _restore = false;
};

} // namespace hip_kernel_provider::device
