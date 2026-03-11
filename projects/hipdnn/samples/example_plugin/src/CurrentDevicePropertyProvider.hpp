// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "hip/HipUtils.hpp"
#include "hip/IDevicePropertyProvider.hpp"

namespace example_plugin
{

/// Concrete IDevicePropertyProvider that queries the current HIP device.
///
/// Calls hipGetDevice() and hipGetDeviceProperties() to retrieve the
/// device properties at runtime.
class CurrentDevicePropertyProvider : public IDevicePropertyProvider
{
public:
    hipDeviceProp_t getDeviceProperties() const override
    {
        int deviceId = 0;
        HIP_CHECK(hipGetDevice(&deviceId));

        hipDeviceProp_t props;
        HIP_CHECK(hipGetDeviceProperties(&props, deviceId));
        return props;
    }
};

} // namespace example_plugin
