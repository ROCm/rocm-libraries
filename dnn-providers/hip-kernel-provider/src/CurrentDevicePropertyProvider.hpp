// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "IDevicePropertyProvider.hpp"
#include "hip/HipUtils.hpp"

#include <hip/hip_runtime_api.h>

namespace hip_kernel_provider
{

class CurrentDevicePropertyProvider : public IDevicePropertyProvider
{
public:
    hipDeviceProp_t getDeviceProperties() const override
    {
        int device = 0;
        HIP_CHECK(hipGetDevice(&device));
        hipDeviceProp_t deviceProps;
        HIP_CHECK(hipGetDeviceProperties(&deviceProps, device));
        return deviceProps;
    }
};

} // namespace hip_kernel_provider
