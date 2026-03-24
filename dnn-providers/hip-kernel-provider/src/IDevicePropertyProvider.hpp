// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_runtime_api.h>

namespace hip_kernel_provider
{

class IDevicePropertyProvider
{
public:
    virtual ~IDevicePropertyProvider() = default;

    virtual hipDeviceProp_t getDeviceProperties() const = 0;
};

} // namespace hip_kernel_provider
