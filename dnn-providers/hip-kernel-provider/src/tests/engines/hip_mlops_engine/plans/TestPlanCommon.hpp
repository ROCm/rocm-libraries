// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdio>
#include <hip/hip_runtime_api.h>

inline hipDeviceProp_t createTestDeviceProps(const char* archName = "gfx942")
{
    hipDeviceProp_t deviceProps = {};
    deviceProps.multiProcessorCount = 60;
    deviceProps.warpSize = 64;
    std::snprintf(deviceProps.gcnArchName, sizeof(deviceProps.gcnArchName), "%s", archName);
    return deviceProps;
}
