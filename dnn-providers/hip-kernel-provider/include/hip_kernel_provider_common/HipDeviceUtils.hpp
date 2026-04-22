// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

namespace hip_kernel_provider_common
{

/**
 * @brief Gets the device architecture string (e.g., "gfx942") from a HIP stream.
 *
 * @param stream The HIP stream to query the device from
 * @return The device architecture string without the colon suffix
 * @throws std::runtime_error if the device query fails
 */
inline std::string getDeviceString(hipStream_t stream)
{
    int deviceId = -1;
    auto status = hipStreamGetDevice(stream, &deviceId);
    if(status != hipSuccess)
    {
        throw std::runtime_error("hipStreamGetDevice failed with error code: "
                                 + std::to_string(status));
    }

    hipDeviceProp_t props;
    status = hipGetDeviceProperties(&props, deviceId);
    if(status != hipSuccess)
    {
        throw std::runtime_error("hipGetDeviceProperties failed with error code: "
                                 + std::to_string(status));
    }
    std::string archStr(props.gcnArchName);

    return archStr.substr(0, archStr.find(':'));
}

} // namespace hip_kernel_provider_common
