// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_runtime_api.h>
#include <stdexcept>
#include <string>

namespace hipdnn_gpu_ref::detail
{

inline void throwOnHipError(hipError_t err, const char* msg)
{
    if(err != hipSuccess)
    {
        throw std::runtime_error(std::string(msg) + ": " + hipGetErrorString(err));
    }
}

} // namespace hipdnn_gpu_ref::detail
