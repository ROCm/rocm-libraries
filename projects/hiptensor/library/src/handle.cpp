/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#include <cstdio>

#include <hip/hip_runtime_api.h>

#include <hiptensor/hiptensor.h>

#include "data_types.hpp"
#include "handle.hpp"
#include "logger.hpp"

namespace hiptensor
{
    hiptensorStatus_t checkHandleDevice(const hiptensorHandle_t handle, char const* apiFuncName)
    {
        char msg[128];
        int  currentDeviceId = -1;

        auto hipStatus = hipGetDevice(&currentDeviceId);
        if(hipStatus != hipSuccess)
        {
            auto errorCode = HIPTENSOR_STATUS_HIP_ERROR;
            snprintf(msg,
                     sizeof(msg),
                     "Cannot query current device: %s (%s)",
                     hipGetErrorString(hipStatus),
                     hiptensorGetErrorString(errorCode));
            Logger::instance()->logError(apiFuncName, msg);
            return errorCode;
        }

        if(currentDeviceId != handle->getDevice().getDeviceId())
        {
            auto errorCode = HIPTENSOR_STATUS_ARCH_MISMATCH;
            snprintf(msg,
                     sizeof(msg),
                     "Device mismatch error: current device id: %d, handle device id: %d (%s)",
                     currentDeviceId,
                     (int)handle->getDevice().getDeviceId(),
                     hiptensorGetErrorString(errorCode));
            Logger::instance()->logError(apiFuncName, msg);
            return errorCode;
        }

        return HIPTENSOR_STATUS_SUCCESS;
    }
} // namespace hiptensor
