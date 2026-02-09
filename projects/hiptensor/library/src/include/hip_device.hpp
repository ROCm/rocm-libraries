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

#pragma once

#include <hip/hip_runtime_api.h>

#include "hiptensor/hiptensor_types.hpp"

namespace hiptensor
{
    class HipDevice
    {
    public:
        enum hipGcnArch_t : uint32_t
        {
            GFX908           = 0x908,
            GFX90A           = 0x90A,
            GFX942           = 0x942,
            GFX950           = 0x950,
            GFX1100          = 0x1100,
            GFX1101          = 0x1101,
            GFX1102          = 0x1102,
            GFX1103          = 0x1103,
            GFX1150          = 0x1150,
            GFX1151          = 0x1151,
            GFX1152          = 0x1152,
            GFX1153          = 0x1153,
            GFX1200          = 0x1200,
            GFX1201          = 0x1201,
            UNSUPPORTED_ARCH = 0x0,
        };

        enum hipWarpSize_t : uint32_t
        {
            Wave32                = 32u,
            Wave64                = 64u,
            UNSUPPORTED_WARP_SIZE = 0u,
        };

        HipDevice();
        ~HipDevice() = default;

        hipDevice_t  getDeviceId() const;
        hipGcnArch_t getGcnArch() const;

        int warpSize() const;

        bool supportsF64() const;

        bool matrixCoreSupport(hiptensorComputeDescriptor_t typeCompute) const;

    private:
        hipDevice_t     mDeviceId;
        hipDeviceProp_t mProps;
        hipDeviceArch_t mArch;
        hipGcnArch_t    mGcnArch;
        int             mWarpSize;
        int             mSharedMemSize;
        int             mCuCount;
        int             mMaxFreqMhz;
    };

} // namespace hiptensor
