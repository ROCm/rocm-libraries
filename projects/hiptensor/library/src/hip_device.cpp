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

#include <string>

#include "hip_device.hpp"
#include "platform.hpp"

namespace hiptensor
{
    HipDevice::HipDevice()
        : mDeviceId(-1)
        , mProps{}
        , mArch{}
        , mGcnArch(hipGcnArch_t::UNSUPPORTED_ARCH)
        , mWarpSize(hipWarpSize_t::UNSUPPORTED_WARP_SIZE)
        , mSharedMemSize(0)
        , mCuCount(0)
        , mMaxFreqMhz(0)
        , mQueryStatus(hipSuccess)
    {
        // A failed query (e.g. on a host without a visible GPU) is recorded rather than
        // fatal: hiptensorCreate reports it, so loading the library never ends the process.
        mQueryStatus = hipGetDevice(&mDeviceId);
        if(mQueryStatus == hipSuccess)
        {
            mQueryStatus = hipGetDeviceProperties(&mProps, mDeviceId);
        }
        if(mQueryStatus != hipSuccess)
        {
            mDeviceId = -1;
            return;
        }

        mArch = mProps.arch;

        std::string deviceName(mProps.gcnArchName);

        if(deviceName.find("gfx908") != std::string::npos)
        {
            mGcnArch = hipGcnArch_t::GFX908;
        }
        else if(deviceName.find("gfx90a") != std::string::npos)
        {
            mGcnArch = hipGcnArch_t::GFX90A;
        }
        else if(deviceName.find("gfx942") != std::string::npos)
        {
            mGcnArch = hipGcnArch_t::GFX942;
        }
        else if(deviceName.find("gfx950") != std::string::npos)
        {
            mGcnArch = hipGcnArch_t::GFX950;
        }
        else if(deviceName.find("gfx1100") != std::string::npos)
        {
            mGcnArch = hipGcnArch_t::GFX1100;
        }
        else if(deviceName.find("gfx1101") != std::string::npos)
        {
            mGcnArch = hipGcnArch_t::GFX1101;
        }
        else if(deviceName.find("gfx1102") != std::string::npos)
        {
            mGcnArch = hipGcnArch_t::GFX1102;
        }
        else if(deviceName.find("gfx1103") != std::string::npos)
        {
            mGcnArch = hipGcnArch_t::GFX1103;
        }
        else if(deviceName.find("gfx1150") != std::string::npos)
        {
            mGcnArch = hipGcnArch_t::GFX1150;
        }
        else if(deviceName.find("gfx1151") != std::string::npos)
        {
            mGcnArch = hipGcnArch_t::GFX1151;
        }
        else if(deviceName.find("gfx1152") != std::string::npos)
        {
            mGcnArch = hipGcnArch_t::GFX1152;
        }
        else if(deviceName.find("gfx1153") != std::string::npos)
        {
            mGcnArch = hipGcnArch_t::GFX1153;
        }
        else if(deviceName.find("gfx1200") != std::string::npos)
        {
            mGcnArch = hipGcnArch_t::GFX1200;
        }
        else if(deviceName.find("gfx1201") != std::string::npos)
        {
            mGcnArch = hipGcnArch_t::GFX1201;
        }
        else if(deviceName.find("gfx1250") != std::string::npos)
        {
            mGcnArch = hipGcnArch_t::GFX1250;
        }

        // Test hook: classify the device as unsupported, so that the handling of an unsupported
        // architecture can be tested on any GPU. It can only take support away: it never makes a
        // device supported or classifies it as another architecture.
        if(auto forceUnsupported = getEnvironmentVariable("HIPTENSOR_TEST_FORCE_UNSUPPORTED_ARCH");
           forceUnsupported && (*forceUnsupported == "1" || *forceUnsupported == "ON"))
        {
            mGcnArch = hipGcnArch_t::UNSUPPORTED_ARCH;
        }

        switch(mProps.warpSize)
        {
        case hipWarpSize_t::Wave32:
        case hipWarpSize_t::Wave64:
            mWarpSize = mProps.warpSize;
        default:;
        }

        mSharedMemSize = mProps.sharedMemPerBlock;
        mCuCount       = mProps.multiProcessorCount;
        mMaxFreqMhz    = static_cast<int>(static_cast<double>(mProps.clockRate) / 1000.0);
    }

    hipDevice_t HipDevice::getDeviceId() const
    {
        return mDeviceId;
    }

    HipDevice::hipGcnArch_t HipDevice::getGcnArch() const
    {
        return mGcnArch;
    }

    std::string HipDevice::getArchName() const
    {
        return std::string(mProps.gcnArchName);
    }

    int HipDevice::getCuCount() const
    {
        return mCuCount;
    }

    int HipDevice::warpSize() const
    {
        return mWarpSize;
    }

    hipError_t HipDevice::queryStatus() const
    {
        return mQueryStatus;
    }

    bool HipDevice::isSupported() const
    {
        return mQueryStatus == hipSuccess && mGcnArch != hipGcnArch_t::UNSUPPORTED_ARCH
               && mWarpSize != hipWarpSize_t::UNSUPPORTED_WARP_SIZE;
    }

    bool HipDevice::supportsF64() const
    {
        return (mGcnArch == HipDevice::hipGcnArch_t::GFX90A
                || mGcnArch == HipDevice::hipGcnArch_t::GFX942
                || mGcnArch == HipDevice::hipGcnArch_t::GFX950
                || mGcnArch == HipDevice::hipGcnArch_t::GFX1250);
    }

    bool HipDevice::matrixCoreSupport(hiptensorComputeDescriptor_t typeCompute) const
    {
        switch(typeCompute)
        {
        case HIPTENSOR_COMPUTE_DESC_64F:
        case HIPTENSOR_COMPUTE_DESC_C64F:
            return (mGcnArch == HipDevice::hipGcnArch_t::GFX90A
                    || mGcnArch == HipDevice::hipGcnArch_t::GFX942
                    || mGcnArch == HipDevice::hipGcnArch_t::GFX950);
        case HIPTENSOR_COMPUTE_DESC_32F:
        case HIPTENSOR_COMPUTE_DESC_C32F:
            return (mGcnArch == HipDevice::hipGcnArch_t::GFX908
                    || mGcnArch == HipDevice::hipGcnArch_t::GFX90A
                    || mGcnArch == HipDevice::hipGcnArch_t::GFX942
                    || mGcnArch == HipDevice::hipGcnArch_t::GFX950
                    || mGcnArch == HipDevice::hipGcnArch_t::GFX1250);
        case HIPTENSOR_COMPUTE_DESC_16F:
        case HIPTENSOR_COMPUTE_DESC_16BF:
            return (mGcnArch == HipDevice::hipGcnArch_t::GFX908
                    || mGcnArch == HipDevice::hipGcnArch_t::GFX90A
                    || mGcnArch == HipDevice::hipGcnArch_t::GFX942
                    || mGcnArch == HipDevice::hipGcnArch_t::GFX950
                    || mGcnArch == HipDevice::hipGcnArch_t::GFX1100
                    || mGcnArch == HipDevice::hipGcnArch_t::GFX1101
                    || mGcnArch == HipDevice::hipGcnArch_t::GFX1102
                    || mGcnArch == HipDevice::hipGcnArch_t::GFX1103
                    || mGcnArch == HipDevice::hipGcnArch_t::GFX1150
                    || mGcnArch == HipDevice::hipGcnArch_t::GFX1151
                    || mGcnArch == HipDevice::hipGcnArch_t::GFX1152
                    || mGcnArch == HipDevice::hipGcnArch_t::GFX1153
                    || mGcnArch == HipDevice::hipGcnArch_t::GFX1200
                    || mGcnArch == HipDevice::hipGcnArch_t::GFX1201
                    || mGcnArch == HipDevice::hipGcnArch_t::GFX1250);
        default:
            return false;
        }
    }

} // namespace hiptensor
