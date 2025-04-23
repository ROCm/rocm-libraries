/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "hip_device.hpp"
#include <hiptensor/internal/hiptensor_utility.hpp>

namespace hiptensor
{
    HipDevice::HipDevice()
        : mDeviceId(-1)
        , mGcnArch(hipGcnArch_t::UNSUPPORTED_ARCH)
        , mWarpSize(hipWarpSize_t::UNSUPPORTED_WARP_SIZE)
        , mSharedMemSize(0)
        , mCuCount(0)
        , mMaxFreqMhz(0)
    {
        CHECK_HIP_ERROR(hipGetDevice(&mDeviceId));
        CHECK_HIP_ERROR(hipGetDeviceProperties(&mProps, mDeviceId));

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

        switch(mProps.warpSize)
        {
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

    hipDeviceProp_t HipDevice::getDeviceProps() const
    {
        return mProps;
    }

    hipDeviceArch_t HipDevice::getDeviceArch() const
    {
        return mArch;
    }

    HipDevice::hipGcnArch_t HipDevice::getGcnArch() const
    {
        return mGcnArch;
    }

    int HipDevice::warpSize() const
    {
        return mWarpSize;
    }

    int HipDevice::sharedMemSize() const
    {
        return mSharedMemSize;
    }

    int HipDevice::cuCount() const
    {
        return mCuCount;
    }

    int HipDevice::maxFreqMhz() const
    {
        return mMaxFreqMhz;
    }

    bool HipDevice::supportsF64() const
    {
        return (mGcnArch == HipDevice::hipGcnArch_t::GFX90A
                || mGcnArch == HipDevice::hipGcnArch_t::GFX942
                || mGcnArch == HipDevice::hipGcnArch_t::GFX950);
    }

    // Need to check the host device target support statically before hip modules attempt
    // to load any kernels. Not safe to proceed if the host device is unsupported.
    struct HipStaticDeviceGuard
    {
        static bool testSupportedDevice()
        {
            auto device = HipDevice();

            if((device.getGcnArch() == HipDevice::hipGcnArch_t::UNSUPPORTED_ARCH)
               || (device.warpSize() == HipDevice::hipWarpSize_t::UNSUPPORTED_WARP_SIZE))
            {
                std::cerr << "Cannot proceed: unsupported host device detected. Exiting."
                          << std::endl;
                exit(EXIT_FAILURE);
            }
            return true;
        }
        static bool sResult;
    };

    bool HipStaticDeviceGuard::sResult = HipStaticDeviceGuard::testSupportedDevice();

} // namespace hiptensor
