/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef ROCWMMA_TEST_HIP_DEVICE_HPP
#define ROCWMMA_TEST_HIP_DEVICE_HPP

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#if ROCWMMA_BENCHMARK_TESTS
#include <amd_smi/amdsmi.h>
#endif // ROCWMMA_BENCHMARK_TESTS
#include <rocwmma/internal/constants.hpp>

#include "performance.hpp"
#include "singleton.hpp"

// Single source of truth for the architectures supported by the test
// framework. Each entry expands to X(ARCH, FWD_ARG), where ARCH is the
// hipGcnArch_t enumerator name (matching Constants::AMDGCN_ARCH_ID_<ARCH>).
// Keep one arch per line so that adding an arch is a one-line change.
#define ROCWMMA_FOR_EACH_TEST_ARCH(X, FWD_ARG) \
    X(GFX908, FWD_ARG)                         \
    X(GFX90A, FWD_ARG)                         \
    X(GFX942, FWD_ARG)                         \
    X(GFX950, FWD_ARG)                         \
    X(GFX1100, FWD_ARG)                        \
    X(GFX1101, FWD_ARG)                        \
    X(GFX1102, FWD_ARG)                        \
    X(GFX1103, FWD_ARG)                        \
    X(GFX1150, FWD_ARG)                        \
    X(GFX1151, FWD_ARG)                        \
    X(GFX1152, FWD_ARG)                        \
    X(GFX1153, FWD_ARG)                        \
    X(GFX1170, FWD_ARG)                        \
    X(GFX1171, FWD_ARG)                        \
    X(GFX1200, FWD_ARG)                        \
    X(GFX1201, FWD_ARG)                        \
    X(GFX1250, FWD_ARG)                        \
    /* End of list */

namespace rocwmma
{

    class HipDevice : public LazySingleton<HipDevice>
    {
    public:
        // For static initialization
        friend std::unique_ptr<HipDevice> std::make_unique<HipDevice>();

        enum hipGcnArch_t : uint32_t
        {
            UNSUPPORTED_ARCH = Constants::AMDGCN_ARCH_ID_NONE,
#define ROCWMMA_TEST_ARCH_ENUM(ARCH, UNUSED) ARCH = Constants::AMDGCN_ARCH_ID_##ARCH,
            ROCWMMA_FOR_EACH_TEST_ARCH(ROCWMMA_TEST_ARCH_ENUM, )
#undef ROCWMMA_TEST_ARCH_ENUM
        };

        enum hipWarpSize_t : uint32_t
        {
            Wave32                = Constants::AMDGCN_WAVE_SIZE_32,
            Wave64                = Constants::AMDGCN_WAVE_SIZE_64,
            UNSUPPORTED_WARP_SIZE = Constants::AMDGCN_WAVE_SIZE_NONE,
        };

    protected:
        HipDevice();

    public:
        hipDevice_t     getDeviceHandle() const;
        hipDeviceProp_t getDeviceProps() const;
        hipDeviceArch_t getDeviceArch() const;
        hipGcnArch_t    getGcnArch() const;

        int warpSize() const;
        int sharedMemSize() const;
        int cuCount() const;
        int maxFreqMhz() const;
        int curFreqMhz() const;

        template <typename InputT>
        double peakGFlopsPerSec() const;

        ~HipDevice();

    private:
        hipDevice_t     mHandle;
        hipDeviceProp_t mProps;
        hipDeviceArch_t mArch;
        hipGcnArch_t    mGcnArch;
        int             mWarpSize;
        int             mSharedMemSize;
        int             mCuCount;
        int             mMaxFreqMhz;
        int             mCurFreqMhz;
    };

    template <typename InputT>
    double HipDevice::peakGFlopsPerSec() const
    {
        double result = -1.0;
        switch(mGcnArch)
        {
        case hipGcnArch_t::GFX908:
            result = calculatePeakGFlopsPerSec<InputT, ArchGfx908>(mCurFreqMhz, mCuCount);
            break;

        case hipGcnArch_t::GFX90A:
            result = calculatePeakGFlopsPerSec<InputT, ArchGfx90a>(mCurFreqMhz, mCuCount);
            break;

        default:
            result = calculatePeakGFlopsPerSec<InputT>(mCurFreqMhz, mCuCount);
        }
        return result;
    }
} // namespace rocwmma

#endif // ROCWMMA_TEST_HIP_DEVICE_HPP
