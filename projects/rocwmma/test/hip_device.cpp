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

#include <vector>

#include "hip_device.hpp"
#include "common.hpp"

namespace rocwmma
{
    HipDevice::HipDevice()
        : mHandle(-1)
        , mGcnArch(hipGcnArch_t::UNSUPPORTED_ARCH)
        , mWarpSize(hipWarpSize_t::UNSUPPORTED_WARP_SIZE)
        , mSharedMemSize(0)
        , mCuCount(0)
        , mMaxFreqMhz(0)
        , mCurFreqMhz(0)
    {
        CHECK_HIP_ERROR(hipGetDevice(&mHandle));
        CHECK_HIP_ERROR(hipGetDeviceProperties(&mProps, mHandle));

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
        mCurFreqMhz    = mMaxFreqMhz;

#if ROCWMMA_BENCHMARK_TESTS
        bool smiErrorFlag = false;
        CHECK_AMDSMI_ERROR(amdsmi_init(AMDSMI_INIT_AMD_GPUS), smiErrorFlag);
        if(!smiErrorFlag)
        {
            uint64_t hipPCIID = 0;
            hipPCIID |= mProps.pciDeviceID & 0xFF;
            hipPCIID |= ((mProps.pciBusID & 0xFF) << 8);
            hipPCIID |= (mProps.pciDomainID) << 16;

            // Get processor handles for all sockets
            uint32_t socket_count = 0;
            CHECK_AMDSMI_ERROR(amdsmi_get_socket_handles(&socket_count, nullptr), smiErrorFlag);
            if(!smiErrorFlag && socket_count > 0)
            {
                std::vector<amdsmi_socket_handle> sockets(socket_count);
                CHECK_AMDSMI_ERROR(amdsmi_get_socket_handles(&socket_count, sockets.data()), smiErrorFlag);
                
                if(!smiErrorFlag)
                {
                    amdsmi_processor_handle smiDeviceHandle = nullptr;
                    bool deviceFound = false;
                    
                    // Iterate through all sockets to find matching device
                    for(uint32_t socket_idx = 0; socket_idx < socket_count && !deviceFound; socket_idx++)
                    {
                        uint32_t processor_count = 0;
                        CHECK_AMDSMI_ERROR(amdsmi_get_processor_handles(sockets[socket_idx], &processor_count, nullptr), smiErrorFlag);
                        if(smiErrorFlag || processor_count == 0)
                            continue;
                        
                        std::vector<amdsmi_processor_handle> processors(processor_count);
                        CHECK_AMDSMI_ERROR(amdsmi_get_processor_handles(sockets[socket_idx], &processor_count, processors.data()), smiErrorFlag);
                        if(smiErrorFlag)
                            continue;
                        
                        // Check each processor for PCI ID match
                        for(uint32_t proc_idx = 0; proc_idx < processor_count; proc_idx++)
                        {
                            amdsmi_bdf_t bdf;
                            CHECK_AMDSMI_ERROR(amdsmi_get_gpu_device_bdf(processors[proc_idx], &bdf), smiErrorFlag);
                            if(smiErrorFlag)
                                continue;
                            
                            // Construct PCI ID from BDF info
                            uint64_t amdsmiPCIID = 0;
                            amdsmiPCIID |= bdf.device_number & 0xFF;
                            amdsmiPCIID |= ((bdf.bus_number & 0xFF) << 8);
                            amdsmiPCIID |= (bdf.domain_number) << 16;
                            
                            if(hipPCIID == amdsmiPCIID)
                            {
                                smiDeviceHandle = processors[proc_idx];
                                deviceFound = true;
                                break;
                            }
                        }
                    }
                    if(!smiErrorFlag && deviceFound)
                    {
                        amdsmi_frequencies_t freq;
                        CHECK_AMDSMI_ERROR(
                            amdsmi_get_clk_freq(smiDeviceHandle, AMDSMI_CLK_TYPE_SYS, &freq),
                            smiErrorFlag);
                        if(!smiErrorFlag)
                        {
                            mCurFreqMhz = freq.frequency[freq.current] / 1000000;
                        }
                    }
                }
            }
        }
#endif // ROCWMMA_BENCHMARK_TESTS
    }

    hipDevice_t HipDevice::getDeviceHandle() const
    {
        return mHandle;
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

    int HipDevice::curFreqMhz() const
    {
        return mCurFreqMhz;
    }

    HipDevice::~HipDevice()
    {
#if ROCWMMA_BENCHMARK_TESTS
        bool smiErrorFlag = false;
        CHECK_AMDSMI_ERROR(amdsmi_shut_down(), smiErrorFlag);
#endif // ROCWMMA_BENCHMARK_TESTS
    }

    // Need to check the host device target support statically before hip modules attempt
    // to load any kernels. Not safe to proceed if the host device is unsupported.
    struct HipStaticDeviceGuard
    {
        static bool testSupportedDevice()
        {
            auto& device = HipDevice::instance();

            if((device->getGcnArch() == HipDevice::hipGcnArch_t::UNSUPPORTED_ARCH)
               || (device->warpSize() == HipDevice::hipWarpSize_t::UNSUPPORTED_WARP_SIZE))
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

} // namespace rocwmma
