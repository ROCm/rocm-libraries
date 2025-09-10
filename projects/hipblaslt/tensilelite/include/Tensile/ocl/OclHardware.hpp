/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef OCL_HARDWARE_HPP
#define OCL_HARDWARE_HPP

#include <Tensile/AMDGPU.hpp>
#include <Tensile/Tensile.hpp>

namespace TensileLite
{
    namespace ocl
    {
        struct oclDeviceProp_t
        {
            std::string name;
            size_t      totalGlobalMem;
            size_t      sharedMemPerBlock;
            int         warpSize;
            int         maxThreadsPerBlock;
            int         maxThreadsDim[3];
            int         maxGridSize[3];
            int         clockRate;
            int         multiProcessorCount;
            int         pciBusID;
            int         pciDeviceID;
            size_t      maxSharedMemoryPerMultiProcessor;
            int         gcnArch;
        };

        struct OclAMDGPU : public AMDGPU
        {
            OclAMDGPU() = default;
            OclAMDGPU(oclDeviceProp_t const& prop);

            oclDeviceProp_t properties;
        };

        std::shared_ptr<Hardware> GetCurrentDevice();
        std::shared_ptr<Hardware> GetDevice(int deviceId);
        std::shared_ptr<Hardware> GetDevice(oclDeviceProp_t const& prop);

    } // namespace ocl
} // namespace TensileLite

#endif // OCL_HARDWARE_HPP
