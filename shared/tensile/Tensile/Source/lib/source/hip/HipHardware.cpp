/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/AMDGPU.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipUtils.hpp>

namespace Tensile
{
    namespace hip
    {
        HipAMDGPU::HipAMDGPU(hipDeviceProp_t const& prop)
            : AMDGPU(std::string(prop.gcnArchName),
                     prop.multiProcessorCount,
                     prop.directManagedMemAccessFromHost,
                     std::string(prop.name))
            , properties(prop)
        {
        }

        std::string HipAMDGPU::archName() const
        {
            return properties.gcnArchName;
        }

        std::shared_ptr<Hardware> GetCurrentDevice()
        {
            int deviceId = 0;
            HIP_CHECK_EXC(hipGetDevice(&deviceId));
            return GetDevice(deviceId);
        }

        std::shared_ptr<Hardware> GetDevice(int deviceId)
        {
            hipDeviceProp_t prop;
            HIP_CHECK_EXC(hipGetDeviceProperties(&prop, deviceId));
#if HIP_VERSION >= 50220730
            int hip_version;
            HIP_CHECK_EXC(hipRuntimeGetVersion(&hip_version));
            if(hip_version >= 50220730)
            {
                HIP_CHECK_EXC(hipDeviceGetAttribute(&prop.multiProcessorCount,
                                                    hipDeviceAttributePhysicalMultiProcessorCount,
                                                    deviceId));
                HIP_CHECK_EXC(
                    hipDeviceGetAttribute(&prop.directManagedMemAccessFromHost,
                                          hipDeviceAttributeDirectManagedMemAccessFromHost,
                                          deviceId));
            }
#endif

            return GetDevice(prop);
        }

        std::shared_ptr<Hardware> GetDevice(hipDeviceProp_t const& prop)
        {
            return std::make_shared<HipAMDGPU>(prop);
        }
    } // namespace hip
} // namespace Tensile
