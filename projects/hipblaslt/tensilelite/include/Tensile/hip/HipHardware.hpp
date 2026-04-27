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

#pragma once

#include <Tensile/AMDGPU.hpp>
#include <Tensile/Tensile.hpp>
#include <origami/hardware.hpp>

#include <hip/hip_runtime.h>

namespace TensileLite
{
    namespace hip
    {
        struct HipAMDGPU : public AMDGPU
        {
            HipAMDGPU() = default;
            HipAMDGPU(hipDeviceProp_t const& prop,
                      int                    deviceId,
                      std::optional<int>     pciChipId = std::nullopt);

            hipDeviceProp_t properties;

            // HIP device ID this instance was constructed from. Retained so
            // multi-handle apps can recover which physical device an
            // AMDGPU/HipAMDGPU entry corresponds to without re-querying
            // hipGetDevice (which only returns the current device for the
            // calling thread). -1 means "unknown" (default-constructed).
            int deviceId = -1;

            std::shared_ptr<origami::hardware_t> analyticalHardware;

            virtual std::string archName() const override;
        };

        std::shared_ptr<Hardware> GetCurrentDevice();
        std::shared_ptr<Hardware> GetDevice(int deviceId);
        std::shared_ptr<Hardware> GetDevice(hipDeviceProp_t const& prop, int deviceId);
    } // namespace hip
} // namespace TensileLite
