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

#include <tensilelitehost/export.h>

#include <cstdint>
#include <optional>
#include <string>

namespace TensileLite
{
    namespace hip
    {
        struct TENSILELITEHOST_EXPORT HipAMDGPU : public AMDGPU
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

            // Read once when the device is constructed: /sys/module/amdgpu/version and the
            // device's /sys/bus/pci/devices/<bdf>/xgmi_physical_id. Empty and -1 when
            // unreadable.
            std::string amdgpuVersion;
            long long   xgmiPhysicalId = -1;

            // The AIDPlacementBit kernel argument: aidPlacementBit(amdgpuVersion, xgmiPhysicalId).
            std::optional<int32_t> aidPlacementBit() const;

            virtual std::string archName() const override;
        };

        // Whether the near-die rule, xgmi_physical_id & 1, holds on this amdgpu
        // version: amdgpu 6.19 and newer.
        TENSILELITEHOST_EXPORT bool page24RuleHolds(std::string const& amdgpuVersion);

        // xgmiPhysicalId & 1 where the near-die rule holds and the id is known,
        // otherwise nullopt.
        TENSILELITEHOST_EXPORT std::optional<int32_t> aidPlacementBit(std::string const& amdgpuVersion,
                                                                      long long xgmiPhysicalId);

        TENSILELITEHOST_EXPORT std::shared_ptr<Hardware> GetCurrentDevice();
        TENSILELITEHOST_EXPORT std::shared_ptr<Hardware> GetDevice(int deviceId);
        TENSILELITEHOST_EXPORT std::shared_ptr<Hardware> GetDevice(hipDeviceProp_t const& prop, int deviceId);
    } // namespace hip
} // namespace TensileLite

