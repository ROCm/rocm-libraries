/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <Tensile/AMDGPUPredicates.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace TensileLite
{
    namespace hip
    {
        namespace
        {
            // A copy of `prop` naming the base architecture instead of the
            // silicon revision the runtime reported.
            //
            // Origami matches gcnArchName exactly against a table of base
            // architectures, so a v0 part reporting gfx1250-strict matches
            // nothing and leaves analyticalHardware null, which throws on the
            // assert guarding every query that consults the analytical model.
            // The base entry's machine constants are the closest model a
            // revision has -- a deliberate approximation, since Origami
            // consumes CU counts and clocks rather than the ISA.
            //
            // Rebuilt through toProcessor rather than trimmed as a string so
            // the revisions folded here stay the ones Tensile recognises.
            // Feature suffixes such as `:xnack-` need no handling; Origami
            // drops everything past the first colon itself.
            hipDeviceProp_t propNamingBaseArchitecture(hipDeviceProp_t const& prop)
            {
                hipDeviceProp_t   base = prop;
                std::string const name = AMDGPU::toString(AMDGPU::toProcessor(prop.gcnArchName));
                std::snprintf(base.gcnArchName, sizeof(base.gcnArchName), "%s", name.c_str());
                return base;
            }

            std::string const& amdgpuModuleVersion()
            {
                static std::string const version = [] {
                    std::ifstream file("/sys/module/amdgpu/version");
                    std::string   text;
                    file >> text;
                    return text;
                }();
                return version;
            }

            long long readXgmiPhysicalId(hipDeviceProp_t const& prop)
            {
                char bdf[32]{};
                std::snprintf(bdf,
                              sizeof(bdf),
                              "%04x:%02x:%02x.0",
                              static_cast<unsigned>(prop.pciDomainID),
                              static_cast<unsigned>(prop.pciBusID),
                              static_cast<unsigned>(prop.pciDeviceID));
                std::ifstream file(std::string("/sys/bus/pci/devices/") + bdf + "/xgmi_physical_id");
                long long     id = -1;
                if(!(file >> id) || id < 0)
                    return -1;
                return id;
            }
        } // namespace

        bool page24RuleHolds(std::string const& amdgpuVersion)
        {
            std::istringstream in(amdgpuVersion);
            int                major = -1, minor = -1;
            char               dot   = 0;
            // The die bit is xgmi_physical_id & 1 from amdgpu 6.19 on. Older
            // drivers, including 6.16, do not provide that bit.
            return (in >> major >> dot >> minor) && dot == '.' && major >= 0 && minor >= 0
                   && (major > 6 || (major == 6 && minor >= 19));
        }

        std::optional<int32_t> aidPlacementBit(std::string const& amdgpuVersion,
                                               long long          xgmiPhysicalId)
        {
            if(!page24RuleHolds(amdgpuVersion) || xgmiPhysicalId < 0)
                return std::nullopt;
            return static_cast<int32_t>(xgmiPhysicalId & 1);
        }

        HipAMDGPU::HipAMDGPU(hipDeviceProp_t const& prop,
                             int                    deviceId,
                             std::optional<int>     pciChipId)
            : AMDGPU(AMDGPU::toProcessor(prop.gcnArchName),
                     prop.multiProcessorCount,
                     std::string(prop.name),
                     pciChipId)
            , properties(prop)
            , deviceId(deviceId)
            , amdgpuVersion(amdgpuModuleVersion())
            , xgmiPhysicalId(readXgmiPhysicalId(prop))
        {
            // Both entry points read gcnArchName out of the properties they are
            // handed, so the same renamed copy has to reach both: renaming only
            // the support check would trade the null for a throw out of
            // get_default_num_xcds when the lookup misses.
            hipDeviceProp_t const analyticalProp = propNamingBaseArchitecture(prop);
            if(origami::hardware_t::is_hardware_supported(analyticalProp))
            {
                // Route analyticalHardware construction through Origami's device-id
                // entry point so selection uses the runtime-queried XCC count
                // (hipDeviceAttributeNumberOfXccs on HIP 7+) instead of the hardcoded
                // per-architecture default in get_default_num_xcds().
                //
                // Pass a copy of the caller-provided `prop` so any adjustments made upstream
                // (e.g. overriding multiProcessorCount with
                // hipDeviceAttributePhysicalMultiProcessorCount on multi-XCC
                // architectures) are preserved. The int-only overload re-queries
                // `hipGetDeviceProperties` and would discard those adjustments.
                //
                // Pass PCI chip ID when known (e.g. from hipDeviceAttributePciChipId upstream);
                // Origami does not query PCI chip ID itself—nullopt selects gfx950 id75a0.
                analyticalHardware
                    = std::make_shared<origami::hardware_t>(origami::hardware_t::
                                                                get_hardware_for_device(
                                                                    deviceId,
                                                                    analyticalProp,
                                                                    pciChipId));
            }
        }

        std::optional<int32_t> HipAMDGPU::aidPlacementBit() const
        {
            return hip::aidPlacementBit(amdgpuVersion, xgmiPhysicalId);
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
            int hip_version;
            HIP_CHECK_EXC(hipRuntimeGetVersion(&hip_version));
            if(hip_version >= 50220730)
            {
                HIP_CHECK_EXC(hipDeviceGetAttribute(&prop.multiProcessorCount,
                                                    hipDeviceAttributePhysicalMultiProcessorCount,
                                                    deviceId));
            }
            return GetDevice(prop, deviceId);
        }

        std::shared_ptr<Hardware> GetDevice(hipDeviceProp_t const& prop, int deviceId)
        {
            const auto processor = AMDGPU::toProcessor(prop.gcnArchName);
            if(!ChipIdRegistry::supportsChipIdPredicate(processor))
            {
                return std::make_shared<HipAMDGPU>(prop, deviceId);
            }

            int        pciChipId = 0;
            hipError_t chipIdResult
                = hipDeviceGetAttribute(&pciChipId, hipDeviceAttributePciChipId, deviceId);

            // Check hip runtime support for PCI Chip ID attribute
            if(chipIdResult == hipErrorInvalidValue)
                throw std::runtime_error(pciChipIdUnsupportedErrorMessage(prop));

            // For any other error, use standard error checking
            HIP_CHECK_EXC(chipIdResult);

            if(!ChipIdRegistry::isKnownChipId(pciChipId))
                logUnregisteredPciChipIdWarningOnce(prop, pciChipId);

            return std::make_shared<HipAMDGPU>(prop, deviceId, std::make_optional(pciChipId));
        }
    } // namespace hip
} // namespace TensileLite
