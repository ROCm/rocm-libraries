// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime_api.h>

#include <memory>
#include <unordered_map>

namespace rocblaslt
{
    /// The properties and the hardware description of each HIP device, by device
    /// ID. Devices of one architecture can still differ, in CU count and clocks,
    /// so each has an entry of its own.
    template <typename Hardware>
    class DeviceHardwareCache
    {
    public:
        /// Fills the entries of devices [0, count), asking getProperties(device)
        /// for its hipDeviceProp_t and getHardware(device) for its
        /// std::shared_ptr<Hardware>. Every device is asked, whether or not an
        /// earlier one has the same architecture.
        template <typename GetProperties, typename GetHardware>
        void populate(int count, GetProperties&& getProperties, GetHardware&& getHardware)
        {
            for(int device = 0; device < count; device++)
            {
                m_properties[device] = std::make_shared<hipDeviceProp_t>(getProperties(device));
                m_hardware[device]   = getHardware(device);
            }
        }

        /// Both accessors throw std::out_of_range for a device that was not filled.
        const std::shared_ptr<hipDeviceProp_t>& properties(int device) const
        {
            return m_properties.at(device);
        }

        std::shared_ptr<Hardware> hardware(int device) const
        {
            return m_hardware.at(device);
        }

    private:
        std::unordered_map<int, std::shared_ptr<hipDeviceProp_t>> m_properties;
        std::unordered_map<int, std::shared_ptr<Hardware>>        m_hardware;
    };
}
