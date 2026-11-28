// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <cstdio>

namespace TensileLite
{
    namespace Client
    {
        struct PciBusId
        {
            unsigned domain   = 0;
            unsigned bus      = 0;
            unsigned device   = 0;
            unsigned function = 0;
        };

        // Parses the "domain:bus:device.function" hex form of hipDeviceGetPCIBusId.
        inline bool parsePciBusId(char const* text, PciBusId& id)
        {
            return std::sscanf(text, "%x:%x:%x.%x", &id.domain, &id.bus, &id.device, &id.function)
                   == 4;
        }

        // The device's HSA_AMD_AGENT_INFO_BDFID: bus in [15:8], device in [7:3], function in [2:0].
        inline uint32_t pciBdfId(PciBusId const& id)
        {
            return (id.bus << 8) | (id.device << 3) | id.function;
        }
    } // namespace Client
} // namespace TensileLite
