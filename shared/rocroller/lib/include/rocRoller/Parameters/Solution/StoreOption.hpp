// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/DataTypes/DataTypes.hpp>

#include <string>

namespace rocRoller
{
    namespace Parameters
    {
        namespace Solution
        {
            enum class StorePath : int
            {
                VGPRToGlobalMemoryWithBuffer, // Store from VGPR to buffer using buffer_store_X
                VGPRToGlobalMemoryWithGlobal, // Store from VGPR to global using global_store_X
                VGPRToGlobalMemoryViaLDSWithBuffer, // Store to LDS first, then to buffer (former storeLDSD=true)
                VGPRToGlobalMemoryViaLDSWithGlobal, // Store to LDS first, then to global
                Count,
            };

            std::string   toString(StorePath path);
            std::ostream& operator<<(std::ostream& stream, StorePath const& path);
            std::istream& operator>>(std::istream& stream, StorePath& path);

            MemoryType GetMemoryType(StorePath const& path);
            bool       IsLDSStore(StorePath const& path);
        } // namespace Solution
    } // namespace Parameters
} // namespace rocRoller
