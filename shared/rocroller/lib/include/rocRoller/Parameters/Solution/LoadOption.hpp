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
            enum class LoadPath : int
            {
                BufferToVGPR,
                BufferToLDSViaVGPR,
                BufferToLDS,
                GlobalToVGPR,
                GlobalToLDSViaVGPR,
                Count,
            };

            std::string   toString(LoadPath path);
            std::ostream& operator<<(std::ostream& stream, LoadPath const& path);
            std::istream& operator>>(std::istream& stream, LoadPath& path);

            MemoryType GetMemoryType(LoadPath const& path);
            bool       IsBufferToLDS(LoadPath const& path);
            bool       IsPathToLDS(LoadPath const& path);
        } // namespace Solution
    } // namespace Parameters

} // namespace rocRoller
