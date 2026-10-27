// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/**
 * Translates hipBLASLt's description of a kernel into rocRoller's.
 *
 * hipBLASLt describes a kernel's operands the way the rest of the library
 * does: an element format is a hipDataType. rocRoller, the generator hipBLASLt
 * drives when it is enabled, has an enumeration of its own for the same
 * formats, so the two are translated here and nowhere else. A kernel's scaling
 * arrangement never reaches rocRoller as a value, so it needs no translation.
 *
 * This mirrors hip2TensileType in tensile_host.cpp, which does the same job for
 * the TensileLite backend. Only sources built against rocRoller's headers
 * include this file; everything the tile ranker reaches stops at the hipBLASLt
 * types. When rocRoller is retired this header goes with it, and the code that
 * describes a kernel does not change.
 *
 * The translation rejects a format it does not name, so a format that gains a
 * kernel without gaining a translation stops the run instead of being silently
 * generated as some other format.
 */

#include <hipblaslt/hipblaslt.h>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace hipblaslt
{
    /**
     * @brief The rocRoller element format that stands for this hipDataType.
     */
    inline rocRoller::DataType hip2RocRollerType(hipDataType type)
    {
        // Compared as integers because a HIP release that predates these
        // formats declares them as constants rather than enumerators, which is
        // how the rest of hipBLASLt reads them too (see utility.cpp).
        if(static_cast<int>(type) == HIP_R_6F_E2M3)
            return rocRoller::DataType::FP6;
        if(static_cast<int>(type) == HIP_R_6F_E3M2)
            return rocRoller::DataType::BF6;
        if(static_cast<int>(type) == HIP_R_4F_E2M1)
            return rocRoller::DataType::FP4;

        switch(type)
        {
        case HIP_R_32F:
            return rocRoller::DataType::Float;
        case HIP_R_64F:
            return rocRoller::DataType::Double;
        case HIP_R_16F:
            return rocRoller::DataType::Half;
        case HIP_R_16BF:
            return rocRoller::DataType::BFloat16;
        case HIP_R_32I:
            return rocRoller::DataType::Int32;
        case HIP_R_8F_E4M3:
            return rocRoller::DataType::FP8;
        case HIP_R_8F_E5M2:
            return rocRoller::DataType::BF8;
        case HIP_R_8F_UE8M0:
            return rocRoller::DataType::E8M0;
        default:
            break;
        }

        rocRoller::Throw<rocRoller::FatalError>(
            "hipBLASLt asked rocRoller for a kernel whose element format rocRoller has no "
            "translation for: hipDataType ",
            static_cast<int>(type));
    }

} // namespace hipblaslt
