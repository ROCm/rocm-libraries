// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_fp16.h>
#endif

#include <functional>

#include <rocRoller/DataTypes/DistinctType.hpp>

namespace rocRoller
{
//TODO: If ROCROLLER_USE_HIP is defined, invoke and use the HIP compiler
#if defined(ROCROLLER_USE_HIP) || defined(ROCROLLER_USE_FLOAT16_BUILTIN)
    /**
     * \ingroup DataTypes
     */
    using Half = __half;
#define ROCROLLER_USE_HALF
#else
    /**
     * \ingroup DataTypes
     */
    struct Half : public DistinctType<uint16_t, Half>
    {
    };
#endif
} // namespace rocRoller

namespace std
{
    inline ostream& operator<<(ostream& stream, const rocRoller::Half val)
    {
        return stream << static_cast<float>(val);
    }

    template <>
    struct hash<rocRoller::Half>
    {
        inline size_t operator()(rocRoller::Half const& h) const noexcept
        {
            hash<float> float_hash;
            return float_hash(static_cast<float>(h));
        }
    };

    template <>
    struct is_floating_point<rocRoller::Half> : std::true_type
    {
    };

    template <>
    struct is_signed<rocRoller::Half> : std::true_type
    {
    };
} // namespace std
