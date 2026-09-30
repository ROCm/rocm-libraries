// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>
#ifdef ROCROLLER_USE_HIP
#include <hip/hip_runtime.h>
#endif

#include <rocRoller/DataTypes/DataTypes_F8_Utils.hpp>
#include <rocRoller/DataTypes/DataTypes_Scale_Utils.hpp>
#include <rocRoller/DataTypes/DistinctType.hpp>

namespace rocRoller
{
    /**
     * \ingroup DataTypes
     */
    struct E5M3
    {
        E5M3()
            : scale(0)
        {
        }

        explicit E5M3(uint8_t scale)
            : scale(scale)
        {
        }

        explicit E5M3(float scale)
        {
            this->scale = floatToE5M3(scale);
        }

        uint8_t scale;

        inline operator float() const
        {
            return E5M3ToFloat(this->scale);
        }

        explicit inline operator uint8_t() const
        {
            return this->scale;
        }
    };
    static_assert(sizeof(E5M3) == 1, "E5M3 must be 1 byte.");

    inline E5M3 operator-(E5M3 const& a)
    {
        return static_cast<E5M3>(-static_cast<float>(a));
    }

    inline std::ostream& operator<<(std::ostream& os, const E5M3 val)
    {
        os << val.scale;
        return os;
    }
} // namespace rocRoller

namespace std
{

    template <typename T>
    requires(std::is_convertible_v<T, uint8_t>&& std::is_integral_v<T>) inline bool
        operator==(rocRoller::E5M3 const& a, T const& b)
    {
        return a.scale == static_cast<uint8_t>(b);
    }

    template <typename T>
    requires(std::is_convertible_v<T, uint8_t>&& std::is_integral_v<T>) inline bool
        operator!=(rocRoller::E5M3 const& a, T const& b)
    {
        return a.scale != static_cast<uint8_t>(b);
    }

    template <>
    struct is_floating_point<rocRoller::E5M3> : true_type
    {
    };

    template <>
    struct hash<rocRoller::E5M3>
    {
        size_t operator()(const rocRoller::E5M3& a) const
        {
            return hash<uint8_t>()(a.scale);
        }
    };
} // namespace std
