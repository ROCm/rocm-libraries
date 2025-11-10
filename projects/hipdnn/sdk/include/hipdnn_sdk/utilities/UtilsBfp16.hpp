// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Workaround: Operator<< in hip_bfloat16 is missing the std::ostream type info without this include.
#include <ostream>

#include <hip/hip_bfloat16.h>
#include <hipdnn_sdk/logging/Logger.hpp>
#include <string>

#define HIPDNN_NAN_BF16 ushort_as_bfloat16(static_cast<unsigned short>(0x7FFFU))

inline __HOST_DEVICE__ hip_bfloat16 operator""_bf(long double value)
{
    return hip_bfloat16(static_cast<float>(value));
}

namespace hipdnn_sdk::utilities::bfp16
{

inline __HOST_DEVICE__ hip_bfloat16 ushort_as_bfloat16(const unsigned short int a)
{
    hip_bfloat16 val;
    val.data = a;
    return val;
}

inline __HOST_DEVICE__ hip_bfloat16 habs(const hip_bfloat16 a)
{
    hip_bfloat16 abs = a;
    abs.data &= 0x7FFF;
    return abs;
}

inline __HOST_DEVICE__ bool hisnan(const hip_bfloat16 a)
{
    hip_bfloat16 hr = a;
    return !static_cast<bool>(~hr.data & 0x7f80) && static_cast<bool>(+(hr.data & 0x7f));
}

inline __HOST_DEVICE__ hip_bfloat16 hmax(const hip_bfloat16 a, const hip_bfloat16 b)
{
    auto aNan = hisnan(a);
    auto bNan = hisnan(b);

    if(aNan || bNan)
    {
        if(aNan && bNan)
        {
            return HIPDNN_NAN_BF16; // return canonical NaN
        }

        return aNan ? b : a;
    }

    return a.data > b.data ? a : b;
}

} // namespace hipdnn_sdk::utilities::bfp16

namespace std
{

inline __HOST_DEVICE__ hip_bfloat16 fabs(hip_bfloat16 num)
{
    return hipdnn_sdk::utilities::bfp16::habs(num);
}

inline __HOST_DEVICE__ hip_bfloat16 abs(hip_bfloat16 num)
{
    return hipdnn_sdk::utilities::bfp16::habs(num);
}

inline __HOST_DEVICE__ hip_bfloat16 max(hip_bfloat16 a, hip_bfloat16 b)
{
    return hipdnn_sdk::utilities::bfp16::hmax(a, b);
}

} // namespace std

template <>
struct fmt::formatter<hip_bfloat16> : fmt::formatter<float>
{
    template <typename FormatContext>
    auto format(hip_bfloat16 bf, FormatContext& ctx) const
    {
        return fmt::formatter<float>::format(static_cast<float>(bf), ctx);
    }
};
