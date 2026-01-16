// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_fp16.h>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <string>

#define HIPDNN_NAN_FP16 ushort_as_half(static_cast<unsigned short>(0x7FFFU))

inline __HOST_DEVICE__ half operator""_h(long double value)
{
    return {static_cast<float>(value)};
}

inline __HOST_DEVICE__ half operator-(half a)
{
    auto r = static_cast<__half_raw>(a);
    r.x ^= 0x8000u;
    return r;
}

inline __HOST_DEVICE__ bool operator==(half a, half b)
{
    return static_cast<float>(a) == static_cast<float>(b);
}

inline __HOST_DEVICE__ bool operator!=(half a, half b)
{
    return static_cast<float>(a) != static_cast<float>(b);
}

inline __HOST_DEVICE__ bool operator<(half a, half b)
{
    return static_cast<float>(a) < static_cast<float>(b);
}

inline __HOST_DEVICE__ bool operator>(half a, half b)
{
    return static_cast<float>(a) > static_cast<float>(b);
}

inline __HOST_DEVICE__ bool operator<=(half a, half b)
{
    return static_cast<float>(a) <= static_cast<float>(b);
}

inline __HOST_DEVICE__ bool operator>=(half a, half b)
{
    return static_cast<float>(a) >= static_cast<float>(b);
}

namespace hipdnn_data_sdk::utilities::fp16
{

inline __HOST_DEVICE__ half ushort_as_half(unsigned short x)
{
    __half_raw r;
    r.x = x;
    return r;
}

inline __HOST_DEVICE__ half habs(half num)
{
    auto raw = static_cast<__half_raw>(num);
    raw.x &= 0x7FFFu;
    return raw;
}

inline __HOST_DEVICE__ bool hisnan(__half x)
{
    auto hr = static_cast<__half_raw>(x);
    return (hr.x & 0x7FFFU) > 0x7C00u;
}

inline __HOST_DEVICE__ half hmax(const half a, const half b)
{
    auto aNan = hisnan(a);
    auto bNan = hisnan(b);

    if(aNan || bNan)
    {
        if(aNan && bNan)
        {
            return HIPDNN_NAN_FP16; // return canonical NaN
        }

        return aNan ? b : a;
    }

    return a > b ? a : b;
}

} // namespace hipdnn_data_sdk::utilities::fp16

namespace std
{

inline __HOST_DEVICE__ half fabs(half num)
{
    return hipdnn_data_sdk::utilities::fp16::habs(num);
}

inline __HOST_DEVICE__ half abs(half num)
{
    return hipdnn_data_sdk::utilities::fp16::habs(num);
}

inline __HOST_DEVICE__ half max(half a, half b)
{
    return hipdnn_data_sdk::utilities::fp16::hmax(a, b);
}

} // namespace std

template <>
struct fmt::formatter<half> : fmt::formatter<float>
{
    template <typename FormatContext>
    auto format(half h, FormatContext& ctx) const
    {
        return fmt::formatter<float>::format(static_cast<float>(h), ctx);
    }
};
