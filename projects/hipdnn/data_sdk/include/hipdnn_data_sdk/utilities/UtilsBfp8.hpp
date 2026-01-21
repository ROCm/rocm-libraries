// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_fp8.h>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <string>
#include <type_traits>

#define HIPDNN_NAN_BFP8 \
    hipdnn_data_sdk::utilities::bfp8::uchar_as_bfp8(static_cast<unsigned char>(0x7F))

using hip_fp8_e5m2 = __hip_fp8_e5m2;

inline __HOST_DEVICE__ hip_fp8_e5m2 operator""_bfp8(long double val)
{
    return {static_cast<float>(val)};
}

inline __HOST_DEVICE__ hip_fp8_e5m2 operator-(hip_fp8_e5m2 a)
{
    hip_fp8_e5m2 val = a;
    val.__x ^= 0x80;
    return val;
}

inline __HOST_DEVICE__ bool operator==(hip_fp8_e5m2 a, hip_fp8_e5m2 b)
{
    return float(a) == float(b);
}

inline __HOST_DEVICE__ bool operator!=(hip_fp8_e5m2 a, hip_fp8_e5m2 b)
{
    return float(a) != float(b);
}

inline __HOST_DEVICE__ bool operator>(hip_fp8_e5m2 a, hip_fp8_e5m2 b)
{
    return float(a) > float(b);
}

inline __HOST_DEVICE__ bool operator>=(hip_fp8_e5m2 a, hip_fp8_e5m2 b)
{
    return float(a) >= float(b);
}

inline __HOST_DEVICE__ bool operator<(hip_fp8_e5m2 a, hip_fp8_e5m2 b)
{
    return float(a) < float(b);
}

inline __HOST_DEVICE__ bool operator<=(hip_fp8_e5m2 a, hip_fp8_e5m2 b)
{
    return float(a) <= float(b);
}

namespace hipdnn_data_sdk::utilities::bfp8
{

inline __HOST_DEVICE__ hip_fp8_e5m2 uchar_as_bfp8(const unsigned char a)
{
    hip_fp8_e5m2 val;
    val.__x = a;
    return val;
}

inline __HOST_DEVICE__ hip_fp8_e5m2 bfp8abs(hip_fp8_e5m2 a)
{
    hip_fp8_e5m2 abs = a;
    abs.__x &= 0x7f;
    return abs;
}

inline __HOST_DEVICE__ bool bfp8isnan(hip_fp8_e5m2 a)
{
    return (a.__x & 0x7f) > 0x7c;
}

inline __HOST_DEVICE__ hip_fp8_e5m2 bfp8max(const hip_fp8_e5m2 a, const hip_fp8_e5m2 b)
{
    auto aNan = bfp8isnan(a);
    auto bNan = bfp8isnan(b);

    if(aNan || bNan)
    {
        if(aNan && bNan)
        {
            return HIPDNN_NAN_BFP8; // return canonical NaN
        }

        return aNan ? b : a;
    }

    return a >= b ? a : b;
}

} // namespace hipdnn_data_sdk::utilities::bfp8

namespace std
{

inline __HOST_DEVICE__ hip_fp8_e5m2 fabs(hip_fp8_e5m2 num)
{
    return hipdnn_data_sdk::utilities::bfp8::bfp8abs(num);
}

inline __HOST_DEVICE__ hip_fp8_e5m2 abs(hip_fp8_e5m2 num)
{
    return hipdnn_data_sdk::utilities::bfp8::bfp8abs(num);
}

inline __HOST_DEVICE__ hip_fp8_e5m2 max(hip_fp8_e5m2 a, hip_fp8_e5m2 b)
{
    return hipdnn_data_sdk::utilities::bfp8::bfp8max(a, b);
}

} // namespace std

template <>
struct fmt::formatter<hip_fp8_e5m2> : fmt::formatter<float>
{
    template <typename FormatContext>
    auto format(hip_fp8_e5m2 h, FormatContext& ctx) const
    {
        return fmt::formatter<float>::format(static_cast<float>(h), ctx);
    }
};
