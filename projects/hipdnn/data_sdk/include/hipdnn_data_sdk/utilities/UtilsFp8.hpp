// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_fp8.h>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <string>
#include <type_traits>

#define HIPDNN_NAN_FP8 \
    hipdnn_data_sdk::utilities::fp8::uchar_as_fp8(static_cast<unsigned char>(0x7F))

using hip_fp8_e4m3 = __hip_fp8_e4m3;

inline __HOST_DEVICE__ hip_fp8_e4m3 operator""_fp8(long double val)
{
    return {static_cast<float>(val)};
}

inline __HOST_DEVICE__ hip_fp8_e4m3 operator-(hip_fp8_e4m3 a)
{
    hip_fp8_e4m3 val = a;
    val.__x ^= 0x80;
    return val;
}

inline __HOST_DEVICE__ bool operator==(hip_fp8_e4m3 a, hip_fp8_e4m3 b)
{
    return float(a) == float(b);
}

inline __HOST_DEVICE__ bool operator!=(hip_fp8_e4m3 a, hip_fp8_e4m3 b)
{
    return float(a) != float(b);
}

inline __HOST_DEVICE__ bool operator>(hip_fp8_e4m3 a, hip_fp8_e4m3 b)
{
    return float(a) > float(b);
}

inline __HOST_DEVICE__ bool operator>=(hip_fp8_e4m3 a, hip_fp8_e4m3 b)
{
    return float(a) >= float(b);
}

inline __HOST_DEVICE__ bool operator<(hip_fp8_e4m3 a, hip_fp8_e4m3 b)
{
    return float(a) < float(b);
}

inline __HOST_DEVICE__ bool operator<=(hip_fp8_e4m3 a, hip_fp8_e4m3 b)
{
    return float(a) <= float(b);
}

namespace hipdnn_data_sdk::utilities::fp8
{
inline __HOST_DEVICE__ hip_fp8_e4m3 uchar_as_fp8(const unsigned char a)
{
    hip_fp8_e4m3 val;
    val.__x = a;
    return val;
}

inline __HOST_DEVICE__ hip_fp8_e4m3 fp8abs(hip_fp8_e4m3 a)
{
    hip_fp8_e4m3 abs = a;
    abs.__x &= 0x7f;
    return abs;
}

inline __HOST_DEVICE__ bool fp8isnan(hip_fp8_e4m3 x)
{
    return (x.__x & 0x7f) == 0x7f;
}

inline __HOST_DEVICE__ hip_fp8_e4m3 fp8max(const hip_fp8_e4m3 a, const hip_fp8_e4m3 b)
{
    auto aNan = fp8isnan(a);
    auto bNan = fp8isnan(b);

    if(aNan || bNan)
    {
        if(aNan && bNan)
        {
            return HIPDNN_NAN_FP8; // return canonical NaN
        }

        return aNan ? b : a;
    }

    return a >= b ? a : b;
}

} // namespace hipdnn_data_sdk::utilities::fp8

namespace std
{

inline __HOST_DEVICE__ hip_fp8_e4m3 fabs(hip_fp8_e4m3 num)
{
    return hipdnn_data_sdk::utilities::fp8::fp8abs(num);
}

inline __HOST_DEVICE__ hip_fp8_e4m3 abs(hip_fp8_e4m3 num)
{
    return hipdnn_data_sdk::utilities::fp8::fp8abs(num);
}

inline __HOST_DEVICE__ hip_fp8_e4m3 max(hip_fp8_e4m3 a, hip_fp8_e4m3 b)
{
    return hipdnn_data_sdk::utilities::fp8::fp8max(a, b);
}

} // namespace std

template <>
struct fmt::formatter<hip_fp8_e4m3> : fmt::formatter<float>
{
    template <typename FormatContext>
    auto format(hip_fp8_e4m3 h, FormatContext& ctx) const
    {
        return fmt::formatter<float>::format(static_cast<float>(h), ctx);
    }
};
