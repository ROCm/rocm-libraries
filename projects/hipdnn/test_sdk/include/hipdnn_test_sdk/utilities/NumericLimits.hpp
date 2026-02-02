// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>
#include <type_traits>

#include <hipdnn_data_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_data_sdk/utilities/UtilsFp16.hpp>

namespace hipdnn_test_sdk::utilities
{

/**
 * @brief Returns the machine epsilon for a given floating-point type.
 *
 * The epsilon value represents the difference between 1.0 and the next representable value.
 * It is related to the number of mantissa bits by the formula: epsilon = 2^(-mantissa_bits).
 *
 * Type         | Mantissa Bits | Epsilon (2^-bits)
 * -------------|---------------|-------------------
 * half (fp16)  | 10            | 2^-10 ≈ 0.0009765625
 * hip_bfloat16 | 7             | 2^-7  = 0.0078125
 * float (fp32) | 23            | 2^-23 ≈ 1.19209e-07
 * double (fp64)| 52            | 2^-52 ≈ 2.22044e-16
 *
 * @tparam T The floating-point type.
 * @return The machine epsilon as a double.
 */
template <typename T>
constexpr double getEpsilon()
{
    if constexpr(std::is_same_v<T, half>)
    {
        // 2^-10 = 0.0009765625
        return 0.0009765625;
    }
    else if constexpr(std::is_same_v<T, hip_bfloat16>)
    {
        // 2^-7 = 0.0078125
        return 0.0078125;
    }
    else
    {
        static_assert(std::numeric_limits<T>::is_specialized,
                      "Type not supported and std::numeric_limits not specialized");
        return static_cast<double>(std::numeric_limits<T>::epsilon());
    }
}

/**
 * @brief Returns the maximum finite value for a given floating-point type.
 *
 * @tparam T The floating-point type.
 * @return The maximum finite value as a double.
 */
template <typename T>
constexpr double getMax()
{
    if constexpr(std::is_same_v<T, half>)
    {
        // Max value for IEEE 754 half-precision (fp16):
        // Sign: 1 bit, Exponent: 5 bits, Mantissa: 10 bits
        // Max exponent value (E_max) = 15 (since 31 is reserved for Inf/NaN)
        // Max value = 2^E_max * (1 + (2^10 - 1) / 2^10)
        //           = 2^15 * (1 + 1023/1024)
        //           = 32768 * (2047/1024)
        //           = 32 * 2047
        //           = 65504.0
        return 65504.0;
    }
    else if constexpr(std::is_same_v<T, hip_bfloat16>)
    {
        // BFloat16 has same range as float
        return static_cast<double>(std::numeric_limits<float>::max());
    }
    else
    {
        static_assert(std::numeric_limits<T>::is_specialized,
                      "Type not supported and std::numeric_limits not specialized");
        return static_cast<double>(std::numeric_limits<T>::max());
    }
}

} // namespace hipdnn_test_sdk::utilities
