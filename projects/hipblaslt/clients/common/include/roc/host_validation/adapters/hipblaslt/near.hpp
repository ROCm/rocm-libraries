// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt tolerance policy.

#include "hipblaslt_ostream.hpp"

#include <limits>
#include <roc/host_validation/adapters/hipblaslt/Types.hpp>

template <class Tc, class Ti, class To>
static constexpr double sum_error_tolerance_for_gfx11 = std::numeric_limits<Tc>::epsilon();

template <>
inline constexpr double sum_error_tolerance_for_gfx11<float, hip_bfloat16, float> = 1 / 10.0;

template <>
inline constexpr double sum_error_tolerance_for_gfx11<float, hip_bfloat16, hip_bfloat16> = 1 / 10.0;

template <>
inline constexpr double sum_error_tolerance_for_gfx11<float, hipblasLtHalf, float> = 1 / 100.0;

template <>
inline constexpr double sum_error_tolerance_for_gfx11<float, hipblasLtHalf, hipblasLtHalf>
    = 1 / 100.0;

template <>
inline constexpr double sum_error_tolerance_for_gfx11<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf>
    = 1 / 100.0;

inline double sum_error_tolerance_for_gfx11_type(hipDataType computeType,
                                                 hipDataType inputType,
                                                 hipDataType outputType)
{
    if(computeType == HIP_R_32F && inputType == HIP_R_16BF
       && (outputType == HIP_R_32F || outputType == HIP_R_16BF))
        return 1 / 10.0;
    if(computeType == HIP_R_32F && inputType == HIP_R_16F
       && (outputType == HIP_R_32F || outputType == HIP_R_16F))
        return 1 / 100.0;
    if(computeType == HIP_R_16F && inputType == HIP_R_16F && outputType == HIP_R_16F)
        return 1 / 100.0;

    switch(computeType)
    {
    case HIP_R_32F:
    case HIP_C_32F:
        return std::numeric_limits<float>::epsilon();
    case HIP_R_64F:
    case HIP_C_64F:
        return std::numeric_limits<double>::epsilon();
    case HIP_R_16F:
        return std::numeric_limits<hipblasLtHalf>::epsilon();
    case HIP_R_32I:
        return std::numeric_limits<int32_t>::epsilon();
    default:
        hipblaslt_cerr << "Error type in sum_error_tolerance_for_gfx11_type" << std::endl;
        return 0.0;
    }
}
