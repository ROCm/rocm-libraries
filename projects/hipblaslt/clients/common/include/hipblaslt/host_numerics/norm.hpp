// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt Frobenius acceptance policy.

#include <algorithm>
#include <hipblaslt/host_numerics/Types.hpp>

// These are product-level acceptance policies, not numerical mechanics.
// Problem/architecture-specific callers may widen them.
inline double norm_tolerance(hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
    case HIP_C_32F:
        return 0.00001;
    case HIP_R_64F:
    case HIP_C_64F:
        return 0.000000000001;
    case HIP_R_16F:
        return 0.01;
    case HIP_R_16BF:
        return 0.1;
    case HIP_R_8F_E4M3_FNUZ:
    case HIP_R_8F_E4M3:
        return 0.125;
    case HIP_R_8F_E5M2_FNUZ:
    case HIP_R_8F_E5M2:
        return 0.25;
    case HIP_R_32I:
        return 0.0001;
    case HIP_R_8I:
        return 0.01;
    case HIP_R_4F_E2M1:
        return 0.3;
    case HIP_R_6F_E2M3:
    case HIP_R_6F_E3M2:
        return 0.5;
    default:
        return 0.0;
    }
}

inline bool norm_check(double normError, hipDataType type)
{
    const double tolerance = norm_tolerance(type);
    return tolerance > 0.0 && normError < tolerance;
}

inline bool norm_check(double               normError,
                       hipDataType          outputType,
                       hipblasComputeType_t computeType,
                       hipDataType          inputTypeA = static_cast<hipDataType>(-1),
                       hipDataType          inputTypeB = static_cast<hipDataType>(-1))
{
    double tolerance = norm_tolerance(outputType);
    if(computeType == HIPBLAS_COMPUTE_32F_FAST_16BF && outputType == HIP_R_32F)
        tolerance = std::max(tolerance, 0.5);
    if(static_cast<int>(inputTypeA) >= 0)
        tolerance = std::max(tolerance, norm_tolerance(inputTypeA));
    if(static_cast<int>(inputTypeB) >= 0)
        tolerance = std::max(tolerance, norm_tolerance(inputTypeB));
    return tolerance > 0.0 && normError < tolerance;
}
