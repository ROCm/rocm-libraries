// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt Frobenius acceptance policy.

#include <algorithm>
#include <hipblaslt/hipblaslt.h>
#include <optional>
#include <roc/host_numerics/scalar.hpp>

// These are product-level acceptance policies, not numerical mechanics.
// Problem/architecture-specific callers may widen them.
inline double norm_tolerance(roc::host_numerics::ScalarType type)
{
    using roc::host_numerics::ScalarType;
    switch(type)
    {
    case ScalarType::Float32:
    case ScalarType::ComplexFloat32:
        return 0.00001;
    case ScalarType::Float64:
    case ScalarType::ComplexFloat64:
        return 0.000000000001;
    case ScalarType::Float16:
        return 0.01;
    case ScalarType::BFloat16:
        return 0.1;
    case ScalarType::Float8E4M3Fnuz:
    case ScalarType::Float8E4M3:
        return 0.125;
    case ScalarType::Float8E5M2Fnuz:
    case ScalarType::Float8E5M2:
        return 0.25;
    case ScalarType::Int32:
        return 0.0001;
    case ScalarType::Int8:
        return 0.01;
    case ScalarType::Float4E2M1:
        return 0.3;
    case ScalarType::Float6E2M3:
    case ScalarType::Float6E3M2:
        return 0.5;
    default:
        return 0.0;
    }
}

inline bool norm_check(double normError, roc::host_numerics::ScalarType type)
{
    const double tolerance = norm_tolerance(type);
    return tolerance > 0.0 && normError < tolerance;
}

inline bool norm_check(double                                        normError,
                       roc::host_numerics::ScalarType                outputType,
                       hipblasComputeType_t                          computeType,
                       std::optional<roc::host_numerics::ScalarType> inputTypeA = std::nullopt,
                       std::optional<roc::host_numerics::ScalarType> inputTypeB = std::nullopt)
{
    double tolerance = norm_tolerance(outputType);
    if(computeType == HIPBLAS_COMPUTE_32F_FAST_16BF
       && outputType == roc::host_numerics::ScalarType::Float32)
        tolerance = std::max(tolerance, 0.5);
    if(inputTypeA)
        tolerance = std::max(tolerance, norm_tolerance(*inputTypeA));
    if(inputTypeB)
        tolerance = std::max(tolerance, norm_tolerance(*inputTypeB));
    return tolerance > 0.0 && normError < tolerance;
}
