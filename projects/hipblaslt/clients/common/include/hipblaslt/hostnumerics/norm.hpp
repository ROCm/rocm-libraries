// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt Frobenius acceptance policy.

#include <algorithm>
#include <hipblaslt/hipblaslt.h>
#include <optional>
#include <roc/hostnumerics/scalar.hpp>

// These are product-level acceptance policies, not numerical mechanics.
// Problem/architecture-specific callers may widen them. Types without a
// policy have no limit, and a norm check of them fails.
inline std::optional<double> norm_tolerance(roc::hostnumerics::ScalarType type)
{
    using roc::hostnumerics::ScalarType;
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
        return std::nullopt;
    }
}

inline bool norm_check(double normError, roc::hostnumerics::ScalarType type)
{
    const auto tolerance = norm_tolerance(type);
    return tolerance && normError < *tolerance;
}

// The norm limit for a GEMM: the widest limit of its output and input types,
// widened further for reduced-precision compute.
inline std::optional<double> norm_tolerance(roc::hostnumerics::ScalarType outputType,
                                            hipblasComputeType_t          computeType,
                                            roc::hostnumerics::ScalarType inputTypeA,
                                            roc::hostnumerics::ScalarType inputTypeB)
{
    std::optional<double> tolerance;
    for(const auto type : {outputType, inputTypeA, inputTypeB})
        if(const auto limit = norm_tolerance(type))
            tolerance = std::max(tolerance.value_or(0.0), *limit);
    if(tolerance && computeType == HIPBLAS_COMPUTE_32F_FAST_16BF
       && outputType == roc::hostnumerics::ScalarType::Float32)
        tolerance = std::max(*tolerance, 0.5);
    return tolerance;
}

inline bool norm_check(double                        normError,
                       roc::hostnumerics::ScalarType outputType,
                       hipblasComputeType_t          computeType,
                       roc::hostnumerics::ScalarType inputTypeA,
                       roc::hostnumerics::ScalarType inputTypeB)
{
    const auto tolerance = norm_tolerance(outputType, computeType, inputTypeA, inputTypeB);
    return tolerance && normError < *tolerance;
}
