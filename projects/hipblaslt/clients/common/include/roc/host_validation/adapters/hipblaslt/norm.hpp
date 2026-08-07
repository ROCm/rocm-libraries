// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt descriptor and tolerance-policy adapter.
// Frobenius evidence is computed by roc::host-validation.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <roc/host_validation/adapters/hipblaslt/Comparison.hpp>
#include <stdexcept>

inline double norm_check_general(char        normType,
                                 int64_t     M,
                                 int64_t     N,
                                 int64_t     lda,
                                 int64_t     stride,
                                 void*       hCPU,
                                 void*       hGPU,
                                 int64_t     batchCount,
                                 hipDataType type)
{
    if(M == 0 || N == 0 || batchCount == 0)
        return 0.0;
    if(normType != 'F' && normType != 'f')
        throw std::invalid_argument(
            "The consolidated host comparison currently exposes Frobenius norm evidence.");

    using namespace roc::host_validation;
    using namespace roc::host_validation::hipblaslt_adapter;

    const ScalarType scalar      = scalarType(type);
    const size_t     storageBits = scalarTypeInfo(scalar).storageBits;
    if(storageBits % 8 != 0)
        throw std::invalid_argument(
            "hipBLASLt norm comparison requires byte-addressable output storage.");
    const size_t elementBytes = storageBits / 8;

    ComparisonOptions options;
    options.pointwise                  = false;
    options.equalNaNs                  = true;
    options.computePointwiseStatistics = false;
    options.computeFrobenius           = true;
    options.maxReportedMismatches      = 0;

    double cumulativeError = 0.0;
    for(int64_t batch = 0; batch < batchCount; ++batch)
    {
        const size_t           byteOffset = static_cast<size_t>(batch * stride) * elementBytes;
        const auto*            expected   = static_cast<const std::byte*>(hCPU) + byteOffset;
        const auto*            observed   = static_cast<const std::byte*>(hGPU) + byteOffset;
        const ComparisonResult report
            = compareBuffers(M, N, lda, 0, expected, observed, 1, type, options);
        cumulativeError += report.relativeFrobeniusError;
    }
    return cumulativeError;
}

inline double norm_check_general(
    char normType, int64_t M, int64_t N, int64_t lda, void* hCPU, void* hGPU, hipDataType type)
{
    return norm_check_general(normType, M, N, lda, 0, hCPU, hGPU, 1, type);
}

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
