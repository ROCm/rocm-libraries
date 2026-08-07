// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt descriptor adapter. Unit comparison, IEEE-class
// consistency, mismatch aggregation, and ULP evidence are component-owned.

#include "hipblaslt_test.hpp"

#include <roc/host_validation/adapters/hipblaslt/Comparison.hpp>

namespace roc::host_validation::hipblaslt_adapter
{
    inline ComparisonOptions unitComparisonOptions(hipDataType type)
    {
        const ScalarType  scalar = scalarType(type);
        ComparisonOptions options;
        options.equalNaNs                  = true;
        options.computePointwiseStatistics = false;
        options.computeFrobenius           = false;
        options.maxReportedMismatches      = 10;

        if(scalar == ScalarType::Float32 || scalar == ScalarType::Float64
           || scalar == ScalarType::ComplexFloat32 || scalar == ScalarType::ComplexFloat64)
        {
            options.pointwise           = false;
            options.computeUlp          = true;
            options.ulpType             = scalar;
            options.maximumUlpTolerance = 4.0;
        }
        return options;
    }
} // namespace roc::host_validation::hipblaslt_adapter

inline void unit_check_general(int64_t     M,
                               int64_t     N,
                               int64_t     lda,
                               int64_t     stride,
                               void*       hCPU,
                               void*       hGPU,
                               int64_t     batchCount,
                               hipDataType type)
{
#ifdef GOOGLE_TEST
    using namespace roc::host_validation;
    using namespace roc::host_validation::hipblaslt_adapter;

    const ComparisonResult report = compareBuffers(
        M, N, lda, stride, hCPU, hGPU, batchCount, type, unitComparisonOptions(type));
    ASSERT_TRUE(report.passed()) << "unit comparison failed for " << report.mismatches << " of "
                                 << report.compared << " values; non-finite class mismatches "
                                 << report.nonFiniteMismatches << ", max absolute difference "
                                 << report.maxAbsoluteDifference << ", max ULP "
                                 << report.maximumUlp;
#else
    (void)M;
    (void)N;
    (void)lda;
    (void)stride;
    (void)hCPU;
    (void)hGPU;
    (void)batchCount;
    (void)type;
#endif
}

inline void check_special_value_consistency(int64_t     M,
                                            int64_t     N,
                                            int64_t     lda,
                                            int64_t     stride,
                                            void*       hCPU,
                                            void*       hGPU,
                                            int64_t     batchCount,
                                            hipDataType type)
{
#ifdef GOOGLE_TEST
    using namespace roc::host_validation;
    using namespace roc::host_validation::hipblaslt_adapter;

    ComparisonOptions options;
    options.pointwise             = false;
    options.equalNaNs             = true;
    options.computeFrobenius      = false;
    options.maxReportedMismatches = 0;
    const ComparisonResult report
        = compareBuffers(M, N, lda, stride, hCPU, hGPU, batchCount, type, options);
    ASSERT_EQ(report.nonFiniteMismatches, 0)
        << "CPU and GPU disagree on NaN/infinity classification.";
#else
    (void)M;
    (void)N;
    (void)lda;
    (void)stride;
    (void)hCPU;
    (void)hGPU;
    (void)batchCount;
    (void)type;
#endif
}
