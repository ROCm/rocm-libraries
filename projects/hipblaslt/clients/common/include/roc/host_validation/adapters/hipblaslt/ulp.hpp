// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt type adapter. ULP arithmetic and aggregation are
// owned by roc::host-validation.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <roc/host_validation/adapters/hipblaslt/Comparison.hpp>
#include <span>

inline int ulp_mantissa_bits(hipDataType type)
{
    return roc::host_validation::ulpMantissaBits(
        roc::host_validation::hipblaslt_adapter::scalarType(type));
}

template <typename T>
inline double ulp_as_double(T value)
{
    return static_cast<double>(value);
}

inline double ulp_distance(double exact, double approximation, int mantissaBits)
{
    return roc::host_validation::ulpDistance(exact, approximation, mantissaBits);
}

template <typename T>
inline void ulp_accumulate_general(int64_t M,
                                   int64_t N,
                                   int64_t lda,
                                   int64_t stride,
                                   T*      hCPU,
                                   T*      hGPU,
                                   int64_t batchCount,
                                   int,
                                   double& maxUlp,
                                   double& sumUlp,
                                   size_t& count)
{
    if(M == 0 || N == 0 || batchCount == 0)
        return;

    using namespace roc::host_validation;
    using namespace roc::host_validation::hipblaslt_adapter;

    const Layout      layout          = comparisonLayout(M, N, lda, stride, batchCount);
    const size_t      storageElements = comparisonStorageElements(layout);
    ComparisonOptions options;
    options.pointwise                  = false;
    options.computePointwiseStatistics = false;
    options.computeFrobenius           = false;
    options.computeUlp                 = true;
    options.ulpType                    = scalarType<T>();
    options.maxReportedMismatches      = 0;
    options.selection.indexOrder       = ComparisonIndexOrder::FirstDimensionFastest;

    const ComparisonResult report = compare(std::span<const T>(hGPU, storageElements),
                                            layout,
                                            std::span<const T>(hCPU, storageElements),
                                            layout,
                                            options);
    maxUlp                        = std::max(maxUlp, report.maximumUlp);
    sumUlp += report.sumUlp;
    count += report.ulpCompared;
}

inline void ulp_check_general(int64_t     M,
                              int64_t     N,
                              int64_t     lda,
                              int64_t     stride,
                              void*       hCPU,
                              void*       hGPU,
                              int64_t     batchCount,
                              double&     maxUlp,
                              double&     sumUlp,
                              size_t&     count,
                              hipDataType type)
{
    if(M == 0 || N == 0 || batchCount == 0)
        return;

    using namespace roc::host_validation;
    using namespace roc::host_validation::hipblaslt_adapter;

    ComparisonOptions options;
    options.pointwise                  = false;
    options.computePointwiseStatistics = false;
    options.computeFrobenius           = false;
    options.computeUlp                 = true;
    options.ulpType                    = scalarType(type);
    options.maxReportedMismatches      = 0;
    const ComparisonResult report
        = compareBuffers(M, N, lda, stride, hCPU, hGPU, batchCount, type, options);
    maxUlp = std::max(maxUlp, report.maximumUlp);
    sumUlp += report.sumUlp;
    count += report.ulpCompared;
}
