// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt descriptor adapter. The tolerance search and all
// elementwise numerical decisions are owned by roc::host-validation.

#include <array>
#include <roc/host_validation/adapters/hipblaslt/Comparison.hpp>

inline bool allclose_check_general(char,
                                   int64_t     M,
                                   int64_t     N,
                                   int64_t     lda,
                                   int64_t     stride,
                                   void*       hCPU,
                                   void*       hGPU,
                                   int64_t     batchCount,
                                   double&     hipblasltAtol,
                                   double&     hipblasltRtol,
                                   hipDataType type)
{
    if(M == 0 || N == 0 || batchCount == 0)
        return false;

    using namespace roc::host_validation;
    using namespace roc::host_validation::hipblaslt_adapter;

    const Layout      layout = comparisonLayout(M, N, lda, stride, batchCount);
    ComparisonOptions options;
    options.computePointwiseStatistics = false;
    options.computeFrobenius           = false;
    options.maxReportedMismatches      = 0;
    options.selection.indexOrder       = ComparisonIndexOrder::FirstDimensionFastest;

    constexpr std::array<double, 6> candidates{1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1};
    const auto tolerance = findAllCloseTolerance(comparisonView(hGPU, type, layout),
                                                 comparisonView(hCPU, type, layout),
                                                 std::span<const double>(candidates),
                                                 std::span<const double>(candidates),
                                                 options);
    if(!tolerance)
    {
        hipblasltAtol = 1.0;
        hipblasltRtol = 1.0;
        return false;
    }

    hipblasltAtol = tolerance->absolute;
    hipblasltRtol = tolerance->relative;
    return true;
}
